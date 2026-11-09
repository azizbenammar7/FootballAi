"""Offline AI3 calibration of an existing analysis run from human-reviewed landmarks.

Three explicit steps, each writing a NEW file and never touching its inputs:

  merge   video-bound template + per-frame exports from the annotate pages
          -> merged annotations (every landmark stays ``pending``)
  review  merged annotations + a decisions file written by the human reviewer
          -> reviewed annotations (only the reviewer's explicit decisions apply;
             every landmark must be decided, nothing defaults to accepted)
  run     reviewed annotations + the run's cached AI1 parquet files + the video
          -> the same pitch-calibration, projected-observations and diagnostic
             artifacts the ``pitch_calibration`` worker stage would publish,
             computed with the worker's own functions. No YOLO, no tracking,
             no cloned run, no network.

All outputs must be written outside a Git working tree: they derive from
private footage. Usage:

  PYTHONPATH=v2/src python v2/dev/ai3_real_calibration.py merge \
      --template annotations-video-bound.json \
      --frame frame-1535-landmarks.json --frame frame-1595-landmarks.json \
      --output annotations-merged.json
  PYTHONPATH=v2/src python v2/dev/ai3_real_calibration.py review \
      --annotations annotations-merged.json --decisions review-decisions.json \
      --output annotations-reviewed.json
  PYTHONPATH=v2/src python v2/dev/ai3_real_calibration.py run \
      --annotations annotations-reviewed.json --review-record annotations-reviewed.review-record.json \
      --run-id <run id> --tracklets-sha256 <recorded tracklets sha256> \
      --detections <run>/artifacts/detections.parquet --tracklets <run>/artifacts/tracklets.parquet \
      --video clip.mp4 --output-dir ai3-offline-<date>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from footballai_v2.execution.ai1.artifacts import sha256_bytes
from footballai_v2.execution.ai3.annotations import (
    AnnotatedFrame,
    AnnotationOrigin,
    LandmarkAnnotation,
    PitchAnnotations,
    ReviewStatus,
    annotations_from_json_bytes,
    annotations_to_json_bytes,
)
from footballai_v2.execution.ai3.cli import _file_sha256, _inside_git_tree, _json
from footballai_v2.execution.ai3.pipeline_stage import (
    SpatialStageError,
    load_pitch_annotations_parameter,
    pitch_annotations_parameter,
    run_identity_stage,
    run_pitch_calibration_stage,
    verify_ai1_artifacts,
)

# A page export may carry a rounded timestamp; the template's AI1 timestamp is
# authoritative. Anything further apart than this is a different frame.
EXPORT_TIMESTAMP_TOLERANCE_S = 0.005
REVIEW_RECORD_SCHEMA = "footballai.ai3-landmark-review-record/v1"


def _fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def _check_output(path: Path) -> None:
    if path.exists():
        _fail(f"refusing to overwrite {path}")
    if _inside_git_tree(path):
        _fail(f"refusing to write inside a Git working tree: {path}")


def _write_new(path: Path, content: bytes) -> str:
    _check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:  # exclusive create: never replaces a file
        handle.write(content)
    return sha256_bytes(content)


def _load_annotations(path: Path) -> tuple[PitchAnnotations, bytes]:
    content = path.read_bytes()
    try:
        return annotations_from_json_bytes(content), content
    except (ValueError, KeyError, TypeError) as exc:
        _fail(f"{path} is not valid footballai.pitch-landmark-annotations/v1: {exc}")
        raise  # unreachable


def _frame_summary(frame: AnnotatedFrame) -> dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "timestamp_seconds": frame.timestamp_seconds,
        "shot_id": frame.shot_id,
        "landmarks": len(frame.landmarks),
        "by_role": dict(sorted(Counter(item.role.value for item in frame.landmarks).items())),
        "by_quality": dict(sorted(Counter(item.quality.value for item in frame.landmarks).items())),
        "by_review_status": dict(sorted(Counter(item.review_status.value for item in frame.landmarks).items())),
        "annotators": sorted({item.annotator for item in frame.landmarks}),
        "frame_image_sha256": frame.frame_image_sha256,
    }


# --- merge -------------------------------------------------------------------------------


def merge(args: argparse.Namespace) -> int:
    template, template_bytes = _load_annotations(Path(args.template))
    source = template.source
    if source.origin is not AnnotationOrigin.MANUAL:
        _fail("the template must have origin 'manual'")
    if source.video_sha256 is None or source.detections_sha256 is not None or source.tracklets_sha256 is not None:
        _fail("the template must be bound to the video only (create it with `template --bind video`)")
    by_index = {frame.frame_index: frame for frame in template.frames}
    replaced: dict[int, AnnotatedFrame] = {}
    inputs = []
    for name in args.frame:
        path = Path(name)
        content = path.read_bytes()
        try:
            exported = AnnotatedFrame.from_dict(json.loads(content))
        except (ValueError, KeyError, TypeError) as exc:
            _fail(f"{path} is not a valid exported frame: {exc}")
        slot = by_index.get(exported.frame_index)
        if slot is None:
            _fail(f"{path}: frame {exported.frame_index} is not a keyframe of the template {sorted(by_index)}")
        if exported.frame_index in replaced:
            _fail(f"{path}: frame {exported.frame_index} was exported more than once")
        if slot.landmarks:
            _fail(f"{path}: template frame {exported.frame_index} already has landmarks; refusing to overwrite them")
        if exported.shot_id != slot.shot_id:
            _fail(f"{path}: shot {exported.shot_id!r} differs from the template's {slot.shot_id!r}")
        drift = abs(exported.timestamp_seconds - slot.timestamp_seconds)
        if drift > EXPORT_TIMESTAMP_TOLERANCE_S:
            _fail(f"{path}: timestamp {exported.timestamp_seconds} is {drift:.6f} s from the AI1 timestamp "
                  f"{slot.timestamp_seconds}; the page was made for a different frame")
        not_pending = [item.landmark_id for item in exported.landmarks if item.review_status is not ReviewStatus.PENDING]
        if not_pending:
            _fail(f"{path}: exported landmarks must be 'pending'; found reviewed points {not_pending}")
        if not exported.landmarks:
            _fail(f"{path}: the export contains no landmarks")
        unnamed = sorted({item.annotator for item in exported.landmarks if item.annotator.strip() in ("", "unnamed")})
        if unnamed:
            _fail(f"{path}: landmarks without a named annotator; fill 'Annotator' on the page and export again")
        # Keep the template's AI1 timestamp: the calibrator requires it to match AI1 within 1 µs.
        replaced[exported.frame_index] = dataclasses.replace(exported, timestamp_seconds=slot.timestamp_seconds)
        inputs.append({
            "path": str(path), "sha256": sha256_bytes(content), "frame_index": exported.frame_index,
            "export_timestamp_seconds": exported.timestamp_seconds, "timestamp_replaced_by_ai1": drift > 0,
        })
    frames = tuple(replaced.get(frame.frame_index, frame) for frame in template.frames)
    try:
        merged = dataclasses.replace(template, frames=frames)  # re-runs every schema and geometry check
    except ValueError as exc:
        _fail(f"merged annotations are invalid: {exc}")
    output = Path(args.output)
    content = annotations_to_json_bytes(merged)
    sha = _write_new(output, content)
    print(json.dumps({
        "step": "merge",
        "template": {"path": args.template, "sha256": sha256_bytes(template_bytes)},
        "video_sha256": source.video_sha256,
        "pitch": template.pitch.to_dict(),
        "exports": inputs,
        "output": {"path": str(output), "sha256": sha},
        "frames": [_frame_summary(frame) for frame in merged.frames],
        "frames_without_landmarks": [frame.frame_index for frame in merged.frames if not frame.landmarks],
        "note": "Every landmark is still 'pending'. Only the review step, with your explicit decisions, accepts points.",
    }, indent=2, sort_keys=True))
    return 0


# --- review ------------------------------------------------------------------------------


def review(args: argparse.Namespace) -> int:
    annotations, content = _load_annotations(Path(args.annotations))
    decisions_path = Path(args.decisions)
    decisions_bytes = decisions_path.read_bytes()
    try:
        record = json.loads(decisions_bytes)
    except json.JSONDecodeError as exc:
        _fail(f"{decisions_path} is not valid JSON: {exc}")
    if not isinstance(record, dict) or set(record) != {"reviewer", "decisions"}:
        _fail('decisions must be {"reviewer": "<name>", "decisions": {"<frame_index>": {"<landmark_id>": "accepted|rejected"}}}')
    reviewer, decisions = record["reviewer"], record["decisions"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        _fail("reviewer must be a non-empty name")
    if not isinstance(decisions, dict):
        _fail("decisions must be an object keyed by frame index")
    annotated = {str(frame.frame_index): frame for frame in annotations.frames if frame.landmarks}
    if set(decisions) != set(annotated):
        _fail(f"decisions must cover exactly the annotated frames {sorted(annotated, key=int)}; got {sorted(decisions)}")
    frames = []
    tally: Counter[str] = Counter()
    for frame in annotations.frames:
        if not frame.landmarks:
            frames.append(frame)
            continue
        chosen = decisions[str(frame.frame_index)]
        ids = {item.landmark_id for item in frame.landmarks}
        if not isinstance(chosen, dict) or set(chosen) != ids:
            missing = sorted(ids - set(chosen)) if isinstance(chosen, dict) else sorted(ids)
            extra = sorted(set(chosen) - ids) if isinstance(chosen, dict) else []
            _fail(f"frame {frame.frame_index}: every landmark needs a decision (missing {missing}, unknown {extra})")
        updated: list[LandmarkAnnotation] = []
        for item in frame.landmarks:
            if item.review_status is not ReviewStatus.PENDING:
                _fail(f"frame {frame.frame_index} {item.landmark_id}: already {item.review_status.value}; review once")
            decision = chosen[item.landmark_id]
            if decision not in (ReviewStatus.ACCEPTED.value, ReviewStatus.REJECTED.value):
                _fail(f"frame {frame.frame_index} {item.landmark_id}: decision must be 'accepted' or 'rejected'")
            tally[decision] += 1
            updated.append(dataclasses.replace(item, review_status=ReviewStatus(decision)))
        frames.append(dataclasses.replace(frame, landmarks=tuple(updated)))
    reviewed = dataclasses.replace(annotations, frames=tuple(frames))
    output = Path(args.output)
    review_record = output.with_name(output.stem + ".review-record.json")
    _check_output(review_record)
    _check_output(output)
    reviewed_bytes = annotations_to_json_bytes(reviewed)
    sha = sha256_bytes(reviewed_bytes)
    summary = {
        "schema": REVIEW_RECORD_SCHEMA,
        "reviewer": reviewer.strip(),
        "input": {"path": args.annotations, "sha256": sha256_bytes(content)},
        "decisions": {"path": str(decisions_path), "sha256": sha256_bytes(decisions_bytes)},
        "output": {"path": str(output), "sha256": sha},
        "decisions_by_status": dict(sorted(tally.items())),
        "frames": [_frame_summary(frame) for frame in reviewed.frames],
    }
    # The record is written first, so accepted points never exist without it.
    _write_new(review_record, _json(summary))
    _write_new(output, reviewed_bytes)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


# --- run ---------------------------------------------------------------------------------


def _coverage_by_shot(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    """Frame states per declared shot; frames outside every shot are grouped under ``None``."""
    states = diagnostic["frame_states"]
    shots = {item["shot_id"]: item for item in diagnostic["calibration"]["shots"]}
    statuses: dict[str | None, Counter[str]] = {key: Counter() for key in [*shots, None]}
    reasons: dict[str | None, Counter[str]] = {key: Counter() for key in [*shots, None]}
    for index, status, reason in zip(states["frame_index"], states["status"], states["reason"]):
        key = next((k for k, s in shots.items() if s["start_frame"] <= index <= s["end_frame"]), None)
        statuses[key][status] += 1
        if reason:
            reasons[key][reason] += 1
    return [
        {
            "shot_id": key,
            "frames": [shots[key]["start_frame"], shots[key]["end_frame"]] if key else None,
            "camera_motion": shots[key]["camera_motion"] if key else None,
            "sampled_frames": sum(statuses[key].values()),
            "by_status": dict(sorted(statuses[key].items())),
            "by_reason": dict(sorted(reasons[key].items())),
        }
        for key in statuses
        if key is not None or statuses[key]
    ]


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    if out.exists():
        _fail(f"refusing to reuse existing output directory {out}")
    if _inside_git_tree(out):
        _fail(f"refusing to write inside a Git working tree: {out}")
    annotations, annotations_bytes = _load_annotations(Path(args.annotations))
    record_bytes = Path(args.review_record).read_bytes()
    record = json.loads(record_bytes)
    if record.get("schema") != REVIEW_RECORD_SCHEMA or record.get("output", {}).get("sha256") != sha256_bytes(annotations_bytes):
        _fail("the review record does not belong to these annotations (its output SHA-256 differs)")
    pending = sum(item.review_status is ReviewStatus.PENDING for frame in annotations.frames for item in frame.landmarks)
    if pending:
        _fail(f"{pending} landmarks are still 'pending'; every landmark needs a human decision first")
    detections_bytes = Path(args.detections).read_bytes()
    tracklets_bytes = Path(args.tracklets).read_bytes()
    video_sha = _file_sha256(Path(args.video))
    try:
        ai1 = verify_ai1_artifacts(
            detections_bytes, tracklets_bytes, video_sha256=video_sha, recorded_tracklets_sha256=args.tracklets_sha256,
        )
        parameter = pitch_annotations_parameter(annotations_bytes, video_sha256=video_sha, data_origin="real")
        identities = run_identity_stage(ai1, tracklets_bytes, source_id=f"run-{args.run_id}")
        result = run_pitch_calibration_stage(
            ai1, identities, run_id=args.run_id, data_origin="real",
            annotations=load_pitch_annotations_parameter(parameter),
        )
    except SpatialStageError as exc:
        _fail(f"{exc.code}: {exc}")
    out.mkdir(parents=True)
    written = {
        "player-identities.json": _write_new(out / "player-identities.json", identities.content),
        "pitch-space-diagnostic.json": _write_new(out / "pitch-space-diagnostic.json", _json(result.diagnostic)),
    }
    if result.calibration_bytes is not None:
        written["pitch-calibration.json"] = _write_new(out / "pitch-calibration.json", result.calibration_bytes)
    if result.observations_bytes is not None:
        written["projected-player-observations.parquet"] = _write_new(
            out / "projected-player-observations.parquet", result.observations_bytes,
        )
    diagnostic = result.diagnostic
    keyframes = [
        {
            "frame_index": item["frame_index"], "timestamp_seconds": item["timestamp_seconds"],
            "shot_id": item["shot_id"], "status": item["status"], "reason": item["reason"],
            "metrics": item["metrics"],
        }
        for item in diagnostic["calibration"]["keyframes"]
    ]
    report = {
        "step": "run",
        "no_inference": "YOLO and ByteTrack were not run; cached AI1 artifacts were verified and reused.",
        "run_id": args.run_id,
        "inputs": {
            "video_sha256": video_sha,
            "detections_sha256": ai1.detections_sha256,
            "tracklets_sha256": ai1.tracklets_sha256,
            "annotations_file_sha256": sha256_bytes(annotations_bytes),
            "annotations_canonical_sha256": parameter["sha256"],
            "review_record_sha256": sha256_bytes(record_bytes),
            "reviewer": record.get("reviewer"),
        },
        "frames": diagnostic["frames"],
        "coverage_by_shot": _coverage_by_shot(diagnostic),
        "calibration_summary": diagnostic["calibration"]["summary"],
        "suspected_cut_frames": diagnostic["calibration"]["suspected_cut_frames"],
        "keyframes": keyframes,
        "observations": diagnostic["observations"],
        "identity": diagnostic["identity"],
        "outputs": {"directory": str(out), "sha256": written},
        "evidence": diagnostic["evidence"],
    }
    report_bytes = _json(report)
    _write_new(out / "offline-report.json", report_bytes)
    sys.stdout.write(report_bytes.decode())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    step = commands.add_parser("merge", help="merge page exports into the video-bound template (stays pending)")
    step.add_argument("--template", required=True)
    step.add_argument("--frame", action="append", required=True, help="one exported frame JSON; repeat per frame")
    step.add_argument("--output", required=True)
    step.set_defaults(handler=merge)
    step = commands.add_parser("review", help="apply the human reviewer's explicit accept/reject decisions")
    step.add_argument("--annotations", required=True)
    step.add_argument("--decisions", required=True)
    step.add_argument("--output", required=True)
    step.set_defaults(handler=review)
    step = commands.add_parser("run", help="offline pitch_calibration stage over cached AI1 artifacts")
    step.add_argument("--annotations", required=True)
    step.add_argument("--run-id", required=True)
    step.add_argument("--detections", required=True)
    step.add_argument("--tracklets", required=True)
    step.add_argument("--tracklets-sha256", required=True,
                      help="tracklets SHA-256 recorded in the run (analysis-diagnostics tracking.tracklets_sha256)")
    step.add_argument("--review-record", required=True, help="the .review-record.json written by the review step")
    step.add_argument("--video", required=True)
    step.add_argument("--output-dir", required=True)
    step.set_defaults(handler=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
