"""Offline AI3 pitch-calibration CLI.

No detector runs and nothing touches the network. ``annotate-frame`` is the
only command that reads a video (one frame, via OpenCV when installed) and it
refuses to write inside a Git working tree, so private frames stay out of Git.
Outputs are never overwritten unless ``--overwrite`` is passed.

Workflow for real footage::

    landmarks       → plan SVG + landmark table for the configured pitch
    template        → empty annotation file bound to the AI1 checksums
    annotate-frame  → frame PNG + local click page (outside the repo)
    (paste exported frames into the annotation file; review → "accepted")
    validate        → structural check + per-keyframe dry run
    calibrate       → footballai.pitch-calibration/v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from footballai_v2.execution.ai1.artifacts import (
    detections_from_parquet_bytes,
    sha256_bytes,
    tracklets_from_parquet_bytes,
)
from footballai_v2.execution.ai3.annotation_page import annotation_page_html
from footballai_v2.execution.ai3.annotations import (
    PITCH_ANNOTATIONS_SCHEMA,
    AnnotationOrigin,
    AnnotationSourceInfo,
    CameraMotion,
    PitchAnnotations,
    ShotSegment,
    annotations_from_json_bytes,
    annotations_to_json_bytes,
    check_source_bindings,
)
from footballai_v2.execution.ai3.artifact import calibrate, pitch_calibration_to_json_bytes
from footballai_v2.execution.ai3.benchmark import run_benchmark
from footballai_v2.execution.ai3.calibration import CalibrationPolicy, calibrate_keyframe
from footballai_v2.execution.ai3.cuts import analysed_frames, birth_burst_source, suspected_cuts_from_tracklets
from footballai_v2.execution.ai3.geometry import (
    IFAB_LAW_1_MARKINGS,
    LANDMARK_DESCRIPTIONS,
    CameraSide,
    PitchGeometry,
    pitch_plan_svg,
)


MARKINGS = {"ifab-law-1": IFAB_LAW_1_MARKINGS}


def _write(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _json(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_git_tree(path: Path) -> bool:
    for parent in (path.resolve(), *path.resolve().parents):
        if (parent / ".git").exists():
            return True
    return False


def _geometry(args: argparse.Namespace) -> PitchGeometry:
    return PitchGeometry(args.length_m, args.width_m, MARKINGS[args.markings], args.dimensions_source)


def _add_pitch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--length-m", type=float, required=True, help="measured pitch length (no default)")
    parser.add_argument("--width-m", type=float, required=True, help="measured pitch width (no default)")
    parser.add_argument("--markings", choices=sorted(MARKINGS), required=True)
    parser.add_argument("--dimensions-source", required=True, help="where the dimensions come from, e.g. 'club ground survey 2026'")


def _policy(path: str | None) -> CalibrationPolicy:
    if path is None:
        return CalibrationPolicy()
    return CalibrationPolicy.from_dict(json.loads(Path(path).read_bytes()))


def _landmarks(args: argparse.Namespace) -> int:
    geometry = _geometry(args)
    table = [
        {"landmark_id": name, "pitch_x_m": x, "pitch_y_m": y, "description": LANDMARK_DESCRIPTIONS[name]}
        for name, (x, y) in geometry.landmarks().items()
    ]
    print(json.dumps({"pitch": geometry.to_dict(), "landmarks": table}, indent=2, sort_keys=True))
    if args.svg:
        _write(Path(args.svg), pitch_plan_svg(geometry).encode(), args.overwrite)
    return 0


def _parse_shot(value: str) -> ShotSegment:
    parts = value.split(":")
    if len(parts) != 5:
        raise SystemExit("--shot must be ID:START:END:MOTION:CAMERA_SIDE")
    return ShotSegment(parts[0], int(parts[1]), int(parts[2]), CameraMotion(parts[3]), CameraSide(parts[4]))


def _template(args: argparse.Namespace) -> int:
    from footballai_v2.execution.ai3.annotations import AnnotatedFrame

    geometry = _geometry(args)
    video_sha = detections_sha = tracklets_sha = None
    width, height = args.frame_width, args.frame_height
    timestamps: dict[int, float] = {}
    if args.detections:
        content = Path(args.detections).read_bytes()
        detections = detections_from_parquet_bytes(content)
        detections_sha = sha256_bytes(content)
        video_sha = detections.provenance.source_video_sha256
        width, height = detections.provenance.source_width, detections.provenance.source_height
        timestamps.update({item.frame_index: item.timestamp_seconds for item in detections.sampled_frames})
    if args.tracklets:
        content = Path(args.tracklets).read_bytes()
        tracklets_sha = sha256_bytes(content)
        timestamps.update(dict(analysed_frames(tracklets_from_parquet_bytes(content))))
    if args.bind == "video":
        # For attaching to an analysis run: AI1 bytes differ per attempt, so only
        # the video checksum (and frame size/timestamps read above) is recorded.
        if video_sha is None:
            raise SystemExit("--bind video needs --detections to read the source video SHA-256")
        detections_sha = tracklets_sha = None
    if width is None or height is None:
        raise SystemExit("frame size is unknown: pass --detections or --frame-width/--frame-height")
    shots = tuple(_parse_shot(item) for item in args.shot)
    frames = []
    for value in args.keyframe:
        index_text, _, time_text = value.partition(":")
        index = int(index_text)
        if time_text:
            timestamp = float(time_text)
        elif index in timestamps:
            timestamp = timestamps[index]
        else:
            raise SystemExit(f"keyframe {index} is not an analysed AI1 frame; give INDEX:TIMESTAMP explicitly")
        shot = next((item for item in shots if item.contains(index)), None)
        if shot is None:
            raise SystemExit(f"keyframe {index} is not inside any --shot")
        frames.append(AnnotatedFrame(index, timestamp, shot.shot_id, ()))
    annotations = PitchAnnotations(
        PITCH_ANNOTATIONS_SCHEMA,
        AnnotationSourceInfo(args.source_id, AnnotationOrigin.MANUAL, width, height, video_sha, detections_sha, tracklets_sha),
        geometry, shots, tuple(sorted(frames, key=lambda item: item.frame_index)),
    )
    _write(Path(args.output), annotations_to_json_bytes(annotations), args.overwrite)
    return 0


def _annotate_frame(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    if _inside_git_tree(output):
        raise SystemExit(f"refusing to write video frames inside a Git working tree: {output}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the worker image
        raise SystemExit("annotate-frame needs OpenCV (pip install -r v2/requirements-v1-compat.txt)") from exc
    video = Path(args.video)
    capture = cv2.VideoCapture(str(video))
    try:
        # Decode sequentially, like AI1: seeking with CAP_PROP_POS_FRAMES is not
        # frame-accurate for many codecs and could label the wrong frame.
        ok = True
        for _ in range(args.frame_index):
            if not capture.grab():
                ok = False
                break
        image = None
        if ok:
            ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame_index} from {video}")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise SystemExit("could not encode the frame as PNG")
    png = encoded.tobytes()
    png_sha = sha256_bytes(png)
    stem = f"frame-{args.frame_index:07d}"
    _write(output / f"{stem}.png", png, args.overwrite)
    page = annotation_page_html(
        image_png=png, frame_index=args.frame_index, timestamp_seconds=args.timestamp, shot_id=args.shot_id,
        frame_image_sha256=png_sha, geometry=_geometry(args),
    )
    _write(output / f"{stem}-annotate.html", page.encode(), args.overwrite)
    print(json.dumps({
        "video_sha256": _file_sha256(video),
        "frame_index": args.frame_index,
        "frame_size": [int(image.shape[1]), int(image.shape[0])],
        "frame_image_sha256": png_sha,
        "page": str(output / f"{stem}-annotate.html"),
    }, indent=2, sort_keys=True))
    return 0


def _bindings(args: argparse.Namespace, annotations: PitchAnnotations):
    tracklets = tracklets_sha = detections_sha = video_sha = frame_size = None
    if getattr(args, "tracklets", None):
        content = Path(args.tracklets).read_bytes()
        tracklets, tracklets_sha = tracklets_from_parquet_bytes(content), sha256_bytes(content)
    if getattr(args, "detections", None):
        content = Path(args.detections).read_bytes()
        detections = detections_from_parquet_bytes(content)
        detections_sha = sha256_bytes(content)
        video_sha = detections.provenance.source_video_sha256
        frame_size = (detections.provenance.source_width, detections.provenance.source_height)
        if tracklets is not None and tracklets.provenance.detections_sha256 != detections_sha:
            raise SystemExit("tracklets were not built from these detections")
    check_source_bindings(
        annotations, tracklets_sha256=tracklets_sha, detections_sha256=detections_sha,
        video_sha256=video_sha, frame_size=frame_size,
    )
    return tracklets, tracklets_sha, detections_sha, video_sha


def _validate(args: argparse.Namespace) -> int:
    content = Path(args.annotations).read_bytes()
    annotations = annotations_from_json_bytes(content)
    _bindings(args, annotations)
    policy = _policy(args.policy)
    report = []
    for frame in annotations.frames:
        keyframe = calibrate_keyframe(
            frame, annotations.shot(frame.shot_id), annotations.pitch,
            (annotations.source.frame_width, annotations.source.frame_height), policy,
        )
        report.append({
            "frame_index": frame.frame_index,
            "status": keyframe.status.value,
            "reason": keyframe.reason.value if keyframe.reason else None,
            "metrics": dict(keyframe.metrics),
        })
    print(json.dumps({
        "annotations_sha256": sha256_bytes(content),
        "origin": annotations.source.origin.value,
        "structurally_valid": True,
        "keyframes": report,
    }, indent=2, sort_keys=True))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    content = Path(args.annotations).read_bytes()
    annotations = annotations_from_json_bytes(content)
    tracklets, tracklets_sha, detections_sha, video_sha = _bindings(args, annotations)
    cuts: tuple[int, ...] = ()
    cuts_source = None
    if args.suspected_cuts_from_tracklets:
        if tracklets is None:
            raise SystemExit("--suspected-cuts-from-tracklets needs --tracklets")
        cuts = suspected_cuts_from_tracklets(tracklets, min_births=args.min_births)
        cuts_source = birth_burst_source(args.min_births)
    artifact = calibrate(
        annotations,
        annotations_sha256=sha256_bytes(content),
        policy=_policy(args.policy),
        frames=analysed_frames(tracklets) if tracklets is not None else (),
        suspected_cut_frames=cuts,
        suspected_cuts_source=cuts_source,
        tracklets_sha256=tracklets_sha,
        detections_sha256=detections_sha,
        video_sha256=video_sha,
    )
    _write(Path(args.output), pitch_calibration_to_json_bytes(artifact), args.overwrite)
    print(json.dumps(dict(artifact.summary), indent=2, sort_keys=True))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    result = run_benchmark(_policy(args.policy))
    _write(Path(args.output), _json(result), args.overwrite)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="footballai_v2.execution.ai3.cli", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)

    landmarks = commands.add_parser("landmarks", help="landmark table and plan SVG for a configured pitch")
    _add_pitch(landmarks)
    landmarks.add_argument("--svg")
    landmarks.add_argument("--overwrite", action="store_true")
    landmarks.set_defaults(handler=_landmarks)

    template = commands.add_parser("template", help="empty annotation file bound to AI1 checksums")
    _add_pitch(template)
    template.add_argument("--source-id", required=True)
    template.add_argument("--detections")
    template.add_argument("--tracklets")
    template.add_argument("--frame-width", type=int)
    template.add_argument("--frame-height", type=int)
    template.add_argument("--shot", action="append", required=True, help="ID:START:END:static|moving|unknown:CAMERA_SIDE")
    template.add_argument("--keyframe", action="append", required=True, help="INDEX or INDEX:TIMESTAMP")
    template.add_argument(
        "--bind", choices=("ai1", "video"), default="ai1",
        help="ai1: pin the detections/tracklets checksums (offline calibrate); "
             "video: pin only the video checksum, required to attach annotations to an analysis run",
    )
    template.add_argument("--output", required=True)
    template.add_argument("--overwrite", action="store_true")
    template.set_defaults(handler=_template)

    page = commands.add_parser("annotate-frame", help="extract one frame and write a local click page (outside Git)")
    _add_pitch(page)
    page.add_argument("--video", required=True)
    page.add_argument("--frame-index", type=int, required=True)
    page.add_argument("--timestamp", type=float, required=True)
    page.add_argument("--shot-id", required=True)
    page.add_argument("--output-dir", required=True)
    page.add_argument("--overwrite", action="store_true")
    page.set_defaults(handler=_annotate_frame)

    validate = commands.add_parser("validate", help="structural check and per-keyframe dry run")
    validate.add_argument("--annotations", required=True)
    validate.add_argument("--tracklets")
    validate.add_argument("--detections")
    validate.add_argument("--policy")
    validate.set_defaults(handler=_validate)

    run = commands.add_parser("calibrate", help="write footballai.pitch-calibration/v1")
    run.add_argument("--annotations", required=True)
    run.add_argument("--tracklets")
    run.add_argument("--detections")
    run.add_argument("--policy")
    run.add_argument("--suspected-cuts-from-tracklets", action="store_true")
    run.add_argument("--min-births", type=int, default=8)
    run.add_argument("--output", required=True)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=_calibrate)

    bench = commands.add_parser("benchmark", help="synthetic offline benchmark")
    bench.add_argument("--policy")
    bench.add_argument("--output", required=True)
    bench.add_argument("--overwrite", action="store_true")
    bench.set_defaults(handler=_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
