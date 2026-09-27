"""Developer CLI for bounded AI2A manual review and evaluation.

The CLI never imports the detector and never assigns player identities
automatically. ``assign`` is the only command that writes identity labels.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from footballai_v2.execution.ai1.artifacts import (
    detections_from_parquet_bytes,
    sha256_bytes,
    tracklets_from_parquet_bytes,
)
from footballai_v2.execution.ai2a.association import (
    build_candidate_graph,
    classify_team,
    generate_temporal_cannot_links,
)
from footballai_v2.execution.ai2a.evaluation import (
    build_evaluation_report,
    evaluation_to_json_bytes,
)
from footballai_v2.execution.ai2a.model import (
    GROUND_TRUTH_SCHEMA,
    AnnotationStatus,
    IdentityGroundTruthArtifact,
    PlayerIdentityId,
    ReviewState,
    SourceReference,
    TeamLabel,
    TrackletAnnotation,
    TrackletId,
    ground_truth_from_json_bytes,
    ground_truth_to_json_bytes,
    tracklet_id,
)
from footballai_v2.execution.ai2a.review import (
    appearances_from_review_samples,
    build_review_samples,
    render_contact_sheets,
    render_review_sheet,
    review_samples_from_json_bytes,
    review_samples_to_json_bytes,
)


def _read_tracklets(path: Path):
    return tracklets_from_parquet_bytes(path.read_bytes())


def _prepare(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_content = args.detections.read_bytes()
    tracklet_content = args.tracklets.read_bytes()
    detections = detections_from_parquet_bytes(detection_content)
    tracklets = tracklets_from_parquet_bytes(tracklet_content)
    video_sha256 = hashlib.sha256(args.video.read_bytes()).hexdigest()
    if detections.provenance.source_video_sha256 != video_sha256:
        raise ValueError("video does not match cached detections")
    if tracklets.provenance.detections_sha256 != sha256_bytes(detection_content):
        raise ValueError("tracklets do not reference the supplied detections artifact")
    source = SourceReference(
        source_id=args.source_id,
        video_sha256=video_sha256,
        duration_seconds=detections.provenance.source_duration_seconds,
        detections_sha256=sha256_bytes(detection_content),
        tracklets_sha256=sha256_bytes(tracklet_content),
    )
    samples = build_review_samples(args.video, tracklets, source, output_dir)
    ground_truth = IdentityGroundTruthArtifact(GROUND_TRUTH_SCHEMA, source, ())
    (output_dir / "review-samples.json").write_bytes(review_samples_to_json_bytes(samples))
    (output_dir / "identity-ground-truth.json").write_bytes(ground_truth_to_json_bytes(ground_truth))
    (output_dir / "review.html").write_text(
        render_review_sheet(tracklets, samples, ground_truth), encoding="utf-8"
    )
    contact_sheets = render_contact_sheets(samples, output_dir)
    print(
        f"prepared {len(tracklets.tracklets)} tracklets and {len(samples.samples)} deterministic crops "
        f"across {len(contact_sheets)} contact sheets in {output_dir}"
    )
    return 0


def _list(args: argparse.Namespace) -> int:
    tracklets = _read_tracklets(args.tracklets)
    samples = review_samples_from_json_bytes(args.review_samples.read_bytes())
    ground_truth = ground_truth_from_json_bytes(args.ground_truth.read_bytes(), tracklets=tracklets)
    annotations = {item.tracklet_id: item for item in ground_truth.annotations}
    sample_paths: dict[TrackletId, list[str]] = {}
    for sample in samples.samples:
        sample_paths.setdefault(sample.tracklet_id, []).append(sample.relative_path)
    for item in sorted(tracklets.tracklets, key=lambda value: value.track_id):
        identifier = tracklet_id(item.track_id)
        annotation = annotations.get(identifier)
        identity = annotation.ground_truth_player_id.value if annotation and annotation.ground_truth_player_id else "UNKNOWN"
        team = annotation.team_label.value if annotation else TeamLabel.UNKNOWN.value
        status = annotation.status.value if annotation else "unreviewed"
        print(
            f"{identifier.value} {item.start_time_seconds:.3f}-{item.end_time_seconds:.3f}s "
            f"observations={item.observation_count} team={team} identity={identity} status={status} "
            f"samples={','.join(sample_paths.get(identifier, []))}"
        )
    return 0


def _assign(args: argparse.Namespace) -> int:
    tracklets = _read_tracklets(args.tracklets)
    artifact = ground_truth_from_json_bytes(args.ground_truth.read_bytes(), tracklets=tracklets)
    identifier = TrackletId(args.tracklet)
    by_id = {tracklet_id(item.track_id): item for item in tracklets.tracklets}
    item = by_id.get(identifier)
    if item is None:
        raise ValueError(f"unknown tracklet reference: {identifier.value}")
    identity = None if args.identity == "UNKNOWN" else PlayerIdentityId(args.identity)
    status = AnnotationStatus(args.status)
    if identity is None:
        status = AnnotationStatus.UNKNOWN
    annotation = TrackletAnnotation(
        tracklet_id=identifier,
        ground_truth_player_id=identity,
        team_label=TeamLabel(args.team),
        role_label=args.role,
        start_frame=item.start_frame,
        end_frame=item.end_frame,
        start_time_seconds=item.start_time_seconds,
        end_time_seconds=item.end_time_seconds,
        annotation_confidence=args.confidence,
        status=status,
        annotator=args.annotator,
        source=args.source,
        notes=args.notes,
        review_state=ReviewState(args.review_state),
    )
    annotations = [value for value in artifact.annotations if value.tracklet_id != identifier]
    annotations.append(annotation)
    updated = replace(artifact, annotations=tuple(sorted(annotations, key=lambda value: value.tracklet_id)))
    updated.validate_against_tracklets(tracklets)
    constraints = generate_temporal_cannot_links(tracklets)
    updated.validate_identity_assignments({frozenset((item.left, item.right)) for item in constraints})
    args.ground_truth.write_bytes(ground_truth_to_json_bytes(updated))
    print(f"saved {identifier.value}: {args.team} / {args.identity} / {status.value}")
    return 0


def _render(args: argparse.Namespace) -> int:
    tracklets = _read_tracklets(args.tracklets)
    samples = review_samples_from_json_bytes(args.review_samples.read_bytes())
    ground_truth = ground_truth_from_json_bytes(args.ground_truth.read_bytes(), tracklets=tracklets)
    args.output.write_text(render_review_sheet(tracklets, samples, ground_truth), encoding="utf-8")
    print(f"rendered {args.output}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    tracklet_content = args.tracklets.read_bytes()
    tracklets = tracklets_from_parquet_bytes(tracklet_content)
    samples = review_samples_from_json_bytes(args.review_samples.read_bytes())
    ground_truth = ground_truth_from_json_bytes(args.ground_truth.read_bytes(), tracklets=tracklets)
    if samples.source != ground_truth.source:
        raise ValueError("review samples and ground truth reference different sources")
    if ground_truth.source.tracklets_sha256 != sha256_bytes(tracklet_content):
        raise ValueError("ground truth does not reference the supplied tracklets")
    appearances = appearances_from_review_samples(samples, args.review_samples.parent)
    team_a_anchor = TrackletId(args.team_a_anchor)
    team_b_anchor = TrackletId(args.team_b_anchor) if args.team_b_anchor else None
    for anchor in tuple(value for value in (team_a_anchor, team_b_anchor) if value is not None):
        if anchor not in appearances:
            raise ValueError(f"team prototype tracklet has no appearance evidence: {anchor.value}")
    prototypes = {
        TeamLabel.TEAM_A: (
            appearances[team_a_anchor].hue,
            appearances[team_a_anchor].saturation,
            appearances[team_a_anchor].value,
        ),
    }
    if team_b_anchor is not None:
        prototypes[TeamLabel.TEAM_B] = (
            appearances[team_b_anchor].hue,
            appearances[team_b_anchor].saturation,
            appearances[team_b_anchor].value,
        )
    teams = {key: classify_team(value, prototypes) for key, value in appearances.items()}
    constraints = generate_temporal_cannot_links(tracklets)
    graph = build_candidate_graph(tracklets, teams, appearances, constraints)
    report = build_evaluation_report(
        ground_truth,
        tracklets,
        teams,
        constraints,
        graph,
        team_prototype_tracklets={
            key: value
            for key, value in {"TEAM_A": team_a_anchor, "TEAM_B": team_b_anchor}.items()
            if value is not None
        },
        annotation_effort_minutes=args.annotation_effort_minutes,
    )
    args.output.write_bytes(evaluation_to_json_bytes(report))
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="extract deterministic review crops")
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--detections", type=Path, required=True)
    prepare.add_argument("--tracklets", type=Path, required=True)
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(handler=_prepare)

    listing = commands.add_parser("list", help="list tracklets, evidence paths, and current labels")
    listing.add_argument("--tracklets", type=Path, required=True)
    listing.add_argument("--review-samples", type=Path, required=True)
    listing.add_argument("--ground-truth", type=Path, required=True)
    listing.set_defaults(handler=_list)

    assign = commands.add_parser("assign", help="manually assign or replace one tracklet label")
    assign.add_argument("--tracklets", type=Path, required=True)
    assign.add_argument("--ground-truth", type=Path, required=True)
    assign.add_argument("--tracklet", required=True)
    assign.add_argument("--identity", required=True, help="pseudonym such as TEAM_A_04 or UNKNOWN")
    assign.add_argument("--team", choices=[item.value for item in TeamLabel], required=True)
    assign.add_argument("--role", default="PLAYER")
    assign.add_argument("--status", choices=[item.value for item in AnnotationStatus], default="confirmed")
    assign.add_argument("--confidence", type=float)
    assign.add_argument("--annotator", required=True)
    assign.add_argument("--source", default="manual_crop_review")
    assign.add_argument("--notes", default="")
    assign.add_argument("--review-state", choices=[item.value for item in ReviewState], default="reviewed")
    assign.set_defaults(handler=_assign)

    render = commands.add_parser("render", help="reload labels into the local HTML review sheet")
    render.add_argument("--tracklets", type=Path, required=True)
    render.add_argument("--review-samples", type=Path, required=True)
    render.add_argument("--ground-truth", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.set_defaults(handler=_render)

    evaluate = commands.add_parser("evaluate", help="compute the bounded AI2A baseline")
    evaluate.add_argument("--tracklets", type=Path, required=True)
    evaluate.add_argument("--review-samples", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--team-a-anchor", required=True)
    evaluate.add_argument("--team-b-anchor", help="optional when the bounded clip contains no TEAM_B players")
    evaluate.add_argument("--annotation-effort-minutes", type=float, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
