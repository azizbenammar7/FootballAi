"""Offline developer CLI for the bounded AI2B appearance benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from footballai_v2.execution.ai1.artifacts import tracklets_from_parquet_bytes
from footballai_v2.execution.ai2a.association import classify_team, generate_temporal_cannot_links
from footballai_v2.execution.ai2a.model import TeamLabel, TrackletId, ground_truth_from_json_bytes
from footballai_v2.execution.ai2a.review import (
    appearances_from_review_samples,
    review_samples_from_json_bytes,
)
from footballai_v2.execution.ai2b.appearance import MobileNetV3SmallEncoder
from footballai_v2.execution.ai2b.artifact import (
    appearance_embeddings_from_json_bytes,
    appearance_embeddings_to_json_bytes,
)
from footballai_v2.execution.ai2b.evaluation import (
    ai2b_evaluation_to_json_bytes,
    build_ai2b_evaluation_report,
)
from footballai_v2.execution.ai2b.pipeline import build_appearance_embeddings


def _embed(args: argparse.Namespace) -> int:
    samples = review_samples_from_json_bytes(args.review_samples.read_bytes())
    encoder = MobileNetV3SmallEncoder(
        args.weights,
        model_sha256=args.model_sha256,
        model_version=args.model_version,
        device=args.device,
    )
    started = time.perf_counter()
    artifact = build_appearance_embeddings(samples, args.review_samples.parent, encoder)
    elapsed = time.perf_counter() - started
    content = appearance_embeddings_to_json_bytes(artifact)
    args.output.write_bytes(content)
    performance = {
        "embedding_inference_seconds": round(elapsed, 6),
        "crop_embeddings": len(artifact.crop_embeddings),
        "rejected_crops": len(artifact.rejections),
        "embedding_dimension": artifact.model.embedding_dimension,
        "embeddings_per_second": round(len(artifact.crop_embeddings) / elapsed, 6) if elapsed else 0.0,
        "device": artifact.model.device,
        "artifact_size_bytes": len(content),
    }
    args.performance_output.write_text(
        json.dumps(performance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(performance, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    tracklets = tracklets_from_parquet_bytes(args.tracklets.read_bytes())
    samples = review_samples_from_json_bytes(args.review_samples.read_bytes())
    ground_truth = ground_truth_from_json_bytes(args.ground_truth.read_bytes(), tracklets=tracklets)
    appearance = appearance_embeddings_from_json_bytes(args.appearance_embeddings.read_bytes())
    appearance.validate_review_samples(samples)
    colors = appearances_from_review_samples(samples, args.review_samples.parent)
    anchors = {
        TeamLabel.TEAM_A: colors[TrackletId(args.team_a_anchor)],
        TeamLabel.TEAM_B: colors[TrackletId(args.team_b_anchor)],
    }
    prototypes = {
        team: (value.hue, value.saturation, value.value) for team, value in anchors.items()
    }
    teams = {identifier: classify_team(value, prototypes) for identifier, value in colors.items()}
    constraints = generate_temporal_cannot_links(tracklets)
    performance = json.loads(args.performance.read_text(encoding="utf-8"))
    started = time.perf_counter()
    report = build_ai2b_evaluation_report(
        ground_truth,
        appearance,
        samples,
        teams,
        constraints,
        total_tracklets=len(tracklets.tracklets),
        annotation_effort_minutes=args.annotation_effort_minutes,
        team_method="central_upper_body_hsv_nearest_manual_anchor/v1",
        detector_executions=args.detector_executions,
        crop_extraction_seconds=args.crop_extraction_seconds,
        performance=performance,
    )
    pair_seconds = time.perf_counter() - started
    report["performance"]["pair_scoring_evaluation_seconds"] = round(pair_seconds, 6)
    args.output.write_bytes(ai2b_evaluation_to_json_bytes(report))
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    embed = commands.add_parser("embed")
    embed.add_argument("--review-samples", type=Path, required=True)
    embed.add_argument("--weights", type=Path, required=True)
    embed.add_argument("--model-sha256", required=True)
    embed.add_argument("--model-version", required=True)
    embed.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--performance-output", type=Path, required=True)
    embed.set_defaults(handler=_embed)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--tracklets", type=Path, required=True)
    evaluate.add_argument("--review-samples", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--appearance-embeddings", type=Path, required=True)
    evaluate.add_argument("--performance", type=Path, required=True)
    evaluate.add_argument("--team-a-anchor", required=True)
    evaluate.add_argument("--team-b-anchor", required=True)
    evaluate.add_argument("--annotation-effort-minutes", type=float, required=True)
    evaluate.add_argument("--detector-executions", type=int, required=True)
    evaluate.add_argument("--crop-extraction-seconds", type=float, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
