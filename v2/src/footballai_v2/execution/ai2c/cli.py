"""Offline AI2C developer CLI.

It reads cached artifacts only: it imports no detector, opens no video, and
makes no network call. Outputs are written to new paths; existing files are
never overwritten unless ``--overwrite`` is passed.

``resolve`` defaults to ``--mode review_only``: only human corrections create
identities. ``temporal_only`` and ``temporal_kinematic`` are experimental and
must be passed explicitly; their identities are always ``provisional``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from footballai_v2.execution.ai1.artifacts import sha256_bytes, tracklets_from_parquet_bytes
from footballai_v2.execution.ai2a.model import PlayerIdentityId, ground_truth_from_json_bytes
from footballai_v2.execution.ai2a.review import review_samples_from_json_bytes
from footballai_v2.execution.ai2c.artifact import EvidenceInput, player_identities_from_json_bytes, player_identities_to_json_bytes
from footballai_v2.execution.ai2c.benchmark import run_benchmark, run_legacy_scale_diagnostic
from footballai_v2.execution.ai2c.corrections import (
    CORRECTIONS_SCHEMA,
    corrections_from_json_bytes,
    corrections_to_json_bytes,
)
from footballai_v2.execution.ai2c.evaluation import evaluate_identities
from footballai_v2.execution.ai2c.evidence import ai2a_evidence_from_evaluation_bytes
from footballai_v2.execution.ai2c.inputs import (
    source_for_legacy_raw_tracks,
    source_for_review_samples,
    source_for_tracklets,
    summaries_from_legacy_raw_tracks,
    summaries_from_review_samples,
    summaries_from_tracklets,
)
from footballai_v2.execution.ai2c.resolver import (
    AUTOMATIC_MODES,
    SELECTED_TEMPORAL_KINEMATIC,
    ResolverConfig,
    ResolverMode,
    resolve_identities,
)


# Numeric parameters only; the mode default is review_only (see build_parser).
_DEFAULTS = SELECTED_TEMPORAL_KINEMATIC


def _write(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _json(value: dict) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _config(args: argparse.Namespace) -> ResolverConfig:
    return ResolverConfig(
        mode=ResolverMode(args.mode),
        max_gap_seconds=args.max_gap_seconds,
        max_speed_heights_per_second=args.max_speed_heights_per_second,
        min_ambiguity_margin=args.min_ambiguity_margin,
    )


def _resolve(args: argparse.Namespace) -> int:
    config = _config(args)
    if config.mode in AUTOMATIC_MODES:
        print(
            f"warning: --mode {config.mode.value} is experimental automatic association "
            "(zero held-out true positives, one certain false merge in the AI2C benchmark); "
            "its identities are provisional review candidates, not verified players",
            file=sys.stderr,
        )
    if args.tracklets:
        content = args.tracklets.read_bytes()
        artifact = tracklets_from_parquet_bytes(content)
        if not args.source_id:
            raise SystemExit("--source-id is required with --tracklets")
        summaries = summaries_from_tracklets(artifact)
        source = source_for_tracklets(content, artifact, source_id=args.source_id, video_sha256=args.video_sha256)
    elif args.review_samples:
        if args.source_id or args.video_sha256:
            raise SystemExit("--source-id/--video-sha256 come from the review samples; do not pass them")
        content = args.review_samples.read_bytes()
        review = review_samples_from_json_bytes(content)
        summaries = summaries_from_review_samples(review)
        source = source_for_review_samples(content, review)
    else:
        if args.video_sha256:
            raise SystemExit("legacy raw tracks carry no video checksum; do not pass --video-sha256")
        content = args.legacy_raw_tracks.read_bytes()
        if not args.source_id:
            raise SystemExit("--source-id is required with --legacy-raw-tracks")
        summaries = summaries_from_legacy_raw_tracks(content)
        source = source_for_legacy_raw_tracks(content, source_id=args.source_id)
    teams, links, inputs = {}, (), []
    if args.ai2a_evaluation:
        teams, links, ai2a_inputs = ai2a_evidence_from_evaluation_bytes(
            args.ai2a_evaluation.read_bytes(), tracklets_sha256=source.tracklets_sha256
        )
        inputs.extend(ai2a_inputs)
    corrections = None
    if args.corrections:
        raw = args.corrections.read_bytes()
        corrections = corrections_from_json_bytes(raw)
        if corrections_to_json_bytes(corrections) != raw:
            # The recorded SHA-256 must be verifiable from the log itself, so
            # only the canonical serialization is accepted.
            raise SystemExit(
                f"{args.corrections} is not a canonical footballai.identity-corrections/v1 file "
                "(sorted keys, 2-space indent, trailing newline); rewrite it with corrections_to_json_bytes"
            )
        inputs.append(EvidenceInput("corrections", CORRECTIONS_SCHEMA, sha256_bytes(raw)))
    result = resolve_identities(
        summaries, source, config=config, team_evidence=teams, cannot_links=links,
        corrections=corrections, evidence_inputs=inputs,
    )
    content = player_identities_to_json_bytes(result)
    _write(args.output, content, args.overwrite)
    print(json.dumps({"output": str(args.output), "sha256": sha256_bytes(content), **result.summary}, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    identities = player_identities_from_json_bytes(args.identities.read_bytes())
    truth = ground_truth_from_json_bytes(args.ground_truth.read_bytes())
    scope = [PlayerIdentityId(item) for item in args.identity_scope.split(",")] if args.identity_scope else None
    links = ()
    if args.ai2a_evaluation:
        _, links, _ = ai2a_evidence_from_evaluation_bytes(
            args.ai2a_evaluation.read_bytes(), tracklets_sha256=identities.source.tracklets_sha256
        )
    report = evaluate_identities(identities, truth, identity_scope=scope, cannot_links=links)
    _write(args.output, _json(report), args.overwrite)
    print(json.dumps(report["pairwise"], sort_keys=True))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    extra = []
    for item in args.extra_clip or []:
        parts = [Path(value) for value in item.split(",")]
        if len(parts) not in (2, 3):
            raise SystemExit("--extra-clip takes REVIEW_SAMPLES,GROUND_TRUTH[,AI2A_EVALUATION]")
        extra.append((parts[0].read_bytes(), parts[1].read_bytes(), parts[2].read_bytes() if len(parts) == 3 else None))
    report = run_benchmark(
        review_samples=args.review_samples.read_bytes(),
        ground_truth=args.ground_truth.read_bytes(),
        development_identities=args.development_identities.split(","),
        evaluation_identities=args.evaluation_identities.split(","),
        ai2a_evaluation=args.ai2a_evaluation.read_bytes() if args.ai2a_evaluation else None,
        extra_clips=extra,
    )
    if args.legacy_raw_tracks:
        config = ResolverConfig.from_dict(report["selected_configuration"])
        report["full_match_scale_diagnostic"] = run_legacy_scale_diagnostic(
            args.legacy_raw_tracks.read_bytes(), source_id=args.legacy_source_id, config=config
        )
    _write(args.output, _json(report), args.overwrite)
    print(json.dumps(report["results"]["temporal_kinematic_selected"]["evaluation"]["pairwise"], sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI2C conservative player-identity association (offline)")
    commands = parser.add_subparsers(dest="command", required=True)

    resolve = commands.add_parser("resolve", help="write a footballai.player-identities/v1 artifact")
    source = resolve.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracklets", type=Path, help="AI1 footballai.tracklets/v1 Parquet")
    source.add_argument("--review-samples", type=Path, help="AI2A review samples (reconstructed extents)")
    source.add_argument("--legacy-raw-tracks", type=Path, help="historical V1 raw_tracks.parquet (diagnostic)")
    resolve.add_argument("--source-id")
    resolve.add_argument("--video-sha256")
    resolve.add_argument("--ai2a-evaluation", type=Path, help="AI2A identity-evaluation JSON (team guard, cannot-links)")
    resolve.add_argument("--corrections", type=Path, help="footballai.identity-corrections/v1 JSON")
    resolve.add_argument(
        "--mode",
        choices=[item.value for item in ResolverMode],
        default=ResolverMode.REVIEW_ONLY.value,
        help="review_only (default, safe); temporal_only and temporal_kinematic are experimental opt-ins",
    )
    resolve.add_argument("--max-gap-seconds", type=float, default=_DEFAULTS.max_gap_seconds)
    resolve.add_argument(
        "--max-speed-heights-per-second", type=float, default=_DEFAULTS.max_speed_heights_per_second
    )
    resolve.add_argument("--min-ambiguity-margin", type=float, default=_DEFAULTS.min_ambiguity_margin)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--overwrite", action="store_true")
    resolve.set_defaults(handler=_resolve)

    evaluate = commands.add_parser("evaluate", help="score identities against AI2A ground truth")
    evaluate.add_argument("--identities", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--identity-scope", help="comma-separated ground-truth identities to score")
    evaluate.add_argument("--ai2a-evaluation", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_evaluate)

    benchmark = commands.add_parser("benchmark", help="run the bounded leakage-aware comparison")
    benchmark.add_argument("--review-samples", type=Path, required=True)
    benchmark.add_argument("--ground-truth", type=Path, required=True)
    benchmark.add_argument("--ai2a-evaluation", type=Path)
    benchmark.add_argument("--development-identities", required=True)
    benchmark.add_argument("--evaluation-identities", required=True)
    benchmark.add_argument("--extra-clip", action="append")
    benchmark.add_argument("--legacy-raw-tracks", type=Path)
    benchmark.add_argument("--legacy-source-id", default="v1-legacy-full-match")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.set_defaults(handler=_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
