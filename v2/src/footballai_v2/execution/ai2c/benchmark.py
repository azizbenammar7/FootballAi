"""Bounded, leakage-aware AI2C benchmark protocol.

Protocol (fixed before evaluation identities are scored):

1. Baselines use a-priori defaults: ``review_only`` (current state: no
   automatic association), ``temporal_only`` and ``temporal_kinematic``.
2. ``temporal_kinematic`` parameters are swept on *development identities
   only*. Selection rule: zero development false merges and zero labelled
   non-player contamination; then most development true-positive pairs; ties go
   to the more conservative configuration (smaller gap, slower speed, larger
   margin).
3. The selected configuration is scored once on held-out *evaluation
   identities*, and on any extra clips supplied (never used for selection).

The resolver always runs over every tracklet in a clip (labelled or not); only
the scored pairs are restricted, so unlabelled distractors are present.
Runtime and peak traced Python memory are measurements on the executing host.
"""

from __future__ import annotations

import statistics
import time
import tracemalloc
from itertools import product
from typing import Sequence

from footballai_v2.execution.ai2a.model import (
    PlayerIdentityId,
    ground_truth_from_json_bytes,
)
from footballai_v2.execution.ai2a.review import review_samples_from_json_bytes
from footballai_v2.execution.ai2c.artifact import AssignmentStatus, PlayerIdentitiesArtifact
from footballai_v2.execution.ai2c.evaluation import evaluate_identities, label_counts
from footballai_v2.execution.ai2c.evidence import ai2a_evidence_from_evaluation_bytes
from footballai_v2.execution.ai2c.inputs import (
    source_for_legacy_raw_tracks,
    source_for_review_samples,
    summaries_from_legacy_raw_tracks,
    summaries_from_review_samples,
)
from footballai_v2.execution.ai1.artifacts import sha256_bytes
from footballai_v2.execution.ai2c.resolver import ResolverConfig, ResolverMode, resolve_identities


BENCHMARK_SCHEMA = "footballai.identity-resolution-evaluation/v1"
DEFAULT_GRID = {
    "max_gap_seconds": (1.0, 2.0, 3.0),
    "max_speed_heights_per_second": (3.0, 5.0, 8.0),
    "min_ambiguity_margin": (0.1, 0.2, 0.3),
}
BASELINES = {
    "review_only": ResolverConfig(mode=ResolverMode.REVIEW_ONLY),
    "temporal_only": ResolverConfig(mode=ResolverMode.TEMPORAL_ONLY, max_gap_seconds=2.0),
    "temporal_kinematic_a_priori": ResolverConfig(
        mode=ResolverMode.TEMPORAL_KINEMATIC, max_gap_seconds=2.0, max_speed_heights_per_second=5.0
    ),
}


class _Clip:
    def __init__(self, review_bytes: bytes, ground_truth_bytes: bytes, evaluation_bytes: bytes | None) -> None:
        review = review_samples_from_json_bytes(review_bytes)
        self.source = source_for_review_samples(review_bytes, review)
        self.summaries = summaries_from_review_samples(review)
        self.ground_truth = ground_truth_from_json_bytes(ground_truth_bytes)
        if self.ground_truth.source.tracklets_sha256 != self.source.tracklets_sha256:
            raise ValueError("ground truth and review samples reference different tracklets")
        self.teams, self.cannot_links, self.evidence_inputs = ({}, (), ())
        if evaluation_bytes is not None:
            self.teams, self.cannot_links, self.evidence_inputs = ai2a_evidence_from_evaluation_bytes(
                evaluation_bytes, tracklets_sha256=self.source.tracklets_sha256
            )
        self.dataset = {
            "source_id": self.source.source_id,
            "tracklets_sha256": self.source.tracklets_sha256,
            "video_sha256": self.source.video_sha256,
            "review_samples_sha256": sha256_bytes(review_bytes),
            "ground_truth_sha256": sha256_bytes(ground_truth_bytes),
            "ai2a_evaluation_sha256": sha256_bytes(evaluation_bytes) if evaluation_bytes else None,
            "tracklets": len(self.summaries),
            "labels": dict(label_counts(self.ground_truth)),
            "team_evidence_available": bool(self.teams),
            "explicit_cannot_links_available": len(self.cannot_links),
        }

    def resolve(self, config: ResolverConfig, *, use_evidence: bool = True) -> PlayerIdentitiesArtifact:
        return resolve_identities(
            self.summaries,
            self.source,
            config=config,
            team_evidence=self.teams if use_evidence else None,
            cannot_links=self.cannot_links if use_evidence else (),
            evidence_inputs=self.evidence_inputs if use_evidence else (),
        )

    def score(self, artifact: PlayerIdentitiesArtifact, scope: Sequence[PlayerIdentityId] | None) -> dict:
        return evaluate_identities(
            artifact, self.ground_truth, identity_scope=scope, cannot_links=self.cannot_links
        )


def _measure(callable_):
    """Time one untraced run, then trace a second run for peak Python memory."""
    started = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - started
    tracemalloc.start()
    try:
        callable_()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {"seconds": round(elapsed, 4), "peak_traced_python_bytes": peak}


def _compact(metrics: dict) -> dict:
    return {
        "pairwise": metrics["pairwise"],
        "false_merges": metrics["false_merges"]["count"],
        "labelled_non_player_contamination_pairs": metrics["labelled_non_player_contamination_pairs"],
        "cross_team_violations": metrics["cross_team_violations"],
        "uncertain_label_disagreements": metrics["uncertain_label_disagreements"]["count"],
        "temporal_overlap_violations": metrics["temporal_overlap_violations"],
        "explicit_cannot_link_violations": metrics["explicit_cannot_link_violations"],
        "fragmentation": metrics["fragmentation"],
        "unresolved": metrics["unresolved"],
    }


def _selection_key(item: dict) -> tuple:
    dev = item["development"]
    config = item["configuration"]
    safe = dev["false_merges"] == 0 and dev["labelled_non_player_contamination_pairs"] == 0
    return (
        not safe,
        dev["false_merges"],
        -dev["pairwise"]["true_positive"],
        config["max_gap_seconds"],
        config["max_speed_heights_per_second"],
        -config["min_ambiguity_margin"],
    )


def run_benchmark(
    *,
    review_samples: bytes,
    ground_truth: bytes,
    development_identities: Sequence[str],
    evaluation_identities: Sequence[str],
    ai2a_evaluation: bytes | None = None,
    extra_clips: Sequence[tuple[bytes, bytes, bytes | None]] = (),
    grid: dict = DEFAULT_GRID,
) -> dict:
    development = tuple(PlayerIdentityId(item) for item in development_identities)
    evaluation = tuple(PlayerIdentityId(item) for item in evaluation_identities)
    if set(development) & set(evaluation):
        raise ValueError("development and evaluation identities must be disjoint")
    clip = _Clip(review_samples, ground_truth, ai2a_evaluation)

    sweep = []
    for gap, speed, margin in product(
        grid["max_gap_seconds"], grid["max_speed_heights_per_second"], grid["min_ambiguity_margin"]
    ):
        config = ResolverConfig(
            mode=ResolverMode.TEMPORAL_KINEMATIC,
            max_gap_seconds=gap,
            max_speed_heights_per_second=speed,
            min_ambiguity_margin=margin,
        )
        sweep.append({
            "configuration": config.to_dict(),
            "development": _compact(clip.score(clip.resolve(config), development)),
            "_config": config,
        })
    selected = min(sweep, key=_selection_key)
    selected_config = selected["_config"]

    results = {}
    for name, config in {**BASELINES, "temporal_kinematic_selected": selected_config}.items():
        artifact, cost = _measure(lambda config=config: clip.resolve(config))
        results[name] = {
            "configuration": config.to_dict(),
            "development": _compact(clip.score(artifact, development)),
            "evaluation": _compact(clip.score(artifact, evaluation)),
            "all_labelled_diagnostic": _compact(clip.score(artifact, None)),
            "artifact_summary": dict(artifact.summary),
            "runtime": cost,
        }

    extra = []
    for review_bytes, truth_bytes, evaluation_bytes in extra_clips:
        other = _Clip(review_bytes, truth_bytes, evaluation_bytes)
        entry = {"dataset": other.dataset, "results": {}}
        for name, config in {
            "review_only": BASELINES["review_only"],
            "temporal_only": BASELINES["temporal_only"],
            "temporal_kinematic_selected": selected_config,
        }.items():
            entry["results"][name] = _compact(other.score(other.resolve(config, use_evidence=False), None))
            if other.teams or other.cannot_links:
                entry["results"][name + "+ai2a_evidence"] = _compact(other.score(other.resolve(config), None))
        extra.append(entry)

    for item in sweep:
        item.pop("_config")
    return {
        "schema": BENCHMARK_SCHEMA,
        "protocol": {
            "selection_rule": (
                "temporal_kinematic grid on development identities only: zero false merges and zero "
                "labelled non-player contamination, then most true-positive pairs, ties to the more "
                "conservative configuration"
            ),
            "development_identities": [item.value for item in development],
            "evaluation_identities": [item.value for item in evaluation],
            "grid": {key: list(value) for key, value in grid.items()},
            "extra_clips_used_for_selection": False,
        },
        "primary_dataset": clip.dataset,
        "selected_configuration": selected_config.to_dict(),
        "results": results,
        "development_sweep": sweep,
        "extra_clips": extra,
        "metric_classes": {
            "measured": [
                "pairwise precision/recall/F1 with TP/FP/FN/TN counts",
                "false merges",
                "labelled non-player contamination",
                "cross-team violations",
                "temporal overlap and explicit cannot-link violations",
                "fragmentation (units per labelled identity)",
                "unresolved rates",
            ],
            "diagnostic_proxies": ["artifact_summary", "runtime (host dependent)"],
            "not_measured": ["HOTA", "IDF1", "ID_switches", "appearance channel (AI2B cache not available)"],
        },
    }


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 3),
        "max": round(ordered[-1], 3),
    }


def run_legacy_scale_diagnostic(
    raw_tracks: bytes, *, source_id: str, config: ResolverConfig
) -> dict:
    """Full-match scale run on historical V1 tracks: no ground truth, proxies only."""
    def build():
        summaries = summaries_from_legacy_raw_tracks(raw_tracks)
        source = source_for_legacy_raw_tracks(raw_tracks, source_id=source_id)
        return summaries, resolve_identities(summaries, source, config=config)

    (summaries, artifact), cost = _measure(build)
    sizes = [len(item.tracklet_ids) for item in artifact.identities]
    observed = [item.observed_seconds for item in artifact.identities]
    durations = {item.tracklet_id: item.duration_seconds for item in summaries}
    unresolved = [
        durations[assignment.tracklet_id]
        for assignment in artifact.assignments
        if assignment.status is AssignmentStatus.UNRESOLVED
    ]
    units = len(artifact.identities) + artifact.summary["unresolved_tracklets"]
    return {
        "input": {
            "source_id": source_id,
            "input_kind": artifact.source.input_kind.value,
            "sha256": artifact.source.input_sha256,
            "tracklets": len(summaries),
            "observations": sum(item.observation_count or 0 for item in summaries),
            "duration_seconds": round(max(item.end_time_seconds for item in summaries), 3),
        },
        "configuration": config.to_dict(),
        "artifact_summary": dict(artifact.summary),
        "units_before": len(summaries),
        "units_after": units,
        "identity_size_distribution": _distribution(sizes),
        "identity_observed_seconds": _distribution(observed),
        "unresolved_tracklet_seconds": _distribution(unresolved),
        "runtime": cost,
        "interpretation": (
            "Diagnostic proxy only. The legacy table has no identity labels, so these counts say "
            "nothing about association correctness."
        ),
    }
