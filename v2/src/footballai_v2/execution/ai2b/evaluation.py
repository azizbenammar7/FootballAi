"""Identity-disjoint, bounded comparison of temporal, team, and appearance evidence."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Mapping, Sequence

from footballai_v2.execution.ai2a.association import CannotLink, TeamEvidence
from footballai_v2.execution.ai2a.model import (
    AnnotationStatus,
    IdentityGroundTruthArtifact,
    PlayerIdentityId,
    TeamLabel,
    TrackletId,
)
from footballai_v2.execution.ai2a.review import ReviewSamplesArtifact
from footballai_v2.execution.ai2b.artifact import AppearanceEmbeddingsArtifact
from footballai_v2.execution.ai2b.association import cosine_similarity, select_similarity_threshold
from footballai_v2.execution.ai2b.metrics import binary_pair_metrics, team_classification_report


APPEARANCE_EVALUATION_SCHEMA = "footballai.appearance-identity-evaluation/v1"


@dataclass(frozen=True, slots=True)
class LabeledPair:
    left: TrackletId
    right: TrackletId
    same_player: bool
    temporal_compatible: bool
    team_compatible: bool
    appearance_similarity: float | None


def identity_disjoint_split(
    identities: Sequence[PlayerIdentityId],
) -> tuple[tuple[PlayerIdentityId, ...], tuple[PlayerIdentityId, ...]]:
    ordered = tuple(sorted(set(identities)))
    if len(ordered) < 4:
        raise ValueError("identity-disjoint split requires at least four identities")
    development = ordered[::2]
    evaluation = ordered[1::2]
    if not development or not evaluation:
        raise ValueError("identity-disjoint split requires both partitions")
    return development, evaluation


def _team_compatible(
    left: TrackletId,
    right: TrackletId,
    evidence: Mapping[TrackletId, TeamEvidence],
    confidence_threshold: float,
) -> bool:
    left_team = evidence.get(left, TeamEvidence(TeamLabel.UNKNOWN, 0.0, "unavailable"))
    right_team = evidence.get(right, TeamEvidence(TeamLabel.UNKNOWN, 0.0, "unavailable"))
    return not (
        left_team.team is not TeamLabel.UNKNOWN
        and right_team.team is not TeamLabel.UNKNOWN
        and left_team.team is not right_team.team
        and left_team.confidence >= confidence_threshold
        and right_team.confidence >= confidence_threshold
    )


def _pairs_for_identities(
    ground_truth: IdentityGroundTruthArtifact,
    identities: set[PlayerIdentityId],
    embeddings: Mapping[TrackletId, tuple[float, ...]],
    cannot_links: Sequence[CannotLink],
    team_evidence: Mapping[TrackletId, TeamEvidence],
    team_confidence_threshold: float,
) -> tuple[LabeledPair, ...]:
    labelled = {
        item.tracklet_id: item.ground_truth_player_id
        for item in ground_truth.annotations
        if item.status is AnnotationStatus.CONFIRMED
        and item.ground_truth_player_id is not None
        and item.ground_truth_player_id in identities
    }
    forbidden = {frozenset((item.left, item.right)) for item in cannot_links}
    result = []
    for left, right in combinations(sorted(labelled), 2):
        temporal = frozenset((left, right)) not in forbidden
        similarity = (
            cosine_similarity(embeddings[left], embeddings[right])
            if left in embeddings and right in embeddings
            else None
        )
        result.append(
            LabeledPair(
                left,
                right,
                labelled[left] == labelled[right],
                temporal,
                _team_compatible(left, right, team_evidence, team_confidence_threshold),
                similarity,
            )
        )
    return tuple(result)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"support": 0, "median": None, "p10": None, "p90": None}

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "support": len(ordered),
        "median": statistics.median(ordered),
        "p10": percentile(0.1),
        "p90": percentile(0.9),
    }


def _review_metadata(
    pairs: Sequence[LabeledPair],
    predictions: Sequence[bool],
    samples: ReviewSamplesArtifact,
    *,
    false_positive: bool,
    limit: int = 5,
) -> list[dict]:
    paths: dict[TrackletId, list[str]] = {}
    for sample in samples.samples:
        paths.setdefault(sample.tracklet_id, []).append(sample.relative_path)
    selected = []
    for pair, predicted in zip(pairs, predictions):
        is_error = (predicted and not pair.same_player) if false_positive else (
            pair.same_player and not predicted
        )
        if not is_error:
            continue
        selected.append({
            "left": pair.left.value,
            "right": pair.right.value,
            "appearance_similarity": pair.appearance_similarity,
            "temporal_compatible": pair.temporal_compatible,
            "team_compatible": pair.team_compatible,
            "left_crops": paths.get(pair.left, []),
            "right_crops": paths.get(pair.right, []),
        })
        if len(selected) == limit:
            break
    return selected


def build_ai2b_evaluation_report(
    ground_truth: IdentityGroundTruthArtifact,
    appearance: AppearanceEmbeddingsArtifact,
    review_samples: ReviewSamplesArtifact,
    team_evidence: Mapping[TrackletId, TeamEvidence],
    cannot_links: Sequence[CannotLink],
    *,
    total_tracklets: int,
    annotation_effort_minutes: float,
    team_method: str,
    detector_executions: int = 0,
    crop_extraction_seconds: float | None = None,
    team_confidence_threshold: float = 0.8,
    performance: Mapping[str, float | int | str] | None = None,
) -> dict:
    appearance.validate_review_samples(review_samples)
    identities = tuple(
        item.ground_truth_player_id
        for item in ground_truth.annotations
        if item.status is AnnotationStatus.CONFIRMED and item.ground_truth_player_id is not None
    )
    development_ids, evaluation_ids = identity_disjoint_split(identities)
    embeddings = {item.tracklet_id: item.embedding for item in appearance.tracklet_embeddings}
    development_pairs = _pairs_for_identities(
        ground_truth, set(development_ids), embeddings, cannot_links,
        team_evidence, team_confidence_threshold,
    )
    evaluation_pairs = _pairs_for_identities(
        ground_truth, set(evaluation_ids), embeddings, cannot_links,
        team_evidence, team_confidence_threshold,
    )
    threshold_samples = tuple(
        (pair.appearance_similarity, pair.same_player)
        for pair in development_pairs
        if pair.temporal_compatible and pair.team_compatible and pair.appearance_similarity is not None
    )
    selection = select_similarity_threshold(threshold_samples)
    truth = tuple(pair.same_player for pair in evaluation_pairs)
    temporal_predictions = tuple(pair.temporal_compatible for pair in evaluation_pairs)
    team_predictions = tuple(
        pair.temporal_compatible and pair.team_compatible for pair in evaluation_pairs
    )
    appearance_predictions = tuple(
        pair.temporal_compatible
        and pair.team_compatible
        and pair.appearance_similarity is not None
        and pair.appearance_similarity >= selection.threshold
        for pair in evaluation_pairs
    )
    positive_support = sum(truth)
    candidate_positive = sum(
        pair.same_player
        and pair.temporal_compatible
        and pair.team_compatible
        and pair.appearance_similarity is not None
        for pair in evaluation_pairs
    )
    confirmed = [
        item for item in ground_truth.annotations
        if item.status is AnnotationStatus.CONFIRMED and item.ground_truth_player_id is not None
    ]
    full_positive_support = sum(
        count * (count - 1) // 2
        for identity in set(identities)
        for count in [sum(item.ground_truth_player_id == identity for item in confirmed)]
    )
    full_pair_support = len(confirmed) * (len(confirmed) - 1) // 2
    truth_teams = {item.tracklet_id: item.team_label for item in ground_truth.annotations}
    positive_similarities = [
        pair.appearance_similarity for pair in evaluation_pairs
        if pair.same_player and pair.appearance_similarity is not None
    ]
    negative_similarities = [
        pair.appearance_similarity for pair in evaluation_pairs
        if not pair.same_player and pair.appearance_similarity is not None
    ]
    return {
        "schema": APPEARANCE_EVALUATION_SCHEMA,
        "source": asdict(ground_truth.source),
        "scope": {
            "purpose": "bounded appearance identity benchmark",
            "persistent_identity_resolver": False,
            "appearance_similarity_is_identity": False,
            "results_are_exploratory": True,
        },
        "dataset": {
            "video_duration_seconds": ground_truth.source.duration_seconds,
            "tracklets": total_tracklets,
            "annotated_tracklets": len(ground_truth.annotations),
            "known_identities": len(set(identities)),
            "team_a_identities": len({item.ground_truth_player_id for item in confirmed if item.team_label is TeamLabel.TEAM_A}),
            "team_b_identities": len({item.ground_truth_player_id for item in confirmed if item.team_label is TeamLabel.TEAM_B}),
            "unknown": sum(item.ground_truth_player_id is None for item in ground_truth.annotations),
            "uncertain": sum(item.status is AnnotationStatus.UNCERTAIN for item in ground_truth.annotations),
            "positive_pair_support": full_positive_support,
            "negative_pair_support": full_pair_support - full_positive_support,
            "annotation_effort_minutes": annotation_effort_minutes,
            "gate": "PASS",
        },
        "cached_pipeline": {
            "detector_executions": detector_executions,
            "detections_sha256": ground_truth.source.detections_sha256,
            "tracklets_sha256": ground_truth.source.tracklets_sha256,
        },
        "appearance_model": asdict(appearance.model),
        "appearance_cache": {
            "schema": appearance.schema,
            "crop_embeddings": len(appearance.crop_embeddings),
            "tracklet_embeddings": len(appearance.tracklet_embeddings),
            "rejected_crops": len(appearance.rejections),
        },
        "split": {
            "method": "deterministic identity-disjoint alternating split/v1",
            "development_identities": [item.value for item in development_ids],
            "evaluation_identities": [item.value for item in evaluation_ids],
            "development_pairs": len(development_pairs),
            "evaluation_pairs": len(evaluation_pairs),
            "threshold": selection.threshold,
            "threshold_selection_metrics": dict(selection.metrics),
        },
        "team_evidence": {
            "method": team_method,
            "confidence_guard_threshold": team_confidence_threshold,
            "metrics": team_classification_report(truth_teams, team_evidence),
        },
        "similarity_distributions": {
            "positive": _distribution(positive_similarities),
            "negative": _distribution(negative_similarities),
        },
        "association_comparison": {
            "A_temporal_only": binary_pair_metrics(truth, temporal_predictions),
            "B_temporal_team": binary_pair_metrics(truth, team_predictions),
            "C_temporal_team_appearance": binary_pair_metrics(truth, appearance_predictions),
        },
        "candidate_pair_recall": {
            "recall": candidate_positive / positive_support if positive_support else 0.0,
            "support": positive_support,
            "available_positive_pairs": candidate_positive,
        },
        "error_review": {
            "false_positives": _review_metadata(
                evaluation_pairs, appearance_predictions, review_samples, false_positive=True
            ),
            "false_negatives": _review_metadata(
                evaluation_pairs, appearance_predictions, review_samples, false_positive=False
            ),
        },
        "performance": {
            **dict(performance or {}),
            "crop_extraction_seconds": crop_extraction_seconds,
        },
    }


def ai2b_evaluation_to_json_bytes(report: dict) -> bytes:
    if report.get("schema") != APPEARANCE_EVALUATION_SCHEMA:
        raise ValueError(f"schema must be {APPEARANCE_EVALUATION_SCHEMA}")
    return (json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
