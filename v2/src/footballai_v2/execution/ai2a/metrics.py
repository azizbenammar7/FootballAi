"""Metrics supported by labelled AI2A tracklet identity ground truth."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

from footballai_v2.execution.ai1.artifacts import TrackletsArtifact
from footballai_v2.execution.ai2a.association import TeamEvidence
from footballai_v2.execution.ai2a.model import (
    AnnotationStatus,
    PlayerIdentityId,
    TeamLabel,
    TrackletAnnotation,
    TrackletId,
    tracklet_id,
)


SUPPORTED_METRICS = frozenset({
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "candidate_pair_recall",
    "team_accuracy",
    "team_rejection_rate",
    "identity_observation_coverage",
})
UNSUPPORTED_METRICS = frozenset({"HOTA", "IDF1", "ID_switches"})


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confirmed_by_tracklet(
    annotations: Sequence[TrackletAnnotation],
) -> dict[TrackletId, PlayerIdentityId]:
    return {
        item.tracklet_id: item.ground_truth_player_id
        for item in annotations
        if item.status is AnnotationStatus.CONFIRMED and item.ground_truth_player_id is not None
    }


def _ground_truth_positive_pairs(
    annotations: Sequence[TrackletAnnotation],
) -> set[frozenset[TrackletId]]:
    labelled = _confirmed_by_tracklet(annotations)
    return {
        frozenset((left, right))
        for left, right in combinations(sorted(labelled), 2)
        if labelled[left] == labelled[right]
    }


def pairwise_association_metrics(
    annotations: Sequence[TrackletAnnotation],
    predicted_pairs: set[frozenset[TrackletId]],
) -> dict[str, float | int]:
    labelled = _confirmed_by_tracklet(annotations)
    truth = _ground_truth_positive_pairs(annotations)
    evaluable_predictions = {
        pair for pair in predicted_pairs if len(pair) == 2 and all(item in labelled for item in pair)
    }
    true_positives = len(truth & evaluable_predictions)
    precision = _safe_divide(true_positives, len(evaluable_predictions))
    recall = _safe_divide(true_positives, len(truth))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": len(truth)}


def candidate_pair_recall(
    annotations: Sequence[TrackletAnnotation],
    candidate_pairs: set[frozenset[TrackletId]],
) -> dict[str, float | int]:
    truth = _ground_truth_positive_pairs(annotations)
    return {"recall": _safe_divide(len(truth & candidate_pairs), len(truth)), "support": len(truth)}


def team_classification_metrics(
    annotations: Sequence[TrackletAnnotation],
    evidence: Mapping[TrackletId, TeamEvidence],
) -> dict[str, float | int]:
    eligible = list(annotations)
    predictions = [evidence.get(item.tracklet_id) for item in eligible]
    correct = sum(
        prediction is not None and prediction.team is item.team_label
        for item, prediction in zip(eligible, predictions)
    )
    unknown = sum(
        prediction is None or prediction.team is TeamLabel.UNKNOWN for prediction in predictions
    )
    return {
        "accuracy": _safe_divide(correct, len(eligible)),
        "rejection_rate": _safe_divide(unknown, len(eligible)),
        "evaluated": len(eligible),
        "correct": correct,
        "unknown_predictions": unknown,
    }


def identity_observation_coverage(
    annotations: Sequence[TrackletAnnotation],
    predicted_identities: Mapping[TrackletId, PlayerIdentityId],
    tracklets: TrackletsArtifact,
) -> dict[str, float | int]:
    labelled = _confirmed_by_tracklet(annotations)
    counts = {tracklet_id(item.track_id): item.observation_count for item in tracklets.tracklets}
    annotated = sum(counts.get(item, 0) for item in labelled)
    correct = sum(
        counts.get(item, 0)
        for item, identity in labelled.items()
        if predicted_identities.get(item) == identity
    )
    return {
        "coverage": _safe_divide(correct, annotated),
        "correct_observations": correct,
        "annotated_observations": annotated,
    }
