"""Support-counted metrics for bounded AI2B experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

from footballai_v2.execution.ai2a.association import TeamEvidence
from footballai_v2.execution.ai2a.model import TeamLabel, TrackletId


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_pair_metrics(
    truth: Sequence[bool], predictions: Sequence[bool]
) -> dict[str, float | int]:
    if len(truth) != len(predictions):
        raise ValueError("pairwise truth and predictions must have equal length")
    positive_support = sum(bool(value) for value in truth)
    negative_support = len(truth) - positive_support
    true_positive = sum(bool(actual) and bool(predicted) for actual, predicted in zip(truth, predictions))
    false_positive = sum(not bool(actual) and bool(predicted) for actual, predicted in zip(truth, predictions))
    false_negative = positive_support - true_positive
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, positive_support)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_support": positive_support,
        "negative_support": negative_support,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def team_classification_report(
    truth: Mapping[TrackletId, TeamLabel],
    evidence: Mapping[TrackletId, TeamEvidence],
) -> dict[str, dict[str, float | int]]:
    predicted = {
        identifier: evidence.get(identifier, TeamEvidence(TeamLabel.UNKNOWN, 0.0, "unavailable")).team
        for identifier in truth
    }
    result: dict[str, dict[str, float | int]] = {}
    for team in (TeamLabel.TEAM_A, TeamLabel.TEAM_B):
        support = sum(value is team for value in truth.values())
        selected = sum(value is team for value in predicted.values())
        correct = sum(truth[key] is team and predicted[key] is team for key in truth)
        result[team.value] = {
            "precision": _divide(correct, selected),
            "recall": _divide(correct, support),
            "support": support,
            "predicted": selected,
        }
    unknown_support = sum(value is TeamLabel.UNKNOWN for value in truth.values())
    unknown_correct = sum(
        truth[key] is TeamLabel.UNKNOWN and predicted[key] is TeamLabel.UNKNOWN for key in truth
    )
    result[TeamLabel.UNKNOWN.value] = {
        "rejection_rate": _divide(unknown_correct, unknown_support),
        "support": unknown_support,
        "rejected": unknown_correct,
        "overall_prediction_rate": _divide(
            sum(value is TeamLabel.UNKNOWN for value in predicted.values()), len(predicted)
        ),
    }
    return result
