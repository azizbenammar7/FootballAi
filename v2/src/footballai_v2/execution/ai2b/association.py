"""Experimental candidate association scoring over implemented evidence only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from footballai_v2.execution.ai2a.association import CannotLink, TeamEvidence
from footballai_v2.execution.ai2a.model import TeamLabel, TrackletId
from footballai_v2.execution.ai2b.metrics import binary_pair_metrics


@dataclass(frozen=True, slots=True)
class AssociationDecision:
    left: TrackletId
    right: TrackletId
    temporal_compatible: bool
    team_compatible: bool
    appearance_similarity: float | None
    candidate_match: bool
    rejection_reason: str | None
    experimental: bool = True


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    metrics: Mapping[str, float | int]
    development_pairs: int


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("cosine vectors must have the same non-zero dimension")
    if not all(math.isfinite(float(value)) for value in (*left, *right)):
        raise ValueError("cosine vectors must contain finite values")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("cosine vectors must have non-zero norm")
    value = sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm)
    return round(max(-1.0, min(1.0, value)), 12)


def score_candidate_pair(
    left: TrackletId,
    right: TrackletId,
    embeddings: Mapping[TrackletId, Sequence[float]],
    cannot_links: Sequence[CannotLink],
    team_evidence: Mapping[TrackletId, TeamEvidence],
    *,
    appearance_threshold: float,
    team_confidence_threshold: float = 0.8,
) -> AssociationDecision:
    if left == right:
        raise ValueError("candidate association requires two distinct tracklets")
    pair = frozenset((left, right))
    forbidden = {frozenset((item.left, item.right)) for item in cannot_links}
    if pair in forbidden:
        return AssociationDecision(left, right, False, True, None, False, "temporal_cannot_link")
    left_team = team_evidence.get(left, TeamEvidence(TeamLabel.UNKNOWN, 0.0, "unavailable"))
    right_team = team_evidence.get(right, TeamEvidence(TeamLabel.UNKNOWN, 0.0, "unavailable"))
    team_conflict = (
        left_team.team is not TeamLabel.UNKNOWN
        and right_team.team is not TeamLabel.UNKNOWN
        and left_team.team is not right_team.team
        and left_team.confidence >= team_confidence_threshold
        and right_team.confidence >= team_confidence_threshold
    )
    if team_conflict:
        return AssociationDecision(left, right, True, False, None, False, "confident_team_conflict")
    if left not in embeddings or right not in embeddings:
        return AssociationDecision(left, right, True, True, None, False, "appearance_unavailable")
    similarity = cosine_similarity(embeddings[left], embeddings[right])
    return AssociationDecision(
        left, right, True, True, similarity, similarity >= appearance_threshold,
        None if similarity >= appearance_threshold else "below_appearance_threshold",
    )


def select_similarity_threshold(samples: Sequence[tuple[float, bool]]) -> ThresholdSelection:
    if not samples:
        raise ValueError("threshold selection requires labelled development pairs")
    values = tuple((float(similarity), bool(label)) for similarity, label in samples)
    if not all(math.isfinite(similarity) for similarity, _ in values):
        raise ValueError("threshold similarities must be finite")
    thresholds = sorted({similarity for similarity, _ in values})
    thresholds.append(math.nextafter(max(thresholds), math.inf))
    truth = tuple(label for _, label in values)
    choices = []
    for threshold in thresholds:
        metrics = binary_pair_metrics(truth, tuple(similarity >= threshold for similarity, _ in values))
        choices.append((float(metrics["f1"]), float(metrics["precision"]), float(metrics["recall"]), threshold, metrics))
    _, _, _, threshold, metrics = max(choices, key=lambda item: (item[0], item[1], item[2], item[3]))
    return ThresholdSelection(threshold, metrics, len(values))
