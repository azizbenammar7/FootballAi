"""Measured identity-association metrics against AI2A-format ground truth.

Only metrics that the tracklet-level annotation format supports are computed.
Ratios carry their numerators and denominators, and a ratio whose denominator
is zero is ``null`` rather than a misleading ``0``. HOTA/IDF1/ID-switches need
per-frame ground-truth boxes for every visible player and are not measured.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from footballai_v2.execution.ai2a.association import CannotLink
from footballai_v2.execution.ai2a.model import (
    AnnotationStatus,
    IdentityGroundTruthArtifact,
    PlayerIdentityId,
    TeamLabel,
    TrackletId,
)
from footballai_v2.execution.ai2c.artifact import AssignmentStatus, PlayerIdentitiesArtifact


NON_PLAYER_ROLES = frozenset({"OFFICIAL", "STAFF", "REFEREE", "SPECTATOR", "MEDIA"})
NOT_MEASURED = {
    "HOTA": "requires per-frame ground-truth boxes for every visible player",
    "IDF1": "requires per-frame ground-truth boxes for every visible player",
    "ID_switches": "requires per-frame ground-truth boxes for every visible player",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def check_ground_truth_matches(artifact: PlayerIdentitiesArtifact, ground_truth: IdentityGroundTruthArtifact) -> None:
    expected = artifact.source.tracklets_sha256
    if expected is None or ground_truth.source.tracklets_sha256 != expected:
        raise ValueError("ground truth is pinned to a different tracklets artifact")
    known = {item.tracklet_id for item in artifact.assignments}
    for annotation in ground_truth.annotations:
        if annotation.tracklet_id not in known:
            raise ValueError(f"ground truth references {annotation.tracklet_id.value}, absent from the identities")


def _pairs(identifiers: Iterable[TrackletId]) -> Iterable[tuple[TrackletId, TrackletId]]:
    return combinations(sorted(identifiers), 2)


def evaluate_identities(
    artifact: PlayerIdentitiesArtifact,
    ground_truth: IdentityGroundTruthArtifact,
    *,
    identity_scope: Sequence[PlayerIdentityId] | None = None,
    cannot_links: Sequence[CannotLink] = (),
) -> dict:
    check_ground_truth_matches(artifact, ground_truth)
    predicted = artifact.identity_by_tracklet()
    assignments = {item.tracklet_id: item for item in artifact.assignments}
    scope = set(identity_scope) if identity_scope is not None else None
    confirmed = {
        item.tracklet_id: item.ground_truth_player_id
        for item in ground_truth.annotations
        if item.status is AnnotationStatus.CONFIRMED
        and item.ground_truth_player_id is not None
        and (scope is None or item.ground_truth_player_id in scope)
    }
    if scope is not None:
        missing = scope - set(confirmed.values())
        if missing:
            raise ValueError(f"identity scope names unlabelled identities: {sorted(item.value for item in missing)}")

    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    false_merges = []
    for left, right in _pairs(confirmed):
        truth = confirmed[left] == confirmed[right]
        same = predicted.get(left) is not None and predicted.get(left) == predicted.get(right)
        key = ("true_" if truth == same else "false_") + ("positive" if same else "negative")
        counts[key] += 1
        if same and not truth:
            false_merges.append({
                "left": left.value, "right": right.value,
                "player_identity_id": predicted[left].value,
                "ground_truth": [confirmed[left].value, confirmed[right].value],
            })
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["false_positive"] + counts["true_negative"]
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    precision = _ratio(counts["true_positive"], predicted_positive)
    recall = _ratio(counts["true_positive"], positives)
    f1 = (
        _ratio(2 * counts["true_positive"], 2 * counts["true_positive"] + counts["false_positive"] + counts["false_negative"])
        if predicted_positive and positives
        else None
    )

    # Contamination: a scoped player merged with a labelled non-player or a
    # known opposing-team tracklet. These are certain errors even when the
    # other tracklet has no player identity label.
    annotations = {item.tracklet_id: item for item in ground_truth.annotations}
    contamination = []
    cross_team = []
    uncertain_disagreements = []
    for left, right in _pairs(annotations):
        if predicted.get(left) is None or predicted.get(left) != predicted.get(right):
            continue
        a, b = annotations[left], annotations[right]
        in_scope = left in confirmed or right in confirmed
        if a.team_label is not TeamLabel.UNKNOWN and b.team_label is not TeamLabel.UNKNOWN and a.team_label is not b.team_label:
            if scope is None or in_scope:
                cross_team.append({"left": left.value, "right": right.value})
        roles = {a.role_label, b.role_label}
        if in_scope and roles & NON_PLAYER_ROLES and "PLAYER" in roles:
            contamination.append({"left": left.value, "right": right.value})
        if (
            (scope is None or in_scope)
            and AnnotationStatus.UNCERTAIN in (a.status, b.status)
            and a.ground_truth_player_id is not None
            and b.ground_truth_player_id is not None
            and a.ground_truth_player_id != b.ground_truth_player_id
        ):
            uncertain_disagreements.append({"left": left.value, "right": right.value})

    # Temporal and explicit cannot-link violations over *all* assignments.
    members: dict[PlayerIdentityId, list[TrackletId]] = defaultdict(list)
    for identifier, identity in predicted.items():
        members[identity].append(identifier)
    temporal = 0
    for identity, items in members.items():
        extents = sorted((assignments[item].start_frame, assignments[item].end_frame) for item in items)
        temporal += sum(1 for (_, end), (start, _) in zip(extents, extents[1:]) if start <= end)
    explicit = sum(
        1
        for item in cannot_links
        if predicted.get(item.left) is not None and predicted.get(item.left) == predicted.get(item.right)
    )

    # Fragmentation: tracklets per labelled identity before vs. predicted units after.
    by_truth: dict[PlayerIdentityId, list[TrackletId]] = defaultdict(list)
    for identifier, identity in confirmed.items():
        by_truth[identity].append(identifier)
    fragments_before = sum(len(items) for items in by_truth.values())
    fragments_after = sum(
        len({predicted.get(item) or item for item in items}) for items in by_truth.values()
    )

    # Purity of predicted identities touching scoped labelled tracklets.
    touched: dict[PlayerIdentityId, set[PlayerIdentityId]] = defaultdict(set)
    for identifier, truth in confirmed.items():
        if predicted.get(identifier) is not None:
            touched[predicted[identifier]].add(truth)
    contaminated = sum(1 for values in touched.values() if len(values) > 1)

    statuses = [item.status for item in artifact.assignments]
    eligible = len(statuses) - statuses.count(AssignmentStatus.EXCLUDED)
    unresolved = statuses.count(AssignmentStatus.UNRESOLVED)
    labelled_unresolved = sum(
        1 for item in confirmed if assignments[item].status is AssignmentStatus.UNRESOLVED
    )
    return {
        "scope": {
            "identities": sorted(item.value for item in set(confirmed.values())),
            "confirmed_labelled_tracklets": len(confirmed),
            "positive_pairs": positives,
            "negative_pairs": negatives,
        },
        "pairwise": {
            **counts,
            "predicted_positive_pairs": predicted_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "false_merges": {
            "count": counts["false_positive"],
            "rate_of_predicted_positive_pairs": _ratio(counts["false_positive"], predicted_positive),
            "rate_of_negative_pairs": _ratio(counts["false_positive"], negatives),
            "pairs": false_merges[:20],
        },
        "labelled_non_player_contamination_pairs": len(contamination),
        "cross_team_violations": len(cross_team),
        "uncertain_label_disagreements": {
            "count": len(uncertain_disagreements),
            "classification": "diagnostic: merged tracklets whose labels differ but one label is uncertain",
            "pairs": uncertain_disagreements[:20],
        },
        "temporal_overlap_violations": temporal,
        "explicit_cannot_link_violations": explicit,
        "explicit_cannot_links_checked": len(cannot_links),
        "contaminated_predicted_identities": contaminated,
        "predicted_identities_touching_scope": len(touched),
        "fragmentation": {
            "labelled_tracklets": fragments_before,
            "predicted_units_after_association": fragments_after,
            "units_per_identity_before": _ratio(fragments_before, len(by_truth)),
            "units_per_identity_after": _ratio(fragments_after, len(by_truth)),
        },
        "unresolved": {
            "all_eligible_tracklets": eligible,
            "unresolved_tracklets": unresolved,
            "rate": _ratio(unresolved, eligible),
            "scoped_labelled_unresolved": labelled_unresolved,
            "scoped_labelled_rate": _ratio(labelled_unresolved, len(confirmed)),
        },
        "not_measured": dict(NOT_MEASURED),
    }


def label_counts(ground_truth: IdentityGroundTruthArtifact) -> Mapping[str, int]:
    return {
        "annotations": len(ground_truth.annotations),
        "confirmed": sum(item.status is AnnotationStatus.CONFIRMED for item in ground_truth.annotations),
        "uncertain": sum(item.status is AnnotationStatus.UNCERTAIN for item in ground_truth.annotations),
        "unknown": sum(item.status is AnnotationStatus.UNKNOWN for item in ground_truth.annotations),
        "identities": len({
            item.ground_truth_player_id
            for item in ground_truth.annotations
            if item.ground_truth_player_id is not None
        }),
    }
