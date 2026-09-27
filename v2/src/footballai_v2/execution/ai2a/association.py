"""Bounded baseline evidence and candidate pairs; no global identity resolver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

from footballai_v2.execution.ai1.artifacts import BoundingBox, TrackletsArtifact
from footballai_v2.execution.ai2a.model import TeamLabel, TrackletId, tracklet_id


CANDIDATE_GRAPH_SCHEMA = "footballai.identity-candidate-graph/v1"


@dataclass(frozen=True, slots=True)
class TrackletAppearance:
    tracklet_id: TrackletId
    hue: float
    saturation: float
    value: float


@dataclass(frozen=True, slots=True)
class TeamEvidence:
    team: TeamLabel
    confidence: float
    method: str


@dataclass(frozen=True, order=True, slots=True)
class CannotLink:
    left: TrackletId
    right: TrackletId
    reason: str
    frame_index: int
    timestamp_seconds: float


@dataclass(frozen=True, order=True, slots=True)
class CandidateNode:
    tracklet_id: TrackletId
    start_time_seconds: float
    end_time_seconds: float
    team: TeamLabel


@dataclass(frozen=True, order=True, slots=True)
class CandidateEdge:
    left: TrackletId
    right: TrackletId
    team_compatible: bool
    temporal_compatible: bool
    appearance_similarity: float | None
    baseline_match: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateGraph:
    schema: str
    nodes: tuple[CandidateNode, ...]
    edges: tuple[CandidateEdge, ...]


def _spatially_distinct(left: BoundingBox, right: BoundingBox) -> bool:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    iou = intersection / union if union else 0.0
    center_distance = math.hypot(left.center_x - right.center_x, left.center_y - right.center_y)
    minimum_diagonal = min(
        math.hypot(left.x2 - left.x1, left.y2 - left.y1),
        math.hypot(right.x2 - right.x1, right.y2 - right.y1),
    )
    return iou <= 0.05 and center_distance >= minimum_diagonal * 0.25


def generate_temporal_cannot_links(tracklets: TrackletsArtifact) -> tuple[CannotLink, ...]:
    """Conservatively link pairs observed simultaneously in disjoint pixel boxes."""
    result: list[CannotLink] = []
    ordered = sorted(tracklets.tracklets, key=lambda item: item.track_id)
    for left, right in combinations(ordered, 2):
        right_by_frame = {item.frame_index: item for item in right.observations}
        witness = next(
            (
                item
                for item in left.observations
                if item.frame_index in right_by_frame and _spatially_distinct(item.bbox, right_by_frame[item.frame_index].bbox)
            ),
            None,
        )
        if witness is None:
            continue
        result.append(
            CannotLink(
                left=tracklet_id(left.track_id),
                right=tracklet_id(right.track_id),
                reason="simultaneous_spatially_distinct",
                frame_index=witness.frame_index,
                timestamp_seconds=witness.timestamp_seconds,
            )
        )
    return tuple(result)


def classify_team(
    appearance: TrackletAppearance,
    prototypes: Mapping[TeamLabel, tuple[float, float, float]],
    *,
    minimum_saturation: float = 25,
    maximum_distance: float = 0.55,
    minimum_margin: float = 0.08,
) -> TeamEvidence:
    """Return jersey-colour evidence, rejecting weak or ambiguous observations."""
    if appearance.saturation < minimum_saturation:
        return TeamEvidence(TeamLabel.UNKNOWN, 0.0, "jersey_color_v1")
    distances = sorted(
        (
            _hsv_distance((appearance.hue, appearance.saturation, appearance.value), prototype),
            team,
        )
        for team, prototype in prototypes.items()
        if team is not TeamLabel.UNKNOWN
    )
    if not distances:
        raise ValueError("at least one known-team prototype is required")
    nearest_distance, nearest_team = distances[0]
    if nearest_distance > maximum_distance:
        return TeamEvidence(TeamLabel.UNKNOWN, 0.0, "jersey_color_v1")
    if len(distances) == 1:
        confidence = 1.0 - nearest_distance / maximum_distance
        return TeamEvidence(nearest_team, round(max(0.0, confidence), 6), "jersey_color_v1")
    margin = distances[1][0] - nearest_distance
    if margin < minimum_margin:
        return TeamEvidence(TeamLabel.UNKNOWN, 0.0, "jersey_color_v1")
    confidence = min(1.0, max(0.0, margin / (distances[1][0] + 1e-12)))
    return TeamEvidence(nearest_team, round(confidence, 6), "jersey_color_v1")


def _hsv_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    hue_delta = abs(left[0] - right[0]) % 180
    hue_delta = min(hue_delta, 180 - hue_delta) / 90
    saturation_delta = abs(left[1] - right[1]) / 255
    value_delta = abs(left[2] - right[2]) / 255
    return math.sqrt(hue_delta**2 + 0.25 * saturation_delta**2 + 0.1 * value_delta**2)


def _appearance_similarity(left: TrackletAppearance, right: TrackletAppearance) -> float:
    return round(max(0.0, 1.0 - _hsv_distance(
        (left.hue, left.saturation, left.value),
        (right.hue, right.saturation, right.value),
    )), 6)


def build_candidate_graph(
    tracklets: TrackletsArtifact,
    team_evidence: Mapping[TrackletId, TeamEvidence],
    appearances: Mapping[TrackletId, TrackletAppearance],
    cannot_links: Sequence[CannotLink],
    *,
    baseline_similarity_threshold: float = 0.8,
) -> CandidateGraph:
    """Create deterministic possible same-player pairs without resolving identities."""
    forbidden = {frozenset((item.left, item.right)) for item in cannot_links}
    ordered = sorted(tracklets.tracklets, key=lambda item: item.track_id)
    nodes = tuple(
        CandidateNode(
            tracklet_id(item.track_id),
            item.start_time_seconds,
            item.end_time_seconds,
            team_evidence.get(
                tracklet_id(item.track_id), TeamEvidence(TeamLabel.UNKNOWN, 0, "unavailable")
            ).team,
        )
        for item in ordered
    )
    edges: list[CandidateEdge] = []
    for left, right in combinations(ordered, 2):
        left_id, right_id = tracklet_id(left.track_id), tracklet_id(right.track_id)
        pair = frozenset((left_id, right_id))
        if pair in forbidden:
            continue
        left_team = team_evidence.get(left_id, TeamEvidence(TeamLabel.UNKNOWN, 0, "unavailable")).team
        right_team = team_evidence.get(right_id, TeamEvidence(TeamLabel.UNKNOWN, 0, "unavailable")).team
        team_compatible = (
            TeamLabel.UNKNOWN in (left_team, right_team) or left_team is right_team
        )
        if not team_compatible:
            continue
        similarity = None
        evidence = ["temporal_compatible", "team_compatible"]
        if left_id in appearances and right_id in appearances:
            similarity = _appearance_similarity(appearances[left_id], appearances[right_id])
            evidence.append("jersey_color_similarity")
        baseline_match = (
            left_team is not TeamLabel.UNKNOWN
            and left_team is right_team
            and similarity is not None
            and similarity >= baseline_similarity_threshold
        )
        edges.append(
            CandidateEdge(
                left_id,
                right_id,
                True,
                True,
                similarity,
                baseline_match,
                tuple(evidence),
            )
        )
    return CandidateGraph(CANDIDATE_GRAPH_SCHEMA, nodes, tuple(edges))
