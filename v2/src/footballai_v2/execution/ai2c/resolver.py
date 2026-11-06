"""Conservative tracklet-to-identity association (AI2C baseline).

Pipeline::

    tracklet summaries
      -> eligibility (class, human exclusions)
      -> hard constraints (temporal overlap, explicit cannot-links,
         confident team conflict, class mismatch)
      -> gated successor candidates (short gap; image-space kinematics and
         box-scale continuity in ``temporal_kinematic`` mode)
      -> mutual-best + ambiguity margin (otherwise abstain)
      -> group-level constraint re-check at every merge
      -> provisional/confirmed identities, or UNRESOLVED (UNKNOWN identity)

Design rules:

* No appearance, jersey-number, pitch-position or camera-motion evidence is
  used: none is validated in this repository yet.
* Image-space kinematics are only meaningful across short gaps; the broadcast
  camera pans and zooms, so the gate is deliberately strict and recall is
  expected to be low. Precision (no false merges) is prioritised.
* ``association_score`` is ``1 - cost``: an uncalibrated ordering heuristic,
  not a probability.
* The algorithm is deterministic: every ordering has an explicit total key.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from footballai_v2.execution.ai1.artifacts import BoundingBox
from footballai_v2.execution.ai2a.association import CannotLink, TeamEvidence
from footballai_v2.execution.ai2a.model import PlayerIdentityId, TeamLabel, TrackletId
from footballai_v2.execution.ai2c.artifact import (
    PLAYER_IDENTITIES_SCHEMA,
    Abstention,
    AssignmentStatus,
    EvidenceInput,
    IdentityLink,
    LinkSource,
    PlayerIdentitiesArtifact,
    PlayerIdentity,
    TrackletAssignment,
)
from footballai_v2.execution.ai2c.corrections import IdentityCorrections
from footballai_v2.execution.ai2c.inputs import InputKind, SourceDescriptor, TrackletSummary


RESOLVER_NAME = "footballai.ai2c.conservative-stitching"
RESOLVER_VERSION = "1.0.0"


class ResolverMode(str, Enum):
    REVIEW_ONLY = "review_only"
    TEMPORAL_ONLY = "temporal_only"
    TEMPORAL_KINEMATIC = "temporal_kinematic"


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    mode: ResolverMode = ResolverMode.TEMPORAL_KINEMATIC
    # Defaults are the configuration selected on AI2B development identities
    # only (see benchmark.py); the a-priori values were 2.0 s and 5.0.
    max_gap_seconds: float = 1.0
    static_allowance_heights: float = 1.0
    max_speed_heights_per_second: float = 3.0
    max_scale_ratio: float = 1.6
    min_ambiguity_margin: float = 0.2
    team_conflict_confidence: float = 0.8
    shot_cut_min_births: int | None = 8
    eligible_semantic_classes: tuple[str, ...] = ("person",)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ResolverMode):
            raise ValueError("mode must be typed")
        for name in (
            "max_gap_seconds",
            "static_allowance_heights",
            "max_speed_heights_per_second",
            "max_scale_ratio",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_scale_ratio < 1:
            raise ValueError("max_scale_ratio must be at least one")
        if not 0 <= self.min_ambiguity_margin < 1:
            raise ValueError("min_ambiguity_margin must be in [0, 1)")
        if not 0 < self.team_conflict_confidence <= 1:
            raise ValueError("team_conflict_confidence must be in (0, 1]")
        if self.shot_cut_min_births is not None and (
            not isinstance(self.shot_cut_min_births, int)
            or isinstance(self.shot_cut_min_births, bool)
            or self.shot_cut_min_births < 2
        ):
            raise ValueError("shot_cut_min_births must be an integer of at least two, or None")
        if not self.eligible_semantic_classes or not all(
            isinstance(item, str) and item.strip() for item in self.eligible_semantic_classes
        ):
            raise ValueError("eligible_semantic_classes must name at least one class")

    @classmethod
    def from_dict(cls, value: Mapping) -> "ResolverConfig":
        return cls(**{
            **value,
            "mode": ResolverMode(value["mode"]),
            "eligible_semantic_classes": tuple(value["eligible_semantic_classes"]),
        })

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["eligible_semantic_classes"] = list(self.eligible_semantic_classes)
        return value


@dataclass(frozen=True, slots=True)
class _Features:
    gap_frames: int
    gap_seconds: float
    normalized_displacement: float
    scale_ratio: float


@dataclass(frozen=True, order=True, slots=True)
class _Candidate:
    cost: float
    earlier: TrackletId
    later: TrackletId
    features: _Features = field(compare=False)


def _foot(box: BoundingBox) -> tuple[float, float]:
    return box.center_x, box.y2


def _height(box: BoundingBox) -> float:
    return box.y2 - box.y1


def _features(earlier: TrackletSummary, later: TrackletSummary) -> _Features:
    a, b = earlier.last_bbox, later.first_bbox
    ax, ay = _foot(a)
    bx, by = _foot(b)
    reference_height = (_height(a) + _height(b)) / 2
    displacement = math.hypot(bx - ax, by - ay) / reference_height
    ratio = max(_height(a), _height(b)) / min(_height(a), _height(b))
    return _Features(
        gap_frames=later.start_frame - earlier.end_frame,
        gap_seconds=round(later.start_time_seconds - earlier.end_time_seconds, 9),
        normalized_displacement=round(displacement, 6),
        scale_ratio=round(ratio, 6),
    )


class _Constraints:
    """Pairwise hard constraints shared by automatic and human links."""

    def __init__(
        self,
        summaries: Mapping[TrackletId, TrackletSummary],
        team_evidence: Mapping[TrackletId, TeamEvidence],
        cannot_links: set[frozenset[TrackletId]],
        team_confidence: float,
    ) -> None:
        self._summaries = summaries
        self._cannot = cannot_links
        self._teams = {
            identifier: evidence.team
            for identifier, evidence in team_evidence.items()
            if evidence.team is not TeamLabel.UNKNOWN and evidence.confidence >= team_confidence
        }

    def violation(self, left: TrackletId, right: TrackletId, *, allow_team_override: bool = False) -> str | None:
        a, b = self._summaries[left], self._summaries[right]
        if a.start_frame <= b.end_frame and b.start_frame <= a.end_frame:
            return "temporal_overlap"
        if frozenset((left, right)) in self._cannot:
            return "explicit_cannot_link"
        if a.semantic_class and b.semantic_class and a.semantic_class != b.semantic_class:
            return "semantic_class_mismatch"
        if not allow_team_override:
            left_team, right_team = self._teams.get(left), self._teams.get(right)
            if left_team is not None and right_team is not None and left_team is not right_team:
                return "confident_team_conflict"
        return None

    def confident_team(self, identifier: TrackletId) -> TeamLabel | None:
        return self._teams.get(identifier)


class _Groups:
    def __init__(self, identifiers: Sequence[TrackletId]) -> None:
        self._parent = {item: item for item in identifiers}
        self._members = {item: [item] for item in identifiers}
        self.human_override_pairs: set[frozenset[TrackletId]] = set()

    def find(self, item: TrackletId) -> TrackletId:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def members(self, item: TrackletId) -> list[TrackletId]:
        return self._members[self.find(item)]

    def conflict(
        self, left: TrackletId, right: TrackletId, constraints: _Constraints, *, human: bool
    ) -> str | None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return "already_same_identity"
        for x in self._members[left_root]:
            for y in self._members[right_root]:
                override = human or frozenset((x, y)) in self.human_override_pairs
                reason = constraints.violation(x, y, allow_team_override=override)
                if reason is not None:
                    return reason
        return None

    def union(self, left: TrackletId, right: TrackletId) -> None:
        left_root, right_root = sorted((self.find(left), self.find(right)))
        self._parent[right_root] = left_root
        self._members[left_root].extend(self._members.pop(right_root))


def _top(candidates: Sequence[_Candidate], *, successor: bool) -> tuple[tuple[TrackletId, float], ...]:
    return tuple(
        ((item.later if successor else item.earlier), item.cost) for item in candidates[:3]
    )


def resolve_identities(
    summaries: Sequence[TrackletSummary],
    source: SourceDescriptor,
    *,
    config: ResolverConfig = ResolverConfig(),
    team_evidence: Mapping[TrackletId, TeamEvidence] | None = None,
    cannot_links: Sequence[CannotLink] = (),
    corrections: IdentityCorrections | None = None,
    evidence_inputs: Sequence[EvidenceInput] = (),
) -> PlayerIdentitiesArtifact:
    by_id = {item.tracklet_id: item for item in summaries}
    if len(by_id) != len(summaries):
        raise ValueError("tracklet summaries must be unique")
    teams = dict(team_evidence or {})
    for identifier in teams:
        if identifier not in by_id:
            raise ValueError(f"team evidence references unknown {identifier.value}")
    explicit = set()
    for item in cannot_links:
        if item.left not in by_id or item.right not in by_id:
            raise ValueError("cannot-link references an unknown tracklet")
        explicit.add(frozenset((item.left, item.right)))

    effective = None
    if corrections is not None:
        if source.tracklets_sha256 is None or corrections.tracklets_sha256 != source.tracklets_sha256:
            raise ValueError("corrections are pinned to a different tracklets artifact")
        effective = corrections.effective()
        for entry in corrections.entries:
            for identifier in entry.tracklet_ids:
                if identifier not in by_id:
                    raise ValueError(f"correction {entry.sequence} references unknown {identifier.value}")
        explicit.update(frozenset((left, right)) for left, right, _ in effective.cannot_links)

    constraints = _Constraints(by_id, teams, explicit, config.team_conflict_confidence)
    excluded: dict[TrackletId, str] = {}
    for identifier, item in by_id.items():
        if item.semantic_class is not None and item.semantic_class not in config.eligible_semantic_classes:
            excluded[identifier] = f"ineligible_semantic_class:{item.semantic_class}"
    if effective is not None:
        for identifier, sequence in effective.excluded:
            excluded[identifier] = f"human_exclude:correction_{sequence}"
    eligible = sorted(identifier for identifier in by_id if identifier not in excluded)
    groups = _Groups(eligible)
    links: list[IdentityLink] = []
    applied: list[dict] = []

    # Human must-links first: reviewer assertions, but never against physics.
    if effective is not None:
        for left, right, sequence in effective.must_links:
            earlier, later = sorted((left, right), key=lambda item: (by_id[item].start_frame, item))
            reason = groups.conflict(earlier, later, constraints, human=True)
            if reason == "already_same_identity":
                applied.append({"sequence": sequence, "action": "must_link", "outcome": "redundant"})
                continue
            if reason is not None:
                raise ValueError(f"correction {sequence} must_link violates a hard constraint: {reason}")
            features = _features(by_id[earlier], by_id[later])
            groups.human_override_pairs.add(frozenset((earlier, later)))
            groups.union(earlier, later)
            links.append(IdentityLink(
                earlier, later, LinkSource.HUMAN, sequence, features.gap_frames, features.gap_seconds,
                features.normalized_displacement, features.scale_ratio, None, None,
            ))
            applied.append({"sequence": sequence, "action": "must_link", "outcome": "applied"})
        outcomes = {item["sequence"]: item["outcome"] for item in applied}
        revoked = {item.revokes_sequence for item in corrections.entries if item.revokes_sequence}
        applied = []
        for entry in corrections.entries:
            if entry.action.value == "revoke":
                outcome = "recorded"
            elif entry.sequence in revoked:
                outcome = "revoked"
            else:
                outcome = outcomes.get(entry.sequence, "applied")
            applied.append({
                "sequence": entry.sequence,
                "action": entry.action.value,
                "outcome": outcome,
                "tracklet_ids": [item.value for item in entry.tracklet_ids],
                "revokes_sequence": entry.revokes_sequence,
                "reviewer": entry.reviewer,
                "reviewed_at": entry.reviewed_at,
            })

    # Suspected shot cuts: many simultaneous track births in one sampled frame
    # (camera cut, replay). Automatic links never cross one; humans may.
    births: dict[int, int] = {}
    for identifier in eligible:
        births[by_id[identifier].start_frame] = births.get(by_id[identifier].start_frame, 0) + 1
    first_frame = min((by_id[item].start_frame for item in eligible), default=0)
    cut_frames = sorted(
        frame
        for frame, count in births.items()
        if config.shot_cut_min_births is not None and count >= config.shot_cut_min_births and frame > first_frame
    )

    # Automatic candidates: strictly later start, within the gap window.
    ordered = sorted(eligible, key=lambda item: (by_id[item].start_frame, item))
    start_frames = [by_id[item].start_frame for item in ordered]
    successors: dict[TrackletId, list[_Candidate]] = {}
    predecessors: dict[TrackletId, list[_Candidate]] = {}
    evaluated = 0
    automatic_ids = eligible if config.mode is not ResolverMode.REVIEW_ONLY else []
    for earlier_id in automatic_ids:
        earlier = by_id[earlier_id]
        index = bisect.bisect_right(start_frames, earlier.end_frame)
        while index < len(ordered):
            later = by_id[ordered[index]]
            index += 1
            gap = later.start_time_seconds - earlier.end_time_seconds
            if gap > config.max_gap_seconds:
                break
            evaluated += 1
            if constraints.violation(earlier_id, later.tracklet_id) is not None:
                continue
            cut = bisect.bisect_right(cut_frames, earlier.end_frame)
            if cut < len(cut_frames) and cut_frames[cut] <= later.start_frame:
                continue
            features = _features(earlier, later)
            if config.mode is ResolverMode.TEMPORAL_ONLY:
                cost = gap / config.max_gap_seconds
            else:
                if features.scale_ratio > config.max_scale_ratio:
                    continue
                allowed = config.static_allowance_heights + config.max_speed_heights_per_second * gap
                if features.normalized_displacement > allowed:
                    continue
                cost = features.normalized_displacement / allowed
            candidate = _Candidate(round(cost, 6), earlier_id, later.tracklet_id, features)
            successors.setdefault(earlier_id, []).append(candidate)
            predecessors.setdefault(later.tracklet_id, []).append(candidate)
    for values in (*successors.values(), *predecessors.values()):
        values.sort()

    abstentions: list[Abstention] = []
    proposals: list[_Candidate] = []
    for earlier_id in sorted(successors):
        options = successors[earlier_id]
        best = options[0]
        reverse = predecessors[best.later]
        if reverse[0] != best:
            abstentions.append(Abstention(earlier_id, "successor", "not_mutual_best", _top(options, successor=True)))
            continue
        if len(options) > 1 and options[1].cost - best.cost < config.min_ambiguity_margin:
            abstentions.append(Abstention(earlier_id, "successor", "ambiguous_successor", _top(options, successor=True)))
            continue
        if len(reverse) > 1 and reverse[1].cost - best.cost < config.min_ambiguity_margin:
            abstentions.append(
                Abstention(best.later, "predecessor", "ambiguous_predecessor", _top(reverse, successor=False))
            )
            continue
        proposals.append(best)

    for candidate in sorted(proposals):
        reason = groups.conflict(candidate.earlier, candidate.later, constraints, human=False)
        if reason is not None:
            abstentions.append(Abstention(
                candidate.earlier, "successor", "group_constraint_conflict",
                ((candidate.later, candidate.cost),),
            ))
            continue
        groups.union(candidate.earlier, candidate.later)
        features = candidate.features
        links.append(IdentityLink(
            candidate.earlier, candidate.later, LinkSource.AUTOMATIC, None, features.gap_frames,
            features.gap_seconds, features.normalized_displacement, features.scale_ratio,
            candidate.cost, round(1 - candidate.cost, 6),
        ))

    return _build_artifact(
        by_id, source, config, constraints, groups, eligible, excluded, links, abstentions, applied,
        evaluated, sum(len(item) for item in successors.values()), cut_frames, evidence_inputs,
        bool(teams), bool(cannot_links), corrections is not None,
    )


def _build_artifact(
    by_id: Mapping[TrackletId, TrackletSummary],
    source: SourceDescriptor,
    config: ResolverConfig,
    constraints: _Constraints,
    groups: _Groups,
    eligible: Sequence[TrackletId],
    excluded: Mapping[TrackletId, str],
    links: Sequence[IdentityLink],
    abstentions: Sequence[Abstention],
    applied: Sequence[dict],
    evaluated: int,
    gated: int,
    cut_frames: Sequence[int],
    evidence_inputs: Sequence[EvidenceInput],
    team_supplied: bool,
    cannot_links_supplied: bool,
    corrections_supplied: bool,
) -> PlayerIdentitiesArtifact:
    roots: dict[TrackletId, list[TrackletId]] = {}
    for identifier in eligible:
        roots.setdefault(groups.find(identifier), []).append(identifier)
    multi = [
        sorted(members, key=lambda item: (by_id[item].start_frame, item))
        for members in roots.values()
        if len(members) > 1
    ]
    multi.sort(key=lambda members: (by_id[members[0]].start_frame, members[0]))
    link_root = {}
    for link in links:
        link_root.setdefault(groups.find(link.earlier), []).append(link)

    identities: list[PlayerIdentity] = []
    identity_of: dict[TrackletId, tuple[PlayerIdentityId, AssignmentStatus]] = {}
    for index, members in enumerate(multi, start=1):
        identifier = PlayerIdentityId(f"PID_{index:06d}")
        group_links = link_root[groups.find(members[0])]
        automatic = [item for item in group_links if item.source is LinkSource.AUTOMATIC]
        human = len(group_links) - len(automatic)
        status = AssignmentStatus.PROVISIONAL if automatic else AssignmentStatus.CONFIRMED
        known_teams = {
            team for team in (constraints.confident_team(item) for item in members) if team is not None
        }
        channels = {"temporal_non_overlap"}
        if automatic:
            channels.add("temporal_gap")
            if config.mode is ResolverMode.TEMPORAL_KINEMATIC:
                channels.update({"image_space_kinematics", "bbox_scale_continuity"})
            if config.shot_cut_min_births is not None:
                channels.add("track_birth_burst_cut_barrier")
        if human:
            channels.add("human_must_link")
        if team_supplied:
            channels.add("team_conflict_guard")
        if cannot_links_supplied:
            channels.add("explicit_cannot_links")
        counts = [by_id[item].observation_count for item in members]
        identities.append(PlayerIdentity(
            player_identity_id=identifier,
            status=status,
            tracklet_ids=tuple(members),
            start_frame=by_id[members[0]].start_frame,
            end_frame=max(by_id[item].end_frame for item in members),
            start_time_seconds=by_id[members[0]].start_time_seconds,
            end_time_seconds=max(by_id[item].end_time_seconds for item in members),
            observed_seconds=round(sum(by_id[item].duration_seconds for item in members), 6),
            observation_count=None if any(item is None for item in counts) else sum(counts),
            team=next(iter(known_teams)) if len(known_teams) == 1 else TeamLabel.UNKNOWN,
            association_score_min=min((item.association_score for item in automatic), default=None),
            automatic_links=len(automatic),
            human_links=human,
            evidence_channels=tuple(sorted(channels)),
        ))
        for member in members:
            identity_of[member] = (identifier, status)

    abstained = {item.tracklet_id for item in abstentions}
    for item in abstentions:
        abstained.update(candidate for candidate, _ in item.candidates)
    assignments = []
    for identifier in sorted(by_id):
        summary = by_id[identifier]
        if identifier in excluded:
            status, identity, reason = AssignmentStatus.EXCLUDED, None, excluded[identifier]
        elif identifier in identity_of:
            identity, status = identity_of[identifier]
            reason = "human_must_link" if status is AssignmentStatus.CONFIRMED else "automatic_association"
        else:
            status, identity = AssignmentStatus.UNRESOLVED, None
            reason = "abstained_ambiguous_or_conflicting" if identifier in abstained else "no_accepted_candidate"
        assignments.append(TrackletAssignment(
            identifier, identity, status, reason, summary.start_frame, summary.end_frame,
        ))

    statuses = [item.status for item in assignments]
    known_counts = [by_id[item].observation_count for item in sorted(by_id)]
    observation_coverage = None
    if all(item is not None for item in known_counts):
        total = sum(known_counts)
        assigned = sum(
            by_id[item.tracklet_id].observation_count
            for item in assignments
            if item.player_identity_id is not None
        )
        observation_coverage = round(assigned / total, 6) if total else None
    summary = {
        "tracklets": len(by_id),
        "eligible_tracklets": len(eligible),
        "excluded_tracklets": statuses.count(AssignmentStatus.EXCLUDED),
        "unresolved_tracklets": statuses.count(AssignmentStatus.UNRESOLVED),
        "provisional_tracklets": statuses.count(AssignmentStatus.PROVISIONAL),
        "confirmed_tracklets": statuses.count(AssignmentStatus.CONFIRMED),
        "identities": len(identities),
        "provisional_identities": sum(item.status is AssignmentStatus.PROVISIONAL for item in identities),
        "confirmed_identities": sum(item.status is AssignmentStatus.CONFIRMED for item in identities),
        "automatic_links": sum(item.source is LinkSource.AUTOMATIC for item in links),
        "human_links": sum(item.source is LinkSource.HUMAN for item in links),
        "abstentions": len(abstentions),
        "candidate_pairs_in_gap_window": evaluated,
        "gated_candidates": gated,
        "suspected_shot_cut_frames": list(cut_frames),
        "identity_observation_share": observation_coverage,
        "identity_observation_share_note": (
            "share of input observations assigned to any confirmed/provisional identity; "
            "not a correctness metric; null when observation counts are unavailable"
        ),
    }
    used = ["temporal_non_overlap"]
    if config.mode is not ResolverMode.REVIEW_ONLY:
        used.append("temporal_gap")
    if config.mode is ResolverMode.TEMPORAL_KINEMATIC:
        used += ["image_space_kinematics", "bbox_scale_continuity"]
    if config.mode is not ResolverMode.REVIEW_ONLY and config.shot_cut_min_births is not None:
        used.append("track_birth_burst_cut_barrier")
    if team_supplied:
        used.append("team_conflict_guard")
    if cannot_links_supplied:
        used.append("explicit_cannot_links")
    if corrections_supplied:
        used.append("human_corrections")
    resolver = {
        "name": RESOLVER_NAME,
        "version": RESOLVER_VERSION,
        "configuration": config.to_dict(),
        "evidence_channels_used": used,
        "evidence_channels_unavailable": {
            "appearance_embedding": "AI2B is unmerged and did not justify identity association",
            "jersey_number": "not implemented",
            "pitch_position": "no validated pitch calibration (AI3)",
            "camera_motion_compensation": "not implemented",
        },
        "association_score_semantics": "1 - gated cost; uncalibrated ordering heuristic, not a probability",
    }
    limitations = [
        "Identities are short-gap stitched tracklet groups, not verified full-match players.",
        "Identity IDs are scoped to this artifact and are not stable across artifacts or matches.",
        "Image-space kinematics ignore camera pan/zoom; links are limited to short gaps.",
        "Unresolved tracklets have UNKNOWN player identity and must not feed player statistics.",
        "Officials, staff and spectators are only excluded by human correction; there is no role classifier.",
    ]
    if source.input_kind is InputKind.REVIEW_SAMPLES:
        limitations.append(
            "Input reconstructed from review samples: extents and endpoint boxes are exact, "
            "observation counts are partly unknown."
        )
    if source.input_kind is InputKind.LEGACY_RAW_TRACKS:
        limitations.append(
            "Legacy V1 raw tracks: diagnostic input only; short fragments were already removed "
            "and frame labels use the legacy convention."
        )
    return PlayerIdentitiesArtifact(
        schema=PLAYER_IDENTITIES_SCHEMA,
        source=source,
        resolver=resolver,
        evidence_inputs=tuple(sorted(evidence_inputs, key=lambda item: (item.role, item.sha256))),
        identities=tuple(identities),
        assignments=tuple(assignments),
        links=tuple(sorted(links, key=lambda item: (by_id[item.earlier].start_frame, item.earlier, item.later))),
        abstentions=tuple(sorted(abstentions, key=lambda item: (item.tracklet_id, item.direction, item.reason))),
        corrections_applied=tuple(applied),
        summary=summary,
        limitations=tuple(limitations),
    )
