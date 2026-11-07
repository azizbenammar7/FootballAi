"""``footballai.player-identities/v1``: versioned identity assignments.

The artifact is a deterministic JSON document (sorted keys, no wall-clock
fields) so the same inputs and configuration always produce the same bytes and
SHA-256. It deliberately records *why* each tracklet is or is not part of an
identity, including abstentions, so every association is reviewable.

Identity IDs are scoped to one artifact. ``PID_000001`` in two artifacts is not
the same person; cross-match identity is a later milestone.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from footballai_v2.execution.ai1.artifacts import sha256_bytes
from footballai_v2.execution.ai2a.model import PlayerIdentityId, TeamLabel, TrackletId
from footballai_v2.execution.ai2c.inputs import SourceDescriptor


PLAYER_IDENTITIES_SCHEMA = "footballai.player-identities/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESOLVER_IDENTITY_RE = re.compile(r"^PID_[0-9]{6,}$")
_HUMAN_EXCLUDE_RE = re.compile(r"^human_exclude:correction_([1-9][0-9]*)$")


class AssignmentStatus(str, Enum):
    """How a tracklet relates to a player identity.

    ``confirmed``   every link in its identity was asserted by a human reviewer.
    ``provisional`` its identity contains at least one automatic link.
    ``unresolved``  eligible, but no association was accepted: identity UNKNOWN.
    ``excluded``    not eligible for player identity (human exclusion or class).
    """

    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    UNRESOLVED = "unresolved"
    EXCLUDED = "excluded"


class LinkSource(str, Enum):
    AUTOMATIC = "automatic"
    HUMAN = "human"


def _finite_or_none(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _score(value: float | None, name: str) -> float | None:
    result = _finite_or_none(value, name)
    if result is not None and not 0 <= result <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return result


@dataclass(frozen=True, order=True, slots=True)
class IdentityLink:
    earlier: TrackletId
    later: TrackletId
    source: LinkSource
    correction_sequence: int | None
    gap_frames: int
    gap_seconds: float
    normalized_displacement: float
    scale_ratio: float
    cost: float | None
    association_score: float | None

    def __post_init__(self) -> None:
        if self.earlier == self.later:
            raise ValueError("a link requires two distinct tracklets")
        if self.gap_frames <= 0 or _finite_or_none(self.gap_seconds, "gap_seconds") <= 0:
            raise ValueError("linked tracklets must not overlap in time")
        _finite_or_none(self.normalized_displacement, "normalized_displacement")
        _finite_or_none(self.scale_ratio, "scale_ratio")
        if self.source is LinkSource.AUTOMATIC:
            if self.correction_sequence is not None:
                raise ValueError("automatic links cannot cite a correction")
            _score(self.cost, "cost")
            if _score(self.association_score, "association_score") is None:
                raise ValueError("automatic links require an association score")
        else:
            if not isinstance(self.correction_sequence, int) or self.correction_sequence < 1:
                raise ValueError("human links must cite a correction sequence")
            if self.cost is not None or self.association_score is not None:
                raise ValueError("human links carry no automatic score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "earlier": self.earlier.value,
            "later": self.later.value,
            "source": self.source.value,
            "correction_sequence": self.correction_sequence,
            "gap_frames": self.gap_frames,
            "gap_seconds": self.gap_seconds,
            "normalized_displacement": self.normalized_displacement,
            "scale_ratio": self.scale_ratio,
            "cost": self.cost,
            "association_score": self.association_score,
        }


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    player_identity_id: PlayerIdentityId
    status: AssignmentStatus
    tracklet_ids: tuple[TrackletId, ...]
    start_frame: int
    end_frame: int
    start_time_seconds: float
    end_time_seconds: float
    observed_seconds: float
    observation_count: int | None
    team: TeamLabel
    association_score_min: float | None
    automatic_links: int
    human_links: int
    evidence_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _RESOLVER_IDENTITY_RE.fullmatch(self.player_identity_id.value):
            raise ValueError("resolver identities use the PID_ namespace")
        if self.status not in (AssignmentStatus.CONFIRMED, AssignmentStatus.PROVISIONAL):
            raise ValueError("an identity is either confirmed or provisional")
        if len(self.tracklet_ids) < 2:
            raise ValueError("an identity requires at least two associated tracklets")
        if self.automatic_links + self.human_links != len(self.tracklet_ids) - 1:
            raise ValueError("identity links must form a spanning tree over its tracklets")
        if self.status is AssignmentStatus.CONFIRMED and self.automatic_links:
            raise ValueError("confirmed identities contain human links only")
        if self.status is AssignmentStatus.PROVISIONAL and not self.automatic_links:
            raise ValueError("provisional identities contain an automatic link")
        if (self.association_score_min is None) != (self.automatic_links == 0):
            raise ValueError("association_score_min summarises automatic links only")
        _score(self.association_score_min, "association_score_min")

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_identity_id": self.player_identity_id.value,
            "status": self.status.value,
            "tracklet_ids": [item.value for item in self.tracklet_ids],
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "observed_seconds": self.observed_seconds,
            "observation_count": self.observation_count,
            "team": self.team.value,
            "association_score_min": self.association_score_min,
            "automatic_links": self.automatic_links,
            "human_links": self.human_links,
            "evidence_channels": list(self.evidence_channels),
        }


@dataclass(frozen=True, slots=True)
class TrackletAssignment:
    tracklet_id: TrackletId
    player_identity_id: PlayerIdentityId | None
    status: AssignmentStatus
    reason: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not 0 <= self.start_frame <= self.end_frame:
            raise ValueError("assignment frame extent is invalid")
        has_identity = self.player_identity_id is not None
        if has_identity != (self.status in (AssignmentStatus.CONFIRMED, AssignmentStatus.PROVISIONAL)):
            raise ValueError("only confirmed/provisional assignments carry a player identity")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("every assignment records a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id.value,
            "player_identity_id": self.player_identity_id.value if self.player_identity_id else None,
            "status": self.status.value,
            "reason": self.reason,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


@dataclass(frozen=True, slots=True)
class Abstention:
    """An association the resolver considered but refused to make."""

    tracklet_id: TrackletId
    direction: str
    reason: str
    candidates: tuple[tuple[TrackletId, float], ...]

    def __post_init__(self) -> None:
        if self.direction not in ("successor", "predecessor"):
            raise ValueError("abstention direction must be successor or predecessor")
        if self.reason not in (
            "ambiguous_successor",
            "ambiguous_predecessor",
            "not_mutual_best",
            "group_constraint_conflict",
        ):
            raise ValueError("unknown abstention reason")
        if not self.candidates or len(self.candidates) > 3:
            raise ValueError("abstentions record between one and three candidates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id.value,
            "direction": self.direction,
            "reason": self.reason,
            "candidates": [
                {"tracklet_id": identifier.value, "cost": cost} for identifier, cost in self.candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    role: str
    schema: str
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in ("team_evidence", "cannot_links", "corrections"):
            raise ValueError("unknown evidence input role")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("evidence input schema must be non-empty")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("evidence input sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "schema": self.schema, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PlayerIdentitiesArtifact:
    schema: str
    source: SourceDescriptor
    resolver: Mapping[str, Any]
    evidence_inputs: tuple[EvidenceInput, ...]
    identities: tuple[PlayerIdentity, ...]
    assignments: tuple[TrackletAssignment, ...]
    links: tuple[IdentityLink, ...]
    abstentions: tuple[Abstention, ...]
    corrections_applied: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != PLAYER_IDENTITIES_SCHEMA:
            raise ValueError(f"schema must be {PLAYER_IDENTITIES_SCHEMA}")
        assigned = [item.tracklet_id for item in self.assignments]
        if len(assigned) != len(set(assigned)):
            raise ValueError("a tracklet has exactly one assignment")
        if assigned != sorted(assigned):
            raise ValueError("assignments must be sorted by tracklet_id")
        by_tracklet = {item.tracklet_id: item for item in self.assignments}
        identity_ids = [item.player_identity_id for item in self.identities]
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError("player identity IDs must be unique")
        members: dict[TrackletId, PlayerIdentityId] = {}
        for identity in self.identities:
            for identifier in identity.tracklet_ids:
                assignment = by_tracklet.get(identifier)
                if assignment is None:
                    raise ValueError(f"identity references unassigned {identifier.value}")
                if assignment.player_identity_id != identity.player_identity_id:
                    raise ValueError(f"{identifier.value} assignment disagrees with its identity")
                if assignment.status is not identity.status:
                    raise ValueError(f"{identifier.value} status disagrees with its identity")
                if identifier in members:
                    raise ValueError(f"{identifier.value} belongs to two identities")
                members[identifier] = identity.player_identity_id
            extents = sorted(
                (by_tracklet[identifier].start_frame, by_tracklet[identifier].end_frame)
                for identifier in identity.tracklet_ids
            )
            for (_, previous_end), (next_start, _) in zip(extents, extents[1:]):
                if next_start <= previous_end:
                    raise ValueError(
                        f"{identity.player_identity_id.value} contains temporally overlapping tracklets"
                    )
            if (extents[0][0], max(end for _, end in extents)) != (identity.start_frame, identity.end_frame):
                raise ValueError(f"{identity.player_identity_id.value} extent disagrees with its tracklets")
        for assignment in self.assignments:
            if assignment.player_identity_id is not None and assignment.tracklet_id not in members:
                raise ValueError(f"{assignment.tracklet_id.value} names an identity that does not list it")
        for link in self.links:
            if members.get(link.earlier) is None or members.get(link.earlier) != members.get(link.later):
                raise ValueError("every link must join two tracklets of the same identity")
            if by_tracklet[link.later].start_frame - by_tracklet[link.earlier].end_frame != link.gap_frames:
                raise ValueError("link gap disagrees with its tracklet extents")
        # Link sources must match each identity's declared counts (so a link
        # cannot be relabelled to turn a provisional identity into a confirmed
        # one), and the links must connect every member: a spanning tree, not
        # merely the right number of edges.
        source_counts = Counter((members[link.earlier], link.source) for link in self.links)
        parent = {identifier: identifier for identifier in members}

        def root(item: TrackletId) -> TrackletId:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for link in self.links:
            parent[root(link.earlier)] = root(link.later)
        for identity in self.identities:
            identifier = identity.player_identity_id
            if (
                source_counts[(identifier, LinkSource.AUTOMATIC)] != identity.automatic_links
                or source_counts[(identifier, LinkSource.HUMAN)] != identity.human_links
            ):
                raise ValueError(f"{identifier.value} link count disagrees with links")
            if len({root(item) for item in identity.tracklet_ids}) != 1:
                raise ValueError(f"{identifier.value} links do not connect all of its tracklets")
        self._check_human_links_are_traceable()

    def _check_human_links_are_traceable(self) -> None:
        """Every human link cites one applied must-link of a checksummed log."""
        human = [link for link in self.links if link.source is LinkSource.HUMAN]
        applied: dict[int, frozenset[str]] = {}
        excludes: dict[int, frozenset[str]] = {}
        for item in self.corrections_applied:
            if not isinstance(item, Mapping):
                raise ValueError("corrections_applied entries must be objects")
            target = {"must_link": applied, "exclude": excludes}.get(item.get("action"))
            if target is None or item.get("outcome") != "applied":
                continue
            sequence, members = item.get("sequence"), item.get("tracklet_ids")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                raise ValueError("corrections_applied sequences must be integers")
            if not isinstance(members, list) or not all(isinstance(value, str) for value in members):
                raise ValueError("corrections_applied tracklet_ids must be a list of strings")
            if sequence in applied or sequence in excludes:
                raise ValueError("corrections_applied repeats a sequence")
            target[sequence] = frozenset(members)
        human_excluded = [
            item for item in self.assignments
            if item.status is AssignmentStatus.EXCLUDED and item.reason.startswith("human_exclude:")
        ]
        if (human or applied or human_excluded) and not any(
            item.role == "corrections" for item in self.evidence_inputs
        ):
            raise ValueError("human decisions require a corrections evidence input (its SHA-256)")
        for item in human_excluded:
            match = _HUMAN_EXCLUDE_RE.fullmatch(item.reason)
            if match is None or item.tracklet_id.value not in excludes.get(int(match.group(1)), ()):
                raise ValueError(f"{item.tracklet_id.value} exclusion does not cite an applied exclude correction")
        cited = Counter(link.correction_sequence for link in human)
        if set(cited) != set(applied) or any(count != 1 for count in cited.values()):
            raise ValueError("human links and applied must-link corrections must correspond one to one")
        for link in human:
            if applied[link.correction_sequence] != frozenset((link.earlier.value, link.later.value)):
                raise ValueError(
                    f"human link {link.earlier.value}-{link.later.value} does not match "
                    f"correction {link.correction_sequence}"
                )

    def identity_by_tracklet(self) -> dict[TrackletId, PlayerIdentityId]:
        return {
            item.tracklet_id: item.player_identity_id
            for item in self.assignments
            if item.player_identity_id is not None
        }


def player_identities_to_json_bytes(artifact: PlayerIdentitiesArtifact) -> bytes:
    value = {
        "schema": artifact.schema,
        "source": artifact.source.to_dict(),
        "resolver": dict(artifact.resolver),
        "evidence_inputs": [item.to_dict() for item in artifact.evidence_inputs],
        "identities": [item.to_dict() for item in artifact.identities],
        "assignments": [item.to_dict() for item in artifact.assignments],
        "links": [item.to_dict() for item in artifact.links],
        "abstentions": [item.to_dict() for item in artifact.abstentions],
        "corrections_applied": [dict(item) for item in artifact.corrections_applied],
        "summary": dict(artifact.summary),
        "limitations": list(artifact.limitations),
    }
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _identity_id(value: str | None) -> PlayerIdentityId | None:
    return PlayerIdentityId(value) if value is not None else None


def player_identities_from_json_bytes(content: bytes) -> PlayerIdentitiesArtifact:
    try:
        value = json.loads(content)
        if value.get("schema") != PLAYER_IDENTITIES_SCHEMA:
            raise ValueError(f"schema must be {PLAYER_IDENTITIES_SCHEMA}")
        return PlayerIdentitiesArtifact(
            schema=value["schema"],
            source=SourceDescriptor.from_dict(value["source"]),
            resolver=value["resolver"],
            evidence_inputs=tuple(EvidenceInput(**item) for item in value["evidence_inputs"]),
            identities=tuple(
                PlayerIdentity(
                    player_identity_id=PlayerIdentityId(item["player_identity_id"]),
                    status=AssignmentStatus(item["status"]),
                    tracklet_ids=tuple(TrackletId(identifier) for identifier in item["tracklet_ids"]),
                    start_frame=item["start_frame"],
                    end_frame=item["end_frame"],
                    start_time_seconds=item["start_time_seconds"],
                    end_time_seconds=item["end_time_seconds"],
                    observed_seconds=item["observed_seconds"],
                    observation_count=item["observation_count"],
                    team=TeamLabel(item["team"]),
                    association_score_min=item["association_score_min"],
                    automatic_links=item["automatic_links"],
                    human_links=item["human_links"],
                    evidence_channels=tuple(item["evidence_channels"]),
                )
                for item in value["identities"]
            ),
            assignments=tuple(
                TrackletAssignment(
                    tracklet_id=TrackletId(item["tracklet_id"]),
                    player_identity_id=_identity_id(item["player_identity_id"]),
                    status=AssignmentStatus(item["status"]),
                    reason=item["reason"],
                    start_frame=item["start_frame"],
                    end_frame=item["end_frame"],
                )
                for item in value["assignments"]
            ),
            links=tuple(
                IdentityLink(
                    earlier=TrackletId(item["earlier"]),
                    later=TrackletId(item["later"]),
                    source=LinkSource(item["source"]),
                    correction_sequence=item["correction_sequence"],
                    gap_frames=item["gap_frames"],
                    gap_seconds=item["gap_seconds"],
                    normalized_displacement=item["normalized_displacement"],
                    scale_ratio=item["scale_ratio"],
                    cost=item["cost"],
                    association_score=item["association_score"],
                )
                for item in value["links"]
            ),
            abstentions=tuple(
                Abstention(
                    tracklet_id=TrackletId(item["tracklet_id"]),
                    direction=item["direction"],
                    reason=item["reason"],
                    candidates=tuple(
                        (TrackletId(candidate["tracklet_id"]), candidate["cost"])
                        for candidate in item["candidates"]
                    ),
                )
                for item in value["abstentions"]
            ),
            corrections_applied=tuple(value["corrections_applied"]),
            summary=value["summary"],
            limitations=tuple(value["limitations"]),
        )
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid player identities artifact") from exc


def player_identities_sha256(artifact: PlayerIdentitiesArtifact) -> str:
    return sha256_bytes(player_identities_to_json_bytes(artifact))
