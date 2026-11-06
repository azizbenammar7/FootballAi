"""Downstream contract: per-observation player view for AI3/AI7 consumers.

``iter_player_observations`` joins AI1 tracklet observations with an AI2C
identity artifact, one observation at a time, so a full match never needs a
second in-memory copy. It never fabricates physical coordinates: ``pitch_xy_m``
is ``None`` and ``pitch_calibration`` is ``"uncalibrated"`` until a validated
AI3 calibration supplies them.

By default only ``confirmed`` (human-reviewed) identities are yielded. The
AI2C benchmark measured a false merge on a held-out clip, so ``provisional``
automatic identities are review candidates: a caller must opt in with
``include_provisional=True`` and propagate that status to anything it derives.
``unresolved`` observations are tracklet-only diagnostics, never player data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from footballai_v2.execution.ai1.artifacts import BoundingBox, TrackletsArtifact
from footballai_v2.execution.ai2a.model import PlayerIdentityId, TeamLabel, TrackletId, tracklet_id
from footballai_v2.execution.ai2c.artifact import AssignmentStatus, PlayerIdentitiesArtifact
from footballai_v2.execution.ai2c.inputs import InputKind


PLAYER_OBSERVATION_CONTRACT = "footballai.player-observation/v1"


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    frame_index: int
    timestamp_seconds: float
    tracklet_id: TrackletId
    player_identity_id: PlayerIdentityId | None
    identity_status: AssignmentStatus
    identity_association_score: float | None
    team: TeamLabel
    bbox: BoundingBox
    foot_pixel_xy: tuple[float, float]
    tracking_confidence: float
    pitch_xy_m: tuple[float, float] | None
    pitch_calibration: str
    source_tracklets_sha256: str
    source_identities_sha256: str

    def __post_init__(self) -> None:
        if self.pitch_xy_m is not None or self.pitch_calibration != "uncalibrated":
            raise ValueError("pitch coordinates require a validated calibration (AI3); none exists")
        if (self.player_identity_id is None) == (
            self.identity_status in (AssignmentStatus.CONFIRMED, AssignmentStatus.PROVISIONAL)
        ):
            raise ValueError("identity status and player identity disagree")


def iter_player_observations(
    tracklets: TrackletsArtifact,
    tracklets_sha256: str,
    identities: PlayerIdentitiesArtifact,
    identities_sha256: str,
    *,
    include_provisional: bool = False,
    include_unresolved: bool = False,
) -> Iterator[PlayerObservation]:
    if identities.source.input_kind is not InputKind.TRACKLETS:
        raise ValueError("observations can only be joined to identities resolved from AI1 tracklets")
    if identities.source.tracklets_sha256 != tracklets_sha256:
        raise ValueError("identities were resolved from a different tracklets artifact")
    assignments = {item.tracklet_id: item for item in identities.assignments}
    details = {item.player_identity_id: item for item in identities.identities}
    for tracklet in sorted(tracklets.tracklets, key=lambda item: item.track_id):
        identifier = tracklet_id(tracklet.track_id)
        assignment = assignments.get(identifier)
        if assignment is None:
            raise ValueError(f"{identifier.value} has no identity assignment")
        if assignment.status is AssignmentStatus.EXCLUDED:
            continue
        if assignment.status is AssignmentStatus.UNRESOLVED and not include_unresolved:
            continue
        if assignment.status is AssignmentStatus.PROVISIONAL and not include_provisional:
            continue
        identity = details.get(assignment.player_identity_id) if assignment.player_identity_id else None
        for item in tracklet.observations:
            yield PlayerObservation(
                frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds,
                tracklet_id=identifier,
                player_identity_id=assignment.player_identity_id,
                identity_status=assignment.status,
                identity_association_score=identity.association_score_min if identity else None,
                team=identity.team if identity else TeamLabel.UNKNOWN,
                bbox=item.bbox,
                foot_pixel_xy=(item.bbox.center_x, item.bbox.y2),
                tracking_confidence=item.confidence,
                pitch_xy_m=None,
                pitch_calibration="uncalibrated",
                source_tracklets_sha256=tracklets_sha256,
                source_identities_sha256=identities_sha256,
            )
