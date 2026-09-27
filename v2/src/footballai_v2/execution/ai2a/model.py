"""Typed, versioned ground truth for AI2A identity evaluation.

``player_identity_id`` is deliberately an evaluation label in AI2A. Nothing in
this module infers or publishes a production player identity.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from footballai_v2.execution.ai1.artifacts import TrackletsArtifact


GROUND_TRUTH_SCHEMA = "footballai.identity-ground-truth/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DETECTION_ID_RE = re.compile(r"^detection_[0-9]{6,}_[0-9]{3,}$")
_TRACKLET_ID_RE = re.compile(r"^tracklet_[0-9]{6,}$")
_PLAYER_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, order=True, slots=True)
class DetectionId:
    """Stable evaluation reference to one ordered AI1 detection row."""

    value: str

    def __post_init__(self) -> None:
        if not _DETECTION_ID_RE.fullmatch(self.value):
            raise ValueError("invalid detection_id")

    @classmethod
    def from_frame_ordinal(cls, frame_index: int, ordinal: int) -> "DetectionId":
        if frame_index < 0 or ordinal < 0:
            raise ValueError("detection frame and ordinal must be non-negative")
        return cls(f"detection_{frame_index:06d}_{ordinal:03d}")


@dataclass(frozen=True, order=True, slots=True)
class TrackletId:
    """Stable evaluation reference to one AI1 temporary track."""

    value: str

    def __post_init__(self) -> None:
        if not _TRACKLET_ID_RE.fullmatch(self.value):
            raise ValueError("invalid tracklet_id")

    @classmethod
    def from_ai1_track_id(cls, value: int) -> "TrackletId":
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("AI1 track ID must be a positive integer")
        return cls(f"tracklet_{value:06d}")

    @property
    def ai1_track_id(self) -> int:
        return int(self.value.removeprefix("tracklet_"))


def tracklet_id(value: int) -> TrackletId:
    return TrackletId.from_ai1_track_id(value)


@dataclass(frozen=True, order=True, slots=True)
class PlayerIdentityId:
    """Pseudonymous ground-truth/manual/future-resolver identity label."""

    value: str

    def __post_init__(self) -> None:
        if not _PLAYER_ID_RE.fullmatch(self.value):
            raise ValueError("player_identity_id must be a stable pseudonymous label")


class TeamLabel(str, Enum):
    TEAM_A = "TEAM_A"
    TEAM_B = "TEAM_B"
    UNKNOWN = "UNKNOWN"


class AnnotationStatus(str, Enum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ReviewState(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class SourceReference:
    source_id: str
    video_sha256: str
    duration_seconds: float
    detections_sha256: str
    tracklets_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        for name in ("video_sha256", "detections_sha256", "tracklets_sha256"):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if _finite(self.duration_seconds, "duration_seconds") <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True, slots=True)
class TrackletAnnotation:
    tracklet_id: TrackletId
    ground_truth_player_id: PlayerIdentityId | None
    team_label: TeamLabel
    role_label: str | None
    start_frame: int
    end_frame: int
    start_time_seconds: float
    end_time_seconds: float
    annotation_confidence: float | None
    status: AnnotationStatus
    annotator: str
    source: str
    notes: str
    review_state: ReviewState

    def __post_init__(self) -> None:
        if not isinstance(self.tracklet_id, TrackletId):
            raise ValueError("tracklet_id must be typed")
        if self.ground_truth_player_id is not None and not isinstance(
            self.ground_truth_player_id, PlayerIdentityId
        ):
            raise ValueError("ground_truth_player_id must be typed")
        if self.status is AnnotationStatus.CONFIRMED and self.ground_truth_player_id is None:
            raise ValueError("confirmed annotations require a player identity")
        if self.status is AnnotationStatus.UNKNOWN and self.ground_truth_player_id is not None:
            raise ValueError("unknown annotations cannot claim a player identity")
        if self.ground_truth_player_id is not None:
            prefix = self.ground_truth_player_id.value.split("_", 2)[:2]
            expected_team = "_".join(prefix)
            if expected_team in {TeamLabel.TEAM_A.value, TeamLabel.TEAM_B.value}:
                if self.team_label.value != expected_team:
                    raise ValueError("player identity and team label disagree")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("annotation frame interval is invalid")
        start = _finite(self.start_time_seconds, "start_time_seconds")
        end = _finite(self.end_time_seconds, "end_time_seconds")
        if start < 0 or end < start:
            raise ValueError("annotation time interval is invalid")
        if self.annotation_confidence is not None:
            confidence = _finite(self.annotation_confidence, "annotation_confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("annotation_confidence must be between zero and one")
        for name in ("annotator", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.role_label is not None and (not isinstance(self.role_label, str) or not self.role_label.strip()):
            raise ValueError("role_label must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class IdentityGroundTruthArtifact:
    schema: str
    source: SourceReference
    annotations: tuple[TrackletAnnotation, ...]

    def __post_init__(self) -> None:
        if self.schema != GROUND_TRUTH_SCHEMA:
            raise ValueError(f"schema must be {GROUND_TRUTH_SCHEMA}")
        if not isinstance(self.annotations, tuple):
            raise ValueError("annotations must be a tuple")
        ids = [item.tracklet_id for item in self.annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("a tracklet may have only one annotation")

    def validate_identity_assignments(
        self,
        cannot_link_pairs: set[frozenset[TrackletId]] | None = None,
    ) -> None:
        labelled = [item for item in self.annotations if item.ground_truth_player_id is not None]
        forbidden = cannot_link_pairs or set()
        for index, left in enumerate(labelled):
            for right in labelled[index + 1 :]:
                if left.ground_truth_player_id != right.ground_truth_player_id:
                    continue
                if frozenset((left.tracklet_id, right.tracklet_id)) in forbidden:
                    raise ValueError(
                        f"identity {left.ground_truth_player_id.value} violates a cannot-link constraint"
                    )

    def validate_against_tracklets(self, tracklets: TrackletsArtifact) -> None:
        by_id = {tracklet_id(item.track_id): item for item in tracklets.tracklets}
        for annotation in self.annotations:
            item = by_id.get(annotation.tracklet_id)
            if item is None:
                raise ValueError(f"unknown tracklet reference: {annotation.tracklet_id.value}")
            expected = (
                item.start_frame,
                item.end_frame,
                item.start_time_seconds,
                item.end_time_seconds,
            )
            actual = (
                annotation.start_frame,
                annotation.end_frame,
                annotation.start_time_seconds,
                annotation.end_time_seconds,
            )
            if expected[:2] != actual[:2] or not all(
                math.isclose(left, right, abs_tol=1e-9) for left, right in zip(expected[2:], actual[2:])
            ):
                raise ValueError(f"annotation extent does not match {annotation.tracklet_id.value}")


def _annotation_to_dict(item: TrackletAnnotation) -> dict[str, Any]:
    return {
        "tracklet_id": item.tracklet_id.value,
        "ground_truth_player_id": item.ground_truth_player_id.value if item.ground_truth_player_id else None,
        "team_label": item.team_label.value,
        "role_label": item.role_label,
        "start_frame": item.start_frame,
        "end_frame": item.end_frame,
        "start_time_seconds": item.start_time_seconds,
        "end_time_seconds": item.end_time_seconds,
        "annotation_confidence": item.annotation_confidence,
        "status": item.status.value,
        "annotator": item.annotator,
        "source": item.source,
        "notes": item.notes,
        "review_state": item.review_state.value,
    }


def ground_truth_to_json_bytes(artifact: IdentityGroundTruthArtifact) -> bytes:
    value = {
        "schema": artifact.schema,
        "source": asdict(artifact.source),
        "annotations": [_annotation_to_dict(item) for item in artifact.annotations],
    }
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def ground_truth_from_json_bytes(
    content: bytes,
    *,
    tracklets: TrackletsArtifact | None = None,
) -> IdentityGroundTruthArtifact:
    try:
        value = json.loads(content)
        source = SourceReference(**value["source"])
        annotations = tuple(
            TrackletAnnotation(
                tracklet_id=TrackletId(item["tracklet_id"]),
                ground_truth_player_id=(
                    PlayerIdentityId(item["ground_truth_player_id"])
                    if item.get("ground_truth_player_id")
                    else None
                ),
                team_label=TeamLabel(item["team_label"]),
                role_label=item.get("role_label"),
                start_frame=item["start_frame"],
                end_frame=item["end_frame"],
                start_time_seconds=item["start_time_seconds"],
                end_time_seconds=item["end_time_seconds"],
                annotation_confidence=item.get("annotation_confidence"),
                status=AnnotationStatus(item["status"]),
                annotator=item["annotator"],
                source=item["source"],
                notes=item.get("notes", ""),
                review_state=ReviewState(item["review_state"]),
            )
            for item in value["annotations"]
        )
        artifact = IdentityGroundTruthArtifact(value["schema"], source, annotations)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid identity ground-truth artifact") from exc
    if tracklets is not None:
        artifact.validate_against_tracklets(tracklets)
        from footballai_v2.execution.ai2a.association import generate_temporal_cannot_links

        cannot_links = generate_temporal_cannot_links(tracklets)
        artifact.validate_identity_assignments({
            frozenset((item.left, item.right)) for item in cannot_links
        })
    else:
        artifact.validate_identity_assignments()
    return artifact
