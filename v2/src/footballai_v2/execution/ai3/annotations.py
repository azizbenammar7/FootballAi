"""``footballai.pitch-landmark-annotations/v1``: manual pixel↔pitch correspondences.

One JSON document per clip. It names the source (video/AI1 checksums and the
native frame size), the explicit pitch geometry, the camera shots, and, per
annotated frame, the landmarks a person clicked.

Every record carries the landmark's pitch coordinates even though they follow
from the geometry: the reader checks they agree, so an annotation made for a
different pitch size is rejected instead of silently re-interpreted.

Roles are chosen by the annotator *before* calibration and are never changed
by the estimator: ``fit`` points may be used to estimate the homography,
``validation`` points never are. That keeps the held-out error independent of
the fit.

Structural problems (unknown landmark, duplicate, out-of-image pixel,
geometry conflict, frame outside its shot...) raise on load. Problems that
only make one frame uncalibratable (too few points, collinear layout,
unreviewed points) become ``UNCALIBRATED`` keyframes with a reason; the CLI's
``validate`` command reports them before calibration.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from footballai_v2.execution.ai3.geometry import CameraSide, PitchGeometry, landmark_matches_geometry


PITCH_ANNOTATIONS_SCHEMA = "footballai.pitch-landmark-annotations/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MIN_DISTINCT_PIXEL_DISTANCE = 1.0
"""Two different landmarks clicked within one pixel are a conflict."""


class CorrespondenceRole(str, Enum):
    FIT = "fit"
    VALIDATION = "validation"


class AnnotationQuality(str, Enum):
    CLEAR = "clear"
    ESTIMATED = "estimated"
    """Partly occluded or blurred; the annotator estimated the intersection."""


class ReviewStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class AnnotationOrigin(str, Enum):
    MANUAL = "manual"
    SYNTHETIC = "synthetic"
    """Generated from a known camera for tests and benchmarks; never real evidence."""


class CameraMotion(str, Enum):
    STATIC = "static"
    MOVING = "moving"
    UNKNOWN = "unknown"


def _sha(value: Any, name: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _int(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a short identifier")
    return value


def _fields(value: Any, name: str, required: set[str], optional: frozenset[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    keys = set(value)
    if required - keys:
        raise ValueError(f"{name} is missing {sorted(required - keys)}")
    if keys - required - optional:
        raise ValueError(f"{name} has unknown fields {sorted(keys - required - optional)}")
    return value


@dataclass(frozen=True, slots=True)
class AnnotationSourceInfo:
    source_id: str
    origin: AnnotationOrigin
    frame_width: int
    frame_height: int
    video_sha256: str | None
    detections_sha256: str | None
    tracklets_sha256: str | None

    def __post_init__(self) -> None:
        _identifier(self.source_id, "source_id")
        if not isinstance(self.origin, AnnotationOrigin):
            raise ValueError("origin must be an AnnotationOrigin")
        _int(self.frame_width, "frame_width", minimum=1)
        _int(self.frame_height, "frame_height", minimum=1)
        _sha(self.video_sha256, "video_sha256", required=False)
        _sha(self.detections_sha256, "detections_sha256", required=False)
        _sha(self.tracklets_sha256, "tracklets_sha256", required=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "origin": self.origin.value,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "video_sha256": self.video_sha256,
            "detections_sha256": self.detections_sha256,
            "tracklets_sha256": self.tracklets_sha256,
        }


@dataclass(frozen=True, slots=True)
class ShotSegment:
    """An inclusive frame interval filmed by one continuous camera take.

    ``boundary_source`` says who drew the boundary; only ``manual`` exists in v1.
    ``camera_motion`` is a claim by that person; AI3 still requires agreeing
    keyframes before treating a shot as static.
    """

    shot_id: str
    start_frame: int
    end_frame: int
    camera_motion: CameraMotion
    camera_side: CameraSide
    boundary_source: str = "manual"

    def __post_init__(self) -> None:
        _identifier(self.shot_id, "shot_id")
        _int(self.start_frame, "start_frame", minimum=0)
        _int(self.end_frame, "end_frame", minimum=self.start_frame)
        if not isinstance(self.camera_motion, CameraMotion) or not isinstance(self.camera_side, CameraSide):
            raise ValueError("camera_motion and camera_side must be enums")
        if self.boundary_source != "manual":
            raise ValueError("only manual shot boundaries are supported in v1")

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "camera_motion": self.camera_motion.value,
            "camera_side": self.camera_side.value,
            "boundary_source": self.boundary_source,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ShotSegment":
        item = _fields(value, "shot", {"shot_id", "start_frame", "end_frame", "camera_motion", "camera_side", "boundary_source"})
        return cls(
            item["shot_id"], item["start_frame"], item["end_frame"],
            CameraMotion(item["camera_motion"]), CameraSide(item["camera_side"]), item["boundary_source"],
        )


@dataclass(frozen=True, slots=True)
class LandmarkAnnotation:
    landmark_id: str
    pixel_x: float
    pixel_y: float
    pitch_x_m: float
    pitch_y_m: float
    role: CorrespondenceRole
    quality: AnnotationQuality
    review_status: ReviewStatus
    annotator: str

    def __post_init__(self) -> None:
        for name in ("pixel_x", "pixel_y", "pitch_x_m", "pitch_y_m"):
            object.__setattr__(self, name, _float(getattr(self, name), name))
        if not isinstance(self.role, CorrespondenceRole) or not isinstance(self.quality, AnnotationQuality):
            raise ValueError("role and quality must be enums")
        if not isinstance(self.review_status, ReviewStatus):
            raise ValueError("review_status must be a ReviewStatus")
        if not isinstance(self.annotator, str) or not self.annotator.strip() or len(self.annotator) > 128:
            raise ValueError("annotator must be a short non-empty name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "pitch_x_m": self.pitch_x_m,
            "pitch_y_m": self.pitch_y_m,
            "role": self.role.value,
            "quality": self.quality.value,
            "review_status": self.review_status.value,
            "annotator": self.annotator,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LandmarkAnnotation":
        item = _fields(value, "landmark", {
            "landmark_id", "pixel_x", "pixel_y", "pitch_x_m", "pitch_y_m", "role", "quality", "review_status", "annotator",
        })
        if not isinstance(item["landmark_id"], str):
            raise ValueError("landmark_id must be a string")
        return cls(
            item["landmark_id"], item["pixel_x"], item["pixel_y"], item["pitch_x_m"], item["pitch_y_m"],
            CorrespondenceRole(item["role"]), AnnotationQuality(item["quality"]), ReviewStatus(item["review_status"]),
            item["annotator"],
        )


@dataclass(frozen=True, slots=True)
class AnnotatedFrame:
    frame_index: int
    timestamp_seconds: float
    shot_id: str
    landmarks: tuple[LandmarkAnnotation, ...]
    frame_image_sha256: str | None = None

    def __post_init__(self) -> None:
        _int(self.frame_index, "frame_index", minimum=0)
        if _float(self.timestamp_seconds, "timestamp_seconds") < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        _identifier(self.shot_id, "shot_id")
        if not isinstance(self.landmarks, tuple):
            raise ValueError("landmarks must be a tuple")
        _sha(self.frame_image_sha256, "frame_image_sha256", required=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "shot_id": self.shot_id,
            "frame_image_sha256": self.frame_image_sha256,
            "landmarks": [item.to_dict() for item in self.landmarks],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "AnnotatedFrame":
        item = _fields(value, "frame", {"frame_index", "timestamp_seconds", "shot_id", "landmarks"}, frozenset({"frame_image_sha256"}))
        if not isinstance(item["landmarks"], list):
            raise ValueError("frame landmarks must be a list")
        return cls(
            item["frame_index"], item["timestamp_seconds"], item["shot_id"],
            tuple(LandmarkAnnotation.from_dict(entry) for entry in item["landmarks"]),
            item.get("frame_image_sha256"),
        )


@dataclass(frozen=True, slots=True)
class PitchAnnotations:
    schema: str
    source: AnnotationSourceInfo
    pitch: PitchGeometry
    shots: tuple[ShotSegment, ...]
    frames: tuple[AnnotatedFrame, ...]

    def __post_init__(self) -> None:
        if self.schema != PITCH_ANNOTATIONS_SCHEMA:
            raise ValueError(f"schema must be {PITCH_ANNOTATIONS_SCHEMA}")
        if not isinstance(self.source, AnnotationSourceInfo) or not isinstance(self.pitch, PitchGeometry):
            raise ValueError("source and pitch are required")
        if not isinstance(self.shots, tuple) or not isinstance(self.frames, tuple):
            raise ValueError("shots and frames must be tuples")
        previous_end = -1
        shot_ids: set[str] = set()
        for shot in self.shots:
            if shot.shot_id in shot_ids:
                raise ValueError(f"duplicate shot_id {shot.shot_id}")
            if shot.start_frame <= previous_end:
                raise ValueError("shots must be sorted and must not overlap")
            shot_ids.add(shot.shot_id)
            previous_end = shot.end_frame
        shots = {shot.shot_id: shot for shot in self.shots}
        previous_frame, previous_time = -1, -1.0
        for frame in self.frames:
            if frame.frame_index <= previous_frame or frame.timestamp_seconds <= previous_time:
                raise ValueError("annotated frames must be unique and strictly increasing in index and time")
            previous_frame, previous_time = frame.frame_index, frame.timestamp_seconds
            shot = shots.get(frame.shot_id)
            if shot is None or not shot.contains(frame.frame_index):
                raise ValueError(f"frame {frame.frame_index} is not inside its shot {frame.shot_id}")
            self._check_frame(frame)

    def _check_frame(self, frame: AnnotatedFrame) -> None:
        width, height = self.source.frame_width, self.source.frame_height
        seen: dict[str, LandmarkAnnotation] = {}
        for item in frame.landmarks:
            where = f"frame {frame.frame_index} landmark {item.landmark_id!r}"
            if item.landmark_id not in self.pitch.landmarks():
                raise ValueError(f"{where}: unknown pitch landmark")
            if item.landmark_id in seen:
                raise ValueError(f"{where}: duplicate landmark in one frame")
            if not landmark_matches_geometry(self.pitch, item.landmark_id, item.pitch_x_m, item.pitch_y_m):
                raise ValueError(f"{where}: pitch coordinates conflict with the configured pitch geometry")
            if not (0 <= item.pixel_x <= width and 0 <= item.pixel_y <= height):
                raise ValueError(f"{where}: pixel lies outside the {width}×{height} source frame")
            for other in seen.values():
                if math.hypot(other.pixel_x - item.pixel_x, other.pixel_y - item.pixel_y) < MIN_DISTINCT_PIXEL_DISTANCE:
                    raise ValueError(f"{where}: conflicts with {other.landmark_id!r} at the same pixel")
            seen[item.landmark_id] = item

    def shot(self, shot_id: str) -> ShotSegment:
        return next(item for item in self.shots if item.shot_id == shot_id)

    def shot_for_frame(self, frame_index: int) -> ShotSegment | None:
        return next((item for item in self.shots if item.contains(frame_index)), None)


def annotations_to_json_bytes(annotations: PitchAnnotations) -> bytes:
    value = {
        "schema": annotations.schema,
        "source": annotations.source.to_dict(),
        "pitch": annotations.pitch.to_dict(),
        "shots": [item.to_dict() for item in annotations.shots],
        "frames": [item.to_dict() for item in annotations.frames],
    }
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def annotations_from_json_bytes(content: bytes) -> PitchAnnotations:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("annotations are not valid JSON") from exc
    item = _fields(value, "annotations", {"schema", "source", "pitch", "shots", "frames"})
    if item["schema"] != PITCH_ANNOTATIONS_SCHEMA:
        raise ValueError(f"schema must be {PITCH_ANNOTATIONS_SCHEMA}")
    source = _fields(item["source"], "source", {
        "source_id", "origin", "frame_width", "frame_height", "video_sha256", "detections_sha256", "tracklets_sha256",
    })
    if not isinstance(item["shots"], list) or not isinstance(item["frames"], list):
        raise ValueError("shots and frames must be lists")
    return PitchAnnotations(
        schema=item["schema"],
        source=AnnotationSourceInfo(
            source["source_id"], AnnotationOrigin(source["origin"]), source["frame_width"], source["frame_height"],
            source["video_sha256"], source["detections_sha256"], source["tracklets_sha256"],
        ),
        pitch=PitchGeometry.from_dict(item["pitch"]),
        shots=tuple(ShotSegment.from_dict(entry) for entry in item["shots"]),
        frames=tuple(AnnotatedFrame.from_dict(entry) for entry in item["frames"]),
    )


def check_source_bindings(
    annotations: PitchAnnotations,
    *,
    tracklets_sha256: str | None = None,
    detections_sha256: str | None = None,
    video_sha256: str | None = None,
    frame_size: tuple[int, int] | None = None,
) -> None:
    """Reject annotations recorded against a different source than the one supplied."""
    source = annotations.source
    for name, declared, supplied in (
        ("tracklets_sha256", source.tracklets_sha256, tracklets_sha256),
        ("detections_sha256", source.detections_sha256, detections_sha256),
        ("video_sha256", source.video_sha256, video_sha256),
    ):
        if supplied is not None and declared is not None and declared != supplied:
            raise ValueError(f"source checksum mismatch: annotations declare {name}={declared}, input is {supplied}")
    if frame_size is not None and frame_size != (source.frame_width, source.frame_height):
        raise ValueError(
            f"source frame size mismatch: annotations use {source.frame_width}×{source.frame_height}, "
            f"input is {frame_size[0]}×{frame_size[1]}"
        )
