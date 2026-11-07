"""``footballai.pitch-calibration/v1``: per-frame calibration states with their evidence.

The artifact is deterministic JSON (sorted keys, no wall-clock fields). It
stores every input needed to recompute itself — pitch geometry, policy, frame
size, shots, suspected cuts, each keyframe's landmark correspondences, and the
analysed frame list — plus every output. Constructing or loading an artifact
re-derives all keyframe homographies, intervals, frame states and the summary
from that evidence and rejects any disagreement. A homography therefore cannot
be edited into an artifact (or marked calibrated) without also fabricating
landmark evidence that independently passes validation; the evidence itself is
pinned by ``source.annotations_sha256``.

``integrity.content_sha256`` covers the canonical JSON of every other field
and catches accidental corruption. Consumers additionally pin the file's own
SHA-256 (:func:`pitch_calibration_sha256`).
"""

from __future__ import annotations

import json
import math
import platform
import re
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from footballai_v2.execution.ai1.artifacts import sha256_bytes
from footballai_v2.execution.ai3.annotations import (
    AnnotatedFrame,
    AnnotationOrigin,
    AnnotationQuality,
    CorrespondenceRole,
    LandmarkAnnotation,
    PitchAnnotations,
    ReviewStatus,
    ShotSegment,
    check_source_bindings,
)
from footballai_v2.execution.ai3.calibration import (
    CalibrationMethod,
    CalibrationPolicy,
    CalibrationStatus,
    CorrespondenceRecord,
    FrameCalibration,
    KeyframeCalibration,
    PropagationInterval,
    Reason,
    Usage,
    calibrate_keyframe,
    frame_states,
    propagation_intervals,
)
from footballai_v2.execution.ai3.geometry import PitchGeometry


PITCH_CALIBRATION_SCHEMA = "footballai.pitch-calibration/v1"
PIPELINE_NAME = "footballai.ai3.manual-landmark-homography"
PIPELINE_VERSION = "1.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_TOLERANCE_S = 1e-6
_RELATIVE_TOLERANCE = 1e-6
"""Rounded metrics may differ in their last (6th) decimal across platforms."""
_MATRIX_TOLERANCE = 1e-9
"""Homographies are rounded to 12 significant digits; allow only last-digit noise."""

LIMITATIONS = (
    "Homographies come from manually clicked landmarks; no automatic pitch keypoint or line detector exists yet.",
    "A homography assumes a planar pitch and a distortion-free (pinhole) image; broadcast lens distortion is not modelled.",
    "Held-out error is measured on annotated validation landmarks only; it is not football tracking accuracy.",
    "Propagation inside a shot relies on a person declaring the camera static plus two agreeing keyframes; "
    "camera motion between keyframes is not observed.",
    "Suspected-cut input is a heuristic; it can only remove calibration, never add it.",
)


def _sha(value: Any, name: str, *, required: bool) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    source_id: str
    origin: AnnotationOrigin
    frame_width: int
    frame_height: int
    annotations_sha256: str
    video_sha256: str | None
    detections_sha256: str | None
    tracklets_sha256: str | None
    suspected_cuts_source: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, AnnotationOrigin):
            raise ValueError("origin must be an AnnotationOrigin")
        for name in ("frame_width", "frame_height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        _sha(self.annotations_sha256, "annotations_sha256", required=True)
        for name in ("video_sha256", "detections_sha256", "tracklets_sha256"):
            _sha(getattr(self, name), name, required=False)
        if self.suspected_cuts_source is not None and (
            not isinstance(self.suspected_cuts_source, str) or not self.suspected_cuts_source.strip()
        ):
            raise ValueError("suspected_cuts_source must be a non-empty description or None")

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_width, self.frame_height

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "origin": self.origin.value,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "annotations_sha256": self.annotations_sha256,
            "video_sha256": self.video_sha256,
            "detections_sha256": self.detections_sha256,
            "tracklets_sha256": self.tracklets_sha256,
            "suspected_cuts_source": self.suspected_cuts_source,
        }


def software_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__}


def _evidence_frame(keyframe: KeyframeCalibration) -> AnnotatedFrame:
    return AnnotatedFrame(
        keyframe.frame_index, keyframe.timestamp_seconds, keyframe.shot_id,
        tuple(LandmarkAnnotation(
            item.landmark_id, item.pixel_xy[0], item.pixel_xy[1], item.pitch_xy_m[0], item.pitch_xy_m[1],
            item.role, item.quality, item.review_status, item.annotator,
        ) for item in keyframe.correspondences),
    )


def _derive(
    pitch: PitchGeometry,
    policy: CalibrationPolicy,
    frame_size: tuple[int, int],
    shots: Sequence[ShotSegment],
    annotated: Sequence[AnnotatedFrame],
    suspected_cuts: Sequence[int],
    frames: Sequence[tuple[int, float]],
) -> tuple[tuple[KeyframeCalibration, ...], tuple[PropagationInterval, ...], tuple[FrameCalibration, ...]]:
    by_id = {shot.shot_id: shot for shot in shots}
    keyframes = tuple(
        calibrate_keyframe(frame, by_id[frame.shot_id], pitch, frame_size, policy) for frame in annotated
    )
    intervals = propagation_intervals(shots, keyframes, suspected_cuts, policy)
    return keyframes, intervals, frame_states(frames, shots, keyframes, intervals, policy)


def _summary(
    keyframes: Sequence[KeyframeCalibration],
    intervals: Sequence[PropagationInterval],
    frames: Sequence[FrameCalibration],
) -> dict[str, Any]:
    statuses = Counter(item.status.value for item in frames)
    validation = sorted(
        record.error_m
        for keyframe in keyframes if keyframe.status is CalibrationStatus.CALIBRATED
        for record in keyframe.correspondences if record.usage is Usage.VALIDATION and record.error_m is not None
    )
    return {
        "frames_total": len(frames),
        "frames_by_status": {status.value: statuses.get(status.value, 0) for status in CalibrationStatus},
        "frames_by_reason": dict(sorted(Counter(item.reason.value for item in frames if item.reason).items())),
        "frames_by_method": dict(sorted(Counter(item.method.value for item in frames).items())),
        "calibrated_fraction": round(statuses.get("calibrated", 0) / len(frames), 6) if frames else None,
        "keyframes_total": len(keyframes),
        "keyframes_by_status": dict(sorted(Counter(item.status.value for item in keyframes).items())),
        "keyframes_by_reason": dict(sorted(Counter(item.reason.value for item in keyframes if item.reason).items())),
        "intervals_by_status": dict(sorted(Counter(item.status.value for item in intervals).items())),
        "validation_points_on_calibrated_keyframes": len(validation),
        "validation_error_median_m": round(float(np.median(validation)), 6) if validation else None,
        "validation_error_max_m": validation[-1] if validation else None,
    }


def _close(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b:
            raise ValueError(f"recorded value at {path or 'root'} disagrees with the re-derived value")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if "homography" in path:
            relative, absolute = _MATRIX_TOLERANCE, _MATRIX_TOLERANCE * 1e-3
        else:
            relative = absolute = _RELATIVE_TOLERANCE
        if not math.isclose(float(a), float(b), rel_tol=relative, abs_tol=absolute):
            raise ValueError(f"recorded value at {path} disagrees with the re-derived value")
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise ValueError(f"recorded fields at {path or 'root'} disagree with the re-derived fields")
        for key in a:
            _close(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"recorded list at {path} has a different length than re-derived")
        for index, (left, right) in enumerate(zip(a, b)):
            _close(left, right, f"{path}[{index}]")
        return
    raise ValueError(f"recorded value at {path} has an unexpected type")


@dataclass(frozen=True, slots=True)
class PitchCalibrationArtifact:
    schema: str
    pipeline: Mapping[str, Any]
    source: CalibrationSource
    pitch: PitchGeometry
    policy: CalibrationPolicy
    shots: tuple[ShotSegment, ...]
    suspected_cut_frames: tuple[int, ...]
    keyframes: tuple[KeyframeCalibration, ...]
    intervals: tuple[PropagationInterval, ...]
    frames: tuple[FrameCalibration, ...]
    summary: Mapping[str, Any]
    limitations: tuple[str, ...]
    _by_frame: Mapping[int, FrameCalibration] = field(init=False, repr=False, compare=False)
    _sha256: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema != PITCH_CALIBRATION_SCHEMA:
            raise ValueError(f"schema must be {PITCH_CALIBRATION_SCHEMA}")
        if self.pipeline.get("name") != PIPELINE_NAME or self.pipeline.get("version") != PIPELINE_VERSION:
            raise ValueError(f"pipeline must be {PIPELINE_NAME} {PIPELINE_VERSION}")
        object.__setattr__(self, "pipeline", MappingProxyType(dict(self.pipeline)))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in self.suspected_cut_frames):
            raise ValueError("suspected cut frames must be non-negative integers")
        if list(self.suspected_cut_frames) != sorted(set(self.suspected_cut_frames)):
            raise ValueError("suspected cut frames must be sorted and unique")
        indices = [item.frame_index for item in self.frames]
        if indices != sorted(set(indices)):
            raise ValueError("frames must be sorted and unique")
        if not {item.frame_index for item in self.keyframes} <= set(indices):
            raise ValueError("every keyframe must appear in the frame list")
        # Re-derive everything from the recorded evidence; the shots are
        # re-validated by rebuilding a PitchAnnotations container.
        evidence = PitchAnnotations(
            "footballai.pitch-landmark-annotations/v1",
            _annotation_source(self.source), self.pitch, self.shots,
            tuple(_evidence_frame(item) for item in self.keyframes),
        )
        keyframes, intervals, frames = _derive(
            self.pitch, self.policy, self.source.frame_size, evidence.shots, evidence.frames,
            self.suspected_cut_frames, [(item.frame_index, item.timestamp_seconds) for item in self.frames],
        )
        _close([item.to_dict() for item in self.keyframes], [item.to_dict() for item in keyframes], "keyframes")
        _close([item.to_dict() for item in self.intervals], [item.to_dict() for item in intervals], "intervals")
        _close([item.to_dict() for item in self.frames], [item.to_dict() for item in frames], "frames")
        _close(dict(self.summary), _summary(keyframes, intervals, frames), "summary")
        if not all(isinstance(item, str) and item.strip() for item in self.limitations):
            raise ValueError("limitations must be non-empty strings")
        object.__setattr__(self, "_by_frame", MappingProxyType({item.frame_index: item for item in self.frames}))
        object.__setattr__(self, "_sha256", None)

    def sha256(self) -> str:
        """SHA-256 of this artifact's canonical file bytes (computed once)."""
        if self._sha256 is None:
            object.__setattr__(self, "_sha256", sha256_bytes(pitch_calibration_to_json_bytes(self)))
        return self._sha256  # type: ignore[return-value]

    def frame(self, frame_index: int) -> FrameCalibration | None:
        return self._by_frame.get(frame_index)

    def keyframe(self, frame_index: int) -> KeyframeCalibration | None:
        return next((item for item in self.keyframes if item.frame_index == frame_index), None)


def _annotation_source(source: CalibrationSource):
    from footballai_v2.execution.ai3.annotations import AnnotationSourceInfo

    return AnnotationSourceInfo(
        source.source_id, source.origin, source.frame_width, source.frame_height,
        source.video_sha256, source.detections_sha256, source.tracklets_sha256,
    )


def calibrate(
    annotations: PitchAnnotations,
    *,
    annotations_sha256: str,
    policy: CalibrationPolicy,
    frames: Iterable[tuple[int, float]] = (),
    suspected_cut_frames: Iterable[int] = (),
    suspected_cuts_source: str | None = None,
    tracklets_sha256: str | None = None,
    detections_sha256: str | None = None,
    video_sha256: str | None = None,
) -> PitchCalibrationArtifact:
    """Calibrate annotated keyframes and assign a state to every analysed frame.

    ``frames`` are the analysed ``(frame_index, timestamp_seconds)`` pairs, for
    example every AI1 sampled frame. Annotated keyframes are always included.
    Supplied checksums must agree with those the annotations declare. Only
    checksums of inputs actually supplied (and therefore verified by the
    caller) are recorded as the calibration's source binding; a checksum the
    annotation merely declares is not, so projection can never trust an
    unverified binding. The annotation file itself pins its declarations.
    """
    check_source_bindings(
        annotations, tracklets_sha256=tracklets_sha256, detections_sha256=detections_sha256, video_sha256=video_sha256,
    )
    cuts = tuple(sorted(set(int(item) for item in suspected_cut_frames)))
    if cuts and suspected_cuts_source is None:
        raise ValueError("suspected cuts need a source description")
    timeline: dict[int, float] = {}
    for frame_index, timestamp in frames:
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("frame indices must be non-negative integers")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("frame timestamps must be finite and non-negative")
        if timeline.setdefault(frame_index, timestamp) != timestamp:
            raise ValueError(f"frame {frame_index} has two timestamps")
    for frame in annotations.frames:
        existing = timeline.setdefault(frame.frame_index, frame.timestamp_seconds)
        if abs(existing - frame.timestamp_seconds) > _TIMESTAMP_TOLERANCE_S:
            raise ValueError(
                f"annotated frame {frame.frame_index} timestamp {frame.timestamp_seconds} disagrees with the analysed "
                f"frame timestamp {existing}: frame conventions differ"
            )
        timeline[frame.frame_index] = frame.timestamp_seconds
    ordered = sorted(timeline.items())
    if any(b[1] <= a[1] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("frame timestamps must increase with frame index")
    source = CalibrationSource(
        annotations.source.source_id, annotations.source.origin,
        annotations.source.frame_width, annotations.source.frame_height, annotations_sha256,
        video_sha256,
        detections_sha256,
        tracklets_sha256,
        suspected_cuts_source if cuts or suspected_cuts_source else None,
    )
    keyframes, intervals, states = _derive(
        annotations.pitch, policy, source.frame_size, annotations.shots, annotations.frames, cuts, ordered,
    )
    return PitchCalibrationArtifact(
        schema=PITCH_CALIBRATION_SCHEMA,
        pipeline={"name": PIPELINE_NAME, "version": PIPELINE_VERSION, "software": software_versions()},
        source=source,
        pitch=annotations.pitch,
        policy=policy,
        shots=annotations.shots,
        suspected_cut_frames=cuts,
        keyframes=keyframes,
        intervals=intervals,
        frames=states,
        summary=_summary(keyframes, intervals, states),
        limitations=LIMITATIONS,
    )


def _content(artifact: PitchCalibrationArtifact) -> dict[str, Any]:
    return {
        "schema": artifact.schema,
        "pipeline": dict(artifact.pipeline),
        "source": artifact.source.to_dict(),
        "pitch": artifact.pitch.to_dict(),
        "policy": artifact.policy.to_dict(),
        "shots": [item.to_dict() for item in artifact.shots],
        "suspected_cut_frames": list(artifact.suspected_cut_frames),
        "keyframes": [item.to_dict() for item in artifact.keyframes],
        "propagation_intervals": [item.to_dict() for item in artifact.intervals],
        "frames": [item.to_dict() for item in artifact.frames],
        "summary": dict(artifact.summary),
        "limitations": list(artifact.limitations),
    }


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def pitch_calibration_to_json_bytes(artifact: PitchCalibrationArtifact) -> bytes:
    content = _content(artifact)
    content["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON (sorted keys, compact) of every top-level field except integrity",
        "content_sha256": sha256_bytes(_canonical(content)),
    }
    return (json.dumps(content, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def pitch_calibration_sha256(artifact: PitchCalibrationArtifact) -> str:
    return artifact.sha256()


def _require(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} has missing or unknown fields")
    return value


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-number list")
    return float(value[0]), float(value[1])


def _matrix(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError("a homography must be a list of nine numbers")
    return tuple(float(item) for item in value)


def _record(entry: Mapping[str, Any]) -> CorrespondenceRecord:
    entry = _require(entry, "correspondence", {
        "landmark_id", "pixel_xy", "pitch_xy_m", "role", "quality", "review_status", "annotator", "usage",
        "exclusion", "ransac_inlier", "error_m",
    })
    return CorrespondenceRecord(
        entry["landmark_id"], _pair(entry["pixel_xy"], "pixel_xy"), _pair(entry["pitch_xy_m"], "pitch_xy_m"),
        CorrespondenceRole(entry["role"]), AnnotationQuality(entry["quality"]), ReviewStatus(entry["review_status"]),
        entry["annotator"], Usage(entry["usage"]), entry["exclusion"], entry["ransac_inlier"], entry["error_m"],
    )


def _reason(value: Any) -> Reason | None:
    return None if value is None else Reason(value)


def _keyframe_from(item: Any) -> KeyframeCalibration:
    item = _require(item, "keyframe", {
        "frame_index", "timestamp_seconds", "shot_id", "status", "reason", "method", "homography_image_to_pitch",
        "correspondences", "metrics", "fit_hull_pixels",
    })
    if item["method"] != CalibrationMethod.MANUAL_LANDMARKS.value:
        raise ValueError("keyframes are manual-landmark calibrations")
    if not isinstance(item["correspondences"], list) or not isinstance(item["fit_hull_pixels"], list):
        raise ValueError("keyframe correspondences and hull must be lists")
    if not isinstance(item["metrics"], Mapping):
        raise ValueError("keyframe metrics must be an object")
    return KeyframeCalibration(
        item["frame_index"], float(item["timestamp_seconds"]), item["shot_id"], CalibrationStatus(item["status"]),
        _reason(item["reason"]), _matrix(item["homography_image_to_pitch"]),
        tuple(_record(entry) for entry in item["correspondences"]), MappingProxyType(dict(item["metrics"])),
        tuple(_pair(point, "fit_hull_pixels") for point in item["fit_hull_pixels"]),
    )


def _interval_from(item: Any) -> PropagationInterval:
    item = _require(item, "propagation interval", {
        "shot_id", "start_keyframe", "end_keyframe", "status", "reason", "max_cross_error_m",
    })
    return PropagationInterval(
        item["shot_id"], item["start_keyframe"], item["end_keyframe"], CalibrationStatus(item["status"]),
        _reason(item["reason"]), item["max_cross_error_m"],
    )


def _frame_from(item: Any) -> FrameCalibration:
    item = _require(item, "frame", {
        "frame_index", "timestamp_seconds", "shot_id", "status", "method", "reason", "source_keyframe",
        "homography_image_to_pitch",
    })
    if not isinstance(item["frame_index"], int) or isinstance(item["frame_index"], bool):
        raise ValueError("frame_index must be an integer")
    return FrameCalibration(
        item["frame_index"], float(item["timestamp_seconds"]), item["shot_id"], CalibrationStatus(item["status"]),
        CalibrationMethod(item["method"]), _reason(item["reason"]), item["source_keyframe"],
        _matrix(item["homography_image_to_pitch"]),
    )


def pitch_calibration_from_json_bytes(content: bytes) -> PitchCalibrationArtifact:
    """Load the recorded values, then re-derive and compare everything.

    The returned artifact holds exactly the recorded values, so it
    re-serialises to the same bytes on any platform; construction rejects any
    recorded value that the evidence does not reproduce.
    """
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("calibration artifact is not valid JSON") from exc
    top = {
        "schema", "pipeline", "source", "pitch", "policy", "shots", "suspected_cut_frames", "keyframes",
        "propagation_intervals", "frames", "summary", "limitations", "integrity",
    }
    value = _require(value, "calibration artifact", top)
    if value["schema"] != PITCH_CALIBRATION_SCHEMA:
        raise ValueError(f"schema must be {PITCH_CALIBRATION_SCHEMA}")
    integrity = _require(value["integrity"], "integrity", {"algorithm", "scope", "content_sha256"})
    body = {key: item for key, item in value.items() if key != "integrity"}
    if integrity["algorithm"] != "sha256" or integrity["content_sha256"] != sha256_bytes(_canonical(body)):
        raise ValueError("calibration artifact integrity checksum does not match its content")
    try:
        source = _require(value["source"], "source", {
            "source_id", "origin", "frame_width", "frame_height", "annotations_sha256", "video_sha256",
            "detections_sha256", "tracklets_sha256", "suspected_cuts_source",
        })
        lists = ("shots", "keyframes", "propagation_intervals", "frames", "suspected_cut_frames", "limitations")
        if not all(isinstance(value[key], list) for key in lists):
            raise ValueError(f"{', '.join(lists)} must be lists")
        if not isinstance(value["pipeline"], Mapping) or not isinstance(value["summary"], Mapping):
            raise ValueError("pipeline and summary must be objects")
        artifact = PitchCalibrationArtifact(
            schema=value["schema"],
            pipeline=value["pipeline"],
            source=CalibrationSource(
                source["source_id"], AnnotationOrigin(source["origin"]), source["frame_width"], source["frame_height"],
                source["annotations_sha256"], source["video_sha256"], source["detections_sha256"],
                source["tracklets_sha256"], source["suspected_cuts_source"],
            ),
            pitch=PitchGeometry.from_dict(value["pitch"]),
            policy=CalibrationPolicy.from_dict(value["policy"]),
            shots=tuple(ShotSegment.from_dict(item) for item in value["shots"]),
            suspected_cut_frames=tuple(value["suspected_cut_frames"]),
            keyframes=tuple(_keyframe_from(item) for item in value["keyframes"]),
            intervals=tuple(_interval_from(item) for item in value["propagation_intervals"]),
            frames=tuple(_frame_from(item) for item in value["frames"]),
            summary=value["summary"],
            limitations=tuple(value["limitations"]),
        )
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"calibration artifact is malformed: {exc!r}") from exc
    if pitch_calibration_to_json_bytes(artifact) != (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode():
        raise ValueError("calibration artifact does not re-serialise to its own content")
    return artifact
