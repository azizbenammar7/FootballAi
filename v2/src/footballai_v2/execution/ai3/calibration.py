"""Quality-gated calibration: keyframe homographies and per-frame validity.

Two layers, deliberately separate:

1. **Keyframes**: one annotated frame → one RANSAC homography from its ``fit``
   landmarks, then independent checks. Its ``validation`` landmarks are never
   used for fitting; their reprojection error is the held-out accuracy.
2. **Frame states**: every analysed frame gets ``CALIBRATED``, ``UNCALIBRATED``
   or ``INVALIDATED``. A homography is only valid for the exact keyframe, or —
   within a shot a person declared *static* — between two consecutive
   calibrated keyframes that each reproject the other's landmarks within the
   validation tolerance, with no suspected cut between them. Moving and
   unknown shots never propagate. Shot boundaries always stop propagation.

``UNCALIBRATED`` means there is not enough evidence to estimate a homography
for the frame. ``INVALIDATED`` means a homography was estimated or propagated
but evidence contradicts it (failed check, disagreeing keyframes, suspected
cut). Neither carries a matrix, and neither falls back to any pixel→metre
scale.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from footballai_v2.execution.ai3 import homography as hg
from footballai_v2.execution.ai3.annotations import (
    AnnotatedFrame,
    AnnotationQuality,
    CameraMotion,
    CorrespondenceRole,
    ReviewStatus,
    ShotSegment,
)
from footballai_v2.execution.ai3.geometry import CameraSide, PitchGeometry


CALIBRATION_POLICY_VERSION = "ai3-calibration-policy/v1"
_METRIC_DECIMALS = 6


class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    INVALIDATED = "invalidated"


class CalibrationMethod(str, Enum):
    MANUAL_LANDMARKS = "manual_landmark_homography"
    STATIC_SHOT_PROPAGATION = "static_shot_keyframe_propagation"
    NONE = "none"


class Reason(str, Enum):
    # UNCALIBRATED: not enough evidence.
    NO_SHOT_SEGMENT = "no_shot_segment"
    NO_KEYFRAME_FOR_FRAME = "no_keyframe_for_frame"
    CAMERA_NOT_DECLARED_STATIC = "camera_not_declared_static"
    OUTSIDE_VERIFIED_INTERVAL = "outside_verified_keyframe_interval"
    NO_VERIFIED_KEYFRAME_PAIR = "no_verified_keyframe_pair"
    INSUFFICIENT_FIT_POINTS = "insufficient_fit_points"
    INSUFFICIENT_VALIDATION_POINTS = "insufficient_validation_points"
    DEGENERATE_GEOMETRY = "degenerate_or_collinear_geometry"
    INSUFFICIENT_INLIERS = "insufficient_ransac_inliers"
    ILL_CONDITIONED = "ill_conditioned_estimate"
    # INVALIDATED: evidence contradicts an estimate.
    NON_FINITE_OR_SINGULAR = "non_finite_or_singular_homography"
    HORIZON_CROSSING = "landmarks_beyond_camera_horizon"
    REFLECTED_ORIENTATION = "reflected_orientation"
    CAMERA_SIDE_INCONSISTENT = "camera_side_inconsistent"
    IMPLAUSIBLE_SCALE = "implausible_metres_per_pixel"
    INSUFFICIENT_COVERAGE = "insufficient_spatial_coverage"
    VALIDATION_ERROR_EXCEEDED = "validation_error_exceeds_threshold"
    KEYFRAMES_DISAGREE = "static_shot_keyframes_disagree"
    SUSPECTED_CAMERA_CUT = "suspected_camera_cut"


INVALIDATING_REASONS = frozenset({
    Reason.NON_FINITE_OR_SINGULAR, Reason.HORIZON_CROSSING, Reason.REFLECTED_ORIENTATION,
    Reason.CAMERA_SIDE_INCONSISTENT, Reason.IMPLAUSIBLE_SCALE, Reason.INSUFFICIENT_COVERAGE,
    Reason.VALIDATION_ERROR_EXCEEDED, Reason.KEYFRAMES_DISAGREE, Reason.SUSPECTED_CAMERA_CUT,
})


def status_for(reason: Reason | None) -> CalibrationStatus:
    if reason is None:
        return CalibrationStatus.CALIBRATED
    return CalibrationStatus.INVALIDATED if reason in INVALIDATING_REASONS else CalibrationStatus.UNCALIBRATED


def _number(value: Any, name: str, *, minimum: float, maximum: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return float(value)


def _count(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    """Versioned acceptance thresholds. See docs/v2/AI3_PITCH_CALIBRATION.md for rationale.

    Chosen from geometry and annotation-precision arguments before any
    evaluation; they were not tuned on synthetic or real results.
    """

    version: str = CALIBRATION_POLICY_VERSION
    min_fit_points: int = 4
    min_fit_inliers: int = 4
    min_inlier_ratio: float = 0.6
    min_validation_points: int = 2
    ransac_threshold_m: float = 1.0
    ransac_max_subsets: int = 5000
    ransac_seed: int = 0
    ransac_max_refinements: int = 5
    min_triangle_ratio: float = 0.02
    min_dlt_singular_ratio: float = 1e-9
    max_validation_median_error_m: float = 0.75
    max_validation_error_m: float = 1.5
    min_image_hull_fraction: float = 0.05
    min_metres_per_pixel: float = 0.002
    max_metres_per_pixel: float = 1.0
    accepted_review_statuses: tuple[str, ...] = (ReviewStatus.ACCEPTED.value,)
    validation_requires_clear: bool = True
    propagate_static_shots: bool = True

    def __post_init__(self) -> None:
        if self.version != CALIBRATION_POLICY_VERSION:
            raise ValueError(f"policy version must be {CALIBRATION_POLICY_VERSION}")
        _count(self.min_fit_points, "min_fit_points", minimum=4)
        _count(self.min_fit_inliers, "min_fit_inliers", minimum=4)
        _count(self.min_validation_points, "min_validation_points", minimum=1)
        _count(self.ransac_max_subsets, "ransac_max_subsets", minimum=1)
        _count(self.ransac_seed, "ransac_seed", minimum=0)
        _count(self.ransac_max_refinements, "ransac_max_refinements", minimum=1)
        _number(self.min_inlier_ratio, "min_inlier_ratio", minimum=0, maximum=1)
        _number(self.ransac_threshold_m, "ransac_threshold_m", minimum=1e-6)
        _number(self.min_triangle_ratio, "min_triangle_ratio", minimum=0, maximum=1)
        _number(self.min_dlt_singular_ratio, "min_dlt_singular_ratio", minimum=0, maximum=1)
        _number(self.max_validation_median_error_m, "max_validation_median_error_m", minimum=1e-6)
        _number(self.max_validation_error_m, "max_validation_error_m", minimum=self.max_validation_median_error_m)
        _number(self.min_image_hull_fraction, "min_image_hull_fraction", minimum=0, maximum=1)
        _number(self.min_metres_per_pixel, "min_metres_per_pixel", minimum=1e-9)
        _number(self.max_metres_per_pixel, "max_metres_per_pixel", minimum=self.min_metres_per_pixel)
        if not isinstance(self.accepted_review_statuses, tuple) or not self.accepted_review_statuses:
            raise ValueError("accepted_review_statuses must be a non-empty tuple")
        for status in self.accepted_review_statuses:
            if ReviewStatus(status) is ReviewStatus.REJECTED:
                raise ValueError("rejected annotations can never be accepted")
        if not isinstance(self.validation_requires_clear, bool) or not isinstance(self.propagate_static_shots, bool):
            raise ValueError("policy switches must be booleans")

    def to_dict(self) -> dict[str, Any]:
        value = {name: getattr(self, name) for name in self.__dataclass_fields__}
        value["accepted_review_statuses"] = list(self.accepted_review_statuses)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationPolicy":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("calibration policy has missing or unknown fields")
        items = dict(value)
        if not isinstance(items["accepted_review_statuses"], list):
            raise ValueError("accepted_review_statuses must be a list")
        items["accepted_review_statuses"] = tuple(items["accepted_review_statuses"])
        return cls(**items)


# --- correspondences ----------------------------------------------------------------


class Usage(str, Enum):
    FIT = "fit"
    VALIDATION = "validation"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class CorrespondenceRecord:
    landmark_id: str
    pixel_xy: tuple[float, float]
    pitch_xy_m: tuple[float, float]
    role: CorrespondenceRole
    quality: AnnotationQuality
    review_status: ReviewStatus
    annotator: str
    usage: Usage
    exclusion: str | None
    ransac_inlier: bool | None
    error_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "pixel_xy": list(self.pixel_xy),
            "pitch_xy_m": list(self.pitch_xy_m),
            "role": self.role.value,
            "quality": self.quality.value,
            "review_status": self.review_status.value,
            "annotator": self.annotator,
            "usage": self.usage.value,
            "exclusion": self.exclusion,
            "ransac_inlier": self.ransac_inlier,
            "error_m": self.error_m,
        }


def _classify(frame: AnnotatedFrame, policy: CalibrationPolicy) -> list[CorrespondenceRecord]:
    records = []
    for item in sorted(frame.landmarks, key=lambda entry: entry.landmark_id):
        exclusion = None
        if item.review_status.value not in policy.accepted_review_statuses:
            exclusion = f"review_{item.review_status.value}"
        elif (
            item.role is CorrespondenceRole.VALIDATION
            and policy.validation_requires_clear
            and item.quality is not AnnotationQuality.CLEAR
        ):
            exclusion = "validation_point_not_clear"
        usage = Usage.EXCLUDED if exclusion else Usage(item.role.value)
        records.append(CorrespondenceRecord(
            item.landmark_id, (item.pixel_x, item.pixel_y), (item.pitch_x_m, item.pitch_y_m),
            item.role, item.quality, item.review_status, item.annotator, usage, exclusion, None, None,
        ))
    return records


def _array(records: Sequence[CorrespondenceRecord], attribute: str) -> np.ndarray:
    return np.asarray([getattr(item, attribute) for item in records], dtype=float).reshape(-1, 2)


def _r(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), _METRIC_DECIMALS)


# --- keyframes ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyframeCalibration:
    frame_index: int
    timestamp_seconds: float
    shot_id: str
    status: CalibrationStatus
    reason: Reason | None
    homography: hg.Matrix | None
    correspondences: tuple[CorrespondenceRecord, ...]
    metrics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    fit_hull_pixels: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if (self.status is CalibrationStatus.CALIBRATED) != (self.homography is not None):
            raise ValueError("only a calibrated keyframe carries a homography")
        if status_for(self.reason) is not self.status:
            raise ValueError("keyframe status and reason disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "shot_id": self.shot_id,
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "method": CalibrationMethod.MANUAL_LANDMARKS.value,
            "homography_image_to_pitch": list(self.homography) if self.homography else None,
            "correspondences": [item.to_dict() for item in self.correspondences],
            "metrics": dict(self.metrics),
            "fit_hull_pixels": [list(item) for item in self.fit_hull_pixels],
        }


def _keyframe(
    frame: AnnotatedFrame, records: Sequence[CorrespondenceRecord], reason: Reason | None,
    matrix: hg.Matrix | None = None, metrics: Mapping[str, Any] | None = None,
    hull: Sequence[Sequence[float]] = (),
) -> KeyframeCalibration:
    status = status_for(reason)
    return KeyframeCalibration(
        frame.frame_index, frame.timestamp_seconds, frame.shot_id, status, reason,
        matrix if status is CalibrationStatus.CALIBRATED else None,
        tuple(records), MappingProxyType(dict(metrics or {})),
        tuple((float(x), float(y)) for x, y in hull) if status is CalibrationStatus.CALIBRATED else (),
    )


def _counts(records: Sequence[CorrespondenceRecord]) -> dict[str, int]:
    return {
        "fit_points": sum(item.usage is Usage.FIT for item in records),
        "validation_points": sum(item.usage is Usage.VALIDATION for item in records),
        "excluded_points": sum(item.usage is Usage.EXCLUDED for item in records),
    }


def calibrate_keyframe(
    frame: AnnotatedFrame,
    shot: ShotSegment,
    pitch: PitchGeometry,
    frame_size: tuple[int, int],
    policy: CalibrationPolicy,
) -> KeyframeCalibration:
    records = _classify(frame, policy)
    counts = _counts(records)
    fit = [item for item in records if item.usage is Usage.FIT]
    validation = [item for item in records if item.usage is Usage.VALIDATION]
    if len(fit) < policy.min_fit_points:
        return _keyframe(frame, records, Reason.INSUFFICIENT_FIT_POINTS, metrics=counts)
    if len(validation) < policy.min_validation_points:
        return _keyframe(frame, records, Reason.INSUFFICIENT_VALIDATION_POINTS, metrics=counts)
    fit_pixels, fit_pitch = _array(fit, "pixel_xy"), _array(fit, "pitch_xy_m")
    if not hg.has_general_position_subset(fit_pixels, fit_pitch, policy.min_triangle_ratio):
        return _keyframe(frame, records, Reason.DEGENERATE_GEOMETRY, metrics=counts)
    result = hg.ransac(
        fit_pixels, fit_pitch,
        threshold_m=policy.ransac_threshold_m,
        max_subsets=policy.ransac_max_subsets,
        seed=policy.ransac_seed + frame.frame_index,
        min_triangle_ratio=policy.min_triangle_ratio,
        max_refinements=policy.ransac_max_refinements,
    )
    if result is None:
        return _keyframe(frame, records, Reason.DEGENERATE_GEOMETRY, metrics=counts)
    inliers = int(result.inliers.sum())
    ransac_metrics = {
        **counts,
        "fit_inliers": inliers,
        "fit_outliers": len(fit) - inliers,
        "ransac_subsets_evaluated": result.subsets_evaluated,
        "ransac_subsets_degenerate": result.subsets_degenerate,
        "ransac_exhaustive": result.exhaustive,
        "dlt_singular_ratio": float(f"{result.singular_ratio:.6e}"),
    }
    if inliers < policy.min_fit_inliers or inliers / len(fit) < policy.min_inlier_ratio:
        return _keyframe(frame, records, Reason.INSUFFICIENT_INLIERS, metrics=ransac_metrics)
    if result.singular_ratio < policy.min_dlt_singular_ratio:
        return _keyframe(frame, records, Reason.ILL_CONDITIONED, metrics=ransac_metrics)
    matrix = hg.round_matrix(result.matrix)
    evaluated, metrics, reason, hull = evaluate_model(
        hg.as_matrix(matrix), records, result.inliers, shot, pitch, frame_size, policy,
    )
    return _keyframe(frame, evaluated, reason, matrix, {**ransac_metrics, **metrics}, hull)


def evaluate_model(
    matrix: np.ndarray,
    records: Sequence[CorrespondenceRecord],
    fit_inliers: np.ndarray,
    shot: ShotSegment,
    pitch: PitchGeometry,
    frame_size: tuple[int, int],
    policy: CalibrationPolicy,
) -> tuple[list[CorrespondenceRecord], dict[str, Any], Reason | None, list[tuple[float, float]]]:
    """Independent checks of one estimated homography. Never refits anything."""
    fit_index = [i for i, item in enumerate(records) if item.usage is Usage.FIT]
    validation_index = [i for i, item in enumerate(records) if item.usage is Usage.VALIDATION]
    pixels, pitch_points = _array(records, "pixel_xy"), _array(records, "pitch_xy_m")
    errors = hg.errors_m(matrix, pixels, pitch_points) if len(records) else np.zeros(0)
    updated = list(records)
    inlier_of = dict(zip(fit_index, (bool(value) for value in fit_inliers)))
    for i, item in enumerate(records):
        if item.usage is Usage.EXCLUDED:
            continue
        updated[i] = CorrespondenceRecord(
            item.landmark_id, item.pixel_xy, item.pitch_xy_m, item.role, item.quality, item.review_status,
            item.annotator, item.usage, item.exclusion, inlier_of.get(i) if item.usage is Usage.FIT else None,
            _r(errors[i]) if math.isfinite(errors[i]) else None,
        )
    inlier_index = [i for i in fit_index if inlier_of[i]]
    checked = inlier_index + validation_index
    checks: dict[str, str] = {}
    metrics: dict[str, Any] = {}

    finite = bool(np.all(np.isfinite(matrix))) and abs(float(np.linalg.det(matrix))) > 1e-18
    checks["finite_invertible"] = "pass" if finite else "fail"
    _, w = hg.apply(matrix, pixels[checked]) if finite else (None, np.zeros(0))
    in_front = finite and bool(np.all(w > 0))
    checks["landmarks_in_front_of_horizon"] = "pass" if in_front else "fail"
    jacobian = hg.jacobian_determinant(matrix, pixels[checked]) if in_front else np.zeros(0)
    oriented = in_front and bool(np.all(jacobian > 0))
    checks["orientation_preserved"] = "pass" if oriented else "fail"
    if oriented:
        scale = np.sqrt(jacobian)
        metrics["metres_per_pixel_min"] = _r(scale.min())
        metrics["metres_per_pixel_max"] = _r(scale.max())
        plausible = bool(scale.min() >= policy.min_metres_per_pixel and scale.max() <= policy.max_metres_per_pixel)
        checks["plausible_scale"] = "pass" if plausible else "fail"
    else:
        checks["plausible_scale"] = "not_checked"

    if in_front and shot.camera_side is not CameraSide.UNKNOWN:
        centroid = pixels[inlier_index].mean(axis=0)
        dx, dy = hg.derivative_along_image_down(matrix, centroid)
        expected = {
            CameraSide.BOTTOM_TOUCHLINE: dy > 0,
            CameraSide.TOP_TOUCHLINE: dy < 0,
            CameraSide.LEFT_GOAL_LINE: dx < 0,
            CameraSide.RIGHT_GOAL_LINE: dx > 0,
        }[shot.camera_side]
        checks["camera_side_consistent"] = "pass" if expected else "fail"
    else:
        checks["camera_side_consistent"] = "not_checked"

    hull = hg.convex_hull(pixels[inlier_index])
    width, height = frame_size
    metrics["image_hull_fraction"] = _r(hg.polygon_area(hull) / (width * height))
    pitch_hull = hg.convex_hull(pitch_points[inlier_index])
    metrics["pitch_hull_area_m2"] = _r(hg.polygon_area(pitch_hull))
    metrics["pitch_hull_fraction"] = _r(hg.polygon_area(pitch_hull) / (pitch.length_m * pitch.width_m))
    checks["spatial_coverage"] = "pass" if metrics["image_hull_fraction"] >= policy.min_image_hull_fraction else "fail"

    fit_errors = errors[inlier_index]
    metrics["fit_rmse_m"] = _r(math.sqrt(float(np.mean(fit_errors ** 2))))
    metrics["fit_max_error_m"] = _r(float(fit_errors.max()))
    metrics["fit_degrees_of_freedom"] = 2 * len(inlier_index) - 8
    held_out = errors[validation_index]
    finite_held_out = bool(np.all(np.isfinite(held_out)))
    metrics["validation_median_error_m"] = _r(float(np.median(held_out))) if finite_held_out else None
    metrics["validation_max_error_m"] = _r(float(held_out.max())) if finite_held_out else None
    metrics["validation_p90_error_m"] = (
        _r(float(np.quantile(held_out, 0.9, method="higher"))) if finite_held_out and len(held_out) >= 10 else None
    )
    metrics["validation_points_outside_fit_hull"] = sum(
        not hg.inside_convex_polygon(hull, pixels[i]) for i in validation_index
    )
    checks["validation_error"] = "pass" if (
        finite_held_out
        and metrics["validation_median_error_m"] <= policy.max_validation_median_error_m
        and metrics["validation_max_error_m"] <= policy.max_validation_error_m
    ) else "fail"
    metrics["checks"] = checks

    order = (
        ("finite_invertible", Reason.NON_FINITE_OR_SINGULAR),
        ("landmarks_in_front_of_horizon", Reason.HORIZON_CROSSING),
        ("orientation_preserved", Reason.REFLECTED_ORIENTATION),
        ("camera_side_consistent", Reason.CAMERA_SIDE_INCONSISTENT),
        ("plausible_scale", Reason.IMPLAUSIBLE_SCALE),
        ("spatial_coverage", Reason.INSUFFICIENT_COVERAGE),
        ("validation_error", Reason.VALIDATION_ERROR_EXCEEDED),
    )
    reason = next((value for name, value in order if checks[name] == "fail"), None)
    return updated, metrics, reason, [(float(x), float(y)) for x, y in hull]


# --- temporal policy ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PropagationInterval:
    shot_id: str
    start_keyframe: int
    end_keyframe: int
    status: CalibrationStatus
    reason: Reason | None
    max_cross_error_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "start_keyframe": self.start_keyframe,
            "end_keyframe": self.end_keyframe,
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "max_cross_error_m": self.max_cross_error_m,
        }


@dataclass(frozen=True, slots=True)
class FrameCalibration:
    frame_index: int
    timestamp_seconds: float
    shot_id: str | None
    status: CalibrationStatus
    method: CalibrationMethod
    reason: Reason | None
    source_keyframe: int | None
    homography: hg.Matrix | None

    def __post_init__(self) -> None:
        calibrated = self.status is CalibrationStatus.CALIBRATED
        if calibrated != (self.homography is not None) or calibrated != (self.source_keyframe is not None):
            raise ValueError("only a calibrated frame carries a homography and its source keyframe")
        if calibrated == (self.method is CalibrationMethod.NONE):
            raise ValueError("calibrated frames name their method; other frames use method 'none'")
        if status_for(self.reason) is not self.status:
            raise ValueError("frame status and reason disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "shot_id": self.shot_id,
            "status": self.status.value,
            "method": self.method.value,
            "reason": self.reason.value if self.reason else None,
            "source_keyframe": self.source_keyframe,
            "homography_image_to_pitch": list(self.homography) if self.homography else None,
        }


def _used(keyframe: KeyframeCalibration) -> list[CorrespondenceRecord]:
    return [
        item for item in keyframe.correspondences
        if item.usage is Usage.VALIDATION or (item.usage is Usage.FIT and item.ransac_inlier)
    ]


def cross_error_m(a: KeyframeCalibration, b: KeyframeCalibration) -> float:
    """Largest error when each keyframe's homography reprojects the other's landmarks."""
    worst = 0.0
    for model, evidence in ((a, b), (b, a)):
        used = _used(evidence)
        errors = hg.errors_m(hg.as_matrix(model.homography), _array(used, "pixel_xy"), _array(used, "pitch_xy_m"))
        worst = max(worst, float(errors.max()) if errors.size else math.inf)
    return worst


def propagation_intervals(
    shots: Sequence[ShotSegment],
    keyframes: Sequence[KeyframeCalibration],
    suspected_cuts: Sequence[int],
    policy: CalibrationPolicy,
) -> tuple[PropagationInterval, ...]:
    intervals = []
    cuts = sorted(set(suspected_cuts))
    for shot in shots:
        if shot.camera_motion is not CameraMotion.STATIC or not policy.propagate_static_shots:
            continue
        in_shot = sorted((item for item in keyframes if item.shot_id == shot.shot_id), key=lambda item: item.frame_index)
        for a, b in zip(in_shot, in_shot[1:]):
            if a.status is not CalibrationStatus.CALIBRATED or b.status is not CalibrationStatus.CALIBRATED:
                intervals.append(PropagationInterval(
                    shot.shot_id, a.frame_index, b.frame_index, CalibrationStatus.UNCALIBRATED,
                    Reason.NO_VERIFIED_KEYFRAME_PAIR, None,
                ))
                continue
            error = cross_error_m(a, b)
            if any(a.frame_index < cut <= b.frame_index for cut in cuts):
                reason: Reason | None = Reason.SUSPECTED_CAMERA_CUT
            elif not error <= policy.max_validation_error_m:
                reason = Reason.KEYFRAMES_DISAGREE
            else:
                reason = None
            intervals.append(PropagationInterval(
                shot.shot_id, a.frame_index, b.frame_index, status_for(reason), reason,
                _r(error) if math.isfinite(error) else None,
            ))
    return tuple(intervals)


def _none(frame_index: int, timestamp: float, shot_id: str | None, reason: Reason) -> FrameCalibration:
    return FrameCalibration(frame_index, timestamp, shot_id, status_for(reason), CalibrationMethod.NONE, reason, None, None)


def frame_states(
    frames: Iterable[tuple[int, float]],
    shots: Sequence[ShotSegment],
    keyframes: Sequence[KeyframeCalibration],
    intervals: Sequence[PropagationInterval],
    policy: CalibrationPolicy,
) -> tuple[FrameCalibration, ...]:
    by_frame = {item.frame_index: item for item in keyframes}
    shot_starts = [item.start_frame for item in shots]  # shots are sorted and disjoint
    by_shot: dict[str, list[PropagationInterval]] = {}
    for item in intervals:
        by_shot.setdefault(item.shot_id, []).append(item)
    interval_starts = {key: [item.start_keyframe for item in value] for key, value in by_shot.items()}
    states = []
    for frame_index, timestamp in frames:
        position = bisect.bisect_right(shot_starts, frame_index) - 1
        shot = shots[position] if position >= 0 and shots[position].contains(frame_index) else None
        if shot is None:
            states.append(_none(frame_index, timestamp, None, Reason.NO_SHOT_SEGMENT))
            continue
        keyframe = by_frame.get(frame_index)
        if keyframe is not None:
            if keyframe.status is CalibrationStatus.CALIBRATED:
                states.append(FrameCalibration(
                    frame_index, timestamp, shot.shot_id, CalibrationStatus.CALIBRATED,
                    CalibrationMethod.MANUAL_LANDMARKS, None, frame_index, keyframe.homography,
                ))
            else:
                states.append(_none(frame_index, timestamp, shot.shot_id, keyframe.reason))  # type: ignore[arg-type]
            continue
        if shot.camera_motion is not CameraMotion.STATIC:
            states.append(_none(frame_index, timestamp, shot.shot_id, Reason.CAMERA_NOT_DECLARED_STATIC))
            continue
        if not policy.propagate_static_shots:
            states.append(_none(frame_index, timestamp, shot.shot_id, Reason.NO_KEYFRAME_FOR_FRAME))
            continue
        candidates = by_shot.get(shot.shot_id, [])
        slot = bisect.bisect_left(interval_starts.get(shot.shot_id, []), frame_index) - 1
        interval = candidates[slot] if slot >= 0 and frame_index < candidates[slot].end_keyframe else None
        if interval is None:
            states.append(_none(frame_index, timestamp, shot.shot_id, Reason.OUTSIDE_VERIFIED_INTERVAL))
        elif interval.status is not CalibrationStatus.CALIBRATED:
            states.append(_none(frame_index, timestamp, shot.shot_id, interval.reason))  # type: ignore[arg-type]
        else:
            nearest = (
                interval.start_keyframe
                if frame_index - interval.start_keyframe <= interval.end_keyframe - frame_index
                else interval.end_keyframe
            )
            states.append(FrameCalibration(
                frame_index, timestamp, shot.shot_id, CalibrationStatus.CALIBRATED,
                CalibrationMethod.STATIC_SHOT_PROPAGATION, None, nearest, by_frame[nearest].homography,
            ))
    return tuple(states)
