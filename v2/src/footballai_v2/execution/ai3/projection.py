"""``footballai.projected-player-observation/v1``: AI2C observations placed on the pitch.

This is a separate contract layered on ``footballai.player-observation/v1``.
The AI2C record is embedded unchanged, so its guard still holds: an AI2C
``PlayerObservation`` never carries pitch coordinates itself. Pitch
coordinates live only here, and only when:

- the calibration artifact's file checksum matches what the caller loaded,
- the calibration was made for the same AI1 tracklets artifact,
- the calibration was gated by exactly the policy the caller expects (the
  default policy unless one is passed), so a relaxed policy cannot slip in,
- the frame index exists in the calibration with the same timestamp,
- that exact frame is ``CALIBRATED`` (a keyframe, or verified propagation),
- the foot point lies inside the source frame and in front of the horizon,
- the projected point is finite and within the pitch plus a margin.

Otherwise ``pitch_xy_m`` is ``None`` and ``projection_status`` says why.

``ProjectedPlayerObservation`` holds the (self-verifying) calibration
artifact it came from and re-checks, on construction, that the checksum is
that artifact's, that its frame is calibrated with exactly the embedded
homography, and that ``pitch_xy_m`` is the projection of the foot pixel. A
caller cannot construct one with arbitrary coordinates without also
constructing a calibration artifact, which re-derives every homography from
landmark evidence that must pass validation. Identity status (``confirmed``/``provisional``/
``unresolved``) passes through untouched; projection never upgrades it.
Synthetic calibrations project only when explicitly allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator

import numpy as np

from footballai_v2.execution.ai2c.observations import PlayerObservation
from footballai_v2.execution.ai3 import homography as hg
from footballai_v2.execution.ai3.annotations import AnnotationOrigin
from footballai_v2.execution.ai3.artifact import PitchCalibrationArtifact, pitch_calibration_sha256
from footballai_v2.execution.ai3.calibration import CalibrationMethod, CalibrationPolicy, CalibrationStatus


PROJECTED_OBSERVATION_CONTRACT = "footballai.projected-player-observation/v1"
DEFAULT_PITCH_MARGIN_M = 5.0
"""Players may stand a few metres outside the lines (throw-ins, run-off)."""
_TIMESTAMP_TOLERANCE_S = 1e-6
_HORIZON_EPSILON = 1e-12


class ProjectionStatus(str, Enum):
    PROJECTED = "projected"
    FRAME_NOT_IN_CALIBRATION = "frame_not_in_calibration"
    FRAME_UNCALIBRATED = "frame_uncalibrated"
    FRAME_INVALIDATED = "frame_invalidated"
    FOOT_POINT_OUTSIDE_IMAGE = "foot_point_outside_image"
    BEYOND_HORIZON = "foot_point_beyond_camera_horizon"
    NON_FINITE = "non_finite_projection"
    OUTSIDE_PITCH_MARGIN = "outside_pitch_margin"


def _project(matrix: tuple[float, ...], pixel: tuple[float, float]) -> tuple[tuple[float, float] | None, float]:
    points, w = hg.apply(hg.as_matrix(matrix), np.asarray([pixel], dtype=float))
    x, y = float(points[0, 0]), float(points[0, 1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None, float(w[0])
    return (x, y), float(w[0])


@dataclass(frozen=True, slots=True)
class ProjectedPlayerObservation:
    observation: PlayerObservation
    calibration: PitchCalibrationArtifact = field(repr=False, compare=False)
    projection_status: ProjectionStatus
    pitch_xy_m: tuple[float, float] | None
    inside_pitch_lines: bool | None
    extrapolated_beyond_landmarks: bool | None
    calibration_status: CalibrationStatus | None
    calibration_method: CalibrationMethod | None
    calibration_reason: str | None
    calibration_keyframe: int | None
    homography: tuple[float, ...] | None
    source_calibration_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, PlayerObservation):
            raise ValueError("observation must be an AI2C PlayerObservation")
        if not isinstance(self.calibration, PitchCalibrationArtifact):
            raise ValueError("calibration must be a PitchCalibrationArtifact")
        if self.calibration.sha256() != self.source_calibration_sha256:
            raise ValueError("source_calibration_sha256 is not the calibration artifact's checksum")
        projected = self.projection_status is ProjectionStatus.PROJECTED
        if projected:
            if self.calibration_status is not CalibrationStatus.CALIBRATED or self.homography is None:
                raise ValueError("pitch coordinates require a calibrated frame and its homography")
            state = self.calibration.frame(self.observation.frame_index)
            if state is None or state.status is not CalibrationStatus.CALIBRATED or state.homography != self.homography:
                raise ValueError("the homography is not the calibration's homography for this frame")
            expected, _ = _project(self.homography, self.observation.foot_pixel_xy)
            if expected is None or self.pitch_xy_m != expected:
                raise ValueError("pitch_xy_m must be the homography projection of the foot point")
            if self.inside_pitch_lines is None or self.extrapolated_beyond_landmarks is None:
                raise ValueError("projected observations report pitch-line and extrapolation flags")
        elif self.pitch_xy_m is not None or self.homography is not None:
            raise ValueError("only a projected observation carries pitch coordinates")

    @property
    def contract(self) -> str:
        return PROJECTED_OBSERVATION_CONTRACT


def project_player_observations(
    observations: Iterable[PlayerObservation],
    calibration: PitchCalibrationArtifact,
    calibration_sha256: str,
    *,
    pitch_margin_m: float = DEFAULT_PITCH_MARGIN_M,
    expected_policy: CalibrationPolicy | None = None,
    allow_synthetic: bool = False,
) -> Iterator[ProjectedPlayerObservation]:
    """Stream projected observations; one output per input, in input order."""
    if pitch_calibration_sha256(calibration) != calibration_sha256:
        raise ValueError("calibration_sha256 does not match the calibration artifact")
    if calibration.policy != (expected_policy or CalibrationPolicy()):
        raise ValueError("calibration was gated by a different policy than the caller expects")
    if calibration.source.tracklets_sha256 is None:
        raise ValueError("calibration is not bound to an AI1 tracklets artifact; refusing to project")
    if calibration.source.origin is AnnotationOrigin.SYNTHETIC and not allow_synthetic:
        raise ValueError("calibration comes from synthetic annotations; pass allow_synthetic=True only in tests/benchmarks")
    margin = float(pitch_margin_m)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("pitch_margin_m must be a non-negative number")
    width, height = calibration.source.frame_size
    hulls = {
        item.frame_index: np.asarray(item.fit_hull_pixels, dtype=float).reshape(-1, 2)
        for item in calibration.keyframes if item.fit_hull_pixels
    }
    for observation in observations:
        if observation.source_tracklets_sha256 != calibration.source.tracklets_sha256:
            raise ValueError("observation and calibration come from different AI1 tracklets artifacts")
        state = calibration.frame(observation.frame_index)

        def result(status: ProjectionStatus, **kwargs) -> ProjectedPlayerObservation:
            return ProjectedPlayerObservation(
                observation=observation,
                calibration=calibration,
                projection_status=status,
                pitch_xy_m=kwargs.get("pitch_xy_m"),
                inside_pitch_lines=kwargs.get("inside"),
                extrapolated_beyond_landmarks=kwargs.get("extrapolated"),
                calibration_status=state.status if state else None,
                calibration_method=state.method if state else None,
                calibration_reason=state.reason.value if state and state.reason else None,
                calibration_keyframe=state.source_keyframe if state else None,
                homography=kwargs.get("homography"),
                source_calibration_sha256=calibration_sha256,
            )

        if state is None:
            yield result(ProjectionStatus.FRAME_NOT_IN_CALIBRATION)
            continue
        if abs(state.timestamp_seconds - observation.timestamp_seconds) > _TIMESTAMP_TOLERANCE_S:
            raise ValueError(
                f"frame {observation.frame_index}: calibration timestamp {state.timestamp_seconds} and observation "
                f"timestamp {observation.timestamp_seconds} differ; frame conventions do not match"
            )
        if state.status is CalibrationStatus.UNCALIBRATED:
            yield result(ProjectionStatus.FRAME_UNCALIBRATED)
            continue
        if state.status is CalibrationStatus.INVALIDATED:
            yield result(ProjectionStatus.FRAME_INVALIDATED)
            continue
        x_px, y_px = observation.foot_pixel_xy
        if not (0 <= x_px <= width and 0 <= y_px <= height):
            yield result(ProjectionStatus.FOOT_POINT_OUTSIDE_IMAGE)
            continue
        point, w = _project(state.homography, observation.foot_pixel_xy)  # type: ignore[arg-type]
        if not math.isfinite(w) or w <= _HORIZON_EPSILON:
            yield result(ProjectionStatus.BEYOND_HORIZON)
            continue
        if point is None:
            yield result(ProjectionStatus.NON_FINITE)
            continue
        if not calibration.pitch.contains(*point, margin_m=margin):
            yield result(ProjectionStatus.OUTSIDE_PITCH_MARGIN)
            continue
        hull = hulls.get(state.source_keyframe)  # type: ignore[arg-type]
        yield result(
            ProjectionStatus.PROJECTED,
            pitch_xy_m=point,
            inside=calibration.pitch.contains(*point),
            extrapolated=hull is None or not hg.inside_convex_polygon(hull, observation.foot_pixel_xy),
            homography=state.homography,
        )
