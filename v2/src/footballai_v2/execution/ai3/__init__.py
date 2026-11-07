"""AI3: pitch calibration, homography and spatial foundation.

Pixel positions become pitch coordinates in metres only when a frame has an
evidence-backed, independently validated homography. Every analysed frame is
``calibrated``, ``uncalibrated`` or ``invalidated``; the last two never carry
coordinates and never fall back to a fixed pixel→metre scale.

Contracts:

- ``footballai.pitch-landmark-annotations/v1`` (manual input)
- ``footballai.pitch-calibration/v1`` (per-frame states with their evidence)
- ``footballai.projected-player-observation/v1`` (AI2C observations on the pitch)
- ``footballai.pitch-calibration-benchmark/v1`` (synthetic offline benchmark)
"""

from footballai_v2.execution.ai3.annotations import PITCH_ANNOTATIONS_SCHEMA, PitchAnnotations
from footballai_v2.execution.ai3.artifact import (
    PITCH_CALIBRATION_SCHEMA,
    PitchCalibrationArtifact,
    calibrate,
    pitch_calibration_from_json_bytes,
    pitch_calibration_sha256,
    pitch_calibration_to_json_bytes,
)
from footballai_v2.execution.ai3.calibration import CalibrationPolicy, CalibrationStatus
from footballai_v2.execution.ai3.geometry import IFAB_LAW_1_MARKINGS, PitchGeometry
from footballai_v2.execution.ai3.projection import (
    PROJECTED_OBSERVATION_CONTRACT,
    ProjectedPlayerObservation,
    ProjectionStatus,
    project_player_observations,
)

__all__ = [
    "IFAB_LAW_1_MARKINGS",
    "PITCH_ANNOTATIONS_SCHEMA",
    "PITCH_CALIBRATION_SCHEMA",
    "PROJECTED_OBSERVATION_CONTRACT",
    "CalibrationPolicy",
    "CalibrationStatus",
    "PitchAnnotations",
    "PitchCalibrationArtifact",
    "PitchGeometry",
    "ProjectedPlayerObservation",
    "ProjectionStatus",
    "calibrate",
    "pitch_calibration_from_json_bytes",
    "pitch_calibration_sha256",
    "pitch_calibration_to_json_bytes",
    "project_player_observations",
]
