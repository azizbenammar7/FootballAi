"""Worker glue for the AI2C ``identity_resolution`` and AI3 ``pitch_calibration`` stages.

The V1-compatible adapter produces the AI1 artifacts (``detections`` and
``tracklets``) during the detection stage. These two stages then turn those
exact bytes into versioned spatial artifacts through
:mod:`footballai_v2.execution.ai3.pipeline_stage`. Every output is a
deterministic function of the AI1 bytes, the run's video checksum, its data
origin and the landmark annotations pinned in its parameters, so a redelivered
job republishes byte-identical artifacts (:meth:`SpatialStages.ensure`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

from footballai_v2.contracts.v1 import AnalysisRun, ArtifactCategory
from footballai_v2.execution.ai1.artifacts import DETECTIONS_SCHEMA, TRACKLETS_SCHEMA, sha256_bytes
from footballai_v2.execution.ai2c.artifact import PLAYER_IDENTITIES_SCHEMA
from footballai_v2.execution.ai3.artifact import PITCH_CALIBRATION_SCHEMA
from footballai_v2.execution.ai3.pipeline_stage import (
    PARQUET_MEDIA_TYPE,
    PITCH_ANNOTATIONS_PARAMETER,
    PITCH_SPACE_DIAGNOSTIC_SCHEMA,
    PROJECTED_OBSERVATIONS_SCHEMA,
    IdentityStageResult,
    SpatialStageError,
    diagnostic_to_json_bytes,
    load_pitch_annotations_parameter,
    run_identity_stage,
    run_pitch_calibration_stage,
    verify_ai1_artifacts,
)
from footballai_v2.execution.errors import ExecutionFailure
from footballai_v2.execution.generated_artifact import GeneratedArtifact


IDENTITIES_ARTIFACT_ID = "player-identities"
CALIBRATION_ARTIFACT_ID = "pitch-calibration"
OBSERVATIONS_ARTIFACT_ID = "projected-player-observations"
DIAGNOSTIC_ARTIFACT_ID = "pitch-space-diagnostic"
SPATIAL_ARTIFACT_IDS = (CALIBRATION_ARTIFACT_ID, OBSERVATIONS_ARTIFACT_ID, DIAGNOSTIC_ARTIFACT_ID)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    message: str
    output_count: int
    frames_processed: int


def _failure(error: SpatialStageError) -> ExecutionFailure:
    # Coded, deterministic refusals: retrying the same inputs cannot succeed.
    return ExecutionFailure(error.code, error.message, retryable=False)


class SpatialStages:
    """Identity (review-only) and pitch calibration over one run's AI1 payloads."""

    def __init__(self) -> None:
        self._identity: IdentityStageResult | None = None

    @staticmethod
    def _ai1(run: AnalysisRun, payloads: MutableMapping[str, Any]):
        detections, tracklets = payloads.get("detections"), payloads.get("tracklets")
        if not (
            isinstance(detections, GeneratedArtifact) and detections.schema_version == DETECTIONS_SCHEMA
            and isinstance(tracklets, GeneratedArtifact) and tracklets.schema_version == TRACKLETS_SCHEMA
        ):
            raise ExecutionFailure(
                "ai1_artifacts_missing", "Verified AI1 tracking artifacts are unavailable for spatial stages.",
                retryable=False,
            )
        diagnostics = payloads.get("analysis-diagnostics")
        tracking = diagnostics.get("tracking", {}) if isinstance(diagnostics, dict) else {}
        recorded = tracking.get("tracklets_sha256") if isinstance(tracking, dict) else None
        try:
            ai1 = verify_ai1_artifacts(
                detections.content, tracklets.content, video_sha256=run.input.sha256,
                recorded_tracklets_sha256=recorded if isinstance(recorded, str) else None,
            )
        except SpatialStageError as exc:
            raise _failure(exc) from exc
        return ai1, tracklets.content

    def _run_identity(self, run: AnalysisRun, payloads: MutableMapping[str, Any]) -> IdentityStageResult:
        ai1, tracklets_bytes = self._ai1(run, payloads)
        try:
            result = run_identity_stage(ai1, tracklets_bytes, source_id=f"run-{run.run_id}")
        except SpatialStageError as exc:
            raise _failure(exc) from exc
        self._identity = result
        payloads[IDENTITIES_ARTIFACT_ID] = GeneratedArtifact(
            IDENTITIES_ARTIFACT_ID, "Player identities (review-only)", ArtifactCategory.TRACKS,
            f"artifacts/{IDENTITIES_ARTIFACT_ID}.json", result.content, "application/json", PLAYER_IDENTITIES_SCHEMA,
        )
        return result

    def identity(self, run: AnalysisRun, payloads: MutableMapping[str, Any]) -> StageOutcome:
        result = self._run_identity(run, payloads)
        counts = result.counts()
        return StageOutcome(
            message=(
                "Review-only identity association: no automatic links were made; "
                f"{counts['unresolved']} person tracklets keep an UNKNOWN identity and "
                f"{counts['excluded']} non-person tracklets are excluded."
            ),
            output_count=len(result.artifact.assignments),
            frames_processed=0,
        )

    def pitch(self, run: AnalysisRun, payloads: MutableMapping[str, Any]) -> StageOutcome:
        ai1, tracklets_bytes = self._ai1(run, payloads)
        identity = self._identity
        published = payloads.get(IDENTITIES_ARTIFACT_ID)
        if identity is None or not isinstance(published, GeneratedArtifact) or sha256_bytes(published.content) != identity.sha256:
            identity = self._run_identity(run, payloads)
        try:
            raw = run.parameters.get(PITCH_ANNOTATIONS_PARAMETER)
            annotations = load_pitch_annotations_parameter(raw) if raw is not None else None
            result = run_pitch_calibration_stage(
                ai1, identity, run_id=run.run_id, data_origin=run.data_origin.value, annotations=annotations,
            )
        except SpatialStageError as exc:
            raise _failure(exc) from exc
        for artifact_id in SPATIAL_ARTIFACT_IDS:
            payloads.pop(artifact_id, None)
        if result.calibration_bytes is not None:
            payloads[CALIBRATION_ARTIFACT_ID] = GeneratedArtifact(
                CALIBRATION_ARTIFACT_ID, "Pitch calibration", ArtifactCategory.OTHER,
                f"artifacts/{CALIBRATION_ARTIFACT_ID}.json", result.calibration_bytes, "application/json",
                PITCH_CALIBRATION_SCHEMA,
            )
        if result.observations_bytes is not None:
            payloads[OBSERVATIONS_ARTIFACT_ID] = GeneratedArtifact(
                OBSERVATIONS_ARTIFACT_ID, "Projected tracklet observations", ArtifactCategory.TRACKS,
                f"artifacts/{OBSERVATIONS_ARTIFACT_ID}.parquet", result.observations_bytes, PARQUET_MEDIA_TYPE,
                PROJECTED_OBSERVATIONS_SCHEMA,
            )
        payloads[DIAGNOSTIC_ARTIFACT_ID] = GeneratedArtifact(
            DIAGNOSTIC_ARTIFACT_ID, "Pitch-space diagnostic (tracklet level)", ArtifactCategory.VISUALIZATION,
            f"artifacts/{DIAGNOSTIC_ARTIFACT_ID}.json", diagnostic_to_json_bytes(result.diagnostic),
            "application/json", PITCH_SPACE_DIAGNOSTIC_SCHEMA,
        )
        frames = result.diagnostic["frames"]
        if result.calibration is None:
            return StageOutcome(
                    message=(
                    f"No pitch landmark annotations attached: all {frames['total']} analysed frames are UNCALIBRATED "
                    "and no pitch coordinates were produced."
                ),
                output_count=0,
                frames_processed=frames["total"],
            )
        by_status = frames["by_status"]
        return StageOutcome(
            message=(
                f"Pitch calibration from manual landmarks: {by_status['calibrated']} calibrated, "
                f"{by_status['uncalibrated']} uncalibrated, {by_status['invalidated']} invalidated of "
                f"{frames['total']} analysed frames; {result.projected_observations} tracklet foot points projected."
            ),
            output_count=result.projected_observations,
            frames_processed=frames["total"],
        )

    def ensure(self, run: AnalysisRun, payloads: MutableMapping[str, Any]) -> None:
        """Recompute spatial artifacts when payloads were rebuilt after a redelivery."""
        if IDENTITIES_ARTIFACT_ID not in payloads:
            self.identity(run, payloads)
        if DIAGNOSTIC_ARTIFACT_ID not in payloads:
            self.pitch(run, payloads)
