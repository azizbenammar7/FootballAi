"""Stable weighted progress derivation."""

from footballai_v2.contracts.v1 import AnalysisRun, StageName, StageStatus


STAGE_WEIGHTS = {
    StageName.INGESTION: 5,
    StageName.VIDEO_VALIDATION: 5,
    StageName.DETECTION: 35,
    StageName.TRACKING: 20,
    StageName.IDENTITY_RESOLUTION: 5,
    StageName.PITCH_CALIBRATION: 5,
    StageName.METRICS: 10,
    StageName.WORKLOAD_ADVISORY: 5,
    StageName.ARTIFACT_PUBLICATION: 10,
}


def overall_progress(run: AnalysisRun) -> float:
    by_name = {stage.stage_name: stage for stage in run.stages}
    value = sum(STAGE_WEIGHTS[name] * (1 if stage.status is StageStatus.SKIPPED else float(stage.progress_percent) / 100) for name, stage in by_name.items())
    if run.status.value != "succeeded":
        value = min(value, 99.0)
    return round(value, 1)


def active_stage(run: AnalysisRun) -> str | None:
    for stage in run.stages:
        if stage.status is StageStatus.RUNNING:
            return stage.stage_name.value
    return None
