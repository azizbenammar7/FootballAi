"""Contract, lifecycle, attempt-chain, and stage-consistency tests."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from footballai_v2.contracts.v1 import (
    ANALYSIS_RUN_CONTRACT_VERSION,
    AnalysisRun,
    AnalysisRunStatus,
    ArtifactCategory,
    ArtifactReference,
    CodeReference,
    ContractValidationError,
    DataOrigin,
    FailureDetail,
    InputReference,
    InvalidStatusTransition,
    ModelReference,
    StageError,
    StageExecution,
    StageName,
    StageStatus,
)


CREATED = datetime(2026, 7, 26, 10, tzinfo=timezone.utc)
LOGICAL_ID = "11111111-1111-4111-8111-111111111111"
RUN_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"
RUN_ID_2 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"
SHA_A = "a" * 64
SHA_B = "b" * 64


def new_run(**overrides) -> AnalysisRun:
    values = {
        "logical_analysis_id": LOGICAL_ID,
        "run_id": RUN_ID,
        "data_origin": DataOrigin.SYNTHETIC,
        "input": InputReference("fixture://synthetic/match.mp4", SHA_A, "video/mp4"),
        "code": CodeReference(
            repository="https://github.com/example/FootballAi",
            revision="84b5457e0058b64fb2cbf31ea795c168debdcae5",
        ),
        "pipeline_version": "2.0.0",
        "parameters": {"effective_fps": 5, "nested": {"enabled": True}},
        "models": [ModelReference("detector-adapter", "benchmark-placeholder", SHA_B)],
        "created_at": CREATED,
    }
    values.update(overrides)
    return AnalysisRun.new(**values)


def artifact(
    artifact_id: str = "team-summary",
    *,
    category: ArtifactCategory = ArtifactCategory.SUMMARY,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        name="Team summary",
        category=category,
        relative_path=f"artifacts/{artifact_id}.json",
        media_type="application/json",
        sha256=SHA_B,
        size_bytes=42,
        schema_version="footballai.team-summary/v1",
    )


def error(at: datetime | None = None) -> StageError:
    return StageError(
        error_code="tracking_failed",
        safe_message="Tracking ended before completion.",
        retryable=True,
        occurred_at=at or CREATED + timedelta(seconds=2),
        technical_details={"frames_processed": 10},
    )


def stage(
    status: StageStatus,
    *,
    stage_id: str = "metrics-1",
    stage_name: StageName = StageName.METRICS,
    required: bool = True,
    artifact_ids: tuple[str, ...] = (),
    attempt_number: int = 1,
    progress: float | None = None,
) -> StageExecution:
    started = CREATED + timedelta(seconds=1)
    finished = CREATED + timedelta(seconds=2)
    values = {
        StageStatus.QUEUED: (0, None, None, None),
        StageStatus.RUNNING: (50, started, None, None),
        StageStatus.SUCCEEDED: (100, started, finished, None),
        StageStatus.PARTIAL: (60, started, finished, None),
        StageStatus.FAILED: (40, started, finished, error(finished)),
        StageStatus.CANCELLED: (30, started, finished, None),
        StageStatus.SKIPPED: (0, None, finished, None),
    }
    default_progress, started_at, finished_at, stage_error = values[status]
    return StageExecution(
        stage_id=stage_id,
        stage_name=stage_name,
        required=required,
        status=status,
        progress_percent=default_progress if progress is None else progress,
        attempt_number=attempt_number,
        started_at=started_at,
        finished_at=finished_at,
        produced_artifact_ids=artifact_ids,
        error=stage_error,
        performance_metrics={"duration_seconds": 1.0, "frames_processed": 5},
        message="Safe stage update.",
    )


def running_run() -> AnalysisRun:
    return new_run().start(CREATED + timedelta(seconds=1), stages=[stage(StageStatus.RUNNING)])


def succeeded_run() -> AnalysisRun:
    item = artifact()
    return running_run().succeed(
        [item],
        CREATED + timedelta(seconds=3),
        stages=[stage(StageStatus.SUCCEEDED, artifact_ids=(item.artifact_id,))],
    )


def partial_run() -> AnalysisRun:
    item = artifact()
    return running_run().complete_partial(
        [item],
        "Reviewable output exists, but mandatory work is incomplete.",
        CREATED + timedelta(seconds=3),
        stages=[stage(StageStatus.PARTIAL, artifact_ids=(item.artifact_id,))],
    )


def failed_run() -> AnalysisRun:
    failed_stage = stage(StageStatus.FAILED)
    return running_run().fail(
        FailureDetail(
            "attempt_failed",
            "The attempt ended safely.",
            True,
            CREATED + timedelta(seconds=3),
        ),
        CREATED + timedelta(seconds=3),
        stages=[failed_stage],
    )


def test_contract_has_stable_first_public_identifier():
    assert ANALYSIS_RUN_CONTRACT_VERSION == "footballai.analysis-run/v1"
    assert new_run().contract_version == ANALYSIS_RUN_CONTRACT_VERSION


def test_generated_logical_and_run_ids_are_uuid_v4():
    generated = AnalysisRun.new(
        data_origin=DataOrigin.SYNTHETIC,
        input=InputReference("fixture://input", SHA_A, "video/mp4"),
        code=CodeReference("https://github.com/example/FootballAi", "8" * 40),
        pipeline_version="2.0.0",
    )
    assert uuid.UUID(generated.logical_analysis_id).version == 4
    assert uuid.UUID(generated.run_id).version == 4


def test_run_states_are_exact_and_partial_is_terminal():
    assert [item.value for item in AnalysisRunStatus] == [
        "queued",
        "running",
        "succeeded",
        "partial",
        "failed",
        "cancelled",
    ]
    assert AnalysisRunStatus.PARTIAL.is_terminal


def test_partial_serializes_and_round_trips():
    completed = partial_run()
    restored = AnalysisRun.from_dict(json.loads(json.dumps(completed.to_dict())))
    assert restored == completed
    assert restored.status is AnalysisRunStatus.PARTIAL
    assert restored.partial_reason.startswith("Reviewable output")


def test_running_to_partial_is_allowed_but_partial_to_running_is_rejected():
    completed = partial_run()
    with pytest.raises(InvalidStatusTransition, match="partial"):
        completed.start()


def test_unknown_run_status_is_rejected():
    payload = new_run().to_dict()
    payload["status"] = "retrying"
    with pytest.raises(ContractValidationError, match="invalid enum"):
        AnalysisRun.from_dict(payload)


@pytest.mark.parametrize("origin", list(DataOrigin))
def test_all_four_data_origins_are_accepted(origin):
    assert new_run(data_origin=origin).data_origin is origin


def test_unknown_data_origin_is_rejected():
    payload = new_run().to_dict()
    payload["data_origin"] = "downloaded"
    with pytest.raises(ContractValidationError, match="invalid enum"):
        AnalysisRun.from_dict(payload)


def test_valid_first_attempt_relationship_fields():
    run = new_run()
    assert run.attempt_number == 1
    assert run.previous_attempt_run_id is None


@pytest.mark.parametrize(
    ("attempt_number", "previous_run_id", "message"),
    [(1, RUN_ID_2, "first attempt"), (2, None, "retry attempt")],
)
def test_invalid_attempt_relationship_rules(attempt_number, previous_run_id, message):
    payload = new_run().to_dict()
    payload["attempt_number"] = attempt_number
    payload["previous_attempt_run_id"] = previous_run_id
    with pytest.raises(ContractValidationError, match=message):
        AnalysisRun.from_dict(payload)


def test_previous_attempt_cannot_equal_current_run():
    payload = new_run().to_dict()
    payload["attempt_number"] = 2
    payload["previous_attempt_run_id"] = payload["run_id"]
    with pytest.raises(ContractValidationError, match="cannot equal"):
        AnalysisRun.from_dict(payload)


@pytest.mark.parametrize("previous_factory", [failed_run, partial_run])
def test_retry_preserves_logical_identity_and_increments_attempt(previous_factory):
    previous = previous_factory()
    retry = AnalysisRun.retry_from(previous, run_id=RUN_ID_2)
    assert retry.logical_analysis_id == previous.logical_analysis_id
    assert retry.input == previous.input
    assert retry.data_origin == previous.data_origin
    assert retry.attempt_number == 2
    assert retry.previous_attempt_run_id == previous.run_id
    assert retry.run_id != previous.run_id
    assert retry.status is AnalysisRunStatus.QUEUED


def test_retry_may_change_attempt_specific_provenance_visibly():
    previous = failed_run()
    retry = AnalysisRun.retry_from(
        previous,
        run_id=RUN_ID_2,
        code=CodeReference("https://github.com/example/FootballAi", "9" * 40),
        pipeline_version="2.0.1",
        parameters={"tracking_threshold": 0.4},
        models=[ModelReference("detector-adapter", "candidate-b")],
    )
    assert retry.code != previous.code
    assert retry.pipeline_version == "2.0.1"
    assert retry.parameters != previous.parameters
    assert retry.models != previous.models


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(new_run(), id="queued"),
        pytest.param(running_run(), id="running"),
        pytest.param(succeeded_run(), id="succeeded"),
        pytest.param(
            new_run().cancel(CREATED + timedelta(seconds=1), reason="Cancelled by operator."),
            id="cancelled",
        ),
    ],
)
def test_retry_rejects_unapproved_states(run):
    with pytest.raises(InvalidStatusTransition, match="cannot retry"):
        AnalysisRun.retry_from(run)


@pytest.mark.parametrize("status", list(StageStatus))
def test_every_stage_status_has_a_valid_record(status):
    record = stage(status)
    restored = StageExecution.from_dict(record.to_dict())
    assert restored == record


@pytest.mark.parametrize("progress", [-0.01, 100.01, float("inf"), float("nan")])
def test_stage_progress_must_be_finite_and_bounded(progress):
    with pytest.raises(ContractValidationError, match="between 0 and 100"):
        replace(stage(StageStatus.RUNNING), progress_percent=progress)


def test_running_stage_requires_start_timestamp():
    with pytest.raises(ContractValidationError, match="requires started_at"):
        replace(stage(StageStatus.RUNNING), started_at=None)


@pytest.mark.parametrize(
    "status",
    [
        StageStatus.SUCCEEDED,
        StageStatus.PARTIAL,
        StageStatus.FAILED,
        StageStatus.CANCELLED,
        StageStatus.SKIPPED,
    ],
)
def test_terminal_stage_requires_finish_timestamp(status):
    with pytest.raises(ContractValidationError, match="requires finished_at"):
        replace(stage(status), finished_at=None)


def test_stage_finish_cannot_precede_start():
    with pytest.raises(ContractValidationError, match="cannot precede"):
        replace(
            stage(StageStatus.SUCCEEDED),
            finished_at=CREATED,
        )


def test_failed_stage_requires_structured_error():
    with pytest.raises(ContractValidationError, match="requires structured error"):
        replace(stage(StageStatus.FAILED), error=None)


def test_non_finite_performance_metric_is_rejected():
    with pytest.raises(ContractValidationError, match="non-finite"):
        replace(stage(StageStatus.RUNNING), performance_metrics={"processing_fps": float("nan")})


def test_stage_attempt_number_must_be_at_least_one():
    with pytest.raises(ContractValidationError, match="at least 1"):
        replace(stage(StageStatus.RUNNING), attempt_number=0)


def test_stage_messages_and_error_details_reject_secrets_and_tracebacks():
    with pytest.raises(ContractValidationError, match="unsafe"):
        replace(stage(StageStatus.RUNNING), message="Authorization: Bearer private")
    with pytest.raises(ContractValidationError, match="secret-like"):
        StageError(
            "safe_error",
            "Safe public message.",
            False,
            CREATED,
            {"api_key": "not-persisted"},
        )


def test_duplicate_stage_ids_are_rejected():
    first = stage(StageStatus.QUEUED, stage_id="stage-1", stage_name=StageName.INGESTION)
    second = stage(StageStatus.QUEUED, stage_id="stage-1", stage_name=StageName.METRICS)
    with pytest.raises(ContractValidationError, match="stage IDs"):
        new_run(stages=[first, second])


def test_duplicate_stage_names_are_rejected():
    first = stage(StageStatus.QUEUED, stage_id="stage-1", stage_name=StageName.INGESTION)
    second = stage(StageStatus.QUEUED, stage_id="stage-2", stage_name=StageName.INGESTION)
    with pytest.raises(ContractValidationError, match="stage names"):
        new_run(stages=[first, second])


def test_stage_artifact_reference_must_exist_in_manifest():
    bad_stage = stage(StageStatus.QUEUED, artifact_ids=("missing-artifact",))
    with pytest.raises(ContractValidationError, match="unknown artifact"):
        new_run(stages=[bad_stage])


def test_stage_attempt_number_must_match_manifest_attempt():
    with pytest.raises(ContractValidationError, match="must match"):
        new_run(stages=[stage(StageStatus.QUEUED, attempt_number=2)])


def test_succeeded_run_with_running_stage_is_rejected():
    item = artifact()
    with pytest.raises(ContractValidationError, match="active stages"):
        replace(
            succeeded_run(),
            stages=[stage(StageStatus.RUNNING)],
        )


def test_partial_run_with_running_stage_is_rejected():
    item = artifact()
    with pytest.raises(ContractValidationError, match="terminal stage records"):
        replace(
            partial_run(),
            stages=[stage(StageStatus.RUNNING)],
            artifacts=[item],
        )


def test_partial_run_without_useful_artifact_is_rejected():
    with pytest.raises(ContractValidationError, match="useful artifact"):
        replace(
            partial_run(),
            artifacts=(),
            stages=[stage(StageStatus.PARTIAL)],
        )


def test_partial_run_without_reason_is_rejected():
    with pytest.raises(ContractValidationError, match="safe reason"):
        replace(partial_run(), partial_reason=None)


def test_partial_run_requires_incomplete_mandatory_stage():
    item = artifact()
    with pytest.raises(ContractValidationError, match="incomplete mandatory"):
        replace(
            partial_run(),
            stages=[
                stage(
                    StageStatus.PARTIAL,
                    required=False,
                    artifact_ids=(item.artifact_id,),
                )
            ],
        )


def test_failed_run_without_structured_error_is_rejected():
    with pytest.raises(ContractValidationError, match="completion/error"):
        replace(failed_run(), failure=None)


def test_failed_and_cancelled_runs_cannot_contain_running_stage():
    with pytest.raises(ContractValidationError, match="running stage"):
        replace(failed_run(), stages=[stage(StageStatus.RUNNING)])
    cancelled = running_run().cancel(
        CREATED + timedelta(seconds=2),
        reason="Cancelled by operator.",
        stages=[stage(StageStatus.CANCELLED)],
    )
    with pytest.raises(ContractValidationError, match="running stage"):
        replace(cancelled, stages=[stage(StageStatus.RUNNING)])


def test_terminal_attempts_reject_all_lifecycle_mutations():
    for run in (succeeded_run(), partial_run(), failed_run()):
        with pytest.raises(InvalidStatusTransition):
            run.with_stages([])


@pytest.mark.parametrize(
    "relative_path",
    [
        "/artifacts/result.json",
        "../result.json",
        "artifacts/../result.json",
        "other/result.json",
        "artifacts\\result.json",
        "artifacts",
        "artifacts//result.json",
        "artifacts/./result.json",
    ],
)
def test_artifact_paths_cannot_escape_or_bypass_run_namespace(relative_path):
    with pytest.raises(ContractValidationError):
        ArtifactReference(
            "bad",
            "Bad artifact",
            ArtifactCategory.OTHER,
            relative_path,
            "application/json",
            SHA_A,
            1,
        )


def test_workload_advisory_is_a_stable_artifact_category_and_stage_name():
    assert ArtifactCategory.WORKLOAD_ADVISORY.value == "workload_advisory"
    assert StageName.WORKLOAD_ADVISORY.value == "workload_advisory"


def test_contract_rejects_unknown_fields_and_non_json_parameters():
    payload = new_run().to_dict()
    payload["surprise"] = True
    with pytest.raises(ContractValidationError, match="unknown fields"):
        AnalysisRun.from_dict(payload)
    with pytest.raises(ContractValidationError, match="non-finite"):
        new_run(parameters={"threshold": float("nan")})
    with pytest.raises(ContractValidationError, match="unsupported type"):
        new_run(parameters={"thresholds": (1, 2)})
