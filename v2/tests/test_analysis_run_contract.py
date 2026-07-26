"""Contract and lifecycle tests for footballai.analysis-run/v1."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from footballai_v2.contracts.v1 import (
    ANALYSIS_RUN_CONTRACT_VERSION,
    AnalysisRun,
    AnalysisRunStatus,
    ArtifactReference,
    CodeReference,
    ContractValidationError,
    DataOrigin,
    FailureDetail,
    InputReference,
    InvalidStatusTransition,
    ModelReference,
)


CREATED = datetime(2026, 7, 26, 10, tzinfo=timezone.utc)
RUN_ID = "018f47d2-5c88-7a1f-8b57-42f89c6a13a5"
SHA_A = "a" * 64
SHA_B = "b" * 64


def new_run(**overrides):
    values = {
        "analysis_run_id": RUN_ID,
        "data_origin": DataOrigin.SYNTHETIC,
        "input": InputReference("file:///fixtures/match.mp4", SHA_A, "video/mp4"),
        "code": CodeReference(
            repository="https://github.com/example/FootballAi",
            revision="84b5457e0058b64fb2cbf31ea795c168debdcae5",
        ),
        "pipeline_version": "2.0.0",
        "parameters": {"effective_fps": 5, "nested": {"enabled": True}},
        "models": [ModelReference("player-detector", "yolov8m", SHA_B)],
        "created_at": CREATED,
    }
    values.update(overrides)
    return AnalysisRun.new(**values)


def output() -> ArtifactReference:
    return ArtifactReference(
        name="summary",
        relative_path="artifacts/player_summary.json",
        media_type="application/json",
        sha256=SHA_B,
        size_bytes=42,
        schema_version="footballai.player-summary/v1",
    )


def test_contract_has_a_stable_explicit_version():
    assert ANALYSIS_RUN_CONTRACT_VERSION == "footballai.analysis-run/v1"
    assert new_run().contract_version == ANALYSIS_RUN_CONTRACT_VERSION


def test_successful_lifecycle_round_trips_without_information_loss():
    queued = new_run()
    running = queued.start(CREATED + timedelta(seconds=1))
    succeeded = running.succeed([output()], CREATED + timedelta(minutes=2))

    restored = AnalysisRun.from_dict(json.loads(json.dumps(succeeded.to_dict())))

    assert restored == succeeded
    assert restored.status is AnalysisRunStatus.SUCCEEDED
    assert restored.outputs[0].schema_version == "footballai.player-summary/v1"
    assert restored.to_dict()["completed_at"].endswith("Z")


def test_real_and_synthetic_origins_are_not_inferred_from_a_path():
    assert new_run(data_origin=DataOrigin.REAL).data_origin is DataOrigin.REAL
    assert new_run(data_origin=DataOrigin.SYNTHETIC).data_origin is DataOrigin.SYNTHETIC


def test_failed_run_has_structured_failure_and_no_outputs():
    failed = (
        new_run()
        .start(CREATED + timedelta(seconds=1))
        .fail(FailureDetail("TRACKING_FAILED", "worker stopped", True), CREATED + timedelta(seconds=2))
    )

    assert failed.status is AnalysisRunStatus.FAILED
    assert failed.failure.retryable is True
    assert failed.outputs == ()


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
def test_artifact_paths_cannot_escape_or_bypass_the_run_namespace(relative_path):
    with pytest.raises(ContractValidationError):
        ArtifactReference("bad", relative_path, "application/json", SHA_A, 1)


def test_contract_rejects_unknown_fields_instead_of_silently_ignoring_them():
    payload = new_run().to_dict()
    payload["surprise"] = True
    with pytest.raises(ContractValidationError, match="unknown fields"):
        AnalysisRun.from_dict(payload)


def test_contract_rejects_non_finite_parameter_values():
    with pytest.raises(ContractValidationError, match="non-finite"):
        new_run(parameters={"threshold": float("nan")})


def test_contract_rejects_non_json_parameter_container_types():
    with pytest.raises(ContractValidationError, match="unsupported type"):
        new_run(parameters={"thresholds": (1, 2)})


def test_invalid_terminal_transition_is_rejected():
    succeeded = (
        new_run()
        .start(CREATED + timedelta(seconds=1))
        .succeed([output()], CREATED + timedelta(seconds=2))
    )
    with pytest.raises(InvalidStatusTransition):
        succeeded.fail(FailureDetail("LATE_FAILURE", "too late", False))


def test_committed_example_is_accepted_by_the_python_contract():
    example = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "analysis-run"
        / "examples"
        / "succeeded.json"
    )
    restored = AnalysisRun.from_dict(json.loads(example.read_text()))
    assert restored.status is AnalysisRunStatus.SUCCEEDED
    assert restored.data_origin is DataOrigin.REAL
