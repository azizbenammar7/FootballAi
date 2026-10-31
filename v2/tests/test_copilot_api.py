"""End-to-end API tests for the explanation endpoint and its feature gate."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from footballai_v2.api import create_app
from footballai_v2.copilot import CopilotSettings, DeterministicFakeProvider
from footballai_v2.execution.coordinator import AnalysisCoordinator, ExecutionSettings
from footballai_v2.execution.executor import AnalysisExecutor


@pytest.fixture
def settings(tmp_path, monkeypatch):
    execution = ExecutionSettings(tmp_path / "runs", tmp_path / "queue", max_upload_bytes=1024 * 1024, allow_test_profiles=True)
    monkeypatch.setattr(
        AnalysisCoordinator,
        "probe_video",
        lambda self, path, extension: {"duration_seconds": 60.0, "container": "mp4", "size_bytes": path.stat().st_size},
    )
    return execution


def enabled_client(settings: ExecutionSettings) -> TestClient:
    return TestClient(
        create_app(
            settings.run_root,
            settings=settings,
            copilot_settings=CopilotSettings(enabled=True, environment="test"),
            copilot_provider=DeterministicFakeProvider(),
        )
    )


def upload(client: TestClient, *, profile: str = "demo_fast") -> str:
    response = client.post(
        "/api/v1/analyses",
        files={"video": ("match.mp4", b"bounded-video-bytes", "video/mp4")},
        data={"match_name": "Tunis Derby", "pipeline_profile": profile, "data_origin": "evaluation"},
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def execute(client: TestClient, settings: ExecutionSettings) -> None:
    job = AnalysisCoordinator(settings).queue.claim("test-worker")
    assert job is not None
    status = AnalysisExecutor(client.app.state.run_store, client.app.state.object_storage, stage_delay_seconds=0).execute(job, "test-worker")
    if status.value in {"succeeded", "partial"}:
        AnalysisCoordinator(settings).queue.complete(job)
    else:
        AnalysisCoordinator(settings).queue.fail(job)


def test_explain_a_completed_synthetic_run(settings):
    client = enabled_client(settings)
    run_id = upload(client)
    execute(client, settings)
    response = client.post(f"/api/v1/runs/{run_id}/explain", json={"question": None})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"] == run_id
    assert body["support_level"] == "supported"
    assert body["model_mode"] == "offline_demo_fake"
    assert "synthetic" in body["answer"].lower()
    assert body["citations"], "a grounded answer should cite evidence"
    assert all(citation["run_id"] == run_id for citation in body["citations"])


def test_explain_unsupported_question_abstains(settings):
    client = enabled_client(settings)
    run_id = upload(client)
    execute(client, settings)
    response = client.post(f"/api/v1/runs/{run_id}/explain", json={"question": "How many passes were completed?"})
    assert response.status_code == 200
    body = response.json()
    assert body["support_level"] == "unavailable"
    assert body["citations"] == []
    assert body["unavailable_reason"]


def test_explain_rejects_non_terminal_run(settings):
    client = enabled_client(settings)
    run_id = upload(client)  # queued, not executed
    response = client.post(f"/api/v1/runs/{run_id}/explain", json={})
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "run_not_eligible"


def test_explain_unknown_run_is_404(settings):
    client = enabled_client(settings)
    response = client.post(
        "/api/v1/runs/00000000-0000-4000-8000-000000000000/explain", json={}
    )
    assert response.status_code == 404


def test_endpoint_absent_when_gate_disabled(settings, monkeypatch):
    monkeypatch.delenv("FOOTBALLAI_COPILOT_ENABLED", raising=False)
    monkeypatch.delenv("FOOTBALLAI_ENVIRONMENT", raising=False)
    client = TestClient(create_app(settings.run_root, settings=settings))  # gate from env -> disabled
    run_id = upload(client)
    execute(client, settings)
    response = client.post(f"/api/v1/runs/{run_id}/explain", json={})
    assert response.status_code == 404  # route is not registered at all
    # Existing endpoints keep working with the gate off.
    assert client.get("/api/health").status_code == 200
    assert client.get(f"/api/v1/runs/{run_id}").status_code == 200
    assert client.get(f"/api/v1/runs/{run_id}/artifacts").status_code == 200
