"""M2-C: the agent endpoint, its gate, and the trace a client may see.

Nothing here calls a language model. Runs come from the real local store, or from
the versioned fixtures served through the real application, and hostile replies go
through the real parser, loop, registry, grounding, and HTTP layer.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from footballai_v2.api import create_app
from footballai_v2.copilot import CopilotSettings, DeterministicFakeProvider
from footballai_v2.copilot.agent import AgentLimits, ReadOnlyAgent, ToolName, ToolRegistry
from footballai_v2.copilot.agent.trace import (
    AgentAdapter,
    AgentStopReason,
    AgentTrace,
    ToolCallTrace,
    UsageSource,
    publish_trace,
)
from footballai_v2.copilot.agent.tools import ToolCallStatus, ToolErrorCode
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evidence import RunEvidenceService
from footballai_v2.copilot.evaluation import agent_models
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, Step, call, say, select
from footballai_v2.copilot.evaluation.dataset import load_dataset
from footballai_v2.copilot.evaluation.fixtures import build_fixture_run
from footballai_v2.copilot.grounding import build_claim_catalog, eligible_claims, route_question
from footballai_v2.copilot.models import ModelMode
from footballai_v2.copilot.provider import AnthropicAgentModel
from footballai_v2.execution.coordinator import AnalysisCoordinator, ExecutionSettings
from footballai_v2.execution.executor import AnalysisExecutor


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = load_dataset(REPO_ROOT / "evaluation" / "copilot" / "v2")
GATE = CopilotSettings(enabled=True, agent_enabled=True, environment="test")
LOOPBACK = {"client": ("127.0.0.1", 50000), "base_url": "http://127.0.0.1:8000"}
UNKNOWN_RUN = "00000000-0000-4000-8000-000000000000"
DIAGNOSTICS = "What do the tracking diagnostics tell us?"
MODEL = "What model was used in the analysis?"
MISSING = "What evidence is missing?"
PASSES = "How many successful passes did player 7 make?"
CANARY = "canary-value-not-real"
STATUS_TOOL = ToolName.GET_RUN_STATUS.value
TRACKING_TOOL = ToolName.GET_TRACKING_DIAGNOSTICS.value
MODEL_TOOL = ToolName.GET_MODEL_PROVENANCE.value
LIMITS_TOOL = ToolName.GET_EVIDENCE_LIMITATIONS.value


def app_for(execution: ExecutionSettings, *, gate: CopilotSettings | None = GATE, factory=None):
    return create_app(
        execution.run_root,
        settings=execution,
        copilot_settings=gate,
        copilot_provider=DeterministicFakeProvider(),
        copilot_agent_model_factory=factory,
    )


@pytest.fixture
def execution(tmp_path, monkeypatch):
    settings = ExecutionSettings(
        tmp_path / "runs", tmp_path / "queue", max_upload_bytes=1024 * 1024, allow_test_profiles=True
    )
    monkeypatch.setattr(
        AnalysisCoordinator,
        "probe_video",
        lambda self, path, extension: {"duration_seconds": 60.0, "container": "mp4", "size_bytes": path.stat().st_size},
    )
    return settings


@pytest.fixture
def fixture_client(execution, monkeypatch):
    """The real application serving one versioned fixture run."""

    def build(fixture="verified_real", factory=None):
        repository, storage, run_id = build_fixture_run(DATASET.fixtures.fixtures[fixture], DATASET.fixtures)
        coordinator = SimpleNamespace(repository=repository, object_storage=storage, queue=None)
        monkeypatch.setattr("footballai_v2.api.app.AnalysisCoordinator", lambda _settings: coordinator)
        return TestClient(app_for(execution, factory=factory), **LOOPBACK), run_id

    return build


def upload(client: TestClient, *, profile: str = "demo_fast") -> str:
    response = client.post(
        "/api/v1/analyses",
        files={"video": ("match.mp4", b"bounded-video-bytes", "video/mp4")},
        data={"match_name": "Tunis Derby", "pipeline_profile": profile, "data_origin": "evaluation"},
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def execute(client: TestClient, execution: ExecutionSettings) -> None:
    job = AnalysisCoordinator(execution).queue.claim("test-worker")
    assert job is not None
    AnalysisExecutor(client.app.state.run_store, client.app.state.object_storage, stage_delay_seconds=0).execute(
        job, "test-worker"
    )
    AnalysisCoordinator(execution).queue.complete(job)


def completed_run(client: TestClient, execution: ExecutionSettings) -> str:
    run_id = upload(client)
    execute(client, execution)
    return run_id


def ask(client: TestClient, run_id: str, question: str | None = None):
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def scripted(*steps: Step):
    return lambda: ScriptedAgentModel(steps)


# -- the four demo questions, on evidence of each kind -------------------------


def test_diagnostics_are_answered_from_a_really_executed_tool(fixture_client):
    client, run_id = fixture_client()
    body = ask(client, run_id, DIAGNOSTICS)
    explanation, trace = body["explanation"], body["trace"]
    assert explanation["support_level"] == "supported"
    assert explanation["output_nature"] == "verified_real"
    assert explanation["model_mode"] == "offline_demo_fake"
    assert explanation["citations"] and all(item["run_id"] == run_id for item in explanation["citations"])
    assert [item["tool"] for item in trace["tool_calls"]] == [TRACKING_TOOL]
    assert trace["tool_calls"][0]["status"] == "ok"
    assert trace["termination"] == "answered"
    assert "diag_tracking" in trace["claim_ids_rendered"]


def test_the_model_question_reads_provenance(fixture_client):
    client, run_id = fixture_client("real_with_model_reference")
    body = ask(client, run_id, MODEL)
    assert [item["tool"] for item in body["trace"]["tool_calls"]] == [MODEL_TOOL]
    assert body["trace"]["termination"] == "answered"
    assert "model_references" in body["trace"]["claim_ids_rendered"]


def test_a_complete_run_says_nothing_is_missing(fixture_client):
    client, run_id = fixture_client()
    body = ask(client, run_id, MISSING)
    assert body["explanation"]["unavailable_reason"] is None
    assert "recorded nothing missing" in body["explanation"]["answer"]
    assert "passes" in body["explanation"]["answer"]
    assert [item["tool"] for item in body["trace"]["tool_calls"]] == [LIMITS_TOOL]
    assert "no_missing_evidence" in body["trace"]["claim_ids_rendered"]


def test_a_run_with_a_gap_states_the_gap(fixture_client):
    client, run_id = fixture_client("missing_diagnostics")
    body = ask(client, run_id, MISSING)
    assert body["explanation"]["support_level"] == "partial"
    assert "No analysis-diagnostics artifact" in body["explanation"]["answer"]
    assert "recorded nothing missing" not in body["explanation"]["answer"]
    assert "missing_evidence" in body["trace"]["claim_ids_rendered"]


def test_the_passes_question_abstains_and_nothing_runs(fixture_client):
    built = []

    def factory():
        built.append(ScriptedAgentModel())
        return built[-1]

    client, run_id = fixture_client(factory=factory)
    body = ask(client, run_id, PASSES)
    explanation, trace = body["explanation"], body["trace"]
    assert explanation["support_level"] == "unavailable"
    assert explanation["unavailable_reason"] == "unsupported_question"
    assert explanation["model_mode"] == "not_invoked"
    assert explanation["citations"] == []
    assert trace["tool_calls"] == [] and trace["tool_calls_refused"] == 0
    assert trace["model_turns"] == 0
    assert trace["termination"] == "unsupported_question"
    assert built[0].requests == []


def test_a_real_store_run_is_answered_and_labelled_synthetic(execution):
    client = TestClient(app_for(execution), **LOOPBACK)
    run_id = completed_run(client, execution)
    body = ask(client, run_id, DIAGNOSTICS)
    assert body["explanation"]["output_nature"] == "synthetic"
    assert all(item["run_id"] == run_id for item in body["explanation"]["citations"])
    assert body["trace"]["run_id"] == run_id


def test_a_general_request_stays_within_the_tool_limit(fixture_client):
    client, run_id = fixture_client()
    body = ask(client, run_id)
    assert body["trace"]["termination"] == "answered"
    assert len(body["trace"]["tool_calls"]) <= AgentLimits().max_tool_calls


# -- the new claim stays narrow -------------------------------------------------


def _evidence(fixture: str):
    repository, storage, run_id = build_fixture_run(DATASET.fixtures.fixtures[fixture], DATASET.fixtures)
    return RunEvidenceService(repository, storage, CopilotSettings()).build(run_id)


def _catalog(fixture: str):
    return build_claim_catalog(_evidence(fixture))


def test_the_no_gap_claim_is_never_offered_with_a_gap():
    assert "no_missing_evidence" not in {claim.claim_id for claim in _catalog("missing_diagnostics")}
    assert "no_missing_evidence" not in {claim.claim_id for claim in _catalog("corrupt_diagnostics")}
    assert "no_missing_evidence" not in {claim.claim_id for claim in _catalog("bad_checksum")}


@pytest.mark.parametrize("fixture", ["legacy_tracking", "unknown_profile"])
def test_the_no_gap_claim_is_never_offered_on_an_unverified_run(fixture):
    assert "no_missing_evidence" not in {claim.claim_id for claim in _catalog(fixture)}


def test_the_no_gap_claim_is_never_offered_beside_a_failed_artifact():
    evidence = _evidence("verified_real")
    assert "no_missing_evidence" in {claim.claim_id for claim in build_claim_catalog(evidence)}
    failed = tuple(replace(item, integrity_state="invalid") for item in evidence.artifacts)
    object.__setattr__(evidence, "artifacts", failed)
    assert "no_missing_evidence" not in {claim.claim_id for claim in build_claim_catalog(evidence)}


def test_the_no_gap_claim_is_withheld_when_a_measure_was_refused():
    assert "no_missing_evidence" not in {claim.claim_id for claim in _catalog("malformed_short_tracklets")}


def test_the_no_gap_claim_is_withheld_beside_an_impossible_fraction():
    evidence = _evidence("verified_real")
    observed = {**evidence.diagnostics["observed_metrics"], "detection_assignment_fraction": 1.5}
    object.__setattr__(evidence, "diagnostics", {**evidence.diagnostics, "observed_metrics": observed})
    catalog = {claim.claim_id for claim in build_claim_catalog(evidence)}
    assert "diag_assignment" not in catalog and "no_missing_evidence" not in catalog


def test_a_schema_that_has_no_metrics_is_never_parsed_for_them():
    evidence = _evidence("synthetic_demo_fast")
    damaged = {**evidence.diagnostics, "observed_metrics": {"short_tracklets": "not a table"}}
    object.__setattr__(evidence, "diagnostics", damaged)
    assert "no_missing_evidence" in {claim.claim_id for claim in build_claim_catalog(evidence)}


def test_the_no_gap_claim_cites_the_artifact_it_describes():
    claim = next(claim for claim in _catalog("verified_real") if claim.claim_id == "no_missing_evidence")
    assert {citation.field_path for citation in claim.citations} == {"missing_evidence", "diagnostics.schema"}
    cited = next(citation for citation in claim.citations if citation.field_path == "diagnostics.schema")
    assert cited.artifact_id == "analysis-diagnostics"
    assert cited.source_sha256


@pytest.mark.parametrize("question", [None, "Explain this run.", "What are the limitations?"])
def test_the_no_gap_claim_does_not_join_other_routes(question):
    offered = {claim.claim_id for claim in eligible_claims(_catalog("verified_real"), route_question(question))}
    assert "no_missing_evidence" not in offered


def test_the_explanation_route_gets_the_same_honest_answer(execution):
    client = TestClient(app_for(execution), **LOOPBACK)
    run_id = completed_run(client, execution)
    response = client.post(f"/api/v1/runs/{run_id}/explain", json={"question": MISSING})
    assert response.status_code == 200
    assert "recorded nothing missing" in response.json()["answer"]


# -- the response contract ------------------------------------------------------


def test_the_response_holds_the_explanation_contract_and_a_public_trace(fixture_client):
    client, run_id = fixture_client()
    body = ask(client, run_id, DIAGNOSTICS)
    assert set(body) == {"explanation", "trace"}
    assert set(body["explanation"]) == {
        "run_id", "answer", "support_level", "output_nature", "model_mode", "citations", "limitations",
        "unavailable_reason",
    }
    assert set(body["trace"]) == {
        "schema_version", "correlation_id", "run_id", "adapter", "model", "model_mode", "instruction_version",
        "routing_version", "termination", "tool_error_code", "duration_ms", "model_turns", "model_turn_limit",
        "tool_call_limit", "tool_calls", "tool_calls_refused", "claim_ids_returned", "claim_ids_selected",
        "claim_ids_rendered", "selected_ids_rejected",
    }
    assert set(body["trace"]["tool_calls"][0]) == {
        "sequence", "turn", "tool", "status", "duration_ms", "claim_ids_returned", "error_code",
    }


def test_the_trace_names_the_adapter_and_never_claims_a_language_model(fixture_client):
    client, run_id = fixture_client()
    trace = ask(client, run_id, DIAGNOSTICS)["trace"]
    assert trace["adapter"] == "offline_scripted"
    assert trace["model"] == "scripted-stub"
    assert trace["model_mode"] == "offline_demo_fake"
    abstained = ask(client, run_id, PASSES)["trace"]
    assert abstained["adapter"] == "offline_scripted"  # the label survives an abstention
    assert abstained["model_mode"] == "not_invoked"


def test_the_route_applies_the_approved_limits(fixture_client, monkeypatch):
    applied: list[AgentLimits] = []
    original = ReadOnlyAgent.__init__

    def spy(self, *args, **options):
        applied.append(options["limits"])
        original(self, *args, **options)

    monkeypatch.setattr(ReadOnlyAgent, "__init__", spy)
    client, run_id = fixture_client()
    trace = ask(client, run_id, DIAGNOSTICS)["trace"]
    assert (trace["tool_call_limit"], trace["model_turn_limit"]) == (4, 5)
    assert applied == [AgentLimits(4, 5, 20.0, 45.0)]


def test_the_trace_correlates_with_the_response_header(fixture_client):
    client, run_id = fixture_client()
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": DIAGNOSTICS})
    assert response.json()["trace"]["correlation_id"] == response.headers["X-Request-ID"]


def test_the_schema_is_published_only_when_enabled(execution, fixture_client):
    client, _run_id = fixture_client()
    assert "/api/v1/runs/{run_id}/agent" in client.get("/openapi.json").json()["paths"]
    closed = CopilotSettings(enabled=True, environment="test")
    assert "/api/v1/runs/{run_id}/agent" not in TestClient(app_for(execution, gate=closed)).get("/openapi.json").json()["paths"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"question": "x" * 501}, 422),
        ({"question": 7}, 422),
        ({"run_id": UNKNOWN_RUN}, 422),
        ({"question": DIAGNOSTICS, "tools": ["sql"]}, 422),
    ],
)
def test_the_request_accepts_a_question_and_nothing_else(fixture_client, body, status):
    client, run_id = fixture_client()
    assert client.post(f"/api/v1/runs/{run_id}/agent", json=body).status_code == status


def test_a_run_that_has_not_finished_is_refused(execution):
    client = TestClient(app_for(execution), **LOOPBACK)
    run_id = upload(client)
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={})
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "run_not_eligible"


def test_an_unknown_run_is_404_with_its_own_message(execution):
    client = TestClient(app_for(execution), **LOOPBACK)
    response = client.post(f"/api/v1/runs/{UNKNOWN_RUN}/agent", json={})
    assert response.status_code == 404
    assert response.json()["detail"] == "Analysis run not found"


def test_an_invalid_run_id_is_rejected(execution):
    client = TestClient(app_for(execution), **LOOPBACK)
    assert client.post("/api/v1/runs/not-a-uuid/agent", json={}).status_code == 422


# -- the gate -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "copilot", "agent", "open_"),
    [
        ("local", "1", "1", True),
        ("test", "1", "1", True),
        ("LOCAL ", "1", "1", True),
        ("local", "1", None, False),
        ("local", "1", "0", False),
        ("local", "1", "true", False),
        ("local", None, "1", False),
        ("local", "0", "1", False),
        ("", "1", "1", False),
        (None, "1", "1", False),
        ("staging", "1", "1", False),
        ("production", "1", "1", False),
        ("dev", "1", "1", False),
    ],
)
def test_the_gate_needs_both_flags_in_a_local_environment(monkeypatch, environment, copilot, agent, open_):
    for name, value in (
        ("FOOTBALLAI_ENVIRONMENT", environment),
        ("FOOTBALLAI_COPILOT_ENABLED", copilot),
        ("FOOTBALLAI_COPILOT_AGENT_ENABLED", agent),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert CopilotSettings.from_environment().agent_enabled is open_


def test_the_agent_is_off_by_default():
    assert CopilotSettings().agent_enabled is False
    assert CopilotSettings(enabled=True, environment="local").agent_enabled is False


def test_settings_refuse_an_agent_outside_the_copilot_gate():
    with pytest.raises(ValueError):
        CopilotSettings(agent_enabled=True, environment="local")
    with pytest.raises(ValueError):
        CopilotSettings(enabled=True, agent_enabled=True, environment="staging")


def test_the_route_does_not_exist_with_the_copilot_on_and_the_agent_off(execution):
    client = TestClient(app_for(execution, gate=CopilotSettings(enabled=True, environment="test")), **LOOPBACK)
    run_id = completed_run(client, execution)
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert client.post(f"/api/v1/runs/{run_id}/explain", json={}).status_code == 200


def test_the_route_does_not_exist_with_everything_off(execution, monkeypatch):
    for name in ("FOOTBALLAI_COPILOT_ENABLED", "FOOTBALLAI_COPILOT_AGENT_ENABLED", "FOOTBALLAI_ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(app_for(execution, gate=None), **LOOPBACK)
    run_id = completed_run(client, execution)
    assert client.post(f"/api/v1/runs/{run_id}/agent", json={}).status_code == 404
    assert client.post(f"/api/v1/runs/{run_id}/explain", json={}).status_code == 404


@pytest.mark.parametrize(
    "options",
    [
        {"client": ("203.0.113.9", 4000), "base_url": "http://127.0.0.1:8000"},  # a remote peer
        {"client": ("192.168.1.20", 4000), "base_url": "http://127.0.0.1:8000"},  # the local network
        {"client": ("testclient", 4000), "base_url": "http://127.0.0.1:8000"},  # not an address
        {"client": ("127.0.0.1", 4000), "base_url": "http://demo.example.com"},  # a tunnel or rebound name
    ],
)
def test_only_a_loopback_caller_on_a_loopback_host_reaches_the_agent(fixture_client, execution, options):
    local, run_id = fixture_client()
    client = TestClient(local.app, **options)
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": DIAGNOSTICS})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}  # the same as a route that was never registered


@pytest.mark.parametrize("origin", ["https://attacker.example", "http://localhost:9999", "null"])
def test_another_page_in_a_local_browser_cannot_reach_the_agent(fixture_client, origin):
    client, run_id = fixture_client()
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={}, headers={"Origin": origin})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_the_dashboard_origin_and_a_local_tool_are_served(fixture_client):
    client, run_id = fixture_client()
    dashboard = client.post(f"/api/v1/runs/{run_id}/agent", json={}, headers={"Origin": "http://localhost:5173"})
    assert dashboard.status_code == 200
    assert client.post(f"/api/v1/runs/{run_id}/agent", json={}).status_code == 200


def test_a_refused_caller_learns_nothing_about_a_run(fixture_client):
    local, _run_id = fixture_client()
    client = TestClient(local.app, client=("203.0.113.9", 4000), base_url="http://127.0.0.1:8000")
    response = client.post(f"/api/v1/runs/{UNKNOWN_RUN}/agent", json={})
    assert response.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_both_local_names_are_served(fixture_client, host):
    local, run_id = fixture_client()
    client = TestClient(local.app, client=("127.0.0.1", 4000), base_url=f"http://{host}:8000")
    assert client.post(f"/api/v1/runs/{run_id}/agent", json={}).status_code == 200


def test_an_ipv6_loopback_caller_is_served(fixture_client):
    local, run_id = fixture_client()
    client = TestClient(local.app, client=("::1", 4000), headers={"Host": "[::1]:8000"})
    assert client.post(f"/api/v1/runs/{run_id}/agent", json={}).status_code == 200


# -- no paid path ---------------------------------------------------------------


def test_a_real_provider_setting_never_reaches_the_agent_route(execution, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(AnthropicAgentModel, "complete", lambda self, request: calls.append(request))
    for name, value in (
        ("FOOTBALLAI_ENVIRONMENT", "test"),
        ("FOOTBALLAI_COPILOT_ENABLED", "1"),
        ("FOOTBALLAI_COPILOT_AGENT_ENABLED", "1"),
        ("FOOTBALLAI_COPILOT_PROVIDER", "real"),
        ("FOOTBALLAI_COPILOT_REAL_PROVIDER_ENABLED", "1"),
        ("ANTHROPIC_API_KEY", CANARY),
    ):
        monkeypatch.setenv(name, value)
    gate = CopilotSettings.from_environment()
    assert gate.wants_real_provider
    client = TestClient(app_for(execution, gate=gate), **LOOPBACK)
    run_id = completed_run(client, execution)
    body = ask(client, run_id, DIAGNOSTICS)
    assert calls == []
    assert body["trace"]["adapter"] == "offline_scripted"
    assert body["trace"]["model"] == "scripted-stub"
    assert CANARY not in json.dumps(body)


def test_the_loaded_api_has_no_way_to_import_the_sdk():
    code = (
        "import sys, types\n"
        "from footballai_v2.api import create_app\n"
        "from footballai_v2.copilot import CopilotSettings\n"
        "import tempfile\n"
        "root = tempfile.mkdtemp()\n"
        "create_app(root, copilot_settings=CopilotSettings(enabled=True, agent_enabled=True, environment='test'))\n"
        "sys.exit(1 if 'anthropic' in sys.modules else 0)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "v2" / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_an_app_with_the_gate_closed_loads_neither_the_agent_nor_the_evaluation():
    code = (
        "import sys, tempfile\n"
        "from footballai_v2.api import create_app\n"
        "from footballai_v2.copilot import CopilotSettings\n"
        "create_app(tempfile.mkdtemp(), copilot_settings=CopilotSettings(enabled=True, environment='test'))\n"
        "loaded = [m for m in sys.modules if m.startswith(('footballai_v2.copilot.agent', 'footballai_v2.copilot.evaluation'))]\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "v2" / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_each_request_gets_its_own_adapter(execution, monkeypatch):
    """A shared stub would keep every prompt it was sent for the life of the process."""
    remembered: list[int] = []
    original = ScriptedAgentModel.complete

    def spy(self, request):
        reply = original(self, request)
        remembered.append(len(self.requests))
        return reply

    monkeypatch.setattr(ScriptedAgentModel, "complete", spy)
    client = TestClient(app_for(execution), **LOOPBACK)
    run_id = completed_run(client, execution)
    for _ in range(3):
        ask(client, run_id, DIAGNOSTICS)
    assert remembered == [1, 2] * 3  # two turns per question, and each question starts empty


def test_the_default_adapter_is_the_baselined_policy():
    model = agent_models.offline_agent_model()
    assert isinstance(model, ScriptedAgentModel)
    assert model.mode is ModelMode.OFFLINE_DEMO_FAKE
    assert model.model_name == "scripted-stub"
    assert model._policy is agent_models.route_following_policy


# -- run binding and hostile adapters -------------------------------------------


def test_a_run_cannot_be_reached_through_another_run(execution):
    client = TestClient(app_for(execution), **LOOPBACK)
    first = completed_run(client, execution)
    second = completed_run(client, execution)
    assert first != second
    body = ask(client, first, DIAGNOSTICS)
    assert second not in json.dumps(body)
    assert body["trace"]["run_id"] == first


def test_a_tool_call_naming_another_run_stops_the_request(fixture_client):
    client, run_id = fixture_client(factory=scripted(call(STATUS_TOOL, arguments={"run_id": UNKNOWN_RUN})))
    body = ask(client, run_id, DIAGNOSTICS)
    assert body["explanation"]["unavailable_reason"] == "cross_run_rejected"
    assert body["explanation"]["citations"] == []
    assert body["trace"]["termination"] == "cross_run_rejected"
    assert body["trace"]["tool_calls"] == []
    assert body["trace"]["tool_calls_refused"] == 1
    assert body["trace"]["tool_error_code"] == "cross_run_reference"
    assert UNKNOWN_RUN not in json.dumps(body)


def test_a_turn_over_the_tool_limit_runs_nothing(fixture_client):
    client, run_id = fixture_client(factory=scripted(call(*(name.value for name in ToolName))))
    body = ask(client, run_id, DIAGNOSTICS)
    assert body["explanation"]["unavailable_reason"] == "agent_limit_reached"
    assert body["trace"]["termination"] == "tool_budget_exhausted"
    assert body["trace"]["tool_calls"] == []
    assert body["trace"]["tool_calls_refused"] == 5


def test_an_adapter_timeout_is_a_typed_abstention(fixture_client):
    failure = ProviderTimeoutError("slow", telemetry=None)
    client, run_id = fixture_client(factory=scripted(Step(error=failure)))
    body = ask(client, run_id, DIAGNOSTICS)
    assert body["explanation"]["unavailable_reason"] == "provider_timeout"
    assert body["trace"]["termination"] == "model_timeout"


def test_an_unretrieved_selection_is_rejected_and_counted(fixture_client):
    client, run_id = fixture_client(factory=scripted(select("diag_tracking")))
    body = ask(client, run_id, DIAGNOSTICS)
    assert body["explanation"]["unavailable_reason"] == "forged_citation"
    assert body["explanation"]["citations"] == []
    assert body["trace"]["termination"] == "selection_rejected"
    assert body["trace"]["selected_ids_rejected"] == 1


def test_a_call_outside_the_registry_is_shown_as_rejected_without_its_name(fixture_client):
    client, run_id = fixture_client(factory=scripted(call(CANARY), select()))
    body = ask(client, run_id, DIAGNOSTICS)
    entry = body["trace"]["tool_calls"][0]
    assert (entry["tool"], entry["status"], entry["error_code"]) == (None, "rejected", "unknown_tool")
    assert CANARY not in json.dumps(body)


def test_model_prose_reaches_neither_the_answer_nor_the_trace(fixture_client):
    client, run_id = fixture_client(factory=scripted(call(TRACKING_TOOL), say(f"Player 7 made 12 passes {CANARY}")))
    body = ask(client, run_id, DIAGNOSTICS)
    assert body["explanation"]["unavailable_reason"] == "provider_unavailable"
    assert CANARY not in json.dumps(body)
    assert "12 passes" not in json.dumps(body)


def test_a_provider_failure_class_is_kept_from_the_client_and_kept_for_the_operator(fixture_client, caplog):
    failure = ProviderUnavailableError(f"bad key {CANARY}", kind=ProviderFailureKind.AUTHENTICATION)
    client, run_id = fixture_client(factory=scripted(Step(error=failure)))
    with caplog.at_level(logging.INFO, logger="footballai_v2.api"):
        response = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": DIAGNOSTICS})
    assert CANARY not in response.text
    assert "authentication" not in response.text
    body = response.json()
    assert body["trace"]["termination"] == "provider_error"
    assert body["trace"]["tool_error_code"] is None
    logged = next(record.getMessage() for record in caplog.records if "agent_trace" in record.getMessage())
    assert '"error_code":"authentication"' in logged
    assert CANARY not in caplog.text


def test_the_log_holds_no_question_or_answer(fixture_client, caplog):
    client, run_id = fixture_client()
    with caplog.at_level(logging.INFO, logger="footballai_v2.api"):
        body = ask(client, run_id, DIAGNOSTICS)
    assert DIAGNOSTICS not in caplog.text
    assert body["explanation"]["answer"] not in caplog.text


def test_stub_token_counts_and_provider_ids_are_not_published(fixture_client):
    """The stub reports constants. They are not measurements, so they are not shown."""
    client, run_id = fixture_client()
    response = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": DIAGNOSTICS})
    for hidden in ("input_tokens", "output_tokens", "usage", "req_scripted", "peak_prompt_chars", '"turns"'):
        assert hidden not in response.text


def test_the_trace_lists_exactly_what_the_registry_executed(fixture_client, monkeypatch):
    executed: list[tuple[str | None, str]] = []
    original = ToolRegistry.execute

    def spy(self, name, arguments):
        execution = original(self, name, arguments)
        executed.append((execution.tool.value if execution.tool else None, execution.status.value))
        return execution

    monkeypatch.setattr(ToolRegistry, "execute", spy)
    client, run_id = fixture_client(
        factory=scripted(call(TRACKING_TOOL, STATUS_TOOL), call(TRACKING_TOOL), select("diag_tracking"))
    )
    trace = ask(client, run_id, DIAGNOSTICS)["trace"]
    assert [(item["tool"], item["status"]) for item in trace["tool_calls"]] == executed
    assert ("get_tracking_diagnostics", "rejected") in executed  # the repeat is shown as refused, not as a result
    assert [item["sequence"] for item in trace["tool_calls"]] == [1, 2, 3]


# -- the published projection ---------------------------------------------------


def worst_case_trace() -> AgentTrace:
    long_id = "c" * 64
    claims = tuple(f"{long_id[:61]}{index:03d}" for index in range(32))
    return AgentTrace(
        correlation_id="c" * 64,
        run_id="r" * 36,
        model="m" * 64,
        model_mode=ModelMode.REAL_MODEL,
        instruction_version="i" * 64,
        routing_version="v" * 64,
        termination=AgentStopReason.PROVIDER_ERROR,
        error_code=ToolErrorCode.INVALID_TOOL_OUTPUT,
        duration_ms=45_000.123,
        model_turns=5,
        turns=(),
        tool_calls=tuple(
            ToolCallTrace(
                sequence=index,
                turn=index,
                tool=ToolName.GET_TRACKING_DIAGNOSTICS,
                status=ToolCallStatus.FAILED,
                duration_ms=20_000.123,
                claim_ids_returned=claims,
                error_code=ToolErrorCode.INVALID_TOOL_OUTPUT,
            )
            for index in range(1, 5)
        ),
        tool_calls_refused=4,
        claim_ids_returned=claims,
        claim_ids_selected=claims,
        claim_ids_rendered=claims,
        selected_ids_rejected=64,
        usage=dict(input_tokens=1),
        usage_complete=True,
        usage_source=UsageSource.UNAVAILABLE,
        estimated_cost_usd=0.01,
        cost_is_lower_bound=True,
        valid_tool_calls=4,
        invalid_tool_calls=0,
        model_call_ms=100_000.0,
        tool_call_ms=1_000.0,
        peak_prompt_chars=24_576,
    )


def test_the_published_trace_is_bounded():
    published = publish_trace(worst_case_trace(), AgentAdapter.REAL_MODEL, AgentLimits())
    assert len(published.model_dump_json().encode()) < 17 * 1024  # with every id at its schema maximum


def test_provider_failure_kinds_never_publish():
    for kind in ProviderFailureKind:
        trace = worst_case_trace().model_copy(update={"error_code": kind})
        published = publish_trace(trace, AgentAdapter.OFFLINE_SCRIPTED, AgentLimits())
        assert published.tool_error_code is None


def test_tool_error_codes_do_publish():
    published = publish_trace(worst_case_trace(), AgentAdapter.OFFLINE_SCRIPTED, AgentLimits())
    assert published.tool_error_code is ToolErrorCode.INVALID_TOOL_OUTPUT


def test_the_adapter_follows_the_model_not_a_response_field():
    assert AgentAdapter.for_mode(ModelMode.OFFLINE_DEMO_FAKE) is AgentAdapter.OFFLINE_SCRIPTED
    assert AgentAdapter.for_mode(ModelMode.REAL_MODEL) is AgentAdapter.REAL_MODEL
    with pytest.raises(ValueError):
        AgentAdapter.for_mode(ModelMode.NOT_INVOKED)
