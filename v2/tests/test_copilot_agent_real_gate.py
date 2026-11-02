"""M3-A: every gate in front of a billed model, and the guard that caps its cost.

Nothing here reaches a network. The adapter is replaced by a counting stand-in, so a
test fails if a refused request would have cost anything.
"""

from __future__ import annotations

import math
import subprocess
import sys
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from footballai_v2.api import create_app
from footballai_v2.copilot import CopilotSettings, DeterministicFakeProvider
from footballai_v2.copilot.agent import AgentStopReason, ReadOnlyAgent, tool_definitions
from footballai_v2.copilot.agent.trace import UsageSource
from footballai_v2.copilot.budget import BudgetedAgentModel, BudgetExhausted, BudgetGuard, turn_worst_case_usd
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, call, select
from footballai_v2.copilot.evaluation.dataset import load_dataset
from footballai_v2.copilot.evaluation.fixtures import build_fixture_run
from footballai_v2.copilot.models import ModelMode, ProviderTelemetry, TokenUsage
from footballai_v2.copilot.pricing import MODEL_PRICES, request_cost_usd, worst_case_cost_usd
from footballai_v2.copilot.provider import (
    AgentModelRequest,
    AnthropicAgentModel,
    AnthropicTextProvider,
    api_base_url,
    build_provider,
)
from footballai_v2.copilot.real_agent import RealAgentRefused, real_agent_blockers, real_agent_factory
from footballai_v2.execution.coordinator import ExecutionSettings

from test_copilot_agent import STATUS, STATUS_TOOL, env


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = load_dataset(REPO_ROOT / "evaluation" / "copilot" / "v2")
HAIKU = MODEL_PRICES["claude-haiku-5-5"]
CANARY = "canary-value-not-real"
READY = CopilotSettings(
    enabled=True,
    environment="test",
    agent_enabled=True,
    provider_mode="real",
    real_provider_enabled=True,
    agent_real_model_enabled=True,
    agent_max_requests=5,
    agent_max_cost_usd=0.05,
)
ENVIRONMENT = {"ANTHROPIC_API_KEY": CANARY}
LOOPBACK = {"client": ("127.0.0.1", 50000), "base_url": "http://127.0.0.1:8000"}
SDK_PRESENT = lambda: True  # noqa: E731
REQUEST = AgentModelRequest(
    system_instructions="system",
    tools=tool_definitions(),
    messages=({"role": "user", "content": "{}"},),
    offered_claim_ids=("output_nature", "run_status"),
    max_output_tokens=1024,
    timeout_seconds=5.0,
)


def blockers(settings=READY, environ=ENVIRONMENT):
    return real_agent_blockers(settings, environ, sdk_installed=SDK_PRESENT)


# -- the gate ---------------------------------------------------------------------


def test_a_fully_configured_local_path_has_no_blockers():
    assert blockers() == ()


def test_the_real_path_is_off_by_default():
    settings = CopilotSettings()
    assert settings.agent_real_model_enabled is False
    assert settings.agent_max_requests == 0
    assert settings.agent_max_cost_usd == 0.0
    assert "real_agent_not_enabled" in blockers(settings)


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({"agent_real_model_enabled": False}, "real_agent_not_enabled"),
        ({"provider_mode": "fake"}, "provider_not_real"),
        ({"real_provider_enabled": False}, "provider_not_real"),
        ({"agent_max_requests": 0}, "no_request_cap"),
        ({"agent_max_cost_usd": 0.0}, "no_cost_cap"),
        ({"model_name": "claude-unknown-9"}, "unpriced_model"),
    ],
)
def test_each_setting_is_required(change, blocker):
    assert blocker in blockers(replace(READY, **change))


def test_a_missing_key_blocks_and_is_never_echoed():
    found = blockers(READY, {})
    assert found == ("missing_api_key",)


@pytest.mark.parametrize(
    "marker",
    ["CONTAINER_APP_NAME", "CONTAINER_APP_ENV_DNS_SUFFIX", "WEBSITE_INSTANCE_ID", "KUBERNETES_SERVICE_HOST"],
)
def test_a_hosted_platform_is_refused_even_when_it_claims_to_be_local(marker):
    assert "hosted_environment" in blockers(READY, {**ENVIRONMENT, marker: "anything"})


@pytest.mark.parametrize("marker", ["CI", "GITHUB_ACTIONS"])
def test_continuous_integration_is_refused(marker):
    assert "continuous_integration" in blockers(READY, {**ENVIRONMENT, marker: "true"})


def test_the_sdk_must_be_installed():
    assert real_agent_blockers(READY, ENVIRONMENT, sdk_installed=lambda: False) == ("sdk_not_installed",)


@pytest.mark.parametrize(
    "url", ["https://proxy.example.com", "http://api.anthropic.com", "https://api.anthropic.com.evil.test"]
)
def test_a_key_is_never_sent_to_an_unapproved_endpoint(url):
    assert "unapproved_api_endpoint" in blockers(READY, {**ENVIRONMENT, "ANTHROPIC_BASE_URL": url})


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "https://api.anthropic.com"),
        ("", "https://api.anthropic.com"),
        ("https://api.anthropic.com", "https://api.anthropic.com"),
        ("http://127.0.0.1:9999", "http://127.0.0.1:9999"),
        ("http://localhost:9999", "http://localhost:9999"),
        ("https://proxy.example.com", None),
        ("http://127.0.0.1.evil.test", None),
        ("ftp://127.0.0.1", None),
    ],
)
def test_the_endpoint_rule(configured, expected):
    assert api_base_url({} if configured is None else {"ANTHROPIC_BASE_URL": configured}) == expected


def test_a_blocked_configuration_refuses_to_build_rather_than_falling_back():
    with pytest.raises(RealAgentRefused) as caught:
        real_agent_factory(replace(READY, agent_max_cost_usd=0.0), {**ENVIRONMENT}, sdk_installed=SDK_PRESENT)
    assert caught.value.blockers == ("no_cost_cap",)
    assert CANARY not in str(caught.value)


@pytest.mark.parametrize(
    ("variable", "value"),
    [("REQUESTS", "abc"), ("REQUESTS", "-1"), ("REQUESTS", "201"), ("COST_USD", "nan"), ("COST_USD", "inf"), ("COST_USD", "5.01")],
)
def test_an_invalid_cap_is_no_cap(monkeypatch, variable, value):
    monkeypatch.setenv("FOOTBALLAI_ENVIRONMENT", "test")
    monkeypatch.setenv("FOOTBALLAI_COPILOT_ENABLED", "1")
    monkeypatch.setenv("FOOTBALLAI_COPILOT_AGENT_ENABLED", "1")
    monkeypatch.setenv(f"FOOTBALLAI_COPILOT_AGENT_MAX_{variable}", value)
    settings = CopilotSettings.from_environment()
    assert settings.agent_max_requests == 0 and settings.agent_max_cost_usd == 0.0


def test_the_flag_reads_only_with_the_agent_gate(monkeypatch):
    monkeypatch.setenv("FOOTBALLAI_ENVIRONMENT", "staging")
    monkeypatch.setenv("FOOTBALLAI_COPILOT_ENABLED", "1")
    monkeypatch.setenv("FOOTBALLAI_COPILOT_AGENT_ENABLED", "1")
    monkeypatch.setenv("FOOTBALLAI_COPILOT_AGENT_REAL_MODEL_ENABLED", "1")
    assert CopilotSettings.from_environment().agent_real_model_enabled is False


def test_the_setting_cannot_be_built_without_the_agent_gate():
    with pytest.raises(ValueError):
        CopilotSettings(agent_real_model_enabled=True)


# -- the route ----------------------------------------------------------------------


@pytest.fixture
def execution(tmp_path):
    return ExecutionSettings(tmp_path / "runs", tmp_path / "queue", max_upload_bytes=1024, allow_test_profiles=True)


def _app(execution, monkeypatch, gate, factory=None):
    repository, storage, run_id = build_fixture_run(DATASET.fixtures.fixtures["verified_real"], DATASET.fixtures)
    coordinator = SimpleNamespace(repository=repository, object_storage=storage, queue=None)
    monkeypatch.setattr("footballai_v2.api.app.AnalysisCoordinator", lambda _settings: coordinator)
    app = create_app(
        execution.run_root,
        settings=execution,
        copilot_settings=gate,
        copilot_provider=DeterministicFakeProvider(),
        copilot_agent_model_factory=factory,
    )
    return TestClient(app, **LOOPBACK), run_id


def test_a_blocked_server_refuses_to_start_instead_of_serving_the_script(execution, monkeypatch):
    for name in ("CI", "GITHUB_ACTIONS", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RealAgentRefused) as caught:
        _app(execution, monkeypatch, READY)
    assert "missing_api_key" in caught.value.blockers


def test_the_scripted_default_ignores_a_real_provider_setting(execution, monkeypatch):
    gate = replace(READY, agent_real_model_enabled=False, agent_max_requests=0, agent_max_cost_usd=0.0)
    client, run_id = _app(execution, monkeypatch, gate)
    assert client.get("/api/v1/copilot/agent").json() == {"adapter": "offline_scripted", "model": "scripted-stub"}
    body = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS}).json()
    assert body["trace"]["adapter"] == "offline_scripted"
    assert body["explanation"]["model_mode"] == "offline_demo_fake"


def test_the_adapter_route_reports_a_real_adapter_truthfully(execution, monkeypatch):
    gate = CopilotSettings(enabled=True, agent_enabled=True, environment="test")
    real = SimpleNamespace(mode=ModelMode.REAL_MODEL, model_name="claude-haiku-5-5")
    client, _ = _app(execution, monkeypatch, gate, factory=lambda: real)
    assert client.get("/api/v1/copilot/agent").json() == {"adapter": "real_model", "model": "claude-haiku-5-5"}


def test_the_adapter_route_is_local_like_the_agent(execution, monkeypatch):
    gate = CopilotSettings(enabled=True, agent_enabled=True, environment="test")
    client, _ = _app(execution, monkeypatch, gate)
    assert client.get("/api/v1/copilot/agent", headers={"Host": "tunnel.example.com"}).status_code == 404
    assert client.get("/api/v1/copilot/agent", headers={"Origin": "https://evil.example.com"}).status_code == 404


def test_the_adapter_route_does_not_exist_when_the_agent_is_off(execution, monkeypatch):
    client, _ = _app(execution, monkeypatch, CopilotSettings(enabled=True, environment="test"))
    assert client.get("/api/v1/copilot/agent").status_code == 404


def test_a_client_cannot_choose_the_model_or_supply_a_key(execution, monkeypatch):
    gate = CopilotSettings(enabled=True, agent_enabled=True, environment="test")
    client, run_id = _app(execution, monkeypatch, gate)
    for extra in ({"model": "claude-opus-5-5"}, {"api_key": CANARY}, {"run_id": run_id}):
        response = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS, **extra})
        assert response.status_code == 422  # rejected before any model or adapter is built


def test_the_real_agent_never_turns_explain_into_an_unbudgeted_billed_route():
    """``/explain`` has no cost guard and no loopback guard, so it stays on the fake."""
    assert isinstance(build_provider(READY), DeterministicFakeProvider)
    alone = replace(READY, agent_real_model_enabled=False, agent_max_requests=0, agent_max_cost_usd=0.0)
    assert isinstance(build_provider(alone), AnthropicTextProvider)  # the M1 path is unchanged


def test_a_factory_model_is_closed_after_its_request(execution, monkeypatch):
    gate = CopilotSettings(enabled=True, agent_enabled=True, environment="test")
    closed = []

    class Closing(ScriptedAgentModel):
        def close(self):
            closed.append(True)

    client, run_id = _app(execution, monkeypatch, gate, factory=lambda: Closing([call(STATUS_TOOL), select("run_status")]))
    assert client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS}).status_code == 200
    assert closed == [True]


def test_the_real_path_does_not_load_the_evaluation_package():
    code = (
        "import sys, types\n"
        "sys.modules['anthropic'] = types.ModuleType('anthropic')\n"
        "from footballai_v2.copilot import CopilotSettings\n"
        "from footballai_v2.copilot.real_agent import real_agent_factory\n"
        "settings = CopilotSettings(enabled=True, environment='test', agent_enabled=True, provider_mode='real',"
        " real_provider_enabled=True, agent_real_model_enabled=True, agent_max_requests=1, agent_max_cost_usd=0.01)\n"
        "real_agent_factory(settings, {'ANTHROPIC_API_KEY': 'x'}, sdk_installed=lambda: True)()\n"
        "import footballai_v2.api.agent_route\n"
        "sys.exit(1 if [m for m in sys.modules if m.startswith('footballai_v2.copilot.evaluation')] else 0)\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "v2" / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_dashboard_never_handles_a_credential():
    source = REPO_ROOT / "v2" / "dashboard" / "src"
    for path in source.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        assert "ANTHROPIC" not in text and "api_key" not in text, path


@pytest.mark.parametrize("name", ["api.Dockerfile", "worker.Dockerfile"])
def test_no_runtime_image_installs_the_sdk(name):
    assert "requirements-copilot-eval" not in (REPO_ROOT / "docker" / name).read_text(encoding="utf-8")


def test_hosted_terraform_never_opens_the_gate():
    text = (REPO_ROOT / "infra/terraform/environments/staging/workloads.tf").read_text(encoding="utf-8")
    assert "FOOTBALLAI_COPILOT" not in text and "ANTHROPIC" not in text


# -- the cost guard -----------------------------------------------------------------


class Priced:
    """A stand-in for the billed adapter. It counts calls, and reports what it is told."""

    mode = ModelMode.REAL_MODEL
    model_name = "claude-haiku-5-5"

    def __init__(self, usage=TokenUsage(1000, 100), error=None):
        self.usage = usage
        self.error = error
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        telemetry = ProviderTelemetry(model=self.model_name, latency_seconds=0.1, usage=self.usage)
        if self.error == "refusal":
            raise ProviderUnavailableError("x", kind=ProviderFailureKind.REFUSAL, telemetry=telemetry)
        if self.error == "timeout":
            raise ProviderTimeoutError("x", telemetry=ProviderTelemetry(model=self.model_name, latency_seconds=5.0))
        return SimpleNamespace(tool_calls=(), claim_ids=(), telemetry=telemetry)


def budgeted(inner, requests=10, cost=1.0):
    guard = BudgetGuard(requests, cost, HAIKU)
    return BudgetedAgentModel(inner, guard, READY), guard


def test_a_request_past_the_request_cap_is_never_sent():
    inner = Priced()
    model, _ = budgeted(inner, requests=1)
    model.complete(REQUEST)
    with pytest.raises(ProviderUnavailableError) as caught:
        model.complete(REQUEST)
    assert caught.value.kind is ProviderFailureKind.BUDGET_EXHAUSTED
    assert inner.calls == 1


def test_a_request_whose_worst_case_does_not_fit_is_never_sent():
    inner = Priced()
    model, _ = budgeted(inner, cost=turn_worst_case_usd(HAIKU, REQUEST) * 0.99)
    with pytest.raises(ProviderUnavailableError):
        model.complete(REQUEST)
    assert inner.calls == 0


def test_reported_usage_replaces_the_reservation():
    model, guard = budgeted(Priced())
    model.complete(REQUEST)
    assert guard.spent_usd == pytest.approx(request_cost_usd(HAIKU, TokenUsage(1000, 100)))
    assert guard.reserved_usd == 0
    assert guard.requests_at_worst_case == 0


def test_a_request_refused_locally_is_not_counted_or_charged():
    class Unconfigured(Priced):
        def complete(self, request):
            raise ProviderUnavailableError("x", kind=ProviderFailureKind.NOT_CONFIGURED)

    model, guard = budgeted(Unconfigured())
    with pytest.raises(ProviderUnavailableError):
        model.complete(REQUEST)
    assert (guard.requests_sent, guard.spent_usd, guard.reserved_usd) == (0, 0.0, 0.0)


def test_a_billed_refusal_is_charged():
    model, guard = budgeted(Priced(error="refusal"))
    with pytest.raises(ProviderUnavailableError):
        model.complete(REQUEST)
    assert guard.spent_usd == pytest.approx(request_cost_usd(HAIKU, TokenUsage(1000, 100)))


def test_a_timeout_is_charged_at_its_worst_case():
    model, guard = budgeted(Priced(error="timeout"))
    with pytest.raises(ProviderTimeoutError):
        model.complete(REQUEST)
    assert guard.spent_usd == pytest.approx(turn_worst_case_usd(HAIKU, REQUEST))
    assert guard.requests_at_worst_case == 1


def test_concurrent_requests_cannot_share_the_same_headroom():
    guard = BudgetGuard(10, 1.0, HAIKU)
    guard.admit(0.6)
    with pytest.raises(BudgetExhausted) as caught:
        guard.admit(0.6)  # the first is in flight and has not been charged yet
    assert caught.value.limit == "max_cost_usd"


def test_the_worst_case_grows_with_every_earlier_turn():
    assistant = {"role": "assistant", "content": ["unserialisable sdk objects"]}
    results = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x" * 100}]}
    later = replace(REQUEST, messages=(*REQUEST.messages, assistant, results))
    assert turn_worst_case_usd(HAIKU, later) > turn_worst_case_usd(HAIKU, REQUEST) + 1024 * HAIKU.input / 1_000_000


def test_the_worst_case_counts_bytes_not_characters():
    wide = replace(REQUEST, messages=({"role": "user", "content": "é" * 1000},))
    narrow = replace(REQUEST, messages=({"role": "user", "content": "e" * 1000},))
    assert turn_worst_case_usd(HAIKU, wide) > turn_worst_case_usd(HAIKU, narrow)


def test_the_budget_stop_is_typed_and_public_as_a_limit():
    repository, storage, run_id = env()
    inner = Priced()
    model, _ = budgeted(inner, requests=0)
    result = ReadOnlyAgent(repository, storage, model, READY).answer(run_id, STATUS)
    assert result.trace.termination is AgentStopReason.BUDGET_EXHAUSTED
    assert result.response.unavailable_reason == "agent_limit_reached"
    assert result.trace.model_turns == 0  # nothing was sent, so no turn happened
    assert result.response.model_mode is ModelMode.NOT_INVOKED
    assert inner.calls == 0


def test_verified_prices_match_the_documentation_reviewed_for_this_milestone():
    assert (HAIKU.input, HAIKU.output, HAIKU.cache_write, HAIKU.cache_read) == (0.10, 0.50, 0.125, 0.01)
    assert HAIKU.long_prompt_multiplier == 5.0
    sonnet = MODEL_PRICES["claude-sonnet-5-5"]
    assert (sonnet.input, sonnet.output, sonnet.cache_write, sonnet.cache_read) == (2.0, 10.0, 2.5, 0.10)


def test_the_long_prompt_rate_starts_above_the_threshold():
    below = request_cost_usd(HAIKU, TokenUsage(100_000, 0))
    above = request_cost_usd(HAIKU, TokenUsage(100_001, 0))
    assert above == pytest.approx(below * 5, rel=1e-3)


def test_a_worst_case_bound_is_finite_and_positive():
    bound = worst_case_cost_usd(HAIKU, 24_576, 1024)
    assert 0 < bound < 0.01 and math.isfinite(bound)


# -- usage accounting -----------------------------------------------------------------


def test_stub_tokens_are_never_reported_as_measurements():
    repository, storage, run_id = env()
    model = ScriptedAgentModel([call(STATUS_TOOL), select("run_status")])
    trace = ReadOnlyAgent(repository, storage, model, CopilotSettings()).answer(run_id, STATUS).trace
    assert trace.usage_source is UsageSource.SCRIPTED_CONSTANT
    assert trace.estimated_cost_usd is None


def test_a_real_adapter_reports_provider_usage_and_a_list_price_estimate():
    repository, storage, run_id = env()
    model = Priced()
    model.complete = lambda _request: SimpleNamespace(
        tool_calls=(),
        claim_ids=(),
        telemetry=ProviderTelemetry(model="claude-haiku-5-5", latency_seconds=0.1, usage=TokenUsage(1000, 100)),
    )
    trace = ReadOnlyAgent(repository, storage, model, READY).answer(run_id, STATUS).trace
    assert trace.usage_source is UsageSource.PROVIDER_REPORTED
    assert trace.estimated_cost_usd == pytest.approx(request_cost_usd(HAIKU, TokenUsage(1000, 100)))
    assert trace.cost_is_lower_bound is False


def test_a_turn_with_no_usage_makes_the_estimate_a_lower_bound():
    repository, storage, run_id = env()
    model = Priced(error="timeout")
    trace = ReadOnlyAgent(repository, storage, model, READY).answer(run_id, STATUS).trace
    assert trace.usage_complete is False
    assert trace.usage_source is UsageSource.UNAVAILABLE
    assert trace.cost_is_lower_bound is True


def test_an_adapter_without_a_real_key_reports_no_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    repository, storage, run_id = env()
    trace = ReadOnlyAgent(repository, storage, AnthropicAgentModel(READY), READY).answer(run_id, STATUS).trace
    assert trace.termination is AgentStopReason.PROVIDER_ERROR
    assert trace.error_code is ProviderFailureKind.NOT_CONFIGURED
    assert trace.model_turns == 0
