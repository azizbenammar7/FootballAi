"""The real agent adapter against a stand-in SDK, so these run with no SDK installed."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest

from footballai_v2.copilot import CopilotSettings
from footballai_v2.copilot.agent import AgentStopReason, ReadOnlyAgent, tool_definitions
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evaluation.agent_models import text_block, thinking_block, tool_use_block
from footballai_v2.copilot.grounding import AGENT_SYSTEM_INSTRUCTIONS
from footballai_v2.copilot.models import ModelMode, TokenUsage
from footballai_v2.copilot.provider import (
    AgentModelRequest,
    AnthropicAgentModel,
    build_agent_model_request,
    parse_agent_message,
)

from test_copilot_agent import STATUS, STATUS_TOOL, env


SETTINGS = CopilotSettings(provider_mode="real", real_provider_enabled=True)
CANARY = "canary-value-not-real"
REQUEST = AgentModelRequest(
    system_instructions=AGENT_SYSTEM_INSTRUCTIONS,
    tools=tool_definitions(),
    messages=({"role": "user", "content": "{}"},),
    offered_claim_ids=("output_nature", "run_status"),
    max_output_tokens=1024,
    timeout_seconds=7.5,
)


class StubTimeout(Exception):
    pass


class StubStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream said {CANARY}")
        self.status_code = status_code


def message(content, stop_reason="end_turn", request_id="req_ok"):
    usage = types.SimpleNamespace(input_tokens=11, output_tokens=7)
    reply = types.SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)
    reply._request_id = request_id
    return reply


TOOL_TURN = message([thinking_block(), tool_use_block(STATUS_TOOL, {}, "toolu_1")], "tool_use")
FINAL_TURN = message([thinking_block(), text_block('{"claim_ids": ["run_status"]}')])


@pytest.fixture
def sdk(monkeypatch):
    state = types.SimpleNamespace(outcomes=[], calls=[], constructed=[])

    def create(**kwargs):
        state.calls.append(kwargs)
        outcome = state.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    module = types.ModuleType("anthropic")

    class Anthropic:
        def __init__(self, **kwargs) -> None:
            state.constructed.append(kwargs)
            self.messages = types.SimpleNamespace(create=create)

    module.Anthropic = Anthropic
    module.APITimeoutError = StubTimeout
    module.APIConnectionError = type("StubConnection", (Exception,), {})
    module.APIResponseValidationError = type("StubValidation", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    return state


def test_a_request_holds_the_tools_the_schema_and_nothing_else():
    arguments = build_agent_model_request(REQUEST, SETTINGS)
    assert set(arguments) == {"model", "max_tokens", "system", "messages", "tools", "tool_choice", "output_config"}
    assert arguments["model"] == "claude-haiku-5-5"
    assert arguments["max_tokens"] == 1024
    assert arguments["tool_choice"] == {"type": "auto"}
    assert arguments["output_config"]["effort"] == "low"
    schema = arguments["output_config"]["format"]["schema"]
    assert schema["properties"]["claim_ids"]["items"]["enum"] == ["output_nature", "run_status"]
    assert schema["additionalProperties"] is False


def test_the_adapter_never_retries_and_passes_the_turn_timeout(sdk):
    sdk.outcomes = [FINAL_TURN]
    AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert sdk.constructed[0]["max_retries"] == 0
    assert sdk.calls[0]["timeout"] == 7.5


def test_a_tool_turn_keeps_the_content_object_for_the_next_turn(sdk):
    sdk.outcomes = [TOOL_TURN]
    reply = AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert reply.assistant_content is TOOL_TURN.content
    assert [(call.call_id, call.name, call.arguments) for call in reply.tool_calls] == [("toolu_1", STATUS_TOOL, {})]
    assert reply.claim_ids == ()
    assert reply.telemetry.usage == TokenUsage(11, 7)
    assert reply.telemetry.request_id == "req_ok"
    assert reply.telemetry.stop_reason == "tool_use"


def test_a_final_turn_is_a_selection(sdk):
    sdk.outcomes = [FINAL_TURN]
    reply = AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert reply.tool_calls == () and reply.claim_ids == ("run_status",)


def test_the_real_adapter_drives_the_loop_and_reports_a_real_mode(sdk):
    sdk.outcomes = [TOOL_TURN, FINAL_TURN]
    repository, storage, run_id = env()
    result = ReadOnlyAgent(repository, storage, AnthropicAgentModel(SETTINGS), SETTINGS).answer(run_id, STATUS)
    assert result.response.model_mode is ModelMode.REAL_MODEL
    assert result.trace.termination is AgentStopReason.ANSWERED
    assert result.trace.model == "claude-haiku-5-5"
    assert result.trace.usage.input_tokens == 22
    assert len(sdk.calls) == 2
    assert len(sdk.constructed) == 1  # one client, so one connection pool, for the whole question
    assert sdk.calls[1]["messages"][1]["content"] is TOOL_TURN.content


@pytest.mark.parametrize(
    "settings",
    [
        CopilotSettings(),
        CopilotSettings(provider_mode="real"),
        CopilotSettings(real_provider_enabled=True),
    ],
)
def test_a_billed_call_needs_the_explicit_opt_in(sdk, settings):
    with pytest.raises(ProviderUnavailableError) as caught:
        AnthropicAgentModel(settings).complete(REQUEST)
    assert caught.value.kind is ProviderFailureKind.NOT_CONFIGURED
    assert sdk.calls == [] and sdk.constructed == []


def test_a_missing_key_never_reaches_the_sdk(sdk, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ProviderUnavailableError) as caught:
        AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert caught.value.kind is ProviderFailureKind.NOT_CONFIGURED
    assert sdk.constructed == []


def test_a_missing_sdk_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    with pytest.raises(ProviderUnavailableError) as caught:
        AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert caught.value.kind is ProviderFailureKind.NOT_CONFIGURED


def test_sdk_timeouts_are_typed(sdk):
    sdk.outcomes = [StubTimeout(f"timed out {CANARY}")]
    with pytest.raises(ProviderTimeoutError) as caught:
        AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert CANARY not in str(caught.value)


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, ProviderFailureKind.AUTHENTICATION), (429, ProviderFailureKind.RATE_LIMITED), (500, ProviderFailureKind.SERVER_ERROR)],
)
def test_sdk_status_errors_are_typed_and_keep_nothing_of_the_body(sdk, status, kind):
    sdk.outcomes = [StubStatusError(status)]
    with pytest.raises(ProviderUnavailableError) as caught:
        AnthropicAgentModel(SETTINGS).complete(REQUEST)
    assert caught.value.kind is kind
    assert CANARY not in str(caught.value)
    assert len(sdk.calls) == 1


def test_an_unusable_turn_keeps_its_billed_usage(sdk):
    sdk.outcomes = [message([], "refusal")]
    repository, storage, run_id = env()
    result = ReadOnlyAgent(repository, storage, AnthropicAgentModel(SETTINGS), SETTINGS).answer(run_id, STATUS)
    assert result.trace.termination is AgentStopReason.MODEL_REFUSAL
    assert result.trace.usage.input_tokens == 11


def test_the_stub_parser_is_the_real_parser():
    reply = parse_agent_message(message([text_block(json.dumps({"claim_ids": []}))]))
    assert reply.claim_ids == () and reply.tool_calls == ()


def test_changing_the_model_changes_only_the_model():
    settings = replace(SETTINGS, model_name="claude-sonnet-5-5", effort="medium")
    arguments = build_agent_model_request(REQUEST, settings)
    assert arguments["model"] == "claude-sonnet-5-5"
    assert arguments["output_config"]["effort"] == "medium"
