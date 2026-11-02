"""The agent adapter against the real SDK and a local HTTP server.

This checks what a stub cannot: that the installed SDK serializes a multi-turn tool
conversation as documented, including returned thinking blocks, and parses the
documented responses. Requests go to 127.0.0.1 only: nothing is sent to Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from footballai_v2.copilot import CopilotSettings
from footballai_v2.copilot.agent import AgentLimits, AgentStopReason, ReadOnlyAgent
from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.models import ModelMode
from footballai_v2.copilot.provider import AnthropicAgentModel

from test_copilot_agent import STATUS, STATUS_TOOL, TRACKING_TOOL, env
from test_copilot_sdk_contract import CANARY, error_body, local_api  # noqa: F401  (a fixture)


SETTINGS = CopilotSettings(provider_mode="real", real_provider_enabled=True)
THINKING = {"type": "thinking", "thinking": "", "signature": "sig-from-the-api"}
TOOL_USE = {"type": "tool_use", "id": "toolu_01", "name": STATUS_TOOL, "input": {}}


def reply(content, stop_reason="end_turn", **changes):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 800, "output_tokens": 40},
        **changes,
    }


TOOL_TURN = reply([THINKING, TOOL_USE], "tool_use")
FINAL_TURN = reply([THINKING, {"type": "text", "text": '{"claim_ids": ["run_status"]}'}])


def ask(question=STATUS, settings=SETTINGS, **options):
    repository, storage, run_id = env()
    return ReadOnlyAgent(repository, storage, AnthropicAgentModel(settings), settings, **options).answer(
        run_id, question
    )


def test_the_tool_conversation_on_the_wire_matches_the_api_contract(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    result = ask()

    assert result.trace.termination is AgentStopReason.ANSWERED
    assert result.response.model_mode is ModelMode.REAL_MODEL
    assert len(local_api.requests) == 2

    method, path, headers, first = local_api.requests[0]
    assert (method, path) == ("POST", "/v1/messages")
    assert {key.lower(): value for key, value in headers.items()}["authorization"] == f"Bearer {CANARY}"
    assert first["tool_choice"] == {"type": "auto"}
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert first["output_config"]["effort"] == "low"
    assert not {"temperature", "top_p", "top_k", "output_format", "thinking"} & set(first)
    assert "stream" not in first or first["stream"] is False
    assert [tool["name"] for tool in first["tools"]][0] == STATUS_TOOL
    for tool in first["tools"]:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False
        assert tool["input_schema"]["properties"] == {}

    second = local_api.requests[1][3]
    assert second["tools"] == first["tools"]
    assert second["output_config"] == first["output_config"]
    assert [message["role"] for message in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][0] == first["messages"][0]


def test_thinking_blocks_go_back_byte_for_byte(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    ask()
    returned = local_api.requests[1][3]["messages"][1]["content"]
    assert returned[0] == THINKING
    assert returned[1]["type"] == "tool_use"
    assert (returned[1]["id"], returned[1]["name"], returned[1]["input"]) == ("toolu_01", STATUS_TOOL, {})


def test_tool_results_answer_every_call_in_one_user_message(local_api):
    second_call = {**TOOL_USE, "id": "toolu_02", "name": TRACKING_TOOL}
    local_api.queue = [reply([THINKING, TOOL_USE, second_call], "tool_use"), FINAL_TURN]
    ask()
    blocks = local_api.requests[1][3]["messages"][2]["content"]
    assert [block["type"] for block in blocks] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in blocks] == ["toolu_01", "toolu_02"]
    assert all(isinstance(block["content"], str) for block in blocks)


def test_a_rejected_call_sets_is_error_on_the_wire(local_api):
    local_api.queue = [reply([{**TOOL_USE, "name": "execute_sql"}], "tool_use"), reply([{"type": "text", "text": '{"claim_ids": []}'}])]
    result = ask()
    block = local_api.requests[1][3]["messages"][2]["content"][0]
    assert block["is_error"] is True
    assert block["content"] == '{"error": "unknown_tool"}'
    assert result.response.unavailable_reason == "no_relevant_claim"


def test_real_usage_and_request_id_reach_the_trace(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    trace = ask().trace
    assert trace.usage.input_tokens == 1600 and trace.usage.output_tokens == 80
    assert trace.usage_complete is True
    assert [turn.request_id for turn in trace.turns] == ["req_local", "req_local"]
    assert [turn.stop_reason for turn in trace.turns] == ["tool_use", "end_turn"]


@pytest.mark.parametrize(
    ("status", "error_type", "kind"),
    [
        (400, "invalid_request_error", ProviderFailureKind.BAD_REQUEST),
        (401, "authentication_error", ProviderFailureKind.AUTHENTICATION),
        (429, "rate_limit_error", ProviderFailureKind.RATE_LIMITED),
        (529, "overloaded_error", ProviderFailureKind.SERVER_ERROR),
    ],
)
def test_real_sdk_errors_stop_the_loop_with_no_retry_or_leak(local_api, status, error_type, kind):
    local_api.status = status
    local_api.body = error_body(error_type)
    result = ask()
    assert result.trace.termination is AgentStopReason.PROVIDER_ERROR
    assert result.trace.error_code is kind
    assert len(local_api.requests) == 1  # a hidden retry would be a hidden charge
    assert CANARY not in result.response.model_dump_json()
    assert CANARY not in result.trace.model_dump_json()


def test_a_real_sdk_timeout_stops_the_loop_with_no_retry(local_api):
    local_api.delay = 1.0
    result = ask(limits=AgentLimits(model_timeout_seconds=0.1))
    assert result.trace.termination is AgentStopReason.MODEL_TIMEOUT
    assert result.response.unavailable_reason == "provider_timeout"
    assert len(local_api.requests) == 1


@pytest.mark.parametrize(
    ("body", "stop"),
    [
        ("not json at all", AgentStopReason.MALFORMED_RESPONSE),
        ({"unexpected": True}, AgentStopReason.MALFORMED_RESPONSE),
        (reply(None), AgentStopReason.MALFORMED_RESPONSE),
        (reply([{"type": "text", "text": "Player 7 ran 10.5 km"}]), AgentStopReason.MALFORMED_RESPONSE),
        (reply([TOOL_USE]), AgentStopReason.MALFORMED_RESPONSE),
        (reply([THINKING], "tool_use"), AgentStopReason.MALFORMED_RESPONSE),
        (reply([], "pause_turn"), AgentStopReason.MALFORMED_RESPONSE),
        (reply([], "refusal", stop_details={"type": "refusal", "category": None}), AgentStopReason.MODEL_REFUSAL),
        (reply([{"type": "text", "text": '{"claim_ids":'}], "max_tokens"), AgentStopReason.MODEL_TRUNCATED),
    ],
)
def test_unusable_responses_fail_closed_with_the_real_sdk(local_api, body, stop):
    local_api.body = body
    result = ask()
    assert result.trace.termination is stop
    assert result.response.support_level.value == "unavailable"
    assert "10.5" not in result.response.answer
