"""M3-A: the Anthropic protocol, exercised by the pinned SDK over real HTTP.

Everything is served from 127.0.0.1, so nothing reaches Anthropic and nothing is
billed. The SDK does its own serialization and parsing here: this is what a stubbed
adapter cannot show. It says nothing about how a live model behaves, or whether the
live API accepts these requests. See docs/v2/COPILOT_AGENT_REAL_MODEL.md.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("anthropic")

from footballai_v2.copilot import CopilotSettings
from footballai_v2.copilot.agent import AgentLimits, AgentStopReason, ReadOnlyAgent, ToolName
from footballai_v2.copilot.agent.trace import UsageSource
from footballai_v2.copilot.budget import BudgetedAgentModel, BudgetGuard
from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.pricing import MODEL_PRICES
from footballai_v2.copilot.provider import AnthropicAgentModel
from footballai_v2.execution.coordinator import ExecutionSettings

from test_copilot_agent import STATUS, STATUS_TOOL, TRACKING_TOOL, env
from test_copilot_agent_real_gate import READY, _app
from test_copilot_agent_sdk_contract import FINAL_TURN, SETTINGS, THINKING, TOOL_TURN, TOOL_USE, ask, reply
from test_copilot_sdk_contract import CANARY, error_body, local_api  # noqa: F401  (a fixture)


USAGE = {"input_tokens": 700, "output_tokens": 30}
SUMMARY = "What was the outcome of this run?"
ARTIFACTS = ToolName.GET_ARTIFACT_METADATA.value
MODEL_TOOL = ToolName.GET_MODEL_PROVENANCE.value
LIMITS_TOOL = ToolName.GET_EVIDENCE_LIMITATIONS.value


def use(name, call_id="toolu_01", arguments=None):
    return {"type": "tool_use", "id": call_id, "name": name, "input": {} if arguments is None else arguments}


def turn(*blocks, stop="tool_use", **changes):
    return reply(list(blocks), stop, **changes)


def final(*claim_ids):
    return reply([THINKING, {"type": "text", "text": json.dumps({"claim_ids": list(claim_ids)})}])


# -- tool use and tool results --------------------------------------------------------


def test_two_calls_in_one_turn_both_run_and_answer_in_order(local_api):
    local_api.queue = [turn(THINKING, use(STATUS_TOOL), use(TRACKING_TOOL, "toolu_02")), final("run_status")]
    result = ask()
    assert result.trace.termination is AgentStopReason.ANSWERED
    assert [call.turn for call in result.trace.tool_calls] == [1, 1]
    blocks = local_api.requests[1][3]["messages"][2]["content"]
    assert [block["tool_use_id"] for block in blocks] == ["toolu_01", "toolu_02"]
    assert {block["type"] for block in blocks} == {"tool_result"}


def test_calls_across_turns_append_to_the_same_history(local_api):
    local_api.queue = [
        turn(THINKING, use(STATUS_TOOL)),
        turn(THINKING, use(TRACKING_TOOL, "toolu_02")),
        final("run_status"),
    ]
    result = ask()
    assert result.trace.model_turns == 3
    first, second, third = (request[3]["messages"] for request in local_api.requests)
    assert second[: len(first)] == first and third[: len(second)] == second
    assert [message["role"] for message in third] == ["user", "assistant", "user", "assistant", "user"]


def test_text_beside_a_tool_call_goes_back_and_is_never_shown(local_api):
    note = {"type": "text", "text": "Player 7 ran 10.5 km"}
    local_api.queue = [turn(THINKING, note, use(STATUS_TOOL)), final("run_status")]
    result = ask()
    assert local_api.requests[1][3]["messages"][1]["content"][1]["text"] == note["text"]
    assert "10.5" not in result.response.answer


def test_a_tool_result_carries_server_claims_with_citations_and_no_prompt_echo(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    ask()
    content = json.loads(local_api.requests[1][3]["messages"][2]["content"][0]["content"])
    assert set(content) == {"tool", "status", "output_nature", "claims"}
    assert content["tool"] == STATUS_TOOL
    assert all(claim["citations"] for claim in content["claims"])


def test_redacted_and_unknown_blocks_go_back_unchanged(local_api):
    redacted = {"type": "redacted_thinking", "data": "opaque"}
    local_api.queue = [turn(redacted, use(STATUS_TOOL)), final("run_status")]
    result = ask()
    assert result.trace.termination is AgentStopReason.ANSWERED
    assert local_api.requests[1][3]["messages"][1]["content"][0] == redacted


# -- structured final selections --------------------------------------------------------


def test_the_structured_final_selection_is_rendered_by_the_server(local_api):
    local_api.queue = [TOOL_TURN, final("run_status")]
    result = ask()
    assert result.trace.claim_ids_rendered == ("output_nature", "run_status")
    assert "run_status" not in result.response.answer  # an id is never shown as text


def test_the_schema_on_the_wire_offers_only_this_questions_claims(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    ask()
    schema = local_api.requests[0][3]["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["claim_ids"]["items"]["enum"]) == {"output_nature", "run_status"}
    assert "maxItems" not in schema["properties"]["claim_ids"]  # the API rejects it
    assert "uniqueItems" not in schema["properties"]["claim_ids"]


def test_no_forced_tool_choice_and_no_fallback_is_ever_sent(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    ask()
    for _method, _path, _headers, body in local_api.requests:
        assert body["tool_choice"] == {"type": "auto"}
        assert "fallbacks" not in body and "thinking" not in body and "stream" not in body


@pytest.mark.parametrize(
    "text",
    ['{"claim_ids": ["run_status", "run_status"]}', '{"claim_ids": ["run_status"], "answer": "x"}', "null", '{"claim_ids": "run_status"}'],
)
def test_an_invalid_selection_fails_closed(local_api, text):
    local_api.queue = [TOOL_TURN, reply([{"type": "text", "text": text}])]
    result = ask()
    assert result.trace.termination is AgentStopReason.MALFORMED_RESPONSE
    assert result.response.citations == []


def test_an_id_the_schema_allowed_but_no_tool_returned_is_rejected(local_api):
    local_api.queue = [final("run_status")]
    result = ask()
    assert result.trace.termination is AgentStopReason.SELECTION_REJECTED
    assert result.trace.selected_ids_rejected == 1
    assert result.response.unavailable_reason == "forged_citation"


def test_an_empty_end_turn_fails_closed_with_no_second_call(local_api):
    local_api.queue = [TOOL_TURN, reply([])]
    result = ask()
    assert result.trace.termination is AgentStopReason.MALFORMED_RESPONSE
    assert len(local_api.requests) == 2


# -- missing and invalid tool calls -------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        turn(THINKING),  # a tool stop with no call
        turn({"type": "tool_use", "id": "has spaces!", "name": STATUS_TOOL, "input": {}}),
        turn({"type": "tool_use", "name": STATUS_TOOL, "input": {}}),  # no id
        turn({"type": "tool_use", "id": "toolu_01", "input": {}}),  # no name
        turn(use(STATUS_TOOL), use(TRACKING_TOOL)),  # one id, two calls
        reply([use(STATUS_TOOL)], "end_turn"),  # a call with a finishing stop
    ],
)
def test_a_malformed_tool_turn_fails_closed_with_nothing_run(local_api, body):
    local_api.body = body
    result = ask()
    assert result.trace.termination is AgentStopReason.MALFORMED_RESPONSE
    assert result.trace.tool_calls == ()


def test_a_name_outside_the_registry_runs_nothing_and_spends_the_budget(local_api):
    local_api.queue = [turn(use("drop_table")), final()]
    result = ask()
    call = result.trace.tool_calls[0]
    assert call.tool is None and call.error_code.value == "unknown_tool"
    assert result.trace.invalid_tool_calls == 1 and result.trace.valid_tool_calls == 0
    assert local_api.requests[1][3]["messages"][2]["content"][0]["is_error"] is True


def test_any_argument_is_rejected_with_a_fixed_error(local_api):
    local_api.queue = [turn(use(STATUS_TOOL, arguments={"limit": 1})), final()]
    result = ask()
    block = local_api.requests[1][3]["messages"][2]["content"][0]
    assert block == {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": '{"error": "invalid_arguments"}',
        "is_error": True,
    }
    assert result.trace.tool_calls[0].error_code.value == "invalid_arguments"


def test_a_run_reference_stops_the_request_before_anything_runs(local_api):
    other = "99999999-9999-4999-8999-999999999999"
    local_api.body = turn(use(STATUS_TOOL, arguments={"target": other}), use(TRACKING_TOOL, "toolu_02"))
    result = ask()
    assert result.trace.termination is AgentStopReason.CROSS_RUN_REJECTED
    assert result.trace.tool_calls == () and result.trace.tool_calls_refused == 2
    assert len(local_api.requests) == 1
    assert other not in result.trace.model_dump_json()


# -- budgets -------------------------------------------------------------------------------


def test_exhausting_the_tool_budget_runs_nothing_from_that_turn(local_api):
    five = [use(name, f"toolu_{index}") for index, name in enumerate(ToolName)]
    local_api.body = turn(THINKING, *five)
    result = ask(SUMMARY)
    assert result.trace.termination is AgentStopReason.TOOL_BUDGET_EXHAUSTED
    assert result.trace.tool_calls == () and result.trace.tool_calls_refused == 5
    assert result.response.unavailable_reason == "agent_limit_reached"
    assert len(local_api.requests) == 1


def test_the_tool_budget_spans_turns(local_api):
    names = list(ToolName)
    local_api.queue = [
        turn(use(names[0].value, "toolu_1"), use(names[1].value, "toolu_2"), use(names[2].value, "toolu_3")),
        turn(use(names[3].value, "toolu_4"), use(names[4].value, "toolu_5")),
    ]
    result = ask(SUMMARY)
    assert result.trace.termination is AgentStopReason.TOOL_BUDGET_EXHAUSTED
    assert len(result.trace.tool_calls) == 3 and result.trace.tool_calls_refused == 2


def test_the_money_guard_stops_a_conversation_between_turns(local_api):
    guard = BudgetGuard(1, 1.0, MODEL_PRICES["claude-haiku-5-5"])
    model = BudgetedAgentModel(AnthropicAgentModel(SETTINGS), guard, SETTINGS)
    repository, storage, run_id = env()
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    result = ReadOnlyAgent(repository, storage, model, SETTINGS).answer(run_id, STATUS)
    assert result.trace.termination is AgentStopReason.BUDGET_EXHAUSTED
    assert result.trace.model_turns == 1
    assert len(local_api.requests) == 1
    assert result.trace.usage.input_tokens == 800  # the billed turn stays on the record
    assert guard.requests_sent == 1


# -- timeouts, networks, and malformed bodies ---------------------------------------------


class Drip:
    """A server that keeps a response open, one byte at a time, so no read ever times out."""

    def __init__(self) -> None:
        self.hung_up_after: float | None = None
        self.started: float | None = None
        self.seen = threading.Event()
        state = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", 0)))
                state.started = time.monotonic()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", "100000")
                self.end_headers()
                try:
                    for _ in range(400):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.025)
                except OSError:
                    state.hung_up_after = time.monotonic() - state.started
                    state.seen.set()
                self.close_connection = True

            def log_message(self, *_args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.handle_error = lambda *_args: None
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()


@pytest.fixture
def drip(monkeypatch):
    state = Drip()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{state.server.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    yield state
    state.server.shutdown()
    state.server.server_close()


def test_a_dripping_server_cannot_hold_a_call_past_its_deadline(drip):
    started = time.monotonic()
    result = ask(limits=AgentLimits(model_timeout_seconds=0.3))
    elapsed = time.monotonic() - started
    assert elapsed < 1.5  # it would otherwise run for the full 10 seconds of the drip
    assert result.trace.termination is AgentStopReason.MODEL_TIMEOUT
    assert result.trace.turns[0].aborted is True


def test_the_abort_really_closes_the_connection(drip):
    ask(limits=AgentLimits(model_timeout_seconds=0.3))
    assert drip.seen.wait(3.0), "the server never saw the client hang up"
    assert drip.hung_up_after < 2.0


def test_the_worker_thread_does_not_outlive_the_abort(drip):
    ask(limits=AgentLimits(model_timeout_seconds=0.3))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not any(thread.name == "copilot-model-call" for thread in threading.enumerate()):
            return
        time.sleep(0.02)
    pytest.fail("the model-call thread is still alive")


def test_an_aborted_turn_reports_no_usage_because_none_is_known(drip):
    trace = ask(limits=AgentLimits(model_timeout_seconds=0.3)).trace
    assert trace.turns[0].usage is None
    assert trace.usage_complete is False and trace.usage_source is UsageSource.UNAVAILABLE


def test_the_question_deadline_also_cuts_a_call_short(drip):
    started = time.monotonic()
    result = ask(limits=AgentLimits(model_timeout_seconds=5.0, total_timeout_seconds=0.4))
    assert time.monotonic() - started < 1.5
    assert result.trace.termination is AgentStopReason.MODEL_TIMEOUT


def test_a_slow_silent_server_times_out_with_no_retry(local_api):
    local_api.delay = 1.0
    result = ask(limits=AgentLimits(model_timeout_seconds=0.1))
    assert result.trace.termination is AgentStopReason.MODEL_TIMEOUT
    assert result.response.unavailable_reason == "provider_timeout"
    assert len(local_api.requests) == 1


def test_a_closed_port_is_a_typed_connection_failure(monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    result = ask()
    assert result.trace.termination is AgentStopReason.PROVIDER_ERROR
    assert result.trace.error_code is ProviderFailureKind.CONNECTION
    assert result.trace.model_turns == 1


def test_a_connection_cut_mid_body_is_a_typed_failure(monkeypatch):
    class Cut(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", "5000")
            self.end_headers()
            self.wfile.write(b'{"id": "msg')
            self.connection.close()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Cut)
    server.handle_error = lambda *_args: None
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    try:
        result = ask()
    finally:
        server.shutdown()
        server.server_close()
    assert result.trace.termination in {AgentStopReason.PROVIDER_ERROR, AgentStopReason.MALFORMED_RESPONSE}
    assert result.response.support_level.value == "unavailable"
    assert CANARY not in result.trace.model_dump_json()


@pytest.mark.parametrize(
    ("body", "kind"),
    [
        ('{"id": "msg_1", "content": [', ProviderFailureKind.INVALID_RESPONSE),
        ("", ProviderFailureKind.INVALID_RESPONSE),
        ("[1, 2, 3]", ProviderFailureKind.INVALID_RESPONSE),
    ],
)
def test_malformed_json_fails_closed_with_a_typed_kind(local_api, body, kind):
    local_api.body = body
    result = ask()
    assert result.trace.termination is AgentStopReason.MALFORMED_RESPONSE
    assert result.trace.error_code is kind


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (402, ProviderFailureKind.BILLING),
        (403, ProviderFailureKind.PERMISSION),
        (404, ProviderFailureKind.MODEL_UNAVAILABLE),
        (413, ProviderFailureKind.REQUEST_TOO_LARGE),
        (500, ProviderFailureKind.SERVER_ERROR),
        (503, ProviderFailureKind.SERVER_ERROR),
    ],
)
def test_every_status_is_typed_and_never_retried(local_api, status, kind):
    local_api.status = status
    local_api.body = error_body("api_error")
    result = ask()
    assert result.trace.error_code is kind
    assert len(local_api.requests) == 1
    assert CANARY not in result.trace.model_dump_json()


def test_a_rate_limit_is_not_retried_even_when_the_server_invites_it(local_api):
    local_api.status = 429
    local_api.body = error_body("rate_limit_error")
    result = ask()
    assert result.trace.error_code is ProviderFailureKind.RATE_LIMITED
    assert len(local_api.requests) == 1


# -- stop reasons, refusals, and usage ------------------------------------------------------


@pytest.mark.parametrize(
    ("stop", "termination"),
    [
        ("pause_turn", AgentStopReason.MALFORMED_RESPONSE),
        ("stop_sequence", AgentStopReason.MALFORMED_RESPONSE),
        ("a_future_stop_reason", AgentStopReason.MALFORMED_RESPONSE),
        ("max_tokens", AgentStopReason.MODEL_TRUNCATED),
        ("model_context_window_exceeded", AgentStopReason.MODEL_TRUNCATED),
    ],
)
def test_a_stop_reason_outside_the_protocol_runs_no_tool(local_api, stop, termination):
    local_api.body = reply([THINKING, TOOL_USE], stop)
    result = ask()
    assert result.trace.termination is termination
    assert result.trace.tool_calls == ()


def test_a_truncated_tool_call_is_never_run(local_api):
    local_api.body = reply([THINKING, {**TOOL_USE, "input": {}}], "max_tokens")
    result = ask()
    assert result.trace.termination is AgentStopReason.MODEL_TRUNCATED
    assert result.trace.tool_calls == ()


@pytest.mark.parametrize(
    ("category", "recorded"),
    [("cyber", "cyber"), ("general_harms", "general_harms"), ("a_new_category", "other"), (None, None)],
)
def test_a_refusal_is_typed_with_a_closed_category(local_api, category, recorded):
    local_api.body = reply(
        [{"type": "text", "text": "I can't"}],
        "refusal",
        stop_details={"type": "refusal", "category": category, "explanation": f"model text {CANARY}"},
    )
    result = ask()
    assert result.trace.termination is AgentStopReason.MODEL_REFUSAL
    assert result.trace.turns[0].refusal_category == recorded
    assert CANARY not in result.trace.model_dump_json()
    assert "I can't" not in result.response.answer


def test_a_refusal_with_a_partial_tool_call_runs_nothing(local_api):
    local_api.body = reply([THINKING, TOOL_USE], "refusal", stop_details={"type": "refusal", "category": None})
    result = ask()
    assert result.trace.termination is AgentStopReason.MODEL_REFUSAL
    assert result.trace.tool_calls == ()


@pytest.mark.parametrize(
    ("body", "termination"),
    [
        (reply([], "refusal", usage=USAGE), AgentStopReason.MODEL_REFUSAL),
        (reply([{"type": "text", "text": "{"}], "max_tokens", usage=USAGE), AgentStopReason.MODEL_TRUNCATED),
        (reply([{"type": "text", "text": "prose"}], usage=USAGE), AgentStopReason.MALFORMED_RESPONSE),
        (reply([use(STATUS_TOOL)], usage=USAGE), AgentStopReason.MALFORMED_RESPONSE),
    ],
)
def test_a_failed_turn_keeps_its_billed_usage(local_api, body, termination):
    local_api.body = body
    trace = ask().trace
    assert trace.termination is termination
    assert (trace.usage.input_tokens, trace.usage.output_tokens) == (700, 30)
    assert trace.usage_complete is True and trace.usage_source is UsageSource.PROVIDER_REPORTED
    assert trace.estimated_cost_usd > 0


def test_earlier_turns_stay_counted_when_a_later_turn_fails(local_api):
    local_api.queue = [TOOL_TURN, reply([], "refusal")]
    trace = ask().trace
    assert trace.usage.input_tokens == 1600
    assert [turn.stop_reason for turn in trace.turns] == ["tool_use", "refusal"]


def test_a_http_error_has_no_usage_and_says_so(local_api):
    local_api.status = 500
    local_api.body = error_body("api_error")
    trace = ask().trace
    assert trace.usage_complete is False
    assert trace.cost_is_lower_bound is True


@pytest.mark.parametrize(
    "usage",
    [{"input_tokens": -1, "output_tokens": 1}, {"input_tokens": "many", "output_tokens": 1}, {"output_tokens": 1}],
)
def test_malformed_usage_is_unknown_never_zero(local_api, usage):
    local_api.body = reply([{"type": "text", "text": '{"claim_ids": []}'}], usage=usage)
    trace = ask().trace
    assert trace.usage_complete is False


def test_cache_tokens_are_accounted_separately(local_api):
    usage = {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 200}
    local_api.body = reply([{"type": "text", "text": '{"claim_ids": []}'}], usage=usage)
    trace = ask().trace
    assert trace.usage.cache_read_input_tokens == 500
    assert trace.usage.cache_creation_input_tokens == 200


# -- the local route, through everything -------------------------------------------------


@pytest.fixture
def real_route(local_api, tmp_path, monkeypatch):
    for name in ("CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)

    def build(**changes):
        execution = ExecutionSettings(tmp_path / "runs", tmp_path / "queue", max_upload_bytes=1024, allow_test_profiles=True)
        return _app(execution, monkeypatch, replace(READY, **changes))

    return build


def test_the_local_route_drives_the_billed_adapter_when_every_gate_is_open(local_api, real_route, caplog):
    client, run_id = real_route()
    local_api.queue = [TOOL_TURN, final("run_status")]
    assert client.get("/api/v1/copilot/agent").json() == {"adapter": "real_model", "model": "claude-haiku-5-5"}
    with caplog.at_level(logging.INFO, logger="footballai_v2.api"):
        body = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS}).json()
    assert body["trace"]["adapter"] == "real_model"
    assert body["explanation"]["model_mode"] == "real_model"
    assert body["trace"]["termination"] == "answered"
    assert len(local_api.requests) == 2
    logged = "\n".join(record.getMessage() for record in caplog.records if "agent_trace" in record.getMessage())
    assert '"usage_source":"provider_reported"' in logged
    assert STATUS not in logged and CANARY not in logged


def test_token_counts_stay_out_of_what_a_client_sees(local_api, real_route):
    client, run_id = real_route()
    local_api.queue = [TOOL_TURN, final("run_status")]
    text = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS}).text
    assert "input_tokens" not in text and "estimated_cost_usd" not in text and "req_local" not in text


def test_the_route_stops_at_its_request_cap_across_questions(local_api, real_route):
    client, run_id = real_route(agent_max_requests=2)
    local_api.queue = [TOOL_TURN, final("run_status")]
    assert client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS}).json()["trace"]["termination"] == "answered"
    second = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": STATUS}).json()
    assert second["trace"]["termination"] == "budget_exhausted"
    assert second["explanation"]["unavailable_reason"] == "agent_limit_reached"
    assert len(local_api.requests) == 2


def test_an_unsupported_question_costs_nothing_on_the_real_path(local_api, real_route):
    client, run_id = real_route()
    body = client.post(f"/api/v1/runs/{run_id}/agent", json={"question": "How many passes did player 7 make?"}).json()
    assert body["trace"]["termination"] == "unsupported_question"
    assert local_api.requests == []


def test_explain_stays_offline_beside_the_real_agent(local_api, real_route):
    client, run_id = real_route()
    body = client.post(f"/api/v1/runs/{run_id}/explain", json={"question": STATUS}).json()
    assert body["model_mode"] == "offline_demo_fake"
    assert local_api.requests == []


# -- what the protocol must never leak ---------------------------------------------------------


def test_the_real_path_sends_the_key_as_configured_and_never_records_it(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    result = ask()
    assert CANARY not in result.trace.model_dump_json()
    assert CANARY not in result.response.model_dump_json()


def test_a_question_goes_to_the_model_as_data_and_stays_out_of_the_trace(local_api):
    question = "What is the status of this run?"
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    trace = ask(question).trace
    assert json.loads(local_api.requests[0][3]["messages"][0]["content"])["question"] == question
    assert question not in trace.model_dump_json()


def test_the_tool_definitions_on_the_wire_accept_no_run_id(local_api):
    local_api.queue = [TOOL_TURN, FINAL_TURN]
    ask()
    tools = local_api.requests[0][3]["tools"]
    assert {tool["name"] for tool in tools} == {name.value for name in ToolName}
    assert all(tool["input_schema"]["properties"] == {} for tool in tools)
    assert CopilotSettings().max_output_tokens == local_api.requests[0][3]["max_tokens"]
