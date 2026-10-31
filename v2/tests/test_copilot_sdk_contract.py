"""The real provider against the real SDK and a local HTTP server.

This checks what a stub cannot: that the installed SDK serializes our request as
documented, parses the documented response, and raises the classes the provider
classifies. Requests go to 127.0.0.1 only, so nothing is sent to Anthropic and
nothing is billed.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("anthropic")

from footballai_v2.copilot import CopilotSettings
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evaluation.dataset import load_dataset
from footballai_v2.copilot.evaluation.live import (
    PlannedRequest,
    PreflightFailed,
    plan_live_run,
    preflight,
    real_settings,
)
from footballai_v2.copilot.models import TokenUsage
from footballai_v2.copilot.provider import AnthropicTextProvider

from test_copilot_explanation import real_run_env, service_for
from test_copilot_provider_failures import CANARY, SETTINGS, STATUS_QUESTION, request_for


REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": '{"claim_ids": ["run_status"]}'},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 812, "output_tokens": 44, "cache_read_input_tokens": 0},
}
MODEL = {
    "id": "claude-haiku-5-5",
    "type": "model",
    "display_name": "Claude Haiku 5.5",
    "created_at": "2026-10-07T00:00:00Z",
    "lifecycle": "active",
    "capabilities": {
        "structured_outputs": {"supported": True},
        "effort": {"supported": True, "low": {"supported": True}},
    },
}


class LocalAnthropic:
    """Scripted responses per path, with every received request recorded."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict, dict]] = []
        self.status = 200
        self.body: object = REPLY
        self.delay = 0.0
        self.routes: dict[str, tuple[int, object]] = {}


def _handler(state: LocalAnthropic):
    class Handler(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            length = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            state.requests.append((self.command, self.path, dict(self.headers), json.loads(raw)))
            time.sleep(state.delay)
            status, body = state.status, state.body
            for prefix, route in state.routes.items():
                if self.path.startswith(prefix):
                    status, body = route
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("request-id", "req_local")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:  # the client timed out and hung up
                pass

        do_GET = do_POST = _respond

        def log_message(self, *_args) -> None:
            pass

    return Handler


@pytest.fixture
def local_api(monkeypatch):
    state = LocalAnthropic()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    yield state
    server.shutdown()
    server.server_close()


def error_body(kind: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": f"upstream said {CANARY}"}, "request_id": "req_err"}


def test_request_on_the_wire_matches_the_api_contract(local_api):
    AnthropicTextProvider(SETTINGS).generate(request_for(real_run_env()))
    method, path, headers, body = local_api.requests[0]
    lowered = {key.lower(): value for key, value in headers.items()}
    assert (method, path) == ("POST", "/v1/messages")
    # SDK 2.x sends the key as a bearer token, not in X-Api-Key (migration guide
    # for API version 2026-10-21). A proxy reading X-Api-Key would need changing.
    assert lowered["authorization"] == f"Bearer {CANARY}"
    assert "x-api-key" not in lowered
    assert lowered["anthropic-version"] == "2026-10-21"
    assert "anthropic-beta" not in lowered  # structured outputs are generally available
    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == 1024
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "stream" not in body or body["stream"] is False
    assert not {"temperature", "top_p", "top_k", "output_format"} & set(body)


def test_documented_reply_parses_to_a_selection_with_usage(local_api):
    result = AnthropicTextProvider(SETTINGS).generate(request_for(real_run_env()))
    assert result.claim_ids == ("run_status",)
    assert result.telemetry.usage == TokenUsage(812, 44)
    assert result.telemetry.request_id == "req_local"
    assert result.telemetry.stop_reason == "end_turn"


def test_null_cache_counters_from_the_wire_are_zero(local_api):
    local_api.body = {**REPLY, "usage": {"input_tokens": 5, "output_tokens": 6, "cache_creation_input_tokens": None}}
    assert AnthropicTextProvider(SETTINGS).generate(request_for(real_run_env())).telemetry.usage == TokenUsage(5, 6)


@pytest.mark.parametrize(
    ("status", "error_type", "kind"),
    [
        (400, "invalid_request_error", ProviderFailureKind.BAD_REQUEST),
        (401, "authentication_error", ProviderFailureKind.AUTHENTICATION),
        (403, "permission_error", ProviderFailureKind.PERMISSION),
        (404, "not_found_error", ProviderFailureKind.MODEL_UNAVAILABLE),
        (429, "rate_limit_error", ProviderFailureKind.RATE_LIMITED),
        (500, "api_error", ProviderFailureKind.SERVER_ERROR),
        (529, "overloaded_error", ProviderFailureKind.SERVER_ERROR),
    ],
)
def test_real_sdk_errors_are_classified_without_retry_or_leak(local_api, status, error_type, kind):
    local_api.status = status
    local_api.body = error_body(error_type)
    with pytest.raises(ProviderUnavailableError) as caught:
        AnthropicTextProvider(SETTINGS).generate(request_for(real_run_env()))
    assert caught.value.kind is kind
    assert CANARY not in str(caught.value)
    assert caught.value.telemetry.request_id in {"req_local", "req_err"}
    assert len(local_api.requests) == 1  # a hidden retry would be a hidden charge


def test_real_sdk_timeout_is_classified(local_api):
    local_api.delay = 1.0
    settings = CopilotSettings(provider_mode="real", real_provider_enabled=True, timeout_seconds=0.1)
    repository, storage, run = real_run_env()
    response = service_for(repository, storage, provider=AnthropicTextProvider(settings), settings=settings).explain(
        run.run_id, STATUS_QUESTION
    )
    assert response.unavailable_reason == "provider_timeout"
    with pytest.raises(ProviderTimeoutError):
        AnthropicTextProvider(settings).generate(replace(request_for(real_run_env()), timeout_seconds=0.1))
    assert len(local_api.requests) == 2  # one per call: a timeout is not retried


def test_real_sdk_connection_failure_is_classified(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # the discard port: nothing listens
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    with pytest.raises(ProviderUnavailableError) as caught:
        AnthropicTextProvider(SETTINGS).generate(request_for(real_run_env()))
    assert caught.value.kind is ProviderFailureKind.CONNECTION


@pytest.mark.parametrize(
    "body",
    [
        "not json at all",
        {**REPLY, "content": None},
        {**REPLY, "content": "plain text"},
        {**REPLY, "content": [{"type": "text", "text": "Player 7 ran 10.5 km"}]},
        {**REPLY, "stop_reason": None},
        {"unexpected": True},
    ],
)
def test_malformed_successful_responses_fail_closed_with_the_real_sdk(local_api, body):
    local_api.body = body
    repository, storage, run = real_run_env()
    response = service_for(repository, storage, provider=AnthropicTextProvider(SETTINGS)).explain(
        run.run_id, STATUS_QUESTION
    )
    assert response.support_level.value == "unavailable"
    assert response.unavailable_reason == "provider_unavailable"
    assert "10.5" not in response.answer


def test_preflight_uses_only_the_unbilled_endpoints(local_api):
    plan = _plan()
    local_api.routes = {"/v1/models": (200, MODEL), "/v1/messages/count_tokens": (200, {"input_tokens": 777})}
    report = preflight(plan, SETTINGS)
    assert report["lifecycle"] == "active"
    assert report["input_tokens_total"] == 777
    paths = [(method, path.split("?")[0]) for method, path, _, _ in local_api.requests]
    assert paths == [("GET", "/v1/models/claude-haiku-5-5"), ("POST", "/v1/messages/count_tokens")]
    assert "max_tokens" not in local_api.requests[1][3]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lifecycle": "retired"}, "lifecycle"),
        ({"capabilities": None}, "structured output"),
        ({"capabilities": {"structured_outputs": {"supported": False}}}, "structured output"),
        ({"capabilities": {"structured_outputs": {"supported": True}, "effort": {"supported": False}}}, "effort"),
    ],
)
def test_preflight_refuses_an_unsuitable_model(local_api, change, message):
    local_api.routes = {"/v1/models": (200, {**MODEL, **change})}
    with pytest.raises(PreflightFailed, match=message):
        preflight(_plan(), SETTINGS)


def test_preflight_reports_an_unavailable_model_without_upstream_text(local_api):
    local_api.routes = {"/v1/models": (404, error_body("not_found_error"))}
    with pytest.raises(PreflightFailed) as caught:
        preflight(_plan(), SETTINGS)
    assert CANARY not in str(caught.value)


def _plan() -> tuple[PlannedRequest, ...]:
    dataset = load_dataset(Path(__file__).resolve().parents[2] / "evaluation" / "copilot" / "v1")
    return plan_live_run(dataset, real_settings(SETTINGS.model_name), sample=1)
