"""LLM provider abstraction: a deterministic fake and an opt-in real model.

A provider's only job is to SELECT approved claim ids (and their order) from the
server-built catalog: deciding which verified facts answer the question. It
never produces factual prose or citations, so it cannot fabricate a source or
publish an unverified statistic.

The fake provider is the default: deterministic, offline, and always reporting
``model_mode = offline_demo_fake`` so it can never be mistaken for real
inference. The real provider is disabled unless explicitly configured, imports
its SDK lazily (so the default install never needs it), is bounded by a token
limit and a timeout, and is fed a bounded structured payload that always
contains the run's safety fields. If that minimal safe payload does not fit the
prompt budget, the provider fails closed instead of silently truncating JSON.
Credentials stay server-side and are never logged.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import ValidationError

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import (
    PromptBudgetExceededError,
    ProviderFailure,
    ProviderFailureKind,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from footballai_v2.copilot.evidence import ValidatedRunEvidence
from footballai_v2.copilot.grounding import EvidenceClaim
from footballai_v2.copilot.models import ClaimSelection, ModelMode, ProviderTelemetry, TokenUsage


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Everything a provider may use. The real provider sees only the bounded
    structured payload built from the catalog + safety fields, never raw run data."""

    evidence: ValidatedRunEvidence
    question: str | None
    system_instructions: str
    catalog: tuple[EvidenceClaim, ...]
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """A provider's selection of approved claim ids, in presentation order."""

    claim_ids: tuple[str, ...]
    mode: ModelMode
    telemetry: ProviderTelemetry | None = None


@runtime_checkable
class LLMProvider(Protocol):
    mode: ModelMode

    def generate(self, request: GenerationRequest) -> ProviderResult: ...


class DeterministicFakeProvider:
    """Offline, deterministic claim selector. The safe default."""

    mode = ModelMode.OFFLINE_DEMO_FAKE

    def generate(self, request: GenerationRequest) -> ProviderResult:
        # Surface every claim offered for the question, in catalog order.
        return ProviderResult(
            claim_ids=tuple(claim.claim_id for claim in request.catalog),
            mode=self.mode,
        )


def build_provider_payload(
    evidence: ValidatedRunEvidence, catalog: tuple[EvidenceClaim, ...], question: str | None
) -> dict[str, Any]:
    """A bounded, structured payload for the real model.

    Safety fields are ALWAYS present and are never dropped to make room. Only
    server-authored claims and allow-listed fields are included: no raw snapshot,
    artifact free text, input URI, or storage path.
    """
    return {
        "pipeline_profile": evidence.pipeline_profile,
        "output_nature": evidence.output_nature.value,
        "is_synthetic": evidence.is_synthetic,
        "verification": {
            "integrity_verified_diagnostics": evidence.diagnostics is not None,
            "diagnostics_schema": evidence.diagnostics_schema,
        },
        "limitations": list(evidence.limitations),
        "missing_evidence": list(evidence.missing_evidence),
        "claims": [{"id": claim.claim_id, "text": claim.text} for claim in catalog],
        "question": (question or "").strip() or "Explain this run.",
    }


def build_model_request(request: GenerationRequest, settings: CopilotSettings) -> dict[str, Any]:
    """The exact ``messages.create`` arguments for ``request``, budget-checked.

    Validates the prompt size BEFORE any call. Never slices JSON: if the minimal
    safe evidence does not fit, this fails closed.
    """
    payload = build_provider_payload(request.evidence, request.catalog, request.question)
    user = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    prompt_chars = len(user) + len(request.system_instructions)
    if len(user) > settings.max_evidence_bytes or prompt_chars > settings.max_prompt_chars:
        raise PromptBudgetExceededError("the safe evidence payload exceeds the prompt budget")
    return {
        "model": settings.model_name,
        "max_tokens": request.max_output_tokens,
        "system": request.system_instructions,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": settings.effort,
            "format": _selection_format(tuple(claim.claim_id for claim in request.catalog)),
        },
    }


class AnthropicTextProvider:
    """Opt-in real provider. Selects claim ids only; bounded tokens and timeout."""

    mode = ModelMode.REAL_MODEL

    def __init__(self, settings: CopilotSettings) -> None:
        self._settings = settings

    def generate(self, request: GenerationRequest) -> ProviderResult:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ProviderUnavailableError(
                "ANTHROPIC_API_KEY is not configured for the real provider",
                kind=ProviderFailureKind.NOT_CONFIGURED,
            )

        arguments = build_model_request(request, self._settings)

        try:
            import anthropic  # lazy: the default install never needs it
        except ImportError as exc:
            raise ProviderUnavailableError(
                "the anthropic SDK is not installed", kind=ProviderFailureKind.NOT_CONFIGURED
            ) from exc

        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=request.timeout_seconds,
            max_retries=self._settings.provider_retries,  # the SDK would otherwise retry twice
        )
        started = time.perf_counter()
        try:
            message = client.messages.create(**arguments)
        except Exception as exc:  # classified below; SDK text never leaves this function
            raise _sdk_failure(anthropic, exc, self._settings.model_name, started) from exc

        telemetry = _telemetry(message, self._settings.model_name, started)
        if message.stop_reason != "end_turn":  # refusal or truncation: not a usable selection
            raise ProviderUnavailableError(
                "the real provider did not finish its selection",
                kind=_stop_reason_kind(message.stop_reason),
                telemetry=telemetry,
            )
        return ProviderResult(
            claim_ids=parse_selection(_selection_text(message, telemetry), telemetry=telemetry),
            mode=self.mode,
            telemetry=telemetry,
        )


_STOP_REASON_KINDS = {
    "refusal": ProviderFailureKind.REFUSAL,
    "max_tokens": ProviderFailureKind.TRUNCATED,
    "model_context_window_exceeded": ProviderFailureKind.TRUNCATED,
}
_REFUSAL_CATEGORIES = ("bio", "cyber", "frontier_llm", "general_harms", "reasoning_extraction")
_STATUS_KINDS = {
    400: ProviderFailureKind.BAD_REQUEST,
    401: ProviderFailureKind.AUTHENTICATION,
    402: ProviderFailureKind.BILLING,
    403: ProviderFailureKind.PERMISSION,
    404: ProviderFailureKind.MODEL_UNAVAILABLE,
    413: ProviderFailureKind.REQUEST_TOO_LARGE,
    429: ProviderFailureKind.RATE_LIMITED,
}


def _stop_reason_kind(stop_reason: Any) -> ProviderFailureKind:
    # The SDK does not validate responses, so this value may not even be hashable.
    if isinstance(stop_reason, str):
        return _STOP_REASON_KINDS.get(stop_reason, ProviderFailureKind.INVALID_RESPONSE)
    return ProviderFailureKind.INVALID_RESPONSE


def _sdk_failure(sdk: Any, exc: Exception, model: str, started: float) -> ProviderFailure:
    """Map an SDK exception to a typed failure. Its text may hold upstream content."""
    telemetry = ProviderTelemetry(
        model=model,
        latency_seconds=time.perf_counter() - started,
        request_id=getattr(exc, "request_id", None),
    )
    if isinstance(exc, getattr(sdk, "APITimeoutError", ())):
        return ProviderTimeoutError("the real provider timed out", telemetry=telemetry)
    status = getattr(exc, "status_code", None)
    # The SDK lets a JSON error through when a successful response has an unreadable body.
    if isinstance(exc, (json.JSONDecodeError, getattr(sdk, "APIResponseValidationError", ()))):
        kind = ProviderFailureKind.INVALID_RESPONSE
    elif isinstance(status, int):
        fallback = ProviderFailureKind.SERVER_ERROR if status >= 500 else ProviderFailureKind.UNKNOWN
        kind = _STATUS_KINDS.get(status, fallback)
    elif isinstance(exc, getattr(sdk, "APIConnectionError", ())):
        kind = ProviderFailureKind.CONNECTION
    else:
        kind = ProviderFailureKind.UNKNOWN
    return ProviderUnavailableError("the real provider call failed", kind=kind, telemetry=telemetry)


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(message: Any) -> TokenUsage | None:
    """Read the reported usage, or ``None`` if any reported count is malformed."""
    usage = getattr(message, "usage", None)
    input_tokens = _token_count(getattr(usage, "input_tokens", None))
    output_tokens = _token_count(getattr(usage, "output_tokens", None))
    # The cache counters are nullable in the API: absent means zero, not unknown.
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    counts = (input_tokens, output_tokens, _token_count(cache_write), _token_count(cache_read))
    if any(count is None for count in counts):
        return None
    return TokenUsage(input_tokens, output_tokens, cache_write, cache_read)


def _refusal_category(message: Any) -> str | None:
    """The refusal's category, on a closed set. ``explanation`` is model text and stays out."""
    if getattr(message, "stop_reason", None) != "refusal":
        return None
    category = getattr(getattr(message, "stop_details", None), "category", None)
    if category is None:
        return None
    return category if category in _REFUSAL_CATEGORIES else "other"


def _telemetry(message: Any, model: str, started: float) -> ProviderTelemetry:
    return ProviderTelemetry(
        model=model,
        latency_seconds=time.perf_counter() - started,
        request_id=getattr(message, "_request_id", None),
        stop_reason=getattr(message, "stop_reason", None),
        usage=_usage(message),
        refusal_category=_refusal_category(message),
    )


def _selection_text(message: Any, telemetry: ProviderTelemetry) -> str:
    """The model's text blocks, read by type: a reply can start with thinking."""
    content = getattr(message, "content", None)
    if not isinstance(content, (list, tuple)):  # the SDK does not validate response shape
        raise ProviderUnavailableError(
            "the real provider returned no content",
            kind=ProviderFailureKind.INVALID_RESPONSE,
            telemetry=telemetry,
        )
    parts = [getattr(block, "text", None) for block in content if getattr(block, "type", None) == "text"]
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part).strip()


def _selection_format(claim_ids: tuple[str, ...]) -> dict[str, Any]:
    """Structured-output schema limiting the model to the ids it was offered."""
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "claim_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(claim_ids)},
                }
            },
            "required": ["claim_ids"],
            "additionalProperties": False,
        },
    }


def parse_selection(text: str, *, telemetry: ProviderTelemetry | None = None) -> tuple[str, ...]:
    """Strictly parse a ``{"claim_ids": [...]}`` selection; fail closed otherwise."""
    if not text:
        raise ProviderUnavailableError(
            "the real provider returned no selection",
            kind=ProviderFailureKind.INVALID_RESPONSE,
            telemetry=telemetry,
        )
    try:
        selection = ClaimSelection.model_validate_json(text)
    except ValidationError:
        # No exception chaining: the validation error quotes the model's text.
        raise ProviderUnavailableError(
            "the real provider returned an invalid selection",
            kind=ProviderFailureKind.INVALID_RESPONSE,
            telemetry=telemetry,
        ) from None
    return tuple(selection.claim_ids)


def build_provider(settings: CopilotSettings) -> LLMProvider:
    """Choose a provider from settings; the real one only on an explicit opt-in.

    The real agent's settings would also select the real provider here, and the
    explanation route has no cost guard or loopback guard of its own.
    """
    if settings.wants_real_provider and not settings.agent_real_model_enabled:
        return AnthropicTextProvider(settings)
    return DeterministicFakeProvider()


# -- agent turns ----------------------------------------------------------------


_TOOL_USE_ID_RE = re.compile(r"[a-zA-Z0-9_-]{1,128}")
_ANTHROPIC_API = "https://api.anthropic.com"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_ABORT_THREAD = "copilot-model-call"


def api_base_url(environ: Mapping[str, str] | None = None) -> str | None:
    """Where the agent may send a key: Anthropic, or a local test server. Otherwise ``None``.

    The SDK follows ``ANTHROPIC_BASE_URL`` silently, so an inherited value would send the
    key and every prompt to a third party, and any measurement would describe that
    endpoint. A loopback address exists for the contract tests.
    """
    configured = (os.environ if environ is None else environ).get("ANTHROPIC_BASE_URL", "").strip()
    if not configured:
        return _ANTHROPIC_API
    try:
        parts = urlsplit(configured)
        host = parts.hostname
    except ValueError:
        return None
    if (parts.scheme == "https" and host == "api.anthropic.com") or (parts.scheme == "http" and host in _LOOPBACK_HOSTS):
        return configured
    return None


@dataclass(frozen=True, slots=True)
class AgentModelRequest:
    """One model turn. ``messages`` only ever grows, and tools take no run id."""

    system_instructions: str
    tools: tuple[Mapping[str, Any], ...]
    messages: tuple[Mapping[str, Any], ...]
    offered_claim_ids: tuple[str, ...]
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the model asked for. Name and arguments are untrusted until validated."""

    call_id: str
    name: str
    arguments: Any


@dataclass(frozen=True, slots=True)
class AgentReply:
    """One parsed model turn: tool calls, or a final selection of claim ids.

    ``assistant_content`` is the model's content exactly as received. It goes back
    unchanged on the next turn, because thinking blocks must not be edited, and it
    is never shown to anyone.
    """

    assistant_content: Any
    tool_calls: tuple[ToolCall, ...] = ()
    claim_ids: tuple[str, ...] = ()
    telemetry: ProviderTelemetry | None = None


@runtime_checkable
class AgentModel(Protocol):
    """What the bounded loop needs from a model. Failures raise ``ProviderFailure``."""

    mode: ModelMode
    model_name: str

    def complete(self, request: AgentModelRequest) -> AgentReply: ...


def build_agent_model_request(request: AgentModelRequest, settings: CopilotSettings) -> dict[str, Any]:
    """The exact ``messages.create`` arguments for one agent turn."""
    return {
        "model": settings.model_name,
        "max_tokens": request.max_output_tokens,
        "system": request.system_instructions,
        "messages": list(request.messages),
        "tools": list(request.tools),
        "tool_choice": {"type": "auto"},
        "output_config": {"effort": settings.effort, "format": _selection_format(request.offered_claim_ids)},
    }


def _invalid_reply(telemetry: ProviderTelemetry | None) -> ProviderUnavailableError:
    return ProviderUnavailableError(
        "the model reply did not match the agent protocol",
        kind=ProviderFailureKind.INVALID_RESPONSE,
        telemetry=telemetry,
    )


def _tool_calls(blocks: list[Any], telemetry: ProviderTelemetry | None) -> tuple[ToolCall, ...]:
    calls = tuple(
        ToolCall(getattr(block, "id", None), getattr(block, "name", None), getattr(block, "input", None))
        for block in blocks
    )
    valid = all(
        isinstance(call.call_id, str) and _TOOL_USE_ID_RE.fullmatch(call.call_id) and isinstance(call.name, str)
        for call in calls
    )
    if not valid or len({call.call_id for call in calls}) != len(calls):
        raise _invalid_reply(telemetry)
    return calls


def parse_agent_message(message: Any, telemetry: ProviderTelemetry | None = None) -> AgentReply:
    """Read one Messages API response, failing closed on anything unexpected.

    Reading by block type keeps thinking and any text out of the decision: only a
    ``tool_use`` block or the final structured selection can have an effect.
    """
    stop_reason = getattr(message, "stop_reason", None)
    # A tuple: the SDK does not validate responses, so this value may not be hashable.
    if stop_reason not in ("tool_use", "end_turn"):  # no server tools, so pause_turn is invalid too
        raise ProviderUnavailableError(
            "the model did not finish its turn", kind=_stop_reason_kind(stop_reason), telemetry=telemetry
        )
    content = getattr(message, "content", None)
    if not isinstance(content, (list, tuple)):
        raise _invalid_reply(telemetry)
    blocks = [block for block in content if getattr(block, "type", None) == "tool_use"]
    if stop_reason == "tool_use":
        if not blocks:
            raise _invalid_reply(telemetry)
        return AgentReply(content, tool_calls=_tool_calls(blocks, telemetry), telemetry=telemetry)
    if blocks:
        raise _invalid_reply(telemetry)
    claim_ids = parse_selection(_selection_text(message, telemetry), telemetry=telemetry)
    return AgentReply(content, claim_ids=claim_ids, telemetry=telemetry)


class AnthropicAgentModel:
    """Opt-in real model for the agent loop. One call per turn, never retried.

    The SDK's timeout bounds each network read, not a call: a server sending one byte
    at a time held a 0.3 second timeout for 11 seconds in a local test. So the call runs
    on a worker thread, and at its deadline the caller gives up and closes the client.
    Tests against a dripping local server show the caller returns on time, the
    connection is closed, and the thread ends. That stops this process waiting. It does
    not prove that Anthropic stops generating, or that the request is not billed, so an
    aborted turn is charged at its worst case.
    """

    mode = ModelMode.REAL_MODEL

    def __init__(self, settings: CopilotSettings) -> None:
        self._settings = settings
        self.model_name = settings.model_name
        self._client: Any = None  # one client serves every turn of a question

    def complete(self, request: AgentModelRequest) -> AgentReply:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        base_url = api_base_url()
        if not self._settings.wants_real_provider or not api_key or base_url is None:
            raise ProviderUnavailableError(
                "the real model is not enabled for the agent", kind=ProviderFailureKind.NOT_CONFIGURED
            )
        try:
            import anthropic  # lazy: the default install never needs it
        except ImportError as exc:
            raise ProviderUnavailableError(
                "the anthropic SDK is not installed", kind=ProviderFailureKind.NOT_CONFIGURED
            ) from exc

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)
        started = time.perf_counter()
        arguments = build_agent_model_request(request, self._settings)
        outcome: dict[str, Any] = {}

        def call(client: Any = self._client) -> None:
            try:
                outcome["message"] = client.messages.create(**arguments, timeout=request.timeout_seconds)
            except Exception as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=call, name=_ABORT_THREAD, daemon=True)
        worker.start()
        worker.join(request.timeout_seconds)
        if worker.is_alive():
            self.close()
            raise ProviderTimeoutError(
                "the real provider passed its deadline",
                telemetry=ProviderTelemetry(
                    model=self._settings.model_name,
                    latency_seconds=time.perf_counter() - started,
                    aborted=True,
                ),
            )
        if "error" in outcome:  # classified below; SDK text never leaves this function
            raise _sdk_failure(anthropic, outcome["error"], self._settings.model_name, started) from outcome["error"]
        message = outcome["message"]
        return parse_agent_message(message, _telemetry(message, self._settings.model_name, started))

    def close(self) -> None:
        """Close the connection pool. An in-flight read on another thread fails at once."""
        client, self._client = self._client, None
        if client is not None:
            client.close()


__all__ = [
    "AgentModel",
    "AgentModelRequest",
    "AgentReply",
    "AnthropicAgentModel",
    "AnthropicTextProvider",
    "DeterministicFakeProvider",
    "GenerationRequest",
    "LLMProvider",
    "ProviderResult",
    "ToolCall",
    "api_base_url",
    "build_agent_model_request",
    "build_model_request",
    "build_provider",
    "build_provider_payload",
    "parse_agent_message",
    "parse_selection",
]
