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
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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
        "output_config": {"effort": settings.effort, "format": _selection_format(request.catalog)},
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
    if isinstance(exc, getattr(sdk, "APIResponseValidationError", ())):
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


def _telemetry(message: Any, model: str, started: float) -> ProviderTelemetry:
    return ProviderTelemetry(
        model=model,
        latency_seconds=time.perf_counter() - started,
        request_id=getattr(message, "_request_id", None),
        stop_reason=getattr(message, "stop_reason", None),
        usage=_usage(message),
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


def _selection_format(catalog: tuple[EvidenceClaim, ...]) -> dict[str, Any]:
    """Structured-output schema limiting the model to the ids it was offered."""
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "claim_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": [claim.claim_id for claim in catalog]},
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
    """Choose a provider from settings; the real one only on an explicit opt-in."""
    if settings.wants_real_provider:
        return AnthropicTextProvider(settings)
    return DeterministicFakeProvider()


__all__ = [
    "AnthropicTextProvider",
    "DeterministicFakeProvider",
    "GenerationRequest",
    "LLMProvider",
    "ProviderResult",
    "build_model_request",
    "build_provider",
    "build_provider_payload",
    "parse_selection",
]
