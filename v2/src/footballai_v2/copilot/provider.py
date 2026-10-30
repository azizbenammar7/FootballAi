"""LLM provider abstraction: a deterministic fake and an opt-in real model.

A provider's only job is to SELECT approved claim ids (and their order) from the
server-built catalog: deciding which verified facts answer the question. It
never produces factual prose or citations, so it cannot fabricate a source or
publish an unverified statistic.

The fake provider is the default: deterministic, offline, and always reporting
``model_mode = offline_demo_fake`` so it can never be mistaken for real
inference. The real provider is disabled unless explicitly configured, imports
its SDK lazily (so normal installs and CI never touch it), is bounded by a token
limit and a timeout, and is fed a bounded structured payload that always
contains the run's safety fields. If that minimal safe payload does not fit the
prompt budget, the provider fails closed instead of silently truncating JSON.
Credentials stay server-side and are never logged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import (
    PromptBudgetExceededError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from footballai_v2.copilot.evidence import ValidatedRunEvidence
from footballai_v2.copilot.grounding import EvidenceClaim
from footballai_v2.copilot.models import ModelMode


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


class AnthropicTextProvider:
    """Opt-in real provider. Selects claim ids only; bounded tokens and timeout."""

    mode = ModelMode.REAL_MODEL

    def __init__(self, settings: CopilotSettings) -> None:
        self._settings = settings

    def generate(self, request: GenerationRequest) -> ProviderResult:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ProviderUnavailableError("ANTHROPIC_API_KEY is not configured for the real provider")

        payload = build_provider_payload(request.evidence, request.catalog, request.question)
        user = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        # Validate the exact prompt size BEFORE calling the model. Never slice
        # JSON: if the minimal safe evidence does not fit, fail closed.
        prompt_chars = len(user) + len(request.system_instructions)
        if len(user) > self._settings.max_evidence_bytes or prompt_chars > self._settings.max_prompt_chars:
            raise PromptBudgetExceededError("the safe evidence payload exceeds the prompt budget")

        try:
            import anthropic  # lazy: never imported by the default install or CI
        except ImportError as exc:
            raise ProviderUnavailableError("the anthropic SDK is not installed") from exc

        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=request.timeout_seconds,
            max_retries=self._settings.provider_retries,  # the SDK would otherwise retry twice
        )
        try:
            message = client.messages.create(
                model=self._settings.model_name,
                max_tokens=request.max_output_tokens,
                system=request.system_instructions,
                messages=[{"role": "user", "content": user}],
                output_config={"format": _selection_format(request.catalog)},
            )
        except Exception as exc:  # SDK timeout/transport -> bounded, typed failure
            name = type(exc).__name__.lower()
            if "timeout" in name:
                raise ProviderTimeoutError("the real provider timed out") from exc
            raise ProviderUnavailableError("the real provider call failed") from exc

        if message.stop_reason != "end_turn":  # refusal or truncation: not a usable selection
            raise ProviderUnavailableError("the real provider did not finish its selection")
        parts = [block.text for block in message.content if getattr(block, "type", None) == "text"]
        text = " ".join(part.strip() for part in parts if part).strip()
        return ProviderResult(claim_ids=_parse_claim_ids(text), mode=self.mode)


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


def _parse_claim_ids(text: str) -> tuple[str, ...]:
    """Strictly parse a ``{"claim_ids": [...]}`` selection; fail closed otherwise."""
    if not text:
        raise ProviderUnavailableError("the real provider returned no selection")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ProviderUnavailableError("the real provider returned a non-JSON selection") from exc
    if not isinstance(data, dict) or set(data) != {"claim_ids"}:
        raise ProviderUnavailableError("the real provider selection was not the expected object")
    raw = data["claim_ids"]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ProviderUnavailableError("the real provider selection had no valid claim_ids list")
    return tuple(raw)


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
    "build_provider",
    "build_provider_payload",
]
