"""Providers used only by the evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from footballai_v2.copilot.errors import ProviderFailure
from footballai_v2.copilot.evaluation.dataset import EvaluationCase
from footballai_v2.copilot.models import ModelMode, ProviderTelemetry
from footballai_v2.copilot.provider import GenerationRequest, LLMProvider, ProviderResult, parse_selection


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    "a about all an and are as at be did do does for how i in is it many me much my of on or say show "
    "tell that the there this to was were what which with".split()
)


def _stems(text: str) -> frozenset[str]:
    return frozenset(word.rstrip("s") for word in _WORD_RE.findall(text.lower()) if word not in _STOP_WORDS)


class LexicalOverlapProvider:
    """A deliberately simple rule: select the claims sharing a word with the question.

    A reference point, not a product feature. Beating "select everything" is a
    low bar, so a model should also have to beat a few lines of deterministic code.
    """

    mode = ModelMode.OFFLINE_DEMO_FAKE

    def generate(self, request: GenerationRequest) -> ProviderResult:
        question = _stems(request.question or "")
        if not question:  # a blank question is a general explanation
            return ProviderResult(tuple(claim.claim_id for claim in request.catalog), self.mode)
        selected = tuple(
            claim.claim_id
            for claim in request.catalog
            if question & _stems(f"{claim.text} {claim.claim_id.replace('_', ' ')}")
        )
        return ProviderResult(selected, self.mode)


class ScriptedProvider:
    """Replays one contract case's raw reply through the real selection parser."""

    mode = ModelMode.REAL_MODEL

    def __init__(self, case: EvaluationCase) -> None:
        self._text = case.model_output or ""

    def generate(self, request: GenerationRequest) -> ProviderResult:
        return ProviderResult(parse_selection(self._text), self.mode)


@dataclass(slots=True)
class RecordingProvider:
    """Wraps a provider and keeps what one case's call did, success or failure."""

    inner: LLMProvider
    request: GenerationRequest | None = None
    result: ProviderResult | None = None
    failure: ProviderFailure | None = None
    calls: int = field(default=0)

    @property
    def mode(self) -> ModelMode:
        return self.inner.mode

    def generate(self, request: GenerationRequest) -> ProviderResult:
        self.calls += 1
        self.request = request
        try:
            self.result = self.inner.generate(request)
        except ProviderFailure as exc:
            self.failure = exc
            raise
        return self.result

    @property
    def telemetry(self) -> ProviderTelemetry | None:
        if self.result is not None:
            return self.result.telemetry
        return self.failure.telemetry if self.failure is not None else None
