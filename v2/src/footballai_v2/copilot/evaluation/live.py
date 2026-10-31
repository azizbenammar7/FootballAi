"""Planning, preflight, and budget-bounded execution of a real-model evaluation.

``plan_live_run`` needs no SDK, key, or network. ``preflight`` needs a key but
sends no ``messages.create``, so it is not billed. Only ``run_live`` can spend
money, and it calls the model through ``BudgetedProvider``, which refuses any
request the budget cannot cover at its worst case.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ProviderFailure, ProviderFailureKind
from footballai_v2.copilot.evaluation.cost import (
    ModelPrice,
    estimate_input_tokens,
    estimate_output_tokens,
    request_cost_usd,
    worst_case_cost_usd,
)
from footballai_v2.copilot.evaluation.dataset import EvaluationCase, EvaluationDataset, Suite
from footballai_v2.copilot.evaluation.harness import CaseResult, run_case
from footballai_v2.copilot.evaluation.selectors import RecordingProvider
from footballai_v2.copilot.models import ProviderTelemetry
from footballai_v2.copilot.provider import (
    AnthropicTextProvider,
    DeterministicFakeProvider,
    GenerationRequest,
    ProviderResult,
    build_model_request,
)


# Failures that will repeat for every remaining case: stop rather than spend.
_FATAL_KINDS = frozenset(
    {
        ProviderFailureKind.NOT_CONFIGURED,
        ProviderFailureKind.AUTHENTICATION,
        ProviderFailureKind.PERMISSION,
        ProviderFailureKind.BILLING,
        ProviderFailureKind.MODEL_UNAVAILABLE,
        ProviderFailureKind.BAD_REQUEST,
        ProviderFailureKind.RATE_LIMITED,
    }
)
MAX_CONSECUTIVE_FAILURES = 3


class BudgetExhausted(Exception):
    """Raised, before a request is sent, when the budget cannot cover it."""

    def __init__(self, limit: str) -> None:
        super().__init__(limit)
        self.limit = limit


class PreflightFailed(Exception):
    """Raised when the account, SDK, or model cannot support the planned run."""


def _prompt_chars(arguments: dict[str, Any]) -> int:
    return len(arguments["system"]) + len(arguments["messages"][0]["content"])


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    case: EvaluationCase
    arguments: dict[str, Any]
    catalog_ids: tuple[str, ...]

    @property
    def prompt_chars(self) -> int:
        return _prompt_chars(self.arguments)

    @property
    def selection_chars(self) -> int:
        expected = self.case.expected
        return len(json.dumps({"claim_ids": list(expected.permitted_claims or self.catalog_ids)}))


def real_settings(model: str) -> CopilotSettings:
    return CopilotSettings(
        enabled=True,
        environment="test",
        provider_mode="real",
        real_provider_enabled=True,
        model_name=model,
    )


def _sample(cases: list[EvaluationCase], limit: int | None) -> list[EvaluationCase]:
    """Stratified and deterministic: take from each category in turn, in hash order."""
    if limit is None or limit >= len(cases):
        return cases
    groups: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        groups.setdefault(case.category, []).append(case)
    for group in groups.values():
        group.sort(key=lambda case: hashlib.sha256(case.case_id.encode()).hexdigest())
    chosen: list[EvaluationCase] = []
    while len(chosen) < limit:
        for category in sorted(groups):
            if groups[category] and len(chosen) < limit:
                chosen.append(groups[category].pop(0))
    return chosen


def plan_live_run(
    dataset: EvaluationDataset, settings: CopilotSettings, sample: int | None = None
) -> tuple[PlannedRequest, ...]:
    """The exact requests a live run would send, found by running the fake.

    Only selection cases that reach a provider are planned: a scripted contract
    case or a case the router stops never costs a request.
    """
    reachable: dict[str, PlannedRequest] = {}
    for case in dataset.suite(Suite.SELECTION):
        recorder = RecordingProvider(DeterministicFakeProvider())
        run_case(case, dataset, recorder, settings)
        if recorder.request is not None:
            reachable[case.case_id] = PlannedRequest(
                case=case,
                arguments=build_model_request(recorder.request, settings),
                catalog_ids=tuple(claim.claim_id for claim in recorder.request.catalog),
            )
    chosen = _sample([plan.case for plan in reachable.values()], sample)
    return tuple(reachable[case.case_id] for case in chosen)


def estimate_plan(plan: tuple[PlannedRequest, ...], price: ModelPrice, max_output_tokens: int) -> dict[str, Any]:
    """Offline token and cost estimate at list price. Not billed usage."""
    input_tokens = sum(estimate_input_tokens(item.prompt_chars) for item in plan)
    output_tokens = sum(estimate_output_tokens(item.selection_chars) for item in plan)
    expected = (input_tokens * price.input + output_tokens * price.output) / 1_000_000
    worst = sum(worst_case_cost_usd(price, item.prompt_chars, max_output_tokens) for item in plan)
    count = len(plan)
    return {
        "method": "offline heuristic: chars/3 plus assumed overheads; run --preflight for exact input tokens",
        "requests": count,
        "expected_input_tokens_per_request": round(input_tokens / count, 1) if count else 0,
        "expected_output_tokens_per_request": round(output_tokens / count, 1) if count else 0,
        "estimated_cost_per_request_usd": round(expected / count, 8) if count else 0,
        "estimated_total_cost_usd": round(expected, 8),
        "worst_case_total_cost_usd": round(worst, 8),
    }


class BudgetGuard:
    """Counts requests and money, conservatively.

    A request is admitted only if its worst case still fits. A request whose
    reported usage is missing or malformed, such as one that timed out and may
    still be billed, is charged at that worst case.
    """

    def __init__(self, max_requests: int, max_cost_usd: float, price: ModelPrice) -> None:
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.price = price
        self.requests_sent = 0
        self.spent_usd = 0.0
        self.requests_at_worst_case = 0

    def admit(self, worst_case_usd: float) -> None:
        if self.requests_sent >= self.max_requests:
            raise BudgetExhausted("max_requests")
        if self.spent_usd + worst_case_usd > self.max_cost_usd:
            raise BudgetExhausted("max_cost_usd")
        self.requests_sent += 1

    def charge(self, telemetry: ProviderTelemetry | None, worst_case_usd: float) -> None:
        if telemetry is None or telemetry.usage is None:
            self.requests_at_worst_case += 1
            self.spent_usd += worst_case_usd
        else:
            self.spent_usd += request_cost_usd(self.price, telemetry.usage)


class BudgetedProvider:
    """The only route to the paid provider: admit, call, charge."""

    def __init__(self, inner: AnthropicTextProvider, guard: BudgetGuard, settings: CopilotSettings) -> None:
        self._inner = inner
        self._guard = guard
        self._settings = settings
        self.mode = inner.mode

    def generate(self, request: GenerationRequest) -> ProviderResult:
        arguments = build_model_request(request, self._settings)
        worst_case = worst_case_cost_usd(self._guard.price, _prompt_chars(arguments), arguments["max_tokens"])
        self._guard.admit(worst_case)
        try:
            result = self._inner.generate(request)
        except ProviderFailure as exc:
            self._guard.charge(exc.telemetry, worst_case)
            raise
        self._guard.charge(result.telemetry, worst_case)
        return result


@dataclass(frozen=True, slots=True)
class LiveRun:
    results: tuple[CaseResult, ...]
    stop_reason: str | None


def run_live(
    plan: tuple[PlannedRequest, ...],
    dataset: EvaluationDataset,
    settings: CopilotSettings,
    guard: BudgetGuard,
) -> LiveRun:
    """Send the planned requests until done, out of budget, or a fatal failure."""
    if settings.provider_retries != 0:
        raise ValueError("a bounded evaluation cannot allow hidden SDK retries")
    provider = BudgetedProvider(AnthropicTextProvider(settings), guard, settings)
    results: list[CaseResult] = []
    consecutive = 0
    for item in plan:
        try:
            result = run_case(item.case, dataset, provider, settings)
        except BudgetExhausted as exc:
            return LiveRun(tuple(results), f"budget_exhausted:{exc.limit}")
        results.append(result)
        kind = ProviderFailureKind(result.failure_kind) if result.failure_kind else None
        if kind in _FATAL_KINDS:
            return LiveRun(tuple(results), f"fatal_failure:{kind}")
        consecutive = consecutive + 1 if kind is not None else 0
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            return LiveRun(tuple(results), "consecutive_failures")
    return LiveRun(tuple(results), None)


def preflight(plan: tuple[PlannedRequest, ...], settings: CopilotSettings) -> dict[str, Any]:
    """Verify the model and count input tokens. Free: sends no ``messages.create``.

    Token counting is free but rate limited; the Models API is not billed.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise PreflightFailed("ANTHROPIC_API_KEY is not set")
    try:
        import anthropic  # lazy: dry runs never need it
    except ImportError as exc:
        raise PreflightFailed("the anthropic SDK is not installed") from exc

    client = anthropic.Anthropic(api_key=api_key, timeout=settings.timeout_seconds, max_retries=0)
    try:
        info = client.models.retrieve(settings.model_name)
    except Exception as exc:
        raise PreflightFailed(f"the model could not be verified ({type(exc).__name__})") from None

    capabilities = getattr(info, "capabilities", None)
    structured = getattr(getattr(capabilities, "structured_outputs", None), "supported", None)
    effort = getattr(getattr(capabilities, "effort", None), "supported", None)
    if info.lifecycle != "active":
        raise PreflightFailed(f"model lifecycle is {info.lifecycle!r}, not 'active'")
    if structured is not True:
        raise PreflightFailed("the model does not report structured output support")
    if effort is not True:
        raise PreflightFailed("the model does not report effort support")

    counts: list[int] = []
    try:
        for item in plan:
            arguments = {key: value for key, value in item.arguments.items() if key != "max_tokens"}
            counts.append(client.messages.count_tokens(**arguments).input_tokens)
    except Exception as exc:
        raise PreflightFailed(f"token counting failed ({type(exc).__name__})") from None
    return {
        "model": info.id,
        "lifecycle": info.lifecycle,
        "structured_outputs": True,
        "effort": True,
        "counted_requests": len(counts),
        "input_tokens_total": sum(counts),
        "input_tokens_max": max(counts, default=0),
    }


__all__ = [
    "BudgetExhausted",
    "BudgetGuard",
    "BudgetedProvider",
    "LiveRun",
    "MAX_CONSECUTIVE_FAILURES",
    "PlannedRequest",
    "PreflightFailed",
    "estimate_plan",
    "plan_live_run",
    "preflight",
    "real_settings",
    "run_live",
]
