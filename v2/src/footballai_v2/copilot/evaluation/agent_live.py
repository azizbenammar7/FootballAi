"""A live agent evaluation. Refused unless a person approved exactly this run.

``plan_benchmark`` needs no SDK, key, or network. ``agent_preflight`` needs a key and
sends no ``messages.create``. Only ``run_benchmark_live`` can spend money, and it needs
an approval record naming this model, these caps, this dataset, and this decision
rule. Every model call goes through ``BudgetedAgentModel``. Replay cases are never sent.

Nothing here runs in CI, and no test sends a request anywhere but 127.0.0.1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from footballai_v2.copilot.agent.trace import AgentStopReason
from footballai_v2.copilot.budget import BudgetedAgentModel, BudgetGuard
from footballai_v2.copilot.config import MAX_AGENT_COST_USD, MAX_AGENT_REQUESTS, CopilotSettings
from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.evaluation.agent_benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseMode,
    CaseScore,
    Split,
    run_case,
)
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, route_following_policy
from footballai_v2.copilot.evaluation.live import _FATAL_KINDS, MAX_CONSECUTIVE_FAILURES, PreflightFailed, preflight
from footballai_v2.copilot.evaluation.agent_rule import ANTHROPIC_ENDPOINT
from footballai_v2.copilot.pricing import MODEL_PRICES, agent_question_worst_case_usd
from footballai_v2.copilot.provider import (
    AgentModelRequest,
    AnthropicAgentModel,
    api_base_url,
    build_agent_model_request,
)


APPROVAL_SCHEMA = "footballai.copilot-agent-live-approval/v1"
LOOPBACK_ENDPOINT = "loopback_test_double"


class LiveRefused(Exception):
    """The run was not started. ``code`` says why, and holds no secret."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class LiveApproval(BaseModel):
    """A person's decision, for one run. Copied from live-approval.template.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(alias="schema", pattern=f"^{APPROVAL_SCHEMA}$")
    status: Literal["not_approved", "approved"]
    approved_by: str | None
    approved_on: str | None
    stage: Literal["readiness", "decision"]
    split: Split
    model: str
    max_model_requests: int = Field(ge=0, le=MAX_AGENT_REQUESTS)
    max_cost_usd: float = Field(ge=0, le=MAX_AGENT_COST_USD)
    decision_rule_sha256: str
    cases_sha256: str


def endpoint_kind(environ: Mapping[str, str] | None = None) -> str:
    """Where calls would go. A test double can never produce a model evaluation."""
    base = api_base_url(environ)
    if base is None:
        raise LiveRefused("unapproved_api_endpoint")
    return ANTHROPIC_ENDPOINT if base.startswith("https://") else LOOPBACK_ENDPOINT


def load_approval(path: Path) -> LiveApproval:
    try:
        return LiveApproval.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise LiveRefused("unreadable_approval") from None


def check_approval(
    approval: LiveApproval,
    *,
    split: Split,
    model: str,
    max_requests: int,
    max_cost_usd: float,
    cases_sha256: str,
    rule_sha256: str,
    rule_accepted: bool,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Every condition the owner set for a paid run. All must hold, checked before any call."""
    environment = os.environ if environ is None else environ
    if environment.get("CI") or environment.get("GITHUB_ACTIONS"):
        raise LiveRefused("continuous_integration")
    if not environment.get("ANTHROPIC_API_KEY", "").strip():
        raise LiveRefused("missing_api_key")
    if model not in MODEL_PRICES:
        raise LiveRefused("unpriced_model")
    endpoint_kind(environment)
    conditions = {
        "not_approved": approval.status == "approved" and bool(approval.approved_by) and bool(approval.approved_on),
        "approval_other_split": approval.split is split,
        "approval_other_model": approval.model == model,
        "approval_other_cases": approval.cases_sha256 == cases_sha256,
        "approval_other_rule": approval.decision_rule_sha256 == rule_sha256,
        "requests_over_approval": 0 < max_requests <= approval.max_model_requests,
        "cost_over_approval": 0 < max_cost_usd <= approval.max_cost_usd,
        # A readiness probe may precede acceptance. A decision never can.
        "rule_not_accepted": approval.stage == "readiness" or rule_accepted,
        "decision_needs_heldout": approval.stage == "readiness" or split is Split.HELDOUT,
        "readiness_cannot_read_heldout": approval.stage == "decision" or split is Split.DEV,
    }
    for code, holds in conditions.items():
        if not holds:
            raise LiveRefused(code)


@dataclass(frozen=True, slots=True)
class PlannedCase:
    case: BenchmarkCase
    arguments: dict[str, Any]  # the first turn, for token counting
    schema_key: tuple[str, ...]  # one output schema per distinct set of offered claims


def plan_benchmark(dataset: BenchmarkDataset, split: Split, settings: CopilotSettings) -> tuple[PlannedCase, ...]:
    """The live questions that reach a model, found by running the offline rule.

    A replay case, or a question the router stops, never costs a request.
    """
    planned: list[PlannedCase] = []
    for case in dataset.split(split):
        if case.mode is not CaseMode.LIVE_ELIGIBLE:
            continue
        score = run_case(case, dataset.source, lambda _clock: ScriptedAgentModel(policy=route_following_policy))
        requests = score.run.model.requests
        if requests:
            first: AgentModelRequest = requests[0]
            planned.append(PlannedCase(case, build_agent_model_request(first, settings), tuple(first.offered_claim_ids)))
    return tuple(planned)


def plan_cost(plan: tuple[PlannedCase, ...], settings: CopilotSettings, max_turns: int) -> dict[str, Any]:
    """An upper bound at list price. Each question may use every turn at the full prompt budget."""
    price = MODEL_PRICES[settings.model_name]
    per_question = agent_question_worst_case_usd(
        price, settings.max_prompt_chars, settings.max_output_tokens, max_turns
    )
    return {
        "note": "An upper bound at list price, not a measurement or a bill. Real questions should cost far less.",
        "model": settings.model_name,
        "questions_reaching_a_model": len(plan),
        "distinct_output_schemas": len({item.schema_key for item in plan}),
        "max_model_requests": len(plan) * max_turns,
        "worst_case_usd_per_question": round(per_question, 6),
        "worst_case_total_usd": round(per_question * len(plan), 6),
    }


def agent_preflight(plan: tuple[PlannedCase, ...], settings: CopilotSettings) -> dict[str, Any]:
    """Confirm the model and count input tokens. Sends no ``messages.create``."""
    if api_base_url() is None:
        raise PreflightFailed("ANTHROPIC_BASE_URL is not an approved endpoint")
    items = tuple(SimpleNamespace(arguments=item.arguments) for item in plan)
    return preflight(items, settings)  # type: ignore[arg-type]  # it reads only ``arguments``


@dataclass(frozen=True, slots=True)
class LiveBenchmarkRun:
    scores: tuple[CaseScore, ...]
    first_use: frozenset[str]  # correlation ids of questions that met a new output schema
    stop_reason: str | None


def run_benchmark_live(
    plan: tuple[PlannedCase, ...],
    dataset: BenchmarkDataset,
    settings: CopilotSettings,
    guard: BudgetGuard,
) -> LiveBenchmarkRun:
    """Ask each planned question once, until done, out of budget, or a fatal failure."""
    scores: list[CaseScore] = []
    first_use: set[str] = set()
    seen_schemas: set[tuple[str, ...]] = set()
    consecutive = 0
    stop: str | None = None
    for item in plan:
        model = BudgetedAgentModel(AnthropicAgentModel(settings), guard, settings)
        try:
            score = run_case(item.case, dataset.source, lambda _clock: model, settings=settings, real_time=True)
        except KeyboardInterrupt:  # money was spent, so the report must still be written
            stop = "interrupted"
            break
        except Exception:
            stop = "runner_error"
            break
        finally:
            model.close()
        trace = score.run.result.trace
        if trace.termination is AgentStopReason.BUDGET_EXHAUSTED:
            stop = "budget_exhausted"
            if trace.model_turns:  # it had spent money before the budget stopped it
                scores.append(score)
            break
        scores.append(score)
        if trace.model_turns and item.schema_key not in seen_schemas:
            seen_schemas.add(item.schema_key)
            first_use.add(trace.correlation_id)
        kind = trace.error_code if isinstance(trace.error_code, ProviderFailureKind) else None
        if kind in _FATAL_KINDS:
            stop = f"fatal_failure:{kind}"
            break
        consecutive = consecutive + 1 if kind is not None else 0
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            stop = "consecutive_failures"
            break
    return LiveBenchmarkRun(tuple(scores), frozenset(first_use), stop)


__all__ = [
    "APPROVAL_SCHEMA",
    "LOOPBACK_ENDPOINT",
    "LiveApproval",
    "LiveBenchmarkRun",
    "LiveRefused",
    "PlannedCase",
    "agent_preflight",
    "check_approval",
    "endpoint_kind",
    "load_approval",
    "plan_benchmark",
    "plan_cost",
    "run_benchmark_live",
]
