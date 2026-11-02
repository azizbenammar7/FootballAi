"""Run cases through the real ``ReadOnlyAgent`` and score what came out.

A script or policy decides what the "model" does, so these scores measure the
agent's safety machinery and the harness, never a language model's judgement.
Negative controls exist to prove the metrics can fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from footballai_v2.copilot.agent.loop import AgentResult, ReadOnlyAgent
from footballai_v2.copilot.agent.tools import READ_ONLY_TOOLS, ToolCallStatus, ToolName
from footballai_v2.copilot.agent.trace import AgentStopReason
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ForgedCitationError, ProviderFailureKind
from footballai_v2.copilot.evaluation.agent_dataset import AgentCase, AgentDataset
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, protocol_violations
from footballai_v2.copilot.evaluation.dataset import EvaluationCase, EvaluationDataset
from footballai_v2.copilot.evaluation.fixtures import build_fixture_run
from footballai_v2.copilot.evaluation.harness import CaseResult, score_case
from footballai_v2.copilot.evaluation.routers import routing_for
from footballai_v2.copilot.evidence import RunEvidenceService
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID, build_claim_catalog, validate_citations
from footballai_v2.copilot.provider import AgentModelRequest


EVALUATION_SETTINGS = CopilotSettings(enabled=True, environment="test")
_TOOL_OF_CLAIM = {claim: spec.name for spec in READ_ONLY_TOOLS for claim in spec.claims}


class StepClock:
    """A clock only the script moves, so timing limits are exact and instant."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True, slots=True)
class AgentRun:
    """One finished question, with the exact prompt history the stand-in received."""

    result: AgentResult
    model: ScriptedAgentModel
    catalog: tuple[Any, ...]
    evidence: Any
    run_id: str

    @property
    def prompt(self) -> str:
        if not self.model.requests:
            return ""
        request = self.model.requests[-1]
        return request.system_instructions + json.dumps(request.messages, default=vars)

    @property
    def executed_tools(self) -> tuple[str, ...]:
        return tuple(
            call.tool.value
            for call in self.result.trace.tool_calls
            if call.status in {ToolCallStatus.OK, ToolCallStatus.NO_EVIDENCE}
        )


def run_agent(
    dataset: EvaluationDataset,
    fixture_name: str,
    question: str | None,
    model_for: Callable[[StepClock], ScriptedAgentModel],
    *,
    settings: CopilotSettings = EVALUATION_SETTINGS,
) -> AgentRun:
    repository, storage, run_id = build_fixture_run(dataset.fixtures.fixtures[fixture_name], dataset.fixtures)
    clock = StepClock()
    model = model_for(clock)
    agent = ReadOnlyAgent(repository, storage, model, settings, routing=routing_for(dataset.version), clock=clock)
    evidence = RunEvidenceService(repository, storage, settings).build(run_id)
    result = agent.answer(run_id, question)
    return AgentRun(result, model, build_claim_catalog(evidence), evidence, run_id)


def _failure_kind(run: AgentRun) -> str | None:
    trace = run.result.trace
    if trace.termination is AgentStopReason.SELECTION_REJECTED:
        return ProviderFailureKind.INVALID_RESPONSE.value
    return trace.error_code.value if isinstance(trace.error_code, ProviderFailureKind) else None


def _score_selection_case(case: EvaluationCase, run: AgentRun) -> CaseResult:
    return score_case(
        case,
        run.result.response,
        run.evidence,
        run.run_id,
        claims=run.catalog,
        offered=run.model.requests[0].offered_claim_ids if run.model.requests else (),
        selected=run.result.trace.claim_ids_selected,
        provider_invoked=bool(run.model.requests),
        failure_kind=_failure_kind(run),
        prompt=run.prompt,
        latency=0.0,
        telemetry=None,
    )


@dataclass(frozen=True, slots=True)
class AgentSelectionResult:
    """A claim-selection case answered through the agent, with its tool behaviour."""

    case: CaseResult
    required_tools: frozenset[str]
    forbidden_tools: frozenset[str]
    run: AgentRun

    @property
    def tool_selection_correct(self) -> bool | None:
        if self.case.selection_correct is None:
            return None
        called = set(self.run.executed_tools)
        return self.required_tools <= called and not called & self.forbidden_tools


def _tools_of(claims: Iterable[str]) -> frozenset[str]:
    return frozenset(_TOOL_OF_CLAIM[claim].value for claim in claims)


def run_selection_cases(
    cases: tuple[EvaluationCase, ...],
    dataset: EvaluationDataset,
    policy_for: Callable[[EvaluationCase], Callable[[AgentModelRequest], Any]],
) -> tuple[AgentSelectionResult, ...]:
    """Tool expectations come from each case's existing labels, not from any policy."""
    results = []
    for case in cases:
        run = run_agent(
            dataset, case.fixture, case.question, lambda _clock: ScriptedAgentModel(policy=policy_for(case))
        )
        expected = case.expected
        required = _tools_of(claim for claim in expected.required_claims if claim != OUTPUT_NATURE_CLAIM_ID)
        if not required and expected.required_claims:
            required = _tools_of(expected.required_claims)
        permitted = _tools_of(expected.allowed_claims)
        results.append(
            AgentSelectionResult(
                _score_selection_case(case, run),
                required,
                frozenset(name.value for name in ToolName) - permitted,
                run,
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class ScriptedResult:
    """One scripted case. ``failures`` names every expectation that was not met."""

    case: AgentCase
    run: AgentRun
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_scripted_case(case: AgentCase, dataset: AgentDataset) -> ScriptedResult:
    def model_for(clock: StepClock) -> ScriptedAgentModel:
        return ScriptedAgentModel((step.to_step() for step in case.script), on_call=clock.advance)

    run = run_agent(dataset.source, case.fixture, case.question, model_for)
    return ScriptedResult(case, run, tuple(_unmet(case, run)))


def _unmet(case: AgentCase, run: AgentRun) -> Iterable[str]:
    expected = case.expected
    response, trace = run.result.response, run.result.trace
    called = set(run.executed_tools)
    if trace.termination is not expected.stop:
        yield "stop"
    if response.unavailable_reason != expected.unavailable_reason:
        yield "unavailable_reason"
    if response.output_nature is not expected.output_nature:
        yield "output_nature"
    if run.executed_tools != tuple(tool.value for tool in expected.tools_executed):
        yield "tools_executed"
    if not {tool.value for tool in expected.required_tools} <= called:
        yield "required_tools"
    if called & {tool.value for tool in expected.forbidden_tools}:
        yield "forbidden_tools"
    if trace.model_turns != expected.model_turns:
        yield "model_turns"
    if not _rendered_as_expected(case, run):
        yield "claims"
    if not _server_rendered(run):
        yield "server_rendering"
    if not _citations_valid(run):
        yield "citations"
    if any(text in response.answer or text in run.prompt for text in expected.forbidden_text):
        yield "forbidden_text"
    if protocol_violations(run.model.requests):
        yield "protocol"


def _rendered_as_expected(case: AgentCase, run: AgentRun) -> bool:
    rendered = set(run.result.trace.claim_ids_rendered) - {OUTPUT_NATURE_CLAIM_ID}
    required = set(case.expected.required_claims) - {OUTPUT_NATURE_CLAIM_ID}
    return required <= rendered and rendered <= set(case.expected.permitted_claims)


def _server_rendered(run: AgentRun) -> bool:
    """Every answer is the catalog's own text. An abstention cites nothing."""
    response, trace = run.result.response, run.result.trace
    if response.unavailable_reason is not None:
        return response.citations == [] and trace.claim_ids_rendered == ()
    by_id = {claim.claim_id: claim for claim in run.catalog}
    return response.answer == " ".join(by_id[claim_id].text for claim_id in trace.claim_ids_rendered)


def _citations_valid(run: AgentRun) -> bool:
    """Citations resolve against this run, and are exactly those of the rendered claims."""
    response, trace = run.result.response, run.result.trace
    by_id = {claim.claim_id: claim for claim in run.catalog}
    rendered = {citation.field_path for claim_id in trace.claim_ids_rendered for citation in by_id[claim_id].citations}
    try:
        validate_citations(response.citations, run.evidence, run.run_id)
    except ForgedCitationError:
        return False
    return {citation.field_path for citation in response.citations} == rendered


def tool_metrics(results: Iterable[AgentRun]) -> dict[str, Any]:
    """Tool-call efficiency. A useful call returned a claim that an answer went on to use."""
    calls = useful = rejected = empty = refused = 0
    questions = 0
    for run in results:
        questions += 1
        trace = run.result.trace
        used = set(trace.claim_ids_selected) if trace.claim_ids_rendered else set()
        refused += trace.tool_calls_refused
        for call in trace.tool_calls:
            calls += 1
            useful += bool(used & set(call.claim_ids_returned))
            rejected += call.status is ToolCallStatus.REJECTED
            empty += call.status is ToolCallStatus.NO_EVIDENCE
    return {
        "questions": questions,
        "tool_calls": calls,
        "calls_per_question": round(calls / questions, 4) if questions else None,
        "useful_calls": useful,
        "useful_call_rate": round(useful / calls, 4) if calls else None,
        "rejected_calls": rejected,
        "empty_calls": empty,
        "calls_refused_before_running": refused,
    }


def stop_counts(runs: Iterable[AgentRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        reason = run.result.trace.termination.value
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def answered_without_permission(results: Iterable[ScriptedResult]) -> list[str]:
    """Scripted cases that produced an answer they were not labelled to produce."""
    return sorted(
        result.case.case_id
        for result in results
        if result.run.result.response.unavailable_reason is None
        and result.case.expected.stop is not AgentStopReason.ANSWERED
    )


__all__ = [
    "AgentRun",
    "AgentSelectionResult",
    "EVALUATION_SETTINGS",
    "ScriptedResult",
    "StepClock",
    "answered_without_permission",
    "run_agent",
    "run_scripted_case",
    "run_selection_cases",
    "stop_counts",
    "tool_metrics",
]
