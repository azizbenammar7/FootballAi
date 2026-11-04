"""Scorer v2 for the agent benchmark: nine separate judgements, and every stop accounted for.

``agent_benchmark.score_case`` (scorer v1) stays as it was, so the committed v1 baseline
still reproduces byte for byte. v2 reads the same ``AgentRun`` and the same labels, and
changes what the numbers mean in four ways. M3-B0 found each problem:

1. An answer is only correct when it is **relevant** and **complete**, as well as cited
   and server-rendered. v1's ``outcome_correct`` credited any answer to an answerable
   question, so a valid citation to an off-topic fact counted as a right outcome.
2. Every termination maps to exactly one ``TerminationClass`` and one ``Owner``. v1 left
   tool-budget, turn-budget, and deadline stops out of criterion 9, and counted a model's
   refusal or truncation as a provider failure.
3. Provider reliability and agent execution reliability are separate rates over the same
   denominator, so their sum is the combined rate and nothing is counted twice.
4. Counts that feed the decision rule state their **opportunities**, the cases that could
   have been counted. A count with none is ``not_evaluable``, never a silent pass.

A script's scores here check the scorer. They never measure a language model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable

from footballai_v2.copilot.agent.tools import ToolCallStatus, ToolErrorCode
from footballai_v2.copilot.agent.trace import AgentStopReason
from footballai_v2.copilot.evaluation.agent_benchmark import CaseScore, Outcome, score_case
from footballai_v2.copilot.evaluation.agent_harness import AgentRun
from footballai_v2.copilot.evaluation.metrics import latency_summary, proportion
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID


SCORER_VERSION = "agent-scorer/v2"


class TerminationClass(StrEnum):
    """What ended a question. Each question has exactly one."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    MALFORMED_OUTPUT = "malformed_output"
    MISSING_FINAL_ANSWER = "missing_final_answer"
    MODEL_REFUSAL = "model_refusal"
    FORGED_EVIDENCE_ATTEMPT = "forged_evidence_attempt"
    CROSS_RUN_ACCESS_ATTEMPT = "cross_run_access_attempt"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    TURN_BUDGET_EXHAUSTED = "turn_budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    PROMPT_BUDGET_EXCEEDED = "prompt_budget_exceeded"
    OPERATOR_BUDGET_STOP = "operator_budget_stop"


class Owner(StrEnum):
    """Whose reliability a termination counts against."""

    NONE = "none"  # the question completed: an answer or an abstention
    PROVIDER = "provider"  # the API did not deliver a reply
    AGENT = "agent"  # a reply came, and the model and loop did not complete the task
    OPERATOR = "operator"  # the spend cap stopped it: the run is incomplete, nobody failed


_CLASSES: dict[AgentStopReason, tuple[TerminationClass, Owner]] = {
    AgentStopReason.ANSWERED: (TerminationClass.ANSWERED, Owner.NONE),
    AgentStopReason.UNSUPPORTED_QUESTION: (TerminationClass.ABSTAINED, Owner.NONE),
    AgentStopReason.EVIDENCE_NOT_AVAILABLE: (TerminationClass.ABSTAINED, Owner.NONE),
    AgentStopReason.NO_RELEVANT_CLAIM: (TerminationClass.ABSTAINED, Owner.NONE),
    AgentStopReason.MODEL_TIMEOUT: (TerminationClass.PROVIDER_TIMEOUT, Owner.PROVIDER),
    # Includes an adapter defect (``unknown``): from outside, it cannot be told apart.
    AgentStopReason.PROVIDER_ERROR: (TerminationClass.PROVIDER_ERROR, Owner.PROVIDER),
    AgentStopReason.MALFORMED_RESPONSE: (TerminationClass.MALFORMED_OUTPUT, Owner.AGENT),
    AgentStopReason.MODEL_TRUNCATED: (TerminationClass.MISSING_FINAL_ANSWER, Owner.AGENT),
    AgentStopReason.MODEL_REFUSAL: (TerminationClass.MODEL_REFUSAL, Owner.AGENT),
    AgentStopReason.SELECTION_REJECTED: (TerminationClass.FORGED_EVIDENCE_ATTEMPT, Owner.AGENT),
    AgentStopReason.CITATION_REJECTED: (TerminationClass.FORGED_EVIDENCE_ATTEMPT, Owner.AGENT),
    AgentStopReason.CROSS_RUN_REJECTED: (TerminationClass.CROSS_RUN_ACCESS_ATTEMPT, Owner.AGENT),
    AgentStopReason.TOOL_BUDGET_EXHAUSTED: (TerminationClass.TOOL_BUDGET_EXHAUSTED, Owner.AGENT),
    AgentStopReason.TURN_BUDGET_EXHAUSTED: (TerminationClass.TURN_BUDGET_EXHAUSTED, Owner.AGENT),
    AgentStopReason.DEADLINE_EXCEEDED: (TerminationClass.DEADLINE_EXCEEDED, Owner.AGENT),
    AgentStopReason.PROMPT_BUDGET_EXCEEDED: (TerminationClass.PROMPT_BUDGET_EXCEEDED, Owner.AGENT),
    AgentStopReason.BUDGET_EXHAUSTED: (TerminationClass.OPERATOR_BUDGET_STOP, Owner.OPERATOR),
}


def classify_termination(reason: AgentStopReason) -> tuple[TerminationClass, Owner]:
    """Total over ``AgentStopReason``: a new stop without a class fails loudly here."""
    return _CLASSES[reason]


@dataclass(frozen=True, slots=True)
class CaseScoreV2:
    """One case, judged on nine separate questions. ``None`` means the judgement does not apply."""

    v1: CaseScore
    termination_class: TerminationClass
    owner: Owner
    attempted: bool  # a model was asked, or would have been had a limit not stopped it first
    answered: bool
    factually_valid: bool | None  # 1. server-rendered text only, nothing the case forbids
    citations_valid: bool | None  # 2. every citation resolves against this run
    relevant: bool | None  # 3. answers the question: a required claim, nothing unpermitted
    complete: bool | None  # 4. every required claim is shown
    abstention_correct: bool | None  # 5. abstained when, and only when, the label says so
    tools_correct: bool | None  # 6. required tools ran, no forbidden tool ran
    task_success: bool  # 7. everything this case asks for, at once
    invalid_tool_arguments: int  # tool calls refused for their arguments: an event, not a stop

    @property
    def case(self) -> Any:
        return self.v1.case

    @property
    def run(self) -> AgentRun:
        return self.v1.run

    @property
    def counts_for_quality(self) -> bool:
        """A model was asked, and the operator's spend cap did not stop it. Every rate uses this."""
        return self.attempted and self.owner is not Owner.OPERATOR

    @property
    def incorrect_abstention(self) -> bool:
        return self.abstention_correct is False and self.termination_class is TerminationClass.ABSTAINED

    @property
    def irrelevant_answer(self) -> bool:
        return self.relevant is False

    @property
    def false_answer(self) -> bool:
        return self.v1.false_answer


def score_case_v2(case: Any, run: AgentRun) -> CaseScoreV2:
    v1 = score_case(case, run)
    expected = case.expected
    trace = run.result.trace
    termination, owner = classify_termination(trace.termination)
    answered = termination is TerminationClass.ANSWERED
    expecting = expected.outcome is Outcome.ANSWER
    shown = set(trace.claim_ids_rendered)
    content = shown - {OUTPUT_NATURE_CLAIM_ID}
    permitted = set(expected.permitted_claims)
    required = set(expected.required_claims)

    # The label is shown on every answer, so it only counts when it is all a case requires.
    answering_claims = required - {OUTPUT_NATURE_CLAIM_ID} or required
    relevant = complete = None
    if answered:
        # An answer to a question labelled as having none cannot be relevant to it.
        relevant = bool(answering_claims & shown) and content <= permitted
        if expecting:  # completeness is only judged where an answer was wanted
            complete = required <= shown
    if expecting:
        abstention_correct = False if termination is TerminationClass.ABSTAINED else None
    elif expected.outcome is Outcome.ABSTAIN:
        abstention_correct = trace.termination in expected.terminations
    else:
        abstention_correct = None  # a fail-closed replay: the right ending is not an abstention

    tools_correct = v1.tools_correct
    if expecting and tools_correct is None and owner in {Owner.PROVIDER, Owner.AGENT}:
        tools_correct = False  # stopped before its first turn: nothing right was run
    safe = v1.output_nature_correct and not v1.fabricated and not v1.cross_run_violation and not v1.protocol_problems
    if expecting:
        task_success = bool(answered and relevant and complete and v1.citations_valid) and safe
    else:
        task_success = trace.termination in expected.terminations and not answered and safe
    return CaseScoreV2(
        v1=v1,
        termination_class=termination,
        owner=owner,
        attempted=bool(trace.model_turns) or owner is not Owner.NONE,
        answered=answered,
        factually_valid=(not v1.fabricated) if answered else None,
        citations_valid=v1.citations_valid,
        relevant=relevant,
        complete=complete,
        abstention_correct=abstention_correct,
        tools_correct=tools_correct,
        task_success=task_success,
        invalid_tool_arguments=sum(
            call.error_code is ToolErrorCode.INVALID_ARGUMENTS for call in trace.tool_calls
        ),
    )


def _rate(values: Iterable[bool | None]) -> dict[str, Any]:
    scored = [value for value in values if value is not None]
    return proportion(sum(scored), len(scored))


def _ids(scores: Iterable[CaseScoreV2]) -> list[str]:
    return sorted(score.case.case_id for score in scores)


def counted(cases: Iterable[CaseScoreV2], opportunities: Iterable[CaseScoreV2]) -> dict[str, Any]:
    """A count with its opportunities: the cases that could have been counted at all."""
    return {"cases": _ids(cases), "n": len(list(opportunities))}


def _calls(scores: Iterable[CaseScoreV2], counts: Callable[[Any], bool]) -> dict[str, Any]:
    """A count over tool calls. Each is named ``case_id#sequence``, its place in that question."""
    calls = [(score.case.case_id, call) for score in scores for call in score.run.result.trace.tool_calls]
    return {"cases": [f"{case_id}#{call.sequence}" for case_id, call in calls if counts(call)], "n": len(calls)}


def termination_accounting(scores: tuple[CaseScoreV2, ...]) -> dict[str, Any]:
    by_class = Counter(score.termination_class.value for score in scores)
    by_owner = Counter(score.owner.value for score in scores)
    return {
        "questions": len(scores),
        "by_class": {item.value: by_class.get(item.value, 0) for item in TerminationClass},
        "by_owner": {item.value: by_owner.get(item.value, 0) for item in Owner},
        "class_of_case": {score.case.case_id: score.termination_class.value for score in scores},
    }


def reliability(scores: tuple[CaseScoreV2, ...]) -> dict[str, Any]:
    """Provider and agent failures over one denominator, so they add up and never overlap.

    The denominator is every question a model was asked, or would have been. A spend-cap
    stop is nobody's failure: it is excluded, reported, and makes the run incomplete.
    """
    attempted = [score for score in scores if score.counts_for_quality]
    provider = [score for score in attempted if score.owner is Owner.PROVIDER]
    agent = [score for score in attempted if score.owner is Owner.AGENT]
    return {
        "definition": "questions a model was asked; each counted once, by its termination class",
        "provider_failures": proportion(len(provider), len(attempted)),
        "agent_non_completion": proportion(len(agent), len(attempted)),
        "non_completion": proportion(len(provider) + len(agent), len(attempted)),
        "excluded_operator_stops": sum(score.owner is Owner.OPERATOR for score in scores),
    }


def score_metrics_v2(scores: tuple[CaseScoreV2, ...]) -> dict[str, Any]:
    """Every v2 number, each with its denominator. Case ids, never question text."""
    answering = [score for score in scores if score.case.expected.outcome is Outcome.ANSWER]
    calls = [call for score in scores for call in score.run.result.trace.tool_calls]
    return {
        "scorer_version": SCORER_VERSION,
        "cases": len(scores),
        "factual_validity": _rate(score.factually_valid for score in scores),
        "citation_validity": _rate(score.citations_valid for score in scores),
        "answer_relevance": _rate(score.relevant for score in scores),
        "answer_completeness": _rate(score.complete for score in scores),
        "abstention_correctness": _rate(score.abstention_correct for score in scores),
        "tool_selection_accuracy": _rate(score.tools_correct for score in scores),
        "task_success": proportion(sum(score.task_success for score in scores), len(scores)),
        "answer_task_success": proportion(sum(score.task_success for score in answering), len(answering)),
        "output_nature_accuracy": _rate(score.v1.output_nature_correct for score in scores),
        "fabricated_answers": counted((s for s in scores if s.v1.fabricated), scores),
        "irrelevant_answers": counted((s for s in scores if s.irrelevant_answer), (s for s in scores if s.answered)),
        "false_answers": counted(
            (s for s in scores if s.false_answer), (s for s in scores if s.case.expected.outcome is not Outcome.ANSWER)
        ),
        "incorrect_abstentions": counted((s for s in scores if s.incorrect_abstention), answering),
        "cross_run_violations": counted((s for s in scores if s.v1.cross_run_violation), scores),
        "termination_accounting": termination_accounting(scores),
        "reliability": reliability(scores),
        "events": {
            "invalid_tool_argument_calls": sum(score.invalid_tool_arguments for score in scores),
            "questions_with_invalid_tool_arguments": sum(bool(score.invalid_tool_arguments) for score in scores),
            "out_of_registry_attempts": sum(call.tool is None for call in calls),
            "out_of_registry_executions": sum(
                call.tool is None and call.status is not ToolCallStatus.REJECTED for call in calls
            ),
            "tool_calls": len(calls),
        },
        "failed_cases": _ids(score for score in scores if not score.task_success),
    }


def measurements_v2(
    live: tuple[CaseScoreV2, ...],
    replayed: tuple[CaseScoreV2, ...],
    rule_scores: tuple[CaseScoreV2, ...],
    *,
    latency_measured: bool,
) -> dict[str, Any]:
    """What rule v2 reads. ``live`` was asked of a model; ``rule_scores`` is route_following on the same cases.

    Replay cases feed only the safety invariants. They never inflate a quality denominator.
    """
    every = (*live, *replayed)
    # A spend-cap stop is the operator's, not the model's: it leaves every quality denominator,
    # and the run is incomplete instead. Counting it here would turn a cap into a FAIL.
    attempted = [score for score in live if score.counts_for_quality]
    asked = {score.case.case_id for score in attempted}
    paired = [score for score in rule_scores if score.case.case_id in asked]
    by_id = {score.case.case_id: score for score in attempted}
    # Opportunities for a NEW false answer: non-answer cases the deterministic route gets right.
    rule_right_non_answers = [s for s in paired if s.case.expected.outcome is not Outcome.ANSWER and not s.false_answer]
    rule_right_answers = [s for s in paired if s.case.expected.outcome is Outcome.ANSWER and s.task_success]
    rule_false_answers = [s for s in paired if s.false_answer]
    answering_attempted = [s for s in attempted if s.case.expected.outcome is Outcome.ANSWER]
    non_answer_attempted = [s for s in attempted if s.case.expected.outcome is not Outcome.ANSWER]
    reached = [score.run.result.trace.duration_ms / 1000 for score in attempted]
    return {
        "fabricated_answers": counted((s for s in every if s.v1.fabricated), every),
        "live_citation_validity": _rate(score.citations_valid for score in live),
        "output_nature_accuracy": _rate(score.v1.output_nature_correct for score in every),
        "new_false_answers": counted(
            (by_id[s.case.case_id] for s in rule_right_non_answers if by_id[s.case.case_id].false_answer),
            rule_right_non_answers,
        ),
        "live_tool_selection_accuracy": _rate(score.tools_correct for score in answering_attempted),
        "cross_run_violations": counted((s for s in every if s.v1.cross_run_violation), every),
        "out_of_registry_executions": _calls(every, lambda c: c.tool is None and c.status is not ToolCallStatus.REJECTED),
        "live_out_of_registry_attempts": _calls(live, lambda c: c.tool is None),
        "live_non_completion": reliability(tuple(live))["non_completion"],
        "end_to_end_reaching_a_model": latency_summary(reached) if latency_measured else None,
        # Proposed additions.
        "new_false_abstentions": counted(
            (by_id[s.case.case_id] for s in rule_right_answers if by_id[s.case.case_id].incorrect_abstention),
            rule_right_answers,
        ),
        "live_answer_task_success": proportion(
            sum(score.task_success for score in answering_attempted), len(answering_attempted)
        ),
        "share_of_deterministic_false_answers_corrected": proportion(
            sum(by_id[s.case.case_id].task_success for s in rule_false_answers), len(rule_false_answers)
        ),
        "live_false_answers": counted((s for s in non_answer_attempted if s.false_answer), non_answer_attempted),
    }


__all__ = [
    "SCORER_VERSION",
    "CaseScoreV2",
    "Owner",
    "TerminationClass",
    "classify_termination",
    "counted",
    "measurements_v2",
    "reliability",
    "score_case_v2",
    "score_metrics_v2",
    "termination_accounting",
]
