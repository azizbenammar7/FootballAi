"""Run cases through the real ExplanationService and score what came out.

Every score is computed from the service's actual response, so an evaluation
cannot pass through a path production does not take.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ForgedCitationError, ProviderFailureKind
from footballai_v2.copilot.evaluation.dataset import EvaluationCase, EvaluationDataset, Outcome
from footballai_v2.copilot.evaluation.fixtures import build_fixture_run
from footballai_v2.copilot.evaluation.selectors import RecordingProvider
from footballai_v2.copilot.evidence import RunEvidenceService
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID, validate_citations
from footballai_v2.copilot.models import ProviderTelemetry
from footballai_v2.copilot.provider import LLMProvider, build_provider_payload
from footballai_v2.copilot.service import ExplanationService


ANSWERED = "answered"
ABSTAINED = "abstained"
REJECTED = "rejected"  # the service refused a forged selection with an error
EVALUATION_SETTINGS = CopilotSettings(enabled=True, environment="test")


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    suite: str
    category: str
    outcome: str
    reason: str | None
    rendered_claims: tuple[str, ...]  # without the always-rendered label
    offered_claims: tuple[str, ...]
    provider_invoked: bool
    failure_kind: str | None
    outcome_correct: bool
    selection_correct: bool | None  # None: the case does not expect an answer
    support_level_correct: bool
    output_nature_correct: bool
    citations_valid: bool
    expected_citations_present: bool | None
    answer_server_rendered: bool
    forbidden_text_absent: bool
    unpermitted_claims: tuple[str, ...]
    coverage_gap: bool
    latency_seconds: float
    telemetry: ProviderTelemetry | None

    def to_report(self) -> dict[str, object]:
        """A case's outcome for a report. Holds ids and counts, never prompts."""
        return {
            "case_id": self.case_id,
            "suite": self.suite,
            "category": self.category,
            "outcome": self.outcome,
            "reason": self.reason,
            "rendered_claims": list(self.rendered_claims),
            "provider_invoked": self.provider_invoked,
            "failure_kind": self.failure_kind,
            "outcome_correct": self.outcome_correct,
            "selection_correct": self.selection_correct,
            "unpermitted_claims": list(self.unpermitted_claims),
            "latency_seconds": round(self.latency_seconds, 6),
            "request_id": self.telemetry.request_id if self.telemetry else None,
            "stop_reason": self.telemetry.stop_reason if self.telemetry else None,
            "usage": _usage_report(self.telemetry),
        }


def _usage_report(telemetry: ProviderTelemetry | None) -> dict[str, int] | None:
    if telemetry is None or telemetry.usage is None:
        return None
    usage = telemetry.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
    }


def run_case(
    case: EvaluationCase,
    dataset: EvaluationDataset,
    provider: LLMProvider,
    settings: CopilotSettings = EVALUATION_SETTINGS,
) -> CaseResult:
    fixture = dataset.fixtures.fixtures[case.fixture]
    repository, storage, run_id = build_fixture_run(fixture, dataset.fixtures)
    recorder = RecordingProvider(provider)
    service = ExplanationService(repository, storage, recorder, settings)
    evidence = RunEvidenceService(repository, storage, settings).build(run_id)

    started = time.perf_counter()
    try:
        response = service.explain(run_id, case.question)
    except ForgedCitationError:
        response = None
    latency = time.perf_counter() - started

    expected = case.expected
    offered = tuple(claim.claim_id for claim in recorder.request.catalog) if recorder.request else ()
    selected = recorder.result.claim_ids if recorder.result else ()
    if response is None:
        outcome, reason = REJECTED, ForgedCitationError.code
    elif response.unavailable_reason is not None:
        outcome, reason = ABSTAINED, response.unavailable_reason
    else:
        outcome, reason = ANSWERED, None

    rendered: tuple[str, ...] = ()
    server_rendered = True
    citations_valid = True
    expected_citations_present: bool | None = None
    if response is not None and outcome == ANSWERED:
        assert recorder.request is not None
        by_id = {claim.claim_id: claim for claim in recorder.request.catalog}
        rendered = tuple(dict.fromkeys(item for item in selected if item != OUTPUT_NATURE_CLAIM_ID))
        claims = [by_id[OUTPUT_NATURE_CLAIM_ID], *(by_id[item] for item in rendered)]
        server_rendered = response.answer == " ".join(claim.text for claim in claims)
        cited = {citation.field_path for claim in claims for citation in claim.citations}
        actual = {citation.field_path for citation in response.citations}
        try:
            validate_citations(response.citations, evidence, run_id)
            citations_valid = actual == cited
        except ForgedCitationError:
            citations_valid = False
        if expected.citations:
            expected_citations_present = set(expected.citations) <= actual
    elif response is not None:
        citations_valid = response.citations == []

    allowed = expected.allowed_claims
    unpermitted = tuple(item for item in rendered if item not in allowed)
    outcome_correct = _outcome_correct(case, outcome, reason, unpermitted)
    selection_correct: bool | None = None
    if expected.outcome is Outcome.ANSWER:
        shown = {*rendered, OUTPUT_NATURE_CLAIM_ID}
        selection_correct = (
            outcome == ANSWERED and set(expected.required_claims) <= shown and not unpermitted
        )

    return CaseResult(
        case_id=case.case_id,
        suite=case.suite.value,
        category=case.category,
        outcome=outcome,
        reason=reason,
        rendered_claims=rendered,
        offered_claims=offered,
        provider_invoked=recorder.calls > 0,
        failure_kind=_failure_kind(recorder, outcome),
        outcome_correct=outcome_correct,
        selection_correct=selection_correct,
        support_level_correct=(
            expected.support_level is None
            or (response is not None and response.support_level is expected.support_level)
        ),
        output_nature_correct=response is None or response.output_nature is expected.output_nature,
        citations_valid=citations_valid,
        expected_citations_present=expected_citations_present,
        answer_server_rendered=server_rendered,
        forbidden_text_absent=_forbidden_text_absent(case, response.answer if response else "", recorder),
        unpermitted_claims=unpermitted,
        coverage_gap=expected.coverage_gap,
        latency_seconds=latency,
        telemetry=recorder.telemetry,
    )


def _outcome_correct(case: EvaluationCase, outcome: str, reason: str | None, unpermitted: tuple[str, ...]) -> bool:
    expected = case.expected
    if expected.outcome is Outcome.ANSWER:
        return outcome == ANSWERED
    if expected.outcome is Outcome.EITHER:
        return (outcome == ANSWERED and not unpermitted) or reason == expected.reason
    return outcome != ANSWERED and reason == expected.reason


def _failure_kind(recorder: RecordingProvider, outcome: str) -> str | None:
    if recorder.failure is not None:
        return recorder.failure.kind.value
    return ProviderFailureKind.INVALID_RESPONSE.value if outcome == REJECTED else None


def _forbidden_text_absent(case: EvaluationCase, answer: str, recorder: RecordingProvider) -> bool:
    """Forbidden text must reach neither the reader nor the model."""
    prompt = ""
    if recorder.request is not None:
        request = recorder.request
        prompt = json.dumps(build_provider_payload(request.evidence, request.catalog, request.question))
    return not any(text in answer or text in prompt for text in case.expected.forbidden_text)


def run_cases(
    cases: tuple[EvaluationCase, ...],
    dataset: EvaluationDataset,
    provider_for: Callable[[EvaluationCase], LLMProvider],
    settings: CopilotSettings = EVALUATION_SETTINGS,
) -> tuple[CaseResult, ...]:
    return tuple(run_case(case, dataset, provider_for(case), settings) for case in cases)
