"""One bounded, read-only agent over one server-bound analysis run.

The order mirrors the explanation service, with retrieval added. Evidence is
built once, and a question FootballAI cannot answer is abstained before a model
is called. Otherwise the model may call the closed tool registry, then select
claim ids. It never writes a fact: the server renders the claims' own text, and
only claims that a tool returned in this request can be rendered.

Every limit fails closed with a typed reason, no partial answer is shown, and
nothing is retried. There is no memory: all state here lives for one request.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from footballai_v2.contracts.v1 import AnalysisRunStatus
from footballai_v2.copilot.agent.tools import (
    READ_ONLY_TOOLS,
    ToolErrorCode,
    ToolRegistry,
    ToolSpec,
    references_a_run,
    tool_definitions,
)
from footballai_v2.copilot.agent.trace import (
    MAX_TRACED_TOOL_CALLS,
    MAX_TRACED_TURNS,
    AgentStopReason,
    AgentTrace,
    TraceRecorder,
)
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import (
    ForgedCitationError,
    PromptBudgetExceededError,
    ProviderFailure,
    ProviderFailureKind,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RunNotEligibleError,
)
from footballai_v2.copilot.evidence import RunEvidenceService, ValidatedRunEvidence
from footballai_v2.copilot.grounding import (
    AGENT_INSTRUCTION_VERSION,
    AGENT_SYSTEM_INSTRUCTIONS,
    OUTPUT_NATURE_CLAIM_ID,
    ROUTING_VERSION,
    EvidenceClaim,
    QuestionRoute,
    QuestionRouter,
    QuestionScope,
    build_claim_catalog,
    can_answer,
    claims_to_render,
    eligible_claims,
    render_abstention,
    render_claims,
    route_question,
    validate_citations,
)
from footballai_v2.copilot.models import ExplanationResponse, ModelMode
from footballai_v2.copilot.provider import AgentModel, AgentModelRequest, AgentReply, build_provider_payload
from footballai_v2.copilot.service import (
    NO_CLAIM_ANSWER,
    NO_CLAIM_REASON,
    NO_EVIDENCE_ANSWER,
    NO_EVIDENCE_REASON,
    UNSUPPORTED_REASON,
    support_level_for,
    unavailable_response,
)
from footballai_v2.storage.ports import AnalysisRepository, ObjectStorage


_ELIGIBLE_STATUSES = frozenset({AnalysisRunStatus.SUCCEEDED.value, AnalysisRunStatus.PARTIAL.value})
_FORGED = ForgedCitationError.code
_PROVIDER_UNAVAILABLE = ProviderUnavailableError.code
_LIMIT_REACHED = "agent_limit_reached"
_CORRELATION_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_AGENT_FAILED_ANSWER = (
    "The evidence agent could not finish this question safely, so no answer was produced. "
    "The run's verified evidence remains visible on this page."
)
# What an API client sees. The finer reason stays in the trace.
_PUBLIC_REASONS: dict[AgentStopReason, str] = {
    AgentStopReason.UNSUPPORTED_QUESTION: UNSUPPORTED_REASON,
    AgentStopReason.EVIDENCE_NOT_AVAILABLE: NO_EVIDENCE_REASON,
    AgentStopReason.NO_RELEVANT_CLAIM: NO_CLAIM_REASON,
    AgentStopReason.SELECTION_REJECTED: _FORGED,
    AgentStopReason.CITATION_REJECTED: _FORGED,
    AgentStopReason.CROSS_RUN_REJECTED: "cross_run_rejected",
    AgentStopReason.TOOL_BUDGET_EXHAUSTED: _LIMIT_REACHED,
    AgentStopReason.TURN_BUDGET_EXHAUSTED: _LIMIT_REACHED,
    AgentStopReason.DEADLINE_EXCEEDED: _LIMIT_REACHED,
    AgentStopReason.BUDGET_EXHAUSTED: _LIMIT_REACHED,
    AgentStopReason.PROMPT_BUDGET_EXCEEDED: PromptBudgetExceededError.code,
    AgentStopReason.MODEL_TIMEOUT: ProviderTimeoutError.code,
    AgentStopReason.MODEL_REFUSAL: _PROVIDER_UNAVAILABLE,
    AgentStopReason.MODEL_TRUNCATED: _PROVIDER_UNAVAILABLE,
    AgentStopReason.MALFORMED_RESPONSE: _PROVIDER_UNAVAILABLE,
    AgentStopReason.PROVIDER_ERROR: _PROVIDER_UNAVAILABLE,
}
_FAILURE_STOPS = {
    ProviderFailureKind.REFUSAL: AgentStopReason.MODEL_REFUSAL,
    ProviderFailureKind.TRUNCATED: AgentStopReason.MODEL_TRUNCATED,
    ProviderFailureKind.INVALID_RESPONSE: AgentStopReason.MALFORMED_RESPONSE,
    ProviderFailureKind.BUDGET_EXHAUSTED: AgentStopReason.BUDGET_EXHAUSTED,
}
# Refused locally, so nothing was sent, nothing can be billed, and no turn happened.
_NEVER_SENT = frozenset({ProviderFailureKind.NOT_CONFIGURED, ProviderFailureKind.BUDGET_EXHAUSTED})


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Budgets for one question. A configuration can tighten these, never loosen them."""

    max_tool_calls: int = MAX_TRACED_TOOL_CALLS
    max_model_turns: int = MAX_TRACED_TURNS
    model_timeout_seconds: float = 20.0
    total_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        if not 0 <= self.max_tool_calls <= MAX_TRACED_TOOL_CALLS:
            raise ValueError(f"max_tool_calls must be between 0 and {MAX_TRACED_TOOL_CALLS}")
        if not 1 <= self.max_model_turns <= MAX_TRACED_TURNS:
            raise ValueError(f"max_model_turns must be between 1 and {MAX_TRACED_TURNS}")
        if not 0 < self.model_timeout_seconds <= 20.0:
            raise ValueError("model_timeout_seconds must be above 0 and at most 20")
        if not 0 < self.total_timeout_seconds <= 45.0:
            raise ValueError("total_timeout_seconds must be above 0 and at most 45")


@dataclass(frozen=True, slots=True)
class AgentResult:
    response: ExplanationResponse
    trace: AgentTrace


@dataclass(frozen=True, slots=True)
class _Stop:
    """How the loop ended: a reason, or ``None`` with a selection to ground."""

    reason: AgentStopReason | None = None
    error_code: ProviderFailureKind | ToolErrorCode | None = None
    claim_ids: tuple[str, ...] = ()


class ReadOnlyAgent:
    """Answer one question about one run. The caller binds the run; the model cannot."""

    def __init__(
        self,
        repository: AnalysisRepository,
        object_storage: ObjectStorage,
        model: AgentModel,
        settings: CopilotSettings | None = None,
        *,
        limits: AgentLimits | None = None,
        routing: tuple[str, QuestionRouter] = (ROUTING_VERSION, route_question),
        clock: Callable[[], float] = time.monotonic,
        tools: tuple[ToolSpec, ...] = READ_ONLY_TOOLS,
    ) -> None:
        self._settings = settings or CopilotSettings()
        self._evidence_service = RunEvidenceService(repository, object_storage, self._settings)
        self._model = model
        self._limits = limits or AgentLimits()
        self._routing_version, self._router = routing  # together, so a trace names what really ran
        self._clock = clock
        self._tools = tools

    def answer(self, run_id: str, question: str | None = None, *, correlation_id: str | None = None) -> AgentResult:
        started = self._clock()
        evidence = self._evidence_service.build(run_id)
        if evidence.status not in _ELIGIBLE_STATUSES:
            raise RunNotEligibleError("only completed runs can be explained")
        valid_id = correlation_id is not None and _CORRELATION_ID_RE.fullmatch(correlation_id)
        recorder = TraceRecorder(
            correlation_id if valid_id else uuid.uuid4().hex,
            evidence.run_id,
            self._model.model_name,
            AGENT_INSTRUCTION_VERSION,
            self._routing_version,
        )

        route = self._router(question)
        if route.scope is QuestionScope.UNSUPPORTED:
            return self._abstain(evidence, recorder, started, AgentStopReason.UNSUPPORTED_QUESTION, render_abstention())
        catalog = eligible_claims(build_claim_catalog(evidence), route)
        if not can_answer(catalog, route):
            return self._abstain(evidence, recorder, started, AgentStopReason.EVIDENCE_NOT_AVAILABLE, NO_EVIDENCE_ANSWER)

        retrieved: dict[str, EvidenceClaim] = {}
        stop = self._converse(evidence, question, catalog, recorder, retrieved, started)
        if stop.reason is not None:
            return self._failed(evidence, recorder, started, stop)
        return self._ground(run_id, evidence, catalog, route, retrieved, stop.claim_ids, recorder, started)

    # -- the loop -------------------------------------------------------------

    def _converse(
        self,
        evidence: ValidatedRunEvidence,
        question: str | None,
        catalog: tuple[EvidenceClaim, ...],
        recorder: TraceRecorder,
        retrieved: dict[str, EvidenceClaim],
        started: float,
    ) -> _Stop:
        limits = self._limits
        deadline = started + limits.total_timeout_seconds
        registry = ToolRegistry(evidence, catalog, self._tools)
        definitions = tool_definitions(self._tools)
        user_message = _user_message(evidence, question)
        messages: tuple[Mapping[str, Any], ...] = ({"role": "user", "content": user_message},)
        prompt_chars = len(AGENT_SYSTEM_INSTRUCTIONS) + len(json.dumps(definitions)) + len(user_message)

        for turn in range(1, limits.max_model_turns + 1):
            remaining = deadline - self._clock()
            if remaining <= 0:
                return _Stop(AgentStopReason.DEADLINE_EXCEEDED)
            recorder.peak_prompt_chars = prompt_chars
            if prompt_chars > self._settings.max_prompt_chars:
                return _Stop(AgentStopReason.PROMPT_BUDGET_EXCEEDED, ProviderFailureKind.PROMPT_BUDGET)

            timeout = min(limits.model_timeout_seconds, remaining)
            outcome = self._model_turn(
                AgentModelRequest(
                    system_instructions=AGENT_SYSTEM_INSTRUCTIONS,
                    tools=definitions,
                    messages=messages,
                    offered_claim_ids=tuple(claim.claim_id for claim in catalog),
                    max_output_tokens=self._settings.max_output_tokens,
                    timeout_seconds=timeout,
                ),
                recorder,
            )
            if isinstance(outcome, _Stop):
                return outcome
            if not outcome.tool_calls:
                return _Stop(claim_ids=outcome.claim_ids)

            refused = self._refusal(outcome, recorder, final_turn=turn == limits.max_model_turns)
            if refused is not None:
                return refused
            results = self._run_tools(outcome, registry, recorder, retrieved)
            prompt_chars += sum(len(result["content"]) for result in results)
            # The full content goes back unchanged, and every call is answered in one message.
            messages = (
                *messages,
                {"role": "assistant", "content": outcome.assistant_content},
                {"role": "user", "content": results},
            )
        return _Stop(AgentStopReason.TURN_BUDGET_EXHAUSTED)

    def _model_turn(self, request: AgentModelRequest, recorder: TraceRecorder) -> AgentReply | _Stop:
        called = self._clock()
        try:
            reply = self._model.complete(request)
        except ProviderTimeoutError as exc:
            recorder.record_turn(self._clock() - called, exc.telemetry)
            return _Stop(AgentStopReason.MODEL_TIMEOUT, ProviderFailureKind.TIMEOUT)
        except ProviderFailure as exc:
            if exc.kind not in _NEVER_SENT:
                recorder.record_turn(self._clock() - called, exc.telemetry)
            return _Stop(_FAILURE_STOPS.get(exc.kind, AgentStopReason.PROVIDER_ERROR), exc.kind)
        except Exception:  # a defect in an adapter must not reach the caller as an answer or a traceback
            recorder.record_turn(self._clock() - called, None)
            return _Stop(AgentStopReason.PROVIDER_ERROR, ProviderFailureKind.UNKNOWN)
        elapsed = self._clock() - called
        recorder.record_turn(elapsed, reply.telemetry)
        # Adapters enforce the deadline themselves, but a reply that is late anyway is discarded.
        if elapsed > request.timeout_seconds:
            return _Stop(AgentStopReason.MODEL_TIMEOUT, ProviderFailureKind.TIMEOUT)
        return reply

    def _refusal(self, reply: AgentReply, recorder: TraceRecorder, *, final_turn: bool) -> _Stop | None:
        """Whether this turn's calls must stop the request. Checked before any of them runs."""
        calls = reply.tool_calls
        stop = None
        if any(references_a_run(call.arguments) for call in calls):
            stop = _Stop(AgentStopReason.CROSS_RUN_REJECTED, ToolErrorCode.CROSS_RUN_REFERENCE)
        elif recorder.tool_call_count + len(calls) > self._limits.max_tool_calls:
            stop = _Stop(AgentStopReason.TOOL_BUDGET_EXHAUSTED)
        elif final_turn:  # a result could never be used, so nothing is retrieved for it
            stop = _Stop(AgentStopReason.TURN_BUDGET_EXHAUSTED)
        if stop is not None:
            recorder.tool_calls_refused = len(calls)
        return stop

    def _run_tools(
        self,
        reply: AgentReply,
        registry: ToolRegistry,
        recorder: TraceRecorder,
        retrieved: dict[str, EvidenceClaim],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in reply.tool_calls:
            called = self._clock()
            execution = registry.execute(call.name, call.arguments)
            recorder.record_tool(self._clock() - called, execution)
            retrieved.update({claim.claim_id: claim for claim in execution.claims})
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.call_id,
                "content": execution.content,
            }
            if execution.is_error:
                result["is_error"] = True
            results.append(result)
        return results

    # -- grounding ------------------------------------------------------------

    def _ground(
        self,
        run_id: str,
        evidence: ValidatedRunEvidence,
        catalog: tuple[EvidenceClaim, ...],
        route: QuestionRoute,
        retrieved: Mapping[str, EvidenceClaim],
        claim_ids: tuple[str, ...],
        recorder: TraceRecorder,
        started: float,
    ) -> AgentResult:
        """Publish a selection only if every id passes all five grounding conditions.

        An id must exist in the approved catalog, be eligible for this question, have
        been returned by a tool in this request, and pass citation validation against
        this run. The output-nature label is rendered first whatever was selected, so a
        model can never remove it, but selecting it still requires retrieving it.
        """
        known = tuple(claim_id for claim_id in claim_ids if claim_id in retrieved)
        if len(known) != len(claim_ids):
            return self._failed(
                evidence,
                recorder,
                started,
                _Stop(AgentStopReason.SELECTION_REJECTED),
                selected=known,
                selected_ids_rejected=len(claim_ids) - len(known),
            )
        try:
            shown = claims_to_render(catalog, known, route)
            rendered = None if shown is None else render_claims(shown)
            if rendered is not None:
                validate_citations(rendered[1], evidence, run_id)
        except ForgedCitationError:
            return self._failed(
                evidence, recorder, started, _Stop(AgentStopReason.CITATION_REJECTED), selected=known
            )
        if shown is None or rendered is None:
            return self._finish(
                unavailable_response(
                    evidence,
                    _PUBLIC_REASONS[AgentStopReason.NO_RELEVANT_CLAIM],
                    self._model.mode,
                    NO_CLAIM_ANSWER,
                ),
                recorder,
                started,
                AgentStopReason.NO_RELEVANT_CLAIM,
                selected=known,
            )
        answer, citations = rendered
        response = ExplanationResponse(
            run_id=evidence.run_id,
            answer=answer,
            support_level=support_level_for(evidence),
            output_nature=evidence.output_nature,
            model_mode=self._model.mode,
            citations=citations,
            limitations=list(evidence.limitations),
            unavailable_reason=None,
        )
        return self._finish(
            response,
            recorder,
            started,
            AgentStopReason.ANSWERED,
            selected=known,
            rendered=tuple(claim.claim_id for claim in shown),
        )

    # -- results --------------------------------------------------------------

    def _abstain(
        self,
        evidence: ValidatedRunEvidence,
        recorder: TraceRecorder,
        started: float,
        reason: AgentStopReason,
        answer: str,
    ) -> AgentResult:
        """A deterministic abstention: no model ran, so no token was spent."""
        response = unavailable_response(evidence, _PUBLIC_REASONS[reason], ModelMode.NOT_INVOKED, answer)
        return self._finish(response, recorder, started, reason)

    def _failed(
        self,
        evidence: ValidatedRunEvidence,
        recorder: TraceRecorder,
        started: float,
        stop: _Stop,
        *,
        selected: tuple[str, ...] = (),
        selected_ids_rejected: int = 0,
    ) -> AgentResult:
        assert stop.reason is not None
        response = unavailable_response(
            evidence, _PUBLIC_REASONS[stop.reason], self._mode(recorder), _AGENT_FAILED_ANSWER
        )
        return self._finish(
            response,
            recorder,
            started,
            stop.reason,
            error_code=stop.error_code,
            selected=selected,
            selected_ids_rejected=selected_ids_rejected,
        )

    def _mode(self, recorder: TraceRecorder) -> ModelMode:
        return self._model.mode if recorder.model_turns else ModelMode.NOT_INVOKED

    def _finish(
        self,
        response: ExplanationResponse,
        recorder: TraceRecorder,
        started: float,
        termination: AgentStopReason,
        *,
        error_code: ProviderFailureKind | ToolErrorCode | None = None,
        selected: tuple[str, ...] = (),
        rendered: tuple[str, ...] = (),
        selected_ids_rejected: int = 0,
    ) -> AgentResult:
        trace = recorder.build(
            termination,
            mode=response.model_mode,
            seconds=self._clock() - started,
            error_code=error_code,
            selected=selected,
            rendered=rendered,
            selected_ids_rejected=selected_ids_rejected,
        )
        return AgentResult(response, trace)


def _user_message(evidence: ValidatedRunEvidence, question: str | None) -> str:
    """The question and the run's safety fields. Claims only ever come from tools."""
    payload = build_provider_payload(evidence, (), question)
    del payload["claims"]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = ["AgentLimits", "AgentResult", "ReadOnlyAgent"]
