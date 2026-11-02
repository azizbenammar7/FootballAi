"""A bounded, schema-validated record of one agent request.

The trace holds what happened, never what was said: tool names, statuses,
timings, closed-vocabulary claim ids, and token counts. A value that came from a
model is recorded only after it has been mapped onto a closed set, so a hostile
reply cannot write into a trace. It holds no prompt, question, tool output,
artifact content, storage path, credential, or exception text.

``AgentTrace`` is the operator's record, and goes to the log. A client receives
``PublicAgentTrace``, which also leaves out provider failure classes, provider
request ids, and token counts. The offline stub reports constants for those, and
publishing them would present stub values as measurements.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from footballai_v2.copilot.agent.tools import ToolCallStatus, ToolErrorCode, ToolExecution, ToolName
from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.models import ExplanationResponse, ModelMode, ProviderTelemetry, TokenUsage

if TYPE_CHECKING:
    from footballai_v2.copilot.agent.loop import AgentLimits


TRACE_SCHEMA = "footballai.copilot-agent-trace/v1"
PUBLIC_TRACE_SCHEMA = "footballai.copilot-agent-public-trace/v1"
MAX_TRACED_TOOL_CALLS = 4
MAX_TRACED_TURNS = 5
_MAX_CLAIMS = 32

_KNOWN_STOP_REASONS = frozenset(
    {"end_turn", "tool_use", "max_tokens", "refusal", "pause_turn", "stop_sequence", "model_context_window_exceeded"}
)
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")

ClaimId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
CorrelationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]


class AgentStopReason(StrEnum):
    """Why a request ended. Every path out of the loop is one of these."""

    ANSWERED = "answered"
    UNSUPPORTED_QUESTION = "unsupported_question"
    EVIDENCE_NOT_AVAILABLE = "evidence_not_available"
    NO_RELEVANT_CLAIM = "no_relevant_claim"
    SELECTION_REJECTED = "selection_rejected"
    CITATION_REJECTED = "citation_rejected"
    CROSS_RUN_REJECTED = "cross_run_rejected"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    TURN_BUDGET_EXHAUSTED = "turn_budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    PROMPT_BUDGET_EXCEEDED = "prompt_budget_exceeded"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_REFUSAL = "model_refusal"
    MODEL_TRUNCATED = "model_truncated"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_ERROR = "provider_error"


class AgentAdapter(StrEnum):
    """What drives the agent. Fixed for a request, so an abstention keeps its label."""

    OFFLINE_SCRIPTED = "offline_scripted"
    REAL_MODEL = "real_model"

    @classmethod
    def for_mode(cls, mode: ModelMode) -> AgentAdapter:
        if mode is ModelMode.OFFLINE_DEMO_FAKE:
            return cls.OFFLINE_SCRIPTED
        if mode is ModelMode.REAL_MODEL:
            return cls.REAL_MODEL
        raise ValueError(f"{mode.value} does not describe an adapter")


class _TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TraceUsage(_TraceModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)


class ToolCallTrace(_TraceModel):
    sequence: int = Field(ge=1, le=MAX_TRACED_TOOL_CALLS)
    turn: int = Field(ge=1, le=MAX_TRACED_TURNS)
    tool: ToolName | None  # None: the model named something outside the registry
    status: ToolCallStatus
    duration_ms: float = Field(ge=0)
    claim_ids_returned: tuple[ClaimId, ...] = Field(default=(), max_length=_MAX_CLAIMS)
    error_code: ToolErrorCode | None = None


class ModelTurnTrace(_TraceModel):
    turn: int = Field(ge=1, le=MAX_TRACED_TURNS)
    duration_ms: float = Field(ge=0)
    stop_reason: str | None = Field(default=None, max_length=40)
    request_id: str | None = Field(default=None, max_length=128)
    usage: TraceUsage | None = None


class AgentTrace(_TraceModel):
    schema_version: str = Field(default=TRACE_SCHEMA, pattern=f"^{re.escape(TRACE_SCHEMA)}$")
    correlation_id: CorrelationId
    run_id: str = Field(max_length=64)
    model: str = Field(max_length=64)
    model_mode: ModelMode
    instruction_version: str = Field(max_length=64)
    routing_version: str = Field(max_length=64)
    termination: AgentStopReason
    error_code: ProviderFailureKind | ToolErrorCode | None = None
    duration_ms: float = Field(ge=0)
    model_turns: int = Field(ge=0, le=MAX_TRACED_TURNS)
    turns: tuple[ModelTurnTrace, ...] = Field(max_length=MAX_TRACED_TURNS)
    tool_calls: tuple[ToolCallTrace, ...] = Field(max_length=MAX_TRACED_TOOL_CALLS)
    tool_calls_refused: int = Field(ge=0)  # requested, and stopped before any ran
    claim_ids_returned: tuple[ClaimId, ...] = Field(max_length=_MAX_CLAIMS)
    claim_ids_selected: tuple[ClaimId, ...] = Field(max_length=_MAX_CLAIMS)
    claim_ids_rendered: tuple[ClaimId, ...] = Field(max_length=_MAX_CLAIMS)
    selected_ids_rejected: int = Field(ge=0)  # counted, never quoted: they are model text
    usage: TraceUsage
    usage_complete: bool  # False: some turn reported none, so accounting is a lower bound
    peak_prompt_chars: int = Field(ge=0)


class PublicAgentTrace(_TraceModel):
    """What a client may see of one request. Every entry is a call that really happened."""

    schema_version: str = Field(default=PUBLIC_TRACE_SCHEMA, pattern=f"^{re.escape(PUBLIC_TRACE_SCHEMA)}$")
    correlation_id: CorrelationId
    run_id: str = Field(max_length=64)
    adapter: AgentAdapter
    model: str = Field(max_length=64)
    model_mode: ModelMode
    instruction_version: str = Field(max_length=64)
    routing_version: str = Field(max_length=64)
    termination: AgentStopReason
    tool_error_code: ToolErrorCode | None = None
    duration_ms: float = Field(ge=0)
    model_turns: int = Field(ge=0, le=MAX_TRACED_TURNS)
    model_turn_limit: int = Field(ge=1, le=MAX_TRACED_TURNS)
    tool_call_limit: int = Field(ge=0, le=MAX_TRACED_TOOL_CALLS)
    tool_calls: tuple[ToolCallTrace, ...] = Field(max_length=MAX_TRACED_TOOL_CALLS)
    tool_calls_refused: int = Field(ge=0)
    claim_ids_returned: tuple[ClaimId, ...] = Field(max_length=_MAX_CLAIMS)
    claim_ids_selected: tuple[ClaimId, ...] = Field(max_length=_MAX_CLAIMS)
    claim_ids_rendered: tuple[ClaimId, ...] = Field(max_length=_MAX_CLAIMS)
    selected_ids_rejected: int = Field(ge=0)


class AgentResponse(_TraceModel):
    """The agent route's reply: the existing explanation contract, plus its public trace."""

    explanation: ExplanationResponse
    trace: PublicAgentTrace


def publish_trace(trace: AgentTrace, adapter: AgentAdapter, limits: AgentLimits) -> PublicAgentTrace:
    """Project a request's trace for a client. Provider failure classes stay internal."""
    return PublicAgentTrace(
        correlation_id=trace.correlation_id,
        run_id=trace.run_id,
        adapter=adapter,
        model=trace.model,
        model_mode=trace.model_mode,
        instruction_version=trace.instruction_version,
        routing_version=trace.routing_version,
        termination=trace.termination,
        tool_error_code=trace.error_code if isinstance(trace.error_code, ToolErrorCode) else None,
        duration_ms=trace.duration_ms,
        model_turns=trace.model_turns,
        model_turn_limit=limits.max_model_turns,
        tool_call_limit=limits.max_tool_calls,
        tool_calls=trace.tool_calls,
        tool_calls_refused=trace.tool_calls_refused,
        claim_ids_returned=trace.claim_ids_returned,
        claim_ids_selected=trace.claim_ids_selected,
        claim_ids_rendered=trace.claim_ids_rendered,
        selected_ids_rejected=trace.selected_ids_rejected,
    )


def _milliseconds(seconds: float) -> float:
    return round(max(seconds, 0.0) * 1000, 3)


def _trace_usage(usage: TokenUsage | None) -> TraceUsage | None:
    if usage is None:
        return None
    return TraceUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
    )


class TraceRecorder:
    """Collects one request's facts, reducing model-supplied values to closed sets."""

    def __init__(self, correlation_id: str, run_id: str, model: str, instruction_version: str, routing_version: str):
        self._identity = {
            "correlation_id": correlation_id,
            "run_id": run_id,
            "model": model[:64],  # configuration, so a long value must not fail a finished request
            "instruction_version": instruction_version,
            "routing_version": routing_version,
        }
        self._turns: list[ModelTurnTrace] = []
        self._tool_calls: list[ToolCallTrace] = []
        self._returned: dict[str, None] = {}
        self._usage = TraceUsage()
        self._usage_complete = True
        self.tool_calls_refused = 0
        self.peak_prompt_chars = 0

    @property
    def model_turns(self) -> int:
        return len(self._turns)

    @property
    def tool_call_count(self) -> int:
        return len(self._tool_calls)

    def record_turn(self, seconds: float, telemetry: ProviderTelemetry | None) -> None:
        usage = _trace_usage(telemetry.usage) if telemetry else None
        stop_reason = telemetry.stop_reason if telemetry else None
        request_id = telemetry.request_id if telemetry else None
        self._turns.append(
            ModelTurnTrace(
                turn=self.model_turns + 1,
                duration_ms=_milliseconds(seconds),
                stop_reason=(
                    stop_reason
                    if isinstance(stop_reason, str) and stop_reason in _KNOWN_STOP_REASONS
                    else None
                ),
                request_id=(
                    request_id if isinstance(request_id, str) and _REQUEST_ID_RE.fullmatch(request_id) else None
                ),
                usage=usage,
            )
        )
        if usage is None:
            self._usage_complete = False
            return
        self._usage = TraceUsage(
            **{name: getattr(self._usage, name) + getattr(usage, name) for name in TraceUsage.model_fields}
        )

    def record_tool(self, seconds: float, execution: ToolExecution) -> None:
        returned = tuple(claim.claim_id for claim in execution.claims)
        self._returned.update(dict.fromkeys(returned))
        self._tool_calls.append(
            ToolCallTrace(
                sequence=self.tool_call_count + 1,
                turn=self.model_turns,
                tool=execution.tool,
                status=execution.status,
                duration_ms=_milliseconds(seconds),
                claim_ids_returned=returned,
                error_code=execution.error_code,
            )
        )

    def build(
        self,
        termination: AgentStopReason,
        *,
        mode: ModelMode,
        seconds: float,
        error_code: ProviderFailureKind | ToolErrorCode | None = None,
        selected: tuple[str, ...] = (),
        rendered: tuple[str, ...] = (),
        selected_ids_rejected: int = 0,
    ) -> AgentTrace:
        return AgentTrace(
            **self._identity,
            model_mode=mode,
            termination=termination,
            error_code=error_code,
            duration_ms=_milliseconds(seconds),
            model_turns=self.model_turns,
            turns=tuple(self._turns),
            tool_calls=tuple(self._tool_calls),
            tool_calls_refused=self.tool_calls_refused,
            claim_ids_returned=tuple(self._returned),
            claim_ids_selected=selected,
            claim_ids_rendered=rendered,
            selected_ids_rejected=selected_ids_rejected,
            usage=self._usage,
            usage_complete=self._usage_complete,
            peak_prompt_chars=self.peak_prompt_chars,
        )


__all__ = [
    "AgentAdapter",
    "AgentResponse",
    "AgentStopReason",
    "AgentTrace",
    "MAX_TRACED_TOOL_CALLS",
    "MAX_TRACED_TURNS",
    "ModelTurnTrace",
    "PUBLIC_TRACE_SCHEMA",
    "PublicAgentTrace",
    "TRACE_SCHEMA",
    "ToolCallTrace",
    "TraceRecorder",
    "TraceUsage",
    "publish_trace",
]
