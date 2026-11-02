"""Operational summaries of agent traces: what happened, in numbers.

Everything is read from ``AgentTrace``, so it holds no prompt, question, answer, or
tool output. Latency is only reported when ``latency_measured`` is set, because a
scripted clock measures nothing. Tokens stay separated by ``UsageSource``, and money
is reported tokens at list price: an estimate, never a bill.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from footballai_v2.copilot.agent.trace import AgentStopReason, AgentTrace, UsageSource
from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.evaluation.metrics import latency_summary, proportion
from footballai_v2.copilot.models import ModelMode


COST_BASIS = "reported tokens at list price. An estimate, not a bill: only the Anthropic Console states that."
NOT_MEASURED = "not_measured: this run used a scripted clock"
_GROUNDING_FAILURES = {AgentStopReason.SELECTION_REJECTED, AgentStopReason.CITATION_REJECTED}


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _seconds(milliseconds: Iterable[float]) -> list[float]:
    return [value / 1000 for value in milliseconds]


def _latency(traces: Sequence[AgentTrace], first_use: frozenset[str]) -> dict[str, Any]:
    """End to end, per model call, and per tool. First-use schemas compile, so they are apart."""
    reached = [trace for trace in traces if trace.model_turns]
    repeat = [trace for trace in reached if trace.correlation_id not in first_use]
    tools: dict[str, list[float]] = {}
    for trace in traces:
        for call in trace.tool_calls:
            if call.tool is not None:
                tools.setdefault(call.tool.value, []).append(call.duration_ms)
    return {
        "percentiles": "nearest rank",
        "end_to_end_all_questions": latency_summary(_seconds(trace.duration_ms for trace in traces)),
        "end_to_end_reaching_a_model": latency_summary(_seconds(trace.duration_ms for trace in reached)),
        "end_to_end_repeat_schema": latency_summary(_seconds(trace.duration_ms for trace in repeat)),
        "end_to_end_first_use_schema": latency_summary(
            _seconds(trace.duration_ms for trace in reached if trace.correlation_id in first_use)
        ),
        "per_model_call": latency_summary(_seconds(turn.duration_ms for trace in traces for turn in trace.turns)),
        "per_tool": {name: latency_summary(_seconds(values)) for name, values in sorted(tools.items())},
    }


def _tokens(traces: Sequence[AgentTrace]) -> dict[str, Any]:
    measured = [trace for trace in traces if trace.usage_source is UsageSource.PROVIDER_REPORTED]
    partial = [trace for trace in traces if trace.usage_source is UsageSource.UNAVAILABLE]
    counted = [*measured, *partial]
    return {
        "note": "Only provider-reported tokens are counted. Scripted constants are never added in.",
        "usage_sources": _sorted_counts(trace.usage_source.value for trace in traces),
        "actual_input_tokens": sum(trace.usage.input_tokens for trace in counted),
        "actual_output_tokens": sum(trace.usage.output_tokens for trace in counted),
        "actual_cache_read_tokens": sum(trace.usage.cache_read_input_tokens for trace in counted),
        "actual_cache_write_tokens": sum(trace.usage.cache_creation_input_tokens for trace in counted),
        "totals_are_a_lower_bound": bool(partial),
        "questions_with_unknown_usage": len(partial),
    }


def _cost(traces: Sequence[AgentTrace]) -> dict[str, Any]:
    estimates = [trace.estimated_cost_usd for trace in traces if trace.estimated_cost_usd is not None]
    return {
        "basis": COST_BASIS,
        "estimated_total_usd": round(sum(estimates), 8) if estimates else None,
        "estimated_per_question_usd": round(sum(estimates) / len(estimates), 8) if estimates else None,
        "questions_estimated": len(estimates),
        "is_lower_bound": any(trace.cost_is_lower_bound for trace in traces),
    }


def summarize_traces(
    traces: Sequence[AgentTrace], *, latency_measured: bool, first_use: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """``first_use`` holds the correlation ids of questions that met a new output schema."""
    reached = [trace for trace in traces if trace.model_turns]
    calls = [call for trace in traces for call in trace.tool_calls]
    answered = [trace for trace in traces if trace.termination is AgentStopReason.ANSWERED]
    failures = [
        trace.error_code.value
        for trace in traces
        if isinstance(trace.error_code, ProviderFailureKind)
        and trace.error_code is not ProviderFailureKind.BUDGET_EXHAUSTED
        and trace.model_turns
    ]
    return {
        "questions": len(traces),
        "unique_correlation_ids": len({trace.correlation_id for trace in traces}),
        "questions_reaching_a_model": len(reached),
        "real_model_questions": sum(trace.model_mode is ModelMode.REAL_MODEL for trace in traces),
        "model_turns": {
            "total": sum(trace.model_turns for trace in traces),
            "mean_per_question_reaching_a_model": (
                round(sum(trace.model_turns for trace in reached) / len(reached), 4) if reached else None
            ),
            "max": max((trace.model_turns for trace in traces), default=0),
        },
        "tool_calls": {
            "valid": sum(trace.valid_tool_calls for trace in traces),
            "invalid": sum(trace.invalid_tool_calls for trace in traces),
            "refused_before_running": sum(trace.tool_calls_refused for trace in traces),
            "by_tool": _sorted_counts(call.tool.value for call in calls if call.tool is not None),
            "outside_the_registry": sum(call.tool is None for call in calls),
            "errors": _sorted_counts(call.error_code.value for call in calls if call.error_code is not None),
        },
        "terminations": _sorted_counts(trace.termination.value for trace in traces),
        "provider_failures_by_kind": _sorted_counts(failures),
        "grounding_success": proportion(
            len(answered), len(answered) + sum(trace.termination in _GROUNDING_FAILURES for trace in traces)
        ),
        "turns_aborted_at_their_deadline": sum(turn.aborted for trace in traces for turn in trace.turns),
        "refusal_categories": _sorted_counts(
            turn.refusal_category for trace in traces for turn in trace.turns if turn.refusal_category
        ),
        "latency": _latency(traces, first_use) if latency_measured else NOT_MEASURED,
        "tokens": _tokens(traces),
        "cost": _cost(traces),
    }


__all__ = ["COST_BASIS", "NOT_MEASURED", "summarize_traces"]
