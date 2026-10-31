"""Aggregate case results into metrics that always report their support."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.evaluation.harness import ANSWERED, CaseResult


_Z_95 = 1.96


def _rate(correct: int, total: int) -> dict[str, Any]:
    """A proportion with its count and a Wilson 95% interval. Small samples are
    wide, and the interval says so."""
    if total == 0:
        return {"correct": 0, "n": 0, "rate": None, "ci95": None}
    rate = correct / total
    z2 = _Z_95**2
    centre = (rate + z2 / (2 * total)) / (1 + z2 / total)
    margin = _Z_95 * math.sqrt(rate * (1 - rate) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return {
        "correct": correct,
        "n": total,
        "rate": round(rate, 4),
        "ci95": [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)],
    }


def _count(results: Iterable[CaseResult], check: Callable[[CaseResult], bool | None]) -> dict[str, Any]:
    """Rate of ``check`` over results it applies to (``None`` means not applicable)."""
    values = [value for value in (check(result) for result in results) if value is not None]
    return _rate(sum(values), len(values))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[max(index, 0)], 6)


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": _percentile(values, 1.0),
    }


def selection_metrics(results: tuple[CaseResult, ...]) -> dict[str, Any]:
    answered = [result for result in results if result.outcome == ANSWERED]
    rendered = sum(len(result.rendered_claims) for result in answered)
    unpermitted = sum(len(result.unpermitted_claims) for result in answered)
    invoked = [result for result in results if result.provider_invoked]
    invalid = [
        result for result in invoked if result.failure_kind == ProviderFailureKind.INVALID_RESPONSE.value
    ]
    failures = [result for result in invoked if result.failure_kind is not None]
    false_answers = [r for r in answered if not r.outcome_correct]
    false_abstentions = [r for r in results if r.outcome != ANSWERED and not r.outcome_correct]
    return {
        "cases": len(results),
        "claim_selection_accuracy": _count(results, lambda r: r.selection_correct),
        "abstention_accuracy": _count(results, lambda r: r.outcome_correct),
        "false_answers": sorted(r.case_id for r in false_answers),
        "false_abstentions": sorted(r.case_id for r in false_abstentions),
        "unsupported_claim_rate": {
            "definition": "rendered claims a case does not permit / rendered claims, excluding the label",
            "unpermitted": unpermitted,
            "rendered": rendered,
            "rate": round(unpermitted / rendered, 4) if rendered else None,
        },
        "fabricated_answers": sum(not r.answer_server_rendered for r in results),
        "citation_validity": _count(answered, lambda r: r.citations_valid),
        "expected_citations_present": _count(results, lambda r: r.expected_citations_present),
        "support_level_accuracy": _count(results, lambda r: r.support_level_correct),
        "output_nature_accuracy": _count(results, lambda r: r.output_nature_correct),
        "forbidden_text_absent": _count(results, lambda r: r.forbidden_text_absent),
        "provider_invocations": len(invoked),
        "provider_failures": len(failures),
        "invalid_output_rate": _rate(len(invalid), len(invoked)),
        "latency_in_process": latency_summary([r.latency_seconds for r in results]),
        "by_category": {
            category: {
                "cases": sum(r.category == category for r in results),
                "abstention_accuracy": _count((r for r in results if r.category == category), lambda r: r.outcome_correct),
                "claim_selection_accuracy": _count(
                    (r for r in results if r.category == category), lambda r: r.selection_correct
                ),
            }
            for category in sorted({r.category for r in results})
        },
    }


def contract_metrics(results: tuple[CaseResult, ...]) -> dict[str, Any]:
    passed = _count(
        results,
        lambda r: r.outcome_correct and r.citations_valid and r.forbidden_text_absent and r.outcome != ANSWERED,
    )
    return {"fail_closed": passed, "failed_cases": sorted(r.case_id for r in results if not r.outcome_correct)}


def headroom(baseline: tuple[CaseResult, ...]) -> dict[str, Any]:
    """What a perfect selector could gain, measured without calling a model.

    A selector only acts on cases that reach it, and only helps where the
    baseline is wrong. Everything else is out of its reach by construction.
    """
    wrong = [r for r in baseline if not (r.outcome_correct and r.selection_correct is not False)]
    reachable = [r for r in wrong if r.provider_invoked]
    scored = [r for r in baseline if r.selection_correct is not None]
    correct = sum(bool(r.selection_correct) for r in scored)
    reachable_selection = sum(r.selection_correct is False and r.provider_invoked for r in scored)
    invoked = [r for r in baseline if r.provider_invoked]
    gaps = [r for r in baseline if r.coverage_gap]
    return {
        "selection_cases": len(baseline),
        "reach_the_selector": len(invoked),
        "stopped_before_the_selector": len(baseline) - len(invoked),
        "improvable_cases": sorted(r.case_id for r in reachable),
        "not_improvable_wrong_cases": sorted(r.case_id for r in wrong if not r.provider_invoked),
        "cases_where_a_selector_can_only_tie_or_lose": len(invoked) - len(reachable),
        "baseline_selection_accuracy": _rate(correct, len(scored)),
        "perfect_selector_selection_accuracy": _rate(correct + reachable_selection, len(scored)),
        "coverage_gap_cases": sorted(r.case_id for r in gaps),
        "coverage_gaps_blocked_before_selector": sorted(r.case_id for r in gaps if not r.provider_invoked),
    }


def compare(reference: tuple[CaseResult, ...], candidate: tuple[CaseResult, ...]) -> dict[str, Any]:
    """Paired comparison on the same cases. Only discordant cases carry evidence."""
    by_id = {result.case_id: result for result in reference}
    improved: list[str] = []
    regressed: list[str] = []
    for result in candidate:
        before = by_id[result.case_id]
        was_right = before.outcome_correct and before.selection_correct is not False
        is_right = result.outcome_correct and result.selection_correct is not False
        if is_right and not was_right:
            improved.append(result.case_id)
        elif was_right and not is_right:
            regressed.append(result.case_id)
    return {
        "paired_cases": len(candidate),
        "improved": sorted(improved),
        "regressed": sorted(regressed),
        "unchanged": len(candidate) - len(improved) - len(regressed),
    }


__all__ = ["compare", "contract_metrics", "headroom", "latency_summary", "selection_metrics"]
