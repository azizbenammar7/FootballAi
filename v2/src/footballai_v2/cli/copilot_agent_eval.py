"""Evaluate the bounded agent offline. Nothing here can reach a model.

This has no paid mode, needs no SDK, key, or network, and its report is
byte-reproducible: it holds no timestamp and no latency. Scripts and policies stand
in for the model, so it measures the agent's safety machinery, not model quality.
Real-model evaluation needs separate authorization and its own tooling (M3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from footballai_v2.copilot.agent.loop import AgentLimits
from footballai_v2.copilot.agent.tools import tool_definitions
from footballai_v2.copilot.config import DEFAULT_MODEL
from footballai_v2.copilot.evaluation.agent_dataset import AgentDataset, load_agent_dataset
from footballai_v2.copilot.evaluation.agent_harness import (
    EVALUATION_SETTINGS,
    AgentRun,
    AgentSelectionResult,
    ScriptedResult,
    answered_without_permission,
    run_scripted_case,
    run_selection_cases,
    stop_counts,
    tool_metrics,
)
from footballai_v2.copilot.evaluation.agent_models import (
    every_tool_policy,
    replay_text_policy,
    route_following_policy,
    skip_retrieval_policy,
    wrong_tool_policy,
)
from footballai_v2.copilot.evaluation.cost import MODEL_PRICES, PRICING_AS_OF, PRICING_SOURCE, agent_question_worst_case_usd
from footballai_v2.copilot.evaluation.dataset import Suite
from footballai_v2.copilot.evaluation.harness import run_cases
from footballai_v2.copilot.evaluation.metrics import (
    compare,
    contract_metrics,
    proportion,
    proportion_where,
    selection_metrics,
)
from footballai_v2.copilot.evaluation.routers import routing_for
from footballai_v2.copilot.grounding import AGENT_INSTRUCTION_VERSION
from footballai_v2.copilot.provider import DeterministicFakeProvider


REPORT_SCHEMA = "footballai.copilot-agent-eval-report/v1"
_NOTE = (
    "Offline and deterministic. No language model was called, so nothing here measures model "
    "quality. Tool-selection and relevance scores describe scripts and a rule the authors wrote "
    "alongside the labels: they show that the agent's plumbing and safety checks work, and the "
    "negative controls show that these metrics can fail."
)
_CONTROLS: dict[str, Callable[..., Any]] = {
    "every_tool_at_once": every_tool_policy,
    "always_the_wrong_tool": wrong_tool_policy,
    "skips_retrieval": skip_retrieval_policy,
}


def _deterministic(report: dict[str, Any]) -> dict[str, Any]:
    """Without latency, so the same code always writes the same bytes."""
    return {key: value for key, value in report.items() if key != "latency_in_process"}


def _policy_metrics(results: tuple[AgentSelectionResult, ...]) -> dict[str, Any]:
    cases = tuple(result.case for result in results)
    runs = [result.run for result in results]
    return {
        **_deterministic(selection_metrics(cases)),
        "tool_selection_accuracy": proportion_where(results, lambda r: r.tool_selection_correct),
        "tool_call_efficiency": tool_metrics(runs),
        "stops": stop_counts(runs),
    }


def _scripted_suite(results: tuple[ScriptedResult, ...]) -> dict[str, Any]:
    runs = [result.run for result in results]
    groups = sorted({result.case.group for result in results})
    labelled = [result for result in results if result.case.expected.required_tools]
    answered = [result for result in results if result.run.result.response.unavailable_reason is None]
    rendered = sum(len(_rendered(result)) for result in answered)
    unpermitted = sum(
        len(_rendered(result) - set(result.case.expected.permitted_claims)) for result in answered
    )
    failure_groups = {"adversarial", "limits", "failure_handling"}
    hostile = [result for result in results if result.case.group in failure_groups]
    return {
        "cases": len(results),
        "all_expectations_met": proportion(sum(result.passed for result in results), len(results)),
        "failed_cases": sorted(result.case.case_id for result in results if not result.passed),
        "by_group": {
            group: proportion(
                sum(result.passed for result in results if result.case.group == group),
                sum(result.case.group == group for result in results),
            )
            for group in groups
        },
        "tool_selection_accuracy": proportion(
            sum("required_tools" not in r.failures and "forbidden_tools" not in r.failures for r in labelled),
            len(labelled),
        ),
        "tool_call_efficiency": tool_metrics(runs),
        "stops": stop_counts(runs),
        "unsupported_claim_rate": {
            "definition": "rendered claims a case does not permit / rendered claims, excluding the label",
            "unpermitted": unpermitted,
            "rendered": rendered,
            "rate": round(unpermitted / rendered, 4) if rendered else None,
        },
        "citation_validity": proportion(
            sum("citations" not in result.failures for result in answered), len(answered)
        ),
        "answers_not_labelled_as_answers": answered_without_permission(results),
        "fail_closed_cases": proportion(sum(result.passed for result in hostile), len(hostile)),
        "protocol_violations": sum("protocol" in result.failures for result in results),
    }


def _rendered(result: ScriptedResult) -> set[str]:
    return set(result.run.result.trace.claim_ids_rendered) - {"output_nature"}


def _cost_plan(peak_prompt_chars: int, limits: AgentLimits) -> dict[str, Any]:
    settings = EVALUATION_SETTINGS
    return {
        "note": "Planning only. An upper bound at list price, not a measurement or a bill.",
        "model": DEFAULT_MODEL,
        "peak_prompt_chars_in_this_suite": peak_prompt_chars,
        "prompt_budget_chars": settings.max_prompt_chars,
        "worst_case_usd_per_question": round(
            agent_question_worst_case_usd(
                MODEL_PRICES[DEFAULT_MODEL],
                settings.max_prompt_chars,
                settings.max_output_tokens,
                limits.max_model_turns,
            ),
            6,
        ),
        "pricing": {"source": PRICING_SOURCE, "as_of": PRICING_AS_OF},
    }


def build_report(dataset: AgentDataset) -> dict[str, Any]:
    source = dataset.source
    selection = source.suite(Suite.SELECTION)
    contract = source.suite(Suite.CONTRACT)
    limits = AgentLimits()

    scripted = tuple(run_scripted_case(case, dataset) for case in dataset.cases)
    following = run_selection_cases(selection, source, lambda _case: route_following_policy)
    baseline = run_cases(selection, source, lambda _case: DeterministicFakeProvider())
    replayed = run_selection_cases(contract, source, lambda case: replay_text_policy(case.model_output or ""))
    controls = {
        name: _policy_metrics(run_selection_cases(selection, source, lambda _case, policy=policy: policy))
        for name, policy in _CONTROLS.items()
    }
    runs: list[AgentRun] = [*(result.run for result in scripted), *(result.run for result in following)]

    return {
        "schema": REPORT_SCHEMA,
        "mode": "dry_run",
        "note": _NOTE,
        "live_model_requests": 0,
        "dataset": {
            "version": dataset.version,
            "cases_sha256": dataset.cases_sha256,
            "selection_dataset": source.version,
            "selection_cases_sha256": source.cases_sha256,
            "fixtures_sha256": source.fixtures_sha256,
        },
        "instruction_version": AGENT_INSTRUCTION_VERSION,
        "routing_version": routing_for(source.version)[0],
        "limits": dataclasses.asdict(limits),
        "tools": [definition["name"] for definition in tool_definitions()],
        "scripted_suite": _scripted_suite(scripted),
        "deterministic_comparison": {
            "note": (
                "route_following is a rule, not a model. Agreement with deterministic_fake shows "
                "that the agent path loses nothing, and says nothing about a model's judgement."
            ),
            "deterministic_fake": _deterministic(selection_metrics(baseline)),
            "route_following": _policy_metrics(following),
            "paired": compare(baseline, tuple(result.case for result in following)),
        },
        "negative_controls": controls,
        "invalid_reply_replay": contract_metrics(tuple(result.case for result in replayed)),
        "cost_plan": _cost_plan(max(run.result.trace.peak_prompt_chars for run in runs), limits),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="agent dataset version directory")
    parser.add_argument("--output", type=Path, help="also write the report to this JSON file")
    args = parser.parse_args(argv)
    text = json.dumps(build_report(load_agent_dataset(args.dataset)), indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
