"""The agent benchmark. Replay is the default, and it spends nothing.

Replay needs no SDK, key, or network, and writes a byte-reproducible report. Scripts
and a rule stand in for a model, so it checks the labels, the scorer, and the safety
machinery. It measures no model: its verdict is always ``NOT_A_MODEL_EVALUATION``.

A paid run needs an approval record for exactly that stage and run, and applies decision
rule v2 with scorer v2. Rule v1 and scorer v1 stay for the committed baseline. See
docs/v2/COPILOT_AGENT_REAL_MODEL.md. Exit codes: 2 refused, 3 preflight failed, 4
stopped early, with a report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from footballai_v2.copilot.agent.loop import AgentLimits
from footballai_v2.copilot.agent.tools import tool_definitions
from footballai_v2.copilot.budget import BudgetGuard
from footballai_v2.copilot.config import DEFAULT_MODEL
from footballai_v2.copilot.evaluation.agent_benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseMode,
    CaseScore,
    Outcome,
    Split,
    label_oracle_policy,
    load_benchmark,
    replay_script,
    run_case,
    score_metrics,
)
from footballai_v2.copilot.evaluation.agent_harness import EVALUATION_SETTINGS
from footballai_v2.copilot.evaluation.agent_live import (
    STAGE_SPLITS,
    LiveRefused,
    PlannedCase,
    agent_preflight,
    check_approval,
    current_provenance,
    endpoint_kind,
    load_approval,
    plan_benchmark,
    plan_cost,
    run_benchmark_live,
)
from footballai_v2.copilot.evaluation.agent_llmops import summarize_traces
from footballai_v2.copilot.evaluation.agent_models import (
    ScriptedAgentModel,
    every_tool_policy,
    route_following_policy,
    skip_retrieval_policy,
    wrong_tool_policy,
)
from footballai_v2.copilot.evaluation.agent_rule import (
    RULE_FILE_V2,
    DecisionRule,
    RuleStatus,
    evaluate_rule,
    load_rule,
)
from footballai_v2.copilot.evaluation.agent_scoring import (
    SCORER_VERSION,
    measurements_v2,
    score_case_v2,
    score_metrics_v2,
)
from footballai_v2.copilot.evaluation.fixtures import build_fixture_run
from footballai_v2.copilot.evaluation.live import PreflightFailed, real_settings
from footballai_v2.copilot.evaluation.metrics import proportion
from footballai_v2.copilot.evidence import RunEvidenceService
from footballai_v2.copilot.grounding import AGENT_INSTRUCTION_VERSION, build_claim_catalog
from footballai_v2.copilot.pricing import MODEL_PRICES, PRICING_SOURCE, PRICING_VERIFIED_ON


REPORT_SCHEMA = "footballai.copilot-agent-benchmark-report/v1"
DECISION_REPORT = "live-heldout.json"
_REPLAY_NOTE = (
    "Replay only. No language model was called, so nothing here measures model quality, "
    "latency, or real token use. label_oracle does exactly what the labels ask, which shows "
    "the labels can be reached. route_following is a rule. Hostile scripts show the agent "
    "fails closed. One author wrote every label, and none has had a second review."
)
_CONTROLS: dict[str, Callable[..., Any]] = {
    "every_tool_at_once": every_tool_policy,
    "always_the_wrong_tool": wrong_tool_policy,
    "skips_retrieval": skip_retrieval_policy,
}


def _live_cases(cases: tuple[BenchmarkCase, ...]) -> tuple[BenchmarkCase, ...]:
    return tuple(case for case in cases if case.mode is CaseMode.LIVE_ELIGIBLE)


def _replay_cases(cases: tuple[BenchmarkCase, ...]) -> tuple[BenchmarkCase, ...]:
    return tuple(case for case in cases if case.mode is CaseMode.REPLAY_ONLY)


def _with_policy(
    cases: tuple[BenchmarkCase, ...], dataset: BenchmarkDataset, policy_for: Callable[[BenchmarkCase], Any]
) -> tuple[CaseScore, ...]:
    return tuple(
        run_case(case, dataset.source, lambda _clock, case=case: ScriptedAgentModel(policy=policy_for(case)))
        for case in cases
    )


def _replayed(cases: tuple[BenchmarkCase, ...], dataset: BenchmarkDataset) -> tuple[CaseScore, ...]:
    return tuple(run_case(case, dataset.source, replay_script(case)) for case in cases)


def _counts(cases: tuple[BenchmarkCase, ...]) -> dict[str, Any]:
    by_category: dict[str, int] = {}
    for case in cases:
        by_category[case.category.value] = by_category.get(case.category.value, 0) + 1
    return {
        "cases": len(cases),
        "live_eligible": sum(case.mode is CaseMode.LIVE_ELIGIBLE for case in cases),
        "replay_only": sum(case.mode is CaseMode.REPLAY_ONLY for case in cases),
        "expecting_an_answer": sum(case.expected.outcome is Outcome.ANSWER for case in cases),
        "labels_reviewed_by_a_second_person": sum(case.label.review != "pending_second_person" for case in cases),
        "by_category": dict(sorted(by_category.items())),
    }


def _aggregate(metrics: dict[str, Any]) -> dict[str, Any]:
    """Case ids become counts, so a held-out split can be reported without being tuned against."""
    return {key: len(value) if isinstance(value, list) else value for key, value in metrics.items()}


def measurements(
    live: tuple[CaseScore, ...], replayed: tuple[CaseScore, ...], rule_scores: tuple[CaseScore, ...]
) -> dict[str, Any]:
    """What the decision rule reads. ``live`` was asked of a model, ``rule_scores`` of the rule."""
    every = score_metrics((*live, *replayed))
    only_live = score_metrics(live)
    rule_false_answers = {score.case.case_id for score in rule_scores if score.false_answer}
    live_false_answers = {score.case.case_id for score in live if score.false_answer}
    rule_right = {score.case.case_id for score in rule_scores if score.claims_correct}
    live_right = {score.case.case_id for score in live if score.outcome_correct}
    live_abstained = {score.case.case_id for score in live if score.false_abstention}
    asked = {score.case.case_id for score in live}
    corrected = rule_false_answers & live_right  # a failed call corrected nothing
    traces = [score.run.result.trace for score in live]
    latency = summarize_traces(traces, latency_measured=True)["latency"]
    return {
        "fabricated_answers": every["fabricated_answers"],
        "citation_validity": every["citation_validity"],
        "output_nature_accuracy": every["output_nature_accuracy"],
        "cross_run_violations": every["cross_run_violations"],
        "out_of_registry_executions": every["out_of_registry_executions"],
        "new_false_answers": sorted(live_false_answers - rule_false_answers),
        "new_false_abstentions": sorted(live_abstained & rule_right),
        "tool_selection_accuracy": only_live["tool_selection_accuracy"],
        "claim_selection_accuracy": only_live["claim_selection_accuracy"],
        "live_out_of_registry_attempts": only_live["out_of_registry_attempts"],
        "live_provider_failures_plus_invalid_outputs": only_live["provider_failures_plus_invalid_outputs"],
        "end_to_end_reaching_a_model": latency["end_to_end_reaching_a_model"],
        "share_of_deterministic_false_answers_corrected": proportion(
            len(corrected), len(rule_false_answers & asked)
        ),
    }


def _split_summary(
    dataset: BenchmarkDataset, split: Split, rule: DecisionRule, *, detailed: bool
) -> dict[str, Any]:
    cases = dataset.split(split)
    live, replay = _live_cases(cases), _replay_cases(cases)
    oracle = _with_policy(live, dataset, label_oracle_policy)
    following = _with_policy(live, dataset, lambda _case: route_following_policy)
    hostile = _replayed(replay, dataset)
    shape: Callable[[dict[str, Any]], dict[str, Any]] = (lambda value: value) if detailed else _aggregate
    traces = [score.run.result.trace for score in (*oracle, *hostile)]
    return {
        "sha256": dataset.sha256(split),
        "shape": _counts(cases),
        "label_oracle": shape(score_metrics(oracle)),
        "route_following": shape(score_metrics(following)),
        "hostile_replay": shape(score_metrics(hostile)),
        "operations": summarize_traces(traces, latency_measured=False),
        "decision_rule": evaluate_rule(rule, {}, mode="replay", endpoint=None, split=split.value, completed=True),
    }


def build_replay_report(dataset: BenchmarkDataset, rule: DecisionRule, rule_sha256: str) -> dict[str, Any]:
    limits = AgentLimits()
    settings = real_settings(DEFAULT_MODEL)
    live_dev = _live_cases(dataset.dev)
    return {
        "schema": REPORT_SCHEMA,
        "mode": "replay",
        "note": _REPLAY_NOTE,
        "live_model_requests": 0,
        "real_model_quality_measured": False,
        "dataset": {
            "version": dataset.version,
            "selection_dataset": dataset.source.version,
            "fixtures_sha256": dataset.source.fixtures_sha256,
        },
        "instruction_version": AGENT_INSTRUCTION_VERSION,
        "limits": dataclasses.asdict(limits),
        "tools": [definition["name"] for definition in tool_definitions()],
        "decision_rule": {"sha256": rule_sha256, "status": rule.status.value},
        "dev": _split_summary(dataset, Split.DEV, rule, detailed=True),
        "heldout": _split_summary(dataset, Split.HELDOUT, rule, detailed=False),
        "negative_controls": {
            name: _aggregate(score_metrics(_with_policy(live_dev, dataset, lambda _case, policy=policy: policy)))
            for name, policy in _CONTROLS.items()
        },
        "cost_plan": {
            split.value: plan_cost(plan_benchmark(dataset, split, settings), settings, limits.max_model_turns)
            for split in Split
        },
        "pricing": {"source": PRICING_SOURCE, "verified_on": PRICING_VERIFIED_ON},
    }


def _case_row(score: CaseScore) -> dict[str, Any]:
    """One live question. It holds ids and numbers: no prompt, question, or answer."""
    trace = score.run.result.trace
    return {
        "case_id": score.case.case_id,
        "correlation_id": trace.correlation_id,
        "passed": score.passed,
        "termination": trace.termination.value,
        "error_code": trace.error_code.value if trace.error_code is not None else None,
        "model_turns": trace.model_turns,
        "provider_request_ids": [turn.request_id for turn in trace.turns],
        "tools_executed": list(score.run.executed_tools),
        "claims_rendered": list(trace.claim_ids_rendered),
        "claims_correct": score.claims_correct,
        "tools_correct": score.tools_correct,
        "usage": trace.usage.model_dump(),
        "usage_source": trace.usage_source.value,
        "estimated_cost_usd": trace.estimated_cost_usd,
        "duration_ms": trace.duration_ms,
        "model_call_ms": trace.model_call_ms,
    }


def build_live_report(
    dataset: BenchmarkDataset,
    split: Split,
    rule: DecisionRule,
    rule_sha256: str,
    run: Any,
    guard: BudgetGuard,
    plan: tuple[PlannedCase, ...],
    model: str,
    endpoint: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scored by scorer v2 against ``rule`` (rule v2). Scorer v1's numbers sit beside, for comparison."""
    cases = dataset.split(split)
    asked = tuple(score.case for score in run.scores)
    planned_ids = {item.case.case_id for item in plan}
    # The router or missing evidence stops these before any model, so no model is needed
    # to score them, and they stay in the rule's denominators.
    stopped = _with_policy(
        tuple(case for case in _live_cases(cases) if case.case_id not in planned_ids),
        dataset,
        lambda _case: route_following_policy,
    )
    scored = (*run.scores, *stopped)
    following = _with_policy(tuple(score.case for score in scored), dataset, lambda _case: route_following_policy)
    hostile = _replayed(_replay_cases(cases), dataset)
    completed = run.stop_reason is None and len(run.scores) == len(plan)
    traces = [score.run.result.trace for score in run.scores]
    scored_v2 = tuple(score_case_v2(score.case, score.run) for score in scored)
    hostile_v2 = tuple(score_case_v2(score.case, score.run) for score in hostile)
    following_v2 = tuple(score_case_v2(score.case, score.run) for score in following)
    return {
        "schema": REPORT_SCHEMA,
        "mode": "live",
        "endpoint": endpoint,
        "model": model,
        "split": split.value,
        "cases_sha256": dataset.sha256(split),
        "decision_rule_sha256": rule_sha256,
        "instruction_version": AGENT_INSTRUCTION_VERSION,
        "scorer_version": SCORER_VERSION,
        "provenance": {**(provenance or {}), "model": model, "cases_sha256": dataset.sha256(split),
                       "decision_rule_sha256": rule_sha256},
        "live_model_requests": guard.requests_sent,
        "real_model_quality_measured": endpoint == "anthropic_api",
        "stop_reason": run.stop_reason,
        "questions_planned": len(plan),
        "questions_asked": len(asked),
        "questions_stopped_before_a_model": len(stopped),
        "budget": {
            "max_model_requests": guard.max_requests,
            "max_cost_usd": guard.max_cost_usd,
            "guard_charged_usd": round(guard.spent_usd, 8),
            "requests_charged_at_worst_case": guard.requests_at_worst_case,
            "note": "The guard's charge, not a bill. A timed-out request is charged at its worst case.",
        },
        "metrics": score_metrics_v2(scored_v2),
        "metrics_scorer_v1": score_metrics(scored),
        "hostile_replay": score_metrics_v2(hostile_v2),
        "operations": summarize_traces(traces, latency_measured=True, first_use=run.first_use),
        "decision_rule": evaluate_rule(
            rule,
            measurements_v2(scored_v2, hostile_v2, following_v2, latency_measured=True),
            mode="live",
            endpoint=endpoint,
            split=split.value,
            completed=completed,
        ),
        "cases": [_case_row(score) for score in run.scores],
    }


def show_fixture(dataset: BenchmarkDataset, name: str) -> dict[str, Any]:
    """A fixture's claims, so a reviewer can check a label without reading the code."""
    repository, storage, run_id = build_fixture_run(dataset.source.fixtures.fixtures[name], dataset.source.fixtures)
    evidence = RunEvidenceService(repository, storage, EVALUATION_SETTINGS).build(run_id)
    return {
        "fixture": name,
        "output_nature": evidence.output_nature.value,
        "claims": {claim.claim_id: claim.text for claim in build_claim_catalog(evidence)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="agent-v2 dataset directory")
    parser.add_argument("--output", type=Path, help="also write the report to this JSON file")
    parser.add_argument("--show-fixture", help="print one fixture's claims, for label review")
    parser.add_argument("--split", choices=[split.value for split in Split], default=Split.DEV.value)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--preflight", action="store_true", help="check the model and count tokens. Sends no generation")
    parser.add_argument("--allow-paid-calls", action="store_true", help="needs --stage, --approval, and both caps")
    parser.add_argument("--stage", choices=sorted(STAGE_SPLITS), help="the approved stage: B, C, or D")
    parser.add_argument("--approval", type=Path, help="an approval record for exactly this run")
    parser.add_argument("--max-requests", type=int, default=0, help="billed model calls this run may send")
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    return parser


def _emit(report: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if output is not None:
        output.write_text(text, encoding="utf-8")


def _check_output(args: argparse.Namespace, split: Split) -> None:
    """Paid results are never only on a screen, and a report is never replaced.

    The decision report has one fixed path, so a held-out result that exists blocks
    another attempt: the rule is applied once, and a failed run cannot be tried again.
    """
    if args.output is None:
        raise LiveRefused("output_required")
    if split is Split.HELDOUT and args.output.resolve() != (args.dataset / DECISION_REPORT).resolve():
        raise LiveRefused("decision_report_path")
    if args.output.exists():
        raise LiveRefused("output_exists")


def _paid_run(args: argparse.Namespace, dataset: BenchmarkDataset) -> int:
    split = Split(args.split)
    if args.approval is None:
        raise LiveRefused("not_approved")
    if args.stage is None:
        raise LiveRefused("stage_required")
    rule, rule_sha = load_rule(args.dataset, RULE_FILE_V2)
    if rule.scorer_version != SCORER_VERSION:
        raise LiveRefused("rule_other_scorer")
    provenance = current_provenance()
    check_approval(
        load_approval(args.approval),
        stage=args.stage,
        split=split,
        model=args.model,
        max_requests=args.max_requests,
        max_cost_usd=args.max_cost_usd,
        cases_sha256=dataset.sha256(split),
        rule_sha256=rule_sha,
        rule_accepted=rule.status is RuleStatus.ACCEPTED,
        provenance=provenance,
    )
    _check_output(args, split)
    settings = real_settings(args.model)
    plan = plan_benchmark(dataset, split, settings)
    agent_preflight(plan, settings)  # free, and required: a paid run never starts on an unverified model
    guard = BudgetGuard(args.max_requests, args.max_cost_usd, MODEL_PRICES[args.model])
    run = run_benchmark_live(plan, dataset, settings, guard)
    report = build_live_report(
        dataset, split, rule, rule_sha, run, guard, plan, args.model, endpoint_kind(), provenance
    )
    _emit(report, args.output)
    return 4 if run.stop_reason is not None else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = load_benchmark(args.dataset)
    rule, rule_sha = load_rule(args.dataset)
    if args.show_fixture:
        _emit(show_fixture(dataset, args.show_fixture), None)
        return 0
    try:
        if args.allow_paid_calls:
            return _paid_run(args, dataset)
        if args.preflight:
            split = Split(args.split)
            settings = real_settings(args.model)
            plan = plan_benchmark(dataset, split, settings)
            limits = AgentLimits()
            _emit({"preflight": agent_preflight(plan, settings), "plan": plan_cost(plan, settings, limits.max_model_turns)}, None)
            return 0
    except LiveRefused as exc:
        print(f"refused: {exc.code}")
        return 2
    except PreflightFailed as exc:
        print(f"preflight failed: {exc}")
        return 3
    _emit(build_replay_report(dataset, rule, rule_sha), args.output)
    return 0


__all__ = ["build_live_report", "build_replay_report", "main", "measurements"]


if __name__ == "__main__":
    raise SystemExit(main())
