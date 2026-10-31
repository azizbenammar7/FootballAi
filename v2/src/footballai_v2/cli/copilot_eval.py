"""Evaluate copilot claim selection. Dry run by default; never reachable from the API.

Default          baselines plus a cost plan. No SDK, key, or network.
--preflight      verifies the model and counts input tokens. Needs a key. Not billed.
--allow-paid-calls  sends real requests. Needs --model, --max-requests, and
                 --max-cost-usd, and refuses to run in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from footballai_v2.copilot.config import DEFAULT_MODEL
from footballai_v2.copilot.evaluation.cost import MODEL_PRICES, PRICING_AS_OF, PRICING_SOURCE
from footballai_v2.copilot.evaluation.dataset import EvaluationDataset, Suite, load_dataset
from footballai_v2.copilot.evaluation.harness import CaseResult, run_cases
from footballai_v2.copilot.evaluation.live import (
    BudgetGuard,
    PlannedRequest,
    PreflightFailed,
    estimate_plan,
    plan_live_run,
    preflight,
    real_settings,
    run_live,
)
from footballai_v2.copilot.evaluation.metrics import (
    compare,
    contract_metrics,
    headroom,
    latency_summary,
    selection_metrics,
)
from footballai_v2.copilot.evaluation.selectors import LexicalOverlapProvider, ScriptedProvider
from footballai_v2.copilot.grounding import INSTRUCTION_VERSION
from footballai_v2.copilot.models import ProviderTelemetry
from footballai_v2.copilot.provider import DeterministicFakeProvider


REPORT_SCHEMA = "footballai.copilot-eval-report/v1"
EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_PREFLIGHT_FAILED = 3
EXIT_RUN_INCOMPLETE = 4
# A typo should not be able to authorize a large spend.
HARD_MAX_REQUESTS = 200
HARD_MAX_COST_USD = 5.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset version directory")
    parser.add_argument("--output", type=Path, help="also write the report to this JSON file")
    parser.add_argument("--model", help=f"model id (default for planning: {DEFAULT_MODEL})")
    parser.add_argument("--sample", type=int, help="stratified sample of planned requests")
    parser.add_argument("--preflight", action="store_true", help="verify the model; sends no billed request")
    parser.add_argument("--allow-paid-calls", action="store_true", help="send real, billed requests")
    parser.add_argument("--max-requests", type=int, help="hard request limit for a paid run")
    parser.add_argument("--max-cost-usd", type=float, help="hard list-price budget for a paid run")
    return parser


def _refusal(args: argparse.Namespace) -> str | None:
    """Why this invocation must not proceed, or ``None``. Checked before any work."""
    model = args.model or DEFAULT_MODEL
    if model not in MODEL_PRICES:
        return f"no documented price is configured for {model!r}"
    if args.sample is not None and args.sample < 1:
        return "--sample must be positive"
    if not args.allow_paid_calls:
        if args.max_requests is not None or args.max_cost_usd is not None:
            return "--max-requests and --max-cost-usd only apply with --allow-paid-calls"
        return None
    if args.preflight:
        return "--preflight sends no billed request, so it cannot be combined with --allow-paid-calls"
    if os.getenv("CI", "").strip().lower() not in {"", "0", "false"} or os.getenv("GITHUB_ACTIONS"):
        return "paid calls are refused in CI"
    if args.model is None:
        return "--allow-paid-calls requires an explicit --model"
    if args.max_requests is None or not 0 < args.max_requests <= HARD_MAX_REQUESTS:
        return f"--allow-paid-calls requires --max-requests between 1 and {HARD_MAX_REQUESTS}"
    if args.max_cost_usd is None or not 0 < args.max_cost_usd <= HARD_MAX_COST_USD:
        return f"--allow-paid-calls requires --max-cost-usd above 0 and at most {HARD_MAX_COST_USD}"
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "--allow-paid-calls requires ANTHROPIC_API_KEY"
    return None


def _baseline(dataset: EvaluationDataset) -> dict[str, tuple[CaseResult, ...]]:
    selection = dataset.suite(Suite.SELECTION)
    return {
        "deterministic_fake": run_cases(selection, dataset, lambda _case: DeterministicFakeProvider()),
        "lexical_overlap": run_cases(selection, dataset, lambda _case: LexicalOverlapProvider()),
    }


def _report_base(dataset: EvaluationDataset, mode: str) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "version": dataset.version,
            "cases_sha256": dataset.cases_sha256,
            "fixtures_sha256": dataset.fixtures_sha256,
            "selection_cases": len(dataset.suite(Suite.SELECTION)),
            "contract_cases": len(dataset.suite(Suite.CONTRACT)),
        },
        "instruction_version": INSTRUCTION_VERSION,
        "pricing": {"source": PRICING_SOURCE, "as_of": PRICING_AS_OF},
    }


def _offline_report(dataset: EvaluationDataset, plan: tuple[PlannedRequest, ...], model: str) -> dict[str, Any]:
    baselines = _baseline(dataset)
    contract = run_cases(dataset.suite(Suite.CONTRACT), dataset, ScriptedProvider)
    settings = real_settings(model)
    return {
        **_report_base(dataset, "dry_run"),
        "live_model_requests": 0,
        "baselines": {name: selection_metrics(results) for name, results in baselines.items()},
        "contract_suite": contract_metrics(contract),
        "selector_headroom": headroom(baselines["deterministic_fake"]),
        "lexical_vs_fake": compare(baselines["deterministic_fake"], baselines["lexical_overlap"]),
        "planned_live_run": {
            "model": model,
            "effort": settings.effort,
            "max_output_tokens": settings.max_output_tokens,
            "timeout_seconds": settings.timeout_seconds,
            "retries": settings.provider_retries,
            "estimate": estimate_plan(plan, MODEL_PRICES[model], settings.max_output_tokens),
            "case_ids": [item.case.case_id for item in plan],
        },
    }


def _api_latency(results: tuple[CaseResult, ...], plan: tuple[PlannedRequest, ...]) -> dict[str, Any]:
    """Provider latency, split on a schema's first use: new schemas compile a grammar."""
    schemas = {item.case.case_id: item.catalog_ids for item in plan}
    first_use: list[float] = []
    repeat_use: list[float] = []
    seen: set[tuple[str, ...]] = set()
    for result in results:
        telemetry: ProviderTelemetry | None = result.telemetry
        if telemetry is None:
            continue
        schema = schemas[result.case_id]
        (repeat_use if schema in seen else first_use).append(telemetry.latency_seconds)
        seen.add(schema)
    return {"schema_first_use": latency_summary(first_use), "schema_repeat_use": latency_summary(repeat_use)}


def _live_report(
    dataset: EvaluationDataset,
    plan: tuple[PlannedRequest, ...],
    model: str,
    checks: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    settings = real_settings(model)
    guard = BudgetGuard(args.max_requests, args.max_cost_usd, MODEL_PRICES[model])
    run = run_live(plan, dataset, settings, guard)
    executed = {result.case_id for result in run.results}
    paired = tuple(item.case for item in plan if item.case.case_id in executed)
    fake = run_cases(paired, dataset, lambda _case: DeterministicFakeProvider())
    lexical = run_cases(paired, dataset, lambda _case: LexicalOverlapProvider())
    report = {
        **_report_base(dataset, "paid"),
        "model": model,
        "preflight": checks,
        "live_model_requests": guard.requests_sent,
        "stop_reason": run.stop_reason,
        "budget": {
            "max_requests": guard.max_requests,
            "max_cost_usd": guard.max_cost_usd,
            "list_price_estimate_usd": round(guard.spent_usd, 8),
            "requests_charged_at_worst_case": guard.requests_at_worst_case,
            "note": "list-price estimate from reported tokens, not a billed amount",
        },
        "model_metrics": selection_metrics(run.results),
        "api_latency": _api_latency(run.results, plan),
        "paired_baselines": {
            "deterministic_fake": selection_metrics(fake),
            "lexical_overlap": selection_metrics(lexical),
        },
        "paired_comparison": {
            "against_deterministic_fake": compare(fake, run.results),
            "against_lexical_overlap": compare(lexical, run.results),
        },
        "cases": [result.to_report() for result in run.results],
    }
    return report, EXIT_OK if run.stop_reason is None else EXIT_RUN_INCOMPLETE


def _emit(report: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(report, indent=2)
    print(text)
    if output is not None:
        output.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    refusal = _refusal(args)
    if refusal is not None:
        print(f"refused: {refusal}", file=sys.stderr)
        return EXIT_REFUSED

    dataset = load_dataset(args.dataset)
    model = args.model or DEFAULT_MODEL
    settings = real_settings(model)
    limit = args.sample
    if args.allow_paid_calls:
        limit = min(limit or args.max_requests, args.max_requests)
    plan = plan_live_run(dataset, settings, limit)

    if not (args.preflight or args.allow_paid_calls):
        _emit(_offline_report(dataset, plan, model), args.output)
        return EXIT_OK

    try:
        checks = preflight(plan, settings)
    except PreflightFailed as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return EXIT_PREFLIGHT_FAILED
    if args.preflight:
        _emit({**_report_base(dataset, "preflight"), "live_model_requests": 0, "preflight": checks}, args.output)
        return EXIT_OK

    report, code = _live_report(dataset, plan, model, checks, args)
    _emit(report, args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
