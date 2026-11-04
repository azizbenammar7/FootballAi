"""M3-B0.1: scorer v2, negative controls, and a dataset audit. Offline, and spends nothing.

Writes a byte-reproducible report: rule v1 and rule v2 applied to twelve scripted
controls, scorer v1 beside scorer v2 for the same runs, and what the held-out split can
and cannot evaluate. No SDK, key, or network. Nothing here measures a language model.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from footballai_v2.copilot.agent.loop import AgentLimits
from footballai_v2.copilot.agent.tools import tool_definitions
from footballai_v2.copilot.evaluation.agent_benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseMode,
    CaseScore,
    Outcome,
    Split,
    load_benchmark,
    replay_script,
    run_case,
    score_metrics,
)
from footballai_v2.copilot.evaluation.agent_controls import CONTROLS, Control
from footballai_v2.copilot.evaluation.agent_live import prompt_fingerprint
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, route_following_policy
from footballai_v2.copilot.evaluation.agent_rule import (
    RULE_FILE,
    RULE_FILE_V2,
    DecisionRule,
    diagnostic_criteria,
    load_rule,
)
from footballai_v2.copilot.evaluation.agent_scoring import (
    SCORER_VERSION,
    CaseScoreV2,
    measurements_v2,
    score_case_v2,
    score_metrics_v2,
)
from footballai_v2.copilot.grounding import AGENT_SYSTEM_INSTRUCTIONS


REPORT_SCHEMA = "footballai.copilot-agent-reliability-report/v1"
REPORT_FILE = "scorer-v2-report.json"
RECOMMENDED_ADDITIONS = frozenset({"no_new_false_abstentions", "answer_task_success", "zero_live_false_answers"})
_NOTE = (
    "Replay only. No language model was called. Every row is a script run through the real agent, "
    "to test the scorer and the decision rule, never a model. 'verdict_if_accepted' is what an "
    "accepted rule would say if a model behaved exactly like the script. Both rules are proposed."
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provenance(dataset: BenchmarkDataset, directory: Path) -> dict[str, Any]:
    """Everything that decides a score, by hash. A committed report cannot name its own commit."""
    return {
        "scorer_version": SCORER_VERSION,
        "dataset_version": dataset.version,
        "cases_sha256": {split.value: dataset.sha256(split) for split in Split},
        "fixtures_sha256": dataset.source.fixtures_sha256,
        "decision_rule_v1_sha256": _sha256((directory / RULE_FILE).read_bytes()),
        "decision_rule_v2_sha256": _sha256((directory / RULE_FILE_V2).read_bytes()),
        **prompt_fingerprint(),
        "limits": dataclasses.asdict(AgentLimits()),
        "model": None,
        "sdk_version": "not used: replay",
        "source_commit": "not recorded: a committed file cannot name the commit that contains it",
    }


def _live(cases: tuple[BenchmarkCase, ...]) -> tuple[BenchmarkCase, ...]:
    return tuple(case for case in cases if case.mode is CaseMode.LIVE_ELIGIBLE)


def _run(control: Control, cases: tuple[BenchmarkCase, ...], dataset: BenchmarkDataset) -> tuple[CaseScore, ...]:
    return tuple(
        run_case(
            case,
            dataset.source,
            lambda clock, case=case: ScriptedAgentModel(policy=control.policy_for(case), on_call=clock.advance),
            limits=control.limits,
        )
        for case in cases
    )


def _v2(scores: tuple[CaseScore, ...]) -> tuple[CaseScoreV2, ...]:
    return tuple(score_case_v2(score.case, score.run) for score in scores)


def _v1_measurements(live: tuple[CaseScore, ...], hostile: tuple[CaseScore, ...], rule: tuple[CaseScore, ...]) -> dict[str, Any]:
    from footballai_v2.cli.copilot_agent_benchmark import measurements  # the v1 definition, unchanged

    measured = measurements(live, hostile, rule)
    measured["end_to_end_reaching_a_model"] = None  # a scripted clock measures nothing
    return measured


def control_matrix(
    dataset: BenchmarkDataset, rule_v1: DecisionRule, rule_v2: DecisionRule, split: Split = Split.HELDOUT
) -> dict[str, Any]:
    """Each control under both rules, on the split the rule applies to."""
    cases = dataset.split(split)
    live = _live(cases)
    hostile = tuple(run_case(case, dataset.source, replay_script(case)) for case in cases if case.mode is CaseMode.REPLAY_ONLY)
    following = _run(CONTROLS[1], live, dataset)
    hostile_v2, following_v2 = _v2(hostile), _v2(following)
    all_v2 = frozenset(c.criterion_id for c in rule_v2.criteria)
    rows: dict[str, Any] = {}
    for control in CONTROLS:
        scores = following if control.name == "route_following" else _run(control, live, dataset)
        scores_v2 = _v2(scores)
        metrics = score_metrics_v2(scores_v2)
        measured_v2 = measurements_v2(scores_v2, hostile_v2, following_v2, latency_measured=False)
        proposed = diagnostic_criteria(rule_v2, measured_v2)
        rows[control.name] = {
            "kind": control.kind,
            "description": control.description,
            "questions_reaching_a_model": sum(score.attempted for score in scores_v2),
            "terminations": metrics["termination_accounting"]["by_class"],
            "rule_v1": diagnostic_criteria(rule_v1, _v1_measurements(scores, hostile, following)),
            "rule_v2": {
                **proposed,
                "verdict_if_accepted": {
                    "as_proposed": proposed["verdict_if_accepted"],
                    "with_recommended_additions": diagnostic_criteria(
                        rule_v2,
                        measured_v2,
                        binding=frozenset(c.criterion_id for c in rule_v2.criteria if c.binding) | RECOMMENDED_ADDITIONS,
                    )["verdict_if_accepted"],
                    "with_all_additions": diagnostic_criteria(rule_v2, measured_v2, binding=all_v2)["verdict_if_accepted"],
                },
            },
        }
    return rows


def never_failed(matrix: dict[str, Any], rule_key: str) -> list[str]:
    """Criteria no control fails. Each is either a machinery check or untested by this split."""
    statuses: dict[str, set[str]] = {}
    for row in matrix.values():
        for item in row[rule_key]["criteria"]:
            statuses.setdefault(item["criterion"], set()).add(item["status"])
    return sorted(name for name, seen in statuses.items() if "fail" not in seen)


def old_versus_new(scores: tuple[CaseScore, ...]) -> dict[str, Any]:
    """The same runs under both scorers. v1 numbers are recomputed here, never edited."""
    scores_v2 = _v2(scores)
    v1, v2 = score_metrics(scores), score_metrics_v2(scores_v2)
    return {
        "scorer_v1": {
            key: v1[key]
            for key in (
                "outcome_accuracy",
                "claim_selection_accuracy",
                "citation_validity",
                "tool_selection_accuracy",
                "provider_failures_plus_invalid_outputs",
            )
        },
        "scorer_v2": {
            key: v2[key]
            for key in (
                "task_success",
                "answer_task_success",
                "answer_relevance",
                "answer_completeness",
                "abstention_correctness",
                "citation_validity",
                "tool_selection_accuracy",
            )
        }
        | {"non_completion": v2["reliability"]["non_completion"]},
        "irrelevant_answers_v1_counted_as_correct_outcomes": sum(
            s.irrelevant_answer and s.v1.outcome_correct and s.case.expected.outcome is Outcome.ANSWER
            for s in scores_v2
        ),
    }


def _ids(cases: Any) -> list[str]:
    return sorted(case.case_id for case in cases)


def _plain(question: str) -> str:
    return question.lower().rstrip("?.! ")


def dataset_audit(dataset: BenchmarkDataset) -> dict[str, Any]:
    """Leakage and coverage. Reported, and not fixed here: the labels change only in M3-B0.2."""
    dev_live = _live(dataset.dev)
    held_live = _live(dataset.heldout)
    dev_questions = {case.question for case in dev_live}
    overlap = [case for case in held_live if case.question in dev_questions]
    tool_text = json.dumps(tool_definitions()).lower()
    prompt_text = AGENT_SYSTEM_INSTRUCTIONS.lower()

    def leaks(text: str) -> list[str]:
        """A question, without its closing punctuation, found word for word in ``text``."""
        return _ids(case for case in held_live if case.question and _plain(case.question) in text)

    following = {
        case.case_id: run_case(case, dataset.source, lambda _clock: ScriptedAgentModel(policy=route_following_policy))
        for case in held_live
    }
    reaches = {case_id for case_id, score in following.items() if score.run.result.trace.model_turns}
    by_category: dict[str, dict[str, int]] = {}
    for case in held_live:
        row = by_category.setdefault(case.category.value, {"live": 0, "reach_a_model": 0})
        row["live"] += 1
        row["reach_a_model"] += case.case_id in reaches
    return {
        "heldout_questions_also_in_dev": {
            "cases": _ids(overlap),
            "reaching_a_model": _ids(case for case in overlap if case.case_id in reaches),
            "note": "Same question text, another fixture. load_benchmark compares (fixture, question), so it allows these.",
        },
        "heldout_questions_inside_tool_descriptions": leaks(tool_text),
        "heldout_questions_inside_system_instructions": leaks(prompt_text),
        "heldout_live_cases_by_category": dict(sorted(by_category.items())),
        "heldout_router_stopped": _ids(case for case in held_live if case.case_id not in reaches),
        "labels_reviewed_by_a_second_person": sum(
            case.label.review != "pending_second_person" for case in (*dataset.dev, *dataset.heldout)
        ),
    }


def build_report(dataset: BenchmarkDataset, directory: Path) -> dict[str, Any]:
    rule_v1, _ = load_rule(directory)
    rule_v2, _ = load_rule(directory, RULE_FILE_V2)
    matrix = control_matrix(dataset, rule_v1, rule_v2)
    comparison = {}
    for split in Split:
        live = _live(dataset.split(split))
        comparison[split.value] = {
            name: old_versus_new(_run(control, live, dataset))
            for name, control in (("label_oracle", CONTROLS[0]), ("route_following", CONTROLS[1]))
        }
    return {
        "schema": REPORT_SCHEMA,
        "mode": "replay",
        "note": _NOTE,
        "live_model_requests": 0,
        "real_model_quality_measured": False,
        "provenance": provenance(dataset, directory),
        "rules": {
            "v1": {"version": rule_v1.rule_version, "status": rule_v1.status.value},
            "v2": {"version": rule_v2.rule_version, "status": rule_v2.status.value},
            "recommended_additions": sorted(RECOMMENDED_ADDITIONS),
        },
        "old_versus_new": comparison,
        "negative_controls": {
            "split": Split.HELDOUT.value,
            "controls": matrix,
            "criteria_no_control_fails": {
                "rule_v1": never_failed(matrix, "rule_v1"),
                "rule_v2": never_failed(matrix, "rule_v2"),
            },
        },
        "dataset_audit": dataset_audit(dataset),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="agent-v2 dataset directory")
    parser.add_argument("--output", type=Path, help="also write the report to this JSON file")
    args = parser.parse_args(argv)
    report = build_report(load_benchmark(args.dataset), args.dataset)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    return 0


__all__ = ["REPORT_FILE", "build_report", "control_matrix", "dataset_audit", "main", "old_versus_new", "provenance"]


if __name__ == "__main__":
    raise SystemExit(main())
