"""M3-A: the agent benchmark. Reproducible, offline, unable to pass without a real model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from footballai_v2.cli import copilot_agent_benchmark as cli
from footballai_v2.copilot.agent import AgentStopReason, ToolName
from footballai_v2.copilot.agent.trace import AgentTrace, ToolCallTrace, TraceUsage, UsageSource
from footballai_v2.copilot.agent.tools import ToolCallStatus, ToolErrorCode
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evaluation.agent_benchmark import (
    BenchmarkCase,
    CaseMode,
    Category,
    Expected,
    Label,
    Outcome,
    Split,
    label_oracle_policy,
    load_benchmark,
    replay_script,
    run_case,
)
from footballai_v2.copilot.evaluation.agent_live import (
    LiveApproval,
    LiveRefused,
    check_approval,
    endpoint_kind,
    plan_benchmark,
    plan_cost,
)
from footballai_v2.copilot.evaluation.agent_llmops import NOT_MEASURED, summarize_traces
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, Step, route_following_policy
from footballai_v2.copilot.evaluation.live import real_settings
from footballai_v2.copilot.evaluation.agent_rule import (
    Criterion,
    RuleStatus,
    Verdict,
    apply_criterion,
    evaluate_rule,
    load_rule,
)
from footballai_v2.copilot.models import ModelMode, OutputNature


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "evaluation" / "copilot" / "agent-v2"
ENVIRONMENT = {"ANTHROPIC_API_KEY": "canary-value-not-real"}
DECISION = dict(stage="decision", split=Split.HELDOUT, cases_sha256="c" * 64, decision_rule_sha256="r" * 64)


@pytest.fixture(scope="module")
def dataset():
    return load_benchmark(DATASET_DIR)


@pytest.fixture(scope="module")
def rule():
    return load_rule(DATASET_DIR)


@pytest.fixture(scope="module")
def report(dataset, rule):
    return cli.build_replay_report(dataset, rule[0], rule[1])


# -- the dataset --------------------------------------------------------------------


def test_the_committed_report_is_exactly_what_the_code_produces(report):
    committed = (DATASET_DIR / "baseline-report.json").read_text(encoding="utf-8")
    assert committed == json.dumps(report, indent=2) + "\n"


def test_there_are_at_least_sixty_cases_covering_every_category(dataset):
    assert len(dataset.dev) + len(dataset.heldout) >= 60
    for split in Split:
        assert {case.category for case in dataset.split(split)} == set(Category), split


def test_no_live_question_appears_in_both_splits(dataset):
    dev = {(case.fixture, case.question) for case in dataset.dev if case.mode is CaseMode.LIVE_ELIGIBLE}
    held = {(case.fixture, case.question) for case in dataset.heldout if case.mode is CaseMode.LIVE_ELIGIBLE}
    assert not dev & held


def test_a_question_shared_between_the_splits_is_refused(tmp_path, dataset):
    root = tmp_path / "copilot"
    (root / "agent-v2").mkdir(parents=True)
    (root / "v2").symlink_to(REPO_ROOT / "evaluation" / "copilot" / "v2")
    dev = json.loads((DATASET_DIR / "cases-dev.json").read_text(encoding="utf-8"))
    held = json.loads((DATASET_DIR / "cases-heldout.json").read_text(encoding="utf-8"))
    first = next(case for case in dev["cases"] if case["mode"] == "live_eligible")
    held["cases"].append({**first, "case_id": "copied-from-dev"})
    (root / "agent-v2" / "cases-dev.json").write_text(json.dumps(dev), encoding="utf-8")
    (root / "agent-v2" / "cases-heldout.json").write_text(json.dumps(held), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out question"):
        load_benchmark(root / "agent-v2")


def test_a_split_in_the_wrong_file_is_refused(tmp_path):
    root = tmp_path / "copilot"
    (root / "agent-v2").mkdir(parents=True)
    (root / "v2").symlink_to(REPO_ROOT / "evaluation" / "copilot" / "v2")
    dev = (DATASET_DIR / "cases-dev.json").read_text(encoding="utf-8")
    (root / "agent-v2" / "cases-dev.json").write_text(dev, encoding="utf-8")
    (root / "agent-v2" / "cases-heldout.json").write_text(dev, encoding="utf-8")
    with pytest.raises(ValueError, match="heldout split"):
        load_benchmark(root / "agent-v2")


def test_every_label_states_why_and_none_claims_a_review(dataset):
    for case in (*dataset.dev, *dataset.heldout):
        assert case.label.rationale and case.label.evidence
        assert case.label.review == "pending_second_person"


def test_answer_cases_are_labelled_from_claims_the_route_cannot_always_offer(report):
    """The labels were not written from the router, so some cases are out of its reach."""
    assert report["dev"]["label_oracle"]["failed_cases"] == [
        "adv-inside-the-vocabulary",
        "ans-finish-paraphrase",
        "ans-real-or-synthetic",
        "multi-model-and-status",
        "multi-overview-synthetic",
    ]


def test_no_replay_case_is_ever_planned_for_a_model(dataset):
    settings = real_settings("claude-haiku-5-5")
    for split in Split:
        planned = plan_benchmark(dataset, split, settings)
        assert planned and all(item.case.mode is CaseMode.LIVE_ELIGIBLE for item in planned)


def test_a_question_the_router_stops_is_never_planned(dataset):
    planned = {item.case.case_id for item in plan_benchmark(dataset, Split.DEV, real_settings("claude-haiku-5-5"))}
    assert "uns-passes" not in planned and "run-named-in-the-question" not in planned
    assert "gap-no-tracklets" not in planned  # no evidence, so no model is called


def test_expected_labels_must_agree_with_themselves():
    base = dict(output_nature=OutputNature.SYNTHETIC)
    with pytest.raises(ValueError):
        Expected(outcome=Outcome.ANSWER, terminations=[AgentStopReason.NO_RELEVANT_CLAIM], **base)
    with pytest.raises(ValueError):
        Expected(outcome=Outcome.ANSWER, terminations=[AgentStopReason.ANSWERED], **base)
    with pytest.raises(ValueError):
        Expected(outcome=Outcome.ABSTAIN, terminations=[AgentStopReason.MODEL_TIMEOUT], **base)
    with pytest.raises(ValueError):  # a required claim needs its tool
        Expected(
            outcome=Outcome.ANSWER,
            terminations=[AgentStopReason.ANSWERED],
            required_claims=("run_status",),
            permitted_claims=("run_status",),
            **base,
        )


def test_a_live_case_can_never_carry_a_script():
    expected = Expected(outcome=Outcome.ABSTAIN, terminations=[AgentStopReason.NO_RELEVANT_CLAIM], output_nature=OutputNature.SYNTHETIC)
    label = Label(rationale="a sufficient reason", evidence="a claim")
    with pytest.raises(ValueError):
        BenchmarkCase(
            case_id="x", category=Category.ANSWERABLE, mode=CaseMode.LIVE_ELIGIBLE, fixture="f", question="q",
            script=({"select": []},), expected=expected, label=label,
        )
    with pytest.raises(ValueError):
        BenchmarkCase(
            case_id="x", category=Category.ANSWERABLE, mode=CaseMode.REPLAY_ONLY, fixture="f", question="q",
            expected=expected, label=label,
        )


# -- the scorer ---------------------------------------------------------------------


def by_id(dataset, identifier):
    return next(case for case in (*dataset.dev, *dataset.heldout) if case.case_id == identifier)


def following(dataset, case):
    return run_case(case, dataset.source, lambda _clock: ScriptedAgentModel(policy=route_following_policy))


def test_a_true_claim_that_does_not_answer_is_a_false_answer(dataset):
    score = following(dataset, by_id(dataset, "zero-model-real"))
    assert score.false_answer is True and score.passed is False
    assert score.fabricated is False  # true, cited, and still wrong


def test_the_right_abstention_passes(dataset):
    case = by_id(dataset, "zero-model-real")
    score = run_case(case, dataset.source, lambda _clock: ScriptedAgentModel(policy=label_oracle_policy(case)))
    assert score.passed is True and score.false_answer is False


def test_a_claim_outside_the_permitted_set_fails(dataset):
    case = by_id(dataset, "ans-status")
    narrow = case.model_copy(update={"expected": case.expected.model_copy(update={"permitted_claims": ("run_status",)})})
    wide = run_case(narrow, dataset.source, lambda _clock: ScriptedAgentModel(policy=route_following_policy))
    assert wide.passed is True
    wrong = narrow.model_copy(
        update={"expected": case.expected.model_copy(update={"required_claims": ("provenance",), "permitted_claims": ("provenance",), "required_tools": (ToolName.GET_MODEL_PROVENANCE,)})}
    )
    assert run_case(wrong, dataset.source, lambda _clock: ScriptedAgentModel(policy=route_following_policy)).passed is False


def test_forbidden_text_counts_as_fabrication(dataset):
    case = by_id(dataset, "ans-status")
    trap = case.model_copy(update={"expected": case.expected.model_copy(update={"forbidden_text": ("Analysis run",)})})
    assert following(dataset, trap).fabricated is True


def test_a_wrong_output_nature_is_caught(dataset):
    case = by_id(dataset, "ans-status")
    wrong = case.model_copy(update={"expected": case.expected.model_copy(update={"output_nature": OutputNature.SYNTHETIC})})
    assert following(dataset, wrong).output_nature_correct is False


def test_a_missing_required_tool_is_caught(dataset):
    case = by_id(dataset, "ans-status")
    wrong = case.model_copy(
        update={"expected": case.expected.model_copy(update={"required_tools": (ToolName.GET_RUN_STATUS, ToolName.GET_ARTIFACT_METADATA)})}
    )
    assert following(dataset, wrong).tools_correct is False


def test_a_forbidden_tool_is_caught(dataset):
    case = by_id(dataset, "ans-status")
    wrong = case.model_copy(update={"expected": case.expected.model_copy(update={"forbidden_tools": (ToolName.GET_RUN_STATUS,), "required_tools": ()})})
    assert following(dataset, wrong).tools_correct is False


def test_every_hostile_replay_fails_closed(report):
    for split in ("dev", "heldout"):
        hostile = report[split]["hostile_replay"]
        assert not hostile["failed_cases"]
        assert hostile["protocol_problems"] == 0
    assert report["dev"]["hostile_replay"]["cross_run_attempts_stopped"] == 3
    assert report["dev"]["hostile_replay"]["out_of_registry_attempts"] == 1
    assert report["dev"]["hostile_replay"]["out_of_registry_executions"] == 0


def test_a_hostile_case_is_really_hostile(dataset):
    score = run_case(by_id(dataset, "adv-prose-final"), dataset.source, replay_script(by_id(dataset, "adv-prose-final")))
    assert score.run.result.trace.termination is AgentStopReason.MALFORMED_RESPONSE
    assert "10.5" not in score.run.result.response.answer


def test_the_deterministic_rule_really_fails_where_a_model_could_help(report):
    assert report["dev"]["route_following"]["false_answers"] == [
        "zero-artifact-origin",
        "zero-model-real",
        "zero-model-status",
        "zero-model-version",
        "zero-tracks-verified",
    ]
    assert report["heldout"]["route_following"]["false_answers"] == 4


def test_the_metrics_can_fail(report):
    for control in report["negative_controls"].values():
        assert control["claim_selection_accuracy"]["rate"] < 0.1
        assert control["tool_selection_accuracy"]["rate"] < 0.1
        assert control["fabricated_answers"] == 0  # bad behaviour stays safe


def test_the_held_out_split_is_reported_only_as_totals(report):
    for name in ("label_oracle", "route_following", "hostile_replay"):
        assert isinstance(report["heldout"][name]["failed_cases"], int)
        assert isinstance(report["heldout"][name]["false_answers"], int)


def test_no_fabrication_anywhere_in_replay(report):
    for split in ("dev", "heldout"):
        for name in ("label_oracle", "route_following", "hostile_replay"):
            assert not report[split][name]["fabricated_answers"]
            assert not report[split][name]["cross_run_violations"]


def test_the_no_gap_case_adr_006_asked_for_exists_and_passes(dataset):
    case = by_id(dataset, "gap-none-on-a-complete-run")
    assert following(dataset, case).passed is True
    assert "no_missing_evidence" in case.expected.required_claims


def test_the_gap_claim_is_withheld_when_a_measure_was_refused(dataset):
    score = following(dataset, by_id(dataset, "bad-malformed-gap"))
    assert score.run.result.trace.termination is AgentStopReason.EVIDENCE_NOT_AVAILABLE


# -- the decision rule ----------------------------------------------------------------


def test_the_rule_is_proposed_and_nobody_has_accepted_it(rule):
    assert rule[0].status is RuleStatus.PROPOSED
    assert rule[0].accepted_by is None and rule[0].accepted_on is None


def test_the_rule_states_the_roadmap_thresholds(rule):
    criteria = {criterion.criterion_id: criterion for criterion in rule[0].criteria}
    assert criteria["zero_fabricated_answers"].threshold == 0
    assert criteria["citation_validity"].threshold == 1.0
    assert criteria["output_nature_accuracy"].threshold == 1.0
    assert criteria["tool_selection_accuracy"].threshold == 0.9
    assert criteria["provider_failures_and_invalid_outputs"].threshold == 0.05
    assert criteria["end_to_end_p95"].threshold == 8.0


def test_every_criterion_states_its_denominator_and_every_addition_is_marked(rule):
    for criterion in rule[0].criteria:
        assert criterion.denominator
        assert (criterion.origin == "proposed_addition") == (not criterion.binding)


def _scored(dataset, case_id, step):
    case = by_id(dataset, case_id)
    return run_case(case, dataset.source, lambda _clock: ScriptedAgentModel([step]))


def test_a_failed_call_is_not_a_corrected_false_answer(dataset):
    case = by_id(dataset, "zero-model-real")
    timed_out = _scored(dataset, "zero-model-real", Step(error=ProviderTimeoutError("scripted")))
    rule_score = following(dataset, case)
    assert rule_score.false_answer is True and timed_out.false_answer is False
    measured = cli.measurements((timed_out,), (), (rule_score,))
    assert measured["share_of_deterministic_false_answers_corrected"]["correct"] == 0


def test_a_right_abstention_is_a_corrected_false_answer(dataset):
    case = by_id(dataset, "zero-model-real")
    right = run_case(case, dataset.source, lambda _clock: ScriptedAgentModel(policy=label_oracle_policy(case)))
    measured = cli.measurements((right,), (), (following(dataset, case),))
    assert measured["share_of_deterministic_false_answers_corrected"]["correct"] == 1


def test_a_budget_stop_is_not_held_against_the_model(dataset):
    stopped = _scored(
        dataset, "ans-status", Step(error=ProviderUnavailableError("x", kind=ProviderFailureKind.BUDGET_EXHAUSTED))
    )
    assert stopped.provider_failure is False


def test_a_rate_limit_is_held_against_the_run(dataset):
    limited = _scored(dataset, "ans-status", Step(error=ProviderUnavailableError("x", kind=ProviderFailureKind.RATE_LIMITED)))
    assert limited.provider_failure is True


def test_every_measurement_a_criterion_names_exists(dataset, rule):
    empty = cli.measurements((), (), ())
    for criterion in rule[0].criteria:
        assert criterion.measurement in empty, criterion.measurement


@pytest.mark.parametrize(
    ("kind", "threshold", "measurement", "min_n", "status"),
    [
        ("max_count", 0, [], 0, "pass"),
        ("max_count", 0, ["a-case"], 0, "fail"),
        ("min_rate", 0.9, {"rate": 0.9, "n": 20}, 20, "pass"),
        ("min_rate", 0.9, {"rate": 0.85, "n": 20}, 20, "fail"),
        ("min_rate", 0.9, {"rate": 1.0, "n": 8}, 20, "insufficient_sample"),
        ("min_rate", 0.9, {"rate": 0.5, "n": 8}, 20, "fail"),  # a small failure is still a failure
        ("max_rate", 0.05, {"rate": 0.05, "n": 20}, 20, "pass"),
        ("max_rate", 0.05, {"rate": 0.1, "n": 20}, 20, "fail"),
        ("max_rate", 0.05, {"rate": 0.0, "n": 5}, 20, "insufficient_sample"),
        ("max_seconds", 8.0, {"p95_seconds": 7.9, "n": 20}, 20, "pass"),
        ("max_seconds", 8.0, {"p95_seconds": 8.1, "n": 20}, 20, "fail"),
        ("max_seconds", 8.0, {"p95_seconds": None, "n": 0}, 20, "not_measured"),
        ("min_rate", 0.9, None, 20, "not_measured"),
    ],
)
def test_the_exact_pass_and_fail_calculation(kind, threshold, measurement, min_n, status):
    criterion = Criterion(
        criterion_id="c", statement="s", measurement="m", kind=kind, threshold=threshold, denominator="d",
        min_n=min_n, origin="roadmap",
    )
    assert apply_criterion(criterion, measurement).status == status


def passing(rule):
    return {
        criterion.measurement: []
        if criterion.kind == "max_count"
        else {"rate": 1.0 if criterion.kind == "min_rate" else 0.0, "n": 100, "p95_seconds": 1.0}
        for criterion in rule.criteria
    }


def accepted(rule):
    return rule.model_copy(update={"status": RuleStatus.ACCEPTED, "accepted_by": "owner", "accepted_on": "2026-11-03"})


LIVE = dict(mode="live", endpoint="anthropic_api", split="heldout", completed=True)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "replay"},
        {"endpoint": "loopback_test_double"},
        {"endpoint": None},
        {"split": "dev"},
    ],
)
def test_nothing_but_a_real_held_out_run_gets_a_verdict(rule, change):
    result = evaluate_rule(accepted(rule[0]), passing(rule[0]), **{**LIVE, **change})
    assert result["verdict"] == Verdict.NOT_A_MODEL_EVALUATION.value
    assert result["criteria"] == []  # a script's perfect numbers are not shown beside a pass


def test_a_proposed_rule_cannot_pass_even_on_perfect_numbers(rule):
    assert evaluate_rule(rule[0], passing(rule[0]), **LIVE)["verdict"] == Verdict.RULE_NOT_ACCEPTED.value


def test_an_accepted_rule_can_pass_and_can_fail(rule):
    good = passing(rule[0])
    assert evaluate_rule(accepted(rule[0]), good, **LIVE)["verdict"] == Verdict.PASS.value
    bad = {**good, "fabricated_answers": ["one-case"]}
    assert evaluate_rule(accepted(rule[0]), bad, **LIVE)["verdict"] == Verdict.FAIL.value


def test_a_small_sample_cannot_pass(rule):
    thin = {**passing(rule[0]), "tool_selection_accuracy": {"rate": 1.0, "n": 8}}
    assert evaluate_rule(accepted(rule[0]), thin, **LIVE)["verdict"] == Verdict.INSUFFICIENT_SAMPLE.value


def test_an_unfinished_run_cannot_pass(rule):
    result = evaluate_rule(accepted(rule[0]), passing(rule[0]), **{**LIVE, "completed": False})
    assert result["verdict"] == Verdict.INCOMPLETE_RUN.value


def test_a_proposed_addition_cannot_decide_until_someone_makes_it_binding(rule):
    worse = {**passing(rule[0]), "new_false_abstentions": ["one-case"]}
    assert evaluate_rule(accepted(rule[0]), worse, **LIVE)["verdict"] == Verdict.PASS.value


def test_the_replay_report_gives_no_verdict(report):
    for split in ("dev", "heldout"):
        assert report[split]["decision_rule"]["verdict"] == Verdict.NOT_A_MODEL_EVALUATION.value
        assert report[split]["decision_rule"]["criteria"] == []
    assert report["live_model_requests"] == 0
    assert report["real_model_quality_measured"] is False
    assert "No language model was called" in report["note"]


# -- the approval gate ------------------------------------------------------------------


def approval(**changes):
    values = dict(
        status="approved", approved_by="owner", approved_on="2026-11-03", model="claude-haiku-5-5",
        max_model_requests=120, max_cost_usd=0.5, **DECISION,
    )
    return LiveApproval.model_validate({"schema": "footballai.copilot-agent-live-approval/v1", **{**values, **changes}})


def check(record=None, **changes):
    arguments = dict(
        split=Split.HELDOUT, model="claude-haiku-5-5", max_requests=120, max_cost_usd=0.5, cases_sha256="c" * 64,
        rule_sha256="r" * 64, rule_accepted=True, environ=ENVIRONMENT,
    )
    check_approval(record or approval(), **{**arguments, **changes})


def test_a_complete_approval_passes():
    check()


@pytest.mark.parametrize(
    ("record", "changes", "code"),
    [
        ({"status": "not_approved"}, {}, "not_approved"),
        ({"approved_by": None}, {}, "not_approved"),
        ({"approved_on": None}, {}, "not_approved"),
        ({}, {"split": Split.DEV}, "approval_other_split"),
        ({}, {"model": "claude-sonnet-5-5"}, "approval_other_model"),
        ({}, {"cases_sha256": "d" * 64}, "approval_other_cases"),
        ({}, {"rule_sha256": "d" * 64}, "approval_other_rule"),
        ({}, {"max_requests": 121}, "requests_over_approval"),
        ({}, {"max_requests": 0}, "requests_over_approval"),
        ({}, {"max_cost_usd": 0.51}, "cost_over_approval"),
        ({}, {"max_cost_usd": 0.0}, "cost_over_approval"),
        ({}, {"rule_accepted": False}, "rule_not_accepted"),
        ({}, {"model": "claude-unknown-9"}, "unpriced_model"),
        ({}, {"environ": {}}, "missing_api_key"),
        ({}, {"environ": {**ENVIRONMENT, "CI": "true"}}, "continuous_integration"),
        ({}, {"environ": {**ENVIRONMENT, "GITHUB_ACTIONS": "true"}}, "continuous_integration"),
        ({}, {"environ": {**ENVIRONMENT, "ANTHROPIC_BASE_URL": "https://proxy.example.com"}}, "unapproved_api_endpoint"),
    ],
)
def test_every_condition_is_enforced(record, changes, code):
    with pytest.raises(LiveRefused) as caught:
        check(approval(**record), **changes)
    assert caught.value.code == code


def test_a_readiness_run_may_not_read_the_held_out_split():
    with pytest.raises(LiveRefused) as caught:
        check(approval(stage="readiness"), rule_accepted=False)
    assert caught.value.code == "readiness_cannot_read_heldout"


def test_a_readiness_run_on_development_cases_needs_no_accepted_rule():
    check(approval(stage="readiness", split=Split.DEV), split=Split.DEV, rule_accepted=False)


def test_a_decision_run_must_be_held_out():
    with pytest.raises(LiveRefused) as caught:
        check(approval(split=Split.DEV), split=Split.DEV)
    assert caught.value.code == "decision_needs_heldout"


def test_a_test_double_is_labelled_as_one():
    assert endpoint_kind({}) == "anthropic_api"
    assert endpoint_kind({"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}) == "loopback_test_double"
    with pytest.raises(LiveRefused):
        endpoint_kind({"ANTHROPIC_BASE_URL": "https://proxy.example.com"})


def test_the_template_is_not_an_approval():
    record = json.loads((DATASET_DIR / "live-approval.template.json").read_text(encoding="utf-8"))
    assert record["status"] == "not_approved"
    assert record["max_model_requests"] == 0 and record["max_cost_usd"] == 0


def test_the_cli_refuses_a_paid_run_without_an_approval(capsys):
    code = cli.main(["--dataset", str(DATASET_DIR), "--allow-paid-calls", "--max-requests", "1", "--max-cost-usd", "0.01"])
    assert code == 2
    assert "refused: not_approved" in capsys.readouterr().out


def test_the_cli_refuses_the_template_itself(monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "canary-value-not-real")
    code = cli.main(
        ["--dataset", str(DATASET_DIR), "--allow-paid-calls", "--approval", str(DATASET_DIR / "live-approval.template.json"),
         "--max-requests", "1", "--max-cost-usd", "0.01"]
    )
    assert code == 2
    assert "not_approved" in capsys.readouterr().out


def test_an_unreadable_approval_is_refused(tmp_path, capsys):
    bad = tmp_path / "approval.json"
    bad.write_text("{", encoding="utf-8")
    code = cli.main(["--dataset", str(DATASET_DIR), "--allow-paid-calls", "--approval", str(bad)])
    assert code == 2
    assert "unreadable_approval" in capsys.readouterr().out


def test_no_workflow_can_make_a_paid_call():
    for path in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert "allow-paid-calls" not in text and "ANTHROPIC_API_KEY" not in text, path


def test_a_real_key_never_reaches_a_benchmark_report(report):
    text = json.dumps(report)
    assert "canary" not in text and "sk-ant" not in text


def test_the_default_mode_needs_no_sdk_key_or_network():
    code = (
        "import sys; from pathlib import Path; from footballai_v2.cli import copilot_agent_benchmark as c; "
        f"c.main(['--dataset', {str(DATASET_DIR)!r}]); sys.exit(1 if 'anthropic' in sys.modules else 0)"
    )
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "v2" / "src")}
    environment.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_the_cli_shows_a_fixture_for_label_review(capsys):
    assert cli.main(["--dataset", str(DATASET_DIR), "--show-fixture", "verified_real"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["output_nature"] == "verified_real"
    assert "no_missing_evidence" in shown["claims"]


def test_the_cli_writes_the_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert cli.main(["--dataset", str(DATASET_DIR), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == capsys.readouterr().out


def test_the_cost_plan_is_a_marked_upper_bound(report, dataset):
    plan = report["cost_plan"]["heldout"]
    assert "not a measurement" in plan["note"]
    assert plan["max_model_requests"] == plan["questions_reaching_a_model"] * 5
    assert plan["worst_case_total_usd"] < 1.0
    assert report["pricing"]["verified_on"] == "2026-11-02"


def test_a_planned_request_is_exactly_what_the_adapter_would_send(dataset):
    settings = real_settings("claude-haiku-5-5")
    item = plan_benchmark(dataset, Split.DEV, settings)[0]
    assert set(item.arguments) == {"model", "max_tokens", "system", "messages", "tools", "tool_choice", "output_config"}


def test_a_planned_cost_uses_the_full_limits(dataset):
    settings = real_settings("claude-haiku-5-5")
    plan = plan_benchmark(dataset, Split.DEV, settings)
    assert plan_cost(plan, settings, 5)["worst_case_total_usd"] > plan_cost(plan, settings, 1)["worst_case_total_usd"]
    assert plan_cost(plan, replace(settings, model_name="claude-sonnet-5-5"), 5)["worst_case_total_usd"] > 1


# -- operations ---------------------------------------------------------------------------


def trace(**changes) -> AgentTrace:
    values = dict(
        correlation_id="c1", run_id="r", model="claude-haiku-5-5", model_mode=ModelMode.REAL_MODEL,
        instruction_version="i", routing_version="v", termination=AgentStopReason.ANSWERED, duration_ms=2000.0,
        model_turns=2, turns=(
            dict(turn=1, duration_ms=500.0, usage=dict(input_tokens=900, output_tokens=40)),
            dict(turn=2, duration_ms=700.0, usage=dict(input_tokens=1100, output_tokens=60)),
        ),
        tool_calls=(ToolCallTrace(sequence=1, turn=1, tool=ToolName.GET_RUN_STATUS, status=ToolCallStatus.OK, duration_ms=3.0),),
        tool_calls_refused=0, claim_ids_returned=(), claim_ids_selected=(), claim_ids_rendered=(), selected_ids_rejected=0,
        usage=TraceUsage(input_tokens=2000, output_tokens=100), usage_complete=True,
        usage_source=UsageSource.PROVIDER_REPORTED, estimated_cost_usd=0.00025, cost_is_lower_bound=False,
        valid_tool_calls=1, invalid_tool_calls=0, model_call_ms=1200.0, tool_call_ms=3.0, peak_prompt_chars=4000,
    )
    return AgentTrace(**{**values, **changes})


def test_actual_tokens_exclude_scripted_constants():
    scripted = trace(
        correlation_id="c2", usage_source=UsageSource.SCRIPTED_CONSTANT, estimated_cost_usd=None, model_mode=ModelMode.OFFLINE_DEMO_FAKE
    )
    summary = summarize_traces([trace(), scripted], latency_measured=True)
    assert summary["tokens"]["actual_input_tokens"] == 2000
    assert summary["tokens"]["usage_sources"] == {"provider_reported": 1, "scripted_constant": 1}
    assert summary["real_model_questions"] == 1


def test_a_lower_bound_is_reported_as_one():
    partial = trace(correlation_id="c2", usage_source=UsageSource.UNAVAILABLE, usage_complete=False, cost_is_lower_bound=True)
    summary = summarize_traces([trace(), partial], latency_measured=True)
    assert summary["tokens"]["totals_are_a_lower_bound"] is True
    assert summary["cost"]["is_lower_bound"] is True
    assert "not a bill" in summary["cost"]["basis"]


def test_latency_is_reported_only_when_measured():
    assert summarize_traces([trace()], latency_measured=False)["latency"] == NOT_MEASURED


def test_percentiles_cover_questions_calls_and_tools():
    traces = [trace(correlation_id=f"c{index}", duration_ms=1000.0 * index) for index in range(1, 21)]
    latency = summarize_traces(traces, latency_measured=True)["latency"]
    assert latency["end_to_end_reaching_a_model"]["p50_seconds"] == 10.0
    assert latency["end_to_end_reaching_a_model"]["p95_seconds"] == 19.0
    assert latency["per_model_call"]["n"] == 40
    assert latency["per_tool"]["get_run_status"]["n"] == 20


def test_first_use_schemas_are_measured_apart():
    traces = [trace(correlation_id="first", duration_ms=9000.0), trace(correlation_id="again", duration_ms=1000.0)]
    latency = summarize_traces(traces, latency_measured=True, first_use=frozenset({"first"}))["latency"]
    assert latency["end_to_end_first_use_schema"]["max_seconds"] == 9.0
    assert latency["end_to_end_repeat_schema"]["max_seconds"] == 1.0


def test_failures_terminations_and_refused_calls_are_counted():
    failed = trace(
        correlation_id="c2", termination=AgentStopReason.PROVIDER_ERROR, error_code=ProviderFailureKind.RATE_LIMITED,
        tool_calls=(ToolCallTrace(sequence=1, turn=1, tool=None, status=ToolCallStatus.REJECTED, duration_ms=0.0, error_code=ToolErrorCode.UNKNOWN_TOOL),),
        valid_tool_calls=0, invalid_tool_calls=1, tool_calls_refused=2,
    )
    summary = summarize_traces([trace(), failed], latency_measured=True)
    assert summary["provider_failures_by_kind"] == {"rate_limited": 1}
    assert summary["terminations"] == {"answered": 1, "provider_error": 1}
    assert summary["tool_calls"]["outside_the_registry"] == 1
    assert summary["tool_calls"]["refused_before_running"] == 2
    assert summary["tool_calls"]["errors"] == {"unknown_tool": 1}
    assert summary["unique_correlation_ids"] == 2


def test_a_tool_error_is_not_a_provider_failure():
    stopped = trace(termination=AgentStopReason.CROSS_RUN_REJECTED, error_code=ToolErrorCode.CROSS_RUN_REFERENCE)
    assert summarize_traces([stopped], latency_measured=True)["provider_failures_by_kind"] == {}


def test_a_budget_stop_is_not_a_provider_failure_in_the_summary():
    stopped = trace(termination=AgentStopReason.BUDGET_EXHAUSTED, error_code=ProviderFailureKind.BUDGET_EXHAUSTED)
    assert summarize_traces([stopped], latency_measured=True)["provider_failures_by_kind"] == {}


def test_grounding_success_counts_rejected_selections():
    rejected = trace(correlation_id="c2", termination=AgentStopReason.SELECTION_REJECTED)
    summary = summarize_traces([trace(), rejected], latency_measured=True)
    assert (summary["grounding_success"]["correct"], summary["grounding_success"]["n"]) == (1, 2)


def test_an_empty_run_is_not_a_crash():
    summary = summarize_traces([], latency_measured=True)
    assert summary["questions"] == 0 and summary["cost"]["estimated_total_usd"] is None
