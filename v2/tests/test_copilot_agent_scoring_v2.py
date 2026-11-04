"""M3-B0.1: scorer v2, termination accounting, and a decision rule that cannot pass on nothing.

Every test here runs offline with scripted stand-ins. None measures a language model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from footballai_v2.copilot.evaluation.agent_rule import Criterion, DecisionRule, apply_criterion, evaluate_rule


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "evaluation" / "copilot" / "agent-v2"


def criterion(kind: str, threshold: float, min_n: int) -> Criterion:
    return Criterion(
        criterion_id="c", statement="s", measurement="m", kind=kind, threshold=threshold, denominator="d",
        min_n=min_n, origin="roadmap",
    )


# -- the rule: an unfailable metric never passes --------------------------------------


@pytest.mark.parametrize(
    ("kind", "threshold", "measurement", "min_n", "status", "reason"),
    [
        ("max_count", 0, {"cases": [], "n": 0}, 1, "not_evaluable", "no_opportunities"),
        ("max_count", 0, {"cases": [], "n": 3}, 4, "insufficient_sample", "below_min_n"),
        ("max_count", 0, {"cases": ["x"], "n": 1}, 4, "fail", "threshold_missed"),
        ("max_count", 0, {"cases": [], "n": 4}, 4, "pass", "threshold_met"),
        ("min_rate", 0.9, {"rate": None, "n": 0}, 20, "not_evaluable", "no_opportunities"),
        ("max_rate", 0.05, {"rate": None, "n": 0}, 20, "not_evaluable", "no_opportunities"),
        ("min_rate", 0.9, None, 20, "not_measured", "measurement_missing"),
        ("min_rate", 0.9, {"rate": 1.0, "n": 8}, 20, "insufficient_sample", "below_min_n"),
        ("min_rate", 0.9, {"rate": 0.5, "n": 8}, 20, "fail", "threshold_missed"),
        ("max_count", 0, [], 0, "pass", "threshold_met"),  # v1 measurements keep their meaning
    ],
)
def test_every_criterion_status_carries_a_reason(kind, threshold, measurement, min_n, status, reason):
    result = apply_criterion(criterion(kind, threshold, min_n), measurement)
    assert (result.status, result.reason) == (status, reason)


def _rule(*criteria: Criterion) -> DecisionRule:
    return DecisionRule.model_validate(
        {
            "schema": "footballai.copilot-agent-decision-rule/v1",
            "rule_version": "t",
            "status": "accepted",
            "accepted_by": "test",
            "accepted_on": "2026-11-04",
            "applies_to_split": "heldout",
            "criteria": [c.model_dump() for c in criteria],
        }
    )


LIVE = dict(mode="live", endpoint="anthropic_api", split="heldout", completed=True)


def _named(identifier: str, kind: str = "max_count", min_n: int = 1) -> Criterion:
    return criterion(kind, 0, min_n).model_copy(update={"criterion_id": identifier, "measurement": identifier})


@pytest.mark.parametrize(
    ("measurements", "verdict"),
    [
        ({"a": {"cases": [], "n": 5}, "b": {"cases": [], "n": 5}}, "PASS"),
        ({"a": {"cases": [], "n": 5}, "b": {"cases": [], "n": 0}}, "NOT_EVALUABLE"),
        ({"a": {"cases": ["x"], "n": 5}, "b": {"cases": [], "n": 0}}, "FAIL"),
        ({"a": {"cases": [], "n": 5}}, "NOT_EVALUABLE"),  # a binding measurement that is missing
    ],
)
def test_a_binding_criterion_nothing_could_fail_blocks_a_pass(measurements, verdict):
    rule = _rule(_named("a"), _named("b"))
    result = evaluate_rule(rule, measurements, **LIVE)
    assert result["verdict"] == verdict
    assert all(item["reason"] for item in result["criteria"])


def test_not_evaluable_outranks_a_thin_sample():
    rule = _rule(_named("a", min_n=10), _named("b"))
    result = evaluate_rule(rule, {"a": {"cases": [], "n": 3}, "b": {"cases": [], "n": 0}}, **LIVE)
    assert result["verdict"] == "NOT_EVALUABLE"
    thin = evaluate_rule(rule, {"a": {"cases": [], "n": 3}, "b": {"cases": [], "n": 3}}, **LIVE)
    assert thin["verdict"] == "INSUFFICIENT_SAMPLE"


# -- scorer v2: every termination lands in exactly one class ----------------------------

from footballai_v2.copilot.agent.loop import AgentLimits  # noqa: E402
from footballai_v2.copilot.agent.trace import AgentStopReason  # noqa: E402
from footballai_v2.copilot.errors import (  # noqa: E402
    ProviderFailureKind,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from footballai_v2.copilot.evaluation import agent_scoring as v2  # noqa: E402
from footballai_v2.copilot.evaluation.agent_benchmark import load_benchmark, run_case  # noqa: E402
from footballai_v2.copilot.evaluation.agent_models import (  # noqa: E402
    ScriptedAgentModel,
    Step,
    call,
    every_tool_policy,
    route_following_policy,
    say,
    select,
)

CROSS_RUN = "99999999-9999-4999-8999-999999999999"


@pytest.fixture(scope="module")
def dataset():
    return load_benchmark(DATASET_DIR)


def by_id(dataset, identifier):
    return next(case for case in (*dataset.dev, *dataset.heldout) if case.case_id == identifier)


def scripted(dataset, case_id, steps=None, *, policy=None, limits=None):
    case = by_id(dataset, case_id)

    def model_for(clock):
        if policy is not None:
            return ScriptedAgentModel(policy=policy, on_call=clock.advance)
        return ScriptedAgentModel(steps, on_call=clock.advance)

    return v2.score_case_v2(case, run_case(case, dataset.source, model_for, limits=limits).run)


def test_every_stop_reason_has_exactly_one_class_and_owner():
    for reason in AgentStopReason:
        termination, owner = v2.classify_termination(reason)
        assert isinstance(termination, v2.TerminationClass)
        assert isinstance(owner, v2.Owner)


FAULTS = {
    "provider_timeout": ("ans-status", [Step(error=ProviderTimeoutError("scripted"))], None, "provider"),
    "provider_error": (
        "ans-status",
        [Step(error=ProviderUnavailableError("scripted", kind=ProviderFailureKind.SERVER_ERROR))],
        None,
        "provider",
    ),
    "malformed_output": ("ans-status", [say("this is not the selection")], None, "agent"),
    "missing_final_answer": ("ans-status", [Step((), "max_tokens")], None, "agent"),
    "model_refusal": ("ans-status", [Step((), "refusal")], None, "agent"),
    "tool_budget_exhausted": ("ans-status", [every_tool_policy(None)], None, "agent"),
    "turn_budget_exhausted": (
        "ans-status",
        [call("get_run_status"), call("get_tracking_diagnostics")],
        AgentLimits(max_model_turns=2),
        "agent",
    ),
    "deadline_exceeded": (
        "ans-status",
        [
            call("get_run_status", takes_seconds=15),
            call("get_tracking_diagnostics", takes_seconds=15),
            call("get_evidence_limitations", takes_seconds=15),
        ],
        None,
        "agent",
    ),
    "forged_evidence_attempt": ("ans-status", [select("run_status")], None, "agent"),
    "cross_run_access_attempt": (
        "ans-status",
        [call("get_run_status", arguments={"run_id": CROSS_RUN})],
        None,
        "agent",
    ),
    "operator_budget_stop": (
        "ans-status",
        [Step(error=ProviderUnavailableError("scripted", kind=ProviderFailureKind.BUDGET_EXHAUSTED))],
        None,
        "operator",
    ),
}


@pytest.mark.parametrize("expected", FAULTS)
def test_each_injected_fault_is_classified_once_and_owned(dataset, expected):
    case_id, steps, limits, owner = FAULTS[expected]
    score = scripted(dataset, case_id, steps, limits=limits)
    assert score.termination_class.value == expected
    assert score.owner.value == owner
    assert score.answered is False and score.task_success is False


def test_budget_turn_and_deadline_stops_count_as_agent_non_completion(dataset):
    """Finding C: v1 left these out of criterion 9, so a run that only hit limits read 0%."""
    names = ("tool_budget_exhausted", "turn_budget_exhausted", "deadline_exceeded")
    stops = [scripted(dataset, "ans-status", FAULTS[name][1], limits=FAULTS[name][2]) for name in names]
    reliability = v2.score_metrics_v2(tuple(stops))["reliability"]
    assert reliability["agent_non_completion"]["correct"] == 3
    assert reliability["provider_failures"]["correct"] == 0
    assert reliability["non_completion"]["rate"] == 1.0


def test_no_termination_disappears_and_none_is_counted_twice(dataset):
    scores = tuple(scripted(dataset, case_id, steps, limits=limits) for case_id, steps, limits, _ in FAULTS.values())
    scores += (scripted(dataset, "ans-status", policy=route_following_policy),)
    accounting = v2.score_metrics_v2(scores)["termination_accounting"]
    assert sum(accounting["by_class"].values()) == accounting["questions"] == len(scores)
    assert sum(accounting["by_owner"].values()) == len(scores)
    assert set(accounting["by_class"]) == {item.value for item in v2.TerminationClass}  # zeros are shown
    reliability = v2.score_metrics_v2(scores)["reliability"]
    combined = reliability["non_completion"]["correct"]
    assert combined == reliability["provider_failures"]["correct"] + reliability["agent_non_completion"]["correct"]
    assert reliability["excluded_operator_stops"] == 1
    assert reliability["non_completion"]["n"] == len(scores) - 1  # a spend-cap stop is nobody's failure


def test_invalid_tool_arguments_are_an_event_not_a_termination(dataset):
    score = scripted(
        dataset, "ans-status", [call("get_run_status", arguments={"verbose": True}), call("get_run_status"), select("run_status")]
    )
    assert score.invalid_tool_arguments == 1
    metrics = v2.score_metrics_v2((score,))
    assert metrics["events"]["questions_with_invalid_tool_arguments"] == 1


# -- scorer v2: a valid citation to an irrelevant fact is not a correct answer ------------

from footballai_v2.copilot.evaluation.agent_benchmark import label_oracle_policy  # noqa: E402


def test_a_cited_off_topic_answer_is_irrelevant_and_not_a_success(dataset):
    """Finding D. v1 calls this outcome correct, with valid citations. v2 does not."""
    score = scripted(dataset, "adv-inside-the-vocabulary", policy=route_following_policy)
    assert score.v1.outcome_correct is True and score.v1.citations_valid is True  # the v1 blind spot
    assert score.citations_valid is True and score.factually_valid is True
    assert score.relevant is False and score.task_success is False
    assert "adv-inside-the-vocabulary" in v2.score_metrics_v2((score,))["irrelevant_answers"]["cases"]


def test_the_right_claim_padded_with_an_off_topic_one_is_irrelevant(dataset):
    """The required claim is there, and so is a cited fact nobody asked for."""
    case = by_id(dataset, "multi-status-artifacts")
    narrow = case.model_copy(
        update={
            "expected": case.expected.model_copy(
                update={"required_claims": ("run_status",), "permitted_claims": ("run_status",)}
            )
        }
    )
    steps = [call("get_run_status", "get_artifact_metadata"), select("run_status", "artifact_integrity")]
    run = run_case(narrow, dataset.source, lambda clock: ScriptedAgentModel(steps, on_call=clock.advance)).run
    score = v2.score_case_v2(narrow, run)
    assert score.citations_valid is True and score.complete is True
    assert score.relevant is False and score.task_success is False


def test_an_answer_to_a_question_with_no_answer_is_irrelevant(dataset):
    score = scripted(dataset, "zero-model-real", policy=route_following_policy)
    assert score.answered and score.false_answer
    assert score.relevant is False and score.abstention_correct is False and score.task_success is False


def test_a_relevant_but_partial_answer_is_incomplete(dataset):
    score = scripted(
        dataset, "multi-status-artifacts", [call("get_run_status", "get_artifact_metadata"), select("run_status")]
    )
    assert score.answered and score.relevant is True
    assert score.complete is False and score.task_success is False


def test_abstaining_on_an_answerable_question_is_an_incorrect_abstention(dataset):
    score = scripted(dataset, "ans-status", [call("get_run_status"), select()])
    assert score.termination_class is v2.TerminationClass.ABSTAINED
    assert score.incorrect_abstention is True and score.abstention_correct is False
    assert score.relevant is None and score.citations_valid is None  # nothing was answered


def test_the_label_oracle_succeeds_on_both_kinds_of_case(dataset):
    for case_id in ("ans-status", "multi-status-artifacts", "zero-model-real"):
        case = by_id(dataset, case_id)
        score = scripted(dataset, case_id, policy=label_oracle_policy(case))
        assert score.task_success is True, case_id
    assert scripted(dataset, "zero-model-real", policy=label_oracle_policy(by_id(dataset, "zero-model-real"))).abstention_correct


def test_the_nine_judgements_are_reported_apart(dataset):
    metrics = v2.score_metrics_v2((scripted(dataset, "ans-status", policy=route_following_policy),))
    for key in (
        "factual_validity", "citation_validity", "answer_relevance", "answer_completeness",
        "abstention_correctness", "tool_selection_accuracy", "task_success", "reliability",
    ):
        assert key in metrics
    assert set(metrics["reliability"]) >= {"provider_failures", "agent_non_completion"}


# -- rule v2: proposed, versioned, and honest about what the split can evaluate -------------

import hashlib  # noqa: E402
import json  # noqa: E402

from footballai_v2.copilot.evaluation.agent_benchmark import CaseMode, replay_script  # noqa: E402
from footballai_v2.copilot.evaluation.agent_rule import RULE_FILE_V2, RuleStatus, load_rule  # noqa: E402

RULE_V1_SHA256 = "47feefb33fad491d83d18105b78485c453f376b7b179be86cbf5cb9df34cdf3c"


@pytest.fixture(scope="module")
def rule_v2():
    return load_rule(DATASET_DIR, RULE_FILE_V2)


def test_rule_v1_is_untouched():
    assert hashlib.sha256((DATASET_DIR / "decision-rule.json").read_bytes()).hexdigest() == RULE_V1_SHA256


def test_rule_v2_is_proposed_and_nobody_accepted_it(rule_v2):
    rule, _sha = rule_v2
    assert rule.status is RuleStatus.PROPOSED
    assert rule.accepted_by is None and rule.accepted_on is None
    assert rule.scorer_version == v2.SCORER_VERSION
    assert rule.supersedes["sha256"] == RULE_V1_SHA256
    for item in rule.criteria:
        assert (item.origin == "proposed_addition") == (not item.binding)


def test_every_measurement_rule_v2_names_exists(rule_v2):
    empty = v2.measurements_v2((), (), (), latency_measured=True)
    for item in rule_v2[0].criteria:
        assert item.measurement in empty, item.measurement


def heldout_scores(dataset, policy_for):
    live = tuple(
        v2.score_case_v2(c, run_case(c, dataset.source, lambda clock, c=c: ScriptedAgentModel(policy=policy_for(c), on_call=clock.advance)).run)
        for c in dataset.heldout
        if c.mode is CaseMode.LIVE_ELIGIBLE
    )
    replayed = tuple(
        v2.score_case_v2(c, run_case(c, dataset.source, replay_script(c)).run)
        for c in dataset.heldout
        if c.mode is CaseMode.REPLAY_ONLY
    )
    return live, replayed


def test_finding_b_new_false_answers_has_no_opportunities_on_heldout(dataset):
    live, replayed = heldout_scores(dataset, lambda _c: route_following_policy)
    measured = v2.measurements_v2(live, replayed, live, latency_measured=False)
    assert measured["new_false_answers"] == {"cases": [], "n": 0}
    assert measured["live_false_answers"]["n"] == 4  # the absolute version can be evaluated
    assert len(measured["live_false_answers"]["cases"]) == 4


def test_rule_v2_cannot_pass_on_the_current_heldout_split(dataset, rule_v2):
    """Even the label oracle gets NOT_EVALUABLE: criterion 4 has nothing that could fail it."""
    oracle, replayed = heldout_scores(dataset, label_oracle_policy)
    rules, _ = heldout_scores(dataset, lambda _c: route_following_policy)
    measured = v2.measurements_v2(oracle, replayed, rules, latency_measured=False)
    measured["end_to_end_reaching_a_model"] = {"p95_seconds": 1.0, "n": 24}  # stands in for a measured clock
    accepted = rule_v2[0].model_copy(update={"status": RuleStatus.ACCEPTED, "accepted_by": "t", "accepted_on": "t"})
    result = evaluate_rule(accepted, measured, **LIVE)
    statuses = {item["criterion"]: item["status"] for item in result["criteria"]}
    assert statuses["no_new_false_answers"] == "not_evaluable"
    assert result["verdict"] == "NOT_EVALUABLE"


def test_finding_e_replay_no_longer_counts_toward_citation_validity(dataset):
    oracle, replayed = heldout_scores(dataset, label_oracle_policy)
    rules, _ = heldout_scores(dataset, lambda _c: route_following_policy)
    measured = v2.measurements_v2(oracle, replayed, rules, latency_measured=False)
    assert measured["live_citation_validity"]["n"] == 20  # v1 read 21: one was a scripted replay
    assert measured["live_tool_selection_accuracy"]["n"] == 20
    assert measured["live_answer_task_success"]["n"] == 20  # router-stopped cases are not in it


# -- negative controls: each bad policy must fail something that binds ---------------------

from footballai_v2.cli import copilot_agent_reliability as reliability_cli  # noqa: E402
from footballai_v2.copilot.evaluation.agent_controls import CONTROLS  # noqa: E402

REQUIRED_CONTROLS = {
    "label_oracle", "route_following", "always_abstain", "select_every_claim", "off_topic_only",
    "right_tool_wrong_claims", "no_tool", "unauthorized_tool", "exhaust_tool_budget",
    "exhaust_turn_budget", "exceed_deadline", "malformed_output",
}


@pytest.fixture(scope="module")
def matrix(dataset):
    rule_v1 = load_rule(DATASET_DIR)
    rule_v2 = load_rule(DATASET_DIR, RULE_FILE_V2)
    return reliability_cli.control_matrix(dataset, rule_v1[0], rule_v2[0])


def test_every_requested_control_exists():
    assert {control.name for control in CONTROLS} == REQUIRED_CONTROLS


def test_each_control_produces_the_termination_it_is_named_for(matrix):
    expected = {
        "exhaust_tool_budget": "tool_budget_exhausted",
        "exhaust_turn_budget": "turn_budget_exhausted",
        "exceed_deadline": "deadline_exceeded",
        "malformed_output": "malformed_output",
        "no_tool": "forged_evidence_attempt",
    }
    for name, termination in expected.items():
        by_class = matrix[name]["terminations"]
        assert by_class[termination] == matrix[name]["questions_reaching_a_model"], name


def test_finding_a_v1_lets_the_deterministic_baseline_pass_every_measurable_roadmap_criterion(matrix):
    """A script measures no latency, so p95 is not scored. Every other roadmap criterion passes."""
    v1 = matrix["route_following"]["rule_v1"]
    roadmap = {item["criterion"]: item["status"] for item in v1["criteria"] if item["origin"] == "roadmap"}
    assert len(roadmap) == 10 and roadmap.pop("end_to_end_p95") == "not_measured"
    assert set(roadmap.values()) == {"pass"}


def test_finding_c_v1_reads_zero_failures_when_every_question_hit_the_tool_budget(matrix):
    v1 = {item["criterion"]: item for item in matrix["exhaust_tool_budget"]["rule_v1"]["criteria"]}
    assert v1["provider_failures_and_invalid_outputs"]["value"] == 0.0
    v2_ = {item["criterion"]: item for item in matrix["exhaust_tool_budget"]["rule_v2"]["criteria"]}
    assert v2_["non_completion"]["status"] == "fail" and v2_["non_completion"]["value"] == 1.0


def test_with_the_recommended_additions_every_negative_control_fails_under_v2(matrix):
    for name, row in matrix.items():
        verdict = row["rule_v2"]["verdict_if_accepted"]["with_recommended_additions"]
        if name == "label_oracle":
            assert verdict == "NOT_EVALUABLE", verdict  # criterion 4 cannot be evaluated; nothing fails
        else:
            assert verdict == "FAIL", name


def test_the_correct_control_fails_nothing(matrix):
    statuses = {item["status"] for item in matrix["label_oracle"]["rule_v2"]["criteria"]}
    assert "fail" not in statuses


def test_latency_is_never_scored_from_a_scripted_clock(matrix):
    for row in matrix.values():
        p95 = next(item for item in row["rule_v2"]["criteria"] if item["criterion"] == "end_to_end_p95")
        assert p95["status"] == "not_measured"


# -- dataset audit: reported here, remediated in M3-B0.2 ------------------------------------


@pytest.fixture(scope="module")
def audit(dataset):
    return reliability_cli.dataset_audit(dataset)


def test_finding_f_the_seven_cross_fixture_overlaps_are_reported(audit):
    overlap = audit["heldout_questions_also_in_dev"]
    assert overlap["cases"] == [
        "h-ans-active", "h-ans-duration", "h-ans-status-synthetic", "h-gap-diagnostics",
        "h-gap-legacy", "h-gap-reliability", "h-multi-summary-partial",
    ]
    assert len(overlap["reaching_a_model"]) == 4


def test_a_heldout_question_inside_a_tool_description_is_found(audit):
    assert audit["heldout_questions_inside_tool_descriptions"] == ["h-gap-legacy"]
    assert audit["heldout_questions_inside_system_instructions"] == []


def test_finding_g_no_adversarial_or_cross_run_heldout_question_reaches_a_model(audit):
    categories = audit["heldout_live_cases_by_category"]
    for name in ("adversarial_instructions", "cross_run_attack", "unsupported", "missing_evidence"):
        assert categories[name]["reach_a_model"] == 0, name


def test_the_committed_reliability_report_is_exactly_what_the_code_produces(dataset):
    import json

    committed = (DATASET_DIR / reliability_cli.REPORT_FILE).read_text(encoding="utf-8")
    assert committed == json.dumps(reliability_cli.build_report(dataset, DATASET_DIR), indent=2) + "\n"


def test_the_v1_baseline_scores_are_unchanged():
    """Historical numbers are recomputed by scorer v1 and must match the committed file byte for byte."""
    baseline = (DATASET_DIR / "baseline-report.json").read_bytes()
    assert hashlib.sha256(baseline).hexdigest() == "baa72adf8f07a31ef939b9cc978293330429ddcad4cd224f55edf2466d85c8af"


def test_old_versus_new_shows_the_irrelevant_answer_v1_called_correct(dataset):
    report = json.loads((DATASET_DIR / reliability_cli.REPORT_FILE).read_text(encoding="utf-8"))
    assert report["old_versus_new"]["dev"]["route_following"]["irrelevant_answers_v1_counted_as_correct_outcomes"] == 1
    assert report["live_model_requests"] == 0 and report["real_model_quality_measured"] is False


# -- finding H: an approval pins everything that decides a score ----------------------------

from footballai_v2.copilot.evaluation import agent_live  # noqa: E402
from footballai_v2.copilot.evaluation.agent_benchmark import Split  # noqa: E402

KEY_ONLY = {"ANTHROPIC_API_KEY": "canary-value-not-real"}
PINNED = {
    "scorer_version": "agent-scorer/v2",
    "instruction_version": "copilot.agent-instructions/v1",
    "system_instructions_sha256": "s" * 64,
    "tool_definitions_sha256": "t" * 64,
    "sdk_version": "2.3.0",
    "source_commit": "a" * 40,
    "source_tree_clean": True,
}


def approval_v2(**changes):
    values = dict(
        schema="footballai.copilot-agent-live-approval/v2", status="approved", approved_by="owner",
        approved_on="2026-11-04", stage="decision", split="heldout", model="claude-haiku-5-5",
        max_model_requests=120, max_cost_usd=0.5, decision_rule_sha256="r" * 64, cases_sha256="c" * 64,
        **{key: value for key, value in PINNED.items() if key != "source_tree_clean"},
    )
    return agent_live.LiveApproval.model_validate({**values, **changes})


def check_v2(record, **changes):
    arguments = dict(
        stage="decision", split=Split.HELDOUT, model="claude-haiku-5-5", max_requests=120, max_cost_usd=0.5,
        cases_sha256="c" * 64, rule_sha256="r" * 64, rule_accepted=True, provenance=PINNED, environ=KEY_ONLY,
    )
    agent_live.check_approval(record, **{**arguments, **changes})


def test_a_fully_pinned_approval_passes():
    check_v2(approval_v2())


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"provenance": {**PINNED, "scorer_version": "agent-scorer/v1"}}, "approval_other_scorer"),
        ({"provenance": {**PINNED, "instruction_version": "copilot.agent-instructions/v2"}}, "approval_other_prompt"),
        ({"provenance": {**PINNED, "system_instructions_sha256": "x" * 64}}, "approval_other_prompt"),
        ({"provenance": {**PINNED, "tool_definitions_sha256": "x" * 64}}, "approval_other_tools"),
        ({"provenance": {**PINNED, "sdk_version": "2.4.0"}}, "approval_other_sdk"),
        ({"provenance": {**PINNED, "source_commit": "b" * 40}}, "approval_other_commit"),
        ({"provenance": {**PINNED, "source_commit": None}}, "approval_other_commit"),
        ({"provenance": {**PINNED, "source_tree_clean": False}}, "source_tree_modified"),
        ({"stage": "development"}, "approval_other_stage"),
    ],
)
def test_each_pin_is_enforced(changes, code):
    with pytest.raises(agent_live.LiveRefused) as caught:
        check_v2(approval_v2(), **changes)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("record", "changes", "code"),
    [
        ({"stage": "protocol_probe", "split": "dev", "max_model_requests": 6}, {"stage": "protocol_probe", "split": Split.DEV, "max_requests": 5}, "probe_over_its_cap"),
        ({"stage": "protocol_probe", "split": "heldout"}, {"stage": "protocol_probe"}, "stage_needs_dev"),
        ({"stage": "development", "split": "heldout"}, {"stage": "development"}, "stage_needs_dev"),
        ({"split": "dev"}, {"split": Split.DEV}, "decision_needs_heldout"),
    ],
)
def test_each_stage_has_its_own_bounds(record, changes, code):
    with pytest.raises(agent_live.LiveRefused) as caught:
        check_v2(approval_v2(**record), **changes)
    assert caught.value.code == code


def test_a_probe_and_a_development_run_need_no_accepted_rule():
    probe = approval_v2(stage="protocol_probe", split="dev", max_model_requests=5)
    check_v2(probe, stage="protocol_probe", split=Split.DEV, max_requests=5, rule_accepted=False)
    development = approval_v2(stage="development", split="dev")
    check_v2(development, stage="development", split=Split.DEV, rule_accepted=False)


def test_a_v1_approval_no_longer_parses():
    record = {"schema": "footballai.copilot-agent-live-approval/v1", "status": "approved"}
    with pytest.raises(ValueError):
        agent_live.LiveApproval.model_validate(record)


def test_the_template_pins_every_field_and_approves_nothing():
    record = json.loads((DATASET_DIR / "live-approval.template.json").read_text(encoding="utf-8"))
    assert record["schema"] == "footballai.copilot-agent-live-approval/v2" and record["status"] == "not_approved"
    assert set(PINNED) - {"source_tree_clean"} <= set(record)


def test_current_provenance_names_the_code_that_would_score(monkeypatch):
    monkeypatch.setattr(agent_live, "source_state", lambda: ("f" * 40, True))
    current = agent_live.current_provenance()
    assert current["scorer_version"] == "agent-scorer/v2"
    assert current["source_commit"] == "f" * 40 and current["source_tree_clean"] is True
    assert len(current["system_instructions_sha256"]) == 64 and len(current["tool_definitions_sha256"]) == 64


def test_criterion_4_has_no_opportunities_on_either_split(dataset):
    """route_following selects everything it retrieved, and a question that reaches a model
    always retrieves something. So it false-answers every reachable non-answer question, on
    both splits, and "no new false answers" against it can never fail. See D7."""
    for cases in (dataset.dev, dataset.heldout):
        live = tuple(
            v2.score_case_v2(c, run_case(c, dataset.source, lambda _clock: ScriptedAgentModel(policy=route_following_policy)).run)
            for c in cases
            if c.mode is CaseMode.LIVE_ELIGIBLE
        )
        measured = v2.measurements_v2(live, (), live, latency_measured=False)
        assert measured["new_false_answers"]["n"] == 0
        assert measured["live_false_answers"]["n"] == len(measured["live_false_answers"]["cases"]) > 0


# -- review fixes ----------------------------------------------------------------------------

import subprocess  # noqa: E402


def test_the_always_shown_label_does_not_make_an_answer_relevant(dataset):
    """The label is rendered on every answer, so it cannot be the 'required claim' that was shown."""
    case = by_id(dataset, "multi-status-artifacts")
    labelled = case.model_copy(
        update={
            "expected": case.expected.model_copy(
                update={
                    "required_claims": ("output_nature", "run_status"),
                    "permitted_claims": ("output_nature", "run_status", "artifact_integrity"),
                }
            )
        }
    )
    steps = [call("get_run_status", "get_artifact_metadata"), select("artifact_integrity")]
    run = run_case(labelled, dataset.source, lambda clock: ScriptedAgentModel(steps, on_call=clock.advance)).run
    assert v2.score_case_v2(labelled, run).relevant is False


def test_completeness_is_not_judged_for_a_question_with_no_answer(dataset):
    score = scripted(dataset, "zero-model-real", policy=route_following_policy)
    assert score.complete is None and score.relevant is False


def test_a_question_stopped_before_its_first_turn_counts_as_wrong_tool_selection(dataset):
    score = scripted(
        dataset, "ans-status", [Step(error=ProviderUnavailableError("scripted", kind=ProviderFailureKind.NOT_CONFIGURED))]
    )
    assert score.run.result.trace.model_turns == 0 and score.attempted
    assert score.tools_correct is False
    operator = scripted(dataset, "ans-status", FAULTS["operator_budget_stop"][1])
    assert operator.tools_correct is None  # nobody's failure: left out, like criterion 9


def test_latency_leaves_out_operator_stops(dataset):
    stopped = scripted(dataset, "ans-status", FAULTS["operator_budget_stop"][1])
    answered = scripted(dataset, "ans-status", policy=route_following_policy)
    measured = v2.measurements_v2((stopped, answered), (), (answered,), latency_measured=True)
    assert measured["end_to_end_reaching_a_model"]["n"] == 1


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_an_untracked_module_under_the_source_tree_is_a_modified_tree(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "v2/src").mkdir(parents=True)
    (tmp_path / "evaluation/copilot").mkdir(parents=True)
    (tmp_path / "v2/src/tracked.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", "base")
    assert agent_live.source_state(tmp_path)[1] is True
    (tmp_path / "evaluation/copilot/live-dev.json").write_text("{}")  # a stage report: allowed
    assert agent_live.source_state(tmp_path)[1] is True
    (tmp_path / "v2/src/shadow").mkdir()
    (tmp_path / "v2/src/shadow/__init__.py").write_text("x = 2\n")  # code that would run: refused
    assert agent_live.source_state(tmp_path)[1] is False
