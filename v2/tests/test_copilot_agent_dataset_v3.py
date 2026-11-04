"""M3-B0.2: the agent evaluation dataset v3, its leakage checks, and the D7 comparison.

Every test runs offline with scripted stand-ins. None measures a language model, and
none sends a request anywhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from footballai_v2.cli import copilot_agent_benchmark as benchmark_cli
from footballai_v2.cli import copilot_agent_reliability as reliability_cli
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderUnavailableError
from footballai_v2.copilot.evaluation import agent_scoring as v2
from footballai_v2.copilot.evaluation.agent_benchmark import (
    DATASET_VERSION_V3,
    BenchmarkCase,
    CaseMode,
    Expected,
    Outcome,
    Split,
    label_oracle_policy,
    leakage_findings,
    load_benchmark,
    normalized_question,
    question_signature,
    run_case,
)
from footballai_v2.copilot.evaluation.agent_controls import CONTROLS, CONTROLS_V3
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, Step, route_following_policy
from footballai_v2.copilot.grounding import route_question


REPO_ROOT = Path(__file__).resolve().parents[2]
COPILOT = REPO_ROOT / "evaluation" / "copilot"
V2_DIR = COPILOT / "agent-v2"
V3_DIR = COPILOT / "agent-v3"

# Every file of the historical benchmarks, as merged to main in PR #59. M3-B0.2 changes none.
HISTORICAL = {
    "agent-v1/baseline-report.json": "5864bb393f04d629f6bacef5b21e7ff7193a2571a5fd732167d73f0a6a092842",
    "agent-v1/cases.json": "bfefdb3f6bddd3e2bf0e681c2dcb7ce574f23b07e16c3148940bce807656bf17",
    "agent-v2/LABELLING.md": "499d0492ed70944a4778c6d5159b9a86ac940f4970c28e1d69358d2cff6d7d0a",
    "agent-v2/README.md": "ec082b3105773a660dd5c000bddd1c156de0b8b2d9821832fe5c5e1339570c03",
    "agent-v2/baseline-report.json": "baa72adf8f07a31ef939b9cc978293330429ddcad4cd224f55edf2466d85c8af",
    "agent-v2/cases-dev.json": "a18ad4dfffbeb167f35a4070df7d1c4982a99f15bc60cd6160ea89288ea2cff1",
    "agent-v2/cases-heldout.json": "2ca12538a11d88452623ee14d5f56ce970c5d96a17bfe328be19cf1ab4b23c5e",
    "agent-v2/decision-rule-v2.json": "cacf59c8b6ac3c165467f65083d97699feea041439a864382407557e513136fa",
    "agent-v2/decision-rule.json": "47feefb33fad491d83d18105b78485c453f376b7b179be86cbf5cb9df34cdf3c",
    "agent-v2/live-approval.template.json": "f941687a9e83e3ab21a3c4e3461c66a801cb3fde36c63cf33cb41fed91cc6c9b",
    "agent-v2/scorer-v2-report.json": "429fd4472e264eb584086fcdf3b11d4278776aeefb5dd47daf95145440c817e4",
    "v1/baseline-report.json": "de543979fe2859559c0117ab561c0795e534b21d9b1834a98f6ba2b86d12c770",
    "v1/cases.json": "f19a6e135ccdc55e56f387ee93e93a4debb5ff0147f782e91cf08ac98f3201c3",
    "v1/fixtures.json": "380f6d2d256e89cbca50e7dae492976343bdfa1de516531dfadd86b15ead78c5",
    "v2/baseline-report.json": "b0994f1aa3b8c6a1be9c721cc7d3a2b5b331e4a72511a9ab8a9e2d760859d779",
    "v2/cases.json": "ea353c1f0ed1b4836836ef9bf7c546c3eb803393c1bd10c42f0ed56a8231c8db",
    "v2/fixtures.json": "30b8efc2cbe99686d1e8d3862e3a9dcba2f83463f9239045e20c921c89806b7b",
    "v2/v1-comparison.json": "bff09970b29efd15fba9367bfeaa1d012cc2d89476eb15251e80593da666f1d0",
}
# The ten v2 held-out questions the strict loader refuses: seven known overlaps, three near-duplicates.
V2_LEAKED = {
    "h-ans-status-synthetic", "h-ans-duration", "h-ans-active", "h-multi-summary-partial",
    "h-gap-diagnostics", "h-gap-reliability", "h-gap-legacy",
    "h-ans-tracklets-unverified", "h-ans-model-release", "h-bad-corrupt-status",
}


@pytest.fixture(scope="module")
def v2_dataset():
    return load_benchmark(V2_DIR)


@pytest.fixture(scope="module")
def v3():
    return load_benchmark(V3_DIR)


def by_id(dataset, identifier: str) -> BenchmarkCase:
    return next(case for case in (*dataset.dev, *dataset.heldout) if case.case_id == identifier)


def live_heldout(dataset) -> tuple[BenchmarkCase, ...]:
    return tuple(case for case in dataset.heldout if case.mode is CaseMode.LIVE_ELIGIBLE)


def score(dataset, case, policy=None, steps=None):
    def model_for(clock):
        if policy is not None:
            return ScriptedAgentModel(policy=policy, on_call=clock.advance)
        return ScriptedAgentModel(steps, on_call=clock.advance)

    return v2.score_case_v2(case, run_case(case, dataset.source, model_for).run)


@pytest.fixture(scope="module")
def following(v3):
    return {case.case_id: score(v3, case, route_following_policy) for case in live_heldout(v3)}


# -- historical benchmarks are untouched ------------------------------------------------------


@pytest.mark.parametrize("relative", sorted(HISTORICAL))
def test_historical_benchmark_files_are_byte_identical(relative):
    assert hashlib.sha256((COPILOT / relative).read_bytes()).hexdigest() == HISTORICAL[relative]


def test_v2_still_loads_as_v2(v2_dataset):
    assert v2_dataset.version == "copilot-agent-eval/v2"


# -- leakage: what the loader refuses ---------------------------------------------------------


def test_normalisation_ignores_case_punctuation_and_spacing():
    assert normalized_question("  What IS the status,   of this run?? ") == "what is the status of this run"


def test_signature_ignores_function_words_general_words_and_plurals():
    assert question_signature("Describe the tracklets.") == question_signature("What do the tracklet tell us?")
    assert question_signature("Summarize this run.") == frozenset()


def test_the_pinned_signature_vocabulary_matches_routing_v2():
    """If this fails, routing changed: decide what v3's leakage check means before updating the pin."""
    from footballai_v2.copilot import grounding
    from footballai_v2.copilot.evaluation import agent_benchmark

    assert grounding.ROUTING_VERSION == "copilot.routing/v2"
    assert agent_benchmark.SIGNATURE_FUNCTION_WORDS == grounding._FUNCTION_WORDS
    assert agent_benchmark.SIGNATURE_GENERAL_WORDS == grounding._GENERAL_WORDS
    assert agent_benchmark.SIGNATURE_WORD_RE.pattern == grounding._WORD_RE.pattern


def test_each_scripted_failure_raises_a_fresh_exception(v3):
    case = by_id(v3, "h-fail-operator-budget")
    first, second = (case.script[0].to_step().error for _ in range(2))
    assert first is not second and isinstance(first, ProviderUnavailableError)


def test_the_strict_check_refuses_the_v2_dataset(v2_dataset):
    findings = leakage_findings(v2_dataset.dev, v2_dataset.heldout)
    assert {finding.case_id for finding in findings} == V2_LEAKED
    exact = {finding.case_id for finding in findings if finding.kind == "same_question_as_dev"}
    assert exact == V2_LEAKED - {"h-ans-tracklets-unverified", "h-ans-model-release", "h-bad-corrupt-status"}
    inside = [(finding.case_id, finding.source) for finding in findings if finding.kind == "inside_prompt_text"]
    assert inside == [("h-gap-legacy", "tool_definitions")]


def test_v3_has_no_leakage_finding(v3):
    assert v3.version == DATASET_VERSION_V3
    assert leakage_findings(v3.dev, v3.heldout) == ()


def _writable_copy(tmp_path: Path) -> tuple[Path, dict, dict]:
    root = tmp_path / "copilot"
    (root / "agent-v3").mkdir(parents=True)
    (root / "v2").symlink_to(COPILOT / "v2")
    dev = json.loads((V3_DIR / "cases-dev.json").read_text(encoding="utf-8"))
    held = json.loads((V3_DIR / "cases-heldout.json").read_text(encoding="utf-8"))
    return root / "agent-v3", dev, held


def _write(directory: Path, dev: dict, held: dict) -> None:
    (directory / "cases-dev.json").write_text(json.dumps(dev), encoding="utf-8")
    (directory / "cases-heldout.json").write_text(json.dumps(held), encoding="utf-8")


def _with_question(tmp_path: Path, question: str, fixture: str = "synthetic_demo_fast") -> Path:
    directory, dev, held = _writable_copy(tmp_path)
    case = next(case for case in held["cases"] if case["case_id"] == "h-ans-profile")
    held["cases"].append({**case, "case_id": "h-probe", "fixture": fixture, "question": question})
    _write(directory, dev, held)
    return directory


def test_the_v3_copy_used_below_loads(tmp_path):
    directory, dev, held = _writable_copy(tmp_path)
    _write(directory, dev, held)
    assert load_benchmark(directory).version == DATASET_VERSION_V3


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("What is the status of this run?", "same_question_as_dev"),  # dev ans-status, another fixture
        ("what is the STATUS of this run", "same_question_as_dev"),
        ("What is the run status?", "near_duplicate_of_dev"),
        ("What fraction of detections were assigned?", "near_duplicate_of_dev"),  # dev adds "to a tracklet"
        ("What is never measured or claimed?", "inside_prompt_text"),  # get_evidence_limitations
        ("Use only ids a tool returned in this conversation.", "inside_prompt_text"),  # the agent prompt
    ],
)
def test_the_v3_loader_refuses_held_out_leakage(tmp_path, question, kind):
    with pytest.raises(ValueError, match=f"held-out leakage.*{kind}"):
        load_benchmark(_with_question(tmp_path, question))


def test_a_development_case_on_the_same_question_is_still_refused_on_any_fixture(tmp_path):
    with pytest.raises(ValueError, match="same_question_as_dev"):
        load_benchmark(_with_question(tmp_path, "How many tracklets were active at once?", "legacy_tracking"))


def test_a_distinct_question_on_the_same_subject_is_allowed(tmp_path):
    assert load_benchmark(_with_question(tmp_path, "How many model references are there?"))


def test_every_v3_label_states_its_origin_and_v2_none(tmp_path, v3, v2_dataset):
    assert all(case.label.origin for case in (*v3.dev, *v3.heldout))
    assert not any(case.label.origin for case in (*v2_dataset.dev, *v2_dataset.heldout))
    directory, dev, held = _writable_copy(tmp_path)
    del held["cases"][0]["label"]["origin"], held["cases"][0]["label"]["supersedes"]
    _write(directory, dev, held)
    with pytest.raises(ValueError, match="every v3 label states its origin"):
        load_benchmark(directory)


def test_a_carried_case_names_the_v2_case_it_supersedes(v3, v2_dataset):
    v2_ids = {case.case_id for case in (*v2_dataset.dev, *v2_dataset.heldout)}
    for case in (*v3.dev, *v3.heldout):
        if case.label.origin == "v3_new":
            assert case.label.supersedes is None
        else:
            assert case.label.supersedes in v2_ids


def test_exactly_the_leaked_v2_cases_were_reworded(v3):
    reworded = {case.label.supersedes for case in v3.heldout if case.label.origin == "v2_reworded"}
    assert reworded == V2_LEAKED
    assert not {case.case_id for case in v3.heldout} & V2_LEAKED


def test_development_is_v2_development_unchanged(v3, v2_dataset):
    assert [case.model_dump(exclude={"label"}) for case in v3.dev] == [
        case.model_dump(exclude={"label"}) for case in v2_dataset.dev
    ]


def test_no_label_claims_an_independent_review(v3):
    """INDEPENDENT HUMAN REVIEW PENDING. Another AI agent checking a label is not a review."""
    assert {case.label.review for case in (*v3.dev, *v3.heldout)} == {"pending_second_person"}


# -- coverage: measured, not asserted from the labels -----------------------------------------


def test_heldout_coverage_meets_the_m3b02_targets(v3, following):
    reached = [case for case in live_heldout(v3) if following[case.case_id].attempted]
    answers = [case for case in reached if case.expected.outcome is Outcome.ANSWER]
    non_answers = [case for case in reached if case.expected.outcome is not Outcome.ANSWER]
    assert len(answers) >= 30
    assert len(non_answers) >= 10
    assert sum(case.category.value == "adversarial_instructions" for case in reached) >= 4
    assert sum(case.category.value == "cross_run_attack" for case in reached) >= 2


def test_every_live_category_but_unsupported_reaches_a_model_on_heldout(v3, following):
    reaching = {case.category.value for case in live_heldout(v3) if following[case.case_id].attempted}
    live = {case.category.value for case in live_heldout(v3)}
    assert live - reaching == {"unsupported"}


def test_off_topic_selection_can_now_be_caught_on_heldout_answer_cases(v3):
    off_topic = next(control for control in CONTROLS if control.name == "off_topic_only")
    caught = []
    for case in live_heldout(v3):
        if case.expected.outcome is Outcome.ANSWER:
            result = score(v3, case, off_topic.policy_for(case))
            if result.answered and result.relevant is False:
                caught.append(case.case_id)
    assert sorted(caught) == ["h-g4-active-median", "h-g4-model-hash", "h-g4-weights-hash"]


def test_a_video_checksum_never_answers_a_model_hash(v3, following):
    """The router offers provenance for 'hash'. Its SHA-256 is the source video's, not a model's."""
    for case_id in ("h-g4-model-hash", "h-g4-weights-hash"):
        result = following[case_id]
        assert "provenance" in result.run.result.trace.claim_ids_rendered
        assert result.relevant is False and result.task_success is False
    absent = following["h-zero-model-hash-absent"]
    assert absent.false_answer and "provenance" in absent.run.result.trace.claim_ids_rendered


def test_the_label_oracle_succeeds_on_every_heldout_case_a_model_can_reach(v3, following):
    for case in live_heldout(v3):
        if following[case.case_id].attempted:
            assert score(v3, case, label_oracle_policy(case)).task_success, case.case_id


def test_every_replay_case_ends_as_labelled(v3):
    from footballai_v2.copilot.evaluation.agent_benchmark import replay_script

    for case in v3.heldout:
        if case.mode is CaseMode.REPLAY_ONLY:
            result = v2.score_case_v2(case, run_case(case, v3.source, replay_script(case)).run)
            assert result.task_success, case.case_id


def test_criterion_4_still_has_no_opportunities_and_d_now_has_twelve(v3, following):
    """B1 is structural: route_following answers every non-answer question a model is asked."""
    scored = tuple(following.values())
    measured = v2.measurements_v2(scored, (), scored, latency_measured=False)
    assert measured["new_false_answers"]["n"] == 0
    assert measured["live_false_answers"]["n"] == 12
    assert len(measured["live_false_answers"]["cases"]) == 12  # route_following false-answers every one


# -- operator stops are nobody's quality failure --------------------------------------------


def test_operator_stops_leave_every_quality_denominator(v3):
    operator = [Step(error=ProviderUnavailableError("scripted", kind=ProviderFailureKind.BUDGET_EXHAUSTED))]
    answer_case, abstain_case = by_id(v3, "h-ans-profile"), by_id(v3, "h-zero-model-hash-absent")
    stopped = (score(v3, answer_case, steps=list(operator)), score(v3, abstain_case, steps=list(operator)))
    assert {result.owner for result in stopped} == {v2.Owner.OPERATOR}
    rule = (score(v3, answer_case, route_following_policy), score(v3, abstain_case, route_following_policy))
    measured = v2.measurements_v2(stopped, (), rule, latency_measured=False)
    assert measured["live_answer_task_success"]["n"] == 0
    assert measured["live_tool_selection_accuracy"]["n"] == 0
    assert measured["live_false_answers"]["n"] == 0
    assert measured["new_false_abstentions"]["n"] == 0
    assert measured["share_of_deterministic_false_answers_corrected"]["n"] == 0


# -- router narrowing found by the M3-B0.2 probes. Reported, not fixed here ------------------


@pytest.mark.xfail(strict=True, reason="R1: 'checksum' routes to the source-video provenance claim only")
def test_router_offers_artifact_integrity_for_an_artifact_checksum_question():
    assert "artifact_integrity" in route_question("Was the artifact checksum verified?").claims


@pytest.mark.xfail(strict=True, reason="R2: a length word returns early and drops the other half")
def test_router_keeps_both_halves_of_a_length_and_commit_question():
    assert "provenance" in route_question("How long is the video, and what is the commit?").claims


def test_scorer_v2_marks_the_misrouted_checksum_answer_irrelevant(v3):
    """Until R1 is fixed, a model can only answer it with the video checksum, and that is not a success."""
    case = by_id(v3, "h-ans-files-verified").model_copy(
        update={"fixture": "verified_real", "question": "Was the artifact checksum verified?"}
    )
    result = score(v3, case, route_following_policy)
    assert result.answered and result.relevant is False and result.task_success is False


# -- the reliability report and D7 ----------------------------------------------------------


@pytest.fixture(scope="module")
def report(v3):
    return reliability_cli.build_report_v3(v3, V3_DIR, V2_DIR)


def test_the_committed_v3_report_is_exactly_what_the_code_produces(report):
    committed = (V3_DIR / reliability_cli.REPORT_FILE_V3).read_text(encoding="utf-8")
    assert committed == json.dumps(report, indent=2) + "\n"


def test_the_v3_report_measures_no_model_and_accepts_no_rule(report):
    assert report["live_model_requests"] == 0 and report["real_model_quality_measured"] is False
    assert report["rules"]["v2"]["status"] == "proposed"
    assert report["review"]["independent_human_review"] == "PENDING"


def test_both_new_controls_exist_and_the_v2_set_is_unchanged():
    assert [control.name for control in CONTROLS_V3[: len(CONTROLS)]] == [control.name for control in CONTROLS]
    assert [control.name for control in CONTROLS_V3[len(CONTROLS) :]] == ["invalid_tool_arguments", "operator_budget_stop"]


def test_invalid_arguments_are_counted_as_events_and_fail_quality(report):
    row = report["negative_controls"]["controls"]["invalid_tool_arguments"]
    assert row["invalid_tool_argument_calls"] > 0
    assert row["d7"]["replace_criterion_4_with_d"]["latency_set_aside"] == "FAIL"


def test_a_spend_cap_is_an_incomplete_run_never_a_model_failure(report):
    row = report["negative_controls"]["controls"]["operator_budget_stop"]
    assert row["terminations"]["operator_budget_stop"] == row["questions_reaching_a_model"]
    for option in row["d7"].values():
        assert option == {"as_written": "INCOMPLETE_RUN", "latency_set_aside": "INCOMPLETE_RUN"}


def test_d7_option_3_fails_every_negative_control_and_the_baseline(report):
    controls = report["negative_controls"]["controls"]
    for name, row in controls.items():
        verdict = row["d7"]["replace_criterion_4_with_d"]["latency_set_aside"]
        if row["kind"] == "correct_control":
            assert verdict == "PASS", name
        elif name != "operator_budget_stop":
            assert verdict == "FAIL", name


def test_d7_option_1_can_never_pass_and_option_2_lets_route_following_through(report):
    """Option 1: criterion 4 has no opportunities, so even the oracle is NOT_EVALUABLE.
    Option 2: nothing then catches answering a question that has no answer."""
    controls = report["negative_controls"]["controls"]
    assert controls["label_oracle"]["d7"]["keep_criterion_4"]["latency_set_aside"] == "NOT_EVALUABLE"
    assert controls["label_oracle"]["d7"]["remove_criterion_4"]["latency_set_aside"] == "PASS"
    assert controls["route_following"]["d7"]["remove_criterion_4"]["latency_set_aside"] == "PASS"
    assert controls["select_every_claim"]["d7"]["remove_criterion_4"]["latency_set_aside"] == "PASS"


def test_always_abstain_fails_under_every_option(report):
    for option in report["negative_controls"]["controls"]["always_abstain"]["d7"].values():
        assert option["as_written"] == option["latency_set_aside"] == "FAIL"


def test_coverage_targets_are_reported_with_their_measured_values(report):
    targets = report["coverage"]["targets"]
    assert {name: item["met"] for name, item in targets.items()} == dict.fromkeys(targets, True)
    assert targets["answer_cases_reaching_a_model"]["measured"] >= 30


# -- CLI behaviour on a dataset that is not frozen ---------------------------------------------


def test_a_reviewer_can_show_a_v3_fixture(capsys):
    assert benchmark_cli.main(["--dataset", str(V3_DIR), "--show-fixture", "malicious_artifact"]) == 0
    assert json.loads(capsys.readouterr().out)["fixture"] == "malicious_artifact"


def test_the_review_sheet_is_blind(capsys, v3):
    assert benchmark_cli.main(["--dataset", str(V3_DIR), "--review-sheet", "heldout"]) == 0
    sheet = json.loads(capsys.readouterr().out)
    assert [row["case_id"] for row in sheet["cases"]] == [case.case_id for case in v3.heldout]
    assert {row["reviewer_decision"] for row in sheet["cases"]} == {""}
    text = json.dumps(sheet)
    for leaked in ("route_following", "task_success", "reach_a_model", "attempted", "verdict"):
        assert leaked not in text
    assert set(sheet["fixture_claims"]) == {case.fixture for case in v3.heldout}


def test_the_review_sheet_shows_every_expected_label_field(v3):
    """A reviewer cannot agree with a label field they were never shown."""
    renamed = {"outcome": "expected_outcome", "terminations": "acceptable_terminations"}
    for split in Split:
        sheet = benchmark_cli.review_sheet(v3, split)
        for row, case in zip(sheet["cases"], v3.split(split), strict=True):
            assert {renamed.get(name, name) for name in Expected.model_fields} <= set(row), case.case_id
            assert row["output_nature"] == case.expected.output_nature.value
            assert row["forbidden_text"] == list(case.expected.forbidden_text)


def test_no_paid_or_replay_run_starts_on_a_dataset_without_its_own_rule(capsys, tmp_path):
    approval = tmp_path / "approval.json"
    approval.write_text("{}", encoding="utf-8")
    args = ["--dataset", str(V3_DIR), "--allow-paid-calls", "--stage", "protocol_probe", "--approval", str(approval)]
    assert benchmark_cli.main(args) == 2
    assert "refused: dataset_not_frozen" in capsys.readouterr().out


def test_the_review_sheet_never_overwrites_a_file(tmp_path, capsys):
    existing = tmp_path / "sheet.json"
    existing.write_text("filled in", encoding="utf-8")
    args = ["--dataset", str(V3_DIR), "--review-sheet", "heldout", "--output", str(existing)]
    assert benchmark_cli.main(args) == 2
    assert "refused: output_exists" in capsys.readouterr().out
    assert existing.read_text(encoding="utf-8") == "filled in"


def test_the_reliability_cli_refuses_v3_without_rules_and_rules_on_v2(capsys):
    assert reliability_cli.main(["--dataset", str(V3_DIR)]) == 2
    assert "refused: rules_required" in capsys.readouterr().out
    assert reliability_cli.main(["--dataset", str(V2_DIR), "--rules", str(V3_DIR)]) == 2
    assert "refused: rules_apply_to_v3_only" in capsys.readouterr().out


def test_the_reliability_cli_reads_rules_from_the_named_directory(tmp_path, capsys):
    output = tmp_path / "report.json"
    code = reliability_cli.main(["--dataset", str(V3_DIR), "--rules", str(V2_DIR), "--output", str(output)])
    assert code == 0
    assert output.read_text(encoding="utf-8") == (V3_DIR / reliability_cli.REPORT_FILE_V3).read_text(encoding="utf-8")
