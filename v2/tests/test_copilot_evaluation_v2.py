"""Dataset versioning: v1 stays reproducible, v2 stays honest about what it is.

v1 is the benchmark the accepted decision rule was written against. It may not
change, and its numbers must keep coming out of the real service, so a fix to
production routing can never move a published v1 score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from footballai_v2.cli import copilot_eval
from footballai_v2.copilot.evaluation import routers
from footballai_v2.copilot.evaluation.dataset import Outcome, Suite, load_dataset
from footballai_v2.copilot.evaluation.harness import run_case, run_cases
from footballai_v2.copilot.evaluation.metrics import compare_dataset_versions, contract_metrics, selection_metrics
from footballai_v2.copilot.evaluation.routers import ROUTING_V1, route_question_v1, routing_for
from footballai_v2.copilot.evaluation.selectors import LexicalOverlapProvider, RecordingProvider, ScriptedProvider
from footballai_v2.copilot.grounding import ROUTING_VERSION, INSTRUCTION_VERSION, route_question
from footballai_v2.copilot.models import ModelMode
from footballai_v2.copilot.provider import DeterministicFakeProvider, ProviderResult

from test_copilot_explanation import real_v2_payload, synthetic_v1_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_DIR = REPO_ROOT / "evaluation" / "copilot" / "v1"
V2_DIR = REPO_ROOT / "evaluation" / "copilot" / "v2"
# Recorded when M2-A began. A changed value means v1 was edited, which is never allowed.
V1_FILES = {
    "cases.json": "f19a6e135ccdc55e56f387ee93e93a4debb5ff0147f782e91cf08ac98f3201c3",
    "fixtures.json": "380f6d2d256e89cbca50e7dae492976343bdfa1de516531dfadd86b15ead78c5",
    "baseline-report.json": "de543979fe2859559c0117ab561c0795e534b21d9b1834a98f6ba2b86d12c770",
}
V1_FALSE_ANSWERS = ["dist-known-words", "model-version", "track-short-no-claim"]


@pytest.fixture(scope="module")
def v1():
    return load_dataset(V1_DIR)


@pytest.fixture(scope="module")
def v2():
    return load_dataset(V2_DIR)


def fake(dataset, cases=None):
    return run_cases(cases or dataset.suite(Suite.SELECTION), dataset, lambda _case: DeterministicFakeProvider())


# -- v1 is immutable -------------------------------------------------------------


@pytest.mark.parametrize(("name", "digest"), V1_FILES.items())
def test_v1_files_are_byte_identical(name, digest):
    assert hashlib.sha256((V1_DIR / name).read_bytes()).hexdigest() == digest


def test_v1_holds_no_new_file():
    assert {path.name for path in V1_DIR.iterdir()} == set(V1_FILES)


def test_v1_is_replayed_through_its_frozen_routing():
    version, router = routing_for("copilot-eval/v1")
    assert version == ROUTING_V1 != ROUTING_VERSION
    assert router is route_question_v1


def test_the_frozen_v1_vocabulary_is_the_one_v1_was_labelled_against():
    # Computed from main at 30984c3 before routing changed. The v1 cases do not use
    # every word, so without this a new word could drift in unnoticed.
    tables = {
        "function": sorted(routers._FUNCTION_WORDS),
        "general": sorted(routers._GENERAL_WORDS),
        "topics": routers._TOPIC_WORDS,
    }
    digest = hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()
    assert digest == "ecfc48dd458768afcc9c8db9604998d1fb0a1a130f4780ac44a5606739190048"


def test_the_known_v1_false_answers_still_reproduce(v1):
    # Under current routing these are fixed. Under v1 they must stay wrong, or
    # the accepted decision rule's "corrects 2 of 3" would no longer mean anything.
    metrics = selection_metrics(fake(v1))
    assert metrics["false_answers"] == V1_FALSE_ANSWERS
    assert metrics["claim_selection_accuracy"]["correct"] == 19
    assert metrics["abstention_accuracy"]["correct"] == 39


def test_every_published_v1_metric_except_latency_matches_the_committed_baseline(v1):
    report = json.loads((V1_DIR / "baseline-report.json").read_text())
    published = report["baselines"]["deterministic_fake"]
    measured = selection_metrics(fake(v1))
    for metric in published:
        if metric != "latency_in_process":
            assert measured[metric] == published[metric], metric


def test_new_claims_never_reach_a_v1_prompt(v1):
    for case in v1.suite(Suite.SELECTION):
        recorder = RecordingProvider(DeterministicFakeProvider())
        run_case(case, v1, recorder)
        offered = {claim.claim_id for claim in recorder.request.catalog} if recorder.request else set()
        assert not offered & {"model_details", "diag_short_tracklets", "diag_reliability_scope", "diag_legacy_tracking"}


def test_the_accepted_instruction_version_is_unchanged():
    assert INSTRUCTION_VERSION == "copilot.grounding-instructions/v3"


def test_a_dataset_with_no_bound_routing_is_refused(tmp_path):
    for name in ("cases.json", "fixtures.json"):
        (tmp_path / name).write_text((V1_DIR / name).read_text())
    data = json.loads((tmp_path / "cases.json").read_text())
    data["dataset_version"] = "copilot-eval/v99"
    (tmp_path / "cases.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no routing"):
        load_dataset(tmp_path)


# -- v2 consistency --------------------------------------------------------------


def test_v2_payloads_reuse_the_approved_fixtures_verbatim(v1, v2):
    assert v2.fixtures.payloads["real_v2"] == real_v2_payload()
    assert v2.fixtures.payloads["synthetic_v1"] == synthetic_v1_payload()
    assert v2.fixtures.run_id == v1.fixtures.run_id


def test_every_v1_fixture_survives_unchanged_in_v2(v1, v2):
    for name, fixture in v1.fixtures.fixtures.items():
        assert v2.fixtures.fixtures[name] == fixture


def test_v2_keeps_every_v1_case_id(v1, v2):
    assert {case.case_id for case in v1.cases} <= {case.case_id for case in v2.cases}


def test_every_relabelled_case_says_why(v1, v2):
    after = {case.case_id: case for case in v2.cases}
    for case in v1.cases:
        if after[case.case_id].expected != case.expected:
            assert after[case.case_id].notes.startswith("Relabelled from v1:"), case.case_id


def test_v2_covers_the_required_regressions(v2):
    categories = {case.category for case in v2.cases}
    assert categories >= {
        "model_reference",
        "short_tracklets",
        "tracking_reliability",
        "synthetic_demo_fast",
        "unverified_diagnostics",
        "possession_pass",
        "player_distance",
        "relevance",
        "invalid_question",
        "invalid_model_output",
    }
    ids = {case.case_id for case in v2.cases}
    assert {"label-only-model", "label-only-short", "output-cross-run-id", "cross-run-legacy-claim"} <= ids
    assert {"empty-string", "invalid-punctuation", "off-topic-claim"} <= ids


def test_every_v2_fixture_is_used_and_every_labelled_claim_exists(v2):
    assert {case.fixture for case in v2.cases} == set(v2.fixtures.fixtures)
    for case in v2.suite(Suite.SELECTION):
        result = run_case(case, v2, DeterministicFakeProvider())
        if case.expected.outcome is Outcome.ANSWER:
            assert result.provider_invoked, case.case_id
            assert set(case.expected.required_claims) <= set(result.offered_claims), case.case_id


def test_the_scorer_can_fail_on_v2(v2):
    class Empty:
        mode = ModelMode.REAL_MODEL

        def generate(self, request):
            return ProviderResult((), self.mode)

    metrics = selection_metrics(run_cases(v2.suite(Suite.SELECTION), v2, lambda _case: Empty()))
    assert metrics["claim_selection_accuracy"]["rate"] == 0.0
    assert metrics["false_abstentions"]


def test_lexical_overlap_is_not_perfect_on_v2(v2):
    # The dataset can tell selectors apart, so its score is not vacuous.
    results = run_cases(v2.suite(Suite.SELECTION), v2, lambda _case: LexicalOverlapProvider())
    assert selection_metrics(results)["claim_selection_accuracy"]["rate"] < 1.0


def test_every_scripted_v2_reply_fails_closed(v2):
    results = run_cases(v2.suite(Suite.CONTRACT), v2, ScriptedProvider)
    assert contract_metrics(results)["failed_cases"] == []
    assert contract_metrics(results)["fail_closed"]["rate"] == 1.0


def test_no_v2_answer_is_fabricated_or_mislabelled(v2):
    metrics = selection_metrics(fake(v2))
    assert metrics["fabricated_answers"] == 0
    assert metrics["output_nature_accuracy"]["rate"] == 1.0
    assert metrics["forbidden_text_absent"]["rate"] == 1.0
    assert metrics["citation_validity"]["rate"] == 1.0


def test_v2_baseline_matches_the_committed_report(v2):
    report = json.loads((V2_DIR / "baseline-report.json").read_text())
    assert report["live_model_requests"] == 0
    assert report["routing_version"] == ROUTING_VERSION
    assert report["dataset"]["version"] == "copilot-eval/v2"
    assert report["dataset"]["cases_sha256"] == v2.cases_sha256
    assert report["dataset"]["fixtures_sha256"] == v2.fixtures_sha256
    measured = selection_metrics(fake(v2))
    published = report["baselines"]["deterministic_fake"]
    for metric in published:
        if metric != "latency_in_process":
            assert measured[metric] == published[metric], metric


def test_production_routing_serves_v2(v2):
    assert routing_for(v2.version)[1] is route_question


# -- comparing versions ----------------------------------------------------------


def test_a_relabelled_case_is_not_counted_as_an_improvement(v1, v2):
    comparison = compare_dataset_versions(v1, fake(v1), v2, fake(v2))
    relabelled = set(comparison["relabelled_cases"])
    assert set(V1_FALSE_ANSWERS) <= relabelled
    assert not relabelled & set(comparison["same_label_comparison"]["improved"])
    assert comparison["same_label_cases"] == comparison["same_label_comparison"]["paired_cases"]


def test_a_changed_fixture_payload_removes_a_case_from_the_pairing(v1, v2):
    changed = {**v2.fixtures.payloads, "real_v2": {"schema": "changed"}}
    tampered = type(v2)(
        v2.version, v2.cases, v2.fixtures.model_copy(update={"payloads": changed}), v2.cases_sha256, v2.fixtures_sha256
    )
    comparison = compare_dataset_versions(v1, fake(v1), tampered, fake(v2))
    assert "real-status" in comparison["relabelled_cases"]


def test_no_same_label_case_regressed_from_v1(v1, v2):
    assert compare_dataset_versions(v1, fake(v1), v2, fake(v2))["same_label_comparison"]["regressed"] == []


def test_the_committed_comparison_matches_the_code(capsys):
    assert copilot_eval.main(["--dataset", str(V2_DIR), "--compare-with", str(V1_DIR)]) == copilot_eval.EXIT_OK
    assert json.loads(capsys.readouterr().out) == json.loads((V2_DIR / "v1-comparison.json").read_text())


@pytest.mark.parametrize("extra", [["--preflight"], ["--sample", "5"], ["--model", "claude-haiku-5-5"]])
def test_comparison_mode_refuses_what_it_would_ignore(extra):
    args = ["--dataset", str(V2_DIR), "--compare-with", str(V1_DIR), *extra]
    assert copilot_eval.main(args) == copilot_eval.EXIT_REFUSED
