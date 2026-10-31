"""The evaluation dataset, scorer, budget guard, and runner.

No test here reaches a model. Live paths run against a local stand-in for the SDK,
and the guards are asserted on what the stand-in actually received.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from footballai_v2.cli import copilot_eval
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderUnavailableError
from footballai_v2.copilot.evaluation import harness
from footballai_v2.copilot.evaluation.cost import (
    MODEL_PRICES,
    estimate_input_tokens,
    estimate_output_tokens,
    request_cost_usd,
    worst_case_cost_usd,
)
from footballai_v2.copilot.evaluation.dataset import (
    EvaluationCase,
    Expected,
    Outcome,
    Suite,
    load_dataset,
)
from footballai_v2.copilot.evaluation.harness import ABSTAINED, ANSWERED, REJECTED, run_case, run_cases
from footballai_v2.copilot.evaluation.live import (
    BudgetExhausted,
    BudgetGuard,
    estimate_plan,
    plan_live_run,
    real_settings,
    run_live,
)
from footballai_v2.copilot.evaluation.metrics import compare, contract_metrics, headroom, selection_metrics
from footballai_v2.copilot.evaluation.selectors import LexicalOverlapProvider, ScriptedProvider
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID
from footballai_v2.copilot.models import ModelMode, OutputNature, ProviderTelemetry, TokenUsage
from footballai_v2.copilot.provider import DeterministicFakeProvider, GenerationRequest, ProviderResult

from test_copilot_explanation import RUN_A, real_v2_payload, synthetic_v1_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "evaluation" / "copilot" / "v1"
BASELINE_REPORT = DATASET_DIR / "baseline-report.json"
CANARY = "canary-value-not-real"
HAIKU = MODEL_PRICES["claude-haiku-5-5"]
SELECTION_ACCURACY = "claim_selection_accuracy"
ABSTENTION_ACCURACY = "abstention_accuracy"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_DIR)


def selection_cases(dataset) -> tuple[EvaluationCase, ...]:
    return dataset.suite(Suite.SELECTION)


class OracleProvider:
    """Selects exactly what a case's labels ask for: a perfect selector."""

    mode = ModelMode.REAL_MODEL

    def __init__(self, case: EvaluationCase) -> None:
        self._case = case

    def generate(self, request: GenerationRequest) -> ProviderResult:
        offered = {claim.claim_id for claim in request.catalog}
        expected = self._case.expected
        wanted = expected.required_claims if expected.outcome is Outcome.ANSWER else ()
        return ProviderResult(tuple(item for item in wanted if item in offered), self.mode)


class EmptyProvider:
    mode = ModelMode.REAL_MODEL

    def generate(self, request: GenerationRequest) -> ProviderResult:
        return ProviderResult((), self.mode)


# -- the dataset -----------------------------------------------------------------


def test_payloads_are_the_approved_test_fixtures_verbatim(dataset):
    # The evaluation must not invent measurements: these are the existing fixtures.
    assert dataset.fixtures.payloads["real_v2"] == real_v2_payload()
    assert dataset.fixtures.payloads["synthetic_v1"] == synthetic_v1_payload()
    assert dataset.fixtures.run_id == RUN_A


def test_dataset_size_and_required_coverage(dataset):
    assert 30 <= len(selection_cases(dataset)) + len(dataset.suite(Suite.CONTRACT)) <= 55
    categories = {case.category for case in dataset.cases}
    assert categories >= {
        "verified_real",
        "synthetic_demo_fast",
        "missing_or_corrupt",
        "tracking_fragmentation",
        "model_provenance",
        "unsupported_statistic",
        "player_distance",
        "possession_pass",
        "ambiguous",
        "malicious_artifact",
        "invalid_model_output",
    }
    assert {case.fixture for case in dataset.cases} == set(dataset.fixtures.fixtures)


def test_dataset_is_versioned_and_hashed(dataset):
    assert dataset.version == "copilot-eval/v1"
    assert len(dataset.cases_sha256) == len(dataset.fixtures_sha256) == 64


def test_every_labelled_claim_exists_for_its_fixture(dataset):
    # A typo in a label would silently make a case unwinnable.
    for case in selection_cases(dataset):
        result = run_case(case, dataset, DeterministicFakeProvider())
        if case.expected.outcome is not Outcome.ABSTAIN and result.provider_invoked:
            unknown = set(case.expected.required_claims) - set(result.offered_claims)
            assert not unknown, f"{case.case_id} requires claims never offered: {unknown}"
            unknown = set(case.expected.permitted_claims) - set(result.offered_claims)
            assert not unknown, f"{case.case_id} permits claims never offered: {unknown}"


def test_expected_output_nature_comes_from_evidence_not_labels(dataset):
    results = run_cases(selection_cases(dataset), dataset, lambda _case: DeterministicFakeProvider())
    assert all(result.output_nature_correct for result in results)


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "answer", "output_nature": "verified_real"},
        {"outcome": "answer", "output_nature": "verified_real", "required_claims": ["x"]},
        {"outcome": "abstain", "output_nature": "verified_real"},
        {"outcome": "abstain", "output_nature": "verified_real", "reason": "r", "permitted_claims": ["x"]},
    ],
)
def test_inconsistent_labels_are_rejected(fields):
    with pytest.raises(ValueError):
        Expected.model_validate(fields)


def test_a_scripted_reply_cannot_enter_the_selection_suite():
    expected = {"outcome": "abstain", "output_nature": "verified_real", "reason": "r"}
    with pytest.raises(ValueError):
        EvaluationCase.model_validate(
            {"case_id": "a", "suite": "selection", "category": "c", "fixture": "f", "question": None,
             "expected": expected, "model_output": "{}"}
        )


def test_loading_rejects_a_dangling_fixture_reference(tmp_path):
    for name in ("cases.json", "fixtures.json"):
        (tmp_path / name).write_text((DATASET_DIR / name).read_text())
    data = json.loads((tmp_path / "cases.json").read_text())
    data["cases"][0]["fixture"] = "does_not_exist"
    (tmp_path / "cases.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown fixture"):
        load_dataset(tmp_path)


def test_loading_rejects_duplicate_case_ids(tmp_path):
    for name in ("cases.json", "fixtures.json"):
        (tmp_path / name).write_text((DATASET_DIR / name).read_text())
    data = json.loads((tmp_path / "cases.json").read_text())
    data["cases"].append(data["cases"][0])
    (tmp_path / "cases.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_dataset(tmp_path)


# -- the scorer can fail ---------------------------------------------------------


def test_a_perfect_selector_scores_perfectly(dataset):
    results = run_cases(selection_cases(dataset), dataset, OracleProvider)
    metrics = selection_metrics(results)
    wrong = [r.case_id for r in results if not r.outcome_correct or r.selection_correct is False]
    assert wrong == []
    assert metrics[SELECTION_ACCURACY]["rate"] == 1.0
    assert metrics[ABSTENTION_ACCURACY]["rate"] == 1.0
    assert metrics["unsupported_claim_rate"]["unpermitted"] == 0


def test_a_selector_that_always_abstains_scores_badly(dataset):
    results = run_cases(selection_cases(dataset), dataset, lambda _case: EmptyProvider())
    metrics = selection_metrics(results)
    assert metrics[SELECTION_ACCURACY]["rate"] == 0.0
    assert metrics[ABSTENTION_ACCURACY]["rate"] < 0.5
    assert metrics["false_abstentions"]


def test_over_selection_is_counted_as_unsupported(dataset):
    case = next(item for item in dataset.cases if item.case_id == "real-video-duration")
    result = run_case(case, dataset, DeterministicFakeProvider())
    assert result.outcome == ANSWERED
    assert result.selection_correct is False
    assert set(result.unpermitted_claims) == {"provenance", "model_references"}


def test_answering_what_should_be_abstained_is_a_false_answer(dataset):
    case = next(item for item in dataset.cases if item.case_id == "track-short-no-claim")
    result = run_case(case, dataset, DeterministicFakeProvider())
    assert result.outcome == ANSWERED
    assert result.outcome_correct is False


def test_a_model_abstaining_correctly_scores_as_correct(dataset):
    case = next(item for item in dataset.cases if item.case_id == "track-short-no-claim")
    result = run_case(case, dataset, EmptyProvider())
    assert result.outcome == ABSTAINED
    assert result.reason == "no_relevant_claim"
    assert result.outcome_correct is True


def test_a_provider_failure_is_never_scored_as_a_correct_abstention(dataset):
    class Failing:
        mode = ModelMode.REAL_MODEL

        def generate(self, request):
            raise ProviderUnavailableError("down", kind=ProviderFailureKind.SERVER_ERROR)

    case = next(item for item in dataset.cases if item.case_id == "track-short-no-claim")
    result = run_case(case, dataset, Failing())
    assert result.outcome_correct is False
    assert result.failure_kind == "server_error"


def test_text_outside_the_catalog_is_detected_as_fabricated(dataset, monkeypatch):
    # Proves the structural check is live: tamper with the service's answer.
    original = harness.ExplanationService.explain

    def tampered(self, run_id, question=None):
        response = original(self, run_id, question)
        return response.model_copy(update={"answer": response.answer + " Player 7 ran 10.5 km."})

    monkeypatch.setattr(harness.ExplanationService, "explain", tampered)
    case = next(item for item in dataset.cases if item.case_id == "real-status")
    results = (run_case(case, dataset, DeterministicFakeProvider()),)
    assert selection_metrics(results)["fabricated_answers"] == 1


def test_a_wrong_label_in_the_response_is_detected(dataset, monkeypatch):
    original = harness.ExplanationService.explain

    def tampered(self, run_id, question=None):
        response = original(self, run_id, question)
        return response.model_copy(update={"output_nature": OutputNature.VERIFIED_REAL})

    monkeypatch.setattr(harness.ExplanationService, "explain", tampered)
    case = next(item for item in dataset.cases if item.case_id == "synth-nature")
    assert run_case(case, dataset, DeterministicFakeProvider()).output_nature_correct is False


def test_a_citation_that_does_not_match_the_rendered_claims_is_detected(dataset, monkeypatch):
    original = harness.ExplanationService.explain

    def tampered(self, run_id, question=None):
        response = original(self, run_id, question)
        return response.model_copy(update={"citations": response.citations[:1]})

    monkeypatch.setattr(harness.ExplanationService, "explain", tampered)
    case = next(item for item in dataset.cases if item.case_id == "real-status")
    assert run_case(case, dataset, DeterministicFakeProvider()).citations_valid is False


def test_injected_artifact_text_reaches_neither_reader_nor_model(dataset):
    case = next(item for item in dataset.cases if item.case_id == "mal-explain")
    assert case.expected.forbidden_text
    assert run_case(case, dataset, DeterministicFakeProvider()).forbidden_text_absent is True


def test_forbidden_text_is_also_checked_in_the_prompt(dataset):
    case = next(item for item in dataset.cases if item.case_id == "real-status")
    leaky = case.model_copy(
        update={"expected": case.expected.model_copy(update={"forbidden_text": ("output_nature",)})}
    )
    assert run_case(leaky, dataset, DeterministicFakeProvider()).forbidden_text_absent is False


# -- contract suite and baseline -------------------------------------------------


def test_every_invalid_model_output_fails_closed(dataset):
    results = run_cases(dataset.suite(Suite.CONTRACT), dataset, ScriptedProvider)
    assert contract_metrics(results)["failed_cases"] == []
    assert contract_metrics(results)["fail_closed"]["rate"] == 1.0
    assert {r.outcome for r in results} <= {ABSTAINED, REJECTED}
    assert {r.reason for r in results} == {"provider_unavailable", "forged_citation", "no_relevant_claim"}


def test_contract_suite_detects_a_parser_that_accepts_bad_output(dataset, monkeypatch):
    monkeypatch.setattr(
        "footballai_v2.copilot.evaluation.selectors.parse_selection", lambda _text: ("run_status",)
    )
    results = run_cases(dataset.suite(Suite.CONTRACT), dataset, ScriptedProvider)
    assert contract_metrics(results)["fail_closed"]["rate"] < 0.5


def test_deterministic_baseline_matches_the_committed_report(dataset):
    # The published numbers are reproducible from the dataset, or this fails.
    report = json.loads(BASELINE_REPORT.read_text())
    assert report["live_model_requests"] == 0
    assert report["dataset"]["cases_sha256"] == dataset.cases_sha256
    assert report["dataset"]["fixtures_sha256"] == dataset.fixtures_sha256
    fake = selection_metrics(run_cases(selection_cases(dataset), dataset, lambda _case: DeterministicFakeProvider()))
    lexical = selection_metrics(run_cases(selection_cases(dataset), dataset, lambda _case: LexicalOverlapProvider()))
    for name, measured in (("deterministic_fake", fake), ("lexical_overlap", lexical)):
        published = report["baselines"][name]
        for metric in (SELECTION_ACCURACY, ABSTENTION_ACCURACY, "unsupported_claim_rate", "citation_validity"):
            assert published[metric] == measured[metric], (name, metric)


def test_headroom_counts_what_a_selector_can_reach(dataset):
    results = run_cases(selection_cases(dataset), dataset, lambda _case: DeterministicFakeProvider())
    room = headroom(results)
    assert room["reach_the_selector"] + room["stopped_before_the_selector"] == len(results)
    assert set(room["coverage_gaps_blocked_before_selector"]) == {"amb-median"}
    assert room["not_improvable_wrong_cases"] == []
    assert room["perfect_selector_selection_accuracy"]["rate"] == 1.0


def test_compare_reports_paired_changes(dataset):
    cases = selection_cases(dataset)
    fake = run_cases(cases, dataset, lambda _case: DeterministicFakeProvider())
    oracle = run_cases(cases, dataset, OracleProvider)
    paired = compare(fake, oracle)
    assert paired["regressed"] == []
    assert "real-video-duration" in paired["improved"]
    assert paired["unchanged"] + len(paired["improved"]) == len(cases)
    backwards = compare(oracle, fake)
    assert backwards["improved"] == []
    assert backwards["regressed"] == paired["improved"]


# -- cost ------------------------------------------------------------------------


def test_cost_uses_documented_rates_per_token_class():
    usage = TokenUsage(10_000, 10_000, 10_000, 10_000)
    assert request_cost_usd(HAIKU, usage) == pytest.approx((0.10 + 0.50 + 0.125 + 0.01) / 100)
    assert request_cost_usd(HAIKU, TokenUsage(800, 100)) == pytest.approx(0.00013)


def test_long_prompts_use_the_documented_long_rate_on_haiku_only():
    assert request_cost_usd(HAIKU, TokenUsage(200_000, 0)) == pytest.approx(0.10)
    # Cached tokens count toward the threshold, and the whole request reprices.
    assert request_cost_usd(HAIKU, TokenUsage(1, 0, 60_000, 60_000)) > 5 * 0.00003
    assert request_cost_usd(MODEL_PRICES["claude-opus-4-8"], TokenUsage(200_000, 0)) == pytest.approx(1.0)


def test_an_absurd_usage_report_cannot_abort_a_paid_run(fake_api, dataset, monkeypatch):
    api = fake_api()
    original = api.create

    def huge(**kwargs):
        reply = original(**kwargs)
        reply.usage.input_tokens = 150_000
        return reply

    monkeypatch.setattr(api, "create", huge)
    monkeypatch.setitem(sys.modules, "anthropic", api.module())
    settings = real_settings("claude-haiku-5-5")
    run = run_live(plan_live_run(dataset, settings, sample=2), dataset, settings, BudgetGuard(2, 1.0, HAIKU))
    assert len(run.results) == 2
    assert run.stop_reason is None


def test_the_worst_case_bounds_the_estimate():
    chars, selection = 3_000, 100
    expected = request_cost_usd(HAIKU, TokenUsage(estimate_input_tokens(chars), estimate_output_tokens(selection)))
    assert worst_case_cost_usd(HAIKU, chars, 1024) > expected


def test_plan_estimate_is_reported_with_its_method(dataset):
    settings = real_settings("claude-haiku-5-5")
    plan = plan_live_run(dataset, settings, sample=5)
    estimate = estimate_plan(plan, HAIKU, settings.max_output_tokens)
    assert estimate["requests"] == 5
    assert "heuristic" in estimate["method"]
    assert estimate["worst_case_total_cost_usd"] > estimate["estimated_total_cost_usd"] > 0


def test_the_recommended_model_is_priced_and_the_old_default_is_far_dearer():
    assert MODEL_PRICES["claude-opus-4-8"].input == 50 * HAIKU.input
    assert MODEL_PRICES["claude-opus-4-8"].output == 50 * HAIKU.output


# -- planning --------------------------------------------------------------------


def test_plan_holds_only_cases_that_reach_a_provider(dataset):
    plan = plan_live_run(dataset, real_settings("claude-haiku-5-5"))
    planned = {item.case.case_id for item in plan}
    contract = {case.case_id for case in dataset.suite(Suite.CONTRACT)}
    assert not planned & contract
    assert "stat-goals" not in planned  # the router stops it
    assert "synth-detections" not in planned  # no evidence for it
    assert len(planned) == headroom(
        run_cases(selection_cases(dataset), dataset, lambda _case: DeterministicFakeProvider())
    )["reach_the_selector"]


def test_sample_is_deterministic_and_stratified(dataset):
    settings = real_settings("claude-haiku-5-5")
    first = [item.case.case_id for item in plan_live_run(dataset, settings, sample=8)]
    second = [item.case.case_id for item in plan_live_run(dataset, settings, sample=8)]
    assert first == second
    assert len(set(first)) == 8
    categories = {item.case.category for item in plan_live_run(dataset, settings, sample=8)}
    assert len(categories) >= 6


def test_planned_prompts_hold_no_private_data(dataset):
    for item in plan_live_run(dataset, real_settings("claude-haiku-5-5")):
        prompt = item.arguments["messages"][0]["content"].lower()
        for private in ("fixture://", "file://", "relative_path", "sig=", "accountkey"):
            assert private not in prompt


# -- the budget ------------------------------------------------------------------


def test_request_limit_is_enforced_before_sending():
    guard = BudgetGuard(1, 1.0, HAIKU)
    guard.admit(0.001)
    with pytest.raises(BudgetExhausted) as caught:
        guard.admit(0.001)
    assert caught.value.limit == "max_requests"
    assert guard.requests_sent == 1


def test_cost_limit_reserves_the_worst_case_before_sending():
    guard = BudgetGuard(10, 0.001, HAIKU)
    guard.spent_usd = 0.0009
    with pytest.raises(BudgetExhausted) as caught:
        guard.admit(0.0002)
    assert caught.value.limit == "max_cost_usd"
    assert guard.requests_sent == 0


def test_reported_usage_is_charged_at_list_price():
    guard = BudgetGuard(10, 1.0, HAIKU)
    telemetry = ProviderTelemetry("claude-haiku-5-5", 0.1, usage=TokenUsage(1000, 100))
    guard.charge(telemetry, worst_case_usd=0.5)
    assert guard.spent_usd == pytest.approx(0.00015)
    assert guard.requests_at_worst_case == 0


@pytest.mark.parametrize("telemetry", [None, ProviderTelemetry("claude-haiku-5-5", 0.1)])
def test_unknown_usage_is_charged_at_the_worst_case(telemetry):
    guard = BudgetGuard(10, 1.0, HAIKU)
    guard.charge(telemetry, worst_case_usd=0.5)
    assert guard.spent_usd == 0.5
    assert guard.requests_at_worst_case == 1


# -- a live run against a local stand-in for the SDK -----------------------------


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream said {CANARY}")
        self.status_code = status_code


class FakeAPI:
    """Stands in for Anthropic: counts every billed call it receives."""

    def __init__(self, outcomes=None) -> None:
        self.outcomes = outcomes
        self.created: list[dict] = []
        self.counted: list[dict] = []
        self.retrieved: list[str] = []
        self.client_kwargs: list[dict] = []

    def module(self) -> types.ModuleType:
        api = self
        module = types.ModuleType("anthropic")

        class Anthropic:
            def __init__(self, **kwargs) -> None:
                api.client_kwargs.append(kwargs)
                self.messages = types.SimpleNamespace(create=api.create, count_tokens=api.count_tokens)
                self.models = types.SimpleNamespace(retrieve=api.retrieve)

        module.Anthropic = Anthropic
        return module

    def create(self, **kwargs):
        self.created.append(kwargs)
        if self.outcomes is not None:
            outcome = self.outcomes[min(len(self.created) - 1, len(self.outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
        ids = json.loads(kwargs["messages"][0]["content"])["claims"]
        selected = [claim["id"] for claim in ids][:2]
        block = types.SimpleNamespace(type="text", text=json.dumps({"claim_ids": selected}))
        reply = types.SimpleNamespace(
            content=[block],
            stop_reason="end_turn",
            usage=types.SimpleNamespace(
                input_tokens=900, output_tokens=60, cache_creation_input_tokens=0, cache_read_input_tokens=0
            ),
        )
        reply._request_id = f"req_{len(self.created)}"
        return reply

    def count_tokens(self, **kwargs):
        self.counted.append(kwargs)
        return types.SimpleNamespace(input_tokens=900)

    def retrieve(self, model_id):
        self.retrieved.append(model_id)
        supported = types.SimpleNamespace(supported=True)
        return types.SimpleNamespace(
            id=model_id,
            lifecycle="active",
            capabilities=types.SimpleNamespace(structured_outputs=supported, effort=supported),
        )


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def install(outcomes=None) -> FakeAPI:
        api = FakeAPI(outcomes)
        monkeypatch.setitem(sys.modules, "anthropic", api.module())
        monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
        return api

    return install


def paid_args(**overrides) -> list[str]:
    values = {"--model": "claude-haiku-5-5", "--max-requests": "3", "--max-cost-usd": "0.05"}
    values.update(overrides)
    pairs = [item for key, value in values.items() if value is not None for item in (key, value)]
    return ["--dataset", str(DATASET_DIR), "--allow-paid-calls", *pairs]


def test_paid_run_never_exceeds_the_request_limit(fake_api, capsys):
    api = fake_api()
    assert copilot_eval.main(paid_args()) == copilot_eval.EXIT_OK
    assert len(api.created) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["live_model_requests"] == 3
    assert report["stop_reason"] is None
    assert len(report["cases"]) == 3
    assert report["cases"][0]["request_id"] == "req_1"
    assert report["cases"][0]["usage"]["input_tokens"] == 900


def test_paid_run_stops_when_the_money_runs_out(fake_api, dataset):
    api = fake_api()
    settings = real_settings("claude-haiku-5-5")
    plan = plan_live_run(dataset, settings)
    one = worst_case_cost_usd(HAIKU, plan[0].prompt_chars, settings.max_output_tokens)
    run = run_live(plan, dataset, settings, BudgetGuard(100, one * 1.5, HAIKU))
    assert run.stop_reason == "budget_exhausted:max_cost_usd"
    assert 1 <= len(api.created) < len(plan)


@pytest.mark.parametrize("status", [401, 402, 403, 404, 400, 429])
def test_paid_run_stops_at_once_on_a_failure_that_will_repeat(fake_api, dataset, status):
    api = fake_api([StatusError(status)])
    settings = real_settings("claude-haiku-5-5")
    run = run_live(plan_live_run(dataset, settings), dataset, settings, BudgetGuard(50, 1.0, HAIKU))
    assert len(api.created) == 1
    assert run.stop_reason.startswith("fatal_failure:")


def test_paid_run_stops_after_repeated_transient_failures(fake_api, dataset):
    api = fake_api([StatusError(529)])
    settings = real_settings("claude-haiku-5-5")
    guard = BudgetGuard(50, 1.0, HAIKU)
    run = run_live(plan_live_run(dataset, settings), dataset, settings, guard)
    assert run.stop_reason == "consecutive_failures"
    assert len(api.created) == 3
    assert guard.requests_at_worst_case == 3  # no usage reported: assume the worst


def test_paid_run_keeps_failures_visible(fake_api, capsys):
    fake_api([StatusError(529), None, None])
    assert copilot_eval.main(paid_args()) == copilot_eval.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["model_metrics"]["provider_failures"] == 1
    assert report["cases"][0]["failure_kind"] == "server_error"


def test_failed_run_exits_nonzero_but_still_reports(fake_api, capsys):
    fake_api([StatusError(401)])
    assert copilot_eval.main(paid_args()) == copilot_eval.EXIT_RUN_INCOMPLETE
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "fatal_failure:authentication"


def test_paid_run_disables_sdk_retries(fake_api):
    api = fake_api()
    copilot_eval.main(paid_args())
    assert all(kwargs["max_retries"] == 0 for kwargs in api.client_kwargs)


def test_a_retrying_configuration_is_refused(dataset):
    settings = replace(real_settings("claude-haiku-5-5"), provider_retries=2)
    with pytest.raises(ValueError, match="retries"):
        run_live((), dataset, settings, BudgetGuard(1, 1.0, HAIKU))


def test_report_holds_no_key_prompt_or_question(fake_api, capsys, tmp_path):
    fake_api([StatusError(401)])
    output = tmp_path / "report.json"
    copilot_eval.main(paid_args(**{"--output": str(output)}))
    captured = capsys.readouterr()
    written = output.read_text()
    for text in (captured.out, captured.err, written):
        assert CANARY not in text
        assert "pipeline_profile" not in text  # a prompt field
        assert "What is the status" not in text
        assert "upstream said" not in text


def test_paid_run_compares_the_same_cases_against_the_baselines(fake_api, capsys):
    fake_api()
    copilot_eval.main(paid_args())
    report = json.loads(capsys.readouterr().out)
    assert report["paired_comparison"]["against_deterministic_fake"]["paired_cases"] == 3
    assert report["paired_baselines"]["lexical_overlap"]["cases"] == 3
    assert report["budget"]["list_price_estimate_usd"] == pytest.approx(3 * 0.00012)


def test_scripted_contract_cases_are_never_sent_to_a_model(fake_api, dataset):
    api = fake_api()
    settings = real_settings("claude-haiku-5-5")
    run_live(plan_live_run(dataset, settings), dataset, settings, BudgetGuard(100, 1.0, HAIKU))
    sent = " ".join(call["messages"][0]["content"] for call in api.created)
    assert "Player 7" not in sent


def test_preflight_mode_never_sends_a_billed_request(fake_api, capsys):
    api = fake_api()
    code = copilot_eval.main(["--dataset", str(DATASET_DIR), "--preflight", "--sample", "2"])
    assert code == copilot_eval.EXIT_OK
    assert api.created == []
    assert api.retrieved == ["claude-haiku-5-5"]
    assert len(api.counted) == 2
    assert json.loads(capsys.readouterr().out)["live_model_requests"] == 0


def test_a_failed_preflight_sends_no_paid_request(fake_api, monkeypatch, capsys):
    api = fake_api()

    def unavailable(_model_id):
        raise StatusError(404)

    monkeypatch.setattr(api, "retrieve", unavailable)
    monkeypatch.setitem(sys.modules, "anthropic", api.module())
    assert copilot_eval.main(paid_args()) == copilot_eval.EXIT_PREFLIGHT_FAILED
    assert api.created == []
    assert CANARY not in capsys.readouterr().err


# -- refusals --------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"--model": None},
        {"--model": "claude-unknown-9"},
        {"--max-requests": None},
        {"--max-requests": "0"},
        {"--max-requests": "-1"},
        {"--max-requests": "201"},
        {"--max-cost-usd": None},
        {"--max-cost-usd": "0"},
        {"--max-cost-usd": "-1"},
        {"--max-cost-usd": "5.01"},
    ],
)
def test_paid_mode_needs_explicit_bounded_limits(fake_api, capsys, overrides):
    api = fake_api()
    assert copilot_eval.main(paid_args(**overrides)) == copilot_eval.EXIT_REFUSED
    assert api.created == api.retrieved == []
    assert "refused" in capsys.readouterr().err


def test_paid_mode_needs_a_key(fake_api, monkeypatch):
    api = fake_api()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert copilot_eval.main(paid_args()) == copilot_eval.EXIT_REFUSED
    assert api.created == []


@pytest.mark.parametrize(("name", "value"), [("CI", "true"), ("CI", "1"), ("GITHUB_ACTIONS", "true")])
def test_paid_mode_is_refused_in_ci(fake_api, monkeypatch, name, value):
    api = fake_api()
    monkeypatch.setenv(name, value)
    assert copilot_eval.main(paid_args()) == copilot_eval.EXIT_REFUSED
    assert api.created == []


def test_limits_without_paid_mode_are_refused(capsys):
    code = copilot_eval.main(["--dataset", str(DATASET_DIR), "--max-requests", "5"])
    assert code == copilot_eval.EXIT_REFUSED


def test_preflight_cannot_be_combined_with_paid_mode(fake_api):
    assert copilot_eval.main([*paid_args(), "--preflight"]) == copilot_eval.EXIT_REFUSED


def test_dry_run_is_the_default_and_never_touches_the_sdk(monkeypatch, capsys):
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "anthropic":
            raise AssertionError("a dry run must not import the SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert copilot_eval.main(["--dataset", str(DATASET_DIR)]) == copilot_eval.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run"
    assert report["live_model_requests"] == 0
    assert report["contract_suite"]["fail_closed"]["rate"] == 1.0


def test_dry_run_with_a_key_present_still_sends_nothing(fake_api, capsys):
    api = fake_api()
    assert copilot_eval.main(["--dataset", str(DATASET_DIR)]) == copilot_eval.EXIT_OK
    assert api.created == api.retrieved == api.counted == []


def test_the_api_and_worker_never_load_the_evaluation():
    code = (
        "import sys; import footballai_v2.api; import footballai_v2.worker; import footballai_v2.copilot; "
        "loaded = [m for m in sys.modules if m.startswith('footballai_v2.copilot.evaluation')]; "
        "sys.exit(1 if loaded or 'anthropic' in sys.modules else 0)"
    )
    env_path = str(REPO_ROOT / "v2" / "src")
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": env_path}, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_output_nature_is_always_the_rendered_label(dataset):
    results = run_cases(selection_cases(dataset), dataset, OracleProvider)
    assert all(OUTPUT_NATURE_CLAIM_ID not in result.rendered_claims for result in results)
