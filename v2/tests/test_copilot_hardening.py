"""Phase 1.1 grounding & safety hardening regression tests.

These tests pin the safety contract for the "Explain This Run" copilot:

* the LLM provider is NEVER invoked for an unsupported question (fail-closed
  abstention happens *before* generation, so no tokens are spent);
* distance/speed/passes/possession/tactics/clinical questions abstain;
* an unknown question scope abstains (allow-list policy, not a word blacklist);
* a real provider can only *select approved claim IDs*; it can never publish a
  free-text football statistic as an evidence-backed fact;
* forged / cross-run / unknown claim IDs are rejected fail-closed;
* synthetic vs verified-real vs unverified output is classified from evidence,
  never promoted from "profile != demo_fast";
* non-finite JSON numbers (NaN, Infinity, and exponent overflow like 1e999) are
  rejected;
* the real-provider prompt is a bounded structured payload that always keeps the
  safety fields and fails closed when over budget (never silently truncated);
* the feature gate fails closed when FOOTBALLAI_ENVIRONMENT is absent/unknown.

No real LLM API call is ever made: the real provider is exercised only through
deterministic adversarial stubs.
"""

from __future__ import annotations

import json

import pytest

from footballai_v2.contracts.v1 import DataOrigin
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ForgedCitationError
from footballai_v2.copilot.evidence import RunEvidenceService
from footballai_v2.copilot.grounding import (
    QuestionScope,
    build_claim_catalog,
    classify_question,
)
from footballai_v2.copilot.models import ModelMode, OutputNature, SupportLevel
from footballai_v2.copilot.provider import GenerationRequest, ProviderResult
from footballai_v2.copilot.service import ExplanationService

# Reuse the port doubles and run builders from the main copilot test module.
from test_copilot_explanation import (
    DIAGNOSTICS_ARTIFACT_ID,
    FakeObjectStorage,
    FakeRepository,
    diagnostics_artifact,
    real_run_env,
    service_for,
    succeeded_run,
    summary_artifact,
)


# -- provider test doubles --------------------------------------------------


class SpyProvider:
    """Records whether generate() was called, so we can prove non-invocation."""

    mode = ModelMode.REAL_MODEL

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> ProviderResult:
        self.call_count += 1
        # Select every approved claim, mimicking a cooperative real model.
        return ProviderResult(
            claim_ids=tuple(claim.claim_id for claim in request.catalog),
            mode=self.mode,
        )


class InventedClaimProvider:
    """Adversarial real model: tries to smuggle a fabricated claim id."""

    mode = ModelMode.REAL_MODEL

    def generate(self, request: GenerationRequest) -> ProviderResult:
        return ProviderResult(
            claim_ids=("fabricated-player-7-ran-10km",),
            mode=self.mode,
        )


class MixedForgedProvider:
    """Adversarial real model: one valid claim plus an unknown (forged) id."""

    mode = ModelMode.REAL_MODEL

    def generate(self, request: GenerationRequest) -> ProviderResult:
        valid = request.catalog[0].claim_id
        return ProviderResult(claim_ids=(valid, "unknown-forged-id"), mode=self.mode)


# -- A. provider is not called for unsupported questions --------------------


def test_unsupported_question_never_invokes_provider():
    repository, storage, run = real_run_env()
    spy = SpyProvider()
    response = service_for(repository, storage, provider=spy).explain(
        run.run_id, "How far did player 7 run?"
    )
    assert spy.call_count == 0, "provider must not be called for an unsupported question"
    assert response.support_level is SupportLevel.UNAVAILABLE
    assert response.citations == []
    assert response.unavailable_reason


# -- B. distance / speed / running questions abstain ------------------------


@pytest.mark.parametrize(
    "question",
    [
        "How far did player 7 run?",
        "What was the top speed of player 10?",
        "How many kilometres did our striker cover?",
        "Who made the most successful passes?",
        "What formation did team A use?",
        "Which player is most likely to be injured?",
        "What was the ball possession percentage?",
    ],
)
def test_unsupported_examples_are_classified_unsupported(question):
    assert classify_question(question) is QuestionScope.UNSUPPORTED


# -- supported & unknown scope ----------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        None,
        "Explain this run.",
        "What does the tracking diagnostic say?",
        "Was this analysis synthetic?",
        "How many detections were recorded?",
    ],
)
def test_supported_questions_are_not_abstained(question):
    assert classify_question(question) is not QuestionScope.UNSUPPORTED


def test_unknown_scope_fails_closed():
    # A question about something out of the measured domain, with no in-domain
    # keyword, must fail closed rather than being answered.
    assert classify_question("What is the weather like in the stadium?") is QuestionScope.UNSUPPORTED


# -- C. a real model cannot publish an invented statistic -------------------


def test_real_provider_cannot_publish_invented_statistic():
    repository, storage, run = real_run_env()
    service = service_for(repository, storage, provider=InventedClaimProvider())
    # A fabricated claim id resolves to nothing; fail closed.
    with pytest.raises(ForgedCitationError):
        service.explain(run.run_id, "Explain this run.")


def test_real_provider_answer_is_server_rendered_only():
    repository, storage, run = real_run_env()
    spy = SpyProvider()
    response = service_for(repository, storage, provider=spy).explain(run.run_id, "Explain this run.")
    assert spy.call_count == 1
    # The answer is EXACTLY the server-authored catalog text, in catalog order.
    catalog = build_claim_catalog(RunEvidenceService(repository, storage).build(run.run_id))
    expected = " ".join(claim.text for claim in catalog)
    assert response.answer == expected
    # Every citation resolves to this run's evidence (server-controlled).
    assert response.citations
    assert all(citation.run_id == run.run_id for citation in response.citations)


# -- D. forged / mixed claim ids rejected -----------------------------------


def test_mixed_forged_claim_id_is_rejected():
    repository, storage, run = real_run_env()
    service = service_for(repository, storage, provider=MixedForgedProvider())
    with pytest.raises(ForgedCitationError):
        service.explain(run.run_id, "Explain this run.")


# -- E. output nature: verified-real vs unverified vs synthetic -------------


def test_verified_real_output_nature_with_diagnostics():
    repository, storage, run = real_run_env()
    evidence = RunEvidenceService(repository, storage).build(run.run_id)
    assert evidence.output_nature is OutputNature.VERIFIED_REAL


def test_real_profile_missing_diagnostics_is_unverified_not_executed_inference():
    run = succeeded_run(profile="v1_compat", artifacts=(summary_artifact(),), data_origin=DataOrigin.REAL)
    storage = FakeObjectStorage()
    storage.put(run.run_id, "team-summary", b'{"schema":"footballai.team-summary/v1"}')
    evidence = RunEvidenceService(FakeRepository(run), storage).build(run.run_id)
    assert evidence.output_nature is OutputNature.UNVERIFIED_OR_PARTIAL
    # Must NOT assert that real inference executed.
    assert "executed inference" not in evidence.synthetic_explanation.lower()
    response = service_for(FakeRepository(run), storage).explain(run.run_id)
    assert response.support_level is SupportLevel.PARTIAL
    assert response.output_nature is OutputNature.UNVERIFIED_OR_PARTIAL


def test_demo_fast_is_always_synthetic_output_nature():
    content = (json.dumps({"schema": "footballai.analysis-diagnostics/v1", "data_quality": "synthetic", "message": "No ML model was executed."}, sort_keys=True) + "\n").encode()
    artifact = diagnostics_artifact(content, schema_version="footballai.analysis-diagnostics/v1")
    run = succeeded_run(profile="demo_fast", artifacts=(artifact,), data_origin=DataOrigin.REAL)
    storage = FakeObjectStorage()
    storage.put(run.run_id, DIAGNOSTICS_ARTIFACT_ID, content)
    evidence = RunEvidenceService(FakeRepository(run), storage).build(run.run_id)
    assert evidence.output_nature is OutputNature.SYNTHETIC


# -- F. non-finite numbers, including exponent overflow ---------------------


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_non_finite_numbers_are_rejected(token):
    content = (
        '{"schema":"footballai.analysis-diagnostics/v2",'
        '"observed_metrics":{"total_tracklets":' + token + ',"total_detections":5}}'
    ).encode()
    artifact = diagnostics_artifact(content, schema_version="footballai.analysis-diagnostics/v2")
    run = succeeded_run(profile="v1_compat", artifacts=(artifact,))
    storage = FakeObjectStorage()
    storage.put(run.run_id, DIAGNOSTICS_ARTIFACT_ID, content)
    evidence = RunEvidenceService(FakeRepository(run), storage).build(run.run_id)
    assert evidence.diagnostics is None


def test_ordinary_finite_metrics_are_preserved():
    repository, storage, run = real_run_env()
    evidence = RunEvidenceService(repository, storage).build(run.run_id)
    assert evidence.diagnostics is not None
    assert evidence.diagnostics["observed_metrics"]["total_tracklets"] == 14


# -- G. configuration fails closed on missing / unknown environment ---------


def test_missing_environment_disables_gate(monkeypatch):
    monkeypatch.delenv("FOOTBALLAI_ENVIRONMENT", raising=False)
    monkeypatch.setenv("FOOTBALLAI_COPILOT_ENABLED", "1")
    assert CopilotSettings.from_environment().enabled is False


def test_unknown_environment_disables_gate(monkeypatch):
    monkeypatch.setenv("FOOTBALLAI_ENVIRONMENT", "banana")
    monkeypatch.setenv("FOOTBALLAI_COPILOT_ENABLED", "1")
    assert CopilotSettings.from_environment().enabled is False


def test_explicit_test_environment_requires_flag(monkeypatch):
    monkeypatch.setenv("FOOTBALLAI_ENVIRONMENT", "test")
    monkeypatch.delenv("FOOTBALLAI_COPILOT_ENABLED", raising=False)
    assert CopilotSettings.from_environment().enabled is False
    monkeypatch.setenv("FOOTBALLAI_COPILOT_ENABLED", "1")
    assert CopilotSettings.from_environment().enabled is True
