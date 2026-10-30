"""Orchestrate evidence -> routing -> provider -> validation -> typed response.

Order matters for safety. A question FootballAI cannot answer, or that this run's
evidence cannot answer, is abstained before a provider is called, so it never
spends a token. When a question is answerable, the provider only selects
approved claim ids; the server renders the answer and its citations from the
catalog, never from model text.
"""

from __future__ import annotations

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evidence import RunEvidenceService, ValidatedRunEvidence
from footballai_v2.copilot.grounding import (
    QuestionScope,
    build_claim_catalog,
    build_system_instructions,
    can_answer,
    eligible_claims,
    render_abstention,
    render_from_selection,
    route_question,
    validate_citations,
)
from footballai_v2.copilot.models import ExplanationResponse, ModelMode, OutputNature, SupportLevel
from footballai_v2.copilot.provider import GenerationRequest, LLMProvider
from footballai_v2.storage.ports import AnalysisRepository, ObjectStorage


_UNSUPPORTED_REASON = "unsupported_question"
_NO_EVIDENCE_REASON = "evidence_not_available"
_NO_CLAIM_REASON = "no_relevant_claim"


class ExplanationService:
    """Produce a bounded, grounded explanation for one completed run."""

    def __init__(
        self,
        repository: AnalysisRepository,
        object_storage: ObjectStorage,
        provider: LLMProvider,
        settings: CopilotSettings | None = None,
    ) -> None:
        self._settings = settings or CopilotSettings()
        self._evidence_service = RunEvidenceService(repository, object_storage, self._settings)
        self._provider = provider

    def explain(self, run_id: str, question: str | None = None) -> ExplanationResponse:
        evidence = self._evidence_service.build(run_id)
        route = route_question(question)

        # Fail closed BEFORE the provider: neither of these spends a token.
        if route.scope is QuestionScope.UNSUPPORTED:
            return self._unavailable(evidence, _UNSUPPORTED_REASON, ModelMode.NOT_INVOKED, render_abstention())
        catalog = eligible_claims(build_claim_catalog(evidence), route)
        if not can_answer(catalog, route):
            return self._unavailable(
                evidence,
                _NO_EVIDENCE_REASON,
                ModelMode.NOT_INVOKED,
                "This run's verified evidence does not contain what the question asks for.",
            )

        request = GenerationRequest(
            evidence=evidence,
            question=question,
            system_instructions=build_system_instructions(),
            catalog=catalog,
            max_output_tokens=self._settings.max_output_tokens,
            timeout_seconds=self._settings.timeout_seconds,
        )
        try:
            result = self._provider.generate(request)
        except (ProviderTimeoutError, ProviderUnavailableError) as exc:
            return self._unavailable(
                evidence,
                exc.code,
                self._provider.mode,
                "The explanation model is currently unavailable, so no explanation could be "
                "produced. The run's verified evidence remains visible on this page.",
            )

        # Server-controlled rendering: unknown/forged ids raise here.
        rendered = render_from_selection(catalog, result.claim_ids, route)
        if rendered is None:
            return self._unavailable(
                evidence,
                _NO_CLAIM_REASON,
                result.mode,
                "No verified evidence claim answers that question for this run.",
            )
        answer, citations = rendered
        # Defence in depth even though citations are catalog-derived.
        validate_citations(citations, evidence, evidence.run_id)

        return ExplanationResponse(
            run_id=evidence.run_id,
            answer=answer,
            support_level=self._support_level(evidence),
            output_nature=evidence.output_nature,
            model_mode=result.mode,
            citations=citations,
            limitations=list(evidence.limitations),
            unavailable_reason=None,
        )

    def _support_level(self, evidence: ValidatedRunEvidence) -> SupportLevel:
        degraded = (
            evidence.output_nature is OutputNature.UNVERIFIED_OR_PARTIAL
            or bool(evidence.missing_evidence)
            or any(item.integrity_state != "verified" for item in evidence.artifacts)
        )
        return SupportLevel.PARTIAL if degraded else SupportLevel.SUPPORTED

    def _unavailable(
        self, evidence: ValidatedRunEvidence, reason: str, mode: ModelMode, answer: str
    ) -> ExplanationResponse:
        return ExplanationResponse(
            run_id=evidence.run_id,
            answer=answer,
            support_level=SupportLevel.UNAVAILABLE,
            output_nature=evidence.output_nature,
            model_mode=mode,
            citations=[],
            limitations=list(evidence.limitations),
            unavailable_reason=reason,
        )
