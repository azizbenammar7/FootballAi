"""Orchestrate evidence -> provider -> validation -> typed grounded response."""

from __future__ import annotations

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evidence import RunEvidenceService, ValidatedRunEvidence
from footballai_v2.copilot.grounding import (
    QuestionScope,
    build_citations,
    build_system_instructions,
    classify_question,
    render_abstention,
    validate_citations,
)
from footballai_v2.copilot.models import ExplanationResponse, ModelMode, SupportLevel
from footballai_v2.copilot.provider import GenerationRequest, LLMProvider
from footballai_v2.storage.ports import AnalysisRepository, ObjectStorage


_MAX_ANSWER_CHARS = 4000


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
        scope = classify_question(question)
        request = GenerationRequest(
            evidence=evidence,
            scope=scope,
            question=question,
            system_instructions=build_system_instructions(),
            max_output_tokens=self._settings.max_output_tokens,
            timeout_seconds=self._settings.timeout_seconds,
        )

        try:
            result = self._provider.generate(request)
        except (ProviderTimeoutError, ProviderUnavailableError) as exc:
            return self._provider_unavailable_response(evidence, exc)

        answer = result.text.strip()[:_MAX_ANSWER_CHARS]

        if scope is QuestionScope.UNSUPPORTED:
            return ExplanationResponse(
                run_id=evidence.run_id,
                answer=answer or render_abstention(question),
                support_level=SupportLevel.UNAVAILABLE,
                model_mode=result.mode,
                citations=[],
                limitations=list(evidence.limitations),
                unavailable_reason=(
                    "The question asks for information FootballAI does not measure for this run "
                    "(for example passes, possession, player identities, or clinical risk)."
                ),
            )

        citations = build_citations(evidence)
        # Fail closed: defence in depth even though citations are evidence-derived.
        validate_citations(citations, evidence, evidence.run_id)
        support_level = self._support_level(evidence)
        return ExplanationResponse(
            run_id=evidence.run_id,
            answer=answer,
            support_level=support_level,
            model_mode=result.mode,
            citations=citations,
            limitations=list(evidence.limitations),
            unavailable_reason=None,
        )

    def _support_level(self, evidence: ValidatedRunEvidence) -> SupportLevel:
        if not evidence.is_synthetic and evidence.diagnostics is None:
            return SupportLevel.PARTIAL
        return SupportLevel.SUPPORTED

    def _provider_unavailable_response(
        self, evidence: ValidatedRunEvidence, exc: Exception
    ) -> ExplanationResponse:
        return ExplanationResponse(
            run_id=evidence.run_id,
            answer=(
                "The explanation model is currently unavailable, so no generated explanation "
                "could be produced. The run's verified evidence remains visible on this page."
            ),
            support_level=SupportLevel.UNAVAILABLE,
            model_mode=getattr(self._provider, "mode", ModelMode.REAL_MODEL),
            citations=[],
            limitations=list(evidence.limitations),
            unavailable_reason=getattr(exc, "code", "provider_unavailable"),
        )
