"""Evidence-grounded "Explain This Run" copilot (Phase 1).

A small, read-only bounded context that explains an existing analysis run from
integrity-verified evidence. No autonomous agent, no tool calling, no RAG, and
no MCP: those belong to later phases. See docs/v2/ADR-003.
"""

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import (
    CopilotError,
    FeatureDisabledError,
    ForgedCitationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RunNotEligibleError,
)
from footballai_v2.copilot.evidence import (
    RunEvidenceService,
    ValidatedRunEvidence,
)
from footballai_v2.copilot.models import (
    Citation,
    ExplainRequest,
    ExplanationResponse,
    ModelMode,
    SupportLevel,
)
from footballai_v2.copilot.provider import (
    AnthropicTextProvider,
    DeterministicFakeProvider,
    LLMProvider,
    build_provider,
)
from footballai_v2.copilot.service import ExplanationService

__all__ = [
    "AnthropicTextProvider",
    "Citation",
    "CopilotError",
    "CopilotSettings",
    "DeterministicFakeProvider",
    "ExplainRequest",
    "ExplanationResponse",
    "ExplanationService",
    "FeatureDisabledError",
    "ForgedCitationError",
    "LLMProvider",
    "ModelMode",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RunEvidenceService",
    "RunNotEligibleError",
    "SupportLevel",
    "ValidatedRunEvidence",
]
