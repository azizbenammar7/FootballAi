"""Evidence-grounded "Explain This Run" copilot (Phase 1).

A small, read-only bounded context that explains an existing analysis run from
integrity-verified evidence. This entry point has no tool calling, RAG, or MCP.
The bounded agent lives in ``copilot.agent``, which is deliberately not imported
here. See docs/v2/ADR-003, docs/v2/ADR-005, and docs/v2/ADR-006.
"""

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import (
    CopilotError,
    FeatureDisabledError,
    ForgedCitationError,
    PromptBudgetExceededError,
    ProviderFailure,
    ProviderFailureKind,
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
    OutputNature,
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
    "OutputNature",
    "PromptBudgetExceededError",
    "ProviderFailure",
    "ProviderFailureKind",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RunEvidenceService",
    "RunNotEligibleError",
    "SupportLevel",
    "ValidatedRunEvidence",
]
