"""Typed errors for the evidence-grounded explanation copilot."""

from __future__ import annotations

from enum import StrEnum

from footballai_v2.copilot.models import ProviderTelemetry


class ProviderFailureKind(StrEnum):
    """Why a provider produced no usable selection. Operational detail only:
    API clients see the stable ``code``, never this classification."""

    NOT_CONFIGURED = "not_configured"
    PROMPT_BUDGET = "prompt_budget"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    BILLING = "billing"
    MODEL_UNAVAILABLE = "model_unavailable"
    BAD_REQUEST = "bad_request"
    REQUEST_TOO_LARGE = "request_too_large"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    REFUSAL = "refusal"
    TRUNCATED = "truncated"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class CopilotError(Exception):
    """Base class for copilot domain errors."""

    code = "copilot_error"


class FeatureDisabledError(CopilotError):
    """Raised when the explanation feature gate is disabled."""

    code = "copilot_disabled"


class RunNotEligibleError(CopilotError):
    """Raised when a run is not in a state that can be explained."""

    code = "run_not_eligible"


class ProviderFailure(CopilotError):
    """A provider call that yielded no selection.

    ``telemetry`` is kept because a failed call can still be billed: a refusal or
    a schema-invalid reply consumes tokens, so accounting must not lose it.
    """

    default_kind = ProviderFailureKind.UNKNOWN

    def __init__(
        self,
        message: str = "",
        *,
        kind: ProviderFailureKind | None = None,
        telemetry: ProviderTelemetry | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind or self.default_kind
        self.telemetry = telemetry


class ProviderUnavailableError(ProviderFailure):
    """Raised when a requested LLM provider cannot be used."""

    code = "provider_unavailable"


class PromptBudgetExceededError(ProviderUnavailableError):
    """Raised, before any model call, when the safe evidence does not fit the budget."""

    code = "prompt_budget_exceeded"
    default_kind = ProviderFailureKind.PROMPT_BUDGET


class ProviderTimeoutError(ProviderFailure):
    """Raised when provider generation exceeds the configured timeout."""

    code = "provider_timeout"
    default_kind = ProviderFailureKind.TIMEOUT


class ForgedCitationError(CopilotError):
    """Raised, fail-closed, when a citation does not resolve to supplied evidence."""

    code = "forged_citation"
