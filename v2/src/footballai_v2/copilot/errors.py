"""Typed errors for the evidence-grounded explanation copilot."""

from __future__ import annotations


class CopilotError(Exception):
    """Base class for copilot domain errors."""

    code = "copilot_error"


class FeatureDisabledError(CopilotError):
    """Raised when the explanation feature gate is disabled."""

    code = "copilot_disabled"


class RunNotEligibleError(CopilotError):
    """Raised when a run is not in a state that can be explained."""

    code = "run_not_eligible"


class ProviderUnavailableError(CopilotError):
    """Raised when a requested LLM provider cannot be used."""

    code = "provider_unavailable"


class ProviderTimeoutError(CopilotError):
    """Raised when provider generation exceeds the configured timeout."""

    code = "provider_timeout"


class ForgedCitationError(CopilotError):
    """Raised, fail-closed, when a citation does not resolve to supplied evidence."""

    code = "forged_citation"
