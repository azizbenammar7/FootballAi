"""Feature gate and bounded budgets for the explanation copilot.

The copilot has no authentication or authorization of its own. Because the
application still ships without user auth, the endpoint is a *development-only*
capability: it is DISABLED by default and can never enable itself in a hosted
(staging/production) environment. Run UUIDs are not treated as access control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Environments in which the dev-only gate may be turned on at all.
_LOCAL_ENVIRONMENTS = frozenset({"local", "test"})
_PROVIDER_MODES = frozenset({"fake", "real"})


def _flag(name: str) -> bool:
    return os.getenv(name, "0").strip() == "1"


@dataclass(frozen=True, slots=True)
class CopilotSettings:
    """Explicit, bounded configuration for the explanation feature."""

    enabled: bool = False
    environment: str = "local"
    provider_mode: str = "fake"
    real_provider_enabled: bool = False
    model_name: str = "claude-opus-4-8"
    # Budgets. Deliberately small: one short grounded paragraph over a bounded
    # evidence snapshot, never a long reasoning chain.
    max_artifact_bytes: int = 1 * 1024 * 1024
    max_evidence_bytes: int = 32 * 1024
    max_prompt_chars: int = 24 * 1024
    max_output_tokens: int = 700
    timeout_seconds: float = 20.0
    provider_retries: int = 0

    def __post_init__(self) -> None:
        if self.provider_mode not in _PROVIDER_MODES:
            raise ValueError("provider_mode must be 'fake' or 'real'")
        for name in ("max_artifact_bytes", "max_evidence_bytes", "max_prompt_chars", "max_output_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.provider_retries < 0:
            raise ValueError("provider_retries cannot be negative")

    @property
    def wants_real_provider(self) -> bool:
        """Whether a real, billed model should be used (requires an explicit opt-in)."""
        return self.provider_mode == "real" and self.real_provider_enabled

    @classmethod
    def from_environment(cls) -> "CopilotSettings":
        environment = os.getenv("FOOTBALLAI_ENVIRONMENT", "local").strip().lower()
        requested = _flag("FOOTBALLAI_COPILOT_ENABLED")
        # Fail closed: the gate can only open in a local/dev environment, never in
        # a hosted environment that lacks verified authorization controls.
        enabled = requested and environment in _LOCAL_ENVIRONMENTS
        provider_mode = os.getenv("FOOTBALLAI_COPILOT_PROVIDER", "fake").strip().lower()
        if provider_mode not in _PROVIDER_MODES:
            provider_mode = "fake"
        return cls(
            enabled=enabled,
            environment=environment,
            provider_mode=provider_mode,
            real_provider_enabled=_flag("FOOTBALLAI_COPILOT_REAL_PROVIDER_ENABLED"),
            model_name=os.getenv("FOOTBALLAI_COPILOT_MODEL", "claude-opus-4-8").strip() or "claude-opus-4-8",
        )
