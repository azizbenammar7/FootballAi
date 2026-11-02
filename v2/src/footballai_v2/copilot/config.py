"""Feature gate and bounded budgets for the explanation copilot.

The copilot has no authentication or authorization of its own. Because the
application still ships without user auth, the endpoint is a *development-only*
capability: it is DISABLED by default and can never enable itself in a hosted
(staging/production) environment. Run UUIDs are not treated as access control.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


# Environments in which the dev-only gate may be turned on at all.
_LOCAL_ENVIRONMENTS = frozenset({"local", "test"})
_PROVIDER_MODES = frozenset({"fake", "real"})
_EFFORT_LEVELS = frozenset({"low", "medium", "high"})
# Claim selection is a small classification task; see docs/v2/COPILOT_LLM_EVALUATION.md.
DEFAULT_MODEL = "claude-haiku-5-5"
# Ceilings for the local real-model agent. Configuration can only choose lower caps.
MAX_AGENT_REQUESTS = 200
MAX_AGENT_COST_USD = 5.0


def _flag(name: str) -> bool:
    return os.getenv(name, "0").strip() == "1"


def _cap(name: str, ceiling: float) -> float:
    """A positive, finite number within ``ceiling``. Anything else is no cap, so it refuses."""
    try:
        value = float(os.getenv(name, "0"))
    except ValueError:
        return 0.0
    return value if math.isfinite(value) and 0 < value <= ceiling else 0.0


@dataclass(frozen=True, slots=True)
class CopilotSettings:
    """Explicit, bounded configuration for the explanation feature."""

    enabled: bool = False
    environment: str = ""  # never assumed: must be explicit before the gate can open
    provider_mode: str = "fake"
    real_provider_enabled: bool = False
    agent_enabled: bool = False  # a second key: the agent route also needs the copilot gate
    agent_real_model_enabled: bool = False  # a third key: see copilot/real_agent.py
    agent_max_requests: int = 0  # billed model calls this process may send
    agent_max_cost_usd: float = 0.0
    model_name: str = DEFAULT_MODEL
    effort: str = "low"
    # Budgets. Deliberately small: one short grounded paragraph over a bounded
    # evidence snapshot, never a long reasoning chain.
    max_artifact_bytes: int = 1 * 1024 * 1024
    max_evidence_bytes: int = 32 * 1024
    max_prompt_chars: int = 24 * 1024
    max_output_tokens: int = 1024  # includes any thinking the model does first
    timeout_seconds: float = 20.0
    provider_retries: int = 0

    def __post_init__(self) -> None:
        if self.enabled and self.environment not in _LOCAL_ENVIRONMENTS:
            raise ValueError("the copilot can only be enabled in a local or test environment")
        if self.agent_enabled and not self.enabled:
            raise ValueError("the agent can only be enabled with the copilot")
        if self.agent_real_model_enabled and not self.agent_enabled:
            raise ValueError("the real agent model can only be enabled with the agent")
        if self.provider_mode not in _PROVIDER_MODES:
            raise ValueError("provider_mode must be 'fake' or 'real'")
        if self.effort not in _EFFORT_LEVELS:
            raise ValueError("effort must be 'low', 'medium' or 'high'")
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
        # Fail closed: an absent/empty/unknown environment is NOT assumed to be
        # local. Explicit development/test configuration is required before the
        # gate can open, and it can never open in a hosted environment that lacks
        # verified authorization controls.
        environment = os.getenv("FOOTBALLAI_ENVIRONMENT", "").strip().lower()
        requested = _flag("FOOTBALLAI_COPILOT_ENABLED")
        enabled = requested and environment in _LOCAL_ENVIRONMENTS
        provider_mode = os.getenv("FOOTBALLAI_COPILOT_PROVIDER", "fake").strip().lower()
        if provider_mode not in _PROVIDER_MODES:
            provider_mode = "fake"
        agent_enabled = enabled and _flag("FOOTBALLAI_COPILOT_AGENT_ENABLED")
        return cls(
            enabled=enabled,
            environment=environment,
            provider_mode=provider_mode,
            real_provider_enabled=_flag("FOOTBALLAI_COPILOT_REAL_PROVIDER_ENABLED"),
            agent_enabled=agent_enabled,
            agent_real_model_enabled=agent_enabled and _flag("FOOTBALLAI_COPILOT_AGENT_REAL_MODEL_ENABLED"),
            agent_max_requests=int(_cap("FOOTBALLAI_COPILOT_AGENT_MAX_REQUESTS", MAX_AGENT_REQUESTS)),
            agent_max_cost_usd=_cap("FOOTBALLAI_COPILOT_AGENT_MAX_COST_USD", MAX_AGENT_COST_USD),
            model_name=os.getenv("FOOTBALLAI_COPILOT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        )
