"""The explicit, opt-in path to a billed model for the agent.

The route uses the offline script unless ``FOOTBALLAI_COPILOT_AGENT_REAL_MODEL_ENABLED=1``
is set. With it set, every condition below must hold or the server refuses to start:
a request never falls back to the script, because a demo would then show one thing
while its operator believed another. The route's loopback guard still applies, and
the key stays in the server's environment.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Callable, Mapping

from footballai_v2.copilot.budget import BudgetedAgentModel, BudgetGuard
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.pricing import MODEL_PRICES
from footballai_v2.copilot.provider import AgentModel, AnthropicAgentModel, api_base_url


# Set by the platforms this project deploys to, or could. An environment name alone
# is a string someone typed, so a hosted platform is also recognised for what it is.
_HOSTED_MARKERS = (
    "CONTAINER_APP_NAME",
    "CONTAINER_APP_ENV_DNS_SUFFIX",
    "WEBSITE_INSTANCE_ID",
    "KUBERNETES_SERVICE_HOST",
)
_CI_MARKERS = ("CI", "GITHUB_ACTIONS")


class RealAgentRefused(RuntimeError):
    """The real path was requested, and is not allowed here. Holds codes, never values."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        super().__init__(f"the real agent model is refused: {', '.join(blockers)}")
        self.blockers = blockers


def _sdk_installed() -> bool:
    return importlib.util.find_spec("anthropic") is not None


def real_agent_blockers(
    settings: CopilotSettings,
    environ: Mapping[str, str] | None = None,
    *,
    sdk_installed: Callable[[], bool] = _sdk_installed,
) -> tuple[str, ...]:
    """Every reason the real path may not run here. Empty means it may."""
    environment = os.environ if environ is None else environ
    checks = {
        "real_agent_not_enabled": not settings.agent_real_model_enabled,
        "provider_not_real": not settings.wants_real_provider,
        "hosted_environment": any(environment.get(name) for name in _HOSTED_MARKERS),
        "continuous_integration": any(environment.get(name) for name in _CI_MARKERS),
        "missing_api_key": not environment.get("ANTHROPIC_API_KEY", "").strip(),
        "unapproved_api_endpoint": api_base_url(environment) is None,
        "unpriced_model": settings.model_name not in MODEL_PRICES,
        "no_request_cap": settings.agent_max_requests < 1,
        "no_cost_cap": settings.agent_max_cost_usd <= 0,
        "sdk_not_installed": not sdk_installed(),
    }
    return tuple(code for code, blocked in checks.items() if blocked)


def real_agent_factory(
    settings: CopilotSettings,
    environ: Mapping[str, str] | None = None,
    *,
    sdk_installed: Callable[[], bool] = _sdk_installed,
) -> Callable[[], AgentModel]:
    """One budget for the process, and one budgeted adapter per request."""
    blockers = real_agent_blockers(settings, environ, sdk_installed=sdk_installed)
    if blockers:
        raise RealAgentRefused(blockers)
    guard = BudgetGuard(settings.agent_max_requests, settings.agent_max_cost_usd, MODEL_PRICES[settings.model_name])
    return lambda: BudgetedAgentModel(AnthropicAgentModel(settings), guard, settings)


__all__ = ["RealAgentRefused", "real_agent_blockers", "real_agent_factory"]
