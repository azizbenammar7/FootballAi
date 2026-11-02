"""The cost guard: nothing is sent unless its worst case still fits the budget.

Used by the paid evaluations and by the local real-model route. A request reserves
its worst case before it is sent, and the reservation is replaced by its reported
usage afterwards. A request with no usable usage, such as one that timed out and
may still be billed, stays charged at its worst case.

The guard lives in memory, so it covers one process. A restart resets it, which is
why the Console's spend limit stays the backstop for real money.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ProviderFailure, ProviderFailureKind, ProviderUnavailableError
from footballai_v2.copilot.models import ModelMode, ProviderTelemetry
from footballai_v2.copilot.pricing import ModelPrice, request_cost_usd, worst_case_cost_usd
from footballai_v2.copilot.provider import AgentModel, AgentModelRequest, AgentReply


class BudgetExhausted(Exception):
    """Raised, before a request is sent, when the budget cannot cover it."""

    def __init__(self, limit: str) -> None:
        super().__init__(limit)
        self.limit = limit


class BudgetGuard:
    """Counts requests and money, conservatively, across every thread of a process."""

    def __init__(self, max_requests: int, max_cost_usd: float, price: ModelPrice) -> None:
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.price = price
        self.requests_sent = 0
        self.spent_usd = 0.0
        self.reserved_usd = 0.0  # admitted, and not charged yet
        self.requests_at_worst_case = 0
        self._lock = threading.Lock()

    def admit(self, worst_case_usd: float) -> None:
        with self._lock:
            if self.requests_sent >= self.max_requests:
                raise BudgetExhausted("max_requests")
            if self.spent_usd + self.reserved_usd + worst_case_usd > self.max_cost_usd:
                raise BudgetExhausted("max_cost_usd")
            self.requests_sent += 1
            self.reserved_usd += worst_case_usd

    def release(self, worst_case_usd: float) -> None:
        """Undo an admission for a request that was refused locally and never sent."""
        with self._lock:
            self.requests_sent -= 1
            self.reserved_usd = max(self.reserved_usd - worst_case_usd, 0.0)

    def charge(self, telemetry: ProviderTelemetry | None, worst_case_usd: float) -> None:
        with self._lock:
            self.reserved_usd = max(self.reserved_usd - worst_case_usd, 0.0)
            if telemetry is None or telemetry.usage is None:
                self.requests_at_worst_case += 1
                self.spent_usd += worst_case_usd
            else:
                self.spent_usd += request_cost_usd(self.price, telemetry.usage)


def _server_bytes(request: AgentModelRequest) -> int:
    """Bytes of what the server wrote. No token is shorter than one byte."""
    size = len(request.system_instructions.encode()) + len(json.dumps(list(request.tools)).encode())
    for message in request.messages:
        if message["role"] != "assistant":
            size += len(json.dumps(message["content"], ensure_ascii=False).encode())
    return size


def turn_worst_case_usd(price: ModelPrice, request: AgentModelRequest) -> float:
    """The most this turn can cost, resending at most a full output for each earlier turn.

    An assistant turn holds SDK objects and thinking that cannot be measured here, but
    no turn can have produced more than the output limit.
    """
    earlier_turns = sum(message["role"] == "assistant" for message in request.messages)
    prompt = _server_bytes(request) + earlier_turns * request.max_output_tokens
    return worst_case_cost_usd(price, prompt, request.max_output_tokens)


class BudgetedAgentModel:
    """The only route to the paid agent model: admit, call, charge."""

    mode: ModelMode

    def __init__(self, inner: AgentModel, guard: BudgetGuard, settings: CopilotSettings) -> None:
        if settings.provider_retries != 0:
            raise ValueError("a budgeted model cannot allow hidden SDK retries")
        self._inner = inner
        self._guard = guard
        self.mode = inner.mode
        self.model_name = inner.model_name

    def complete(self, request: AgentModelRequest) -> AgentReply:
        worst_case = turn_worst_case_usd(self._guard.price, request)
        try:
            self._guard.admit(worst_case)
        except BudgetExhausted as exc:
            raise ProviderUnavailableError(
                f"the agent budget is exhausted ({exc.limit})", kind=ProviderFailureKind.BUDGET_EXHAUSTED
            ) from None
        try:
            reply = self._inner.complete(request)
        except ProviderFailure as exc:
            if exc.kind is ProviderFailureKind.NOT_CONFIGURED:
                self._guard.release(worst_case)
            else:
                self._guard.charge(exc.telemetry, worst_case)
            raise
        except Exception:
            self._guard.charge(None, worst_case)
            raise
        self._guard.charge(reply.telemetry, worst_case)
        return reply

    def close(self) -> None:
        close: Any = getattr(self._inner, "close", None)
        if close is not None:
            close()


__all__ = ["BudgetExhausted", "BudgetGuard", "BudgetedAgentModel", "turn_worst_case_usd"]
