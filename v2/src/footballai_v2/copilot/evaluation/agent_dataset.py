"""Typed loader for the scripted agent cases.

``copilot-agent-eval/v1`` adds tool expectations that the claim-selection datasets
cannot state, so it is a new dataset. It owns no fixtures: it replays the immutable
``copilot-eval/v2`` fixtures, pinned by hash so they cannot change underneath it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from footballai_v2.copilot.agent.tools import ToolName
from footballai_v2.copilot.agent.trace import AgentStopReason
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderTimeoutError, ProviderUnavailableError
from footballai_v2.copilot.evaluation.agent_models import Step, call, say, select, tool_use_block, thinking_block
from footballai_v2.copilot.evaluation.dataset import EvaluationDataset, load_dataset
from footballai_v2.copilot.models import OutputNature


AGENT_CASES_FILE = "cases.json"
AGENT_CASES_SCHEMA = "footballai.copilot-agent-eval-cases/v1"
# Factories, so every replay raises its own exception and no traceback or telemetry is shared.
_FAILURES = {
    "timeout": lambda: ProviderTimeoutError("scripted timeout"),
    "server_error": lambda: ProviderUnavailableError("scripted failure", kind=ProviderFailureKind.SERVER_ERROR),
    "rate_limited": lambda: ProviderUnavailableError("scripted failure", kind=ProviderFailureKind.RATE_LIMITED),
    "adapter_defect": lambda: RuntimeError("scripted defect canary-exception-text"),
    # The operator's spend cap, raised by the adapter before anything is sent.
    "operator_budget": lambda: ProviderUnavailableError("scripted spend cap", kind=ProviderFailureKind.BUDGET_EXHAUSTED),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScriptStep(_Strict):
    """Exactly one scripted model action."""

    call: tuple[str, ...] | None = None
    arguments: Any = None
    select: tuple[str, ...] | None = None
    text: str | None = None
    stop_reason: str | None = None
    null_content: bool = False
    tool_use_without_id: bool = False
    failure: str | None = None
    takes_seconds: float = 0.0

    @model_validator(mode="after")
    def _one_action(self) -> "ScriptStep":
        actions = [self.call, self.select, self.text, self.stop_reason, self.failure]
        if sum(action is not None for action in actions) != 1:
            raise ValueError("a script step takes exactly one action")
        if self.failure is not None and self.failure not in _FAILURES:
            raise ValueError(f"unknown scripted failure {self.failure!r}")
        return self

    def to_step(self) -> Step:
        if self.call is not None:
            return call(*self.call, arguments=self.arguments, takes_seconds=self.takes_seconds)
        if self.select is not None:
            return select(*self.select, takes_seconds=self.takes_seconds)
        if self.text is not None:
            return say(self.text, takes_seconds=self.takes_seconds)
        if self.failure is not None:
            return Step(error=_FAILURES[self.failure](), takes_seconds=self.takes_seconds)
        if self.tool_use_without_id:
            block = tool_use_block("get_run_status", {}, None)
            return Step((thinking_block(), block), self.stop_reason)
        return Step(None if self.null_content else (), self.stop_reason)


class AgentExpected(_Strict):
    stop: AgentStopReason
    output_nature: OutputNature
    unavailable_reason: str | None = None
    tools_executed: tuple[ToolName, ...] = ()  # the exact sequence that ran
    required_tools: tuple[ToolName, ...] = ()
    forbidden_tools: tuple[ToolName, ...] = ()
    required_claims: tuple[str, ...] = ()
    permitted_claims: tuple[str, ...] = ()
    forbidden_text: tuple[str, ...] = ()
    model_turns: int = Field(ge=0, le=5)

    @model_validator(mode="after")
    def _consistent(self) -> "AgentExpected":
        answered = self.stop is AgentStopReason.ANSWERED
        if answered != (self.unavailable_reason is None):
            raise ValueError("only an answered case has no unavailable_reason")
        if answered and not self.required_claims:
            raise ValueError("an answered case must name its required claims")
        if not set(self.required_claims) <= set(self.permitted_claims):
            raise ValueError("required_claims must be permitted")
        if set(self.required_tools) & set(self.forbidden_tools):
            raise ValueError("a tool cannot be both required and forbidden")
        return self


class AgentCase(_Strict):
    case_id: str = Field(pattern=r"^[a-z0-9-]+$")
    group: str
    fixture: str
    question: str | None
    script: tuple[ScriptStep, ...] = ()
    expected: AgentExpected
    notes: str = ""


class _FixturePin(_Strict):
    dataset: str
    fixtures_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AgentCaseSet(_Strict):
    schema_id: str = Field(alias="schema", pattern=f"^{AGENT_CASES_SCHEMA}$")
    dataset_version: str
    fixtures: _FixturePin
    cases: tuple[AgentCase, ...]


@dataclass(frozen=True, slots=True)
class AgentDataset:
    version: str
    cases: tuple[AgentCase, ...]
    cases_sha256: str
    source: EvaluationDataset  # the claim-selection dataset supplying fixtures and routing


def load_agent_dataset(directory: Path) -> AgentDataset:
    """Load and cross-check the agent cases; raises ``ValueError`` if invalid."""
    cases_bytes = (directory / AGENT_CASES_FILE).read_bytes()
    case_set = AgentCaseSet.model_validate_json(cases_bytes)
    source = load_dataset(directory.parent / case_set.fixtures.dataset)
    if source.fixtures_sha256 != case_set.fixtures.fixtures_sha256:
        raise ValueError("the pinned fixtures changed: a new agent dataset version is required")
    identifiers = [case.case_id for case in case_set.cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case_id values must be unique")
    for case in case_set.cases:
        if case.fixture not in source.fixtures.fixtures:
            raise ValueError(f"case {case.case_id} references unknown fixture {case.fixture!r}")
    return AgentDataset(
        version=case_set.dataset_version,
        cases=case_set.cases,
        cases_sha256=hashlib.sha256(cases_bytes).hexdigest(),
        source=source,
    )


__all__ = ["AgentCase", "AgentDataset", "AgentExpected", "ScriptStep", "load_agent_dataset"]
