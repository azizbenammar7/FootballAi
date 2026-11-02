"""Deterministic stand-ins for a tool-calling model. None is a language model.

Their replies are shaped like Messages API responses and go through the real
``parse_agent_message``, so a hostile reply exercises the path production takes.
They report ``offline_demo_fake`` and can never be mistaken for real inference.
Everything here proves plumbing and safety, not what a real model would do.

The local-only agent route uses ``offline_agent_model``, so a demo shows the same
policy the evaluation baselines, and is labelled as a script wherever it appears.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

from footballai_v2.copilot.agent.loop import AgentLimits
from footballai_v2.copilot.agent.tools import READ_ONLY_TOOLS, ToolName
from footballai_v2.copilot.errors import ProviderFailureKind, ProviderUnavailableError
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID
from footballai_v2.copilot.models import ModelMode, ProviderTelemetry, TokenUsage
from footballai_v2.copilot.provider import AgentModelRequest, AgentReply, parse_agent_message


SCRIPTED_MODEL_NAME = "scripted-stub"
_REPORTED_USAGE = TokenUsage(100, 20)


@dataclass(frozen=True, slots=True)
class Step:
    """One scripted model turn: a response, or an exception to raise."""

    content: Any = ()
    stop_reason: Any = "end_turn"
    usage: TokenUsage | None = _REPORTED_USAGE
    error: Exception | None = None
    takes_seconds: float = 0.0


def thinking_block() -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="", signature="sig-scripted")


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, arguments: Any, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def call(*names: str, arguments: Any = None, takes_seconds: float = 0.0) -> Step:
    """Call tools in one turn, with empty input unless ``arguments`` is given."""
    blocks = [
        tool_use_block(name, {} if arguments is None else arguments, f"toolu_{index}_{name[:24]}")
        for index, name in enumerate(names)
    ]
    return Step((thinking_block(), *blocks), "tool_use", takes_seconds=takes_seconds)


def say(text: str, *, takes_seconds: float = 0.0) -> Step:
    """Finish with raw text. Valid only if it is the structured selection."""
    return Step((thinking_block(), text_block(text)), takes_seconds=takes_seconds)


def select(*claim_ids: str, takes_seconds: float = 0.0) -> Step:
    return say(json.dumps({"claim_ids": list(claim_ids)}), takes_seconds=takes_seconds)


def tool_result_claims(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    """Claim ids in the tool results of a conversation, in the order they came."""
    claim_ids: list[str] = []
    for message in messages:
        content = message["content"]
        if message["role"] != "user" or not isinstance(content, (list, tuple)):
            continue
        for block in content:
            if not block.get("is_error"):
                claim_ids.extend(claim["id"] for claim in json.loads(block["content"])["claims"])
    return claim_ids


def protocol_violations(requests: list[AgentModelRequest]) -> list[str]:
    """Breaches of the Messages API tool protocol across one conversation."""
    violations: list[str] = []
    for previous, current in zip(requests, requests[1:]):
        if current.messages[: len(previous.messages)] != previous.messages:
            violations.append("history was edited")
    if not requests:
        return violations
    pending: set[str] = set()
    for index, message in enumerate(requests[-1].messages):
        expected_role = "assistant" if index % 2 else "user"
        if message["role"] != expected_role:
            violations.append(f"message {index} has the wrong role")
        content = message["content"]
        if message["role"] == "assistant":
            pending = {block.id for block in content if block.type == "tool_use"}
        elif index:
            if not all(block.get("type") == "tool_result" for block in content):
                violations.append(f"message {index} mixes other blocks with tool results")
            answered = [block["tool_use_id"] for block in content]
            if len(answered) != len(set(answered)) or set(answered) != pending:
                violations.append(f"message {index} does not answer exactly the calls made")
    return violations


class ScriptedAgentModel:
    """Replays steps, or asks ``policy`` for each turn. A test stub, not a model."""

    mode = ModelMode.OFFLINE_DEMO_FAKE
    model_name = SCRIPTED_MODEL_NAME

    def __init__(
        self,
        steps: Iterable[Step] = (),
        *,
        policy: Callable[[AgentModelRequest], Step] | None = None,
        on_call: Callable[[float], None] | None = None,
    ) -> None:
        self._steps = list(steps)
        self._policy = policy
        self._on_call = on_call
        self.requests: list[AgentModelRequest] = []

    def complete(self, request: AgentModelRequest) -> AgentReply:
        self.requests.append(request)
        step = self._next(request)
        if self._on_call is not None:
            self._on_call(step.takes_seconds)
        if step.error is not None:
            raise step.error
        telemetry = ProviderTelemetry(
            model=SCRIPTED_MODEL_NAME,
            latency_seconds=step.takes_seconds,
            request_id=f"req_scripted_{len(self.requests)}",
            stop_reason=step.stop_reason,
            usage=step.usage,
        )
        return parse_agent_message(SimpleNamespace(stop_reason=step.stop_reason, content=step.content), telemetry)

    def _next(self, request: AgentModelRequest) -> Step:
        if self._policy is not None:
            return self._policy(request)
        if not self._steps:  # a script that runs out must fail, never improvise
            return Step(error=ProviderUnavailableError("the script ran out", kind=ProviderFailureKind.INVALID_RESPONSE))
        return self._steps.pop(0)


_RETRIEVAL_ORDER = (
    ToolName.GET_RUN_STATUS,
    ToolName.GET_TRACKING_DIAGNOSTICS,
    ToolName.GET_EVIDENCE_LIMITATIONS,
    ToolName.GET_MODEL_PROVENANCE,
    ToolName.GET_ARTIFACT_METADATA,
)


def _tools_for(claim_ids: set[str]) -> list[str]:
    specs = {spec.name: spec for spec in READ_ONLY_TOOLS}
    return [name.value for name in _RETRIEVAL_ORDER if specs[name].claims & claim_ids][: AgentLimits().max_tool_calls]


def route_following_policy(request: AgentModelRequest) -> Step:
    """Fetch the tools owning the offered claims, then select all that came back.

    The deterministic agent baseline: a broad summary spans five tools, so with a
    limit of four this policy leaves artifact metadata out.
    """
    if len(request.messages) > 1:
        return select(*tool_result_claims(request.messages))
    offered = set(request.offered_claim_ids)
    return call(*_tools_for(offered - {OUTPUT_NATURE_CLAIM_ID} or offered))


def offline_agent_model() -> ScriptedAgentModel:
    """The adapter for one request. It keeps its requests, so it is never shared."""
    return ScriptedAgentModel(policy=route_following_policy)


def replay_text_policy(text: str) -> Callable[[AgentModelRequest], Step]:
    """Retrieve everything offered, label included, then finish with a recorded reply."""

    def policy(request: AgentModelRequest) -> Step:
        return say(text) if len(request.messages) > 1 else call(*_tools_for(set(request.offered_claim_ids)))

    return policy


def every_tool_policy(_request: AgentModelRequest) -> Step:
    """Negative control: asks for all five tools at once, over the limit of four."""
    return call(*(name.value for name in ToolName))


def wrong_tool_policy(request: AgentModelRequest) -> Step:
    """Negative control: always reads artifact metadata, whatever was asked."""
    if len(request.messages) > 1:
        return select(*tool_result_claims(request.messages))
    return call(ToolName.GET_ARTIFACT_METADATA.value)


def skip_retrieval_policy(request: AgentModelRequest) -> Step:
    """Negative control: selects every offered id without retrieving anything."""
    return select(*request.offered_claim_ids)


__all__ = [
    "SCRIPTED_MODEL_NAME",
    "ScriptedAgentModel",
    "Step",
    "call",
    "every_tool_policy",
    "offline_agent_model",
    "protocol_violations",
    "replay_text_policy",
    "route_following_policy",
    "say",
    "select",
    "skip_retrieval_policy",
    "text_block",
    "thinking_block",
    "tool_result_claims",
    "tool_use_block",
    "wrong_tool_policy",
]
