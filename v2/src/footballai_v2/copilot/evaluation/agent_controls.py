"""Controlled policies for testing the decision rule. Scripts, never a language model.

Each control is a deliberate behaviour, run through the real ``ReadOnlyAgent``. A rule
that a bad control passes cannot tell a bad model from a good one. ``label_oracle`` is the
correct control: it does exactly what the labels ask, so it shows the labels are reachable.

Controls read a case's labels only to behave *badly* on purpose (``off_topic_only`` must
know what is on topic to avoid it). A real model never sees a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from footballai_v2.copilot.agent.loop import AgentLimits
from footballai_v2.copilot.agent.tools import ToolName
from footballai_v2.copilot.evaluation.agent_benchmark import BenchmarkCase, label_oracle_policy
from footballai_v2.copilot.evaluation.agent_models import (
    Step,
    _tools_for,
    call,
    every_tool_policy,
    route_following_policy,
    say,
    select,
    tool_result_claims,
)
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID
from footballai_v2.copilot.provider import AgentModelRequest


Policy = Callable[[AgentModelRequest], Step]
UNREGISTERED_TOOL = "execute_sql"
SLOW_TURN_SECONDS = 15.0  # three of these use the whole 45 second deadline


def _offered_tools(request: AgentModelRequest) -> list[str]:
    offered = set(request.offered_claim_ids)
    return _tools_for(offered - {OUTPUT_NATURE_CLAIM_ID} or offered)


def _turn(request: AgentModelRequest) -> int:
    return (len(request.messages) - 1) // 2  # model turns already taken


def _always_abstain(_case: BenchmarkCase) -> Policy:
    return lambda _request: select()


def _select_every_claim(_case: BenchmarkCase) -> Policy:
    def policy(request: AgentModelRequest) -> Step:
        if _turn(request):
            return select(*tool_result_claims(request.messages))
        return call(*_tools_for(set(request.offered_claim_ids)))  # the label's tool too

    return policy


def _off_topic_only(case: BenchmarkCase) -> Policy:
    on_topic = set(case.expected.permitted_claims) | {OUTPUT_NATURE_CLAIM_ID}

    def policy(request: AgentModelRequest) -> Step:
        if _turn(request):
            return select(*(claim for claim in tool_result_claims(request.messages) if claim not in on_topic))
        return call(*_offered_tools(request))

    return policy


def _right_tool_wrong_claims(case: BenchmarkCase) -> Policy:
    required = set(case.expected.required_claims) | {OUTPUT_NATURE_CLAIM_ID}

    def policy(request: AgentModelRequest) -> Step:
        if _turn(request):
            return select(*(claim for claim in tool_result_claims(request.messages) if claim not in required))
        tools = [tool.value for tool in case.expected.required_tools] or _offered_tools(request)
        return call(*tools)

    return policy


def _no_tool(case: BenchmarkCase) -> Policy:
    """Selects without retrieving: what it names is real, and still never fetched."""
    return lambda request: select(*(case.expected.required_claims or request.offered_claim_ids))


def _unauthorized_tool(_case: BenchmarkCase) -> Policy:
    return lambda request: select() if _turn(request) else call(UNREGISTERED_TOOL)


def _one_new_tool_per_turn(takes_seconds: float) -> Callable[[BenchmarkCase], Policy]:
    def policy_for(_case: BenchmarkCase) -> Policy:
        def policy(request: AgentModelRequest) -> Step:
            tools = list(ToolName)
            if _turn(request) >= len(tools):  # never under the limits; finish rather than fail oddly
                return select()
            return call(tools[_turn(request)].value, takes_seconds=takes_seconds)

        return policy

    return policy_for


def _malformed_output(_case: BenchmarkCase) -> Policy:
    return lambda _request: say("I would rather describe the run in my own words.")


@dataclass(frozen=True, slots=True)
class Control:
    name: str
    kind: Literal["correct_control", "deterministic_baseline", "negative_control"]
    description: str
    policy_for: Callable[[BenchmarkCase], Policy]
    limits: AgentLimits | None = None


CONTROLS: tuple[Control, ...] = (
    Control("label_oracle", "correct_control", "Exactly what the labels ask", label_oracle_policy),
    Control(
        "route_following", "deterministic_baseline", "Retrieve the offered tools, select all", lambda _c: route_following_policy
    ),
    Control("always_abstain", "negative_control", "Selects nothing, every time", _always_abstain),
    Control("select_every_claim", "negative_control", "Retrieves every offered tool, label included, selects all", _select_every_claim),
    Control("off_topic_only", "negative_control", "Selects only retrieved claims the case does not permit", _off_topic_only),
    Control("right_tool_wrong_claims", "negative_control", "Calls the required tools, selects what they return except the answer", _right_tool_wrong_claims),
    Control("no_tool", "negative_control", "Selects claims it never retrieved", _no_tool),
    Control("unauthorized_tool", "negative_control", f"Calls {UNREGISTERED_TOOL}, outside the registry", _unauthorized_tool),
    Control("exhaust_tool_budget", "negative_control", "Asks for all five tools at once, over the limit of four", lambda _c: every_tool_policy),
    Control(
        "exhaust_turn_budget",
        "negative_control",
        "One new tool per turn, under a two-turn limit",
        _one_new_tool_per_turn(0.0),
        AgentLimits(max_model_turns=2),
    ),
    Control(
        "exceed_deadline",
        "negative_control",
        f"One new tool per turn, {SLOW_TURN_SECONDS:g} s each on a scripted clock",
        _one_new_tool_per_turn(SLOW_TURN_SECONDS),
    ),
    Control("malformed_output", "negative_control", "Finishes with prose, not the selection", _malformed_output),
)


__all__ = ["CONTROLS", "Control", "SLOW_TURN_SECONDS", "UNREGISTERED_TOOL"]
