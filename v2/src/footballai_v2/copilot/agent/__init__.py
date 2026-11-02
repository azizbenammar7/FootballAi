"""One bounded, read-only tool-calling agent (M2-B). See docs/v2/COPILOT_AGENT.md.

This package is not imported by the API or worker: no endpoint exposes it yet.
"""

from footballai_v2.copilot.agent.loop import AgentLimits, AgentResult, ReadOnlyAgent
from footballai_v2.copilot.agent.tools import (
    READ_ONLY_TOOLS,
    ToolCallStatus,
    ToolErrorCode,
    ToolName,
    ToolRegistry,
    tool_definitions,
)
from footballai_v2.copilot.agent.trace import AgentStopReason, AgentTrace

__all__ = [
    "AgentLimits",
    "AgentResult",
    "AgentStopReason",
    "AgentTrace",
    "READ_ONLY_TOOLS",
    "ReadOnlyAgent",
    "ToolCallStatus",
    "ToolErrorCode",
    "ToolName",
    "ToolRegistry",
    "tool_definitions",
]
