"""One bounded, read-only tool-calling agent (M2-B). See docs/v2/COPILOT_AGENT.md.

The worker never imports this package. The API imports it only when the agent gate
is open, from ``api.agent_route``, so a default deployment does not load it.
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
from footballai_v2.copilot.agent.trace import (
    AgentAdapter,
    AgentResponse,
    AgentStopReason,
    AgentTrace,
    PublicAgentTrace,
    UsageSource,
    publish_trace,
)

__all__ = [
    "AgentAdapter",
    "AgentLimits",
    "AgentResponse",
    "AgentResult",
    "AgentStopReason",
    "AgentTrace",
    "PublicAgentTrace",
    "READ_ONLY_TOOLS",
    "ReadOnlyAgent",
    "ToolCallStatus",
    "ToolErrorCode",
    "ToolName",
    "ToolRegistry",
    "UsageSource",
    "publish_trace",
    "tool_definitions",
]
