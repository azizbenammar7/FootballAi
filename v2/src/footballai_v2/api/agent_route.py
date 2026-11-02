"""The development-only route to the bounded read-only agent.

``create_app`` imports this module only when the agent gate is open, so a default
deployment never loads the agent or its offline adapter. The run comes from the
path and nothing else: the request body holds a question, and the agent's tools
cannot name a run. No configuration reaches a paid model from here. A real adapter
needs an explicit ``model_factory``, which only M3 may add. See
docs/v2/ADR-006-agent-endpoint-and-trace.md.
"""

from __future__ import annotations

import logging
from typing import Callable, Collection

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import UUID4

from footballai_v2.copilot.agent import AgentAdapter, AgentLimits, AgentResponse, ReadOnlyAgent, publish_trace
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.evaluation.agent_models import offline_agent_model
from footballai_v2.copilot.models import ExplainRequest
from footballai_v2.copilot.provider import AgentModel
from footballai_v2.storage.ports import AnalysisRepository, ObjectStorage


logger = logging.getLogger("footballai_v2.api")

_LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1"})
_LOOPBACK_HOSTS = _LOOPBACK_PEERS | {"localhost"}


def _from_this_machine(request: Request, allowed_origins: Collection[str]) -> bool:
    """Whether a request came from a person on this machine, not through something else.

    The application has no authentication, and an environment variable cannot prove
    where a server is listening. The peer check refuses a server bound to a shared
    address, the Host check refuses tunnels and DNS rebinding, which arrive through a
    loopback peer, and the Origin check refuses other pages open in a local browser.
    """
    peer = request.client.host if request.client else None
    origin = request.headers.get("origin")
    return (
        peer in _LOOPBACK_PEERS
        and request.url.hostname in _LOOPBACK_HOSTS
        and (origin is None or origin in allowed_origins)
    )


def register_agent_route(
    app: FastAPI,
    repository: AnalysisRepository,
    object_storage: ObjectStorage,
    settings: CopilotSettings,
    require_completed: Callable[[UUID4], None],
    allowed_origins: Collection[str],
    model_factory: Callable[[], AgentModel] | None = None,
) -> None:
    build_model = model_factory or offline_agent_model
    limits = AgentLimits()

    def local_callers_only(request: Request) -> None:
        # The plain 404 of an unregistered route. A malformed body or a wrong method is
        # answered before this runs, which shows the route exists and reaches no run.
        if not _from_this_machine(request, allowed_origins):
            raise HTTPException(status_code=404, detail="Not Found")

    @app.post(
        "/api/v1/runs/{run_id}/agent",
        response_model=AgentResponse,
        dependencies=[Depends(local_callers_only)],
        summary="Ask the bounded read-only agent about one completed run",
    )
    def ask_agent(run_id: UUID4, body: ExplainRequest, request: Request) -> AgentResponse:
        require_completed(run_id)
        model = build_model()  # one adapter per request: nothing carries over between questions
        agent = ReadOnlyAgent(repository, object_storage, model, settings, limits=limits)
        result = agent.answer(str(run_id), body.question, correlation_id=request.state.request_id)
        logger.info("agent_trace %s", result.trace.model_dump_json())  # holds no question, answer, or tool output
        return AgentResponse(
            explanation=result.response,
            trace=publish_trace(result.trace, AgentAdapter.for_mode(model.mode), limits),
        )
