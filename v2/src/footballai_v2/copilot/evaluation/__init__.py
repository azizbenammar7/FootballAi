"""Offline evaluation of copilot claim selection.

Development tooling only: the worker and ``footballai_v2.copilot`` never import this
package, and the API imports one module of it, ``agent_models``, only when the
local-only agent gate is open. It runs the real ``ExplanationService`` over
versioned, hand-labelled cases from ``evaluation/copilot/``. See
docs/v2/COPILOT_LLM_EVALUATION.md and docs/v2/ADR-006-agent-endpoint-and-trace.md.
"""
