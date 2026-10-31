"""Offline evaluation of copilot claim selection.

Development tooling only: nothing in the API, worker, or ``footballai_v2.copilot``
imports this package. It runs the real ``ExplanationService`` over versioned,
hand-labelled cases from ``evaluation/copilot/``. See
docs/v2/COPILOT_LLM_EVALUATION.md.
"""
