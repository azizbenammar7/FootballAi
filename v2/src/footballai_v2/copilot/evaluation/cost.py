"""Offline cost estimates for planning. Estimates are not billed usage.

Prices and upper bounds live in ``copilot.pricing``, because the runtime cost guard
needs them too. What stays here is the guesswork that only a plan may use.
"""

from __future__ import annotations

import math

from footballai_v2.copilot.pricing import (
    MODEL_PRICES,
    PRICING_SOURCE,
    ModelPrice,
    agent_question_worst_case_usd,
    request_cost_usd,
    worst_case_cost_usd,
)


PRICING_AS_OF = "2026-10-31"  # recorded in the v1, v2, and agent-v1 baselines, so it stays

# Assumptions for the offline estimate. Replace them with --preflight, which asks
# the free token-counting endpoint for exact input counts.
_CHARS_PER_TOKEN = 3.0  # JSON-heavy text tokenizes shorter than prose
_STRUCTURED_OUTPUT_PROMPT_TOKENS = 300  # the API adds an undocumented-size system prompt
_THINKING_TOKENS = 150  # low effort; billed as output and unknown until measured


def estimate_input_tokens(prompt_chars: int) -> int:
    return math.ceil(prompt_chars / _CHARS_PER_TOKEN) + _STRUCTURED_OUTPUT_PROMPT_TOKENS


def estimate_output_tokens(selection_chars: int) -> int:
    return math.ceil(selection_chars / _CHARS_PER_TOKEN) + _THINKING_TOKENS


__all__ = [
    "MODEL_PRICES",
    "ModelPrice",
    "PRICING_AS_OF",
    "PRICING_SOURCE",
    "agent_question_worst_case_usd",
    "estimate_input_tokens",
    "estimate_output_tokens",
    "request_cost_usd",
    "worst_case_cost_usd",
]
