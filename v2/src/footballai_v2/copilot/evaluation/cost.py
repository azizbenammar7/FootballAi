"""Cost estimation from documented prices. Estimates are not billed usage.

Only the Anthropic Console's usage and cost reports state what was billed. An
estimate here says what a request should cost at list price; ``request_cost_usd``
says what reported token counts cost at list price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from footballai_v2.copilot.models import TokenUsage


PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICING_AS_OF = "2026-10-31"

# Haiku 5.5 bills every token class at five times its rate above this prompt size.
# The copilot's prompt budget keeps requests far below it, but reported usage decides.
_LONG_PROMPT_TOKENS = 100_000

# Assumptions for the offline estimate. Replace them with --preflight, which asks
# the free token-counting endpoint for exact input counts.
_CHARS_PER_TOKEN = 3.0  # JSON-heavy text tokenizes shorter than prose
_STRUCTURED_OUTPUT_PROMPT_TOKENS = 300  # the API adds an undocumented-size system prompt
_THINKING_TOKENS = 150  # low effort; billed as output and unknown until measured
# Upper bounds for budget enforcement. No token is shorter than one character.
_STRUCTURED_OUTPUT_PROMPT_TOKENS_MAX = 2_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float
    long_prompt_multiplier: float = 1.0


MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(
        input=0.10, output=0.50, cache_write=0.125, cache_read=0.01, long_prompt_multiplier=5.0
    ),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00, cache_write=6.25, cache_read=0.50),
}


def request_cost_usd(price: ModelPrice, usage: TokenUsage) -> float:
    """List-price cost of one request's reported tokens."""
    multiplier = price.long_prompt_multiplier if usage.prompt_tokens > _LONG_PROMPT_TOKENS else 1.0
    return multiplier * (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * price.cache_write
        + usage.cache_read_input_tokens * price.cache_read
    ) / 1_000_000


def estimate_input_tokens(prompt_chars: int) -> int:
    return math.ceil(prompt_chars / _CHARS_PER_TOKEN) + _STRUCTURED_OUTPUT_PROMPT_TOKENS


def estimate_output_tokens(selection_chars: int) -> int:
    return math.ceil(selection_chars / _CHARS_PER_TOKEN) + _THINKING_TOKENS


def worst_case_cost_usd(price: ModelPrice, prompt_chars: int, max_output_tokens: int) -> float:
    """An upper bound for one request: the amount budgets must reserve."""
    usage = TokenUsage(prompt_chars + _STRUCTURED_OUTPUT_PROMPT_TOKENS_MAX, max_output_tokens)
    return request_cost_usd(price, usage)


__all__ = [
    "MODEL_PRICES",
    "ModelPrice",
    "PRICING_AS_OF",
    "PRICING_SOURCE",
    "estimate_input_tokens",
    "estimate_output_tokens",
    "request_cost_usd",
    "worst_case_cost_usd",
]
