"""List prices and upper bounds. Estimates are not billed usage.

Only the Anthropic Console states what was billed. ``request_cost_usd`` says what
reported tokens cost at list price, and ``worst_case_cost_usd`` is the most a request
can cost, which a budget must reserve before sending it. This module is used at
runtime by the cost guard, so it imports nothing from ``copilot.evaluation``.
"""

from __future__ import annotations

from dataclasses import dataclass

from footballai_v2.copilot.models import TokenUsage


PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICING_VERIFIED_ON = "2026-11-02"  # the agent's prices were last read from the page on this date

# Haiku 5.5 bills every token class at five times its rate above this prompt size.
# The copilot's prompt budget keeps requests far below it, but reported usage decides.
_LONG_PROMPT_TOKENS = 100_000

# The API adds its own system prompts. Tool use is documented at 286 tokens for these
# models. The structured-output prompt's size is not, so this is a generous allowance.
_API_PROMPT_TOKENS_MAX = 2_000


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


def worst_case_cost_usd(price: ModelPrice, prompt_size: int, max_output_tokens: int) -> float:
    """An upper bound for one request. ``prompt_size`` is in characters or bytes."""
    usage = TokenUsage(prompt_size + _API_PROMPT_TOKENS_MAX, max_output_tokens)
    return request_cost_usd(price, usage)


def agent_question_worst_case_usd(
    price: ModelPrice, max_prompt_chars: int, max_output_tokens: int, model_turns: int
) -> float:
    """An upper bound for one agent question, which the loop's limits enforce.

    Every turn sends the whole prompt budget, and each later turn also resends all
    the output of the turns before it, thinking included.
    """
    return sum(
        worst_case_cost_usd(price, max_prompt_chars + turn * max_output_tokens, max_output_tokens)
        for turn in range(model_turns)
    )


__all__ = [
    "MODEL_PRICES",
    "ModelPrice",
    "PRICING_SOURCE",
    "PRICING_VERIFIED_ON",
    "agent_question_worst_case_usd",
    "request_cost_usd",
    "worst_case_cost_usd",
]
