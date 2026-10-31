"""Typed request/response contract for the explanation endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class CopilotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Far above any catalog; a longer list is not a selection.
_MAX_SELECTED_CLAIMS = 64


class ClaimSelection(CopilotModel):
    """The only reply a model may give: approved claim ids, most relevant first.

    Structured outputs cannot express ``uniqueItems``, so duplicates are rejected
    here. Whether an id is approved is decided later, against the eligible claims.
    """

    claim_ids: list[StrictStr] = Field(max_length=_MAX_SELECTED_CLAIMS)

    @field_validator("claim_ids")
    @classmethod
    def _reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("claim_ids must not repeat an id")
        return value


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens reported by the provider for one request."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        # ``input_tokens`` excludes cached tokens, which are billed separately.
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


@dataclass(frozen=True, slots=True)
class ProviderTelemetry:
    """Operational facts about one provider request. Never shown to API clients.

    ``usage`` is ``None`` when the provider reported none, or reported it
    malformed; cost accounting must then assume the worst case.
    """

    model: str
    latency_seconds: float
    request_id: str | None = None
    stop_reason: str | None = None
    usage: TokenUsage | None = None


class SupportLevel(StrEnum):
    """How well the answer is backed by verified evidence."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ModelMode(StrEnum):
    """Which selector chose the claims. Never mislabel the fake as real."""

    OFFLINE_DEMO_FAKE = "offline_demo_fake"
    REAL_MODEL = "real_model"
    NOT_INVOKED = "not_invoked"  # deterministic abstention: no provider ran


class OutputNature(StrEnum):
    """Provenance of the run's output, derived only from evidence.

    ``VERIFIED_REAL`` is asserted *only* when integrity-verified real diagnostics
    corroborate that inference executed. A real uploaded video or a non-demo
    profile is never, by itself, enough: that case is ``UNVERIFIED_OR_PARTIAL``.
    """

    SYNTHETIC = "synthetic"
    VERIFIED_REAL = "verified_real"
    UNVERIFIED_OR_PARTIAL = "unverified_or_partial"


class ExplainRequest(CopilotModel):
    """Optional free-text question about the selected run."""

    question: str | None = Field(default=None, max_length=500)


class Citation(CopilotModel):
    """A pointer into the approved evidence snapshot for one factual claim."""

    run_id: str
    field_path: str = Field(min_length=1, max_length=200)
    artifact_id: str | None = None
    schema_version: str | None = None
    source_sha256: str | None = None


class ExplanationResponse(CopilotModel):
    """The grounded explanation returned to the dashboard."""

    run_id: str
    answer: str
    support_level: SupportLevel
    output_nature: OutputNature
    model_mode: ModelMode
    citations: list[Citation]
    limitations: list[str]
    unavailable_reason: str | None = None
