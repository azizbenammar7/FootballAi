"""Typed request/response contract for the explanation endpoint."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CopilotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
