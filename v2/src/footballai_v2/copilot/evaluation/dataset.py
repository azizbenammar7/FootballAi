"""Typed loader for the versioned evaluation cases and fixtures.

Cases and fixtures are data, kept apart from the logic that scores them. A loaded
dataset carries the SHA-256 of both files, so a report names exactly what it
measured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID
from footballai_v2.copilot.models import OutputNature, SupportLevel


CASES_FILE = "cases.json"
FIXTURES_FILE = "fixtures.json"
CASES_SCHEMA = "footballai.copilot-eval-cases/v1"
FIXTURES_SCHEMA = "footballai.copilot-eval-fixtures/v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Suite(StrEnum):
    SELECTION = "selection"  # labelled questions: any selector, including a real model
    CONTRACT = "contract"  # scripted model replies: never sent to a real model


class Outcome(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    EITHER = "either"  # genuinely ambiguous: a permitted answer or abstaining is correct
    REJECT = "reject"  # a contract case: the reply must not produce an answer


class Expected(_Strict):
    outcome: Outcome
    output_nature: OutputNature
    reason: str | None = None
    support_level: SupportLevel | None = None
    required_claims: tuple[str, ...] = ()
    permitted_claims: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    forbidden_text: tuple[str, ...] = ()
    # Abstaining is the policy today, but the evidence could answer this question.
    coverage_gap: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "Expected":
        if not set(self.required_claims) <= set(self.permitted_claims) | {OUTPUT_NATURE_CLAIM_ID}:
            raise ValueError("required_claims must be permitted")
        if self.outcome is Outcome.ANSWER and not self.required_claims:
            raise ValueError("an answer case must name its required claims")
        if self.outcome in {Outcome.ABSTAIN, Outcome.EITHER, Outcome.REJECT} and self.reason is None:
            raise ValueError("this outcome must name its expected reason")
        if self.outcome is Outcome.ABSTAIN and (self.required_claims or self.permitted_claims):
            raise ValueError("an abstention permits no claims")
        return self

    @property
    def allowed_claims(self) -> frozenset[str]:
        """Claims that may appear; the provenance label is always rendered."""
        return frozenset(self.permitted_claims) | {OUTPUT_NATURE_CLAIM_ID}


class EvaluationCase(_Strict):
    case_id: str = Field(pattern=r"^[a-z0-9-]+$")
    suite: Suite
    category: str
    fixture: str
    question: str | None
    expected: Expected
    model_output: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _scripted_only_in_contract(self) -> "EvaluationCase":
        if (self.suite is Suite.CONTRACT) != (self.model_output is not None):
            raise ValueError("model_output is required for, and only for, contract cases")
        if (self.suite is Suite.CONTRACT) != (self.expected.outcome is Outcome.REJECT):
            raise ValueError("the reject outcome is required for, and only for, contract cases")
        return self


class DiagnosticsFixture(_Strict):
    schema_version: str
    payload: str | None = None
    raw_text: str | None = None
    overlay: dict[str, Any] | None = None
    integrity: bool = True

    @model_validator(mode="after")
    def _one_source(self) -> "DiagnosticsFixture":
        if (self.payload is None) == (self.raw_text is None):
            raise ValueError("diagnostics need exactly one of payload or raw_text")
        return self


class ModelFixture(_Strict):
    name: str
    version: str
    sha256: str | None = None


class RunFixture(_Strict):
    description: str
    profile: str
    data_origin: str
    diagnostics: DiagnosticsFixture | None
    models: tuple[ModelFixture, ...] = ()


class FixtureSet(_Strict):
    schema_id: str = Field(alias="schema", pattern=f"^{FIXTURES_SCHEMA}$")
    run_id: str
    note: str
    payloads: dict[str, dict[str, Any]]
    fixtures: dict[str, RunFixture]


class CaseSet(_Strict):
    schema_id: str = Field(alias="schema", pattern=f"^{CASES_SCHEMA}$")
    dataset_version: str
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    version: str
    cases: tuple[EvaluationCase, ...]
    fixtures: FixtureSet
    cases_sha256: str
    fixtures_sha256: str

    def suite(self, suite: Suite) -> tuple[EvaluationCase, ...]:
        return tuple(case for case in self.cases if case.suite is suite)


def load_dataset(directory: Path) -> EvaluationDataset:
    """Load and cross-check one dataset version; raises ``ValueError`` if invalid."""
    cases_bytes = (directory / CASES_FILE).read_bytes()
    fixtures_bytes = (directory / FIXTURES_FILE).read_bytes()
    case_set = CaseSet.model_validate_json(cases_bytes)
    fixture_set = FixtureSet.model_validate_json(fixtures_bytes)

    identifiers = [case.case_id for case in case_set.cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case_id values must be unique")
    for case in case_set.cases:
        if case.fixture not in fixture_set.fixtures:
            raise ValueError(f"case {case.case_id} references unknown fixture {case.fixture!r}")
    for name, fixture in fixture_set.fixtures.items():
        reference = fixture.diagnostics.payload if fixture.diagnostics else None
        if reference is not None and reference not in fixture_set.payloads:
            raise ValueError(f"fixture {name} references unknown payload {reference!r}")

    return EvaluationDataset(
        version=case_set.dataset_version,
        cases=case_set.cases,
        fixtures=fixture_set,
        cases_sha256=hashlib.sha256(cases_bytes).hexdigest(),
        fixtures_sha256=hashlib.sha256(fixtures_bytes).hexdigest(),
    )


__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "Expected",
    "FixtureSet",
    "Outcome",
    "RunFixture",
    "Suite",
    "load_dataset",
]
