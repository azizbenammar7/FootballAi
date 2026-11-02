"""Apply the agent decision rule to a measured report. It cannot pass on anything else.

The rule is data: ``evaluation/copilot/agent-v2/decision-rule.json``. It is proposed,
not accepted, until a person records that it is. Only a report from a real model at
Anthropic's endpoint can receive a verdict. A replay or a local test double gets
``NOT_A_MODEL_EVALUATION``, whatever its numbers are.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RULE_SCHEMA = "footballai.copilot-agent-decision-rule/v1"
RULE_FILE = "decision-rule.json"
ANTHROPIC_ENDPOINT = "anthropic_api"


class RuleStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"


class Verdict(StrEnum):
    NOT_A_MODEL_EVALUATION = "NOT_A_MODEL_EVALUATION"
    RULE_NOT_ACCEPTED = "RULE_NOT_ACCEPTED"
    INCOMPLETE_RUN = "INCOMPLETE_RUN"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    FAIL = "FAIL"
    PASS = "PASS"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Criterion(_Strict):
    criterion_id: str = Field(pattern=r"^[a-z0-9_]+$")
    statement: str
    # Where the number is read from, in the report's dictionary of measurements.
    measurement: str
    kind: Literal["max_count", "min_rate", "max_rate", "max_seconds"]
    threshold: float = Field(ge=0)
    denominator: str
    min_n: int = Field(ge=0)
    origin: Literal["roadmap", "proposed_addition"]
    binding: bool = True  # False: reported, and never decides the verdict


class DecisionRule(_Strict):
    schema_id: str = Field(alias="schema", pattern=f"^{RULE_SCHEMA}$")
    rule_version: str
    status: RuleStatus
    accepted_by: str | None = None
    accepted_on: str | None = None
    applies_to_split: Literal["heldout"]
    criteria: tuple[Criterion, ...] = Field(min_length=1)


def load_rule(directory: Path) -> tuple[DecisionRule, str]:
    raw = (directory / RULE_FILE).read_bytes()
    return DecisionRule.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class CriterionResult:
    criterion_id: str
    status: Literal["pass", "fail", "insufficient_sample", "not_measured"]
    value: float | None
    n: int | None


def _reading(measurement: Any) -> tuple[float | None, int | None]:
    """A value and its denominator, from a count, a list of case ids, or a proportion."""
    if measurement is None:
        return None, None
    if isinstance(measurement, list):
        return float(len(measurement)), None
    if isinstance(measurement, dict):
        if "rate" in measurement:
            return measurement["rate"], measurement["n"]
        if "p95_seconds" in measurement:
            return measurement["p95_seconds"], measurement["n"]
    return float(measurement), None


def apply_criterion(criterion: Criterion, measurement: Any) -> CriterionResult:
    value, n = _reading(measurement)
    identifier = criterion.criterion_id
    if value is None:
        return CriterionResult(identifier, "not_measured", None, n)
    if criterion.kind == "max_count":
        return CriterionResult(identifier, "pass" if value <= criterion.threshold else "fail", value, n)
    if n is None or n < criterion.min_n:
        # A failure on a small sample is still a failure: only a pass needs the sample.
        failed = n and (value < criterion.threshold if criterion.kind == "min_rate" else value > criterion.threshold)
        return CriterionResult(identifier, "fail" if failed else "insufficient_sample", value, n)
    passed = value >= criterion.threshold if criterion.kind == "min_rate" else value <= criterion.threshold
    return CriterionResult(identifier, "pass" if passed else "fail", value, n)


def _overall(rule: DecisionRule, results: list[CriterionResult], completed: bool) -> Verdict:
    binding = {criterion.criterion_id for criterion in rule.criteria if criterion.binding}
    statuses = {result.status for result in results if result.criterion_id in binding}
    if "fail" in statuses:
        return Verdict.FAIL
    if not completed:
        return Verdict.INCOMPLETE_RUN
    if statuses & {"insufficient_sample", "not_measured"}:
        return Verdict.INSUFFICIENT_SAMPLE
    return Verdict.PASS


def evaluate_rule(
    rule: DecisionRule,
    measurements: dict[str, Any],
    *,
    mode: str,
    endpoint: str | None,
    split: str,
    completed: bool,
) -> dict[str, Any]:
    """The verdict and every criterion. Anything but a real model shows no criterion at all.

    A script's perfect score beside the words "tool_selection_accuracy: pass" would read as
    a result, so a report that measured no model has nothing here to misread.
    """
    if mode != "live" or endpoint != ANTHROPIC_ENDPOINT or split != rule.applies_to_split:
        return {
            "verdict": Verdict.NOT_A_MODEL_EVALUATION.value,
            "rule_version": rule.rule_version,
            "rule_status": rule.status.value,
            "criteria": [],
        }
    results = [apply_criterion(criterion, measurements.get(criterion.measurement)) for criterion in rule.criteria]
    if rule.status is not RuleStatus.ACCEPTED:
        verdict = Verdict.RULE_NOT_ACCEPTED
    else:
        verdict = _overall(rule, results, completed)
    return {
        "verdict": verdict.value,
        "rule_version": rule.rule_version,
        "rule_status": rule.status.value,
        "criteria": [
            {"criterion": item.criterion_id, "status": item.status, "value": item.value, "n": item.n}
            for item in results
        ],
    }


__all__ = [
    "ANTHROPIC_ENDPOINT",
    "Criterion",
    "CriterionResult",
    "DecisionRule",
    "RULE_FILE",
    "RuleStatus",
    "Verdict",
    "apply_criterion",
    "evaluate_rule",
    "load_rule",
]
