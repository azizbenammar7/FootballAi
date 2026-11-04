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
RULE_SCHEMA_V2 = "footballai.copilot-agent-decision-rule/v2"
RULE_FILE = "decision-rule.json"  # rule v1: kept byte for byte, for the v1 baseline and comparison
RULE_FILE_V2 = "decision-rule-v2.json"  # rule v2: the one a paid run applies, once accepted
ANTHROPIC_ENDPOINT = "anthropic_api"


class RuleStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"


class Verdict(StrEnum):
    NOT_A_MODEL_EVALUATION = "NOT_A_MODEL_EVALUATION"
    RULE_NOT_ACCEPTED = "RULE_NOT_ACCEPTED"
    INCOMPLETE_RUN = "INCOMPLETE_RUN"
    NOT_EVALUABLE = "NOT_EVALUABLE"  # a binding criterion had nothing that could fail it
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
    schema_id: str = Field(alias="schema", pattern=f"^({RULE_SCHEMA}|{RULE_SCHEMA_V2})$")
    rule_version: str
    status: RuleStatus
    accepted_by: str | None = None
    accepted_on: str | None = None
    applies_to_split: Literal["heldout"]
    scorer_version: str | None = None  # rule v2 names the scorer whose numbers it reads
    supersedes: dict[str, str] | None = None
    criteria: tuple[Criterion, ...] = Field(min_length=1)


def load_rule(directory: Path, name: str = RULE_FILE) -> tuple[DecisionRule, str]:
    raw = (directory / name).read_bytes()
    return DecisionRule.model_validate_json(raw), hashlib.sha256(raw).hexdigest()


Status = Literal["pass", "fail", "insufficient_sample", "not_evaluable", "not_measured"]


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """One criterion. ``reason`` says why, so no status is ever read without its cause."""

    criterion_id: str
    status: Status
    value: float | None
    n: int | None
    reason: str = ""


def _reading(measurement: Any) -> tuple[float | None, int | None]:
    """A value and its denominator, from a count, a list of case ids, or a proportion.

    A count may state its opportunities as ``{"cases": [...], "n": k}``: ``n`` is how many
    cases could have been counted. A bare list (rule v1) states none.
    """
    if measurement is None:
        return None, None
    if isinstance(measurement, list):
        return float(len(measurement)), None
    if isinstance(measurement, dict):
        if "cases" in measurement:
            return float(len(measurement["cases"])), measurement["n"]
        if "rate" in measurement:
            return measurement["rate"], measurement["n"]
        if "p95_seconds" in measurement:
            return measurement["p95_seconds"], measurement["n"]
    return float(measurement), None


def _misses(criterion: Criterion, value: float) -> bool:
    if criterion.kind == "min_rate":
        return value < criterion.threshold
    return value > criterion.threshold  # max_count, max_rate, max_seconds


def apply_criterion(criterion: Criterion, measurement: Any) -> CriterionResult:
    """A failure on a small sample is still a failure: only a pass needs the sample.

    A measurement with no opportunities at all is ``not_evaluable``: nothing could have
    failed it, so it must never read as a pass.
    """
    value, n = _reading(measurement)
    identifier = criterion.criterion_id

    def result(status: Status, reason: str) -> CriterionResult:
        return CriterionResult(identifier, status, value, n, reason)

    if measurement is None:
        return result("not_measured", "measurement_missing")
    if n == 0:
        return result("not_evaluable", "no_opportunities")
    if value is None:
        return result("not_measured", "measurement_missing")
    if _misses(criterion, value):
        return result("fail", "threshold_missed")
    if criterion.kind == "max_count" and n is None:  # rule v1: a bare list, no sample needed
        return result("pass", "threshold_met")
    if n is None or n < criterion.min_n:
        return result("insufficient_sample", "below_min_n")
    return result("pass", "threshold_met")


def _overall(rule: DecisionRule, results: list[CriterionResult], completed: bool) -> Verdict:
    binding = {criterion.criterion_id for criterion in rule.criteria if criterion.binding}
    statuses = {result.status for result in results if result.criterion_id in binding}
    if "fail" in statuses:
        return Verdict.FAIL
    if not completed:
        return Verdict.INCOMPLETE_RUN
    if statuses & {"not_evaluable", "not_measured"}:
        return Verdict.NOT_EVALUABLE  # more data of the same kind cannot fix this
    if "insufficient_sample" in statuses:
        return Verdict.INSUFFICIENT_SAMPLE
    return Verdict.PASS


def diagnostic_criteria(
    rule: DecisionRule,
    measurements: dict[str, Any],
    *,
    binding: frozenset[str] | None = None,
    completed: bool = True,
) -> dict[str, Any]:
    """Every criterion applied to a scripted control, for testing the rule itself.

    Never a model verdict: ``evaluate_rule`` stays the only way to one. The verdict here is
    what an accepted rule *would* say, with ``binding`` naming the criteria that count.
    ``completed`` is False when the operator's spend cap stopped any question.
    """
    results = [apply_criterion(criterion, measurements.get(criterion.measurement)) for criterion in rule.criteria]
    chosen = binding if binding is not None else frozenset(c.criterion_id for c in rule.criteria if c.binding)
    as_bound = rule.model_copy(
        update={"criteria": tuple(c.model_copy(update={"binding": c.criterion_id in chosen}) for c in rule.criteria)}
    )
    by_id = {criterion.criterion_id: criterion for criterion in rule.criteria}
    return {
        "verdict_if_accepted": _overall(as_bound, results, completed=completed).value,
        "criteria": [
            {
                "criterion": item.criterion_id,
                "origin": by_id[item.criterion_id].origin,
                "binding": item.criterion_id in chosen,
                "status": item.status,
                "reason": item.reason,
                "value": item.value,
                "n": item.n,
            }
            for item in results
        ],
    }


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
            {
                "criterion": item.criterion_id,
                "status": item.status,
                "reason": item.reason,
                "value": item.value,
                "n": item.n,
            }
            for item in results
        ],
    }


__all__ = [
    "ANTHROPIC_ENDPOINT",
    "Criterion",
    "CriterionResult",
    "DecisionRule",
    "RULE_FILE",
    "RULE_FILE_V2",
    "RULE_SCHEMA_V2",
    "RuleStatus",
    "Verdict",
    "apply_criterion",
    "diagnostic_criteria",
    "evaluate_rule",
    "load_rule",
]
