"""Generated-schema drift and committed-example validation."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from footballai_v2.contracts.v1 import AnalysisRun
from footballai_v2.contracts.v1.schema import analysis_run_json_schema


CONTRACT_ROOT = Path(__file__).resolve().parents[1] / "contracts" / "analysis-run"
SCHEMA_PATH = CONTRACT_ROOT / "v1.schema.json"
EXAMPLE_ROOT = CONTRACT_ROOT / "examples"


def test_generated_schema_matches_committed_schema_byte_for_structure():
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == analysis_run_json_schema()


def test_schema_contains_all_approved_enums_and_fields():
    schema = analysis_run_json_schema()
    properties = schema["properties"]
    assert properties["status"]["enum"] == [
        "queued",
        "running",
        "succeeded",
        "partial",
        "failed",
        "cancelled",
    ]
    assert properties["data_origin"]["enum"] == [
        "real",
        "synthetic",
        "evaluation",
        "legacy_v1",
    ]
    assert {
        "logical_analysis_id",
        "run_id",
        "attempt_number",
        "previous_attempt_run_id",
        "stages",
        "artifacts",
    }.issubset(properties)
    stage = schema["$defs"]["stageExecution"]
    assert stage["properties"]["status"]["enum"] == [
        "queued",
        "running",
        "succeeded",
        "partial",
        "failed",
        "cancelled",
        "skipped",
    ]
    assert "workload_advisory" in schema["$defs"]["artifactReference"]["properties"][
        "category"
    ]["enum"]
    assert "workload_advisory" in stage["properties"]["stage_name"]["enum"]


def test_all_committed_examples_validate_against_schema_and_python_contract():
    validator = Draft202012Validator(
        analysis_run_json_schema(),
        format_checker=FormatChecker(),
    )
    examples = sorted(EXAMPLE_ROOT.glob("*.json"))
    assert [path.name for path in examples] == [
        "failed-first-attempt.json",
        "linked-retry.json",
        "partial-first-attempt.json",
        "queued-first-attempt.json",
        "succeeded-with-stages.json",
    ]
    for path in examples:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validator.validate(payload)
        restored = AnalysisRun.from_dict(payload)
        assert restored.to_dict() == payload


def test_linked_retry_example_preserves_logical_identity_and_input():
    failed = AnalysisRun.from_dict(
        json.loads((EXAMPLE_ROOT / "failed-first-attempt.json").read_text(encoding="utf-8"))
    )
    retry = AnalysisRun.from_dict(
        json.loads((EXAMPLE_ROOT / "linked-retry.json").read_text(encoding="utf-8"))
    )
    assert retry.logical_analysis_id == failed.logical_analysis_id
    assert retry.input == failed.input
    assert retry.data_origin == failed.data_origin
    assert retry.attempt_number == failed.attempt_number + 1
    assert retry.previous_attempt_run_id == failed.run_id
