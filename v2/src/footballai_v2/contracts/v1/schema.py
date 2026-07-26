"""Generate the reviewable JSON Schema for the authoritative Python contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from footballai_v2.contracts.v1.analysis_run import (
    ANALYSIS_RUN_CONTRACT_VERSION,
    AnalysisRunStatus,
    ArtifactCategory,
    DataOrigin,
    StageName,
    StageStatus,
)


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def analysis_run_json_schema() -> dict[str, Any]:
    """Return JSON Schema draft 2020-12 for ``footballai.analysis-run/v1``."""
    uuid_v4 = {
        "type": "string",
        "format": "uuid",
        "pattern": (
            "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
            "[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        ),
    }
    identifier = {
        "type": "string",
        "pattern": "^[a-z][a-z0-9._-]{0,127}$",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://footballai.local/contracts/analysis-run/v1.schema.json",
        "title": "FootballAi analysis run v1",
        "description": (
            "Provenance, immutable attempt relationships, stage execution, and isolated "
            "artifact references for one V2 analysis-run attempt."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "contract_version",
            "logical_analysis_id",
            "run_id",
            "attempt_number",
            "previous_attempt_run_id",
            "status",
            "data_origin",
            "input",
            "code",
            "pipeline_version",
            "parameters",
            "models",
            "artifacts",
            "stages",
            "created_at",
        ],
        "properties": {
            "contract_version": {"const": ANALYSIS_RUN_CONTRACT_VERSION},
            "logical_analysis_id": {"$ref": "#/$defs/uuidV4"},
            "run_id": {"$ref": "#/$defs/uuidV4"},
            "attempt_number": {"type": "integer", "minimum": 1},
            "previous_attempt_run_id": {
                "oneOf": [{"$ref": "#/$defs/uuidV4"}, {"type": "null"}]
            },
            "status": {"enum": [item.value for item in AnalysisRunStatus]},
            "data_origin": {"enum": [item.value for item in DataOrigin]},
            "input": {"$ref": "#/$defs/inputReference"},
            "code": {"$ref": "#/$defs/codeReference"},
            "pipeline_version": _string(),
            "parameters": {"type": "object"},
            "models": {
                "type": "array",
                "items": {"$ref": "#/$defs/modelReference"},
            },
            "artifacts": {
                "type": "array",
                "items": {"$ref": "#/$defs/artifactReference"},
            },
            "stages": {
                "type": "array",
                "items": {"$ref": "#/$defs/stageExecution"},
            },
            "created_at": {"type": "string", "format": "date-time"},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "failure": {"$ref": "#/$defs/structuredError"},
            "partial_reason": _string(),
            "cancellation_reason": _string(),
        },
        "allOf": [
            {
                "if": {"properties": {"attempt_number": {"const": 1}}},
                "then": {"properties": {"previous_attempt_run_id": {"type": "null"}}},
                "else": {
                    "properties": {
                        "previous_attempt_run_id": {"$ref": "#/$defs/uuidV4"}
                    }
                },
            },
            {
                "if": {"properties": {"status": {"const": "queued"}}},
                "then": {
                    "properties": {"artifacts": {"maxItems": 0}},
                    "not": {
                        "anyOf": [
                            {"required": ["started_at"]},
                            {"required": ["completed_at"]},
                            {"required": ["failure"]},
                            {"required": ["partial_reason"]},
                            {"required": ["cancellation_reason"]},
                        ]
                    },
                },
            },
            {
                "if": {"properties": {"status": {"const": "running"}}},
                "then": {
                    "required": ["started_at"],
                    "properties": {"artifacts": {"maxItems": 0}},
                    "not": {
                        "anyOf": [
                            {"required": ["completed_at"]},
                            {"required": ["failure"]},
                            {"required": ["partial_reason"]},
                            {"required": ["cancellation_reason"]},
                        ]
                    },
                },
            },
            {
                "if": {"properties": {"status": {"const": "succeeded"}}},
                "then": {
                    "required": ["started_at", "completed_at"],
                    "properties": {"artifacts": {"minItems": 1}, "stages": {"minItems": 1}},
                    "not": {
                        "anyOf": [
                            {"required": ["failure"]},
                            {"required": ["partial_reason"]},
                            {"required": ["cancellation_reason"]},
                        ]
                    },
                },
            },
            {
                "if": {"properties": {"status": {"const": "partial"}}},
                "then": {
                    "required": ["started_at", "completed_at", "partial_reason"],
                    "properties": {"artifacts": {"minItems": 1}, "stages": {"minItems": 1}},
                    "not": {
                        "anyOf": [
                            {"required": ["failure"]},
                            {"required": ["cancellation_reason"]},
                        ]
                    },
                },
            },
            {
                "if": {"properties": {"status": {"const": "failed"}}},
                "then": {
                    "required": ["completed_at", "failure"],
                    "properties": {"artifacts": {"maxItems": 0}},
                    "not": {
                        "anyOf": [
                            {"required": ["partial_reason"]},
                            {"required": ["cancellation_reason"]},
                        ]
                    },
                },
            },
            {
                "if": {"properties": {"status": {"const": "cancelled"}}},
                "then": {
                    "required": ["completed_at"],
                    "properties": {"artifacts": {"maxItems": 0}},
                    "not": {
                        "anyOf": [
                            {"required": ["failure"]},
                            {"required": ["partial_reason"]},
                        ]
                    },
                },
            },
        ],
        "$defs": {
            "uuidV4": uuid_v4,
            "identifier": identifier,
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "inputReference": {
                "type": "object",
                "additionalProperties": False,
                "required": ["uri", "sha256", "media_type"],
                "properties": {
                    "uri": _string(),
                    "sha256": {"$ref": "#/$defs/sha256"},
                    "media_type": _string(),
                },
            },
            "codeReference": {
                "type": "object",
                "additionalProperties": False,
                "required": ["repository", "revision", "dirty"],
                "properties": {
                    "repository": _string(),
                    "revision": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{40,64}$",
                    },
                    "dirty": {"type": "boolean"},
                },
            },
            "modelReference": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "version"],
                "properties": {
                    "name": _string(),
                    "version": _string(),
                    "sha256": {"$ref": "#/$defs/sha256"},
                },
            },
            "artifactReference": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "artifact_id",
                    "name",
                    "category",
                    "relative_path",
                    "media_type",
                    "sha256",
                    "size_bytes",
                ],
                "properties": {
                    "artifact_id": {"$ref": "#/$defs/identifier"},
                    "name": _string(),
                    "category": {"enum": [item.value for item in ArtifactCategory]},
                    "relative_path": {
                        "type": "string",
                        "pattern": (
                            "^artifacts/(?!\\.{1,2}(?:/|$))[^/\\\\]+"
                            "(?:/(?!\\.{1,2}(?:/|$))[^/\\\\]+)*$"
                        ),
                    },
                    "media_type": _string(),
                    "sha256": {"$ref": "#/$defs/sha256"},
                    "size_bytes": {"type": "integer", "minimum": 0},
                    "schema_version": _string(),
                },
            },
            "structuredError": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "error_code",
                    "safe_message",
                    "retryable",
                    "occurred_at",
                    "technical_details",
                ],
                "properties": {
                    "error_code": {"$ref": "#/$defs/identifier"},
                    "safe_message": _string(),
                    "retryable": {"type": "boolean"},
                    "occurred_at": {"type": "string", "format": "date-time"},
                    "technical_details": {
                        "oneOf": [{"type": "object"}, {"type": "null"}]
                    },
                },
            },
            "stageExecution": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "stage_id",
                    "stage_name",
                    "required",
                    "status",
                    "progress_percent",
                    "attempt_number",
                    "started_at",
                    "finished_at",
                    "produced_artifact_ids",
                    "error",
                    "performance_metrics",
                    "message",
                ],
                "properties": {
                    "stage_id": {"$ref": "#/$defs/identifier"},
                    "stage_name": {"enum": [item.value for item in StageName]},
                    "required": {"type": "boolean"},
                    "status": {"enum": [item.value for item in StageStatus]},
                    "progress_percent": {"type": "number", "minimum": 0, "maximum": 100},
                    "attempt_number": {"type": "integer", "minimum": 1},
                    "started_at": {
                        "oneOf": [
                            {"type": "string", "format": "date-time"},
                            {"type": "null"},
                        ]
                    },
                    "finished_at": {
                        "oneOf": [
                            {"type": "string", "format": "date-time"},
                            {"type": "null"},
                        ]
                    },
                    "produced_artifact_ids": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"$ref": "#/$defs/identifier"},
                    },
                    "error": {
                        "oneOf": [
                            {"$ref": "#/$defs/structuredError"},
                            {"type": "null"},
                        ]
                    },
                    "performance_metrics": {"type": "object"},
                    "message": {"oneOf": [_string(), {"type": "null"}]},
                },
            },
        },
    }


def write_schema(path: str | Path) -> Path:
    """Write the generated schema deterministically and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(analysis_run_json_schema(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="schema output path")
    args = parser.parse_args()
    write_schema(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
