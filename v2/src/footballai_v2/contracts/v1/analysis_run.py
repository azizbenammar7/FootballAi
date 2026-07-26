"""Dependency-free implementation of ``footballai.analysis-run/v1``.

The contract is deliberately independent of web frameworks, databases, job
queues, and storage providers. It describes one immutable run attempt within a
logical analysis and can therefore be shared by local and future adapters.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


ANALYSIS_RUN_CONTRACT_VERSION = "footballai.analysis-run/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_UNSAFE_TEXT_RE = re.compile(
    r"traceback \(most recent call last\)|authorization\s*:|bearer\s+[a-z0-9._~-]+"
    r"|(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]",
    re.IGNORECASE,
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:password|passwd|authorization|api[_-]?key|access[_-]?token|secret)",
    re.IGNORECASE,
)

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ContractValidationError(ValueError):
    """Raised when serialized data violates the public contract."""


class InvalidStatusTransition(ContractValidationError):
    """Raised when an analysis run attempts an invalid lifecycle transition."""


class DataOrigin(StrEnum):
    """How the logical input data was produced."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    EVALUATION = "evaluation"
    LEGACY_V1 = "legacy_v1"


class AnalysisRunStatus(StrEnum):
    """Durable lifecycle states for one run attempt."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            AnalysisRunStatus.SUCCEEDED,
            AnalysisRunStatus.PARTIAL,
            AnalysisRunStatus.FAILED,
            AnalysisRunStatus.CANCELLED,
        }


class StageName(StrEnum):
    """Stable initial stage names published by contract v1."""

    INGESTION = "ingestion"
    VIDEO_VALIDATION = "video_validation"
    DETECTION = "detection"
    TRACKING = "tracking"
    IDENTITY_RESOLUTION = "identity_resolution"
    PITCH_CALIBRATION = "pitch_calibration"
    METRICS = "metrics"
    WORKLOAD_ADVISORY = "workload_advisory"
    ARTIFACT_PUBLICATION = "artifact_publication"


class StageStatus(StrEnum):
    """Lifecycle states for one stage record within an attempt."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self in {
            StageStatus.SUCCEEDED,
            StageStatus.PARTIAL,
            StageStatus.FAILED,
            StageStatus.CANCELLED,
            StageStatus.SKIPPED,
        }


class ArtifactCategory(StrEnum):
    """Stable broad categories for published artifacts."""

    SOURCE = "source"
    TRACKS = "tracks"
    METRICS = "metrics"
    SUMMARY = "summary"
    WORKLOAD_ADVISORY = "workload_advisory"
    VISUALIZATION = "visualization"
    OTHER = "other"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def parse_utc_datetime(value: str) -> datetime:
    """Parse an RFC 3339 timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value:
        raise ContractValidationError("timestamp must be a non-empty RFC 3339 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ContractValidationError(f"invalid RFC 3339 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractValidationError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    _require_aware_datetime(value, "timestamp")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_aware_datetime(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractValidationError(f"{name} must be a timezone-aware datetime")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be a non-empty string")


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ContractValidationError(
            f"{name} must start with a lowercase letter and contain only lowercase "
            "letters, numbers, dots, underscores, or hyphens"
        )


def _require_exact_keys(
    data: Mapping[str, Any],
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    if not isinstance(data, Mapping):
        raise ContractValidationError("contract value must be an object")
    missing = required - set(data)
    unknown = set(data) - required - (optional or set())
    if missing:
        raise ContractValidationError(f"missing required fields: {sorted(missing)}")
    if unknown:
        raise ContractValidationError(f"unknown fields: {sorted(unknown)}")


def _validate_json(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{path} contains a non-string object key")
            _validate_json(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json(child, f"{path}[{index}]")
        return
    raise ContractValidationError(f"{path} contains unsupported type {type(value).__name__}")


def _validate_safe_text(value: str, name: str) -> None:
    _require_non_empty(value, name)
    if _UNSAFE_TEXT_RE.search(value):
        raise ContractValidationError(f"{name} contains unsafe diagnostic or secret-like content")


def _validate_safe_details(value: Mapping[str, JsonValue], path: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} must be an object")
    _validate_json(value, path)
    for key, child in value.items():
        if _SENSITIVE_KEY_RE.search(key):
            raise ContractValidationError(f"{path} contains a secret-like key")
        _validate_safe_detail_value(child, f"{path}.{key}")


def _validate_safe_detail_value(value: JsonValue, path: str) -> None:
    if isinstance(value, str) and _UNSAFE_TEXT_RE.search(value):
        raise ContractValidationError(f"{path} contains unsafe diagnostic content")
    if isinstance(value, Mapping):
        _validate_safe_details(value, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_safe_detail_value(child, f"{path}[{index}]")


def validate_uuid_v4(value: str, name: str) -> str:
    """Return a canonical RFC 4122 UUID-v4 string."""
    _require_non_empty(value, name)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ContractValidationError(f"{name} must be a canonical UUID v4") from exc
    if value != str(parsed) or parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ContractValidationError(f"{name} must be a canonical lowercase UUID v4")
    return value


def validate_run_id(value: str) -> str:
    """Validate the UUID-v4 namespace identifier for one attempt."""
    return validate_uuid_v4(value, "run_id")


def validate_relative_artifact_path(value: str) -> str:
    """Validate a storage-neutral, run-relative POSIX artifact path."""
    _require_non_empty(value, "relative_path")
    if "\\" in value:
        raise ContractValidationError("relative_path must use POSIX separators")
    path = PurePosixPath(value)
    if value != str(path):
        raise ContractValidationError("relative_path must be normalized")
    if path.is_absolute() or value.startswith("/") or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ContractValidationError("relative_path must be a normalized path within the run")
    if path.parts[0] != "artifacts":
        raise ContractValidationError("relative_path must be inside the run's artifacts/ directory")
    if len(path.parts) == 1:
        raise ContractValidationError("relative_path must name an artifact")
    return str(path)


@dataclass(frozen=True, slots=True)
class InputReference:
    """Immutable identity and location of a logical analysis input."""

    uri: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        _require_non_empty(self.uri, "input.uri")
        _require_non_empty(self.media_type, "input.media_type")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ContractValidationError("input.sha256 must be 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"uri": self.uri, "sha256": self.sha256, "media_type": self.media_type}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputReference":
        _require_exact_keys(data, {"uri", "sha256", "media_type"})
        return cls(uri=data["uri"], sha256=data["sha256"], media_type=data["media_type"])


@dataclass(frozen=True, slots=True)
class CodeReference:
    """Source revision used for one attempt."""

    repository: str
    revision: str
    dirty: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.repository, "code.repository")
        if not isinstance(self.revision, str) or not _GIT_REVISION_RE.fullmatch(self.revision):
            raise ContractValidationError("code.revision must be a 40-64 character lowercase Git hash")
        if not isinstance(self.dirty, bool):
            raise ContractValidationError("code.dirty must be a boolean")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"repository": self.repository, "revision": self.revision, "dirty": self.dirty}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CodeReference":
        _require_exact_keys(data, {"repository", "revision", "dirty"})
        return cls(repository=data["repository"], revision=data["revision"], dirty=data["dirty"])


@dataclass(frozen=True, slots=True)
class ModelReference:
    """Versioned model dependency used by one attempt."""

    name: str
    version: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "model.name")
        _require_non_empty(self.version, "model.version")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256)
        ):
            raise ContractValidationError("model.sha256 must be 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"name": self.name, "version": self.version}
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelReference":
        _require_exact_keys(data, {"name", "version"}, {"sha256"})
        return cls(name=data["name"], version=data["version"], sha256=data.get("sha256"))


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Content-addressed artifact relative to its run-attempt directory."""

    artifact_id: str
    name: str
    category: ArtifactCategory
    relative_path: str
    media_type: str
    sha256: str
    size_bytes: int
    schema_version: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id, "artifact.artifact_id")
        _require_non_empty(self.name, "artifact.name")
        if not isinstance(self.category, ArtifactCategory):
            raise ContractValidationError("artifact.category must be an ArtifactCategory")
        validate_relative_artifact_path(self.relative_path)
        _require_non_empty(self.media_type, "artifact.media_type")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ContractValidationError("artifact.sha256 must be 64 lowercase hexadecimal characters")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ContractValidationError("artifact.size_bytes must be a non-negative integer")
        if self.schema_version is not None:
            _require_non_empty(self.schema_version, "artifact.schema_version")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "category": self.category.value,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.schema_version is not None:
            result["schema_version"] = self.schema_version
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactReference":
        _require_exact_keys(
            data,
            {
                "artifact_id",
                "name",
                "category",
                "relative_path",
                "media_type",
                "sha256",
                "size_bytes",
            },
            {"schema_version"},
        )
        try:
            category = ArtifactCategory(data["category"])
        except (ValueError, TypeError) as exc:
            raise ContractValidationError(f"invalid artifact category: {exc}") from exc
        return cls(
            artifact_id=data["artifact_id"],
            name=data["name"],
            category=category,
            relative_path=data["relative_path"],
            media_type=data["media_type"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            schema_version=data.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class StructuredError:
    """Sanitized execution error safe to persist and expose."""

    error_code: str
    safe_message: str
    retryable: bool
    occurred_at: datetime
    technical_details: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.error_code, "error.error_code")
        _validate_safe_text(self.safe_message, "error.safe_message")
        if not isinstance(self.retryable, bool):
            raise ContractValidationError("error.retryable must be a boolean")
        _require_aware_datetime(self.occurred_at, "error.occurred_at")
        if self.technical_details is not None:
            _validate_safe_details(self.technical_details, "error.technical_details")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "error_code": self.error_code,
            "safe_message": self.safe_message,
            "retryable": self.retryable,
            "occurred_at": _format_datetime(self.occurred_at),
            "technical_details": (
                dict(self.technical_details) if self.technical_details is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuredError":
        _require_exact_keys(
            data,
            {"error_code", "safe_message", "retryable", "occurred_at", "technical_details"},
        )
        details = data["technical_details"]
        if details is not None and not isinstance(details, Mapping):
            raise ContractValidationError("error.technical_details must be an object or null")
        return cls(
            error_code=data["error_code"],
            safe_message=data["safe_message"],
            retryable=data["retryable"],
            occurred_at=parse_utc_datetime(data["occurred_at"]),
            technical_details=details,
        )


# Public semantic aliases: run failures and stage errors share one safe wire shape.
FailureDetail = StructuredError
StageError = StructuredError


@dataclass(frozen=True, slots=True)
class StageExecution:
    """Execution and progress record for one stable stage in an attempt."""

    stage_id: str
    stage_name: StageName
    required: bool
    status: StageStatus
    progress_percent: float
    attempt_number: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    produced_artifact_ids: tuple[str, ...] = ()
    error: StructuredError | None = None
    performance_metrics: Mapping[str, JsonValue] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.stage_id, "stage.stage_id")
        if not isinstance(self.stage_name, StageName):
            raise ContractValidationError("stage.stage_name must be a StageName")
        if not isinstance(self.required, bool):
            raise ContractValidationError("stage.required must be a boolean")
        if not isinstance(self.status, StageStatus):
            raise ContractValidationError("stage.status must be a StageStatus")
        if (
            isinstance(self.progress_percent, bool)
            or not isinstance(self.progress_percent, (int, float))
            or not math.isfinite(float(self.progress_percent))
            or not 0 <= float(self.progress_percent) <= 100
        ):
            raise ContractValidationError("stage.progress_percent must be finite and between 0 and 100")
        if (
            not isinstance(self.attempt_number, int)
            or isinstance(self.attempt_number, bool)
            or self.attempt_number < 1
        ):
            raise ContractValidationError("stage.attempt_number must be an integer of at least 1")
        if self.started_at is not None:
            _require_aware_datetime(self.started_at, "stage.started_at")
        if self.finished_at is not None:
            _require_aware_datetime(self.finished_at, "stage.finished_at")
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ContractValidationError("stage.finished_at cannot precede stage.started_at")
        if self.status is StageStatus.QUEUED and (
            self.started_at is not None or self.finished_at is not None
        ):
            raise ContractValidationError("queued stage cannot have start or finish timestamps")
        if self.status is StageStatus.RUNNING:
            if self.started_at is None or self.finished_at is not None:
                raise ContractValidationError("running stage requires started_at and no finished_at")
        if self.status.is_terminal and self.finished_at is None:
            raise ContractValidationError("terminal stage requires finished_at")
        if self.status is StageStatus.SUCCEEDED and float(self.progress_percent) != 100:
            raise ContractValidationError("succeeded stage requires 100 percent progress")
        if self.status is StageStatus.FAILED and self.error is None:
            raise ContractValidationError("failed stage requires structured error information")
        if not isinstance(self.produced_artifact_ids, tuple):
            raise ContractValidationError("stage.produced_artifact_ids must be a tuple")
        if len(self.produced_artifact_ids) != len(set(self.produced_artifact_ids)):
            raise ContractValidationError("stage produced artifact IDs must be unique")
        for artifact_id in self.produced_artifact_ids:
            _require_identifier(artifact_id, "stage.produced_artifact_ids item")
        if not isinstance(self.performance_metrics, Mapping):
            raise ContractValidationError("stage.performance_metrics must be an object")
        _validate_json(self.performance_metrics, "stage.performance_metrics")
        if self.message is not None:
            _validate_safe_text(self.message, "stage.message")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name.value,
            "required": self.required,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "attempt_number": self.attempt_number,
            "started_at": _format_datetime(self.started_at) if self.started_at else None,
            "finished_at": _format_datetime(self.finished_at) if self.finished_at else None,
            "produced_artifact_ids": list(self.produced_artifact_ids),
            "error": self.error.to_dict() if self.error else None,
            "performance_metrics": dict(self.performance_metrics),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageExecution":
        _require_exact_keys(
            data,
            {
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
            },
        )
        try:
            stage_name = StageName(data["stage_name"])
            status = StageStatus(data["status"])
        except (ValueError, TypeError) as exc:
            raise ContractValidationError(f"invalid stage enum value: {exc}") from exc
        if not isinstance(data["produced_artifact_ids"], list):
            raise ContractValidationError("stage.produced_artifact_ids must be an array")
        if not isinstance(data["performance_metrics"], Mapping):
            raise ContractValidationError("stage.performance_metrics must be an object")
        return cls(
            stage_id=data["stage_id"],
            stage_name=stage_name,
            required=data["required"],
            status=status,
            progress_percent=data["progress_percent"],
            attempt_number=data["attempt_number"],
            started_at=(
                parse_utc_datetime(data["started_at"]) if data["started_at"] is not None else None
            ),
            finished_at=(
                parse_utc_datetime(data["finished_at"])
                if data["finished_at"] is not None
                else None
            ),
            produced_artifact_ids=tuple(data["produced_artifact_ids"]),
            error=StructuredError.from_dict(data["error"]) if data["error"] else None,
            performance_metrics=data["performance_metrics"],
            message=data["message"],
        )


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """Versioned manifest for one immutable attempt in a logical analysis."""

    logical_analysis_id: str
    run_id: str
    attempt_number: int
    previous_attempt_run_id: str | None
    status: AnalysisRunStatus
    data_origin: DataOrigin
    input: InputReference
    code: CodeReference
    pipeline_version: str
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    models: tuple[ModelReference, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    stages: tuple[StageExecution, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure: StructuredError | None = None
    partial_reason: str | None = None
    cancellation_reason: str | None = None
    contract_version: str = ANALYSIS_RUN_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != ANALYSIS_RUN_CONTRACT_VERSION:
            raise ContractValidationError(
                f"unsupported contract_version {self.contract_version!r}; "
                f"expected {ANALYSIS_RUN_CONTRACT_VERSION!r}"
            )
        validate_uuid_v4(self.logical_analysis_id, "logical_analysis_id")
        validate_run_id(self.run_id)
        if (
            not isinstance(self.attempt_number, int)
            or isinstance(self.attempt_number, bool)
            or self.attempt_number < 1
        ):
            raise ContractValidationError("attempt_number must be an integer of at least 1")
        if self.attempt_number == 1:
            if self.previous_attempt_run_id is not None:
                raise ContractValidationError("first attempt requires previous_attempt_run_id null")
        else:
            if self.previous_attempt_run_id is None:
                raise ContractValidationError("retry attempt requires previous_attempt_run_id")
            validate_run_id(self.previous_attempt_run_id)
        if self.previous_attempt_run_id == self.run_id:
            raise ContractValidationError("previous_attempt_run_id cannot equal run_id")
        if not isinstance(self.status, AnalysisRunStatus):
            raise ContractValidationError("status must be an AnalysisRunStatus")
        if not isinstance(self.data_origin, DataOrigin):
            raise ContractValidationError("data_origin must be a DataOrigin")
        _require_non_empty(self.pipeline_version, "pipeline_version")
        if not isinstance(self.parameters, Mapping):
            raise ContractValidationError("parameters must be an object")
        _validate_json(self.parameters, "parameters")
        _require_aware_datetime(self.created_at, "created_at")
        if self.started_at is not None:
            _require_aware_datetime(self.started_at, "started_at")
            if self.started_at < self.created_at:
                raise ContractValidationError("started_at cannot precede created_at")
        if self.completed_at is not None:
            _require_aware_datetime(self.completed_at, "completed_at")
            if self.completed_at < (self.started_at or self.created_at):
                raise ContractValidationError("completed_at cannot precede the run start")
        if self.partial_reason is not None:
            _validate_safe_text(self.partial_reason, "partial_reason")
        if self.cancellation_reason is not None:
            _validate_safe_text(self.cancellation_reason, "cancellation_reason")
        self._validate_unique_dependencies()
        self._validate_stage_artifact_links()
        self._validate_lifecycle()

    @property
    def analysis_run_id(self) -> str:
        """Compatibility accessor for early pre-merge adapter code."""
        return self.run_id

    @property
    def outputs(self) -> tuple[ArtifactReference, ...]:
        """Compatibility accessor; the published manifest field is ``artifacts``."""
        return self.artifacts

    def _validate_unique_dependencies(self) -> None:
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ContractValidationError("model names must be unique within an analysis run")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        artifact_names = [artifact.name for artifact in self.artifacts]
        artifact_paths = [artifact.relative_path for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ContractValidationError("artifact IDs must be unique within an analysis run")
        if len(artifact_names) != len(set(artifact_names)):
            raise ContractValidationError("artifact names must be unique within an analysis run")
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ContractValidationError("artifact relative paths must be unique within an analysis run")
        stage_ids = [stage.stage_id for stage in self.stages]
        stage_names = [stage.stage_name for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ContractValidationError("stage IDs must be unique within an analysis run")
        if len(stage_names) != len(set(stage_names)):
            raise ContractValidationError("stage names must be unique within an analysis run")

    def _validate_stage_artifact_links(self) -> None:
        artifact_ids = {artifact.artifact_id for artifact in self.artifacts}
        for stage in self.stages:
            if stage.attempt_number != self.attempt_number:
                raise ContractValidationError("stage attempt_number must match the run attempt_number")
            unknown = set(stage.produced_artifact_ids) - artifact_ids
            if unknown:
                raise ContractValidationError(
                    f"stage references unknown artifact IDs: {sorted(unknown)}"
                )

    def _validate_lifecycle(self) -> None:
        active_stages = {StageStatus.QUEUED, StageStatus.RUNNING}
        if self.status is AnalysisRunStatus.QUEUED:
            if (
                self.started_at
                or self.completed_at
                or self.failure
                or self.artifacts
                or self.partial_reason
                or self.cancellation_reason
            ):
                raise ContractValidationError("queued run cannot have execution or terminal fields")
            if any(stage.status is not StageStatus.QUEUED for stage in self.stages):
                raise ContractValidationError("queued run may contain only queued stage records")
        elif self.status is AnalysisRunStatus.RUNNING:
            if (
                self.started_at is None
                or self.completed_at
                or self.failure
                or self.artifacts
                or self.partial_reason
                or self.cancellation_reason
            ):
                raise ContractValidationError("running run requires started_at and no terminal fields")
        elif self.status is AnalysisRunStatus.SUCCEEDED:
            if (
                self.started_at is None
                or self.completed_at is None
                or self.failure is not None
                or self.partial_reason is not None
                or self.cancellation_reason is not None
            ):
                raise ContractValidationError("succeeded run requires start/completion and no failure reason")
            if not self.artifacts or not self.stages:
                raise ContractValidationError("succeeded run requires artifacts and terminal stage records")
            if any(stage.status in active_stages for stage in self.stages):
                raise ContractValidationError("succeeded run cannot contain active stages")
            if any(stage.required and stage.status is not StageStatus.SUCCEEDED for stage in self.stages):
                raise ContractValidationError("succeeded run requires every required stage to succeed")
        elif self.status is AnalysisRunStatus.PARTIAL:
            if (
                self.started_at is None
                or self.completed_at is None
                or self.failure is not None
                or self.partial_reason is None
                or self.cancellation_reason is not None
            ):
                raise ContractValidationError(
                    "partial run requires start/completion/safe reason and no run failure"
                )
            if not self.artifacts:
                raise ContractValidationError("partial run requires at least one useful artifact")
            if not self.stages or any(stage.status in active_stages for stage in self.stages):
                raise ContractValidationError("partial run requires terminal stage records")
            incomplete = {StageStatus.PARTIAL, StageStatus.FAILED, StageStatus.CANCELLED, StageStatus.SKIPPED}
            if not any(stage.required and stage.status in incomplete for stage in self.stages):
                raise ContractValidationError("partial run requires incomplete mandatory stage evidence")
        elif self.status is AnalysisRunStatus.FAILED:
            if (
                self.completed_at is None
                or self.failure is None
                or self.artifacts
                or self.partial_reason is not None
                or self.cancellation_reason is not None
            ):
                raise ContractValidationError("failed run requires completion/error and no artifacts")
            if any(stage.status is StageStatus.RUNNING for stage in self.stages):
                raise ContractValidationError("failed run cannot contain a running stage")
        elif self.status is AnalysisRunStatus.CANCELLED:
            if (
                self.completed_at is None
                or self.failure is not None
                or self.artifacts
                or self.partial_reason is not None
            ):
                raise ContractValidationError("cancelled run requires completion and no failure/artifacts")
            if any(stage.status is StageStatus.RUNNING for stage in self.stages):
                raise ContractValidationError("cancelled run cannot contain a running stage")

    @classmethod
    def new(
        cls,
        *,
        data_origin: DataOrigin,
        input: InputReference,
        code: CodeReference,
        pipeline_version: str,
        parameters: Mapping[str, JsonValue] | None = None,
        models: Sequence[ModelReference] = (),
        stages: Sequence[StageExecution] = (),
        logical_analysis_id: str | None = None,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> "AnalysisRun":
        """Create the first queued attempt for a logical analysis."""
        return cls(
            logical_analysis_id=logical_analysis_id or str(uuid.uuid4()),
            run_id=run_id or str(uuid.uuid4()),
            attempt_number=1,
            previous_attempt_run_id=None,
            status=AnalysisRunStatus.QUEUED,
            data_origin=data_origin,
            input=input,
            code=code,
            pipeline_version=pipeline_version,
            parameters=dict(parameters or {}),
            models=tuple(models),
            stages=tuple(stages),
            created_at=created_at or utc_now(),
        )

    @classmethod
    def retry_from(
        cls,
        previous: "AnalysisRun",
        *,
        code: CodeReference | None = None,
        pipeline_version: str | None = None,
        parameters: Mapping[str, JsonValue] | None = None,
        models: Sequence[ModelReference] | None = None,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> "AnalysisRun":
        """Create a queued, isolated retry from a failed or partial attempt."""
        if previous.status not in {AnalysisRunStatus.FAILED, AnalysisRunStatus.PARTIAL}:
            raise InvalidStatusTransition(
                f"cannot retry run in {previous.status.value} state"
            )
        return cls(
            logical_analysis_id=previous.logical_analysis_id,
            run_id=run_id or str(uuid.uuid4()),
            attempt_number=previous.attempt_number + 1,
            previous_attempt_run_id=previous.run_id,
            status=AnalysisRunStatus.QUEUED,
            data_origin=previous.data_origin,
            input=previous.input,
            code=code or previous.code,
            pipeline_version=pipeline_version or previous.pipeline_version,
            parameters=(dict(parameters) if parameters is not None else dict(previous.parameters)),
            models=(tuple(models) if models is not None else previous.models),
            created_at=created_at or utc_now(),
        )

    def start(
        self,
        at: datetime | None = None,
        *,
        stages: Sequence[StageExecution] | None = None,
    ) -> "AnalysisRun":
        if self.status is not AnalysisRunStatus.QUEUED:
            raise InvalidStatusTransition(f"cannot start run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.RUNNING,
            started_at=at or utc_now(),
            stages=tuple(stages) if stages is not None else self.stages,
        )

    def with_stages(self, stages: Sequence[StageExecution]) -> "AnalysisRun":
        if self.status.is_terminal:
            raise InvalidStatusTransition("terminal attempts cannot update stage records")
        return replace(self, stages=tuple(stages))

    def succeed(
        self,
        artifacts: Sequence[ArtifactReference],
        at: datetime | None = None,
        *,
        stages: Sequence[StageExecution],
    ) -> "AnalysisRun":
        if self.status is not AnalysisRunStatus.RUNNING:
            raise InvalidStatusTransition(f"cannot succeed run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.SUCCEEDED,
            artifacts=tuple(artifacts),
            stages=tuple(stages),
            completed_at=at or utc_now(),
        )

    def complete_partial(
        self,
        artifacts: Sequence[ArtifactReference],
        partial_reason: str,
        at: datetime | None = None,
        *,
        stages: Sequence[StageExecution],
    ) -> "AnalysisRun":
        if self.status is not AnalysisRunStatus.RUNNING:
            raise InvalidStatusTransition(f"cannot complete partial run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.PARTIAL,
            artifacts=tuple(artifacts),
            stages=tuple(stages),
            partial_reason=partial_reason,
            completed_at=at or utc_now(),
        )

    def fail(
        self,
        failure: StructuredError,
        at: datetime | None = None,
        *,
        stages: Sequence[StageExecution] | None = None,
    ) -> "AnalysisRun":
        if self.status not in {AnalysisRunStatus.QUEUED, AnalysisRunStatus.RUNNING}:
            raise InvalidStatusTransition(f"cannot fail run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.FAILED,
            failure=failure,
            stages=tuple(stages) if stages is not None else self.stages,
            completed_at=at or utc_now(),
        )

    def cancel(
        self,
        at: datetime | None = None,
        *,
        reason: str | None = None,
        stages: Sequence[StageExecution] | None = None,
    ) -> "AnalysisRun":
        if self.status not in {AnalysisRunStatus.QUEUED, AnalysisRunStatus.RUNNING}:
            raise InvalidStatusTransition(f"cannot cancel run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.CANCELLED,
            cancellation_reason=reason,
            stages=tuple(stages) if stages is not None else self.stages,
            completed_at=at or utc_now(),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "contract_version": self.contract_version,
            "logical_analysis_id": self.logical_analysis_id,
            "run_id": self.run_id,
            "attempt_number": self.attempt_number,
            "previous_attempt_run_id": self.previous_attempt_run_id,
            "status": self.status.value,
            "data_origin": self.data_origin.value,
            "input": self.input.to_dict(),
            "code": self.code.to_dict(),
            "pipeline_version": self.pipeline_version,
            "parameters": dict(self.parameters),
            "models": [model.to_dict() for model in self.models],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "stages": [stage.to_dict() for stage in self.stages],
            "created_at": _format_datetime(self.created_at),
        }
        if self.started_at is not None:
            result["started_at"] = _format_datetime(self.started_at)
        if self.completed_at is not None:
            result["completed_at"] = _format_datetime(self.completed_at)
        if self.failure is not None:
            result["failure"] = self.failure.to_dict()
        if self.partial_reason is not None:
            result["partial_reason"] = self.partial_reason
        if self.cancellation_reason is not None:
            result["cancellation_reason"] = self.cancellation_reason
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisRun":
        _require_exact_keys(
            data,
            {
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
            },
            {"started_at", "completed_at", "failure", "partial_reason", "cancellation_reason"},
        )
        try:
            status = AnalysisRunStatus(data["status"])
            data_origin = DataOrigin(data["data_origin"])
        except (ValueError, TypeError) as exc:
            raise ContractValidationError(f"invalid enum value: {exc}") from exc
        if not isinstance(data["models"], list):
            raise ContractValidationError("models must be an array")
        if not isinstance(data["artifacts"], list):
            raise ContractValidationError("artifacts must be an array")
        if not isinstance(data["stages"], list):
            raise ContractValidationError("stages must be an array")
        return cls(
            contract_version=data["contract_version"],
            logical_analysis_id=data["logical_analysis_id"],
            run_id=data["run_id"],
            attempt_number=data["attempt_number"],
            previous_attempt_run_id=data["previous_attempt_run_id"],
            status=status,
            data_origin=data_origin,
            input=InputReference.from_dict(data["input"]),
            code=CodeReference.from_dict(data["code"]),
            pipeline_version=data["pipeline_version"],
            parameters=data["parameters"],
            models=tuple(ModelReference.from_dict(item) for item in data["models"]),
            artifacts=tuple(ArtifactReference.from_dict(item) for item in data["artifacts"]),
            stages=tuple(StageExecution.from_dict(item) for item in data["stages"]),
            created_at=parse_utc_datetime(data["created_at"]),
            started_at=(
                parse_utc_datetime(data["started_at"]) if "started_at" in data else None
            ),
            completed_at=(
                parse_utc_datetime(data["completed_at"]) if "completed_at" in data else None
            ),
            failure=StructuredError.from_dict(data["failure"]) if "failure" in data else None,
            partial_reason=data.get("partial_reason"),
            cancellation_reason=data.get("cancellation_reason"),
        )
