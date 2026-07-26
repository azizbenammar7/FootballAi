"""Dependency-free implementation of ``footballai.analysis-run/v1``.

The contract is intentionally independent of FastAPI, a database, and a storage
provider. API, worker, PostgreSQL, and Azure adapters can therefore exchange
the same serialized record without importing one another.
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

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ContractValidationError(ValueError):
    """Raised when serialized data violates the public contract."""


class InvalidStatusTransition(ContractValidationError):
    """Raised when an analysis run attempts an invalid lifecycle transition."""


class DataOrigin(StrEnum):
    """How the input data was produced."""

    REAL = "real"
    SYNTHETIC = "synthetic"


class AnalysisRunStatus(StrEnum):
    """Durable lifecycle states for an analysis run."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            AnalysisRunStatus.SUCCEEDED,
            AnalysisRunStatus.FAILED,
            AnalysisRunStatus.CANCELLED,
        }


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


def _validate_json(value: Any, path: str = "parameters") -> None:
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


def validate_run_id(value: str) -> str:
    """Return the canonical UUID string used as an output namespace."""
    _require_non_empty(value, "analysis_run_id")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ContractValidationError("analysis_run_id must be a canonical UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ContractValidationError("analysis_run_id must use canonical lowercase UUID form")
    return canonical


def validate_relative_artifact_path(value: str) -> str:
    """Validate a storage-neutral, run-relative POSIX artifact path."""
    _require_non_empty(value, "relative_path")
    if "\\" in value:
        raise ContractValidationError("relative_path must use POSIX separators")
    path = PurePosixPath(value)
    if value != str(path):
        raise ContractValidationError("relative_path must be normalized")
    if path.is_absolute() or value.startswith("/") or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractValidationError("relative_path must be a normalized path within the run")
    if path.parts[0] != "artifacts":
        raise ContractValidationError("relative_path must be inside the run's artifacts/ directory")
    if len(path.parts) == 1:
        raise ContractValidationError("relative_path must name an artifact")
    return str(path)


@dataclass(frozen=True, slots=True)
class InputReference:
    """Immutable identity and location of a run input."""

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
    """Source revision that produced a run."""

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
    """Versioned model dependency used by an analysis."""

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
    """Content-addressed output relative to its analysis-run directory."""

    name: str
    relative_path: str
    media_type: str
    sha256: str
    size_bytes: int
    schema_version: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "artifact.name")
        validate_relative_artifact_path(self.relative_path)
        _require_non_empty(self.media_type, "artifact.media_type")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ContractValidationError("artifact.sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ContractValidationError("artifact.size_bytes must be a non-negative integer")
        if self.schema_version is not None:
            _require_non_empty(self.schema_version, "artifact.schema_version")

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "name": self.name,
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
            {"name", "relative_path", "media_type", "sha256", "size_bytes"},
            {"schema_version"},
        )
        return cls(
            name=data["name"],
            relative_path=data["relative_path"],
            media_type=data["media_type"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            schema_version=data.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class FailureDetail:
    """Sanitized terminal failure information safe to persist."""

    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        _require_non_empty(self.code, "failure.code")
        _require_non_empty(self.message, "failure.message")
        if not isinstance(self.retryable, bool):
            raise ContractValidationError("failure.retryable must be a boolean")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureDetail":
        _require_exact_keys(data, {"code", "message", "retryable"})
        return cls(code=data["code"], message=data["message"], retryable=data["retryable"])


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """Versioned provenance record and lifecycle state for one analysis."""

    analysis_run_id: str
    status: AnalysisRunStatus
    data_origin: DataOrigin
    input: InputReference
    code: CodeReference
    pipeline_version: str
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    models: tuple[ModelReference, ...] = ()
    outputs: tuple[ArtifactReference, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure: FailureDetail | None = None
    contract_version: str = ANALYSIS_RUN_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != ANALYSIS_RUN_CONTRACT_VERSION:
            raise ContractValidationError(
                f"unsupported contract_version {self.contract_version!r}; "
                f"expected {ANALYSIS_RUN_CONTRACT_VERSION!r}"
            )
        validate_run_id(self.analysis_run_id)
        if not isinstance(self.status, AnalysisRunStatus):
            raise ContractValidationError("status must be an AnalysisRunStatus")
        if not isinstance(self.data_origin, DataOrigin):
            raise ContractValidationError("data_origin must be a DataOrigin")
        _require_non_empty(self.pipeline_version, "pipeline_version")
        if not isinstance(self.parameters, Mapping):
            raise ContractValidationError("parameters must be an object")
        _validate_json(self.parameters)
        _require_aware_datetime(self.created_at, "created_at")
        if self.started_at is not None:
            _require_aware_datetime(self.started_at, "started_at")
            if self.started_at < self.created_at:
                raise ContractValidationError("started_at cannot precede created_at")
        if self.completed_at is not None:
            _require_aware_datetime(self.completed_at, "completed_at")
            if self.completed_at < (self.started_at or self.created_at):
                raise ContractValidationError("completed_at cannot precede the run start")
        self._validate_unique_dependencies()
        self._validate_lifecycle()

    def _validate_unique_dependencies(self) -> None:
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ContractValidationError("model names must be unique within an analysis run")
        output_names = [output.name for output in self.outputs]
        output_paths = [output.relative_path for output in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ContractValidationError("artifact names must be unique within an analysis run")
        if len(output_paths) != len(set(output_paths)):
            raise ContractValidationError("artifact relative paths must be unique within an analysis run")

    def _validate_lifecycle(self) -> None:
        if self.status is AnalysisRunStatus.QUEUED:
            if self.started_at or self.completed_at or self.failure or self.outputs:
                raise ContractValidationError("queued run cannot have execution or output fields")
        elif self.status is AnalysisRunStatus.RUNNING:
            if self.started_at is None or self.completed_at or self.failure or self.outputs:
                raise ContractValidationError("running run requires started_at and no terminal fields")
        elif self.status is AnalysisRunStatus.SUCCEEDED:
            if self.started_at is None or self.completed_at is None or self.failure is not None:
                raise ContractValidationError("succeeded run requires start/completion and no failure")
            if not self.outputs:
                raise ContractValidationError("succeeded run must reference at least one output")
        elif self.status is AnalysisRunStatus.FAILED:
            if self.started_at is None or self.completed_at is None or self.failure is None or self.outputs:
                raise ContractValidationError("failed run requires start/completion/failure and no outputs")
        elif self.status is AnalysisRunStatus.CANCELLED:
            if self.completed_at is None or self.failure is not None or self.outputs:
                raise ContractValidationError("cancelled run requires completion and no failure/outputs")

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
        analysis_run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> "AnalysisRun":
        """Create a queued run with a caller-supplied or generated UUID."""
        return cls(
            analysis_run_id=analysis_run_id or str(uuid.uuid4()),
            status=AnalysisRunStatus.QUEUED,
            data_origin=data_origin,
            input=input,
            code=code,
            pipeline_version=pipeline_version,
            parameters=dict(parameters or {}),
            models=tuple(models),
            created_at=created_at or utc_now(),
        )

    def start(self, at: datetime | None = None) -> "AnalysisRun":
        if self.status is not AnalysisRunStatus.QUEUED:
            raise InvalidStatusTransition(f"cannot start run in {self.status.value} state")
        return replace(self, status=AnalysisRunStatus.RUNNING, started_at=at or utc_now())

    def succeed(
        self,
        outputs: Sequence[ArtifactReference],
        at: datetime | None = None,
    ) -> "AnalysisRun":
        if self.status is not AnalysisRunStatus.RUNNING:
            raise InvalidStatusTransition(f"cannot succeed run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.SUCCEEDED,
            outputs=tuple(outputs),
            completed_at=at or utc_now(),
        )

    def fail(self, failure: FailureDetail, at: datetime | None = None) -> "AnalysisRun":
        if self.status is not AnalysisRunStatus.RUNNING:
            raise InvalidStatusTransition(f"cannot fail run in {self.status.value} state")
        return replace(
            self,
            status=AnalysisRunStatus.FAILED,
            failure=failure,
            completed_at=at or utc_now(),
        )

    def cancel(self, at: datetime | None = None) -> "AnalysisRun":
        if self.status not in {AnalysisRunStatus.QUEUED, AnalysisRunStatus.RUNNING}:
            raise InvalidStatusTransition(f"cannot cancel run in {self.status.value} state")
        return replace(self, status=AnalysisRunStatus.CANCELLED, completed_at=at or utc_now())

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "contract_version": self.contract_version,
            "analysis_run_id": self.analysis_run_id,
            "status": self.status.value,
            "data_origin": self.data_origin.value,
            "input": self.input.to_dict(),
            "code": self.code.to_dict(),
            "pipeline_version": self.pipeline_version,
            "parameters": dict(self.parameters),
            "models": [model.to_dict() for model in self.models],
            "outputs": [output.to_dict() for output in self.outputs],
            "created_at": _format_datetime(self.created_at),
        }
        if self.started_at is not None:
            result["started_at"] = _format_datetime(self.started_at)
        if self.completed_at is not None:
            result["completed_at"] = _format_datetime(self.completed_at)
        if self.failure is not None:
            result["failure"] = self.failure.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisRun":
        _require_exact_keys(
            data,
            {
                "contract_version",
                "analysis_run_id",
                "status",
                "data_origin",
                "input",
                "code",
                "pipeline_version",
                "parameters",
                "models",
                "outputs",
                "created_at",
            },
            {"started_at", "completed_at", "failure"},
        )
        try:
            status = AnalysisRunStatus(data["status"])
            data_origin = DataOrigin(data["data_origin"])
        except (ValueError, TypeError) as exc:
            raise ContractValidationError(f"invalid enum value: {exc}") from exc
        if not isinstance(data["models"], list) or not isinstance(data["outputs"], list):
            raise ContractValidationError("models and outputs must be arrays")
        return cls(
            contract_version=data["contract_version"],
            analysis_run_id=data["analysis_run_id"],
            status=status,
            data_origin=data_origin,
            input=InputReference.from_dict(data["input"]),
            code=CodeReference.from_dict(data["code"]),
            pipeline_version=data["pipeline_version"],
            parameters=data["parameters"],
            models=tuple(ModelReference.from_dict(item) for item in data["models"]),
            outputs=tuple(ArtifactReference.from_dict(item) for item in data["outputs"]),
            created_at=parse_utc_datetime(data["created_at"]),
            started_at=(
                parse_utc_datetime(data["started_at"]) if "started_at" in data else None
            ),
            completed_at=(
                parse_utc_datetime(data["completed_at"]) if "completed_at" in data else None
            ),
            failure=FailureDetail.from_dict(data["failure"]) if "failure" in data else None,
        )
