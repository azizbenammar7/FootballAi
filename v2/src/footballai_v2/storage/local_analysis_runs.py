"""Filesystem adapter for isolated V2 analysis-run outputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from footballai_v2.contracts.v1 import AnalysisRun, AnalysisRunStatus, ArtifactReference
from footballai_v2.contracts.v1.analysis_run import (
    ContractValidationError,
    validate_relative_artifact_path,
    validate_run_id,
)


class RunAlreadyExistsError(FileExistsError):
    """Raised when a run ID has already reserved an output namespace."""


class RunNotFoundError(FileNotFoundError):
    """Raised when an analysis-run namespace does not exist."""


class ManifestConflictError(RuntimeError):
    """Raised when a manifest update does not match the stored run identity."""


class LocalAnalysisRunStore:
    """Store each analysis run under one non-overlapping directory.

    Layout::

        <root>/<analysis-run-id>/
            manifest.json
            artifacts/...

    Artifact files use exclusive creation. A repeated write cannot silently
    replace either an artifact from the same run or one from another run.
    """

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def create(self, run: AnalysisRun) -> Path:
        """Atomically reserve a new run namespace and persist its manifest."""
        if run.status is not AnalysisRunStatus.QUEUED:
            raise ManifestConflictError("a run namespace must be created in queued state")
        run_dir = self.run_directory(run.analysis_run_id)
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise RunAlreadyExistsError(run.analysis_run_id) from exc
        (run_dir / "artifacts").mkdir()
        try:
            self._write_manifest(run, replace_existing=False)
        except Exception:
            # Only remove the empty namespace created by this call.
            self.manifest_path(run.analysis_run_id).unlink(missing_ok=True)
            (run_dir / "artifacts").rmdir()
            run_dir.rmdir()
            raise
        return run_dir

    def load(self, analysis_run_id: str) -> AnalysisRun:
        run_dir = self.run_directory(analysis_run_id)
        self._validate_run_directory(run_dir)
        manifest_path = run_dir / self.MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RunNotFoundError(analysis_run_id)
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ContractValidationError(f"invalid stored manifest for {analysis_run_id}") from exc
        run = AnalysisRun.from_dict(data)
        if run.analysis_run_id != analysis_run_id:
            raise ManifestConflictError("manifest run ID does not match its directory")
        return run

    def save(self, run: AnalysisRun) -> None:
        """Atomically replace a manifest without changing its run namespace."""
        current = self.load(run.analysis_run_id)
        if current.status.is_terminal:
            raise ManifestConflictError("terminal manifests are immutable")
        allowed = {
            AnalysisRunStatus.QUEUED: {
                AnalysisRunStatus.RUNNING,
                AnalysisRunStatus.CANCELLED,
            },
            AnalysisRunStatus.RUNNING: {
                AnalysisRunStatus.SUCCEEDED,
                AnalysisRunStatus.FAILED,
                AnalysisRunStatus.CANCELLED,
            },
        }
        if run.status not in allowed[current.status]:
            raise ManifestConflictError(
                f"cannot replace {current.status.value} manifest with {run.status.value}"
            )
        immutable_fields = (
            "contract_version",
            "analysis_run_id",
            "data_origin",
            "input",
            "code",
            "pipeline_version",
            "parameters",
            "models",
            "created_at",
        )
        if any(getattr(current, name) != getattr(run, name) for name in immutable_fields):
            raise ManifestConflictError("run provenance cannot change after namespace creation")
        if run.status is AnalysisRunStatus.SUCCEEDED:
            self._verify_outputs(run)
        self._write_manifest(run, replace_existing=True)

    def write_artifact(
        self,
        analysis_run_id: str,
        *,
        name: str,
        relative_path: str,
        content: bytes,
        media_type: str,
        schema_version: str | None = None,
    ) -> ArtifactReference:
        """Write bytes once and return their content-addressed reference."""
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        run_dir = self._require_run_directory(analysis_run_id)
        if self.load(analysis_run_id).status is not AnalysisRunStatus.RUNNING:
            raise ManifestConflictError("artifacts can only be written while a run is running")
        normalized = validate_relative_artifact_path(relative_path)
        destination = run_dir.joinpath(*Path(normalized).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._require_within_run(destination.parent.resolve(), run_dir)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_name, destination)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return ArtifactReference(
            name=name,
            relative_path=normalized,
            media_type=media_type,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            schema_version=schema_version,
        )

    def run_directory(self, analysis_run_id: str) -> Path:
        canonical = validate_run_id(analysis_run_id)
        return self.root / canonical

    def manifest_path(self, analysis_run_id: str) -> Path:
        return self.run_directory(analysis_run_id) / self.MANIFEST_NAME

    def artifact_path(self, analysis_run_id: str, relative_path: str) -> Path:
        run_dir = self._require_run_directory(analysis_run_id)
        normalized = validate_relative_artifact_path(relative_path)
        candidate = run_dir.joinpath(*Path(normalized).parts)
        self._require_within_run(candidate.resolve(), run_dir)
        return candidate

    def _require_run_directory(self, analysis_run_id: str) -> Path:
        run_dir = self.run_directory(analysis_run_id)
        self._validate_run_directory(run_dir)
        manifest_path = run_dir / self.MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RunNotFoundError(analysis_run_id)
        return run_dir

    def _validate_run_directory(self, run_dir: Path) -> None:
        if not run_dir.is_dir():
            raise RunNotFoundError(run_dir.name)
        if run_dir.is_symlink():
            raise ContractValidationError("analysis-run directory cannot be a symlink")
        self._require_within_run(run_dir.resolve(), self.root)

    def _verify_outputs(self, run: AnalysisRun) -> None:
        for output in run.outputs:
            path = self.artifact_path(run.analysis_run_id, output.relative_path)
            if not path.is_file() or path.is_symlink():
                raise ManifestConflictError(
                    f"output {output.name!r} is missing or is not a regular run artifact"
                )
            if path.stat().st_size != output.size_bytes:
                raise ManifestConflictError(f"output {output.name!r} size does not match its manifest")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != output.sha256:
                raise ManifestConflictError(f"output {output.name!r} hash does not match its manifest")

    def _write_manifest(self, run: AnalysisRun, *, replace_existing: bool) -> None:
        path = self.manifest_path(run.analysis_run_id)
        payload = (
            json.dumps(run.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if not replace_existing:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return
        fd, temp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def _require_within_run(candidate: Path, run_dir: Path) -> None:
        resolved_run = run_dir.resolve()
        try:
            candidate.resolve().relative_to(resolved_run)
        except ValueError as exc:
            raise ContractValidationError("path escapes its analysis-run namespace") from exc
