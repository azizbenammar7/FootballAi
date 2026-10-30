"""Read-only, strictly-validated run evidence for grounded explanations.

``RunEvidenceService`` turns one ``run_id`` into a bounded, allow-listed
``ValidatedRunEvidence`` snapshot built *only* from the provider-neutral ports:
the :class:`AnalysisRepository` for the manifest and the :class:`ObjectStorage`
for integrity-verified artifact bytes. It never reads arbitrary filesystem
paths, never issues SQL, never reimplements a storage adapter, and never exposes
raw input URIs, SAS URLs, credentials, or unapproved manifest text.

A real uploaded video does NOT imply real AI inference: a ``demo_fast`` run is
synthetic regardless of input origin, and that is surfaced explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from footballai_v2.contracts.v1 import AnalysisRun
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.storage.ports import AnalysisRepository, ObjectStorage


DIAGNOSTICS_ARTIFACT_ID = "analysis-diagnostics"
DIAGNOSTICS_MEDIA_TYPE = "application/json"
REAL_DIAGNOSTICS_SCHEMA = "footballai.analysis-diagnostics/v2"
SYNTHETIC_DIAGNOSTICS_SCHEMA = "footballai.analysis-diagnostics/v1"
APPROVED_DIAGNOSTICS_SCHEMAS = frozenset({REAL_DIAGNOSTICS_SCHEMA, SYNTHETIC_DIAGNOSTICS_SCHEMA})

# Pipeline profiles whose output is synthetic regardless of the uploaded input.
SYNTHETIC_PROFILES = frozenset({"demo_fast"})

_DISTRIBUTION_KEYS = ("min", "median", "mean", "p90", "p95", "max")
_ACTIVE_KEYS = ("median", "mean", "p95", "max")


class _NonFiniteNumber(ValueError):
    """Raised while parsing JSON that contains NaN/Infinity."""


def _reject_non_finite(_token: str) -> float:
    raise _NonFiniteNumber("artifact contains a non-finite number")


def _parse_finite_json(data: bytes) -> Any:
    """Parse JSON, rejecting NaN/Infinity so only finite numbers survive."""
    return json.loads(data.decode("utf-8"), parse_constant=_reject_non_finite)


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _text(value: Any, *, limit: int = 400) -> str | None:
    if isinstance(value, str):
        return value[:limit]
    return None


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    """Approved, integrity-stamped metadata for one registered artifact."""

    artifact_id: str
    name: str
    category: str
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str
    schema_version: str | None
    integrity_state: str  # "verified" | "invalid"

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "integrity_state": self.integrity_state,
        }


@dataclass(frozen=True, slots=True)
class ValidatedRunEvidence:
    """Typed, bounded evidence snapshot. The single source of truth for citations."""

    run_id: str
    status: str
    pipeline_profile: str
    pipeline_version: str
    data_origin: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    is_synthetic: bool
    synthetic_explanation: str
    source_video: Mapping[str, Any]
    models: tuple[Mapping[str, Any], ...]
    code: Mapping[str, Any]
    artifacts: tuple[EvidenceArtifact, ...]
    diagnostics: Mapping[str, Any] | None
    diagnostics_schema: str | None
    diagnostics_artifact_id: str | None
    diagnostics_sha256: str | None
    limitations: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    _field_paths: frozenset[str] = field(default_factory=frozenset, compare=False)

    def artifact(self, artifact_id: str) -> EvidenceArtifact | None:
        return next((item for item in self.artifacts if item.artifact_id == artifact_id), None)

    def to_snapshot(self) -> dict[str, Any]:
        """The bounded dict handed to the provider and resolved by citations."""
        snapshot: dict[str, Any] = {
            "run": {
                "run_id": self.run_id,
                "status": self.status,
                "pipeline_profile": self.pipeline_profile,
                "pipeline_version": self.pipeline_version,
                "data_origin": self.data_origin,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "output_nature": {
                "is_synthetic": self.is_synthetic,
                "explanation": self.synthetic_explanation,
            },
            "source_video": dict(self.source_video),
            "models": [dict(item) for item in self.models],
            "code": dict(self.code),
            "artifacts": {item.artifact_id: item.to_snapshot() for item in self.artifacts},
            "limitations": list(self.limitations),
            "missing_evidence": list(self.missing_evidence),
        }
        if self.diagnostics is not None:
            snapshot["diagnostics"] = dict(self.diagnostics)
        return snapshot

    def field_paths(self) -> frozenset[str]:
        return self._field_paths


def _flatten_paths(value: Any, prefix: str = "") -> set[str]:
    """Every resolvable dotted path (intermediate nodes and leaves) in a snapshot."""
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= _flatten_paths(child, path)
    return paths


def resolve_field_path(snapshot: Mapping[str, Any], path: str) -> bool:
    """Whether ``path`` resolves against the snapshot (dict segments only)."""
    current: Any = snapshot
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return False
        current = current[segment]
    return True


class RunEvidenceService:
    """Build a validated evidence snapshot for one run via the storage ports."""

    def __init__(
        self,
        repository: AnalysisRepository,
        object_storage: ObjectStorage,
        settings: CopilotSettings | None = None,
    ) -> None:
        self._repository = repository
        self._object_storage = object_storage
        self._settings = settings or CopilotSettings()

    def build(self, run_id: str) -> ValidatedRunEvidence:
        run = self._repository.load(run_id)
        return self.build_from_run(run)

    def build_from_run(self, run: AnalysisRun) -> ValidatedRunEvidence:
        profile = str(run.parameters.get("pipeline_profile", "")) or run.pipeline_version.split("/")[0]
        diagnostics, schema, diag_artifact_id, diag_sha256, missing = self._load_diagnostics(run)

        synthetic_by_data = bool(
            diagnostics is not None and diagnostics.get("data_quality") == "synthetic"
        )
        is_synthetic = profile in SYNTHETIC_PROFILES or synthetic_by_data
        synthetic_explanation = (
            "This run used the synthetic 'demo_fast' profile; its values are deterministic "
            "placeholders and no ML model was executed, regardless of the uploaded input."
            if is_synthetic
            else "This run used a real analysis profile; outputs derive from executed inference."
        )

        source_video: dict[str, Any] = {
            "sha256": run.input.sha256,
            "media_type": run.input.media_type,
        }
        duration = self._safe_duration(run)
        if duration is not None:
            source_video["duration_seconds"] = duration

        artifacts = tuple(
            EvidenceArtifact(
                artifact_id=item.artifact_id,
                name=item.name,
                category=item.category.value,
                relative_path=item.relative_path,
                media_type=item.media_type,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                schema_version=item.schema_version,
                integrity_state=(
                    "verified"
                    if self._object_storage.artifact_integrity(run.run_id, item.artifact_id)
                    else "invalid"
                ),
            )
            for item in run.artifacts
        )

        limitations = self._limitations(is_synthetic, schema, missing)

        evidence = ValidatedRunEvidence(
            run_id=run.run_id,
            status=run.status.value,
            pipeline_profile=profile,
            pipeline_version=run.pipeline_version,
            data_origin=run.data_origin.value,
            created_at=_iso(run.created_at),
            started_at=_iso(run.started_at),
            completed_at=_iso(run.completed_at),
            is_synthetic=is_synthetic,
            synthetic_explanation=synthetic_explanation,
            source_video=source_video,
            models=tuple(
                {k: v for k, v in item.to_dict().items()} for item in run.models
            ),
            code={
                "repository": run.code.repository,
                "revision": run.code.revision,
                "dirty": run.code.dirty,
            },
            artifacts=artifacts,
            diagnostics=diagnostics,
            diagnostics_schema=schema,
            diagnostics_artifact_id=diag_artifact_id,
            diagnostics_sha256=diag_sha256,
            limitations=limitations,
            missing_evidence=tuple(missing),
        )
        # Compute the authoritative citable-field set from the finished snapshot.
        object.__setattr__(evidence, "_field_paths", frozenset(_flatten_paths(evidence.to_snapshot())))
        return evidence

    # -- diagnostics artifact validation ------------------------------------

    def _load_diagnostics(
        self, run: AnalysisRun
    ) -> tuple[dict[str, Any] | None, str | None, str | None, str | None, list[str]]:
        """Strictly validate the registered diagnostics artifact.

        Returns ``(approved_fields, schema, artifact_id, sha256, missing)``. Any
        validation failure yields ``None`` diagnostics plus a safe missing-evidence
        reason; it never raises and never fabricates fields.
        """
        missing: list[str] = []
        reference = next(
            (item for item in run.artifacts if item.artifact_id == DIAGNOSTICS_ARTIFACT_ID),
            None,
        )
        if reference is None:
            missing.append("No analysis-diagnostics artifact is registered for this run.")
            return None, None, None, None, missing

        if reference.media_type != DIAGNOSTICS_MEDIA_TYPE:
            missing.append("The diagnostics artifact is not registered as JSON.")
            return None, None, None, None, missing

        if reference.size_bytes > self._settings.max_artifact_bytes:
            missing.append("The diagnostics artifact exceeds the evidence size limit and was not read.")
            return None, None, None, None, missing

        if not self._object_storage.artifact_integrity(run.run_id, reference.artifact_id):
            missing.append("The diagnostics artifact failed SHA-256 integrity verification.")
            return None, None, None, None, missing

        try:
            data = self._object_storage.read_artifact_bytes(
                run.run_id,
                reference.artifact_id,
                max_bytes=self._settings.max_artifact_bytes,
            )
        except Exception:  # bounded: oversize, integrity, or storage errors all fail safe
            missing.append("The diagnostics artifact could not be read safely.")
            return None, None, None, None, missing

        try:
            payload = _parse_finite_json(data)
        except (_NonFiniteNumber, ValueError, UnicodeDecodeError):
            missing.append("The diagnostics artifact is not valid finite JSON.")
            return None, None, None, None, missing

        if not isinstance(payload, Mapping):
            missing.append("The diagnostics artifact is not a JSON object.")
            return None, None, None, None, missing

        schema = payload.get("schema")
        if schema not in APPROVED_DIAGNOSTICS_SCHEMAS:
            missing.append("The diagnostics artifact uses an unrecognized schema.")
            return None, None, None, None, missing

        # Defense in depth: the registered schema_version must match the payload.
        if reference.schema_version is not None and reference.schema_version != schema:
            missing.append("The diagnostics artifact schema does not match its registration.")
            return None, None, None, None, missing

        if schema == REAL_DIAGNOSTICS_SCHEMA:
            approved = self._approved_real_diagnostics(payload)
        else:
            approved = self._approved_synthetic_diagnostics(payload)

        if approved is None:
            missing.append("The diagnostics artifact did not contain the expected fields.")
            return None, None, None, None, missing

        return approved, schema, reference.artifact_id, reference.sha256, missing

    def _approved_real_diagnostics(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        observed = payload.get("observed_metrics")
        if not isinstance(observed, Mapping):
            return None
        result: dict[str, Any] = {"schema": payload["schema"], "observed_metrics": {}}
        om = result["observed_metrics"]
        for key in (
            "total_detections",
            "total_tracklets",
            "tracklets_per_video_minute",
            "track_start_rate_per_minute",
            "track_end_rate_per_minute",
            "detection_assignment_fraction",
        ):
            value = _number(observed.get(key))
            if value is not None:
                om[key] = value
        dist = observed.get("tracklet_duration_seconds")
        if isinstance(dist, Mapping):
            om["tracklet_duration_seconds"] = {
                k: _number(dist.get(k)) for k in _DISTRIBUTION_KEYS if _number(dist.get(k)) is not None
            }
        opt = observed.get("observations_per_tracklet")
        if isinstance(opt, Mapping):
            om["observations_per_tracklet"] = {
                k: _number(opt.get(k)) for k in ("median", "mean") if _number(opt.get(k)) is not None
            }
        insufficient = observed.get("insufficient_tracklets")
        if isinstance(insufficient, Mapping):
            om["insufficient_tracklets"] = {
                "criterion": _text(insufficient.get("criterion")),
                "count": _number(insufficient.get("count")),
                "percentage": _number(insufficient.get("percentage")),
            }
        active = observed.get("active_simultaneous_tracklets")
        if isinstance(active, Mapping):
            om["active_simultaneous_tracklets"] = {
                k: _number(active.get(k)) for k in _ACTIVE_KEYS if _number(active.get(k)) is not None
            }
        short = observed.get("short_tracklets")
        if isinstance(short, Mapping):
            om["short_tracklets"] = {
                str(threshold): {
                    "count": _number(block.get("count")),
                    "percentage": _number(block.get("percentage")),
                }
                for threshold, block in short.items()
                if isinstance(block, Mapping)
            }
        trend = observed.get("fragmentation_trend")
        if isinstance(trend, list):
            om["fragmentation_trend_bins"] = len(trend)
        interpretation = payload.get("interpretation")
        if isinstance(interpretation, Mapping):
            result["interpretation"] = {
                "status": _text(interpretation.get("status")),
                "summary": _text(interpretation.get("summary"), limit=600),
            }
        ground_truth = payload.get("ground_truth_metrics")
        if isinstance(ground_truth, Mapping):
            result["ground_truth_metrics"] = {
                "status": _text(ground_truth.get("status")),
                "reason": _text(ground_truth.get("reason"), limit=600),
            }
        return result if om else None

    def _approved_synthetic_diagnostics(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        approved: dict[str, Any] = {"schema": payload["schema"]}
        for key in ("generator_profile", "data_quality", "message"):
            value = _text(payload.get(key), limit=600)
            if value is not None:
                approved[key] = value
        for key in ("output_count", "input_count", "deterministic_seed"):
            value = _number(payload.get(key))
            if value is not None:
                approved[key] = value
        return approved

    # -- helpers ------------------------------------------------------------

    def _safe_duration(self, run: AnalysisRun) -> float | None:
        probe = run.parameters.get("video_probe")
        if isinstance(probe, Mapping):
            value = _number(probe.get("duration_seconds"))
            if value is not None and value > 0:
                return float(value)
        return None

    def _limitations(
        self, is_synthetic: bool, schema: str | None, missing: list[str]
    ) -> tuple[str, ...]:
        limitations: list[str] = [
            "Track identifiers are temporary tracklets, not verified persistent player identities.",
            "No passes, shots, possession, formations, tactics, verified physical distances, or "
            "clinical injury risks are measured or claimed.",
        ]
        if is_synthetic:
            limitations.insert(
                0,
                "Output is synthetic: generated deterministically for workflow demonstration, "
                "not a measurement of the match.",
            )
        elif schema == REAL_DIAGNOSTICS_SCHEMA:
            limitations.append(
                "Tracking diagnostics are descriptive fragmentation proxies with no labelled "
                "ground-truth identity metric."
            )
        limitations.extend(missing)
        return tuple(limitations)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")
