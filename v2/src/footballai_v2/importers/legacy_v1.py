"""Read-only importer for committed V1 technical-test artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from footballai_v2.contracts.v1 import (
    AnalysisRun,
    AnalysisRunStatus,
    ArtifactCategory,
    CodeReference,
    DataOrigin,
    FailureDetail,
    InputReference,
    ModelReference,
    StageExecution,
    StageName,
    StageStatus,
    utc_now,
)
from footballai_v2.storage import LocalAnalysisRunStore, RunAlreadyExistsError


class LegacyImportError(RuntimeError):
    """Raised when legacy artifacts cannot be imported safely."""


LEGACY_QUALITY_WARNINGS: Final[tuple[str, ...]] = (
    "Track IDs are not verified player identities.",
    "Halftime stitching is not identity-safe.",
    "Pitch positions are not homography-calibrated.",
    "Camera motion may affect movement estimates.",
    "Active-time and coverage semantics are approximate.",
    "V1 HSR load uses total distance rather than calibrated high-speed running load.",
    "V1 risk values are heuristic and advisory only.",
    "Full-match execution provenance is incomplete.",
)


@dataclass(frozen=True, slots=True)
class LegacyArtifactSpec:
    source_name: str
    destination_name: str
    artifact_id: str
    display_name: str
    category: ArtifactCategory
    media_type: str
    schema_version: str
    completion_required: bool = True


LEGACY_ARTIFACTS: Final[tuple[LegacyArtifactSpec, ...]] = (
    LegacyArtifactSpec(
        "meta.json",
        "meta.json",
        "legacy-meta",
        "Legacy match metadata",
        ArtifactCategory.SUMMARY,
        "application/json",
        "footballai.legacy-meta/v1",
    ),
    LegacyArtifactSpec(
        "player_summary.json",
        "player_summary.json",
        "legacy-player-summary",
        "Legacy player summary",
        ArtifactCategory.SUMMARY,
        "application/json",
        "footballai.legacy-player-summary/v1",
    ),
    LegacyArtifactSpec(
        "risk_scores.json",
        "workload_advisory.json",
        "workload-advisory",
        "Workload and Fatigue Advisory",
        ArtifactCategory.WORKLOAD_ADVISORY,
        "application/json",
        "footballai.legacy-workload-advisory/v1",
    ),
    LegacyArtifactSpec(
        "player_stats.parquet",
        "player_stats.parquet",
        "legacy-player-stats",
        "Legacy player statistics",
        ArtifactCategory.METRICS,
        "application/vnd.apache.parquet",
        "footballai.legacy-player-stats/v1",
    ),
    LegacyArtifactSpec(
        "raw_tracks.parquet",
        "raw_tracks.parquet",
        "legacy-raw-tracks",
        "Legacy raw tracks",
        ArtifactCategory.TRACKS,
        "application/vnd.apache.parquet",
        "footballai.legacy-raw-tracks/v1",
    ),
)

_MINIMUM_REQUIRED: Final[frozenset[str]] = frozenset({"meta.json", "player_summary.json"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_set_checksum(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_json_file(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyImportError(f"{path.name} is not valid UTF-8 JSON") from exc


def _validate_parquet_envelope(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            first = handle.read(4)
            handle.seek(-4, 2)
            last = handle.read(4)
    except (OSError, ValueError) as exc:
        raise LegacyImportError(f"{path.name} is not a readable Parquet artifact") from exc
    if first != b"PAR1" or last != b"PAR1":
        raise LegacyImportError(f"{path.name} does not have a valid Parquet envelope")


class LegacyV1Importer:
    """Copy a validated V1 artifact set into a new isolated V2 run."""

    def __init__(self, store: LocalAnalysisRunStore, code: CodeReference) -> None:
        self.store = store
        self.code = code

    def import_directory(
        self,
        source: str | Path,
        *,
        run_id: str | None = None,
    ) -> AnalysisRun:
        source_root = Path(source).expanduser().resolve()
        if not source_root.is_dir() or source_root.is_symlink():
            raise LegacyImportError("legacy source must be a real directory")

        available: dict[str, tuple[LegacyArtifactSpec, Path]] = {}
        missing: list[str] = []
        for spec in LEGACY_ARTIFACTS:
            path = source_root / spec.source_name
            if path.is_symlink():
                raise LegacyImportError(f"{spec.source_name} must not be a symlink")
            if path.is_file():
                if path.suffix == ".json":
                    _validate_json_file(path)
                elif path.suffix == ".parquet":
                    _validate_parquet_envelope(path)
                available[spec.source_name] = (spec, path)
            else:
                missing.append(spec.source_name)

        missing_minimum = sorted(_MINIMUM_REQUIRED - set(available))
        if missing_minimum:
            raise LegacyImportError(
                f"missing required legacy artifacts: {', '.join(missing_minimum)}"
            )

        source_hashes = {name: _sha256(path) for name, (_, path) in available.items()}
        input_checksum = _artifact_set_checksum([path for _, path in available.values()])
        parameters = {
            "legacy_import": {
                "baseline_tag": "technical-test-v1.0",
                "imported_artifacts": sorted(available),
                "missing_artifacts": sorted(missing),
                "source_checksums": dict(sorted(source_hashes.items())),
            },
            "quality_warnings": list(LEGACY_QUALITY_WARNINGS),
            "advisory_label": "Workload and Fatigue Advisory",
        }
        queued = AnalysisRun.new(
            run_id=run_id,
            data_origin=DataOrigin.LEGACY_V1,
            input=InputReference(
                uri=f"legacy-v1://artifact-set/{input_checksum}",
                sha256=input_checksum,
                media_type="application/vnd.footballai.legacy-artifact-set",
            ),
            code=self.code,
            pipeline_version="legacy-import/1.0.0",
            parameters=parameters,
            models=[ModelReference("legacy-v1-detector", "ultralytics-yolov8")],
        )
        try:
            self.store.create(queued)
        except RunAlreadyExistsError:
            raise

        started_at = utc_now()
        running_stage = StageExecution(
            stage_id="ingestion-1",
            stage_name=StageName.INGESTION,
            required=True,
            status=StageStatus.RUNNING,
            progress_percent=0,
            attempt_number=1,
            started_at=started_at,
            performance_metrics={"input_count": len(available)},
            message="Copying validated legacy artifacts into an isolated V2 run.",
        )
        running = queued.start(started_at, stages=[running_stage])
        self.store.save(running)

        artifacts = []
        try:
            for spec, path in available.values():
                artifacts.append(
                    self.store.write_artifact(
                        running.run_id,
                        artifact_id=spec.artifact_id,
                        name=spec.display_name,
                        category=spec.category,
                        relative_path=f"artifacts/{spec.destination_name}",
                        content=path.read_bytes(),
                        media_type=spec.media_type,
                        schema_version=spec.schema_version,
                    )
                )
            warnings_content = (
                json.dumps(
                    {
                        "label": "Legacy V1 analysis",
                        "warnings": list(LEGACY_QUALITY_WARNINGS),
                        "advisory_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            artifacts.append(
                self.store.write_artifact(
                    running.run_id,
                    artifact_id="quality-warnings",
                    name="Legacy data quality warnings",
                    category=ArtifactCategory.SUMMARY,
                    relative_path="artifacts/quality_warnings.json",
                    content=warnings_content,
                    media_type="application/json",
                    schema_version="footballai.data-quality-warnings/v1",
                )
            )
        except Exception as exc:
            failed_at = utc_now()
            failed_stage = StageExecution(
                stage_id="ingestion-1",
                stage_name=StageName.INGESTION,
                required=True,
                status=StageStatus.FAILED,
                progress_percent=0,
                attempt_number=1,
                started_at=started_at,
                finished_at=failed_at,
                error=FailureDetail(
                    "legacy_copy_failed",
                    "Legacy artifacts could not be copied safely.",
                    True,
                    failed_at,
                    None,
                ),
                performance_metrics={"input_count": len(available), "output_count": len(artifacts)},
                message="The import ended without exposing internal diagnostics.",
            )
            failed = running.fail(
                FailureDetail(
                    "legacy_import_failed",
                    "The legacy import did not complete.",
                    True,
                    failed_at,
                    None,
                ),
                failed_at,
                stages=[failed_stage],
            )
            self.store.save(failed)
            raise LegacyImportError(f"legacy import failed for run {running.run_id}") from exc

        after_hashes = {name: _sha256(path) for name, (_, path) in available.items()}
        if after_hashes != source_hashes:
            raise LegacyImportError("legacy source changed during import")

        finished_at = utc_now()
        artifact_ids = {item.artifact_id for item in artifacts}
        present = set(available)
        stages = self._completed_stages(
            started_at,
            finished_at,
            present,
            artifact_ids,
            len(artifacts),
        )
        incomplete = sorted(
            spec.source_name
            for spec in LEGACY_ARTIFACTS
            if spec.completion_required and spec.source_name not in present
        )
        if incomplete:
            completed = running.complete_partial(
                artifacts,
                "Reviewable legacy artifacts were imported, but the source set was incomplete: "
                + ", ".join(incomplete),
                finished_at,
                stages=stages,
            )
        else:
            completed = running.succeed(artifacts, finished_at, stages=stages)
        self.store.save(completed)
        return completed

    @staticmethod
    def _completed_stages(
        started_at,
        finished_at,
        present: set[str],
        artifact_ids: set[str],
        output_count: int,
    ) -> list[StageExecution]:
        def record(
            stage_id: str,
            stage_name: StageName,
            status: StageStatus,
            *,
            required: bool,
            produced: tuple[str, ...] = (),
            message: str,
        ) -> StageExecution:
            return StageExecution(
                stage_id=stage_id,
                stage_name=stage_name,
                required=required,
                status=status,
                progress_percent=100 if status is StageStatus.SUCCEEDED else 0,
                attempt_number=1,
                started_at=started_at if status is StageStatus.SUCCEEDED else None,
                finished_at=finished_at,
                produced_artifact_ids=tuple(item for item in produced if item in artifact_ids),
                performance_metrics={
                    "duration_seconds": max((finished_at - started_at).total_seconds(), 0),
                    "output_count": len(produced),
                },
                message=message,
            )

        metrics_complete = "player_stats.parquet" in present
        advisory_complete = "risk_scores.json" in present
        tracking_complete = "raw_tracks.parquet" in present
        return [
            record(
                "ingestion-1",
                StageName.INGESTION,
                StageStatus.SUCCEEDED,
                required=True,
                produced=tuple(sorted(artifact_ids)),
                message="Validated artifacts were copied without changing the V1 source.",
            ),
            record(
                "video-validation-1",
                StageName.VIDEO_VALIDATION,
                StageStatus.SKIPPED,
                required=False,
                message="No video was processed; this run imports committed artifacts.",
            ),
            record(
                "tracking-1",
                StageName.TRACKING,
                StageStatus.SUCCEEDED if tracking_complete else StageStatus.PARTIAL,
                required=True,
                produced=("legacy-raw-tracks",) if tracking_complete else (),
                message=(
                    "Legacy tracks were copied; tracking was not executed."
                    if tracking_complete
                    else "The legacy raw-tracks artifact was unavailable."
                ),
            ),
            record(
                "metrics-1",
                StageName.METRICS,
                StageStatus.SUCCEEDED if metrics_complete else StageStatus.PARTIAL,
                required=True,
                produced=tuple(
                    item
                    for item in ("legacy-meta", "legacy-player-summary", "legacy-player-stats")
                    if item in artifact_ids
                ),
                message=(
                    "Legacy metric artifacts were copied; metrics were not recomputed."
                    if metrics_complete
                    else "Some legacy metric artifacts were unavailable."
                ),
            ),
            record(
                "workload-advisory-1",
                StageName.WORKLOAD_ADVISORY,
                StageStatus.SUCCEEDED if advisory_complete else StageStatus.PARTIAL,
                required=True,
                produced=("workload-advisory",) if advisory_complete else (),
                message=(
                    "Legacy heuristic values were imported as advisory-only indicators."
                    if advisory_complete
                    else "The legacy advisory artifact was unavailable."
                ),
            ),
            record(
                "artifact-publication-1",
                StageName.ARTIFACT_PUBLICATION,
                StageStatus.SUCCEEDED,
                required=True,
                produced=(),
                message=f"Registered and verified {output_count} isolated artifacts.",
            ),
        ]
