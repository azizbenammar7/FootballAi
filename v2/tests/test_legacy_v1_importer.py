"""Safe V1-to-V2 legacy artifact importer tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from footballai_v2.cli import import_legacy_v1 as cli
from footballai_v2.contracts.v1 import (
    AnalysisRunStatus,
    CodeReference,
    DataOrigin,
    StageName,
    StageStatus,
)
from footballai_v2.importers import LEGACY_QUALITY_WARNINGS, LegacyImportError, LegacyV1Importer
from footballai_v2.storage import LocalAnalysisRunStore, RunAlreadyExistsError


RUN_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    return {path.name: digest(path) for path in sorted(root.iterdir()) if path.is_file()}


@pytest.fixture
def legacy_source(tmp_path) -> Path:
    source = tmp_path / "legacy-source"
    source.mkdir()
    (source / "meta.json").write_text(
        json.dumps({"duration_s": 90.0, "effective_fps": 5.0}),
        encoding="utf-8",
    )
    (source / "player_summary.json").write_text(
        json.dumps(
            {
                "match_duration_s": 90.0,
                "total_players": 1,
                "players": {
                    "12": {
                        "track_id": 12,
                        "total_distance_m": 100.0,
                        "mean_speed_ms": 2.0,
                        "peak_speed_ms": 6.0,
                        "total_sprints": 1,
                        "active_time_s": 50.0,
                        "coverage_frac": 0.5,
                        "heatmap": [[0.2, 0.8]],
                        "block_metrics": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "risk_scores.json").write_text(
        json.dumps(
            {
                "12": {
                    "track_id": 12,
                    "risk_score": 0.4,
                    "risk_flag": "MEDIUM",
                    "fatigue_indicators": {},
                    "score_breakdown": {},
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "player_stats.parquet").write_bytes(b"PAR1player-statsPAR1")
    (source / "raw_tracks.parquet").write_bytes(b"PAR1raw-tracksPAR1")
    return source


def importer(output_root: Path) -> LegacyV1Importer:
    return LegacyV1Importer(
        LocalAnalysisRunStore(output_root),
        CodeReference("https://github.com/example/FootballAi", "8" * 40),
    )


def test_complete_import_is_succeeded_isolated_and_content_addressed(tmp_path, legacy_source):
    before = snapshot(legacy_source)
    output_root = tmp_path / "configured-runs"
    result = importer(output_root).import_directory(legacy_source, run_id=RUN_ID)
    store = LocalAnalysisRunStore(output_root)

    assert result.status is AnalysisRunStatus.SUCCEEDED
    assert result.data_origin is DataOrigin.LEGACY_V1
    assert result.input.uri.startswith("legacy-v1://artifact-set/")
    assert result.parameters["quality_warnings"] == list(LEGACY_QUALITY_WARNINGS)
    assert len(result.artifacts) == 6
    assert store.run_directory(result.run_id) == output_root.resolve() / result.run_id
    assert store.manifest_path(result.run_id).is_file()
    for item in result.artifacts:
        artifact_path = store.artifact_path(result.run_id, item.relative_path)
        assert digest(artifact_path) == item.sha256
        assert artifact_path.stat().st_size == item.size_bytes
    assert snapshot(legacy_source) == before


def test_legacy_risk_artifact_uses_v2_workload_advisory_terminology(tmp_path, legacy_source):
    result = importer(tmp_path / "runs").import_directory(legacy_source)
    advisory = next(item for item in result.artifacts if item.artifact_id == "workload-advisory")
    assert advisory.relative_path == "artifacts/workload_advisory.json"
    assert advisory.name == "Workload and Fatigue Advisory"
    assert advisory.category.value == "workload_advisory"


def test_import_stages_make_clear_that_pipeline_work_was_not_reexecuted(tmp_path, legacy_source):
    result = importer(tmp_path / "runs").import_directory(legacy_source)
    by_name = {item.stage_name: item for item in result.stages}
    assert by_name[StageName.VIDEO_VALIDATION].status is StageStatus.SKIPPED
    assert "tracking was not executed" in by_name[StageName.TRACKING].message
    assert "metrics were not recomputed" in by_name[StageName.METRICS].message


@pytest.mark.parametrize(
    ("missing_name", "stage_name"),
    [
        ("raw_tracks.parquet", StageName.TRACKING),
        ("player_stats.parquet", StageName.METRICS),
        ("risk_scores.json", StageName.WORKLOAD_ADVISORY),
    ],
)
def test_missing_completion_artifact_produces_partial_run(
    tmp_path, legacy_source, missing_name, stage_name
):
    (legacy_source / missing_name).unlink()
    result = importer(tmp_path / "runs").import_directory(legacy_source)
    affected = next(item for item in result.stages if item.stage_name is stage_name)
    assert result.status is AnalysisRunStatus.PARTIAL
    assert missing_name in result.partial_reason
    assert affected.required is True
    assert affected.status is StageStatus.PARTIAL


def test_missing_minimum_artifact_fails_before_creating_run(tmp_path, legacy_source):
    (legacy_source / "player_summary.json").unlink()
    output_root = tmp_path / "runs"
    with pytest.raises(LegacyImportError, match="missing required"):
        importer(output_root).import_directory(legacy_source)
    assert list(output_root.iterdir()) == []


def test_malformed_json_fails_safely_without_run_or_traceback(tmp_path, legacy_source):
    (legacy_source / "risk_scores.json").write_text("{not-json", encoding="utf-8")
    output_root = tmp_path / "runs"
    with pytest.raises(LegacyImportError, match="not valid") as captured:
        importer(output_root).import_directory(legacy_source)
    assert "Traceback" not in str(captured.value)
    assert list(output_root.iterdir()) == []


def test_invalid_parquet_envelope_fails_safely(tmp_path, legacy_source):
    (legacy_source / "player_stats.parquet").write_bytes(b"not-parquet")
    with pytest.raises(LegacyImportError, match="Parquet"):
        importer(tmp_path / "runs").import_directory(legacy_source)


def test_repeated_imports_create_distinct_runs(tmp_path, legacy_source):
    service = importer(tmp_path / "runs")
    first = service.import_directory(legacy_source)
    second = service.import_directory(legacy_source)
    assert first.run_id != second.run_id
    assert first.logical_analysis_id != second.logical_analysis_id
    assert service.store.run_directory(first.run_id).is_dir()
    assert service.store.run_directory(second.run_id).is_dir()


def test_provided_existing_run_id_is_never_overwritten(tmp_path, legacy_source):
    service = importer(tmp_path / "runs")
    first = service.import_directory(legacy_source, run_id=RUN_ID)
    before = service.store.manifest_path(first.run_id).read_bytes()
    with pytest.raises(RunAlreadyExistsError):
        service.import_directory(legacy_source, run_id=RUN_ID)
    assert service.store.manifest_path(first.run_id).read_bytes() == before


def test_cli_prints_run_and_manifest_without_raw_traceback(
    tmp_path, legacy_source, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli,
        "_local_code_reference",
        lambda _root: CodeReference("https://github.com/example/FootballAi", "8" * 40),
    )
    output_root = tmp_path / "runs"
    result = cli.main(
        ["--source", str(legacy_source), "--output-root", str(output_root), "--run-id", RUN_ID]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert f"run_id={RUN_ID}" in captured.out
    assert "status=succeeded" in captured.out
    assert "manifest=" in captured.out
    assert "Traceback" not in captured.err


def test_cli_expected_failure_is_nonzero_and_sanitized(
    tmp_path, legacy_source, monkeypatch, capsys
):
    (legacy_source / "meta.json").write_text("bad", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_local_code_reference",
        lambda _root: CodeReference("https://github.com/example/FootballAi", "8" * 40),
    )
    result = cli.main(
        ["--source", str(legacy_source), "--output-root", str(tmp_path / "runs")]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "Legacy import failed:" in captured.err
    assert "Traceback" not in captured.err


def test_committed_v1_artifacts_remain_byte_for_byte_unchanged(tmp_path):
    repository_root = Path(__file__).resolve().parents[2]
    source = repository_root / "data" / "processed"
    before = snapshot(source)
    result = importer(tmp_path / "runs").import_directory(source)
    assert result.status is AnalysisRunStatus.SUCCEEDED
    assert snapshot(source) == before
