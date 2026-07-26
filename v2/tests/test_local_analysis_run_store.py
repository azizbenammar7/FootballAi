"""Output isolation and persistence tests for the local storage adapter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from footballai_v2.contracts.v1 import (
    AnalysisRun,
    ArtifactReference,
    CodeReference,
    ContractValidationError,
    DataOrigin,
    InputReference,
)
from footballai_v2.storage import (
    LocalAnalysisRunStore,
    ManifestConflictError,
    RunAlreadyExistsError,
)


CREATED = datetime(2026, 7, 26, 10, tzinfo=timezone.utc)
RUN_IDS = (
    "018f47d2-5c88-7a1f-8b57-42f89c6a13a5",
    "2ff69ffc-2cba-44fe-b240-66fc4188f9fa",
)


def new_run(run_id: str) -> AnalysisRun:
    return AnalysisRun.new(
        analysis_run_id=run_id,
        data_origin=DataOrigin.SYNTHETIC,
        input=InputReference("file:///fixture.mp4", "a" * 64, "video/mp4"),
        code=CodeReference("https://github.com/example/FootballAi", "8" * 40),
        pipeline_version="2.0.0",
        created_at=CREATED,
    )


def start(store: LocalAnalysisRunStore, run: AnalysisRun) -> AnalysisRun:
    store.create(run)
    running = run.start(CREATED + timedelta(seconds=1))
    store.save(running)
    return running


def test_same_artifact_name_in_two_runs_has_two_isolated_files(tmp_path):
    store = LocalAnalysisRunStore(tmp_path / "analysis-runs")
    runs = [start(store, new_run(run_id)) for run_id in RUN_IDS]

    refs = [
        store.write_artifact(
            run.analysis_run_id,
            name="summary",
            relative_path="artifacts/summary.json",
            content=content,
            media_type="application/json",
        )
        for run, content in zip(runs, (b'{"run": 1}', b'{"run": 2}'), strict=True)
    ]

    paths = [
        store.artifact_path(run.analysis_run_id, ref.relative_path)
        for run, ref in zip(runs, refs, strict=True)
    ]
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b'{"run": 1}'
    assert paths[1].read_bytes() == b'{"run": 2}'


def test_artifacts_use_exclusive_creation_and_cannot_be_overwritten(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    run = start(store, new_run(RUN_IDS[0]))
    kwargs = {
        "name": "summary",
        "relative_path": "artifacts/summary.json",
        "content": b"first",
        "media_type": "application/json",
    }
    store.write_artifact(run.analysis_run_id, **kwargs)

    with pytest.raises(FileExistsError):
        store.write_artifact(run.analysis_run_id, **{**kwargs, "content": b"replacement"})
    assert store.artifact_path(run.analysis_run_id, kwargs["relative_path"]).read_bytes() == b"first"


def test_duplicate_run_id_cannot_reuse_a_namespace(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    run = new_run(RUN_IDS[0])
    store.create(run)
    with pytest.raises(RunAlreadyExistsError):
        store.create(run)


def test_namespace_must_be_reserved_before_execution_starts(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = new_run(RUN_IDS[0]).start(CREATED + timedelta(seconds=1))
    with pytest.raises(ManifestConflictError, match="queued state"):
        store.create(running)


def test_artifact_write_requires_running_state(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    run = new_run(RUN_IDS[0])
    store.create(run)
    with pytest.raises(ManifestConflictError, match="only be written"):
        store.write_artifact(
            run.analysis_run_id,
            name="summary",
            relative_path="artifacts/summary.json",
            content=b"{}",
            media_type="application/json",
        )


def test_terminal_manifest_is_immutable(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = start(store, new_run(RUN_IDS[0]))
    artifact = store.write_artifact(
        running.analysis_run_id,
        name="summary",
        relative_path="artifacts/summary.json",
        content=b"{}",
        media_type="application/json",
    )
    succeeded = running.succeed([artifact], CREATED + timedelta(seconds=2))
    store.save(succeeded)
    with pytest.raises(ManifestConflictError, match="immutable"):
        store.save(succeeded)


def test_store_rejects_traversal_before_touching_the_filesystem(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = start(store, new_run(RUN_IDS[0]))
    with pytest.raises(ContractValidationError):
        store.write_artifact(
            running.analysis_run_id,
            name="escape",
            relative_path="artifacts/../../outside.json",
            content=b"unsafe",
            media_type="application/json",
        )
    assert not (tmp_path.parent / "outside.json").exists()


def test_manifest_persists_lifecycle_and_content_hash(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = start(store, new_run(RUN_IDS[0]))
    artifact = store.write_artifact(
        running.analysis_run_id,
        name="summary",
        relative_path="artifacts/summary.json",
        content=b"{}",
        media_type="application/json",
        schema_version="footballai.player-summary/v1",
    )
    succeeded = running.succeed([artifact], CREATED + timedelta(seconds=2))
    store.save(succeeded)

    restored = store.load(running.analysis_run_id)
    assert restored == succeeded
    assert restored.outputs[0].sha256 == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_success_rejects_output_metadata_that_does_not_match_bytes(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = start(store, new_run(RUN_IDS[0]))
    actual = store.write_artifact(
        running.analysis_run_id,
        name="summary",
        relative_path="artifacts/summary.json",
        content=b"{}",
        media_type="application/json",
    )
    forged = ArtifactReference(
        name=actual.name,
        relative_path=actual.relative_path,
        media_type=actual.media_type,
        sha256="f" * 64,
        size_bytes=actual.size_bytes,
    )
    with pytest.raises(ManifestConflictError, match="hash does not match"):
        store.save(running.succeed([forged], CREATED + timedelta(seconds=2)))
