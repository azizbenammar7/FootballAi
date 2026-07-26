"""Output isolation, persistence, integrity, and retry tests."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from footballai_v2.contracts.v1 import (
    AnalysisRun,
    AnalysisRunStatus,
    ArtifactCategory,
    ArtifactReference,
    CodeReference,
    ContractValidationError,
    DataOrigin,
    FailureDetail,
    InputReference,
    InvalidStatusTransition,
    StageError,
    StageExecution,
    StageName,
    StageStatus,
)
from footballai_v2.storage import (
    LocalAnalysisRunStore,
    ManifestConflictError,
    RunAlreadyExistsError,
)


CREATED = datetime(2026, 7, 26, 10, tzinfo=timezone.utc)
LOGICAL_ID = "11111111-1111-4111-8111-111111111111"
RUN_IDS = (
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1",
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2",
    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa3",
)


def new_run(run_id: str, *, logical_id: str = LOGICAL_ID) -> AnalysisRun:
    return AnalysisRun.new(
        logical_analysis_id=logical_id,
        run_id=run_id,
        data_origin=DataOrigin.SYNTHETIC,
        input=InputReference("fixture://synthetic/match.mp4", "a" * 64, "video/mp4"),
        code=CodeReference("https://github.com/example/FootballAi", "8" * 40),
        pipeline_version="2.0.0",
        created_at=CREATED,
    )


def error(at: datetime) -> StageError:
    return StageError(
        "processing_failed",
        "Processing ended safely.",
        True,
        at,
        None,
    )


def stage(status: StageStatus, artifact_ids: tuple[str, ...] = (), attempt: int = 1):
    started = CREATED + timedelta(seconds=1)
    finished = CREATED + timedelta(seconds=2)
    values = {
        StageStatus.QUEUED: (0, None, None, None),
        StageStatus.RUNNING: (50, started, None, None),
        StageStatus.SUCCEEDED: (100, started, finished, None),
        StageStatus.PARTIAL: (60, started, finished, None),
        StageStatus.FAILED: (50, started, finished, error(finished)),
        StageStatus.CANCELLED: (50, started, finished, None),
        StageStatus.SKIPPED: (0, None, finished, None),
    }
    progress, started_at, finished_at, stage_error = values[status]
    return StageExecution(
        stage_id="metrics-1",
        stage_name=StageName.METRICS,
        required=True,
        status=status,
        progress_percent=progress,
        attempt_number=attempt,
        started_at=started_at,
        finished_at=finished_at,
        produced_artifact_ids=artifact_ids,
        error=stage_error,
        performance_metrics={},
        message=None,
    )


def start(store: LocalAnalysisRunStore, run: AnalysisRun) -> AnalysisRun:
    store.create(run)
    running = run.start(
        CREATED + timedelta(seconds=1),
        stages=[stage(StageStatus.RUNNING, attempt=run.attempt_number)],
    )
    store.save(running)
    return running


def write_summary(store: LocalAnalysisRunStore, run: AnalysisRun, content: bytes = b"{}"):
    return store.write_artifact(
        run.run_id,
        artifact_id="team-summary",
        name="Team summary",
        category=ArtifactCategory.SUMMARY,
        relative_path="artifacts/team_summary.json",
        content=content,
        media_type="application/json",
        schema_version="footballai.team-summary/v1",
    )


def succeed(store: LocalAnalysisRunStore, run: AnalysisRun) -> AnalysisRun:
    item = write_summary(store, run)
    completed = run.succeed(
        [item],
        CREATED + timedelta(seconds=3),
        stages=[stage(StageStatus.SUCCEEDED, (item.artifact_id,), run.attempt_number)],
    )
    store.save(completed)
    return completed


def fail(store: LocalAnalysisRunStore, run: AnalysisRun) -> AnalysisRun:
    completed_at = CREATED + timedelta(seconds=3)
    failed = run.fail(
        FailureDetail(
            "attempt_failed",
            "The attempt ended safely.",
            True,
            completed_at,
            None,
        ),
        completed_at,
        stages=[stage(StageStatus.FAILED, attempt=run.attempt_number)],
    )
    store.save(failed)
    return failed


def complete_partial(store: LocalAnalysisRunStore, run: AnalysisRun) -> AnalysisRun:
    item = write_summary(store, run)
    completed = run.complete_partial(
        [item],
        "Reviewable output exists, but mandatory work is incomplete.",
        CREATED + timedelta(seconds=3),
        stages=[stage(StageStatus.PARTIAL, (item.artifact_id,), run.attempt_number)],
    )
    store.save(completed)
    return completed


def test_same_artifact_name_in_two_runs_has_two_isolated_files(tmp_path):
    store = LocalAnalysisRunStore(tmp_path / "configured-runs")
    runs = [
        start(store, new_run(RUN_IDS[0])),
        start(
            store,
            new_run(
                RUN_IDS[1],
                logical_id="22222222-2222-4222-8222-222222222222",
            ),
        ),
    ]
    refs = [write_summary(store, runs[0], b'{"run":1}'), write_summary(store, runs[1], b'{"run":2}')]
    paths = [store.artifact_path(run.run_id, ref.relative_path) for run, ref in zip(runs, refs)]
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b'{"run":1}'
    assert paths[1].read_bytes() == b'{"run":2}'
    assert all(str(path).startswith(str(tmp_path)) for path in paths)


def test_artifacts_use_exclusive_creation_and_cannot_be_overwritten(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    run = start(store, new_run(RUN_IDS[0]))
    first = write_summary(store, run, b"first")
    with pytest.raises(FileExistsError):
        write_summary(store, run, b"replacement")
    assert store.artifact_path(run.run_id, first.relative_path).read_bytes() == b"first"


def test_duplicate_run_id_cannot_reuse_a_namespace(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    run = new_run(RUN_IDS[0])
    store.create(run)
    with pytest.raises(RunAlreadyExistsError):
        store.create(run)


def test_namespace_must_be_reserved_in_queued_state(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = new_run(RUN_IDS[0]).start(
        CREATED + timedelta(seconds=1),
        stages=[stage(StageStatus.RUNNING)],
    )
    with pytest.raises(ManifestConflictError, match="queued state"):
        store.create(running)


def test_artifact_write_requires_running_state(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    run = new_run(RUN_IDS[0])
    store.create(run)
    with pytest.raises(ManifestConflictError, match="only be written"):
        write_summary(store, run)


@pytest.mark.parametrize("terminalizer", [succeed, fail, complete_partial])
def test_terminal_manifest_is_immutable(tmp_path, terminalizer):
    store = LocalAnalysisRunStore(tmp_path)
    terminal = terminalizer(store, start(store, new_run(RUN_IDS[0])))
    before = store.manifest_path(terminal.run_id).read_bytes()
    with pytest.raises(ManifestConflictError, match="immutable"):
        store.save(terminal)
    assert store.manifest_path(terminal.run_id).read_bytes() == before


def test_store_rejects_traversal_before_touching_filesystem(tmp_path):
    store = LocalAnalysisRunStore(tmp_path / "runs")
    running = start(store, new_run(RUN_IDS[0]))
    with pytest.raises(ContractValidationError):
        store.write_artifact(
            running.run_id,
            artifact_id="escape",
            name="Escape",
            category=ArtifactCategory.OTHER,
            relative_path="artifacts/../../outside.json",
            content=b"unsafe",
            media_type="application/json",
        )
    assert not (tmp_path / "outside.json").exists()


def test_store_rejects_symlink_run_namespace(tmp_path):
    store = LocalAnalysisRunStore(tmp_path / "runs")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, store.run_directory(RUN_IDS[0]))
    with pytest.raises(ContractValidationError, match="symlink"):
        store.load(RUN_IDS[0])


def test_manifest_persists_lifecycle_and_content_hash(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    succeeded = succeed(store, start(store, new_run(RUN_IDS[0])))
    restored = store.load(succeeded.run_id)
    assert restored == succeeded
    assert restored.artifacts[0].sha256 == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_success_rejects_artifact_metadata_that_does_not_match_bytes(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    running = start(store, new_run(RUN_IDS[0]))
    actual = write_summary(store, running)
    forged = ArtifactReference(
        artifact_id=actual.artifact_id,
        name=actual.name,
        category=actual.category,
        relative_path=actual.relative_path,
        media_type=actual.media_type,
        sha256="f" * 64,
        size_bytes=actual.size_bytes,
    )
    completed = running.succeed(
        [forged],
        CREATED + timedelta(seconds=3),
        stages=[stage(StageStatus.SUCCEEDED, (forged.artifact_id,))],
    )
    with pytest.raises(ManifestConflictError, match="hash does not match"):
        store.save(completed)


def test_atomic_manifest_failure_leaves_previous_manifest_readable(tmp_path, monkeypatch):
    store = LocalAnalysisRunStore(tmp_path)
    queued = new_run(RUN_IDS[0])
    store.create(queued)
    running = queued.start(CREATED + timedelta(seconds=1), stages=[stage(StageStatus.RUNNING)])

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr("footballai_v2.storage.local_analysis_runs.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.save(running)
    assert store.load(queued.run_id) == queued


@pytest.mark.parametrize("terminalizer", [fail, complete_partial])
def test_retry_creates_isolated_attempt_and_preserves_previous_bytes(tmp_path, terminalizer):
    store = LocalAnalysisRunStore(tmp_path / "configured-root")
    previous = terminalizer(store, start(store, new_run(RUN_IDS[0])))
    previous_manifest = store.manifest_path(previous.run_id).read_bytes()
    previous_artifacts = {
        item.relative_path: store.artifact_path(previous.run_id, item.relative_path).read_bytes()
        for item in previous.artifacts
    }

    retry = store.create_retry_attempt(
        previous.run_id,
        run_id=RUN_IDS[1],
        code=CodeReference("https://github.com/example/FootballAi", "9" * 40),
        pipeline_version="2.0.1",
        parameters={"corrected": True},
    )

    assert retry.logical_analysis_id == previous.logical_analysis_id
    assert retry.data_origin == previous.data_origin
    assert retry.input == previous.input
    assert retry.attempt_number == previous.attempt_number + 1
    assert retry.previous_attempt_run_id == previous.run_id
    assert retry.run_id != previous.run_id
    assert store.run_directory(retry.run_id) != store.run_directory(previous.run_id)
    assert store.run_directory(retry.run_id).is_dir()
    assert store.manifest_path(previous.run_id).read_bytes() == previous_manifest
    assert {
        path: store.artifact_path(previous.run_id, path).read_bytes()
        for path in previous_artifacts
    } == previous_artifacts


def test_retry_attempt_cannot_overwrite_previous_artifact(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    previous = complete_partial(store, start(store, new_run(RUN_IDS[0])))
    retry = store.create_retry_attempt(previous.run_id, run_id=RUN_IDS[1])
    running_retry = retry.start(
        retry.created_at + timedelta(seconds=1),
        stages=[stage(StageStatus.RUNNING, attempt=2)],
    )
    store.save(running_retry)
    new_artifact = write_summary(store, running_retry, b'{"attempt":2}')
    assert store.artifact_path(previous.run_id, previous.artifacts[0].relative_path).read_bytes() == b"{}"
    assert store.artifact_path(retry.run_id, new_artifact.relative_path).read_bytes() == b'{"attempt":2}'


@pytest.mark.parametrize("state", ["queued", "running", "succeeded", "cancelled"])
def test_retry_rejects_unapproved_source_states(tmp_path, state):
    store = LocalAnalysisRunStore(tmp_path)
    queued = new_run(RUN_IDS[0])
    store.create(queued)
    current = queued
    if state != "queued":
        current = queued.start(
            CREATED + timedelta(seconds=1),
            stages=[stage(StageStatus.RUNNING)],
        )
        store.save(current)
    if state == "succeeded":
        current = succeed(store, current)
    elif state == "cancelled":
        current = current.cancel(
            CREATED + timedelta(seconds=2),
            reason="Cancelled by operator.",
            stages=[stage(StageStatus.CANCELLED)],
        )
        store.save(current)
    with pytest.raises(InvalidStatusTransition, match="cannot retry"):
        store.create_retry_attempt(current.run_id, run_id=RUN_IDS[1])


def test_attempt_relationship_fields_cannot_change_in_place(tmp_path):
    store = LocalAnalysisRunStore(tmp_path)
    queued = new_run(RUN_IDS[0])
    store.create(queued)
    running = queued.start(CREATED + timedelta(seconds=1), stages=[stage(StageStatus.RUNNING)])
    changed = replace(
        running,
        logical_analysis_id="22222222-2222-4222-8222-222222222222",
    )
    with pytest.raises(ManifestConflictError, match="provenance"):
        store.save(changed)
