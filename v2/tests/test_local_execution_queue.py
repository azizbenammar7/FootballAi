"""Filesystem queue durability and process-safety tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta

import pytest

from footballai_v2.contracts.v1 import AnalysisRun, CodeReference, DataOrigin, InputReference, StructuredError, utc_now
from footballai_v2.execution.contracts import ExecutionJob
from footballai_v2.execution.queue import DuplicateJobError, LocalFilesystemQueue
from footballai_v2.storage import LocalAnalysisRunStore


def job(run_id: str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1") -> ExecutionJob:
    return ExecutionJob.new(run_id, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1", 1, "demo_fast")


def test_enqueue_claim_and_complete_are_atomic(tmp_path):
    queue = LocalFilesystemQueue(tmp_path / "queue"); queued = job(); queue.enqueue(queued)
    claimed = queue.claim("worker-one")
    assert claimed and claimed.worker_id == "worker-one" and claimed.claimed_at
    assert queue.claim("worker-two") is None
    queue.complete(claimed); queue.complete(claimed)
    assert (tmp_path / "queue/completed" / f"{queued.job_id}.json").is_file()


def test_duplicate_run_enqueue_is_rejected_even_with_new_job_id(tmp_path):
    queue = LocalFilesystemQueue(tmp_path / "queue"); queue.enqueue(job())
    with pytest.raises(DuplicateJobError): queue.enqueue(job())


def test_concurrent_claims_never_duplicate_processing(tmp_path):
    queue = LocalFilesystemQueue(tmp_path / "queue"); queue.enqueue(job())
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda index: queue.claim(f"worker-{index}"), range(8)))
    assert len([item for item in claims if item]) == 1


def test_failure_and_cancellation_have_explicit_terminal_directories(tmp_path):
    queue = LocalFilesystemQueue(tmp_path / "queue"); first = job(); queue.enqueue(first)
    claimed = queue.claim("worker"); assert claimed; queue.fail(claimed)
    second = job("cccccccc-cccc-4ccc-8ccc-ccccccccccc1"); queue.enqueue(second)
    assert queue.cancel(second.run_id)
    assert (tmp_path / "queue/failed" / f"{first.job_id}.json").exists()
    assert (tmp_path / "queue/cancelled" / f"{second.job_id}.json").exists()


def test_malformed_and_oversized_records_are_quarantined(tmp_path):
    queue = LocalFilesystemQueue(tmp_path / "queue")
    (tmp_path / "queue/queued/bad.json").write_text("{" * 20_000)
    assert queue.claim("worker") is None
    assert (tmp_path / "queue/failed/bad.json").exists()


def test_job_ids_and_run_ids_cannot_traverse_paths(tmp_path):
    queue = LocalFilesystemQueue(tmp_path / "queue")
    with pytest.raises(Exception): queue.cancel("../../private")
    with pytest.raises(Exception): replace(job(), job_id="../../private")


def test_abandoned_claim_requeues_nonterminal_run(tmp_path):
    store = LocalAnalysisRunStore(tmp_path / "runs")
    run = AnalysisRun.new(data_origin=DataOrigin.REAL, input=InputReference("run-input://source.mp4", "a" * 64, "video/mp4"), code=CodeReference("https://example.com/repo", "8" * 40), pipeline_version="demo_fast/1.0.0", run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
    store.create(run); queue = LocalFilesystemQueue(tmp_path / "queue"); queued = job(run.run_id); queue.enqueue(queued)
    claimed = queue.claim("dead-worker"); assert claimed
    old = replace(claimed, claimed_at=utc_now() - timedelta(hours=1)); path = tmp_path / "queue/claimed" / f"{claimed.job_id}.json"
    queue._write_replace(path, old)
    assert queue.recover_abandoned(30, store) == 1
    assert queue.claim("new-worker").run_id == run.run_id


def test_terminal_run_is_never_requeued(tmp_path):
    store = LocalAnalysisRunStore(tmp_path / "runs")
    run = AnalysisRun.new(data_origin=DataOrigin.REAL, input=InputReference("run-input://source.mp4", "a" * 64, "video/mp4"), code=CodeReference("https://example.com/repo", "8" * 40), pipeline_version="demo_fast/1.0.0", run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1")
    store.create(run); queue = LocalFilesystemQueue(tmp_path / "queue"); queued = job(run.run_id); queue.enqueue(queued); claimed = queue.claim("dead"); assert claimed
    store.save(run.fail(StructuredError("safe_failure", "Safe failure.", True, utc_now())))
    assert queue.recover_abandoned(0, store) == 0
    assert queue.claim("new") is None
