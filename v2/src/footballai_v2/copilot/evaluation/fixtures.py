"""Build an in-memory run and storage from a dataset fixture.

Evaluation never touches a real run: no repository, object store, or video is
read, so only the versioned fixtures can reach a model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from footballai_v2.contracts.v1 import (
    AnalysisRun,
    AnalysisRunStatus,
    ArtifactCategory,
    ArtifactReference,
    CodeReference,
    DataOrigin,
    InputReference,
    ModelReference,
    StageExecution,
    StageName,
    StageStatus,
)
from footballai_v2.copilot.evaluation.dataset import FixtureSet, RunFixture
from footballai_v2.copilot.evidence import DIAGNOSTICS_ARTIFACT_ID
from footballai_v2.storage import RunNotFoundError


_CREATED = datetime(2026, 10, 30, tzinfo=timezone.utc)
_LOGICAL_ID = "cccccccc-cccc-4ccc-8ccc-ccccccccccc1"
_SUMMARY_ARTIFACT_ID = "team-summary"
_SUMMARY_BYTES = b'{"schema":"footballai.team-summary/v1"}'


class FixtureRepository:
    """The only ``AnalysisRepository`` method the evidence service uses."""

    def __init__(self, run: AnalysisRun) -> None:
        self._run = run

    def load(self, run_id: str) -> AnalysisRun:
        if run_id != self._run.run_id:
            raise RunNotFoundError(run_id)
        return self._run


class FixtureObjectStorage:
    """The two ``ObjectStorage`` methods the evidence service uses."""

    def __init__(self, blobs: dict[str, tuple[bytes, bool]]) -> None:
        self._blobs = blobs

    def artifact_integrity(self, run_id: str, artifact_id: str) -> bool:
        return self._blobs[artifact_id][1]

    def read_artifact_bytes(self, run_id: str, artifact_id: str, *, max_bytes: int) -> bytes:
        content = self._blobs[artifact_id][0]
        if len(content) > max_bytes:
            raise ValueError("registered artifact exceeds the configured read limit")
        return content


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _diagnostics_bytes(fixture: RunFixture, fixtures: FixtureSet) -> bytes:
    diagnostics = fixture.diagnostics
    assert diagnostics is not None
    if diagnostics.raw_text is not None:
        return diagnostics.raw_text.encode()
    assert diagnostics.payload is not None
    payload = _deep_merge(fixtures.payloads[diagnostics.payload], diagnostics.overlay or {})
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def _artifact(artifact_id: str, name: str, category: ArtifactCategory, content: bytes, schema: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        name=name,
        category=category,
        relative_path=f"artifacts/{artifact_id}.json",
        media_type="application/json",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        schema_version=schema,
    )


def build_fixture_run(fixture: RunFixture, fixtures: FixtureSet) -> tuple[FixtureRepository, FixtureObjectStorage, str]:
    """Return ports serving one fixture run, and that run's id."""
    if fixture.diagnostics is None:
        blobs = {_SUMMARY_ARTIFACT_ID: (_SUMMARY_BYTES, True)}
        artifact = _artifact(
            _SUMMARY_ARTIFACT_ID,
            "Team Summary",
            ArtifactCategory.SUMMARY,
            _SUMMARY_BYTES,
            "footballai.team-summary/v1",
        )
    else:
        content = _diagnostics_bytes(fixture, fixtures)
        blobs = {DIAGNOSTICS_ARTIFACT_ID: (content, fixture.diagnostics.integrity)}
        artifact = _artifact(
            DIAGNOSTICS_ARTIFACT_ID,
            "Analysis Diagnostics",
            ArtifactCategory.OTHER,
            content,
            fixture.diagnostics.schema_version,
        )
    started = _CREATED + timedelta(seconds=1)
    finished = started + timedelta(seconds=1)
    run = AnalysisRun(
        logical_analysis_id=_LOGICAL_ID,
        run_id=fixtures.run_id,
        attempt_number=1,
        previous_attempt_run_id=None,
        status=AnalysisRunStatus.SUCCEEDED,
        data_origin=DataOrigin(fixture.data_origin),
        input=InputReference(uri="fixture://synthetic/match.mp4", sha256="d" * 64, media_type="video/mp4"),
        code=CodeReference("https://github.com/azizbenammar7/FootballAi", "e" * 40, False),
        pipeline_version="v1_compat/2.0.0" if fixture.profile == "v1_compat" else f"{fixture.profile}/1.0.0",
        parameters={"pipeline_profile": fixture.profile, "video_probe": {"duration_seconds": 60.0}},
        models=tuple(ModelReference(item.name, item.version, item.sha256) for item in fixture.models),
        artifacts=(artifact,),
        stages=(
            StageExecution(
                stage_id="artifact-publication-1",
                stage_name=StageName.ARTIFACT_PUBLICATION,
                required=True,
                status=StageStatus.SUCCEEDED,
                progress_percent=100,
                attempt_number=1,
                started_at=started,
                finished_at=finished,
                produced_artifact_ids=(artifact.artifact_id,),
            ),
        ),
        created_at=_CREATED,
        started_at=started,
        completed_at=finished,
    )
    return FixtureRepository(run), FixtureObjectStorage(blobs), run.run_id
