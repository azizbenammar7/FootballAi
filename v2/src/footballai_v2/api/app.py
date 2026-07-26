"""FastAPI application factory for the local V2 run store."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import UUID4

from footballai_v2.api.legacy_adapter import LegacyDataError, LegacyRunAdapter
from footballai_v2.api.models import (
    ArtifactListResponse,
    ArtifactView,
    AttemptLink,
    HealthResponse,
    ManifestResponse,
    PlayerDetailResponse,
    PlayerListResponse,
    ProvenanceView,
    RunDetailResponse,
    RunListItem,
    RunListResponse,
    StageView,
    TeamSummaryResponse,
)
from footballai_v2.contracts.v1 import ANALYSIS_RUN_CONTRACT_VERSION, AnalysisRun
from footballai_v2.storage import LocalAnalysisRunStore, RunNotFoundError


logger = logging.getLogger("footballai_v2.api")


def _timestamp(value) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _warnings(run: AnalysisRun) -> list[str]:
    value = run.parameters.get("quality_warnings", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _stage_view(stage) -> StageView:
    payload = stage.to_dict()
    return StageView(
        stage_id=stage.stage_id,
        stage_name=stage.stage_name.value,
        required=stage.required,
        status=stage.status.value,
        progress_percent=float(stage.progress_percent),
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
        produced_artifact_ids=list(stage.produced_artifact_ids),
        error=payload["error"],
        performance_metrics=dict(stage.performance_metrics),
        message=stage.message,
    )


def _public_input_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file" or (not parsed.scheme and uri.startswith("/")):
        return "local-input://redacted"
    return uri


def _public_manifest(run: AnalysisRun) -> dict[str, Any]:
    payload = run.to_dict()
    payload["input"] = {**payload["input"], "uri": _public_input_uri(run.input.uri)}
    return payload


def _validate_local_origins(origins: Sequence[str]) -> list[str]:
    validated = []
    for origin in origins:
        parsed = urlparse(origin)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(f"CORS origin must be an HTTP localhost origin: {origin!r}")
        validated.append(origin.rstrip("/"))
    return validated


def create_app(
    run_root: str | Path,
    *,
    allowed_origins: Sequence[str] = ("http://localhost:5173",),
) -> FastAPI:
    """Create a local, read-oriented API bound to one configured run root."""
    store = LocalAnalysisRunStore(run_root)
    app = FastAPI(
        title="FootballAi V2 local analysis API",
        version="1.0.0",
        description="Read-only local API for versioned FootballAi analysis runs.",
    )
    app.state.run_store = store
    origins = _validate_local_origins(allowed_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    def load_run(run_id: UUID4) -> AnalysisRun:
        try:
            return store.load(str(run_id))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Analysis run not found") from exc

    def adapter(run: AnalysisRun) -> LegacyRunAdapter:
        try:
            return LegacyRunAdapter(store, run)
        except LegacyDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="footballai-v2-local-api",
            contract_version=ANALYSIS_RUN_CONTRACT_VERSION,
        )

    @app.get("/api/v1/runs", response_model=RunListResponse)
    def list_runs() -> RunListResponse:
        items = []
        for run in store.list_runs():
            progress = (
                sum(float(stage.progress_percent) for stage in run.stages) / len(run.stages)
                if run.stages
                else 0
            )
            items.append(
                RunListItem(
                    run_id=run.run_id,
                    logical_analysis_id=run.logical_analysis_id,
                    origin=run.data_origin.value,
                    status=run.status.value,
                    attempt_number=run.attempt_number,
                    created_at=_timestamp(run.created_at),
                    pipeline_version=run.pipeline_version,
                    warning_count=len(_warnings(run)),
                    stage_progress_percent=progress,
                )
            )
        return RunListResponse(runs=items)

    @app.get("/api/v1/runs/{run_id}", response_model=RunDetailResponse)
    def run_detail(run_id: UUID4) -> RunDetailResponse:
        run = load_run(run_id)
        chain = [
            AttemptLink(
                run_id=item.run_id,
                attempt_number=item.attempt_number,
                status=item.status.value,
                created_at=_timestamp(item.created_at),
            )
            for item in sorted(
                (
                    item
                    for item in store.list_runs()
                    if item.logical_analysis_id == run.logical_analysis_id
                ),
                key=lambda item: item.attempt_number,
            )
        ]
        return RunDetailResponse(
            run_id=run.run_id,
            logical_analysis_id=run.logical_analysis_id,
            attempt_number=run.attempt_number,
            previous_attempt_run_id=run.previous_attempt_run_id,
            status=run.status.value,
            origin=run.data_origin.value,
            contract_version=run.contract_version,
            created_at=_timestamp(run.created_at),
            started_at=_timestamp(run.started_at),
            completed_at=_timestamp(run.completed_at),
            partial_reason=run.partial_reason,
            cancellation_reason=run.cancellation_reason,
            failure=run.failure.to_dict() if run.failure else None,
            provenance=ProvenanceView(
                input_uri=_public_input_uri(run.input.uri),
                input_checksum=run.input.sha256,
                input_media_type=run.input.media_type,
                repository=run.code.repository,
                code_revision=run.code.revision,
                code_dirty=run.code.dirty,
                pipeline_version=run.pipeline_version,
                parameters=dict(run.parameters),
                models=[item.to_dict() for item in run.models],
            ),
            warnings=_warnings(run),
            attempt_chain=chain,
            stages=[_stage_view(item) for item in run.stages],
        )

    @app.get("/api/v1/runs/{run_id}/manifest", response_model=ManifestResponse)
    def manifest(run_id: UUID4) -> ManifestResponse:
        return ManifestResponse(manifest=_public_manifest(load_run(run_id)))

    @app.get("/api/v1/runs/{run_id}/artifacts", response_model=ArtifactListResponse)
    def artifacts(run_id: UUID4) -> ArtifactListResponse:
        run = load_run(run_id)
        return ArtifactListResponse(
            run_id=run.run_id,
            artifacts=[
                ArtifactView(
                    artifact_id=item.artifact_id,
                    name=item.name,
                    category=item.category.value,
                    relative_path=item.relative_path,
                    media_type=item.media_type,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                    schema_version=item.schema_version,
                    integrity_state=(
                        "verified" if store.artifact_integrity(run.run_id, item.artifact_id) else "invalid"
                    ),
                )
                for item in run.artifacts
            ],
        )

    @app.get("/api/v1/runs/{run_id}/summary", response_model=TeamSummaryResponse)
    def summary(run_id: UUID4) -> TeamSummaryResponse:
        run = load_run(run_id)
        return adapter(run).team_summary()

    @app.get("/api/v1/runs/{run_id}/players", response_model=PlayerListResponse)
    def players(run_id: UUID4) -> PlayerListResponse:
        run = load_run(run_id)
        return adapter(run).player_list()

    @app.get("/api/v1/runs/{run_id}/players/{player_id}", response_model=PlayerDetailResponse)
    def player(run_id: UUID4, player_id: int) -> PlayerDetailResponse:
        run = load_run(run_id)
        try:
            return adapter(run).player_detail(player_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Player track not found") from exc

    return app
