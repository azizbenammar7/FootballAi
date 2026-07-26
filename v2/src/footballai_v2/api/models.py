"""Typed public response models for the local V2 API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(PublicModel):
    status: str
    service: str
    contract_version: str


class StageView(PublicModel):
    stage_id: str
    stage_name: str
    required: bool
    status: str
    progress_percent: float
    started_at: str | None
    finished_at: str | None
    produced_artifact_ids: list[str]
    error: dict[str, Any] | None
    performance_metrics: dict[str, Any]
    message: str | None


class AttemptLink(PublicModel):
    run_id: str
    attempt_number: int
    status: str
    created_at: str


class RunListItem(PublicModel):
    run_id: str
    logical_analysis_id: str
    origin: str
    status: str
    attempt_number: int
    created_at: str
    pipeline_version: str
    warning_count: int
    stage_progress_percent: float


class RunListResponse(PublicModel):
    runs: list[RunListItem]


class ProvenanceView(PublicModel):
    input_uri: str
    input_checksum: str
    input_media_type: str
    repository: str
    code_revision: str
    code_dirty: bool
    pipeline_version: str
    parameters: dict[str, Any]
    models: list[dict[str, Any]]


class RunDetailResponse(PublicModel):
    run_id: str
    logical_analysis_id: str
    attempt_number: int
    previous_attempt_run_id: str | None
    status: str
    origin: str
    contract_version: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    partial_reason: str | None
    cancellation_reason: str | None
    failure: dict[str, Any] | None
    provenance: ProvenanceView
    warnings: list[str]
    attempt_chain: list[AttemptLink]
    stages: list[StageView]


class ManifestResponse(PublicModel):
    manifest: dict[str, Any]


class ArtifactView(PublicModel):
    artifact_id: str
    name: str
    category: str
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str
    schema_version: str | None
    integrity_state: str


class ArtifactListResponse(PublicModel):
    run_id: str
    artifacts: list[ArtifactView]


class TeamBlock(PublicModel):
    block_index: int
    start_minute: int
    end_minute: int
    average_speed_ms: float
    estimated_distance_m: float


class DistanceSummary(PublicModel):
    total_m: float
    average_per_track_m: float
    maximum_track_m: float


class TeamSummaryResponse(PublicModel):
    run_id: str
    logical_analysis_id: str
    origin: str
    legacy: bool
    match_duration_seconds: float
    total_tracks: int
    scored_tracks: int
    insufficient_tracks: int
    distance: DistanceSummary
    advisory_distribution: dict[str, int]
    blocks: list[TeamBlock]
    warnings: list[str]


class PlayerListItem(PublicModel):
    player_id: str
    label: str
    identity_verified: bool
    total_distance_m: float
    average_speed_ms: float
    peak_speed_ms: float
    sprint_count: int
    active_span_seconds: float
    coverage_fraction: float
    advisory_level: str
    advisory_score: float | None


class PlayerListResponse(PublicModel):
    run_id: str
    players: list[PlayerListItem]
    warnings: list[str]


class TimelinePoint(PublicModel):
    block_index: int
    minute: int
    value: float


class AdvisoryView(PublicModel):
    label: str = "Workload and Fatigue Advisory"
    level: str
    score: float | None
    reason: str | None
    indicators: dict[str, Any]
    breakdown: dict[str, Any]
    advisory_only: bool = True


class PlayerDetailResponse(PublicModel):
    run_id: str
    player_id: str
    label: str
    identity_verified: bool = False
    total_distance_m: float
    average_speed_ms: float
    peak_speed_ms: float
    sprint_count: int
    active_span_seconds: float = Field(description="Approximate legacy active-span metric")
    coverage_fraction: float
    heatmap: list[list[float]]
    speed_timeline: list[TimelinePoint]
    distance_timeline: list[TimelinePoint]
    advisory: AdvisoryView
    warnings: list[str]
