export interface RunListItem {
  run_id: string
  logical_analysis_id: string
  origin: 'real' | 'synthetic' | 'evaluation' | 'legacy_v1'
  status: 'queued' | 'running' | 'succeeded' | 'partial' | 'failed' | 'cancelled'
  attempt_number: number
  created_at: string
  pipeline_version: string
  warning_count: number
  stage_progress_percent: number
}

export interface RunListResponse { runs: RunListItem[] }

export interface PipelineProfile {
  profile_id: string; display_name: string; description: string; available: boolean
  readiness_status: string; readiness_message: string; setup_command: string | null
  missing_requirements: string[]; runtime_errors: string[]; runtime: Record<string, unknown>
  warnings: string[]; purpose: string; gpu: string
}
export interface PipelineProfileList { profiles: PipelineProfile[] }

export interface StageRecord {
  stage_id: string
  stage_name: string
  required: boolean
  status: string
  progress_percent: number
  started_at: string | null
  finished_at: string | null
  produced_artifact_ids: string[]
  error: { error_code: string; safe_message: string; retryable: boolean } | null
  performance_metrics: Record<string, unknown>
  message: string | null
}

export interface AttemptLink {
  run_id: string
  attempt_number: number
  status: string
  created_at: string
}

export interface RunDetail {
  run_id: string
  logical_analysis_id: string
  attempt_number: number
  previous_attempt_run_id: string | null
  status: RunListItem['status']
  origin: RunListItem['origin']
  contract_version: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  partial_reason: string | null
  cancellation_reason: string | null
  failure: Record<string, unknown> | null
  provenance: {
    input_uri: string
    input_checksum: string
    input_media_type: string
    repository: string
    code_revision: string
    code_dirty: boolean
    pipeline_version: string
    parameters: Record<string, unknown>
    models: Record<string, unknown>[]
  }
  warnings: string[]
  attempt_chain: AttemptLink[]
  stages: StageRecord[]
}

export interface RunProgress {
  run_id: string; logical_analysis_id: string; attempt_number: number; status: RunListItem['status']
  overall_progress_percent: number; active_stage: string | null; stages: StageRecord[]
  created_at: string; updated_at: string; can_cancel: boolean; can_retry: boolean; can_create_new_from_input: boolean
}

export interface Artifact {
  artifact_id: string
  name: string
  category: string
  relative_path: string
  media_type: string
  size_bytes: number
  sha256: string
  schema_version: string | null
  integrity_state: 'verified' | 'invalid'
}

export interface ArtifactListResponse { run_id: string; artifacts: Artifact[] }

export interface TeamSummary {
  run_id: string
  logical_analysis_id: string
  origin: string
  legacy: boolean
  match_duration_seconds: number
  total_tracks: number
  scored_tracks: number
  insufficient_tracks: number
  distance: { total_m: number; average_per_track_m: number; maximum_track_m: number }
  advisory_distribution: Record<string, number>
  blocks: Array<{
    block_index: number
    start_minute: number
    end_minute: number
    average_speed_ms: number
    estimated_distance_m: number
  }>
  warnings: string[]
}

export interface PlayerListItem {
  player_id: string
  label: string
  identity_verified: boolean
  total_distance_m: number
  average_speed_ms: number
  peak_speed_ms: number
  sprint_count: number
  active_span_seconds: number
  coverage_fraction: number
  advisory_level: string
  advisory_score: number | null
}

export interface PlayerListResponse {
  run_id: string
  players: PlayerListItem[]
  warnings: string[]
}

export interface TimelinePoint { block_index: number; minute: number; value: number }

export interface PlayerDetail {
  run_id: string
  player_id: string
  label: string
  identity_verified: false
  total_distance_m: number
  average_speed_ms: number
  peak_speed_ms: number
  sprint_count: number
  active_span_seconds: number
  coverage_fraction: number
  heatmap: number[][]
  speed_timeline: TimelinePoint[]
  distance_timeline: TimelinePoint[]
  advisory: {
    label: string
    level: string
    score: number | null
    reason: string | null
    indicators: Record<string, unknown>
    breakdown: Record<string, unknown>
    advisory_only: true
  }
  warnings: string[]
}
