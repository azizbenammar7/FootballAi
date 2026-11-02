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

export type SupportLevel = 'supported' | 'partial' | 'unavailable'
export type ModelMode = 'offline_demo_fake' | 'real_model' | 'not_invoked'
export type OutputNature = 'synthetic' | 'verified_real' | 'unverified_or_partial'

export interface Citation {
  run_id: string
  field_path: string
  artifact_id: string | null
  schema_version: string | null
  source_sha256: string | null
}

export interface ExplanationResponse {
  run_id: string
  answer: string
  support_level: SupportLevel
  output_nature: OutputNature
  model_mode: ModelMode
  citations: Citation[]
  limitations: string[]
  unavailable_reason: string | null
}

export type AgentAdapter = 'offline_scripted' | 'real_model'
export interface AgentAdapterInfo {
  adapter: AgentAdapter
  model: string
}
export type ToolCallStatus = 'ok' | 'no_evidence' | 'rejected' | 'failed'
export type AgentStopReason =
  | 'answered' | 'unsupported_question' | 'evidence_not_available' | 'no_relevant_claim'
  | 'selection_rejected' | 'citation_rejected' | 'cross_run_rejected' | 'tool_budget_exhausted'
  | 'turn_budget_exhausted' | 'deadline_exceeded' | 'budget_exhausted' | 'prompt_budget_exceeded' | 'model_timeout'
  | 'model_refusal' | 'model_truncated' | 'malformed_response' | 'provider_error'

export interface AgentToolCall {
  sequence: number
  turn: number
  tool: string | null
  status: ToolCallStatus
  duration_ms: number
  claim_ids_returned: string[]
  error_code: string | null
}

export interface AgentTrace {
  schema_version: string
  correlation_id: string
  run_id: string
  adapter: AgentAdapter
  model: string
  model_mode: ModelMode
  instruction_version: string
  routing_version: string
  termination: AgentStopReason
  tool_error_code: string | null
  duration_ms: number
  model_turns: number
  model_turn_limit: number
  tool_call_limit: number
  tool_calls: AgentToolCall[]
  tool_calls_refused: number
  claim_ids_returned: string[]
  claim_ids_selected: string[]
  claim_ids_rendered: string[]
  selected_ids_rejected: number
}

export interface AgentResponse {
  explanation: ExplanationResponse
  trace: AgentTrace
}

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
