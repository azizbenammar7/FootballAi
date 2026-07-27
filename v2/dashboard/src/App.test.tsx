import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const run = {
  run_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1', logical_analysis_id: '11111111-1111-4111-8111-111111111111',
  origin: 'legacy_v1', status: 'succeeded', attempt_number: 1, created_at: '2026-07-26T10:00:00Z',
  pipeline_version: 'legacy-import/1.0.0', warning_count: 8, stage_progress_percent: 83,
}
const warnings = ['Track IDs are not verified player identities.']
const detail = {
  ...run, contract_version: 'footballai.analysis-run/v1', previous_attempt_run_id: null, started_at: run.created_at,
  completed_at: run.created_at, partial_reason: null, cancellation_reason: null, failure: null, warnings,
  provenance: { input_uri: 'legacy-v1://artifact-set/test', input_checksum: 'a'.repeat(64), input_media_type: 'application/json', repository: 'https://example.com/repo', code_revision: '8'.repeat(40), code_dirty: false, pipeline_version: run.pipeline_version, parameters: {}, models: [] },
  attempt_chain: [{ run_id: run.run_id, attempt_number: 1, status: 'succeeded', created_at: run.created_at }],
  stages: [{ stage_id: 'ingestion-1', stage_name: 'ingestion', required: true, status: 'succeeded', progress_percent: 100, started_at: run.created_at, finished_at: run.created_at, produced_artifact_ids: [], error: null, performance_metrics: {}, message: 'Artifacts copied safely.' }],
}
const team = {
  run_id: run.run_id, logical_analysis_id: run.logical_analysis_id, origin: 'legacy_v1', legacy: true,
  match_duration_seconds: 5400, total_tracks: 2, scored_tracks: 1, insufficient_tracks: 1,
  distance: { total_m: 1500, average_per_track_m: 750, maximum_track_m: 1000 },
  advisory_distribution: { MEDIUM: 1, INSUFFICIENT: 1 },
  blocks: [{ block_index: 0, start_minute: 0, end_minute: 15, average_speed_ms: 2, estimated_distance_m: 500 }], warnings,
}
const players = { run_id: run.run_id, warnings, players: [{ player_id: '12', label: 'Legacy track 12', identity_verified: false, total_distance_m: 1000, average_speed_ms: 2, peak_speed_ms: 6, sprint_count: 1, active_span_seconds: 50, coverage_fraction: .5, advisory_level: 'MEDIUM', advisory_score: .4 }] }
const player = {
  run_id: run.run_id, player_id: '12', label: 'Unverified player track 12', identity_verified: false,
  total_distance_m: 1000, average_speed_ms: 2, peak_speed_ms: 6, sprint_count: 1, active_span_seconds: 50,
  coverage_fraction: .5, heatmap: [[.2,.8]], speed_timeline: [{ block_index: 0, minute: 0, value: 2 }],
  distance_timeline: [{ block_index: 0, minute: 0, value: 0 }, { block_index: 1, minute: 15, value: 1000 }],
  advisory: { label: 'Workload and Fatigue Advisory', level: 'MEDIUM', score: .4, reason: 'Approximate.', indicators: {}, breakdown: {}, advisory_only: true }, warnings,
}
const demoWarning = 'Synthetic workflow result. This run demonstrates the full upload, processing, progress and dashboard workflow.'
const profiles = { profiles: [
  { profile_id: 'demo_fast', display_name: 'Demo fast', description: 'Deterministic workflow.', available: true, readiness_status: 'ready', readiness_message: 'Ready.', setup_command: null, missing_requirements: [], runtime_errors: [], runtime: { device: 'not_required', model: null }, warnings: [demoWarning], purpose: 'Workflow validation', gpu: 'not_required' },
  { profile_id: 'v1_compat', display_name: 'V1-compatible analysis', description: 'Historical algorithm adapter.', available: false, readiness_status: 'missing_python_packages', readiness_message: 'V1-compatible analysis is unavailable. Run: make v2-v1-compat-setup', setup_command: 'make v2-v1-compat-setup', missing_requirements: ['ultralytics'], runtime_errors: [], runtime: { device: 'unavailable', model: 'yolov8m.pt' }, warnings: ['V1-compatible analysis'], purpose: 'CV execution', gpu: 'optional' },
] }
const progress = {
  run_id: run.run_id, logical_analysis_id: run.logical_analysis_id, attempt_number: 1, status: 'succeeded',
  overall_progress_percent: 100, active_stage: null, stages: detail.stages, created_at: run.created_at,
  updated_at: run.created_at, can_cancel: false, can_retry: false, can_create_new_from_input: true,
}

function jsonResponse(value: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } }))
}
function renderRoute(route: string) { return render(<MemoryRouter initialEntries={[route]}><App /></MemoryRouter>) }

beforeEach(() => { vi.stubGlobal('fetch', vi.fn()) })
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('dashboard states and analysis views', () => {
  it('renders a loading state', async () => {
    vi.mocked(fetch).mockReturnValue(new Promise(() => {}))
    renderRoute('/runs')
    expect(await screen.findByText('Loading local analysis runs…')).toBeInTheDocument()
  })

  it('renders the empty run state', async () => {
    vi.mocked(fetch).mockReturnValue(jsonResponse({ runs: [] }))
    renderRoute('/runs')
    expect(await screen.findByText('No V2 runs yet')).toBeInTheDocument()
  })

  it('renders a safe API error', async () => {
    vi.mocked(fetch).mockReturnValue(jsonResponse({ detail: 'API unavailable' }, 503))
    renderRoute('/runs')
    expect(await screen.findByRole('alert')).toHaveTextContent('API unavailable')
  })

  it('renders run status, origin, warning and progress data', async () => {
    vi.mocked(fetch).mockReturnValue(jsonResponse({ runs: [run] }))
    renderRoute('/runs')
    expect(await screen.findByText('legacy v1')).toBeInTheDocument()
    expect(screen.getByText('succeeded')).toBeInTheDocument()
    expect(screen.getByText(/83% · 8 notices/)).toBeInTheDocument()
  })

  it('renders run stages, artifact integrity and unavoidable legacy warning', async () => {
    vi.mocked(fetch).mockImplementation((input) => String(input).endsWith('/artifacts')
      ? jsonResponse({ run_id: run.run_id, artifacts: [{ artifact_id: 'summary', name: 'Team summary', category: 'summary', relative_path: 'artifacts/summary.json', media_type: 'application/json', size_bytes: 10, sha256: 'a'.repeat(64), schema_version: null, integrity_state: 'verified' }] })
      : jsonResponse(detail))
    renderRoute(`/runs/${run.run_id}`)
    expect(await screen.findByText('Legacy V1 analysis')).toBeInTheDocument()
    expect(screen.getByText('Stage timeline')).toBeInTheDocument()
    expect(await screen.findByText('Team summary')).toBeInTheDocument()
    expect(screen.getByText('verified')).toBeInTheDocument()
  })

  it('renders team metrics and unverified track cards', async () => {
    vi.mocked(fetch).mockImplementation((input) => String(input).endsWith('/players') ? jsonResponse(players) : jsonResponse(team))
    renderRoute(`/runs/${run.run_id}/team`)
    expect(await screen.findByText('Unverified player tracks')).toBeInTheDocument()
    expect(screen.getByText('Legacy track 12')).toBeInTheDocument()
    expect(screen.getByText('1.50 km')).toBeInTheDocument()
  })

  it('renders track heatmap, timelines and approved advisory terminology', async () => {
    vi.mocked(fetch).mockReturnValue(jsonResponse(player))
    renderRoute(`/runs/${run.run_id}/tracks/12`)
    expect(await screen.findByText('Unverified player track 12')).toBeInTheDocument()
    expect(screen.getByText('Workload and Fatigue Advisory')).toBeInTheDocument()
    expect(screen.getByText('Movement profile')).toBeInTheDocument()
    expect(screen.getByText(/not diagnosis or clinical advice/i)).toBeInTheDocument()
  })

  it('uses advisory-only language across the about page', async () => {
    renderRoute('/about')
    expect(await screen.findByText('Advisory language')).toBeInTheDocument()
    await waitFor(() => expect(document.body.textContent?.toLowerCase()).not.toContain('medical prediction'))
  })

  it('renders the New Analysis form without auto-submitting selected files', async () => {
    vi.mocked(fetch).mockReturnValue(jsonResponse(profiles))
    const user = userEvent.setup()
    renderRoute('/analyses/new')
    expect(await screen.findByRole('heading', { name: 'New Analysis' })).toBeInTheDocument()
    const file = new File(['video'], 'fixture.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Football video'), file)
    expect(screen.getByText('fixture.mp4')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start analysis' })).toBeEnabled()
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('shows profile availability and missing requirements', async () => {
    vi.mocked(fetch).mockReturnValue(jsonResponse(profiles))
    renderRoute('/analyses/new')
    expect(await screen.findByText(/V1-compatible analysis unavailable \(missing_python_packages\)/)).toBeInTheDocument()
    expect(screen.getAllByText(/make v2-v1-compat-setup/).length).toBeGreaterThan(0)
    expect(screen.getByRole('option', { name: /V1-compatible analysis — unavailable/ })).toBeDisabled()
  })

  it('enables the V1-compatible option only when readiness is ready', async () => {
    const readyProfiles = { profiles: profiles.profiles.map((profile) => profile.profile_id === 'v1_compat' ? { ...profile, available: true, readiness_status: 'ready', readiness_message: 'V1-compatible analysis is ready.', missing_requirements: [], runtime: { device: 'mps', model: 'yolov8m.pt' } } : profile) }
    vi.mocked(fetch).mockReturnValue(jsonResponse(readyProfiles))
    renderRoute('/analyses/new')
    expect(await screen.findByRole('option', { name: 'V1-compatible analysis' })).toBeEnabled()
  })

  it('renders completed live progress, attempt history and synthetic warning', async () => {
    const demoDetail = { ...detail, origin: 'evaluation', warnings: [demoWarning], pipeline_version: 'demo_fast/1.0.0' }
    vi.mocked(fetch).mockImplementation((input) => String(input).endsWith('/progress') ? jsonResponse(progress) : jsonResponse(demoDetail))
    renderRoute(`/runs/${run.run_id}/progress`)
    expect(await screen.findByRole('heading', { name: 'Analysis complete' })).toBeInTheDocument()
    expect(screen.getByText('Synthetic workflow result')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open results' })).toBeInTheDocument()
    expect(screen.getByText('Attempt chain')).toBeInTheDocument()
  })

  it('shows safe failed-attempt controls without medical prediction language', async () => {
    const failedProgress = { ...progress, status: 'failed', overall_progress_percent: 10, can_retry: true }
    const failedDetail = { ...detail, status: 'failed', completed_at: run.created_at, failure: { error_code: 'test_stage_failure', safe_message: 'Detection stopped safely.' } }
    vi.mocked(fetch).mockImplementation((input) => String(input).endsWith('/progress') ? jsonResponse(failedProgress) : jsonResponse(failedDetail))
    renderRoute(`/runs/${run.run_id}/progress`)
    expect(await screen.findByText('test_stage_failure')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry as new attempt' })).toBeInTheDocument()
    expect(document.body.textContent?.toLowerCase()).not.toContain('medical prediction')
  })
})
