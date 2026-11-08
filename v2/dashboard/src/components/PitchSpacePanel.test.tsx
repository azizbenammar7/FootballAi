import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PitchSpacePanel } from './PitchSpacePanel'

const RUN = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1'
const LABEL = 'Tracklet-level pitch-space diagnostic. This is NOT a verified individual player heatmap.'

const base = {
  run_id: RUN, label: LABEL, level: 'tracklet', data_origin: 'real',
  identity: { resolver_mode: 'review_only', assignments_by_status: { unresolved: 3 }, team_assignment: 'not_available' },
  limitations: ['Tracklets are not players.'],
  evidence: { tracklets_sha256: 'b'.repeat(64), calibration_sha256: null },
  artifacts: [{ artifact_id: 'pitch-space-diagnostic', schema_version: 'footballai.pitch-space-diagnostic/v1', sha256: 'c'.repeat(64), integrity_state: 'verified' }],
  points_truncated: false,
}
const uncalibrated = {
  ...base, evidence_status: 'no_annotations', calibration_state: 'not_attempted', calibration_reason: 'no_landmark_annotations',
  pitch: null, occupancy: null, points: [],
  interval: {
    start_seconds: null, end_seconds: null, analysed_frames: 41,
    frames_by_status: { calibrated: 0, uncalibrated: 41, invalidated: 0 },
    frames_by_reason: { no_landmark_annotations: 41 }, calibrated_fraction: 0,
    segments: [{ start_frame: 0, end_frame: 40, start_seconds: 0, end_seconds: 8, status: 'uncalibrated', reason: 'no_landmark_annotations', frames: 41 }],
  },
  observations: { total: 123, projected: 0, by_projection_status: { frame_uncalibrated: 123 }, distinct_tracklets_projected: 0, projected_extrapolated_beyond_landmarks: 0, projected_inside_pitch_lines: 0 },
}
const calibrated = {
  ...base, data_origin: 'synthetic', evidence_status: 'synthetic_annotations', calibration_state: 'calibrated_frames_available', calibration_reason: null,
  pitch: { length_m: 105, width_m: 68, dimensions_source: 'test', markings: { penalty_area_depth_m: 16.5, penalty_area_width_m: 40.32, goal_area_depth_m: 5.5, goal_area_width_m: 18.32, centre_circle_radius_m: 9.15 } },
  occupancy: { unit: 'projected tracklet observations per cell', cell_size_m: 5, origin_m: [-5, -5], columns: 23, rows: 16, counts: Array.from({ length: 16 }, (_, r) => Array.from({ length: 23 }, (_, c) => (r === 5 && c === 5 ? 21 : 0))) },
  points: [
    { frame_index: 10, timestamp_seconds: 2, tracklet_id: 'TRK_000001', identity_status: 'unresolved', x_m: 20, y_m: 20, inside_pitch_lines: true, extrapolated_beyond_landmarks: false, calibration_keyframe: 10 },
    { frame_index: 11, timestamp_seconds: 2.2, tracklet_id: 'TRK_000002', identity_status: 'unresolved', x_m: 35, y_m: 50, inside_pitch_lines: true, extrapolated_beyond_landmarks: true, calibration_keyframe: 10 },
  ],
  interval: {
    start_seconds: 1, end_seconds: 3, analysed_frames: 11,
    frames_by_status: { calibrated: 6, uncalibrated: 5, invalidated: 0 },
    frames_by_reason: { outside_verified_keyframe_interval: 5 }, calibrated_fraction: 0.545455,
    segments: [
      { start_frame: 5, end_frame: 9, start_seconds: 1, end_seconds: 1.8, status: 'uncalibrated', reason: 'outside_verified_keyframe_interval', frames: 5 },
      { start_frame: 10, end_frame: 15, start_seconds: 2, end_seconds: 3, status: 'calibrated', reason: null, frames: 6 },
    ],
  },
  observations: { total: 33, projected: 18, by_projection_status: { projected: 18, frame_uncalibrated: 15 }, distinct_tracklets_projected: 3, projected_extrapolated_beyond_landmarks: 1, projected_inside_pitch_lines: 18 },
}

function json(value: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } }))
}

beforeEach(() => { vi.stubGlobal('fetch', vi.fn()) })
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('pitch-space diagnostic panel', () => {
  it('states explicitly that an unannotated run has no positions and is not a player heatmap', async () => {
    vi.mocked(fetch).mockImplementation(() => json(uncalibrated))
    render(<PitchSpacePanel runId={RUN} />)
    expect(await screen.findByText(/NOT a verified individual player heatmap/)).toBeInTheDocument()
    expect(screen.getByText(/no pitch landmark annotations were attached/)).toBeInTheDocument()
    expect(screen.getByText(/No fixed pixel-to-metre scale/)).toBeInTheDocument()
    expect(screen.getByText('No pitch landmark annotations')).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(screen.getByText('no_landmark_annotations')).toBeInTheDocument()
  })

  it('draws calibrated positions, coverage and synthetic status, and filters by interval', async () => {
    vi.mocked(fetch).mockImplementation(() => json(calibrated))
    render(<PitchSpacePanel runId={RUN} />)
    expect(await screen.findByRole('img', { name: /105 by 68 metres with 18 projected/ })).toBeInTheDocument()
    expect(screen.getByText('Synthetic annotations — not real evidence')).toBeInTheDocument()
    expect(screen.getByText('54.5%')).toBeInTheDocument()
    expect(screen.getByText('5 / 0')).toBeInTheDocument()
    expect(screen.getByText(/team not available/)).toBeInTheDocument()
    const user = userEvent.setup()
    await user.type(screen.getByLabelText('From (s)'), '1')
    await user.type(screen.getByLabelText('To (s)'), '3')
    await user.click(screen.getByRole('button', { name: 'Apply interval' }))
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith(`/api/v1/runs/${RUN}/pitch-space?start_seconds=1&end_seconds=3`))).toBe(true))
  })

  it('shows the server message when the diagnostic is unavailable', async () => {
    vi.mocked(fetch).mockImplementation(() => json({ detail: { error_code: 'pitch_space_unavailable', message: 'This run has no pitch-space diagnostic.' } }, 404))
    render(<PitchSpacePanel runId={RUN} />)
    expect(await screen.findByText('This run has no pitch-space diagnostic.')).toBeInTheDocument()
  })
})
