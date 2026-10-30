import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import RunDetailPage from './RunDetailPage'

const RUN = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1'

const detail = {
  run_id: RUN, logical_analysis_id: '11111111-1111-4111-8111-111111111111', attempt_number: 1,
  previous_attempt_run_id: null, status: 'succeeded', origin: 'real', contract_version: 'footballai.analysis-run/v1',
  created_at: '2026-10-30T10:00:00Z', started_at: '2026-10-30T10:00:01Z', completed_at: '2026-10-30T10:00:02Z',
  partial_reason: null, cancellation_reason: null, failure: null,
  provenance: { input_uri: 'local-input://redacted', input_checksum: 'd'.repeat(64), input_media_type: 'video/mp4', repository: 'https://example.com/repo', code_revision: 'e'.repeat(40), code_dirty: false, pipeline_version: 'v1_compat/2.0.0', parameters: {}, models: [] },
  warnings: [], attempt_chain: [{ run_id: RUN, attempt_number: 1, status: 'succeeded', created_at: '2026-10-30T10:00:00Z' }],
  stages: [{ stage_id: 'artifact-publication-1', stage_name: 'artifact_publication', required: true, status: 'succeeded', progress_percent: 100, started_at: '2026-10-30T10:00:01Z', finished_at: '2026-10-30T10:00:02Z', produced_artifact_ids: [], error: null, performance_metrics: {}, message: 'done' }],
}
const artifacts = { run_id: RUN, artifacts: [{ artifact_id: 'analysis-diagnostics', name: 'Analysis Diagnostics', category: 'other', relative_path: 'artifacts/analysis-diagnostics.json', media_type: 'application/json', size_bytes: 400, sha256: 'a'.repeat(64), schema_version: 'footballai.analysis-diagnostics/v2', integrity_state: 'verified' }] }
const explanation = {
  run_id: RUN, answer: 'Analysis run aaaaaaaa has status succeeded, produced by the v1_compat profile.',
  support_level: 'supported', output_nature: 'verified_real', model_mode: 'offline_demo_fake',
  citations: [{ run_id: RUN, field_path: 'diagnostics.observed_metrics.total_tracklets', artifact_id: 'analysis-diagnostics', schema_version: 'footballai.analysis-diagnostics/v2', source_sha256: 'a'.repeat(64) }],
  limitations: ['Track identifiers are temporary tracklets, not verified persistent player identities.'],
  unavailable_reason: null,
}

function json(value: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } }))
}
function route() {
  return render(
    <MemoryRouter initialEntries={[`/runs/${RUN}`]}>
      <Routes><Route path="/runs/:runId" element={<RunDetailPage />} /></Routes>
    </MemoryRouter>,
  )
}
function mock(explainResult: () => Promise<Response>) {
  vi.mocked(fetch).mockImplementation((input, init) => {
    const url = String(input)
    if (init?.method === 'POST' && url.endsWith('/explain')) return explainResult()
    if (url.endsWith('/artifacts')) return json(artifacts)
    return json(detail)
  })
}

beforeEach(() => { vi.stubGlobal('fetch', vi.fn()) })
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('run detail explanation panel', () => {
  it('renders the explain action for a completed run', async () => {
    mock(() => json(explanation))
    route()
    expect(await screen.findByRole('button', { name: 'Explain this run' })).toBeEnabled()
  })

  it('shows a grounded answer, mode, citations and limitations after requesting', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    expect(await screen.findByText(/status succeeded/)).toBeInTheDocument()
    expect(screen.getByText('Supported by evidence')).toBeInTheDocument()
    expect(screen.getByText(/Offline deterministic demo/)).toBeInTheDocument()
    expect(screen.getByText('diagnostics.observed_metrics.total_tracklets')).toBeInTheDocument()
    expect(screen.getByText(/temporary tracklets/)).toBeInTheDocument()
  })

  it.each([
    ['synthetic', 'Synthetic output'],
    ['verified_real', 'Real inference — execution verified'],
    ['unverified_or_partial', 'Real inference not verified'],
  ])('labels %s output distinctly', async (output_nature, label) => {
    mock(() => json({ ...explanation, output_nature }))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    expect(await screen.findByText(label)).toBeInTheDocument()
  })

  it('never presents the offline demo as a language model', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    expect(await screen.findByText(/not a language model/)).toBeInTheDocument()
    expect(screen.queryByText(/selected verified claims only/)).not.toBeInTheDocument()
  })

  it('labels a real model as selecting claims, not as a source of facts', async () => {
    mock(() => json({ ...explanation, model_mode: 'real_model' }))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    expect(await screen.findByText('Language model — selected verified claims only')).toBeInTheDocument()
  })

  it('shows a deterministic abstention with no citations and no model', async () => {
    mock(() => json({
      ...explanation, answer: "I can't answer that from verified evidence.", support_level: 'unavailable',
      output_nature: 'verified_real', model_mode: 'not_invoked', citations: [], unavailable_reason: 'unsupported_question',
    }))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    expect(await screen.findByText('Not answerable from evidence')).toBeInTheDocument()
    expect(screen.getByText('No model invoked — deterministic response')).toBeInTheDocument()
    expect(screen.getByText(/does not measure this/)).toBeInTheDocument()
    expect(screen.queryByText('unsupported_question')).not.toBeInTheDocument()
    expect(screen.queryByText('Evidence citations')).not.toBeInTheDocument()
  })

  it('shows an error state when the explanation fails', async () => {
    mock(() => json({ detail: { error_code: 'evidence_validation_failed', message: 'Evidence citation validation failed.' } }, 422))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Explain this run' })).not.toBeInTheDocument())
    expect(screen.queryByText('Supported by evidence')).not.toBeInTheDocument()
    expect(screen.queryByText(/disabled in this environment/)).not.toBeInTheDocument()
  })

  it('shows a disabled notice when the endpoint is not enabled', async () => {
    mock(() => json({ detail: 'Not Found' }, 404))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    await waitFor(() => expect(screen.getByText(/disabled in this environment/)).toBeInTheDocument())
  })
})
