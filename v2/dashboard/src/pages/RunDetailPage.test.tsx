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
    expect(await screen.findByRole('alert')).toHaveTextContent('Evidence citation validation failed.')
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

describe('run detail question interface', () => {
  const question = () => screen.findByRole('textbox', { name: 'Ask about this run' })
  const ask = () => screen.getByRole('button', { name: 'Ask Copilot' })
  const sent = () => vi.mocked(fetch).mock.calls
    .filter(([url, init]) => init?.method === 'POST' && String(url).endsWith('/explain'))
    .map(([, init]) => JSON.parse(String(init?.body)) as { question: string | null })

  it('sends the typed question to the selected run', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'How reliable was the tracking?')
    await user.click(ask())
    expect(await screen.findByText(/status succeeded/)).toBeInTheDocument()
    expect(sent()).toEqual([{ question: 'How reliable was the tracking?' }])
    expect(String(vi.mocked(fetch).mock.calls.at(-1)?.[0])).toContain(`/api/v1/runs/${RUN}/explain`)
  })

  it('submits with the Enter key and keeps keyboard focus in the input', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'Which model was used?{Enter}')
    expect(await screen.findByText(/status succeeded/)).toBeInTheDocument()
    expect(sent()).toEqual([{ question: 'Which model was used?' }])
    expect(await question()).toHaveFocus()
  })

  it('trims the question before sending it', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.type(await question(), '   Was this analysis synthetic?   {Enter}')
    await waitFor(() => expect(sent()).toEqual([{ question: 'Was this analysis synthetic?' }]))
  })

  it('shows what was asked next to the answer', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'Which model was used?{Enter}')
    expect(await screen.findByText('Which model was used?')).toBeInTheDocument()
  })

  it('keeps the general explanation as a request with no question', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Explain this run' }))
    await screen.findByText(/status succeeded/)
    expect(sent()).toEqual([{ question: null }])
  })

  it('offers example questions that fill the input without sending anything', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'How reliable was the tracking?' }))
    expect(await question()).toHaveValue('How reliable was the tracking?')
    expect(await question()).toHaveFocus()
    expect(sent()).toEqual([])
  })

  it.each(['', '     '])('blocks an empty question %j', async (typed) => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    const input = await question()
    if (typed) await user.type(input, typed)
    expect(ask()).toBeDisabled()
    await user.type(input, '{Enter}')
    expect(sent()).toEqual([])
  })

  it('accepts a question at the limit and refuses one past it', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    const input = await question()
    await user.click(input)
    await user.paste('x'.repeat(500))
    expect(ask()).toBeEnabled()
    await user.paste('y')
    expect(ask()).toBeDisabled()
    expect(input).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('alert')).toHaveTextContent(/500 characters/)
    await user.type(input, '{Enter}')
    expect(sent()).toEqual([])
  })

  it('shows loading, and cannot send twice while waiting', async () => {
    let finish: (value: Response) => void = () => {}
    mock(() => new Promise<Response>((resolve) => { finish = resolve }))
    route()
    const user = userEvent.setup()
    const input = await question()
    await user.type(input, 'Which model was used?{Enter}')
    expect(await screen.findByText('Generating grounded explanation…')).toBeInTheDocument()
    expect(ask()).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Explain this run' })).toBeDisabled()
    await user.type(input, '{Enter}')
    expect(sent()).toHaveLength(1)
    finish(new Response(JSON.stringify(explanation), { status: 200 }))
    expect(await screen.findByText(/status succeeded/)).toBeInTheDocument()
    expect(ask()).toBeEnabled()
  })

  it('replaces the previous answer on a follow-up question', async () => {
    const answers = [{ ...explanation, answer: 'First server answer.' }, { ...explanation, answer: 'Second server answer.' }]
    mock(() => json(answers.shift()))
    route()
    const user = userEvent.setup()
    const input = await question()
    await user.type(input, 'What is the status?{Enter}')
    expect(await screen.findByText('First server answer.')).toBeInTheDocument()
    await user.clear(input)
    await user.type(input, 'Which model was used?{Enter}')
    expect(await screen.findByText('Second server answer.')).toBeInTheDocument()
    expect(screen.queryByText('First server answer.')).not.toBeInTheDocument()
    expect(sent()).toEqual([{ question: 'What is the status?' }, { question: 'Which model was used?' }])
  })

  it('renders each citation with its artifact, schema and checksum', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'How many detections were recorded?{Enter}')
    expect(await screen.findByText('Evidence citations')).toBeInTheDocument()
    expect(screen.getByText('diagnostics.observed_metrics.total_tracklets')).toBeInTheDocument()
    expect(screen.getByText(/analysis-diagnostics \(footballai\.analysis-diagnostics\/v2\)/)).toHaveTextContent('aaaaaaaa')
  })

  it('renders a partial answer on an unverified run with its limits', async () => {
    mock(() => json({ ...explanation, support_level: 'partial', output_nature: 'unverified_or_partial' }))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'What do the tracking diagnostics tell us?{Enter}')
    expect(await screen.findByText('Partially supported')).toBeInTheDocument()
    expect(screen.getByText('Real inference not verified')).toBeInTheDocument()
    expect(screen.getByText('Important limitations')).toBeInTheDocument()
  })

  it('treats answer text as text, never as markup', async () => {
    mock(() => json({ ...explanation, answer: '<img src=x onerror=alert(1)>' }))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'What is the status?{Enter}')
    expect(await screen.findByText('<img src=x onerror=alert(1)>')).toBeInTheDocument()
    expect(document.querySelector('.explain-answer img')).toBeNull()
  })

  it('keeps the question after a failure so it can be retried', async () => {
    const replies = [json({ detail: { error_code: 'x', message: 'Evidence citation validation failed.' } }, 422), json(explanation)]
    mock(() => replies.shift() as Promise<Response>)
    route()
    const user = userEvent.setup()
    const input = await question()
    await user.type(input, 'Which model was used?{Enter}')
    expect(await screen.findByRole('alert')).toHaveTextContent('Evidence citation validation failed.')
    expect(input).toHaveValue('Which model was used?')
    await user.click(ask())
    expect(await screen.findByText(/status succeeded/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('does not show a raw network error', async () => {
    mock(() => Promise.reject(new TypeError('connect ECONNREFUSED 127.0.0.1:8000')))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'Which model was used?{Enter}')
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/could not reach/i)
    expect(alert).not.toHaveTextContent('ECONNREFUSED')
  })

  it('does not blame local data when the copilot fails', async () => {
    mock(() => json({ detail: 'Boom' }, 500))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'Which model was used?{Enter}')
    await screen.findByRole('alert')
    expect(screen.queryByText('Local data unavailable')).not.toBeInTheDocument()
  })

  it('removes the input when the feature is disabled', async () => {
    mock(() => json({ detail: 'Not Found' }, 404))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'Which model was used?{Enter}')
    expect(await screen.findByText(/disabled in this environment/)).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: 'Ask about this run' })).not.toBeInTheDocument()
  })

  it.each(['queued', 'running', 'failed'])('does not accept questions on a %s run', async (status) => {
    vi.mocked(fetch).mockImplementation((input) => String(input).endsWith('/artifacts') ? json(artifacts) : json({ ...detail, status }))
    route()
    expect(await question()).toBeDisabled()
    expect(ask()).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Explain this run' })).toBeDisabled()
    expect(screen.getByText(/Only completed/)).toBeInTheDocument()
  })

  it('announces results to assistive technology', async () => {
    mock(() => json(explanation))
    route()
    const user = userEvent.setup()
    await user.type(await question(), 'Which model was used?{Enter}')
    await screen.findByText(/status succeeded/)
    expect(document.querySelector('.explain-result')?.closest('[aria-live="polite"]')).not.toBeNull()
  })
})
