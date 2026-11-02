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

const CORRELATION = '6f1d2c3b-0a4e-4b7c-9d21-0123456789ab'
const toolCall = (overrides: Record<string, unknown> = {}) => ({
  sequence: 1, turn: 1, tool: 'get_tracking_diagnostics', status: 'ok', duration_ms: 0.21,
  claim_ids_returned: ['diag_tracking', 'diag_duration'], error_code: null, ...overrides,
})
const trace = (overrides: Record<string, unknown> = {}) => ({
  schema_version: 'footballai.copilot-agent-public-trace/v1', correlation_id: CORRELATION, run_id: RUN,
  adapter: 'offline_scripted', model: 'scripted-stub', model_mode: 'offline_demo_fake',
  instruction_version: 'copilot.agent-instructions/v1', routing_version: 'copilot.routing/v2', termination: 'answered',
  tool_error_code: null, duration_ms: 12.5, model_turns: 2, model_turn_limit: 5, tool_call_limit: 4,
  tool_calls: [toolCall()], tool_calls_refused: 0, claim_ids_returned: ['diag_tracking', 'diag_duration'],
  claim_ids_selected: ['diag_tracking', 'diag_duration'], claim_ids_rendered: ['output_nature', 'diag_tracking', 'diag_duration'],
  selected_ids_rejected: 0, ...overrides,
})
const agentReply = (explanationOverrides: Record<string, unknown> = {}, traceOverrides: Record<string, unknown> = {}) => ({
  explanation: { ...explanation, answer: 'Tracking diagnostics report 362 temporary tracklets.', ...explanationOverrides },
  trace: trace(traceOverrides),
})
const ABSTENTION = {
  answer: "I can't answer that from verified evidence.", support_level: 'unavailable', model_mode: 'not_invoked',
  citations: [], unavailable_reason: 'unsupported_question',
}
const ABSTAINED_TRACE = {
  model_mode: 'not_invoked', termination: 'unsupported_question', model_turns: 0, tool_calls: [],
  claim_ids_returned: [], claim_ids_selected: [], claim_ids_rendered: [],
}

describe('run detail agent mode', () => {
  const question = () => screen.findByRole('textbox', { name: 'Ask about this run' })
  const agentMode = () => screen.findByRole('radio', { name: 'Read-Only Agent' })
  const explanationMode = () => screen.getByRole('radio', { name: 'Grounded Explanation' })
  const posted = (suffix: string) => vi.mocked(fetch).mock.calls
    .filter(([url, init]) => init?.method === 'POST' && String(url).endsWith(suffix))
    .map(([, init]) => JSON.parse(String(init?.body)) as { question: string | null })
  function mockBoth(agentResult: () => Promise<Response>, explainResult: () => Promise<Response> = () => json(explanation)) {
    vi.mocked(fetch).mockImplementation((input, init) => {
      const url = String(input)
      if (init?.method === 'POST' && url.endsWith('/agent')) return agentResult()
      if (init?.method === 'POST' && url.endsWith('/explain')) return explainResult()
      if (url.endsWith('/artifacts')) return json(artifacts)
      return json(detail)
    })
  }
  async function askAgent(text = 'What do the tracking diagnostics tell us?') {
    const user = userEvent.setup()
    await user.click(await agentMode())
    await user.type(await question(), `${text}{Enter}`)
    return user
  }

  it('starts in grounded explanation mode and does not call the agent', async () => {
    mockBoth(() => json(agentReply()))
    route()
    const user = userEvent.setup()
    expect(await screen.findByRole('radio', { name: 'Grounded Explanation' })).toBeChecked()
    await user.type(await question(), 'Which model was used?{Enter}')
    await screen.findByText(/status succeeded/)
    expect(posted('/agent')).toEqual([])
    expect(posted('/explain')).toHaveLength(1)
    expect(screen.queryByText('Tool execution trace', { exact: false })).not.toBeInTheDocument()
  })

  it('labels the agent as an offline script before anything is asked', async () => {
    mockBoth(() => json(agentReply()))
    route()
    const user = userEvent.setup()
    await user.click(await agentMode())
    expect(screen.getByText('OFFLINE SCRIPTED AGENT')).toBeInTheDocument()
    expect(screen.getByText(/not a language model/)).toBeInTheDocument()
    expect(posted('/agent')).toEqual([])
  })

  it('sends the question to the agent route of the selected run', async () => {
    mockBoth(() => json(agentReply()))
    route()
    await askAgent()
    await screen.findByText(/362 temporary tracklets/)
    expect(posted('/agent')).toEqual([{ question: 'What do the tracking diagnostics tell us?' }])
    expect(posted('/explain')).toEqual([])
    expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).endsWith(`/api/v1/runs/${RUN}/agent`))).toBe(true)
  })

  it('shows the grounded answer with its citations and support and nature badges', async () => {
    mockBoth(() => json(agentReply()))
    route()
    await askAgent()
    expect(await screen.findByText(/362 temporary tracklets/)).toBeInTheDocument()
    expect(screen.getByText('Supported by evidence')).toBeInTheDocument()
    expect(screen.getByText('Real inference — execution verified')).toBeInTheDocument()
    expect(screen.getByText('Evidence citations')).toBeInTheDocument()
    expect(screen.getByText('diagnostics.observed_metrics.total_tracklets')).toBeInTheDocument()
  })

  it('labels every scripted answer and never calls it a language model', async () => {
    mockBoth(() => json(agentReply()))
    route()
    await askAgent()
    await screen.findByText(/362 temporary tracklets/)
    expect(document.querySelector('.explain-result .explain-adapter')).toHaveTextContent('OFFLINE SCRIPTED AGENT')
    expect(screen.queryByText(/selected verified claims only/)).not.toBeInTheDocument()
    expect(screen.queryByText(/LANGUAGE MODEL AGENT/)).not.toBeInTheDocument()
  })

  it('keeps the offline label on an abstention, where no model ran', async () => {
    mockBoth(() => json(agentReply(ABSTENTION, ABSTAINED_TRACE)))
    route()
    await askAgent('How many successful passes did player 7 make?')
    expect(await screen.findByText('Not answerable from evidence')).toBeInTheDocument()
    expect(document.querySelector('.explain-result .explain-adapter')).toHaveTextContent('OFFLINE SCRIPTED AGENT')
    expect(screen.getByText('No model invoked — deterministic response')).toBeInTheDocument()
    expect(screen.queryByText('Evidence citations')).not.toBeInTheDocument()
  })

  it('shows a real adapter only when the server reports one', async () => {
    mockBoth(() => json(agentReply({ model_mode: 'real_model' }, { adapter: 'real_model', model: 'claude-haiku-5-5', model_mode: 'real_model' })))
    route()
    await askAgent()
    await screen.findByText(/362 temporary tracklets/)
    expect(document.querySelector('.explain-result .explain-adapter')).toHaveTextContent('LANGUAGE MODEL AGENT')
  })

  it('keeps the trace collapsed until it is opened', async () => {
    mockBoth(() => json(agentReply()))
    route()
    const user = await askAgent()
    const summary = await screen.findByText(/Tool execution trace/)
    expect(screen.getByText('get_tracking_diagnostics')).not.toBeVisible()
    await user.click(summary)
    expect(screen.getByText('get_tracking_diagnostics')).toBeVisible()
  })

  it('shows each tool, its result, its claims and how the request ended', async () => {
    mockBoth(() => json(agentReply()))
    route()
    await askAgent()
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect(panel).toHaveTextContent('1 of 4 tool calls')
    expect(panel).toHaveTextContent('Answered')
    expect(panel).toHaveTextContent('get_tracking_diagnostics')
    expect(panel).toHaveTextContent('Executed — evidence returned')
    expect(panel).toHaveTextContent('diag_duration')
    expect(panel).toHaveTextContent('2 of 5')
    expect(panel).toHaveTextContent('scripted-stub')
    expect(panel).toHaveTextContent(CORRELATION)
  })

  it('lists only calls the server reported, in the order it reported them', async () => {
    const calls = [toolCall({ sequence: 1, tool: 'get_run_status' }), toolCall({ sequence: 2, tool: 'get_model_provenance', claim_ids_returned: [] , status: 'no_evidence' })]
    mockBoth(() => json(agentReply({}, { tool_calls: calls })))
    route()
    await askAgent()
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect([...panel.querySelectorAll('.agent-trace__tool > div > code')].map((node) => node.textContent)).toEqual(['get_run_status', 'get_model_provenance'])
    expect(panel).not.toHaveTextContent('get_tracking_diagnostics')
    expect(panel).toHaveTextContent('Executed — nothing for this question')
  })

  it('says no tool ran when the server stopped the question first', async () => {
    mockBoth(() => json(agentReply(ABSTENTION, ABSTAINED_TRACE)))
    route()
    await askAgent('How many successful passes did player 7 make?')
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect(panel).toHaveTextContent('No tool ran')
    expect(panel).toHaveTextContent('0 of 4 tool calls')
    expect(panel).toHaveTextContent('Question outside FootballAI')
    expect(panel.querySelectorAll('.agent-trace__tool')).toHaveLength(0)
  })

  it('never counts a rejected call as executed', async () => {
    const calls = [toolCall(), toolCall({ sequence: 2, tool: null, status: 'rejected', claim_ids_returned: [], error_code: 'unknown_tool' })]
    mockBoth(() => json(agentReply({}, { tool_calls: calls })))
    route()
    await askAgent()
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect(panel).toHaveTextContent('1 of 4 tool calls')
    expect(panel).toHaveTextContent('Rejected — never ran')
    expect(panel).toHaveTextContent('Outside the tool registry')
    expect(panel).toHaveTextContent('unknown_tool')
  })

  it('shows a tool error that ended the request', async () => {
    mockBoth(() => json(agentReply({ ...ABSTENTION, unavailable_reason: 'cross_run_rejected' }, { termination: 'cross_run_rejected', tool_error_code: 'cross_run_reference', tool_calls: [], model_turns: 1 })))
    route()
    await askAgent()
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect(panel).toHaveTextContent('cross_run_reference')
    expect(panel).toHaveTextContent('Cross-run reference rejected')
  })

  it('shows a value it does not know as it came, never as a blank or undefined', async () => {
    mockBoth(() => json(agentReply({}, { termination: 'a_new_reason', adapter: 'a_new_adapter', tool_calls: [toolCall({ status: 'a_new_status' })] })))
    route()
    await askAgent()
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect(panel).toHaveTextContent('a_new_reason')
    expect(panel).toHaveTextContent('a_new_status')
    expect(panel).not.toHaveTextContent('undefined')
    expect(document.querySelector('.explain-result .explain-adapter')).toHaveTextContent('a_new_adapter')
  })

  it('reports calls refused before running and rejected selections', async () => {
    mockBoth(() => json(agentReply(
      { ...ABSTENTION, model_mode: 'offline_demo_fake', unavailable_reason: 'agent_limit_reached' },
      { termination: 'tool_budget_exhausted', tool_calls: [], tool_calls_refused: 5, selected_ids_rejected: 2, model_turns: 1 },
    )))
    route()
    await askAgent()
    const panel = (await screen.findByText(/Tool execution trace/)).closest('details') as HTMLElement
    expect(panel).toHaveTextContent('5 requested calls were refused before running')
    expect(panel).toHaveTextContent('2 selected ids were rejected')
    expect(panel).toHaveTextContent('Tool limit reached')
    expect(panel).toHaveTextContent('The adapter called no tool')
  })

  it.each([
    ['cross_run_rejected', /tried to reach another run/],
    ['agent_limit_reached', /safety limit/],
    ['forged_citation', /failed server validation/],
    ['provider_timeout', /timed out/],
  ])('explains %s without showing the raw code', async (unavailable_reason, label) => {
    mockBoth(() => json(agentReply({ ...ABSTENTION, unavailable_reason }, { model_turns: 1, tool_calls: [] })))
    route()
    await askAgent()
    expect(await screen.findByText(label)).toBeInTheDocument()
    expect(screen.queryByText(unavailable_reason)).not.toBeInTheDocument()
  })

  it('shows a neutral loading state, and no tool steps until the server reports them', async () => {
    let finish: (value: Response) => void = () => {}
    mockBoth(() => new Promise<Response>((resolve) => { finish = resolve }))
    route()
    await askAgent()
    expect(await screen.findByText('Running the read-only agent…')).toBeInTheDocument()
    expect(screen.queryByText(/get_/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Tool execution trace/)).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Read-Only Agent' })).toBeDisabled()
    finish(new Response(JSON.stringify(agentReply()), { status: 200 }))
    expect(await screen.findByText(/362 temporary tracklets/)).toBeInTheDocument()
  })

  it('shows an unavailable state for the agent, and keeps grounded explanation working', async () => {
    mockBoth(() => json({ detail: 'Not Found' }, 404))
    route()
    const user = await askAgent()
    expect(await screen.findByText(/read-only agent is disabled in this environment/i)).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: 'Ask about this run' })).not.toBeInTheDocument()
    expect(screen.queryByText('OFFLINE SCRIPTED AGENT')).not.toBeInTheDocument()
    await user.click(explanationMode())
    await user.type(await question(), 'Which model was used?{Enter}')
    expect(await screen.findByText(/status succeeded/)).toBeInTheDocument()
    expect(screen.queryByText(/disabled in this environment/)).not.toBeInTheDocument()
  })

  it('shows a failure and keeps the question for a retry', async () => {
    mockBoth(() => json({ detail: { error_code: 'run_not_eligible', message: 'Only completed runs can be explained.' } }, 409))
    route()
    await askAgent()
    expect(await screen.findByRole('alert')).toHaveTextContent('Only completed runs can be explained.')
    expect(await question()).toHaveValue('What do the tracking diagnostics tell us?')
    expect(screen.queryByText(/Tool execution trace/)).not.toBeInTheDocument()
  })

  it('does not show a raw network error', async () => {
    mockBoth(() => Promise.reject(new TypeError('connect ECONNREFUSED 127.0.0.1:8000')))
    route()
    await askAgent()
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/could not reach/i)
    expect(alert).not.toHaveTextContent('ECONNREFUSED')
  })

  it('clears an answer when the mode changes, so it cannot pass for the other mode', async () => {
    mockBoth(() => json(agentReply()))
    route()
    const user = await askAgent()
    await screen.findByText(/362 temporary tracklets/)
    await user.click(explanationMode())
    expect(screen.queryByText(/362 temporary tracklets/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Tool execution trace/)).not.toBeInTheDocument()
    expect(await question()).toHaveValue('What do the tracking diagnostics tell us?')
  })

  it('treats trace values as text, never as markup', async () => {
    mockBoth(() => json(agentReply({}, { model: '<img src=x onerror=alert(1)>' })))
    route()
    await askAgent()
    await screen.findByText(/Tool execution trace/)
    expect(document.querySelector('.agent-trace img')).toBeNull()
  })

  it('offers the general request in agent mode as a request with no question', async () => {
    mockBoth(() => json(agentReply()))
    route()
    const user = userEvent.setup()
    await user.click(await agentMode())
    await user.click(screen.getByRole('button', { name: 'Explain this run' }))
    await screen.findByText(/362 temporary tracklets/)
    expect(posted('/agent')).toEqual([{ question: null }])
  })

  it('does not accept questions in agent mode on an unfinished run', async () => {
    vi.mocked(fetch).mockImplementation((input) => String(input).endsWith('/artifacts') ? json(artifacts) : json({ ...detail, status: 'running' }))
    route()
    const user = userEvent.setup()
    await user.click(await agentMode())
    expect(await question()).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Ask Copilot' })).toBeDisabled()
  })
})
