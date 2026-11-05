import { AlertTriangle, Bot, Box, CheckCircle2, ChevronRight, Clock3, FileCheck2, GitBranch, ShieldAlert, Sparkles } from 'lucide-react'
import { type FormEvent, useId, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, askAgent, explainRun, useApi } from '../api'
import { AgentTraceView } from '../components/AgentTraceView'
import { AnalysisWarning, ErrorState, LoadingState, OriginBadge, Panel, StatusBadge, formatDate, shortId } from '../components/ui'
import type { AgentAdapterInfo, AgentResponse, AgentTrace, ArtifactListResponse, ExplanationResponse, ModelMode, OutputNature, RunDetail, SupportLevel } from '../types'

const SUPPORT_LABEL: Record<SupportLevel, string> = {
  supported: 'Supported by evidence',
  partial: 'Partially supported',
  unavailable: 'Not answerable from evidence',
}

const MODE_LABEL: Record<ModelMode, string> = {
  offline_demo_fake: 'Offline deterministic demo — not a language model',
  real_model: 'Language model — selected verified claims only',
  not_invoked: 'No model invoked — deterministic response',
}

const NATURE_LABEL: Record<OutputNature, string> = {
  synthetic: 'Synthetic output',
  verified_real: 'Real inference — execution verified',
  unverified_or_partial: 'Real inference not verified',
}

const ADAPTER_LABEL: Record<string, string> = {
  offline_scripted: 'OFFLINE SCRIPTED AGENT',
  real_model: 'LANGUAGE MODEL AGENT',
}

const REASON_LABEL: Record<string, string> = {
  unsupported_question: 'FootballAI does not measure this, or the question is outside the run\u2019s evidence.',
  evidence_not_available: 'This run\u2019s verified evidence does not include this information.',
  no_relevant_claim: 'No verified evidence claim answers this question.',
  provider_unavailable: 'The explanation model is unavailable.',
  provider_timeout: 'The explanation model timed out.',
  prompt_budget_exceeded: 'The evidence for this run exceeds the safe prompt size.',
  cross_run_rejected: 'The request tried to reach another run, so it was stopped.',
  agent_limit_reached: 'The agent reached a safety limit, so no partial answer is shown.',
  forged_citation: 'The selected evidence failed server validation, so it was not shown.',
}

const MAX_QUESTION_CHARS = 500
const EXAMPLE_QUESTIONS = [
  'What do the tracking diagnostics tell us?',
  'How reliable was the tracking?',
  'How many successful passes did player 7 make?',
]

type CopilotMode = 'explanation' | 'agent'

const AGENT_ADAPTER_PATH = '/api/v1/copilot/agent'
type ExplanationState = { phase: 'idle' | 'loading' | 'done' | 'error'; data?: ExplanationResponse; trace?: AgentTrace; error?: string; asked?: string | null }

const MODES: { id: CopilotMode; label: string }[] = [
  { id: 'explanation', label: 'Grounded Explanation' },
  { id: 'agent', label: 'Read-Only Agent' },
]
const DISABLED_NOTICE: Record<CopilotMode, string> = {
  explanation: 'The explanation copilot is disabled in this environment. It is a development-only feature and is not exposed where access controls are not yet in place.',
  agent: 'The read-only agent is disabled in this environment. It needs its own development-only flag, and answers only on this machine. Grounded Explanation is unaffected by this flag.',
}

function failureMessage(reason: unknown) {
  // A network failure's own text can hold addresses, so it is never shown.
  return reason instanceof ApiError ? reason.message : 'The Copilot could not reach the local API. Check that it is running, then ask again.'
}

// The server says what drives the agent, so this banner can never promise the wrong thing.
function adapterBanner(adapter: AgentAdapterInfo['adapter'] | undefined) {
  if (adapter === 'offline_scripted') {
    return <><strong>{ADAPTER_LABEL.offline_scripted}</strong> A fixed script, not a language model, chooses the tools. The tool calls, limits, and evidence checks are real.</>
  }
  if (adapter === 'real_model') {
    return <><strong>{ADAPTER_LABEL.real_model}</strong> A language model chooses the tools and which claims to show. The server writes every sentence and checks every citation.</>
  }
  return 'The badge on each answer names what produced it.'
}

function ExplanationPanel({ runId, status }: { runId: string; status: RunDetail['status'] }) {
  const eligible = status === 'succeeded' || status === 'partial'
  const [mode, setMode] = useState<CopilotMode>('explanation')
  const [disabled, setDisabled] = useState<Record<CopilotMode, boolean>>({ explanation: false, agent: false })
  const [state, setState] = useState<ExplanationState>({ phase: 'idle' })
  const [question, setQuestion] = useState('')
  const adapter = useApi<AgentAdapterInfo>(mode === 'agent' ? AGENT_ADAPTER_PATH : null).data?.adapter
  const inputRef = useRef<HTMLInputElement>(null)
  const questionId = useId()
  const hintId = useId()
  const modeName = useId()
  const trimmed = question.trim()
  const loading = state.phase === 'loading'
  const tooLong = trimmed.length > MAX_QUESTION_CHARS
  const canAsk = eligible && !loading && trimmed.length > 0 && !tooLong

  async function run(asked: string | null) {
    setState({ phase: 'loading', asked })
    try {
      if (mode === 'agent') {
        const { explanation, trace } = await askAgent<AgentResponse>(runId, asked)
        setState({ phase: 'done', data: explanation, trace, asked })
      } else {
        const data = await explainRun<ExplanationResponse>(runId, asked)
        setState({ phase: 'done', data, asked })
      }
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 404) {
        setDisabled((current) => ({ ...current, [mode]: true }))
        setState({ phase: 'idle' })
      } else {
        setState({ phase: 'error', error: failureMessage(reason), asked })
      }
    }
  }

  // An answer belongs to the mode that produced it, so it cannot stay on screen in the other.
  function changeMode(next: CopilotMode) {
    setMode(next)
    setState({ phase: 'idle' })
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    if (canAsk) void run(trimmed)
  }

  function fillExample(example: string) {
    setQuestion(example)
    inputRef.current?.focus()
  }

  return (
    <Panel>
      <div className="panel-head"><div><span className="eyebrow">AI copilot</span><h2>AI Analysis Explanation</h2></div><Sparkles /></div>
      <p className="explain-intro">Ask one question about this run. Answers come only from its verified evidence, and each question stands alone. It never invents player identities, passes, possession, tactics, verified distances, or injury risk.</p>
      <fieldset className="explain-modes" disabled={loading}>
        <legend>Copilot mode</legend>
        {MODES.map((item) => (
          <label key={item.id} className={mode === item.id ? 'explain-mode-option explain-mode-option--selected' : 'explain-mode-option'}>
            <input type="radio" name={modeName} checked={mode === item.id} onChange={() => changeMode(item.id)} />
            {item.label}
          </label>
        ))}
      </fieldset>
      {mode === 'agent' && !disabled.agent && (
        <p className="agent-banner"><Bot /><span>{adapterBanner(adapter)}</span></p>
      )}
      {disabled[mode]
        ? <p className="explain-disabled" role="status"><AlertTriangle /> {DISABLED_NOTICE[mode]}</p>
        : (
          <>
            <form className="explain-form" onSubmit={submit}>
              <label htmlFor={questionId}>Ask about this run</label>
              <div className="explain-ask">
                <input ref={inputRef} id={questionId} type="text" value={question} onChange={(event) => setQuestion(event.target.value)} disabled={!eligible} readOnly={loading} autoComplete="off" placeholder="How reliable was the tracking?" aria-describedby={hintId} aria-invalid={tooLong} />
                <button className="button button--primary" type="submit" disabled={!canAsk}>Ask Copilot</button>
                <button className="button button--secondary" type="button" onClick={() => void run(null)} disabled={!eligible || loading}>Explain this run</button>
              </div>
              <small id={hintId} className="explain-hint">{trimmed.length}/{MAX_QUESTION_CHARS} characters</small>
              {tooLong && <small className="explain-invalid" role="alert">Shorten the question to {MAX_QUESTION_CHARS} characters or fewer.</small>}
              {!eligible && <small className="explain-hint">Only completed (succeeded or partial) runs can be explained.</small>}
            </form>
            {eligible && (
              <div className="explain-examples" aria-label="Example questions">
                {EXAMPLE_QUESTIONS.map((example) => <button key={example} type="button" onClick={() => fillExample(example)} disabled={loading}>{example}</button>)}
              </div>
            )}
          </>
        )}
      <div aria-live="polite">
        {loading && <LoadingState label={mode === 'agent' ? 'Running the read-only agent…' : 'Generating grounded explanation…'} />}
        {state.phase === 'error' && <p className="explain-failure" role="alert"><AlertTriangle /> {state.error}</p>}
        {state.phase === 'done' && state.data && <ExplanationResult data={state.data} trace={state.trace} asked={state.asked ?? null} />}
      </div>
    </Panel>
  )
}

function ExplanationResult({ data, trace, asked }: { data: ExplanationResponse; trace?: AgentTrace; asked: string | null }) {
  return (
    <div className="explain-result">
      {asked && <p className="explain-asked"><span>Question</span>{asked}</p>}
      <div className="explain-badges">
        <span className={`explain-support explain-support--${data.support_level}`}>{SUPPORT_LABEL[data.support_level]}</span>
        <span className={`explain-nature explain-nature--${data.output_nature}`}>{NATURE_LABEL[data.output_nature]}</span>
        <span className={`explain-mode explain-mode--${data.model_mode}`}>{MODE_LABEL[data.model_mode]}</span>
        {trace && <span className={`explain-adapter explain-adapter--${trace.adapter}`}>{ADAPTER_LABEL[trace.adapter] ?? trace.adapter}</span>}
      </div>
      <p className="explain-answer">{data.answer}</p>
      {data.unavailable_reason && <p className="explain-reason">{REASON_LABEL[data.unavailable_reason] ?? data.unavailable_reason}</p>}
      {data.citations.length > 0 && (
        <div className="explain-section">
          <h3>Evidence citations</h3>
          <ul className="explain-citations">
            {data.citations.map((citation, index) => (
              <li key={`${citation.field_path}-${index}`}>
                <code>{citation.field_path}</code>
                {citation.artifact_id && <span> · {citation.artifact_id}{citation.schema_version ? ` (${citation.schema_version})` : ''}{citation.source_sha256 ? ` · SHA-256 ${shortId(citation.source_sha256)}` : ''}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.limitations.length > 0 && (
        <div className="explain-section">
          <h3>Important limitations</h3>
          <ul className="explain-limitations">{data.limitations.map((item, index) => <li key={index}>{item}</li>)}</ul>
        </div>
      )}
      {trace && <AgentTraceView trace={trace} />}
    </div>
  )
}

export default function RunDetailPage() {
  const { runId } = useParams()
  const detail = useApi<RunDetail>(runId ? `/api/v1/runs/${runId}` : null)
  const artifacts = useApi<ArtifactListResponse>(runId ? `/api/v1/runs/${runId}/artifacts` : null)
  if (detail.loading) return <LoadingState label="Loading run provenance…" />
  if (detail.error || !detail.data) return <ErrorState message={detail.error ?? 'Run not found.'} />
  const run = detail.data
  const matchName = typeof run.provenance.parameters.match_name === 'string' ? run.provenance.parameters.match_name : null
  const profile = run.provenance.parameters.pipeline_profile
  const profileLabel = profile === 'demo_fast'
    ? 'Synthetic demo_fast · no YOLO inference'
    : profile === 'v1_compat' ? 'YOLOv8m + ByteTrack profile' : `Pipeline profile: ${String(profile ?? 'unknown')}`
  return (
    <div className="page">
      <div className="breadcrumbs"><Link to="/runs">Analysis runs</Link><ChevronRight /><span>{shortId(run.run_id)}</span></div>
      <div className="page-head run-hero">
        <div><span className="eyebrow">Immutable attempt #{run.attempt_number}</span><h1>{matchName || shortId(run.run_id)}</h1><p>{shortId(run.run_id)} · {run.contract_version} · created {formatDate(run.created_at)}</p><p className="run-profile-label">{profileLabel}</p></div>
        <div className="badge-row"><StatusBadge value={run.status} /><OriginBadge value={run.origin} /></div>
      </div>
      {run.warnings.length > 0 && <AnalysisWarning warnings={run.warnings} legacy={run.origin === 'legacy_v1'} />}
      <div className="action-strip">
        <Link className="button button--primary" to={run.status === 'succeeded' || run.status === 'partial' ? `/runs/${run.run_id}/team` : `/runs/${run.run_id}/progress`}>{run.status === 'succeeded' || run.status === 'partial' ? 'Open team overview' : 'Open live progress'} <ChevronRight /></Link>
        <span><FileCheck2 />{artifacts.data?.artifacts.filter((item) => item.integrity_state === 'verified').length ?? 0} verified artifacts</span>
      </div>
      <ExplanationPanel key={run.run_id} runId={run.run_id} status={run.status} />
      <div className="detail-grid">
        <Panel>
          <div className="panel-head"><div><span className="eyebrow">Execution</span><h2>Stage timeline</h2></div><Clock3 /></div>
          <ol className="stage-list">
            {run.stages.map((stage) => (
              <li key={stage.stage_id} className={`stage stage--${stage.status}`}>
                <span className="stage-dot">{stage.status === 'succeeded' ? <CheckCircle2 /> : <ShieldAlert />}</span>
                <div><div className="stage-title"><strong>{stage.stage_name.replaceAll('_', ' ')}</strong><StatusBadge value={stage.status} /></div><p>{stage.message ?? 'No stage message.'}</p><small>{stage.required ? 'Required stage' : 'Optional stage'} · {stage.progress_percent}% complete</small></div>
              </li>
            ))}
          </ol>
        </Panel>
        <div className="detail-stack">
          <Panel>
            <div className="panel-head"><div><span className="eyebrow">History</span><h2>Attempt chain</h2></div><GitBranch /></div>
            <div className="attempt-chain">{run.attempt_chain.map((attempt) => <Link key={attempt.run_id} to={`/runs/${attempt.run_id}/progress`}><span>Attempt {attempt.attempt_number}</span><strong>{shortId(attempt.run_id)}</strong><StatusBadge value={attempt.status} /></Link>)}</div>
          </Panel>
          <Panel>
            <div className="panel-head"><div><span className="eyebrow">Source</span><h2>Provenance</h2></div><Box /></div>
            <dl className="key-values">
              <div><dt>Input checksum</dt><dd title={run.provenance.input_checksum}>{shortId(run.provenance.input_checksum)}</dd></div>
              <div><dt>Code revision</dt><dd title={run.provenance.code_revision}>{shortId(run.provenance.code_revision)}</dd></div>
              <div><dt>Pipeline</dt><dd>{run.provenance.pipeline_version}</dd></div>
              <div><dt>Worktree</dt><dd>{run.provenance.code_dirty ? 'Dirty snapshot recorded' : 'Clean revision'}</dd></div>
            </dl>
          </Panel>
        </div>
      </div>
      <Panel>
        <div className="panel-head"><div><span className="eyebrow">Registered outputs</span><h2>Artifacts &amp; integrity</h2></div><FileCheck2 /></div>
        {artifacts.loading && <LoadingState label="Verifying artifacts…" />}
        {artifacts.error && <ErrorState message={artifacts.error} />}
        <div className="artifact-grid">{artifacts.data?.artifacts.map((item) => <article key={item.artifact_id}><div><span className={`integrity integrity--${item.integrity_state}`}>{item.integrity_state}</span><small>{item.category.replaceAll('_', ' ')}</small></div><h3>{item.name}</h3><p>{item.relative_path}</p><span>{(item.size_bytes / 1024).toFixed(1)} KB · SHA-256 {shortId(item.sha256)}</span></article>)}</div>
      </Panel>
    </div>
  )
}
