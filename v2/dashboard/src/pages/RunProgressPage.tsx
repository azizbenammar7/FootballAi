import { CheckCircle2, Circle, Clock3, RotateCcw, Square, XCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { apiGet, apiPost } from '../api'
import { AnalysisWarning, ErrorState, LoadingState, Panel, StatusBadge, formatDate, shortId } from '../components/ui'
import type { RunDetail, RunProgress } from '../types'

export default function RunProgressPage() {
  const { runId } = useParams(); const navigate = useNavigate()
  const [progress, setProgress] = useState<RunProgress | null>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (!runId) return
    let stopped = false; let timer: number | undefined
    async function poll() {
      if (document.visibilityState === 'hidden') { timer = window.setTimeout(poll, 1500); return }
      try {
        const next = await apiGet<RunProgress>(`/api/v1/runs/${runId}/progress`)
        if (stopped) return
        setProgress(next); setDetail(await apiGet<RunDetail>(`/api/v1/runs/${runId}`))
        if (!['succeeded', 'partial', 'failed', 'cancelled'].includes(next.status)) timer = window.setTimeout(poll, 1000)
      } catch (reason) { if (!stopped) setError(reason instanceof Error ? reason.message : 'Progress is unavailable.') }
    }
    void poll(); return () => { stopped = true; if (timer) window.clearTimeout(timer) }
  }, [runId])
  async function operation(kind: 'cancel' | 'retry' | 'clone') {
    if (!runId) return
    if (kind === 'cancel' && !window.confirm('Cancel this analysis at the next safe checkpoint?')) return
    setBusy(true); setError(null)
    try {
      const result = await apiPost<{ run_id: string }>(`/api/v1/runs/${runId}/${kind}`)
      if (kind === 'cancel') window.location.reload()
      else navigate(`/runs/${result.run_id}/progress`)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Operation failed safely.'); setBusy(false) }
  }
  if (error && !progress) return <ErrorState message={error} />
  if (!progress || !detail) return <LoadingState label="Loading live analysis progress…" />
  const terminal = ['succeeded', 'partial', 'failed', 'cancelled'].includes(progress.status)
  const failure = detail.failure as { error_code?: string; safe_message?: string } | null
  return (
    <div className="page">
      <div className="page-head run-hero"><div><span className="eyebrow">Attempt {progress.attempt_number} · {shortId(progress.run_id)}</span><h1>{terminal ? progress.status === 'succeeded' ? 'Analysis complete' : 'Analysis stopped' : 'Analysis in progress'}</h1><p>Created {formatDate(progress.created_at)} · {progress.active_stage?.replaceAll('_', ' ') ?? 'No active stage'}</p></div><StatusBadge value={progress.status} /></div>
      {detail.warnings.length > 0 && <AnalysisWarning warnings={detail.warnings} legacy={detail.origin === 'legacy_v1'} />}
      <Panel className="progress-overview"><div className="progress-number"><strong>{progress.overall_progress_percent.toFixed(0)}%</strong><span>overall progress</span></div><div className="progress progress--large"><i style={{ width: `${progress.overall_progress_percent}%` }} /></div><p>{terminal ? `Attempt finished with state ${progress.status}.` : `Current stage: ${progress.active_stage?.replaceAll('_', ' ') ?? 'waiting for worker'}`}</p></Panel>
      {(failure || detail.partial_reason || detail.cancellation_reason) && <div className="failure-card" role="alert"><XCircle /><div><strong>{failure?.error_code ?? progress.status}</strong><p>{failure?.safe_message ?? detail.partial_reason ?? detail.cancellation_reason}</p></div></div>}
      <div className="workflow-actions">
        {progress.can_cancel && <button className="button button--danger" disabled={busy} onClick={() => void operation('cancel')}><Square />Cancel analysis</button>}
        {progress.can_retry && <button className="button button--primary" disabled={busy} onClick={() => void operation('retry')}><RotateCcw />Retry as new attempt</button>}
        {progress.can_create_new_from_input && terminal && <button className="button button--secondary" disabled={busy} onClick={() => void operation('clone')}>Create new analysis from this input</button>}
        {progress.status === 'succeeded' && <Link className="button button--primary" to={`/runs/${runId}/team`}>Open results</Link>}
        <Link className="button button--secondary" to={`/runs/${runId}`}>Open attempt details</Link>
      </div>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="detail-grid">
        <Panel><div className="panel-head"><div><span className="eyebrow">Live execution</span><h2>Stage timeline</h2></div><Clock3 /></div><ol className="stage-list">{progress.stages.map((stage) => <li className={`stage stage--${stage.status}`} key={stage.stage_id}><span className="stage-dot">{stage.status === 'succeeded' ? <CheckCircle2 /> : stage.status === 'failed' ? <XCircle /> : <Circle />}</span><div><div className="stage-title"><strong>{stage.stage_name.replaceAll('_', ' ')}</strong><StatusBadge value={stage.status} /></div><div className="progress"><i style={{ width: `${stage.progress_percent}%` }} /></div><small>{stage.message ?? 'Waiting'} · {stage.progress_percent}%</small></div></li>)}</ol></Panel>
        <Panel><div className="panel-head"><div><span className="eyebrow">Immutable history</span><h2>Attempt chain</h2></div><RotateCcw /></div><div className="attempt-chain">{detail.attempt_chain.map((attempt) => <Link key={attempt.run_id} to={`/runs/${attempt.run_id}/progress`}><span>Attempt {attempt.attempt_number}</span><strong>{shortId(attempt.run_id)}</strong><StatusBadge value={attempt.status} /></Link>)}</div></Panel>
      </div>
    </div>
  )
}
