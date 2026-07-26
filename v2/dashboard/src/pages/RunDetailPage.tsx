import { Box, CheckCircle2, ChevronRight, Clock3, FileCheck2, GitBranch, ShieldAlert } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { useApi } from '../api'
import { ErrorState, LegacyWarning, LoadingState, OriginBadge, Panel, StatusBadge, formatDate, shortId } from '../components/ui'
import type { ArtifactListResponse, RunDetail } from '../types'

export default function RunDetailPage() {
  const { runId } = useParams()
  const detail = useApi<RunDetail>(runId ? `/api/v1/runs/${runId}` : null)
  const artifacts = useApi<ArtifactListResponse>(runId ? `/api/v1/runs/${runId}/artifacts` : null)
  if (detail.loading) return <LoadingState label="Loading run provenance…" />
  if (detail.error || !detail.data) return <ErrorState message={detail.error ?? 'Run not found.'} />
  const run = detail.data
  return (
    <div className="page">
      <div className="breadcrumbs"><Link to="/runs">Analysis runs</Link><ChevronRight /><span>{shortId(run.run_id)}</span></div>
      <div className="page-head run-hero">
        <div><span className="eyebrow">Immutable attempt #{run.attempt_number}</span><h1>{shortId(run.run_id)}</h1><p>{run.contract_version} · created {formatDate(run.created_at)}</p></div>
        <div className="badge-row"><StatusBadge value={run.status} /><OriginBadge value={run.origin} /></div>
      </div>
      {run.origin === 'legacy_v1' && <LegacyWarning warnings={run.warnings} />}
      <div className="action-strip">
        <Link className="button button--primary" to={`/runs/${run.run_id}/team`}>Open team overview <ChevronRight /></Link>
        <span><FileCheck2 />{artifacts.data?.artifacts.filter((item) => item.integrity_state === 'verified').length ?? 0} verified artifacts</span>
      </div>
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
            <div className="attempt-chain">{run.attempt_chain.map((attempt) => <div key={attempt.run_id}><span>Attempt {attempt.attempt_number}</span><strong>{shortId(attempt.run_id)}</strong><StatusBadge value={attempt.status} /></div>)}</div>
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
