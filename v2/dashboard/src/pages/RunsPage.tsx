import { ArrowRight, Database, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useApi } from '../api'
import { EmptyState, ErrorState, LoadingState, OriginBadge, Panel, StatusBadge, formatDate, shortId } from '../components/ui'
import type { RunListResponse } from '../types'

export default function RunsPage() {
  const { data, loading, error } = useApi<RunListResponse>('/api/v1/runs')
  return (
    <div className="page">
      <div className="page-head page-head--hero">
        <div>
          <span className="eyebrow">Versioned analysis registry</span>
          <h1>Analysis runs</h1>
          <p>Inspect immutable attempts, stage progress, provenance and artifact integrity.</p>
        </div>
        <div className="hero-signal"><ShieldCheck /><span><strong>Local &amp; isolated</strong>Caller-configured run storage</span></div>
      </div>
      {loading && <LoadingState label="Loading local analysis runs…" />}
      {error && <ErrorState message={error} />}
      {!loading && !error && data?.runs.length === 0 && (
        <EmptyState title="No V2 runs yet" message="Import the preserved V1 demo artifacts to create the first isolated run." />
      )}
      {data && data.runs.length > 0 && (
        <Panel className="run-registry">
          <div className="panel-head"><div><span className="eyebrow">Run store</span><h2>{data.runs.length} analysis attempt{data.runs.length === 1 ? '' : 's'}</h2></div><Database /></div>
          <div className="run-table" role="table" aria-label="Analysis runs">
            <div className="run-row run-row--head" role="row">
              <span>Run</span><span>State</span><span>Origin</span><span>Attempt</span><span>Stage progress</span><span>Created</span><span aria-label="Actions" />
            </div>
            {data.runs.map((run) => (
              <div className="run-row" role="row" key={run.run_id}>
                <div><strong>{shortId(run.run_id)}</strong><small>Logical {shortId(run.logical_analysis_id)}</small></div>
                <StatusBadge value={run.status} />
                <OriginBadge value={run.origin} />
                <div><strong>#{run.attempt_number}</strong><small>{run.pipeline_version}</small></div>
                <div className="progress-cell"><div className="progress"><i style={{ width: `${Math.min(run.stage_progress_percent, 100)}%` }} /></div><small>{run.stage_progress_percent.toFixed(0)}% · {run.warning_count} notices</small></div>
                <time>{formatDate(run.created_at)}</time>
                <Link className="icon-link" to={`/runs/${run.run_id}`} aria-label={`View run ${run.run_id}`}>View run <ArrowRight /></Link>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  )
}
