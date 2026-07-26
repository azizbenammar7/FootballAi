import { Activity, ChevronRight, Gauge, Map, ShieldAlert, Timer } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { AnalysisWarning, ErrorState, LoadingState, MetricCard, Panel, StatusBadge, formatDistance, shortId } from '../components/ui'
import type { PlayerDetail } from '../types'

function Details({ values }: { values: Record<string, unknown> }) {
  const entries = Object.entries(values)
  if (!entries.length) return <p className="muted">No stable breakdown was recorded.</p>
  return <dl className="breakdown">{entries.map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{typeof value === 'number' ? value.toFixed(2) : String(value)}</dd></div>)}</dl>
}

export default function TrackPage() {
  const { runId, playerId } = useParams()
  const { data, loading, error } = useApi<PlayerDetail>(runId && playerId ? `/api/v1/runs/${runId}/players/${playerId}` : null)
  if (loading) return <LoadingState label="Loading unverified track…" />
  if (error || !data) return <ErrorState message={error ?? 'Track not found.'} />
  const maxHeat = Math.max(...data.heatmap.flat(), 1)
  return (
    <div className="page">
      <div className="breadcrumbs"><Link to="/runs">Analysis runs</Link><ChevronRight /><Link to={`/runs/${runId}`}>{shortId(runId!)}</Link><ChevronRight /><Link to={`/runs/${runId}/team`}>Team</Link><ChevronRight /><span>Track {data.player_id}</span></div>
      <div className="page-head track-hero"><div><span className="eyebrow">Identity not verified</span><h1>{data.label}</h1><p>Approximate legacy movement profile with explicit confidence limitations.</p></div><StatusBadge value={data.advisory.level} /></div>
      <AnalysisWarning warnings={data.warnings} legacy={data.warnings.some((item) => item.toLowerCase().includes('legacy'))} />
      <div className="metrics-grid metrics-grid--five">
        <MetricCard label="Distance" value={formatDistance(data.total_distance_m)} detail="Image-space estimate" />
        <MetricCard label="Average speed" value={`${data.average_speed_ms.toFixed(2)} m/s`} />
        <MetricCard label="Peak speed" value={`${data.peak_speed_ms.toFixed(2)} m/s`} />
        <MetricCard label="Sprint count" value={String(data.sprint_count)} />
        <MetricCard label="Active span" value={`${data.active_span_seconds.toFixed(0)} s`} detail="Approximate track span" />
      </div>
      <div className="chart-grid">
        <Panel><div className="panel-head"><div><span className="eyebrow">Movement profile</span><h2>Speed timeline</h2></div><Activity /></div><div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data.speed_timeline}><CartesianGrid strokeDasharray="3 3" stroke="#1d322a" /><XAxis dataKey="minute" tickFormatter={(value) => `${value}'`} stroke="#8ba096" /><YAxis stroke="#8ba096" /><Tooltip contentStyle={{ background: '#0d1f18', border: '1px solid #284438', borderRadius: 10 }} /><Line type="monotone" dataKey="value" name="Speed (m/s)" stroke="#62a7ff" strokeWidth={3} dot={{ fill: '#62a7ff' }} /></LineChart></ResponsiveContainer></div></Panel>
        <Panel><div className="panel-head"><div><span className="eyebrow">Approximate accumulation</span><h2>Distance timeline</h2></div><Timer /></div><div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data.distance_timeline}><CartesianGrid strokeDasharray="3 3" stroke="#1d322a" /><XAxis dataKey="minute" tickFormatter={(value) => `${value}'`} stroke="#8ba096" /><YAxis stroke="#8ba096" /><Tooltip contentStyle={{ background: '#0d1f18', border: '1px solid #284438', borderRadius: 10 }} /><Line type="monotone" dataKey="value" name="Distance (m)" stroke="#36d889" strokeWidth={3} dot={{ fill: '#36d889' }} /></LineChart></ResponsiveContainer></div></Panel>
      </div>
      <div className="detail-grid">
        <Panel><div className="panel-head"><div><span className="eyebrow">Spatial distribution</span><h2>Legacy heatmap</h2></div><Map /></div><div className="pitch-heatmap" style={{ gridTemplateColumns: `repeat(${data.heatmap[0]?.length ?? 1}, 1fr)` }}>{data.heatmap.flatMap((row, rowIndex) => row.map((value, columnIndex) => <span key={`${rowIndex}-${columnIndex}`} style={{ backgroundColor: `rgba(54, 216, 137, ${0.08 + (value / maxHeat) * 0.88})` }} title={`Cell intensity ${value.toFixed(2)}`} />))}<i className="pitch-line pitch-line--half" /></div><p className="chart-note">Normalized image-space occupancy. Not homography-calibrated to pitch coordinates.</p></Panel>
        <Panel className="advisory-panel"><div className="panel-head"><div><span className="eyebrow">Advisory only</span><h2>{data.advisory.label}</h2></div><Gauge /></div><div className="advisory-score"><StatusBadge value={data.advisory.level} /><strong>{data.advisory.score == null ? '—' : data.advisory.score.toFixed(2)}</strong><span>heuristic score</span></div>{data.advisory.reason && <p>{data.advisory.reason}</p>}<h3>Recorded indicators</h3><Details values={data.advisory.indicators} /><h3>Score breakdown</h3><Details values={data.advisory.breakdown} /><div className="advisory-disclaimer"><ShieldAlert /><span>This is workload context for review, not diagnosis or clinical advice.</span></div></Panel>
      </div>
    </div>
  )
}
