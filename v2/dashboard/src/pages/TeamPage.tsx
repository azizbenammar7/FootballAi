import { Activity, ChevronRight, Gauge, Timer, UsersRound } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api'
import { ErrorState, LegacyWarning, LoadingState, MetricCard, Panel, StatusBadge, formatDistance, shortId } from '../components/ui'
import type { PlayerListResponse, TeamSummary } from '../types'

const advisoryColors: Record<string, string> = { LOW: '#4ae89a', MEDIUM: '#f7c65d', HIGH: '#ff7b67', INSUFFICIENT: '#778a83', UNAVAILABLE: '#55645f' }

export default function TeamPage() {
  const { runId } = useParams()
  const summary = useApi<TeamSummary>(runId ? `/api/v1/runs/${runId}/summary` : null)
  const players = useApi<PlayerListResponse>(runId ? `/api/v1/runs/${runId}/players` : null)
  if (summary.loading || players.loading) return <LoadingState label="Building team overview…" />
  if (summary.error || players.error || !summary.data || !players.data) return <ErrorState message={summary.error ?? players.error ?? 'Team data unavailable.'} />
  const data = summary.data
  const advisoryData = Object.entries(data.advisory_distribution).map(([name, value]) => ({ name, value }))
  return (
    <div className="page">
      <div className="breadcrumbs"><Link to="/runs">Analysis runs</Link><ChevronRight /><Link to={`/runs/${runId}`}>{shortId(runId!)}</Link><ChevronRight /><span>Team overview</span></div>
      <div className="page-head"><div><span className="eyebrow">Legacy team view</span><h1>Team overview</h1><p>Aggregate movement and advisory context across unverified tracks.</p></div><span className="match-clock"><Timer />{(data.match_duration_seconds / 60).toFixed(1)} min</span></div>
      <LegacyWarning warnings={data.warnings} />
      <div className="metrics-grid">
        <MetricCard label="Unverified tracks" value={String(data.total_tracks)} detail={`${data.scored_tracks} advisory-scored`} />
        <MetricCard label="Aggregate distance" value={formatDistance(data.distance.total_m)} detail={`${formatDistance(data.distance.average_per_track_m)} average`} />
        <MetricCard label="Insufficient tracks" value={String(data.insufficient_tracks)} detail="Coverage gate applied" />
        <MetricCard label="Longest track" value={formatDistance(data.distance.maximum_track_m)} detail="Approximate image-space metric" />
      </div>
      <div className="chart-grid">
        <Panel>
          <div className="panel-head"><div><span className="eyebrow">15-minute blocks</span><h2>Estimated distance &amp; mean speed</h2></div><Activity /></div>
          <div className="chart" aria-label="15-minute team block chart">
            <ResponsiveContainer width="100%" height="100%"><BarChart data={data.blocks} margin={{ top: 12, right: 8, left: 0, bottom: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="#1d322a" /><XAxis dataKey="start_minute" tickFormatter={(value) => `${value}'`} stroke="#8ba096" /><YAxis yAxisId="distance" stroke="#8ba096" /><YAxis yAxisId="speed" orientation="right" stroke="#8ba096" /><Tooltip contentStyle={{ background: '#0d1f18', border: '1px solid #284438', borderRadius: 10 }} /><Legend /><Bar yAxisId="distance" name="Estimated distance (m)" dataKey="estimated_distance_m" fill="#36d889" radius={[5, 5, 0, 0]} /><Bar yAxisId="speed" name="Average speed (m/s)" dataKey="average_speed_ms" fill="#62a7ff" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer>
          </div>
        </Panel>
        <Panel>
          <div className="panel-head"><div><span className="eyebrow">Advisory coverage</span><h2>Track distribution</h2></div><Gauge /></div>
          <div className="chart chart--pie" aria-label="Advisory distribution chart">
            <ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={advisoryData} dataKey="value" nameKey="name" innerRadius={62} outerRadius={92} paddingAngle={3}>{advisoryData.map((item) => <Cell key={item.name} fill={advisoryColors[item.name] ?? '#778a83'} />)}</Pie><Tooltip contentStyle={{ background: '#0d1f18', border: '1px solid #284438', borderRadius: 10 }} /><Legend /></PieChart></ResponsiveContainer>
          </div>
          <p className="chart-note">Advisory distribution is heuristic and coverage-gated; it is not a clinical assessment.</p>
        </Panel>
      </div>
      <Panel>
        <div className="panel-head"><div><span className="eyebrow">Track roster</span><h2>Unverified player tracks</h2></div><UsersRound /></div>
        <div className="track-grid">
          {players.data.players.slice(0, 24).map((player) => (
            <Link key={player.player_id} to={`/runs/${runId}/tracks/${player.player_id}`} className="track-card">
              <div><span className="track-number">#{player.player_id}</span><StatusBadge value={player.advisory_level} /></div>
              <h3>{player.label}</h3><p>Identity not verified · {(player.coverage_fraction * 100).toFixed(0)}% coverage</p>
              <dl><div><dt>Distance</dt><dd>{formatDistance(player.total_distance_m)}</dd></div><div><dt>Peak</dt><dd>{player.peak_speed_ms.toFixed(1)} m/s</dd></div><div><dt>Sprints</dt><dd>{player.sprint_count}</dd></div></dl>
              <span className="card-link">Open track analysis <ChevronRight /></span>
            </Link>
          ))}
        </div>
      </Panel>
    </div>
  )
}
