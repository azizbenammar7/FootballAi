import { AlertTriangle, Map as MapIcon } from 'lucide-react'
import { type FormEvent, useId, useState } from 'react'
import { useApi } from '../api'
import type { FrameCalibrationStatus, PitchGeometryView, PitchSpaceView } from '../types'
import { ErrorState, LoadingState, MetricCard, Panel, shortId } from './ui'

const EVIDENCE_LABEL: Record<PitchSpaceView['evidence_status'], string> = {
  no_annotations: 'No pitch landmark annotations',
  manual_annotations: 'Manual landmark annotations',
  synthetic_annotations: 'Synthetic annotations — not real evidence',
}

const STATUS_LABEL: Record<FrameCalibrationStatus, string> = {
  calibrated: 'Calibrated',
  uncalibrated: 'Uncalibrated',
  invalidated: 'Invalidated',
}

function percent(value: number | null) {
  return value === null ? 'n/a' : `${(value * 100).toFixed(1)}%`
}

function seconds(value: number) {
  return `${value.toFixed(1)} s`
}

function PitchMarkings({ pitch }: { pitch: PitchGeometryView }) {
  const { length_m: length, width_m: width, markings } = pitch
  const box = (depth: number, span: number, right: boolean) => (
    <rect x={right ? length - depth : 0} y={(width - span) / 2} width={depth} height={span} />
  )
  return (
    <g className="pitch-svg__lines">
      <rect x={0} y={0} width={length} height={width} />
      <line x1={length / 2} y1={0} x2={length / 2} y2={width} />
      <circle cx={length / 2} cy={width / 2} r={markings.centre_circle_radius_m} />
      {box(markings.penalty_area_depth_m, markings.penalty_area_width_m, false)}
      {box(markings.penalty_area_depth_m, markings.penalty_area_width_m, true)}
      {box(markings.goal_area_depth_m, markings.goal_area_width_m, false)}
      {box(markings.goal_area_depth_m, markings.goal_area_width_m, true)}
    </g>
  )
}

function PitchView({ view }: { view: PitchSpaceView }) {
  const pitch = view.pitch
  const occupancy = view.occupancy
  if (!pitch || !occupancy) return null
  const [ox, oy] = occupancy.origin_m
  const viewWidth = occupancy.columns * occupancy.cell_size_m
  const viewHeight = occupancy.rows * occupancy.cell_size_m
  const peak = Math.max(1, ...occupancy.counts.flat())
  return (
    <figure className="pitch-svg">
      <svg viewBox={`${ox} ${oy} ${viewWidth} ${viewHeight}`} role="img" aria-label={`Pitch plan ${pitch.length_m} by ${pitch.width_m} metres with ${view.observations.projected} projected tracklet observations`}>
        <rect className="pitch-svg__turf" x={ox} y={oy} width={viewWidth} height={viewHeight} />
        {occupancy.counts.map((row, r) => row.map((count, c) => count > 0 && (
          <rect
            key={`${r}-${c}`}
            className="pitch-svg__cell"
            x={ox + c * occupancy.cell_size_m}
            y={oy + r * occupancy.cell_size_m}
            width={occupancy.cell_size_m}
            height={occupancy.cell_size_m}
            fillOpacity={0.15 + 0.7 * (count / peak)}
          ><title>{count} projected tracklet observations</title></rect>
        )))}
        <PitchMarkings pitch={pitch} />
        {view.points.map((point, index) => (
          <circle
            key={index}
            className={point.extrapolated_beyond_landmarks ? 'pitch-svg__point pitch-svg__point--extrapolated' : 'pitch-svg__point'}
            cx={point.x_m}
            cy={point.y_m}
            r={0.6}
          ><title>{`${point.tracklet_id} · ${seconds(point.timestamp_seconds)} · frame ${point.frame_index} · keyframe ${point.calibration_keyframe}${point.extrapolated_beyond_landmarks ? ' · extrapolated beyond landmarks' : ''}`}</title></circle>
        ))}
      </svg>
      <figcaption>
        Plan view in metres ({pitch.length_m} × {pitch.width_m} m, {pitch.dimensions_source}). Shaded cells: {occupancy.unit} ({occupancy.cell_size_m} m).
        Dots: {view.points.length}{view.points_truncated ? ' evenly sampled' : ''} projected foot points; hollow dots are extrapolated beyond the fit landmarks.
      </figcaption>
    </figure>
  )
}

function Timeline({ view }: { view: PitchSpaceView }) {
  const total = view.interval.analysed_frames
  if (!total) return <p className="muted">No AI1 sampled frames fall in this interval.</p>
  return (
    <div className="pitch-timeline" role="list" aria-label="Calibration state of analysed frames over time">
      {view.interval.segments.map((segment) => (
        <span
          key={`${segment.start_frame}-${segment.status}`}
          role="listitem"
          className={`pitch-timeline__segment pitch-timeline__segment--${segment.status}`}
          style={{ flexGrow: segment.frames }}
          title={`${STATUS_LABEL[segment.status]}${segment.reason ? ` (${segment.reason})` : ''}: ${seconds(segment.start_seconds)}–${seconds(segment.end_seconds)}, ${segment.frames} frames`}
        />
      ))}
    </div>
  )
}

export function PitchSpacePanel({ runId }: { runId: string }) {
  const [draft, setDraft] = useState({ start: '', end: '' })
  const [query, setQuery] = useState('')
  const startId = useId()
  const endId = useId()
  const view = useApi<PitchSpaceView>(`/api/v1/runs/${runId}/pitch-space${query}`)

  function apply(event: FormEvent) {
    event.preventDefault()
    const params = new URLSearchParams()
    if (draft.start.trim()) params.set('start_seconds', draft.start.trim())
    if (draft.end.trim()) params.set('end_seconds', draft.end.trim())
    const encoded = params.toString()
    setQuery(encoded ? `?${encoded}` : '')
  }

  function reset() {
    setDraft({ start: '', end: '' })
    setQuery('')
  }

  const data = view.data
  return (
    <Panel className="pitch-space">
      <div className="panel-head"><div><span className="eyebrow">AI3 spatial diagnostic</span><h2>Pitch-space diagnostic (tracklet level)</h2></div><MapIcon /></div>
      <form className="pitch-interval" onSubmit={apply}>
        <label htmlFor={startId}>From (s)</label>
        <input id={startId} type="number" min={0} step="any" inputMode="decimal" value={draft.start} onChange={(event) => setDraft({ ...draft, start: event.target.value })} />
        <label htmlFor={endId}>To (s)</label>
        <input id={endId} type="number" min={0} step="any" inputMode="decimal" value={draft.end} onChange={(event) => setDraft({ ...draft, end: event.target.value })} />
        <button className="button button--secondary" type="submit">Apply interval</button>
        <button className="button button--secondary" type="button" onClick={reset} disabled={!query}>Whole clip</button>
      </form>
      {view.loading && <LoadingState label="Reading verified spatial artifacts…" />}
      {view.error && <ErrorState message={view.error} />}
      {data && (
        <div className="pitch-space__body">
          <p className="pitch-space__label" role="note"><AlertTriangle /> {data.label}</p>
          <div className="explain-badges">
            <span className={`pitch-evidence pitch-evidence--${data.evidence_status}`}>{EVIDENCE_LABEL[data.evidence_status]}</span>
            <span className="pitch-evidence">Data origin: {data.data_origin}</span>
            <span className="pitch-evidence">Identity: {data.identity.resolver_mode.replace('_', ' ')} · team {data.identity.team_assignment.replace('_', ' ')}</span>
          </div>
          <div className="metrics-grid">
            <MetricCard label="Analysed frames" value={String(data.interval.analysed_frames)} detail={data.interval.start_seconds !== null || data.interval.end_seconds !== null ? 'In the selected interval' : 'Whole clip'} />
            <MetricCard label="Calibrated" value={percent(data.interval.calibrated_fraction)} detail={`${data.interval.frames_by_status.calibrated} frames with a validated homography`} />
            <MetricCard label="Missing / invalidated" value={`${data.interval.frames_by_status.uncalibrated} / ${data.interval.frames_by_status.invalidated}`} detail="Uncalibrated / invalidated frames — no positions" />
            <MetricCard label="Projected foot points" value={String(data.observations.projected)} detail={`${data.observations.distinct_tracklets_projected} tracklets · ${data.observations.projected_extrapolated_beyond_landmarks} extrapolated`} />
          </div>
          <Timeline view={data} />
          <div className="pitch-timeline__legend">
            {(['calibrated', 'uncalibrated', 'invalidated'] as const).map((status) => <span key={status} className={`pitch-timeline__key pitch-timeline__key--${status}`}>{STATUS_LABEL[status]}</span>)}
          </div>
          {data.pitch && data.occupancy
            ? <PitchView view={data} />
            : (
              <p className="explain-disabled" role="status">
                <AlertTriangle /> No pitch positions are available{data.calibration_reason === 'no_landmark_annotations' ? ': no pitch landmark annotations were attached to this analysis, so every frame is uncalibrated. No fixed pixel-to-metre scale is ever substituted.' : ': no analysed frame in this run has a validated calibration.'}
              </p>
            )}
          {Object.keys(data.interval.frames_by_reason).length > 0 && (
            <div className="explain-section">
              <h3>Why frames have no position</h3>
              <ul className="explain-limitations">
                {Object.entries(data.interval.frames_by_reason).map(([reason, count]) => <li key={reason}><code>{reason}</code> · {count} frames</li>)}
              </ul>
            </div>
          )}
          <div className="explain-section">
            <h3>Evidence</h3>
            <ul className="explain-citations">
              {data.artifacts.map((item) => <li key={item.artifact_id}><code>{item.artifact_id}</code> <span>· {item.schema_version} · SHA-256 {shortId(item.sha256)} · {item.integrity_state}</span></li>)}
              {Object.entries(data.evidence).filter(([, value]) => value).map(([name, value]) => <li key={name}><code>{name}</code> <span title={value ?? undefined}>· {shortId(value ?? '')}</span></li>)}
            </ul>
          </div>
          <div className="explain-section">
            <h3>Important limitations</h3>
            <ul className="explain-limitations">{data.limitations.map((item, index) => <li key={index}>{item}</li>)}</ul>
          </div>
        </div>
      )}
    </Panel>
  )
}
