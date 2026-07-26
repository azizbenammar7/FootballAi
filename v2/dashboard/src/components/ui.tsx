import type { ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, LoaderCircle, ShieldAlert } from 'lucide-react'

export function StatusBadge({ value }: { value: string }) {
  return <span className={`badge badge--${value.toLowerCase()}`}>{value.replace('_', ' ')}</span>
}

export function OriginBadge({ value }: { value: string }) {
  return <span className={`origin origin--${value}`}>{value.replace('_', ' ')}</span>
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function LoadingState({ label = 'Loading analysis data…' }: { label?: string }) {
  return (
    <div className="state" role="status">
      <LoaderCircle className="spin" aria-hidden="true" />
      <p>{label}</p>
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="state state--error" role="alert">
      <ShieldAlert aria-hidden="true" />
      <h2>Local data unavailable</h2>
      <p>{message}</p>
      <p className="muted">Confirm the V2 API is running at localhost:8000, then refresh this page.</p>
    </div>
  )
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <div className="state">
      <CheckCircle2 aria-hidden="true" />
      <h2>{title}</h2>
      <p>{message}</p>
    </div>
  )
}

export function LegacyWarning({ warnings }: { warnings: string[] }) {
  return (
    <aside className="legacy-warning" role="alert">
      <AlertTriangle aria-hidden="true" />
      <div>
        <strong>Legacy V1 analysis</strong>
        <p>
          These results come from the original technical-test pipeline. Track identities are
          unverified, positions are not homography-calibrated, and workload indicators are advisory only.
        </p>
        {warnings.length > 0 && <span>{warnings.length} data-quality notices attached</span>}
      </div>
    </aside>
  )
}

export function AnalysisWarning({ warnings, legacy = false }: { warnings: string[]; legacy?: boolean }) {
  if (legacy) return <LegacyWarning warnings={warnings} />
  const synthetic = warnings.some((item) => item.toLowerCase().includes('synthetic workflow result'))
  return (
    <aside className="legacy-warning quality-warning" role="alert">
      <AlertTriangle aria-hidden="true" />
      <div>
        <strong>{synthetic ? 'Synthetic workflow result' : 'V1-compatible analysis'}</strong>
        <p>{warnings[0] ?? 'Track identities are unverified and Workload and Fatigue Advisory outputs are heuristic and advisory only.'}</p>
      </div>
    </aside>
  )
}

export function MetricCard({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  )
}

export function formatDate(value: string | null) {
  if (!value) return 'Not recorded'
  return new Intl.DateTimeFormat('en', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export function shortId(value: string) {
  return `${value.slice(0, 8)}…${value.slice(-4)}`
}

export function formatDistance(value: number) {
  return value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${value.toFixed(0)} m`
}
