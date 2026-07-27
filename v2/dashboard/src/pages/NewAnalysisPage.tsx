import { FileVideo2, UploadCloud } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadAnalysis, useApi } from '../api'
import { ErrorState, Panel } from '../components/ui'
import type { PipelineProfileList } from '../types'

export default function NewAnalysisPage() {
  const navigate = useNavigate()
  const profiles = useApi<PipelineProfileList>('/api/v1/pipeline-profiles')
  const [file, setFile] = useState<File | null>(null)
  const [profileId, setProfileId] = useState('demo_fast')
  const [progress, setProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const profile = profiles.data?.profiles.find((item) => item.profile_id === profileId)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(null)
    if (!file) { setError('Choose a video before starting the analysis.'); return }
    if (!profile?.available) { setError('The selected pipeline profile is unavailable.'); return }
    const values = new FormData(event.currentTarget); values.set('video', file)
    setSubmitting(true)
    try {
      const created = await uploadAnalysis(values, setProgress)
      navigate(`/runs/${created.run_id}/progress`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Upload failed safely.')
      setSubmitting(false)
    }
  }

  if (profiles.error) return <ErrorState message={profiles.error} />
  return (
    <div className="page">
      <div className="page-head"><div><span className="eyebrow">Asynchronous V2 workflow</span><h1>New Analysis</h1><p>Upload one football video, review its metadata, then start an isolated analysis attempt.</p></div></div>
      <form className="analysis-form" onSubmit={submit}>
        <Panel className="upload-panel">
          <label className="drop-zone">
            <UploadCloud aria-hidden="true" /><strong>{file ? file.name : 'Choose a football video'}</strong>
            <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type || 'media type unavailable'}` : 'MP4, MOV, MKV, or WebM · selection does not start upload'}</span>
            <input aria-label="Football video" type="file" accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mkv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          </label>
          {submitting && <div className="upload-progress" aria-label="Upload progress"><span style={{ width: `${progress}%` }} /><strong>{progress}% uploaded</strong></div>}
        </Panel>
        <Panel>
          <div className="panel-head"><div><span className="eyebrow">Match context</span><h2>Analysis metadata</h2></div><FileVideo2 /></div>
          <div className="form-grid">
            <label><span>Match name *</span><input name="match_name" required maxLength={160} placeholder="Home vs Away" /></label>
            <label><span>Home team</span><input name="home_team" maxLength={100} /></label>
            <label><span>Away team</span><input name="away_team" maxLength={100} /></label>
            <label><span>Competition</span><input name="competition" maxLength={120} /></label>
            <label><span>Match date</span><input name="match_date" type="date" /></label>
            <label><span>Venue</span><input name="venue" maxLength={160} /></label>
            <label><span>Data origin</span><select name="data_origin" defaultValue="real"><option value="real">Real uploaded footage</option><option value="evaluation">Evaluation fixture</option><option value="synthetic">Synthetic fixture</option></select></label>
            <label><span>Pipeline profile</span><select name="pipeline_profile" value={profileId} onChange={(event) => setProfileId(event.target.value)}>{profiles.data?.profiles.map((item) => <option key={item.profile_id} value={item.profile_id} disabled={!item.available}>{item.display_name}{item.available ? '' : ' — unavailable'}</option>)}</select></label>
            <label className="form-wide"><span>Notes</span><textarea name="notes" maxLength={1000} rows={4} /></label>
          </div>
          {profile && <div className={`profile-card ${profile.available ? '' : 'profile-card--unavailable'}`}><strong>{profile.display_name}</strong><p>{profile.description}</p><small>{profile.available ? `${profile.purpose} · Device: ${String(profile.runtime.device ?? 'not required')} · Model: ${String(profile.runtime.model ?? 'none')}` : profile.readiness_message}</small>{!profile.available && profile.missing_requirements.length > 0 && <span>Missing: {profile.missing_requirements.join(', ')}</span>}{!profile.available && profile.runtime_errors.map((error) => <span key={error}>{error}</span>)}{!profile.available && profile.setup_command && <span>Run: <code>{profile.setup_command}</code></span>}{profile.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div>}
          {profiles.data?.profiles.filter((item) => !item.available).map((item) => <p className="muted" key={item.profile_id}>{item.display_name} unavailable ({item.readiness_status}). Run: {item.setup_command}</p>)}
          {error && <p className="form-error" role="alert">{error}</p>}
          <button className="button button--primary" type="submit" disabled={submitting || !file || !profile?.available}>{submitting ? 'Uploading…' : 'Start analysis'}</button>
        </Panel>
      </form>
    </div>
  )
}
