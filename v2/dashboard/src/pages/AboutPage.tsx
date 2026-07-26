import { Cloud, Database, GitBranch, HeartPulse, Layers3, ShieldCheck } from 'lucide-react'
import { Panel } from '../components/ui'

export default function AboutPage() {
  return (
    <div className="page about-page">
      <div className="page-head page-head--hero"><div><span className="eyebrow">System &amp; methodology</span><h1>Built for traceable football analysis</h1><p>FootballAi V2 separates immutable evidence, local services and review interfaces while preserving the original prototype.</p></div></div>
      <div className="about-grid">
        <Panel><div className="about-icon"><GitBranch /></div><h2>V1 — technical-test baseline</h2><p>The original Ultralytics YOLOv8 pipeline, artifacts and Streamlit dashboard remain historically preserved. V2 reads imported copies; it does not rewrite V1.</p></Panel>
        <Panel><div className="about-icon"><Layers3 /></div><h2>V2 — versioned platform</h2><p>Logical analyses contain isolated immutable attempts. Each records origin, input identity, code, configuration, model references, stages, artifacts and checksums.</p></Panel>
        <Panel><div className="about-icon"><Database /></div><h2>Local architecture today</h2><p>A caller-configured filesystem run store feeds a read-oriented FastAPI service and this React dashboard. There is no database, queue or cloud dependency.</p></Panel>
        <Panel><div className="about-icon"><ShieldCheck /></div><h2>Advisory language</h2><p>Workload and Fatigue Advisory outputs are heuristic review signals. They are not diagnosis, clinical advice, or a substitute for qualified practitioners.</p></Panel>
        <Panel><div className="about-icon"><HeartPulse /></div><h2>Quality before confidence</h2><p>Legacy views surface identity, calibration, camera-motion, coverage and provenance limitations wherever they affect interpretation.</p></Panel>
        <Panel><div className="about-icon"><Cloud /></div><h2>What comes later</h2><p>Detector benchmarking, calibrated tracking and optional Azure deployment remain future milestones after the local platform is reviewed and stable.</p></Panel>
      </div>
      <Panel className="contract-callout"><span className="eyebrow">Published contract</span><code>footballai.analysis-run/v1</code><p>V2 is the platform generation. This is the first public analysis-run contract.</p></Panel>
    </div>
  )
}
