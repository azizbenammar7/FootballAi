import { Activity, CircleHelp, Database, Menu, PlusCircle, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

export default function AppShell() {
  const [open, setOpen] = useState(false)
  return (
    <div className="app-shell">
      <header className="mobile-header">
        <span className="brand-mark">FA</span>
        <strong>FootballAi V2</strong>
        <button type="button" aria-label="Toggle navigation" onClick={() => setOpen((value) => !value)}>
          {open ? <X /> : <Menu />}
        </button>
      </header>
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`}>
        <div className="brand">
          <span className="brand-mark">FA</span>
          <div><strong>FootballAi</strong><small>Analysis workspace · V2</small></div>
        </div>
        <nav aria-label="Primary navigation" onClick={() => setOpen(false)}>
          <NavLink to="/analyses/new"><PlusCircle aria-hidden="true" />New Analysis</NavLink>
          <NavLink to="/runs"><Database aria-hidden="true" />Analysis runs</NavLink>
          <NavLink to="/about"><CircleHelp aria-hidden="true" />System &amp; about</NavLink>
        </nav>
        <div className="sidebar-foot">
          <Activity aria-hidden="true" />
          <div><strong>Local mode</strong><small>No cloud services connected</small></div>
        </div>
      </aside>
      <main className="workspace"><Outlet /></main>
    </div>
  )
}
