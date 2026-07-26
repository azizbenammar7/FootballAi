import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import AppShell from './components/AppShell'
import { LoadingState } from './components/ui'

const AboutPage = lazy(() => import('./pages/AboutPage'))
const RunDetailPage = lazy(() => import('./pages/RunDetailPage'))
const RunsPage = lazy(() => import('./pages/RunsPage'))
const TeamPage = lazy(() => import('./pages/TeamPage'))
const TrackPage = lazy(() => import('./pages/TrackPage'))

export default function App() {
  return (
    <Suspense fallback={<LoadingState label="Loading dashboard view…" />}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Navigate to="/runs" replace />} />
          <Route path="runs" element={<RunsPage />} />
          <Route path="runs/:runId" element={<RunDetailPage />} />
          <Route path="runs/:runId/team" element={<TeamPage />} />
          <Route path="runs/:runId/tracks/:playerId" element={<TrackPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/runs" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}
