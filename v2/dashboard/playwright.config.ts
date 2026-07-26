import { defineConfig, devices } from '@playwright/test'

const pythonExecutable = process.env.FOOTBALLAI_V2_PYTHON ?? '../../.venv-test/bin/python'
const e2eRoot = `/tmp/footballai-v2-e2e-${process.pid}`
const apiPort = Number(process.env.FOOTBALLAI_V2_E2E_API_PORT ?? 18080)
const dashboardPort = Number(process.env.FOOTBALLAI_V2_E2E_DASHBOARD_PORT ?? 14173)
const serviceEnvironment = `FOOTBALLAI_V2_PYTHON=${pythonExecutable} FOOTBALLAI_V2_API_PORT=${apiPort} FOOTBALLAI_V2_RUN_ROOT=${e2eRoot}/runs FOOTBALLAI_V2_QUEUE_ROOT=${e2eRoot}/queue FOOTBALLAI_V2_CORS_ORIGINS=http://127.0.0.1:${dashboardPort} FOOTBALLAI_ENABLE_TEST_PROFILES=1 FOOTBALLAI_DEMO_STAGE_DELAY_SECONDS=0.25`

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  outputDir: 'test-results',
  use: {
    baseURL: `http://127.0.0.1:${dashboardPort}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command:
        `${serviceEnvironment} ../dev/run_e2e_services.sh`,
      url: `http://127.0.0.1:${apiPort}/api/health`,
      timeout: 60_000,
      reuseExistingServer: false,
    },
    {
      command: `VITE_API_BASE=http://127.0.0.1:${apiPort} npm run dev -- --host 127.0.0.1 --port ${dashboardPort} --strictPort`,
      url: `http://127.0.0.1:${dashboardPort}`,
      timeout: 30_000,
      reuseExistingServer: false,
    },
  ],
})
