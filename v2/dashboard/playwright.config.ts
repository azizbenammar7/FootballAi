import { defineConfig, devices } from '@playwright/test'

const pythonExecutable = process.env.FOOTBALLAI_V2_PYTHON ?? '../../.venv-test/bin/python'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  fullyParallel: false,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  outputDir: 'test-results',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command:
        `PYTHONPATH=../src ${pythonExecutable} -m footballai_v2.cli.import_legacy_v1 --source ../../data/processed --output-root ../../data/runs && FOOTBALLAI_V2_RUN_ROOT=../../data/runs FOOTBALLAI_V2_CORS_ORIGINS=http://127.0.0.1:4173 PYTHONPATH=../src ${pythonExecutable} -m uvicorn footballai_v2.api.main:app --host 127.0.0.1 --port 8000`,
      url: 'http://127.0.0.1:8000/api/health',
      timeout: 60_000,
      reuseExistingServer: false,
    },
    {
      command: 'npm run preview -- --host 127.0.0.1 --port 4173',
      url: 'http://127.0.0.1:4173',
      timeout: 30_000,
      reuseExistingServer: false,
    },
  ],
})
