import { execFileSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test, type Page } from '@playwright/test'

const fixtureDirectory = mkdtempSync(join(tmpdir(), 'footballai-agent-'))
const fixturePath = join(fixtureDirectory, 'generated-agent-fixture.mp4')

test.beforeAll(() => {
  execFileSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', 'color=c=0x195f3b:s=320x180:d=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', fixturePath])
})

async function openCompletedRun(page: Page) {
  await page.goto('/analyses/new')
  await page.getByLabel('Football video').setInputFiles(fixturePath)
  await page.getByLabel('Match name *').fill('Generated agent fixture')
  await page.getByLabel('Data origin').selectOption('evaluation')
  await page.getByLabel('Pipeline profile').selectOption('demo_fast')
  await page.getByRole('button', { name: 'Start analysis' }).click()
  await expect(page).toHaveURL(/\/runs\/[0-9a-f-]+\/progress/)
  await expect(page.getByRole('heading', { name: 'Analysis complete' })).toBeVisible({ timeout: 20_000 })
  const runId = /\/runs\/([0-9a-f-]+)\//.exec(page.url())?.[1]
  await page.goto(`/runs/${runId}`)
  await expect(page.getByRole('heading', { name: 'AI Analysis Explanation' })).toBeVisible()
}

async function ask(page: Page, question: string) {
  const input = page.getByRole('textbox', { name: 'Ask about this run' })
  await input.fill(question)
  const answered = page.waitForResponse((response) => response.url().endsWith('/agent') && response.request().method() === 'POST')
  await input.press('Enter')
  const response = await answered
  expect(response.status()).toBe(200)
  await expect(page.locator('.explain-asked')).toContainText(question)
  return (await response.json()) as { trace: { tool_calls: { tool: string | null; status: string }[]; termination: string } }
}

test('the agent answers, abstains, and shows only the tools the server ran', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()) })
  page.on('pageerror', (error) => consoleErrors.push(error.message))
  await openCompletedRun(page)
  await page.getByRole('radio', { name: 'Read-Only Agent' }).check()
  await expect(page.locator('.agent-banner')).toContainText('OFFLINE SCRIPTED AGENT')

  const diagnostics = await ask(page, 'What do the tracking diagnostics tell us?')
  expect(diagnostics.trace.tool_calls.map((item) => item.tool)).toEqual(['get_tracking_diagnostics'])
  await expect(page.locator('.explain-adapter')).toHaveText('OFFLINE SCRIPTED AGENT')
  await expect(page.locator('.explain-nature')).toHaveText('Synthetic output')
  await expect(page.getByText('Evidence citations')).toBeVisible()
  await page.locator('.agent-trace summary').click()
  await expect(page.locator('.agent-trace__tool > div > code')).toHaveText(['get_tracking_diagnostics'])

  const model = await ask(page, 'What model was used in the analysis?')
  expect(model.trace.tool_calls.map((item) => item.tool)).toEqual(['get_model_provenance'])
  await expect(page.locator('.agent-trace__tool > div > code')).toHaveText(['get_model_provenance'])

  const missing = await ask(page, 'What evidence is missing?')
  expect(missing.trace.termination).toBe('answered')
  expect(missing.trace.tool_calls.map((item) => item.tool)).toEqual(['get_evidence_limitations'])
  await expect(page.locator('.explain-answer')).toContainText('recorded nothing missing')

  const passes = await ask(page, 'How many successful passes did player 7 make?')
  expect(passes.trace.termination).toBe('unsupported_question')
  expect(passes.trace.tool_calls).toEqual([])
  await expect(page.locator('.explain-support')).toHaveText('Not answerable from evidence')
  await expect(page.locator('.explain-adapter')).toHaveText('OFFLINE SCRIPTED AGENT')
  await expect(page.getByText('Evidence citations')).toHaveCount(0)
  await expect(page.locator('.agent-trace__tool')).toHaveCount(0)
  await page.locator('.agent-trace summary').click()  // each answer starts with its trace collapsed
  await expect(page.getByText(/^No tool ran/)).toBeVisible()

  await page.screenshot({ path: 'test-results/agent-abstention.png', fullPage: true })
  expect(consoleErrors).toEqual([])
})

test('grounded explanation still works beside the agent', async ({ page }) => {
  await openCompletedRun(page)
  const input = page.getByRole('textbox', { name: 'Ask about this run' })
  await input.fill('What is the status of this run?')
  await input.press('Enter')
  await expect(page.locator('.explain-answer')).toContainText('succeeded')
  await expect(page.locator('.agent-trace')).toHaveCount(0)
  await expect(page.locator('.explain-adapter')).toHaveCount(0)
})
