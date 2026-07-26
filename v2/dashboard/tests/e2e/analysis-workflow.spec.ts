import { execFileSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, test, type Page } from '@playwright/test'

const fixtureDirectory = mkdtempSync(join(tmpdir(), 'footballai-playwright-'))
const fixturePath = join(fixtureDirectory, 'generated-evaluation-fixture.mp4')

test.beforeAll(() => {
  execFileSync('ffmpeg', ['-v', 'error', '-f', 'lavfi', '-i', 'color=c=0x195f3b:s=320x180:d=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', fixturePath])
})

async function submit(page: Page, profile = 'demo_fast') {
  await page.goto('/analyses/new')
  await page.getByLabel('Football video').setInputFiles(fixturePath)
  await page.getByLabel('Match name *').fill(`Generated ${profile} fixture`)
  await page.getByLabel('Data origin').selectOption('evaluation')
  await page.getByLabel('Pipeline profile').selectOption(profile)
  await page.getByRole('button', { name: 'Start analysis' }).click()
  await expect(page).toHaveURL(/\/runs\/[0-9a-f-]+\/progress/)
}

test('workflow success uploads, executes and opens synthetic results', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()) })
  await submit(page)
  await expect(page.getByRole('heading', { name: 'Analysis complete' })).toBeVisible({ timeout: 20_000 })
  await expect(page.getByText('Synthetic workflow result', { exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'Open results' }).click()
  await expect(page.getByRole('heading', { name: 'Team overview' })).toBeVisible()
  await expect(page.getByLabel('15-minute team block chart')).toBeVisible()
  await expect(page.getByText(/must not be treated as match measurements/)).toBeVisible()
  expect(consoleErrors).toEqual([])
})

test('running workflow can be cancelled but not retried', async ({ page }) => {
  await submit(page)
  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: 'Cancel analysis' }).click()
  await expect(page.locator('.run-hero').getByText('cancelled', { exact: true })).toBeVisible({ timeout: 15_000 })
  await expect(page.getByRole('button', { name: 'Retry as new attempt' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Create new analysis from this input' })).toBeVisible()
})

test('failed workflow retries as a separate successful attempt', async ({ page }) => {
  await submit(page, 'test_fail')
  await expect(page.getByText('test_stage_failure')).toBeVisible({ timeout: 15_000 })
  const failedUrl = page.url()
  await page.getByRole('button', { name: 'Retry as new attempt' }).click()
  await expect(page).not.toHaveURL(failedUrl)
  await expect(page.getByRole('heading', { name: 'Analysis complete' })).toBeVisible({ timeout: 20_000 })
  await expect(page.getByRole('link', { name: /Attempt 1/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /Attempt 2/ })).toBeVisible()
})
