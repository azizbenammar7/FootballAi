import { expect, test } from '@playwright/test'

test('local legacy demo navigates without browser errors', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  page.on('pageerror', (error) => consoleErrors.push(error.message))

  await page.goto('/runs')
  await expect(page.getByRole('heading', { name: 'Analysis runs' })).toBeVisible()
  await expect(page.getByText('legacy v1').first()).toBeVisible()
  await page.screenshot({ path: 'test-results/runs-overview.png', fullPage: true })

  await page.getByRole('link', { name: /^View run/ }).first().click()
  await expect(page.getByText('Legacy V1 analysis')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Stage timeline' })).toBeVisible()
  await page.screenshot({ path: 'test-results/run-detail.png', fullPage: true })

  await page.getByRole('link', { name: /Open team overview/ }).click()
  await expect(page.getByRole('heading', { name: 'Team overview' })).toBeVisible()
  await expect(page.getByText('Unverified player tracks')).toBeVisible()
  await page.screenshot({ path: 'test-results/team-overview.png', fullPage: true })

  await page.getByRole('link', { name: /Open track analysis/ }).first().click()
  await expect(page.getByText('Workload and Fatigue Advisory')).toBeVisible()
  await expect(page.getByText(/Identity not verified/i)).toBeVisible()
  await page.screenshot({ path: 'test-results/track-detail.png', fullPage: true })

  expect(consoleErrors).toEqual([])
})

test('dashboard remains usable at tablet width', async ({ page }) => {
  await page.setViewportSize({ width: 820, height: 1100 })
  await page.goto('/runs')
  await expect(page.getByRole('heading', { name: 'Analysis runs' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Toggle navigation' })).toBeVisible()
  await page.getByRole('button', { name: 'Toggle navigation' }).click()
  await expect(page.getByRole('navigation', { name: 'Primary navigation' })).toBeVisible()
  await page.screenshot({ path: 'test-results/tablet-runs.png', fullPage: true })
})
