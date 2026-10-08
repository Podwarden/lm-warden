import { expect, type Page } from '@playwright/test';

// Shared bootstrap for the manual-smoke Playwright specs (chat-playground.spec.ts,
// chat2.spec.ts). Extracted from chat-playground.spec.ts (S8, issue #117) so
// chat2.spec.ts (issue #232) does not fork its own copy of the login / model
// pull-and-load dance.

export async function loginViaUi(page: Page) {
  await page.goto('/login');
  await page.request.get('/api/csrf');
  await page.fill('input[name=username]', 'admin');
  await page.fill('input[name=password]', process.env.E2E_ADMIN_PW || 'change-me');
  await page.click('button:has-text("Log in")');
  await expect(page).toHaveURL(/\/stats/);
}

export async function ensureOptLoaded(page: Page) {
  // Reuse the happy-path pattern: warm CSRF, list models, add+pull+load
  // if missing.
  await page.request.get('/api/csrf');
  const loginRes = await page.request.post('/api/auth/login', {
    data: { username: 'admin', password: process.env.E2E_ADMIN_PW || 'change-me' },
  });
  if (!loginRes.ok()) return;
  const { access_token } = await loginRes.json();
  const csrfRes = await page.request.get('/api/csrf');
  const csrf = csrfRes.ok() ? (await csrfRes.json()).csrf : '';
  const listRes = await page.request.get('/api/models', {
    headers: { Authorization: `Bearer ${access_token}` },
  });
  if (!listRes.ok()) return;
  const { models } = (await listRes.json()) as {
    models: Array<{ id: string; served_model_name: string; status: string }>;
  };
  const existing = models.find((m) => m.served_model_name === 'opt-125m');
  if (existing && existing.status === 'loaded') return;

  // Not loaded — drive through the UI to inherit the same add/pull/load
  // wait semantics as the happy-path spec.
  await page.goto('/models');
  if (!existing) {
    await page.click('button:has-text("Add model")');
    await page.fill('input[name=served_model_name]', 'opt-125m');
    await page.fill('input[name=hf_repo]', 'facebook/opt-125m');
    await page.fill('input[name=gpu_indices]', '0');
    const addResponse = page.waitForResponse(
      (r) =>
        r.url().endsWith('/api/models') &&
        r.request().method() === 'POST' &&
        r.status() === 201,
      { timeout: 30000 },
    );
    await page.locator('[role=dialog]').getByRole('button', { name: /^Add$/ }).click();
    await addResponse;
  }
  await page.getByText('opt-125m').first().click();
  await expect(page.locator('text=pulled').first()).toBeVisible({ timeout: 120000 });
  if (existing?.status !== 'loaded') {
    await page.click('button:has-text("Load")');
    await expect(page.locator('text=loaded').first()).toBeVisible({ timeout: 60000 });
  }
}
