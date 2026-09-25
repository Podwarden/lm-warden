import { test, expect } from '@playwright/test';

// Default test timeout is 30s; this spec waits up to 120s for pull and 60s
// for load, plus all the click/fill overhead. Allow 5 minutes end-to-end
// so a slow HF mirror or a cold load doesn't artificially fail the gate.
test.setTimeout(600_000);

test('login → add tiny model → auto-pull → load → completion → rotate token → unload → delete', async ({ page, request }) => {
  // Best-effort cleanup of any stale opt-125m / leftover tokens from a
  // previous failed run. Without this, the second registration POST hits
  // 409 Conflict and the spec dies in the modal. Uses the page.request
  // fixture (same cookie jar Playwright will use later).
  await page.request.get('/api/csrf');
  const loginRes = await page.request.post('/api/auth/login', {
    data: { username: 'admin', password: process.env.E2E_ADMIN_PW || 'change-me' },
  });
  if (loginRes.ok()) {
    const { access_token } = await loginRes.json();
    const csrfRes = await page.request.get('/api/csrf');
    const csrf = csrfRes.ok() ? (await csrfRes.json()).csrf : '';
    const listRes = await page.request.get('/api/models', {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    if (listRes.ok()) {
      const { models } = (await listRes.json()) as { models: Array<{ id: string; served_model_name: string; status: string }> };
      for (const m of models) {
        if (m.served_model_name !== 'opt-125m') continue;
        if (m.status === 'loaded' || m.status === 'loading') {
          await page.request.post(`/api/models/${m.id}/unload`, {
            headers: { Authorization: `Bearer ${access_token}`, 'X-CSRF-Token': csrf },
          }).catch(() => {});
        }
        await page.request.delete(`/api/models/${m.id}`, {
          headers: { Authorization: `Bearer ${access_token}`, 'X-CSRF-Token': csrf },
        }).catch(() => {});
      }
    }
  }
  // Clear all browser state so login below starts from a clean slate
  // (the cleanup above set a refresh cookie + access token).
  await page.context().clearCookies();

  await page.goto('/login');
  // Warm the CSRF cookie before the first mutating call. auth-fetch.ts
  // does retry on 403 by refetching /api/csrf, but the retry races the
  // SWR list refresh — by the time the second POST lands the test has
  // already given up on `text=opt-125m`. Seeding here removes the race.
  await page.request.get('/api/csrf');
  await page.fill('input[name=username]', 'admin');
  await page.fill('input[name=password]', process.env.E2E_ADMIN_PW || 'change-me');
  await page.click('button:has-text("Log in")');
  await expect(page).toHaveURL(/\/stats/);

  await page.click('button:has-text("Add model")');
  await page.fill('input[name=served_model_name]', 'opt-125m');
  await page.fill('input[name=hf_repo]', 'facebook/opt-125m');
  await page.fill('input[name=gpu_indices]', '0');
  // Wait for the POST /api/models to land before checking the list. The
  // modal closes optimistically on 201, but SWR revalidation is on a 5s
  // poll, so we explicitly wait for the response rather than relying on
  // the next tick.
  const addResponse = page.waitForResponse(
    (r) => r.url().endsWith('/api/models') && r.request().method() === 'POST' && r.status() === 201,
    { timeout: 30000 },
  );
  await page.locator('[role=dialog]').getByRole('button', { name: /^Add$/ }).click();
  await addResponse;
  await expect(page.locator('[role=dialog]')).toBeHidden();

  // v2 UI auto-triggers pull on registration; there is no manual Pull
  // button. Click into the model detail page and wait for the status
  // badge to flip to "pulled" (the actual contract — old "Pull complete"
  // text never existed in this UI).
  await expect(page.getByText('opt-125m').first()).toBeVisible({ timeout: 10000 });
  await page.getByText('opt-125m').first().click();
  await expect(page.locator('text=pulled').first()).toBeVisible({ timeout: 120000 });
  await page.click('button:has-text("Load")');
  await expect(page.locator('text=loaded').first()).toBeVisible({ timeout: 60000 });

  // Mint API token
  await page.goto('/tokens');
  await page.click('button:has-text("Create token")');
  await page.fill('input[name=name]', 'e2e-bot');
  // Scope the click to the dialog — outside the dialog "Create token" also
  // contains "Create", and Playwright's overlay-pointer-events guard will
  // retry forever on the obscured trigger.
  await page.locator('[role=dialog]').getByRole('button', { name: /^Create$/ }).click();
  const plaintext = (await page.textContent('[data-testid=new-token]')) ?? '';
  expect(plaintext.startsWith('vw_')).toBe(true);
  // Dismiss the post-create dialog so the token row is reachable.
  await page.click('button:has-text("Done")');

  // Completion via /v1
  const r = await request.post('/v1/completions', {
    headers: { Authorization: `Bearer ${plaintext}` },
    data: { model: 'opt-125m', prompt: 'hi', max_tokens: 8 },
  });
  expect(r.status()).toBe(200);

  // Rotate: trigger is aria-labeled "Rotate"; confirm is the submit
  // button inside the dialog (text "Rotate"), scoped via role=dialog
  // so the click doesn't collide with the trigger.
  await page.click('button[aria-label="Rotate"]');
  await page.locator('[role=dialog]').getByRole('button', { name: /^Rotate$/ }).click();
  await expect(page.getByText(/vw_/).first()).toBeVisible();

  // Unload + delete. Detail page is /models/{id} (not served_model_name), so
  // navigate by clicking the row from the list rather than hard-coding a path
  // that 404s.
  await page.goto('/models');
  await page.getByText('opt-125m').first().click();
  await page.click('button:has-text("Unload")');
  await expect(page.locator('text=pulled').first()).toBeVisible({ timeout: 60000 });
  // S6 (#105): Delete now opens a confirmation modal instead of firing
  // the DELETE inline. The trigger is still labelled "Delete"; the
  // confirm button inside the dialog is also "Delete". Scope the
  // confirm click to the dialog so Playwright doesn't keep retrying the
  // (now-obscured) trigger.
  await page.click('button:has-text("Delete")');
  await page.locator('[role=dialog]').getByRole('button', { name: /^Delete$/ }).click();
  await expect(page).toHaveURL(/\/models$/);
});
