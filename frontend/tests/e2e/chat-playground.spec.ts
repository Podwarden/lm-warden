import { test, expect } from '@playwright/test';
import { loginViaUi, ensureOptLoaded } from './helpers';

// /chat playground happy-path + abort verification (S8, issue #117).
//
// Out of scope for CI (Playwright is manual-smoke per tests/e2e/README.md);
// this lives next to happy-path.spec.ts and is exercised before each
// release against a live `docker compose` stack with opt-125m loaded.
//
// The spec asserts three load-bearing contracts:
//
//   1. /chat boots and POSTs /api/chat/playground/ensure on mount.
//   2. A streaming completion produces visible tokens in the message
//      list and resolves to phase=idle (Send button returns).
//   3. The abort affordance (Stop button or Esc) cancels the in-flight
//      request and the server-side ``/api/admin/active-requests`` counter
//      returns to 0 within a reasonable window. This is the diagnostic
//      that fences against the SSE leak risk called out in the dispatch
//      (#117 — "SSE cancellation must not leak server-side requests").
//
// Why opt-125m: the existing happy-path.spec already adds + pulls + loads
// it; running this spec immediately after happy-path inherits a loaded
// model. We re-bootstrap (login + pull + load) anyway so the spec is
// runnable in isolation.
//
// `loginViaUi` / `ensureOptLoaded` live in ./helpers.ts (extracted here,
// chat2 Task 12 / issue #232) so chat2.spec.ts can reuse the same
// login + model-bootstrap dance instead of forking its own copy.

test.setTimeout(600_000);

test.describe('/chat playground', () => {
  test('streams a completion, then abort returns active-requests to 0', async ({
    page,
  }) => {
    await loginViaUi(page);
    await ensureOptLoaded(page);

    // Navigate to /chat and wait for ensure() to settle.
    const ensureCall = page.waitForResponse(
      (r) =>
        r.url().includes('/api/chat/playground/ensure') &&
        r.request().method() === 'POST' &&
        r.status() === 200,
      { timeout: 15000 },
    );
    await page.goto('/chat');
    await ensureCall;

    // Picker should auto-select opt-125m once /api/models populates.
    const picker = page.getByTestId('chat-model-picker');
    await expect(picker).toBeEnabled();
    await expect(picker.locator('option:checked')).toHaveText('opt-125m');

    // Drive a turn with a prompt long enough that abort has a chance to
    // fire before completion. Tokens will start arriving within ~1s.
    const input = page.getByTestId('chat-input');
    await input.fill(
      'Please count slowly from one to one hundred, one number per line, no other commentary.',
    );

    // We need to be ready to inspect /api/admin/active-requests both
    // during streaming (count > 0) and after abort (count == 0). Pull
    // a fresh access token via the same login API so we can hit the
    // admin endpoint directly without piggy-backing on the page's
    // session.
    await page.request.get('/api/csrf');
    const loginRes = await page.request.post('/api/auth/login', {
      data: {
        username: 'admin',
        password: process.env.E2E_ADMIN_PW || 'change-me',
      },
    });
    expect(loginRes.ok()).toBe(true);
    const { access_token } = await loginRes.json();

    async function activeCount(): Promise<number> {
      const r = await page.request.get('/api/admin/active-requests', {
        headers: { Authorization: `Bearer ${access_token}` },
      });
      expect(r.status()).toBe(200);
      const { count } = (await r.json()) as { count: number };
      return count;
    }

    // Baseline before any send: 0.
    expect(await activeCount()).toBe(0);

    // Hit Send.
    await page.getByTestId('chat-send').click();

    // The Stop button should appear quickly once /api/chat/completions
    // returns headers + the first byte. (~2s budget covers cold prefill.)
    await expect(page.getByTestId('chat-stop')).toBeVisible({ timeout: 10000 });

    // While streaming the counter should be 1.
    await expect.poll(activeCount, { timeout: 5000 }).toBe(1);

    // Confirm tokens are actually painting in the message list — this
    // proves the SSE delta loop is wired, not just that the server
    // accepted the request. ``chat-streaming`` is the placeholder row
    // rendered while phase === 'streaming'.
    await expect(page.getByTestId('chat-streaming')).toContainText(/\S/, {
      timeout: 10000,
    });

    // Abort via the button (the unit suite covers Esc).
    await page.getByTestId('chat-stop').click();

    // Send button returns — proves phase flipped out of 'streaming'.
    await expect(page.getByTestId('chat-send')).toBeVisible({ timeout: 10000 });

    // The diagnostic counter must drift back to 0 within a few seconds.
    // The server-side finally block (in app/chat/routes_api.py) calls
    // counter.exit() on socket close; we give it a generous 10s budget
    // to allow for the upstream proxy noticing the disconnect.
    await expect.poll(activeCount, { timeout: 10000 }).toBe(0);

    // Partial assistant text should remain in the transcript (session-
    // only history contract: abort preserves what we received). The
    // streaming row is converted to a permanent row when send()
    // resolves, even on abort — the permanent row is rendered with
    // ``data-testid="chat-message"`` + ``data-role="assistant"``.
    const lastAssistant = page
      .locator('[data-testid="chat-message"][data-role="assistant"]')
      .last();
    await expect(lastAssistant).toContainText(/\S/);
  });
});
