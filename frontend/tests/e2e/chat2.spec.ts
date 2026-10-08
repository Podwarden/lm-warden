import { test, expect } from '@playwright/test';
import path from 'node:path';
import { loginViaUi, ensureOptLoaded } from './helpers';

// /chat2 happy-path smoke (chat2 Task 12, issue #232).
//
// Out of scope for CI (Playwright is manual-smoke per tests/e2e/README.md);
// exercised before each release against a live `docker compose` stack with
// opt-125m loaded, same as chat-playground.spec.ts (whose `loginViaUi` /
// `ensureOptLoaded` this spec reuses from ./helpers.ts).
//
// A brand-new chat has no model until one is picked in Settings (the
// backend only seeds `model` from a user's stored `/defaults`, which a
// fresh install has none of — see `app/chat2/routes_chats.py:create_chat`),
// so both tests below open the settings drawer and select opt-125m before
// touching the composer.
//
// Two tests, matching the design doc's §7 e2e list:
//
//   1. create → send → stream → regenerate → fork-here → delete: the
//      linear happy path plus the two "creates work off a real seq"
//      affordances (Regenerate replaces the last turn in place; Fork here
//      cuts a new chat at a message and switches to it).
//   2. paste an image → reload (signed URL refresh) → shrink the context
//      window until it locks → fork out of the lock → two tabs racing the
//      same chat (`turn_in_flight`). The `budget_blocked` banner (spec §7)
//      is NOT covered here — the warden's `BudgetPolicy` is `AlwaysAllow`,
//      so there is no real budget to trip; it is exercised with a stubbed
//      budget by the Banners component test (Task 10).

test.setTimeout(600_000);

test.describe('/chat2', () => {
  test('create → send → stream → regenerate → fork here → delete', async ({ page }) => {
    await loginViaUi(page);
    await ensureOptLoaded(page);

    await page.goto('/chat2');
    await page.getByRole('button', { name: 'New chat' }).click();

    // Pick the model — a fresh chat has none until Settings sets one.
    await page.getByRole('button', { name: 'Settings' }).click();
    await page.getByLabel('Model').selectOption('opt-125m');

    // Send and wait out the full turn (Stop appears, then Send returns) —
    // Regenerate only renders on a message once it is no longer streaming.
    await page.getByRole('textbox', { name: 'Message' }).fill('Say hello in one word.');
    await page.keyboard.press('Enter');
    await expect(page.getByRole('button', { name: 'Stop' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: 'Send' })).toBeVisible({ timeout: 60_000 });

    const lastAssistant = page.locator('[data-role=assistant]').last();
    await expect(lastAssistant).toContainText(/\w+/);

    // Regenerate replaces the same (last) assistant row in place.
    await lastAssistant.hover();
    await lastAssistant.getByRole('button', { name: 'Regenerate' }).click();
    await expect(page.getByRole('button', { name: 'Stop' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: 'Send' })).toBeVisible({ timeout: 60_000 });

    // Fork here (on the first user message) → confirm → the new chat is
    // created and selected. Count the delta rather than an absolute
    // sidebar size, since this is a shared manual-smoke stack that may
    // already carry chats from a previous run.
    const chatOptions = page.getByRole('option');
    const countBeforeFork = await chatOptions.count();

    const firstUser = page.locator('[data-role=user]').first();
    await firstUser.hover();
    await firstUser.getByRole('button', { name: 'Fork here' }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Fork', exact: true }).click();

    await expect(page.getByRole('option', { selected: true })).toHaveCount(1);
    await expect(chatOptions).toHaveCount(countBeforeFork + 1);

    // Delete one of the two chats (whichever sorts first) via its hover
    // delete icon + the destructive confirm dialog, and see the sidebar
    // shrink back down.
    await chatOptions.first().hover();
    await chatOptions.first().getByRole('button', { name: /^Delete / }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Delete', exact: true }).click();
    await expect(chatOptions).toHaveCount(countBeforeFork);
  });

  test('image attach + signed-url refresh on reload, context-full lock, two-tab turn_in_flight', async ({
    page,
    context,
  }) => {
    await loginViaUi(page);
    await ensureOptLoaded(page);

    await page.goto('/chat2');
    await page.getByRole('button', { name: 'New chat' }).click();

    await page.getByRole('button', { name: 'Settings' }).click();
    await page.getByLabel('Model').selectOption('opt-125m');

    // Attach an image, wait for the upload to finish (Send only enables
    // once no draft is `uploading`/`missing`/`error`), then send it.
    await page.getByTestId('file-input').setInputFiles(path.join(__dirname, 'fixtures', 'tiny.png'));
    await expect(page.getByRole('list', { name: 'Attached images' }).locator('img')).toBeVisible();
    await page.getByRole('textbox', { name: 'Message' }).fill('What is this?');
    await expect(page.getByRole('button', { name: 'Send' })).toBeEnabled({ timeout: 20_000 });
    await page.keyboard.press('Enter');

    // The user row's image only resolves to a real <img> (vs. the "expired
    // placeholder") once `attachments` is refreshed by the post-turn
    // reload — wait out the whole turn first.
    await expect(page.getByRole('button', { name: 'Stop' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole('button', { name: 'Send' })).toBeVisible({ timeout: 60_000 });
    await expect(page.locator('[data-role=user] img')).toBeVisible({ timeout: 15_000 });

    // Reload: history (and its signed attachment URL) survives a fresh load.
    await page.reload();
    const userImg = page.locator('[data-role=user] img');
    await expect(userImg).toBeVisible({ timeout: 15_000 });
    const src = await userImg.getAttribute('src');
    expect(src).toMatch(/\/api\/chat2\/attachments\/.+\?t=/);

    // Context-full: shrinking `max_model_len` is heavy to drive through the
    // UI; instead push `max_tokens` far past opt-125m's window so
    // `reserve = min(max_tokens, 4096)` alone exceeds it (spec §9 — derived,
    // not stored, context state). `settingsOpen` is page-local React state
    // and does not survive the reload above, so Settings has to be reopened.
    await page.getByRole('button', { name: 'Settings' }).click();
    const maxTokens = page.getByLabel('Max tokens');
    await maxTokens.fill('4096');
    await maxTokens.blur();
    // Filtered on the banner's own second clause, not "Context window is full":
    // the composer's disabled-reason line (composer.tsx, also role="status")
    // reads "Context window is full — fork to continue" in exactly this state,
    // so filtering on that phrase matches two nodes and trips strict mode.
    const ctxBanner = page.getByRole('status').filter({ hasText: 'The chat stays readable' });
    await expect(ctxBanner).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole('button', { name: 'Send' })).toBeDisabled();

    // Fork out of the lock via the banner's own action (scoped to it — the
    // confirm dialog's button is also named exactly "Fork").
    await ctxBanner.getByRole('button', { name: 'Fork' }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Fork', exact: true }).click();

    // The fork copies the source chat's settings (`max_tokens: 4096`), so
    // it inherits the lock — bring it back down. `updateSettings` awaits
    // the PATCH before updating local state, so waiting for the banner to
    // clear also proves the setting round-tripped the server before the
    // second tab reads this chat.
    await maxTokens.fill('256');
    await maxTokens.blur();
    await expect(ctxBanner).toHaveCount(0, { timeout: 10_000 });

    // Start a long generation, then race a second tab against the same chat.
    await page.getByRole('textbox', { name: 'Message' }).fill('Count from 1 to 500 slowly.');
    await expect(page.getByRole('button', { name: 'Send' })).toBeEnabled();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('button', { name: 'Stop' })).toBeVisible({ timeout: 15_000 });

    const page2 = await context.newPage();
    await page2.goto(page.url());
    await page2.getByRole('textbox', { name: 'Message' }).fill('hi');
    await page2.keyboard.press('Enter');
    await expect(page2.getByText(/Another turn is running/)).toBeVisible({ timeout: 15_000 });
    await page2.close();

    await page.getByRole('button', { name: 'Stop' }).click();
    await expect(page.getByRole('button', { name: 'Send' })).toBeVisible({ timeout: 15_000 });
  });
});
