// Login-flow e2e — pins the contract that the unauthenticated /login
// page renders without any nav chrome:
//
//   - no brand block (the `LM Warden` link to /models)
//   - no live header-metrics widget (`[data-testid=header-metrics]`)
//   - no menu hamburger
//
// This is the #39 + #83 + S2 visual smoke. Bench-style spec — not run
// in CI; ops drive this manually against a live stack the same way
// they drive `happy-path.spec.ts` (see ./README.md).
//
// Once authenticated, the same page (post-redirect to /models) MUST
// surface the brand AND the header-metrics widget — so the spec
// brackets both the negative (logged-out chrome is empty) and
// positive (logged-in chrome carries the widget) sides of the gate.
import { test, expect } from '@playwright/test';

const ADMIN_PW = process.env.E2E_ADMIN_PW || 'change-me';

test('login page has no nav chrome; post-login page surfaces the header widget', async ({
  page,
}) => {
  // Clear cookies so we land unauthenticated regardless of any prior
  // session left by other specs in this run.
  await page.context().clearCookies();

  await page.goto('/login');

  // Negative side — no chrome on /login. Match the brand by ROLE, not by
  // text: /login has its own <h1>LM Warden</h1>, so a bare getByText
  // would find the heading and fail even with the nav correctly hidden.
  // The brand is the only LINK carrying that name (it points at /models).
  // The widget exposes data-testid="header-metrics". Both should be
  // absent — if either is visible, NavBar's hide gate regressed.
  await expect(page.getByRole('link', { name: 'LM Warden' })).toHaveCount(0);
  await expect(page.getByTestId('header-metrics')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /open menu/i })).toHaveCount(0);

  // Authenticate and bounce to /models. Reuse the happy-path's CSRF
  // warm-up so the first POST doesn't race the cookie set.
  await page.request.get('/api/csrf');
  await page.fill('input[name=username]', 'admin');
  await page.fill('input[name=password]', ADMIN_PW);
  await page.click('button:has-text("Log in")');
  await expect(page).toHaveURL(/\/models/);

  // Positive side — both the brand and the widget should surface now.
  await expect(page.getByRole('link', { name: 'LM Warden' })).toBeVisible();
  // The widget is responsive-hidden under md (`hidden md:inline-flex`).
  // Playwright's default viewport is 1280×720 so the md breakpoint is
  // satisfied; the element should be both present AND visible.
  const widget = page.getByTestId('header-metrics');
  await expect(widget).toBeVisible({ timeout: 5_000 });
  // data-status flips to "connected" once the SSE handshake completes.
  // Allow up to 10s — the first probe can be slow on a cold nvidia-smi
  // or when the ticket-mint preflight blocks on CSRF acquisition.
  await expect(widget).toHaveAttribute('data-status', /connected|connecting/, {
    timeout: 10_000,
  });
});
