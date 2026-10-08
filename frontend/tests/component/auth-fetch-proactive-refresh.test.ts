/**
 * Regression suite for #97 — "vllm-warden logs me out every few minutes".
 *
 * Two related fixes covered here:
 *
 *   1. **Proactive refresh scheduler.** Every successful login or refresh
 *      sets a `setTimeout` at 80% of the access-token TTL. The next refresh
 *      runs BEFORE the token expires, so the 15-minute access window stops
 *      being a recurring forced-logout boundary. The chain is self-perpetuating:
 *      each successful refresh re-arms the timer; a failed refresh leaves
 *      it un-armed and the next user action recovers.
 *
 *   2. **Transient vs terminal classification.** Pre-#97, `refresh()`
 *      returned `null` for *every* non-ok response and the 401-replay
 *      handler punted the user to /login regardless. A 502 during a
 *      backend bounce now resolves as `{ error: 'transient' }`; the
 *      401-replay handler surfaces the original 401 to the caller but
 *      does NOT touch `loginRedirectInFlight` and does NOT redirect. Only
 *      an actual HTTP 401/403 from `/api/auth/refresh` (the backend
 *      rejecting the refresh-token cookie) still triggers the redirect.
 *
 * Tests use vi.useFakeTimers() so the 12-minute setTimeout for the
 * default 15-minute TTL can be exercised in milliseconds. This is a
 * unit-test pattern, not the prior fake-timers anti-pattern (which was
 * for components doing real network mocks).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  authFetch,
  setAccessToken,
  setCsrfToken,
  isLoginRedirectInFlight,
  __resetLoginRedirectInFlightForTests,
  __hasProactiveRefreshScheduledForTests,
} from '@/lib/auth-fetch';

/** Replace `window.location` with a stub whose `replace` is a spy.
 *  jsdom 25 marks the real `window.location.replace` non-configurable,
 *  so a bare `vi.spyOn` would throw. */
function stubWindowLocation(pathname = '/'): ReturnType<typeof vi.fn> {
  const replace = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { replace, pathname, origin: 'http://localhost' },
    writable: true,
    configurable: true,
  });
  return replace;
}

describe('authFetch — proactive refresh scheduler (#97 Fix #1)', () => {
  beforeEach(() => {
    setAccessToken(null); // also clears any leftover timer from prior tests
    setCsrfToken(null);
    __resetLoginRedirectInFlightForTests();
    stubWindowLocation('/');
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    // Belt-and-suspenders — ensure no timer leaks into the next test file.
    setAccessToken(null);
  });

  it('successful login schedules a refresh timer at 80% of expires_in', () => {
    // Simulate what login/page.tsx does after POST /api/auth/login:
    // the page reads { access_token, expires_in } from the body and
    // calls setAccessToken with both. Before #97 the page only passed
    // the token, so no timer was ever installed and the session always
    // hit the 15-minute expiry hard.
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);
    setAccessToken('login-token', 900);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(true);
  });

  it('successful refresh re-schedules the timer (chain stays alive)', async () => {
    // Drive a full /api/auth/refresh round-trip and assert that the
    // timer is armed when refresh() resolves. The point of this test
    // is the self-perpetuating property: a refresh fired BY the scheduler
    // arms the next scheduler tick from inside refresh(). If this
    // wiring breaks, the second tick never fires and we regress to
    // pre-#97 "logged out at 15 minutes" behaviour.
    setAccessToken('initial', 900);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(true);

    const fetchMock = vi.fn()
      // First call: /api/version → 401 (forces 401-replay path)
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      // Second call: /api/auth/refresh → 200 with fresh token + expiry
      .mockResolvedValueOnce(
        new Response('{"access_token":"refreshed","expires_in":900}', { status: 200 }),
      )
      // Third call: replay /api/version with refreshed Bearer → 200
      .mockResolvedValueOnce(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/version');
    expect(r.status).toBe(200);

    // Timer is still armed — the successful refresh installed a fresh
    // one (the old one was cleared by setAccessToken).
    expect(__hasProactiveRefreshScheduledForTests()).toBe(true);
  });

  it('proactive refresh receiving a 5xx leaves no timer and does NOT redirect', async () => {
    // The scheduler fires `void refresh()` at the 12-minute mark.
    // refresh() resolves with { error: 'transient' } on a 502, which
    // means: leave accessToken null, leave loginRedirectInFlight false,
    // do NOT call window.location.replace. The next user action will
    // retry. This is the core anti-cascade behaviour — backend bounces
    // must not boot users.
    const replace = stubWindowLocation('/');
    setAccessToken('initial', 900);

    // Mock only the refresh call — it'll be the only fetch the timer
    // callback triggers.
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Bad Gateway', { status: 502 }));
    vi.stubGlobal('fetch', fetchMock);

    // Advance to the scheduled fire time (12 min for expires_in=900).
    // vi.advanceTimersByTimeAsync drains the setTimeout AND awaits the
    // promise chain the callback created. The bare vi.advanceTimersByTime
    // would leave the inner `refresh()` promise unresolved.
    await vi.advanceTimersByTimeAsync(12 * 60 * 1000 + 100);

    // The 502 hit. Timer is gone (it cleared itself before invoking the
    // callback, and the transient failure path doesn't re-arm). No
    // redirect. The flag is clean — caller can still drive a recovery.
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);
    expect(replace).not.toHaveBeenCalled();
    expect(isLoginRedirectInFlight()).toBe(false);

    // And the refresh round-trip actually happened (proves the timer
    // fired — without this assertion a setTimeout typo that NEVER fired
    // would also pass the "no redirect" check vacuously).
    expect(fetchMock).toHaveBeenCalledWith('/api/auth/refresh', expect.objectContaining({
      method: 'POST', credentials: 'include',
    }));
  });

  it('proactive refresh receiving a 401 sets loginRedirectInFlight and redirects (once)', async () => {
    // The "true terminal" branch: backend says the refresh cookie is
    // actually rejected. Same trigger conditions as the request-driven
    // 401-replay path, just initiated by the scheduler instead of a
    // user action. Note: a proactive refresh that fails terminally
    // does NOT itself redirect — only the subsequent user-driven 401
    // does. We assert here by simulating: scheduler fires, refresh
    // 401s, then a user action fires and the 401-replay sees the
    // refresh fail again (terminal), THAT call redirects.
    //
    // Per spec: "The scheduled refresh must NOT trigger a redirect on
    // failure — Fix #2's classification applies here too. A failed
    // proactive refresh just leaves the timer un-scheduled."
    //
    // So this test pins TWO things:
    //   (a) the scheduler fires refresh and gets 401 — no redirect,
    //       no flag set, just an un-armed timer.
    //   (b) the subsequent user-driven 401-replay sees refresh() return
    //       { error: 'invalid' } again, and THAT call fires the
    //       redirect (preserves the cascade-fix invariant: terminal
    //       refresh failures redirect exactly once).
    const replace = stubWindowLocation('/');
    setAccessToken('initial', 900);

    // Three refresh calls total: one from scheduler, two from authFetch
    // (the eager-refresh guard does NOT fire because we set accessToken
    // above; only the 401-replay calls refresh). Wait — we cleared
    // accessToken when the scheduler's refresh returned invalid? No:
    // the invalid branch does NOT touch accessToken. So accessToken is
    // still 'initial' when the user action fires. That request gets
    // 401 (server's own access-token expiry), then 401-replay calls
    // refresh again → invalid → redirects. Good.
    const fetchMock = vi.fn()
      // Call 1: scheduler-fired /api/auth/refresh → 401
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      // Call 2: user GET /api/version → 401 (server rejects token)
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      // Call 3: 401-replay → /api/auth/refresh → 401 (same invalid)
      .mockResolvedValueOnce(new Response('', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    // (a) Drive the scheduler. After it fires, no redirect.
    await vi.advanceTimersByTimeAsync(12 * 60 * 1000 + 100);
    expect(replace).not.toHaveBeenCalled();
    expect(isLoginRedirectInFlight()).toBe(false);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);

    // (b) Now a user-driven request hits 401, refresh-replay also 401,
    // THIS call fires the redirect. Exactly once.
    const r = await authFetch('/api/version');
    expect(r.status).toBe(401);
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
    expect(isLoginRedirectInFlight()).toBe(true);
  });

  it('setAccessToken(null) clears the timer', () => {
    setAccessToken('about-to-logout', 900);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(true);
    // Logout path or session-teardown should cancel the timer so a
    // stale callback can't fire after the user has logged out and
    // accidentally re-mint a session via the refresh cookie.
    setAccessToken(null);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);
  });

  it('clamps to a no-op when expires_in is zero, negative, or non-finite', () => {
    // Defensive: a backend bug that returns expires_in=0 or NaN must
    // not install a 0-ms timer (which would tight-loop refresh calls
    // and DDoS our own auth endpoint). The scheduler treats <=0 and
    // non-finite as "no schedule".
    setAccessToken('token', 0);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);

    setAccessToken('token', -1);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);

    setAccessToken('token', NaN);
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);
  });

  it('single-arg setAccessToken does not install a timer (back-compat)', () => {
    // Existing call sites (tests + any future caller that lacks the
    // expiry) keep the no-schedule semantics. Critical for test
    // isolation — every beforeEach in the existing auth-fetch.test.ts
    // calls `setAccessToken('initial')` and we MUST NOT start a timer
    // there.
    setAccessToken('token-no-expiry');
    expect(__hasProactiveRefreshScheduledForTests()).toBe(false);
  });
});

describe('authFetch — transient vs terminal refresh failure (#97 Fix #2)', () => {
  beforeEach(() => {
    setAccessToken('initial');
    setCsrfToken(null);
    __resetLoginRedirectInFlightForTests();
    stubWindowLocation('/');
  });
  afterEach(() => {
    vi.restoreAllMocks();
    setAccessToken(null);
  });

  it('request-driven 401-replay where refresh returns 502 does NOT redirect', async () => {
    // The core regression. Backend restarts mid-session → the user's
    // next authed request gets 401 (loadbalancer / kubelet drained the
    // pod just-in-time), 401-replay calls refresh, refresh gets 502
    // (next pod still starting up). Pre-#97 this returned null →
    // redirected the user. Now it returns { error: 'transient' } →
    // surfaces the 401 to the caller, no redirect, no flag flip.
    const replace = stubWindowLocation('/');
    const fetchMock = vi.fn()
      // user GET /api/models → 401
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      // 401-replay → /api/auth/refresh → 502
      .mockResolvedValueOnce(new Response('Bad Gateway', { status: 502 }));
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/models');

    // The original 401 is surfaced — SWR / the caller sees the error
    // and can render whatever it wants (re-try later, show a toast, …).
    expect(r.status).toBe(401);
    // The crux of #97: NO redirect.
    expect(replace).not.toHaveBeenCalled();
    // Flag is clean — a later user action when the backend recovers
    // can still drive a fresh refresh + recover the session.
    expect(isLoginRedirectInFlight()).toBe(false);
  });

  it('request-driven 401-replay where refresh returns a network error does NOT redirect', async () => {
    // TypeError from fetch (TCP RST, DNS, offline). Same shape as 5xx
    // — caller must NOT be redirected. Pin this separately because the
    // pre-#97 code's try/catch around fetch was absent — a thrown
    // TypeError would have escaped refresh() entirely and crashed the
    // authFetch promise chain.
    const replace = stubWindowLocation('/');
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fetchMock);

    // Silence the console.warn we emit on transient network errors —
    // otherwise vitest prints it to the test runner log on every run.
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const r = await authFetch('/api/models');

    expect(r.status).toBe(401);
    expect(replace).not.toHaveBeenCalled();
    expect(isLoginRedirectInFlight()).toBe(false);
    // We DID warn — proves we entered the transient-network branch
    // rather than some other code path.
    expect(consoleWarn).toHaveBeenCalled();
  });

  it('request-driven 401-replay where refresh returns 429 does NOT redirect', async () => {
    // Rate-limit class. Backend unlikely to ever 429 /api/auth/refresh
    // in practice, but the spec calls it out explicitly to cover every
    // non-401/403 status. Same contract as 5xx: surface, do not redirect.
    const replace = stubWindowLocation('/');
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      .mockResolvedValueOnce(new Response('Too Many Requests', { status: 429 }));
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/models');

    expect(r.status).toBe(401);
    expect(replace).not.toHaveBeenCalled();
    expect(isLoginRedirectInFlight()).toBe(false);
  });

  it('request-driven 401-replay where refresh returns 401 DOES redirect (preserves cascade-fix invariant)', async () => {
    // The "real" terminal failure. The refresh cookie is genuinely
    // rejected — the user's session is dead and must re-authenticate.
    // This is the v2026.05.15.3 invariant ("first 401 from refresh
    // fires exactly one redirect") and the new transient/terminal
    // split MUST preserve it. Without this regression test, a future
    // refactor that accidentally treats 401 as transient would silently
    // strand users on stale pages.
    const replace = stubWindowLocation('/');
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      .mockResolvedValueOnce(new Response('', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/models');

    expect(r.status).toBe(401);
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
    expect(isLoginRedirectInFlight()).toBe(true);
  });

  it('request-driven 401-replay where refresh returns 403 DOES redirect (treats 403 as terminal)', async () => {
    // 403 from /api/auth/refresh is also terminal — origin check
    // rejection (origin_check_dep in app/auth/routes.py) means the
    // request is structurally wrong, retrying won't help, and the user
    // needs to re-authenticate. Same treatment as 401.
    const replace = stubWindowLocation('/');
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      .mockResolvedValueOnce(new Response('', { status: 403 }));
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/models');

    expect(r.status).toBe(401);
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
    expect(isLoginRedirectInFlight()).toBe(true);
  });
});
