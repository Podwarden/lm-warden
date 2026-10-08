/**
 * Regression for v2026.05.15.4 — /login no longer reload-loops when an
 * authenticated component (NavBar) is mounted in the root layout.
 *
 * Pre-fix scenario (v2026.05.15.3 production):
 *   1. The root layout always renders NavBar. NavBar called
 *      `useSWR('/api/version', authFetchJSON, …)` unconditionally — React
 *      hooks run regardless of any conditional `return null` later in
 *      the component, so the fetch fired even on /login.
 *   2. On /login the operator may not be authenticated. The
 *      `/api/version` request returns 401; the refresh attempt also
 *      returns 401.
 *   3. The 401 fallback in `auth-fetch.ts` called
 *      `window.location.replace('/login')` — but we were already on
 *      /login, so the call triggered a full-page reload of the same
 *      page.
 *   4. The reload re-imported `auth-fetch.ts`, resetting the
 *      `loginRedirectInFlight` module-level flag to `false`. The cycle
 *      repeated indefinitely.
 *
 * Two-pronged fix:
 *   - Primary: NavBar passes `null` as the SWR key on `/login` and
 *     `/setup` so the fetcher never fires (covered by nav-bar.test.tsx).
 *   - Defense in depth (covered here): `auth-fetch.ts` short-circuits
 *     the `window.location.replace('/login')` call when the current
 *     pathname already starts with `/login`. Any peer component that
 *     regresses the primary fix can't re-open the loop.
 *
 * The 401 Response is still returned to the caller in both cases —
 * skipping the redirect does not change the caller's error handling.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  authFetch,
  setAccessToken,
  setCsrfToken,
  isLoginRedirectInFlight,
  __resetLoginRedirectInFlightForTests,
} from '@/lib/auth-fetch';

/** Replace `window.location` with a stub whose `replace` is a spy and
 *  whose `pathname` can be set per-test. jsdom 25 marks the real
 *  `window.location.replace` non-configurable, so a bare `vi.spyOn`
 *  would throw — copy the trick from auth-fetch.test.ts and
 *  log-stream-401-cascade.test.tsx. */
function stubWindowLocation(pathname: string): ReturnType<typeof vi.fn> {
  const replace = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { replace, pathname, origin: 'http://localhost' },
    writable: true,
    configurable: true,
  });
  return replace;
}

describe('authFetch — same-page /login redirect guard', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    __resetLoginRedirectInFlightForTests();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('does NOT call window.location.replace when already on /login (loop guard)', async () => {
    const replace = stubWindowLocation('/login');

    // Sequence:
    //   call 1: GET /api/version → 401
    //   call 2: POST /api/auth/refresh → 401 (refresh fails)
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await authFetch('/api/version');

    // 401 still surfaced to caller — caller's error handling must run.
    expect(response.status).toBe(401);
    // The crux: no redirect fired. /login → /login would have been a
    // full-page reload, resetting the in-flight flag, looping forever.
    expect(replace).not.toHaveBeenCalled();
    // The flag also stayed clean — we never entered the "redirect
    // in flight" state because no redirect was issued.
    expect(isLoginRedirectInFlight()).toBe(false);
  });

  it('DOES call window.location.replace("/login") when on a different page', async () => {
    // Companion case: same fetch sequence, but pathname is "/" — the
    // redirect must still fire so the operator is sent to /login. This
    // pins that the loop guard is narrowly scoped to /login-on-/login.
    const replace = stubWindowLocation('/');

    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await authFetch('/api/version');

    expect(response.status).toBe(401);
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
    expect(isLoginRedirectInFlight()).toBe(true);
  });

  it('does NOT call replace when pathname is /login with a trailing slash', async () => {
    // #39 fix: the guard now exact-matches /login (with or without a
    // trailing slash) rather than ``startsWith('/login')``. /login/
    // is still a same-page no-op so a misconfigured route trail
    // doesn't bounce the operator.
    const replace = stubWindowLocation('/login/');

    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await authFetch('/api/version');

    expect(response.status).toBe(401);
    expect(replace).not.toHaveBeenCalled();
  });

  it('DOES call replace from a non-exact /login sub-path (#39 tightening)', async () => {
    // Counter-test for the #39 tightening: a hypothetical /login-help
    // page (or /loginX) is NOT the same page as /login, so the redirect
    // MUST fire. Pins exact-match semantics so a future refactor that
    // reverts to ``startsWith('/login')`` breaks this test.
    //
    // Real browsers parse window.location.pathname to exclude the
    // query string, so /login?next=… surfaces here as pathname
    // '/login' (suppressed redirect, which is correct — we ARE on
    // the login page). We exercise the sub-path case with a
    // non-query suffix the real DOM would not strip.
    const replace = stubWindowLocation('/login-help');

    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    const response = await authFetch('/api/version');

    expect(response.status).toBe(401);
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
  });
});
