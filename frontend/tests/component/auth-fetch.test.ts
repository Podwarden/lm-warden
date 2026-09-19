import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  authFetch,
  setAccessToken,
  setCsrfToken,
  __resetLoginRedirectInFlightForTests,
} from '@/lib/auth-fetch';

describe('authFetch', () => {
  beforeEach(() => {
    setAccessToken('initial');
    setCsrfToken(null);
    // The "redirects to /login if refresh also 401" test below leaves
    // loginRedirectInFlight === true (by design — production code
    // never resets it; the /login navigation re-imports the module).
    // Without this reset, any later test that exercises the post-401
    // path would silently observe the short-circuit and pass for the
    // wrong reason. Reset explicitly so each test starts in a known
    // state. See auth-fetch.ts for the production-side invariant.
    __resetLoginRedirectInFlightForTests();
    vi.restoreAllMocks();
  });

  it('attaches bearer header', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/tokens');
    expect(fetchMock).toHaveBeenCalledWith('/api/tokens', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer initial' }),
    }));
  });

  it('refreshes on 401 then retries with new token', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('{}', { status: 401 }))
      .mockResolvedValueOnce(new Response('{"access_token":"new","expires_in":900}', { status: 200 }))
      .mockResolvedValueOnce(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const r = await authFetch('/api/tokens');
    expect(r.status).toBe(200);
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/auth/refresh', expect.objectContaining({
      method: 'POST', credentials: 'include',
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/tokens', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer new' }),
    }));
  });

  it('redirects to /login if refresh also 401', async () => {
    const replace = vi.fn();
    Object.defineProperty(window, 'location', { value: { replace }, writable: true });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 401 }))
      .mockResolvedValueOnce(new Response('', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/tokens');
    expect(replace).toHaveBeenCalledWith('/login');
  });

  it('fetches CSRF token before unsafe requests and attaches X-CSRF-Token', async () => {
    const fetchMock = vi.fn()
      // /api/csrf prefetch
      .mockResolvedValueOnce(new Response('{"csrf":"tok-abc"}', { status: 200 }))
      // POST /api/models
      .mockResolvedValueOnce(new Response('{"id":"x"}', { status: 201 }));
    vi.stubGlobal('fetch', fetchMock);
    const r = await authFetch('/api/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    expect(r.status).toBe(201);
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/csrf', expect.objectContaining({
      credentials: 'include',
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/models', expect.objectContaining({
      headers: expect.objectContaining({ 'X-CSRF-Token': 'tok-abc' }),
    }));
  });

  it('does not fetch CSRF for bypass-listed unsafe paths (/api/auth, /api/setup)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/auth/sse-ticket', { method: 'POST', body: '{}' });
    await authFetch('/api/setup/welcome', { method: 'POST', body: '{}' });
    // Exactly one call per authFetch — no /api/csrf prefetch.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).not.toHaveBeenCalledWith('/api/csrf', expect.anything());
  });

  it('does not fetch CSRF for safe (GET) requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/models');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/api/models', expect.anything());
  });

  it('on 403, drops cached token, refetches, and replays unsafe request once', async () => {
    setCsrfToken('stale');
    const fetchMock = vi.fn()
      // First POST attempt with the stale token → 403
      .mockResolvedValueOnce(new Response('{"detail":"csrf token invalid"}', { status: 403 }))
      // /api/csrf refetch
      .mockResolvedValueOnce(new Response('{"csrf":"fresh"}', { status: 200 }))
      // Replay POST with fresh token → 201
      .mockResolvedValueOnce(new Response('{"id":"x"}', { status: 201 }));
    vi.stubGlobal('fetch', fetchMock);
    const r = await authFetch('/api/models', { method: 'POST', body: '{}' });
    expect(r.status).toBe(201);
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/csrf', expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/models', expect.objectContaining({
      headers: expect.objectContaining({ 'X-CSRF-Token': 'fresh' }),
    }));
  });

  it('reuses the cached CSRF token across multiple unsafe requests', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('{"csrf":"tok-1"}', { status: 200 }))
      .mockResolvedValueOnce(new Response('{}', { status: 201 }))
      .mockResolvedValueOnce(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await authFetch('/api/models', { method: 'POST', body: '{}' });
    await authFetch('/api/models/abc', { method: 'DELETE' });
    // Only one /api/csrf prefetch across both POST/DELETE calls.
    const csrfCalls = fetchMock.mock.calls.filter((c) => c[0] === '/api/csrf');
    expect(csrfCalls).toHaveLength(1);
  });

  // ---------------------------------------------------------------------
  // Eager-refresh guard (#50)
  // ---------------------------------------------------------------------

  it('eager-refreshes before the first request when accessToken is null', async () => {
    // Pre-fix scenario: hard reload / new tab / deep link. The session
    // cookie is valid (Set-Cookie from prior login), but the in-memory
    // module-level `accessToken` is null. Three parallel SWR fetchers
    // on /stats fired a 401 each, then recovered via the post-401 path
    // below. That worked but logged three 401s in console on every
    // page load.
    //
    // Post-fix: with accessToken === null, authFetch awaits refresh()
    // BEFORE issuing the resource request. The replay path is no
    // longer exercised on the first request — proven here by asserting
    // /api/version is called EXACTLY ONCE (not twice — i.e. no
    // 401-replay) and carries a Bearer header.
    setAccessToken(null);
    const fetchMock = vi.fn()
      // call 1: /api/auth/refresh → fresh token
      .mockResolvedValueOnce(new Response('{"access_token":"fresh","expires_in":900}', { status: 200 }))
      // call 2: /api/version → 200 with the fresh bearer header
      .mockResolvedValueOnce(new Response('{"version":"v1"}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/version');
    expect(r.status).toBe(200);

    // Exactly two fetch calls — refresh, then the resource. If the
    // eager guard didn't fire, the sequence would be GET 401 →
    // refresh → GET 200 = three calls. Pin the count to prove the
    // 401-replay was NOT exercised here.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    // Order matters: refresh first, then /api/version.
    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/auth/refresh', expect.objectContaining({
      method: 'POST', credentials: 'include',
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/version', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer fresh' }),
    }));
  });

  it('skips eager refresh when accessToken is already set', async () => {
    // Counter-test: the guard is null-gated. If accessToken is present
    // (typical mid-session) we must NOT issue a refresh on every call —
    // that would double the request count for the entire app lifetime.
    setAccessToken('still-valid');
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await authFetch('/api/version');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/api/version', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer still-valid' }),
    }));
  });

  it('three parallel authFetch calls share one /api/auth/refresh', async () => {
    // The whole point of the eager guard (#50): on hard reload, SWR
    // fires several fetchers in parallel — pre-fix each one issued a
    // 401-then-refresh round trip, logging N 401s in console. With
    // the guard, all N callers await the SAME refresh() promise
    // because refresh() de-dupes via the module-level `refreshing`
    // promise (see auth-fetch.ts).
    //
    // We prove this by mocking /api/auth/refresh as a deferred promise
    // so all three authFetch calls reach `await refresh()` BEFORE the
    // refresh resolves. The first caller sets `refreshing` to the
    // deferred promise; the next two synchronously hit
    // `if (refreshing) return refreshing` and share it. When we
    // resolve the deferred, all three proceed in lockstep.
    setAccessToken(null);

    let resolveRefresh!: (r: Response) => void;
    const refreshPromise = new Promise<Response>((resolve) => {
      resolveRefresh = resolve;
    });

    let refreshCallCount = 0;
    const resourceCalls: string[] = [];
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url === '/api/auth/refresh') {
        refreshCallCount++;
        return refreshPromise;
      }
      resourceCalls.push(url);
      return Promise.resolve(new Response('{}', { status: 200 }));
    });
    vi.stubGlobal('fetch', fetchMock);

    // Fire three concurrent authFetch calls (mirrors SWR firing
    // /api/version, /api/models, /api/stats in parallel on hard
    // reload). Don't await yet — we need all three to enter
    // `await refresh()` before the refresh resolves.
    const calls = Promise.all([
      authFetch('/api/version'),
      authFetch('/api/models'),
      authFetch('/api/stats'),
    ]);

    // Yield once so all three authFetch microtasks reach the
    // `await refresh()` line. After this tick, `refreshing` is
    // populated and all three callers are queued on it.
    await Promise.resolve();

    // Resolve the deferred refresh — now all three resource fetches
    // can proceed with the fresh bearer token.
    resolveRefresh(
      new Response('{"access_token":"fresh","expires_in":900}', { status: 200 }),
    );

    const results = await calls;
    expect(results.map((r) => r.status)).toEqual([200, 200, 200]);

    // The contract: exactly ONE refresh, exactly THREE resource calls.
    // Pre-fix this would have been three refreshes (one per parallel
    // 401-recovery path) plus three 401s plus three replays = nine
    // calls total.
    expect(refreshCallCount).toBe(1);
    expect(resourceCalls.sort()).toEqual(
      ['/api/models', '/api/stats', '/api/version'],
    );

    // Bearer header on every resource call must reflect the fresh
    // token from the single refresh.
    for (const url of ['/api/version', '/api/models', '/api/stats']) {
      expect(fetchMock).toHaveBeenCalledWith(url, expect.objectContaining({
        headers: expect.objectContaining({ Authorization: 'Bearer fresh' }),
      }));
    }
  });

  // #251: the SSE ticket mint is under /api/auth/ but requires the JWT. On a
  // hard reload it used to go out with no bearer, 401, refresh, and replay —
  // two ticket POSTs per stream (NavBar's header metrics, on every page load).
  it('eager-refreshes before minting an SSE ticket: one POST, with the bearer', async () => {
    setAccessToken(null);
    const fetchMock = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
      if (url === '/api/auth/refresh') return new Response('{"access_token":"fresh"}', { status: 200 });
      const auth = (init?.headers as Record<string, string> | undefined)?.Authorization;
      return auth ? new Response('{"ticket":"t"}', { status: 200 }) : new Response('{}', { status: 401 });
    });
    vi.stubGlobal('fetch', fetchMock);

    const r = await authFetch('/api/auth/sse-ticket', { method: 'POST', body: '{"path":"/x"}' });

    expect(r.status).toBe(200);
    const urls = fetchMock.mock.calls.map((c) => c[0]);
    expect(urls).toEqual(['/api/auth/refresh', '/api/auth/sse-ticket']);
    expect(fetchMock).toHaveBeenLastCalledWith('/api/auth/sse-ticket', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer fresh' }),
    }));
  });

  it('does NOT eager-refresh on bypass paths (/api/auth/*, /api/csrf) to avoid recursion', async () => {
    // The crux of the bypass: refresh() itself POSTs /api/auth/refresh.
    // If that POST went through authFetch (it doesn't, but a future
    // refactor might), and authFetch's eager guard called refresh()
    // again, we'd deadlock on the `refreshing` promise. Belt-and-
    // suspenders: even when authFetch IS called directly with an
    // /api/auth/* or /api/csrf path while accessToken is null, the
    // guard must short-circuit and let the resource fetch through
    // anonymously.
    setAccessToken(null);
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await authFetch('/api/auth/login', { method: 'POST', body: '{}' });
    await authFetch('/api/csrf');

    // Two calls total — one per authFetch — and NO /api/auth/refresh.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const refreshCalls = fetchMock.mock.calls.filter(
      (c) => c[0] === '/api/auth/refresh',
    );
    expect(refreshCalls).toHaveLength(0);
  });
});
