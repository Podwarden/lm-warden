/**
 * Regression for v2026.05.15.3 — peer-401 cascade no longer strands the
 * live-log panel.
 *
 * Pre-fix scenario (v2026.05.15.2 production):
 *   1. The session token expires (prod uses an itsdangerous-signed
 *      `vw_session` cookie; the cascade fix is auth-mechanism-agnostic
 *      and works the same way for Bearer-JWT setups).
 *   2. Every authenticated fetch in flight returns 401 in parallel —
 *      `/api/version`, `/api/models`, etc.
 *   3. Each 401 ran the `auth-fetch.ts` fallback that calls
 *      `window.location.replace('/login')`. Multiple overlapping
 *      navigations confused the browser and, critically, stranded the
 *      SSE preflight (`POST /api/auth/sse-ticket`) before it could mint a
 *      ticket. `useEventSource` saw `authFetch` resolve to a value it
 *      didn't classify, never transitioned out of `connecting`, and
 *      `LogStream` rendered a blank "Connecting…" placeholder for the
 *      brief window before the navigation completed.
 *
 * Post-fix contract pinned here:
 *
 *   1. Only the FIRST 401 to exhaust refresh fires the redirect. A
 *      module-level `loginRedirectInFlight` flag de-dupes subsequent
 *      401s in the same page lifetime. `window.location.replace` is
 *      called exactly once, no matter how many concurrent 401s land.
 *   2. `useEventSource` checks `isLoginRedirectInFlight()` up front and
 *      transitions to `terminal-error` with code 401 instead of firing
 *      a preflight that will inevitably 401 itself during the unload
 *      window.
 *   3. If the peer 401 fires AFTER the SSE preflight has already
 *      started, the hook completes its preflight normally — the
 *      cascade fix doesn't break the happy path. The redirect is
 *      already in flight, so the operator sees a brief terminal banner
 *      and then the page navigates.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup } from '@testing-library/react';
import { LogStream } from '@/components/models/log-stream';
import {
  authFetch,
  setAccessToken,
  setCsrfToken,
  isLoginRedirectInFlight,
  __resetLoginRedirectInFlightForTests,
} from '@/lib/auth-fetch';

class FakeES {
  static last: FakeES | null = null;
  static constructed = 0;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.last = this;
    FakeES.constructed += 1;
    // Match real-event ordering — fire onopen async so the hook
    // resolves to "connected" only after the preflight settles.
    setTimeout(() => this.onopen?.(), 0);
  }
  close() { this.closed = true; }
}

/** Replace `window.location` with a stub whose `replace` is a spy. jsdom
 *  25 marks the real `window.location.replace` non-configurable, so a
 *  bare `vi.spyOn` would throw. The auth-fetch unit tests use the same
 *  trick — copy the pattern so this regression test reads as a peer to
 *  those. */
function stubWindowLocationReplace(): ReturnType<typeof vi.fn> {
  const replace = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { replace, origin: 'http://localhost' },
    writable: true,
    configurable: true,
  });
  return replace;
}

describe('LogStream — peer-401 cascade', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    FakeES.last = null;
    FakeES.constructed = 0;
    // The flag is module-level; reset between cases so test isolation
    // is preserved. Production code never calls this.
    __resetLoginRedirectInFlightForTests();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('peer 401 BEFORE preflight: flag set, terminal "session expired" banner, replace() called exactly once', async () => {
    const replace = stubWindowLocationReplace();
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);

    // fetch mock:
    //   call 1: the SIBLING `authFetch('/api/version')` → 401
    //   call 2: that 401's refresh attempt          → 401 (refresh fails)
    //   call 3..n: the LogStream preflight should NEVER run a real
    //              network call because the redirect flag short-circuits.
    //              If it does (regression), we'd see a third fetch — the
    //              assertions below pin that.
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      // Any subsequent call (shouldn't happen) returns a benign 200.
      .mockResolvedValue(new Response('{"ticket":"t-should-not-fire"}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    // Fire the sibling 401 BEFORE rendering LogStream.
    expect(isLoginRedirectInFlight()).toBe(false);
    const peerResponse = await authFetch('/api/version');
    expect(peerResponse.status).toBe(401);
    expect(isLoginRedirectInFlight()).toBe(true);
    expect(replace).toHaveBeenCalledTimes(1);

    // Now render LogStream — its preflight should short-circuit on the
    // flag and surface the terminal banner without firing fetch #3.
    render(<LogStream modelId="abc" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // The auth-specific terminal copy is "Stream unavailable — your
    // session expired. Please re-login." — role=alert so a screen-reader
    // announces it synchronously.
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/session expired/i)).toBeInTheDocument();
    expect(screen.queryByText(/connecting/i)).not.toBeInTheDocument();

    // Exactly two fetch calls total — the sibling GET and its refresh.
    // No preflight, no EventSource construction.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(FakeES.constructed).toBe(0);
    // The crux: window.location.replace ran ONCE, not twice.
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
  });

  it('peer 401 DURING preflight: catch-block flag re-check terminates instead of reconnecting', async () => {
    // This is the precise race the new flag re-check at sse.ts catch block
    // exists to defend against. Order of events:
    //
    //   1. LogStream renders → useEventSource fires preflight
    //      `POST /api/auth/sse-ticket`. The fetch is in flight (pending).
    //   2. A sibling authFetch (e.g. `/api/version`) returns 401 in
    //      parallel and exhausts refresh. It sets the flag and fires
    //      replace('/login') exactly once.
    //   3. The browser starts tearing down the page. The pending preflight
    //      fetch rejects with a network error (mocked here by rejecting
    //      the pending promise after the flag has flipped).
    //   4. The catch block in sse.ts runs. Without the flag re-check, it
    //      would call scheduleReconnect(null) and loop into another
    //      preflight that would also fail during the unload window — the
    //      old blank-panel behaviour. WITH the re-check, it short-circuits
    //      to terminal-error(401) which the LogStream renders as the
    //      "session expired" banner.
    const replace = stubWindowLocationReplace();
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);

    let rejectPreflight!: (e: Error) => void;
    const preflightPending = new Promise<Response>((_resolve, reject) => {
      rejectPreflight = reject;
    });

    // fetch sequence:
    //   call 1: preflight POST /api/auth/sse-ticket → HANGS (pending)
    //   call 2: sibling GET /api/version            → 401
    //   call 3: sibling's refresh attempt           → 401 (refresh fails)
    const fetchMock = vi.fn()
      .mockReturnValueOnce(preflightPending)
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    render(<LogStream modelId="abc" />);

    // Microtask flush — the preflight authFetch is in flight, hung on the
    // pending Promise. The flag must still be clean at this point.
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(isLoginRedirectInFlight()).toBe(false);
    expect(FakeES.constructed).toBe(0);

    // Sibling 401 lands WHILE the preflight is in flight. authFetch runs
    // refresh (also 401), exhausts it, sets the flag, fires replace once.
    const peerResponse = await authFetch('/api/version');
    expect(peerResponse.status).toBe(401);
    expect(isLoginRedirectInFlight()).toBe(true);
    expect(replace).toHaveBeenCalledTimes(1);

    // Now simulate the unload-window network drop: the pending preflight
    // rejects. This is the path the new catch-block re-check defends.
    // Without the re-check, the hook would call scheduleReconnect(null);
    // with it, it transitions straight to terminal-error(401).
    rejectPreflight(new Error('network'));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // Contract: terminal "session expired" banner, NOT a reconnecting
    // state. role=alert because tone=error in LogStream's status renderer.
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/session expired/i)).toBeInTheDocument();
    expect(screen.queryByText(/retrying/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/connecting/i)).not.toBeInTheDocument();

    // EventSource was never constructed — preflight never resolved with
    // a ticket. This is the proof the catch path took the early return,
    // not the happy path.
    expect(FakeES.constructed).toBe(0);

    // Redirect fired exactly once (from the sibling, not from the
    // preflight failure). The catch-block re-check is the de-dup here.
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
  });

  it('subsequent peer 401s after redirect already fired: no second replace(), 401 still returned to caller', async () => {
    // Belt-and-suspenders for the de-dup in auth-fetch itself. Once the
    // flag is up, every further 401 returns the 401 Response to its caller
    // (so error handling can run) but does NOT re-fire replace().
    const replace = stubWindowLocationReplace();
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);

    // Sequence:
    //   First peer:  GET /api/version → 401, refresh → 401 (fires replace)
    //   Second peer: GET /api/models  → 401, refresh → 401 (NO replace)
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }))
      .mockResolvedValueOnce(new Response('Unauthorized', { status: 401 }));
    vi.stubGlobal('fetch', fetchMock);

    const first = await authFetch('/api/version');
    expect(first.status).toBe(401);
    expect(isLoginRedirectInFlight()).toBe(true);
    expect(replace).toHaveBeenCalledTimes(1);

    const second = await authFetch('/api/models');
    expect(second.status).toBe(401);
    // The cap: still exactly one replace() across both peers.
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
  });
});
