/**
 * Regression tests for v2026.05.15.2 — useEventSource state machine.
 *
 * Pre-fix behaviour: any error (4xx ticket-mint, 5xx ticket-mint, network
 * drop, EventSource onerror) silently fell into an unbounded exp-backoff
 * loop. The UI surfaced nothing — operators saw a stuck "Connecting…"
 * placeholder for an inaccessible or non-existent stream.
 *
 * Post-fix contract pinned here:
 *
 *   1. 4xx (except 429) on the ticket-mint path → state.status ===
 *      'terminal-error' with errorCode set. No further reconnect attempts.
 *   2. 5xx → reconnecting, with attempts incrementing until MAX_RECONNECT,
 *      then terminal-error.
 *   3. EventSource.onerror (after a successful onopen) → reconnecting,
 *      eventually capping at terminal-error after MAX_RECONNECT.
 *   4. A clean ticket mint + onopen → state.status === 'connected'.
 *   5. The hook tears down its timer on unmount so a slow backoff doesn't
 *      schedule a reconnect after the consumer is gone.
 *
 * The harness mirrors sse-hook.test.tsx (vitest fake timers + FakeES +
 * stubbed fetch). We don't drive the full LogStream component — that's
 * covered in log-stream-states.test.tsx — only the hook contract.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { useEventSource, MAX_RECONNECT, type SseState } from '@/lib/sse';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

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
    // Match the real-event ordering: open fires async after the
    // EventSource is wired up.
    setTimeout(() => this.onopen?.(), 0);
  }
  close() { this.closed = true; }
}

function makeJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Renders a probe component that captures every SSE state emitted by
 *  the hook. Returns the array so the test can assert on transitions. */
function renderProbe(path = '/api/models/abc/logs/stream') {
  const states: SseState[] = [];
  function Probe() {
    const s = useEventSource(path, { onMessage: () => {} });
    states.push(s);
    return null;
  }
  return { states, ...render(<Probe />) };
}

// jsdom 25 marks `Location.replace` non-configurable, so `vi.spyOn` on it
// throws. Replace the whole `window.location` with a plain object whose
// `replace` is a spy (same trick as auth-fetch.test.ts). tests/setup.ts
// restores the real location after every test.
function stubWindowLocation(pathname = '/'): ReturnType<typeof vi.fn> {
  const replace = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { replace, pathname, origin: 'http://localhost' },
    writable: true,
    configurable: true,
  });
  return replace;
}

describe('useEventSource — state machine', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    FakeES.last = null;
    FakeES.constructed = 0;
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('starts in connecting and reaches connected after a successful ticket + onopen', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(makeJsonResponse({ ticket: 't1' }));
    vi.stubGlobal('fetch', fetchMock);

    const { states, unmount } = renderProbe();
    // Initial render — before fetch has resolved.
    expect(states[0].status).toBe('connecting');

    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // After ticket-mint + FakeES onopen → connected.
    const final = states[states.length - 1];
    expect(final.status).toBe('connected');
    expect(final.errorCode).toBeNull();
    expect(final.attempts).toBe(0);
    expect(FakeES.constructed).toBe(1);

    unmount();
  });

  it('classifies a 401 ticket-mint as terminal-error and does NOT reconnect', async () => {
    // R19: the 401 branch of authFetch fires the real
    // window.location.replace('/login'); against jsdom's real location that
    // is the "Not implemented: navigation" error that used to leak into
    // other tests. Stub the location so the redirect is a recorded,
    // deterministic call — and pin that it actually happens exactly once.
    const replace = stubWindowLocation('/');
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('Unauthorized', { status: 401 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { states, unmount } = renderProbe();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    const final = states[states.length - 1];
    expect(final.status).toBe('terminal-error');
    expect(final.errorCode).toBe(401);
    // Critical: no EventSource was constructed for an auth failure.
    expect(FakeES.constructed).toBe(0);
    // The auth failure is also reported as a session bounce — exactly one
    // navigation attempt, de-duped by the loginRedirectInFlight flag.
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');

    // Advance well past any plausible backoff — terminal means terminal.
    // If the implementation regressed to retrying on 4xx, we'd see
    // additional fetch calls or a state flip back to reconnecting.
    const fetchCallsAfterTerminal = fetchMock.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(60_000); });
    expect(fetchMock).toHaveBeenCalledTimes(fetchCallsAfterTerminal);
    expect(states[states.length - 1].status).toBe('terminal-error');

    unmount();
  });

  it('classifies a 404 ticket-mint as terminal-error with the right code', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('Not Found', { status: 404 }),
    ));

    const { states, unmount } = renderProbe();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    const final = states[states.length - 1];
    expect(final.status).toBe('terminal-error');
    expect(final.errorCode).toBe(404);
    expect(FakeES.constructed).toBe(0);

    unmount();
  });

  it('keeps reconnecting on 5xx ticket-mint until MAX_RECONNECT then gives up', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('Bad Gateway', { status: 502 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { states, unmount } = renderProbe();

    // Initial attempt + 5 retries = MAX_RECONNECT+1 fetch calls before
    // the hook surrenders. Advance generously past the cumulative
    // backoff ceiling — post-#53 the sequence is
    // 250ms + 500ms + 1s + 2s + 4s ≈ 7.75s (the 5th attempt triggers
    // terminal-error on entry, no further wait) — but we keep the 120s
    // value to leave headroom for future tuning.
    await act(async () => { await vi.advanceTimersByTimeAsync(120_000); });

    // The hook should be in terminal-error with errorCode=502.
    const final = states[states.length - 1];
    expect(final.status).toBe('terminal-error');
    expect(final.errorCode).toBe(502);
    expect(final.attempts).toBeGreaterThan(MAX_RECONNECT);

    // Number of fetch calls is bounded — verify we DID stop. Allow
    // exactly MAX_RECONNECT+1 (initial + retries before the cap).
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(MAX_RECONNECT + 1);

    unmount();
  });

  it('treats 429 as transient (NOT terminal) — rate-limit deserves a retry', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn()
      // First call returns 429, subsequent calls return a fresh good
      // ticket (factory form keeps Response.body fresh per call —
      // mockResolvedValue would reuse a single Response).
      .mockResolvedValueOnce(new Response('Too Many Requests', { status: 429 }))
      .mockImplementation(async () => makeJsonResponse({ ticket: 't2' }));
    vi.stubGlobal('fetch', fetchMock);

    const { states, unmount } = renderProbe();

    // Flush initial 429 — should schedule a reconnect (NOT terminal).
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(states[states.length - 1].status).toBe('reconnecting');
    expect(states[states.length - 1].errorCode).toBe(429);

    // First reconnect backoff is 250ms (post-#53); advance well past it
    // + flush the FakeES onopen. We keep 1500ms here to leave a comfy
    // margin against future tuning of INITIAL_BACKOFF_MS.
    await act(async () => { await vi.advanceTimersByTimeAsync(1_500); });
    // Now the second fetch should have resolved with a good ticket and
    // FakeES.onopen should have fired → connected.
    expect(states[states.length - 1].status).toBe('connected');

    unmount();
  });

  it('on EventSource.onerror (after onopen) → reconnecting; capped at MAX_RECONNECT', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    // Use mockImplementation (not mockResolvedValue) so each call returns
    // a *fresh* Response — Response.body is one-shot, so reusing one
    // resolved value across reconnects makes the second .json() throw
    // (which then triggers scheduleReconnect with the 200 status and
    // pollutes the final errorCode assertion below).
    const fetchMock = vi.fn().mockImplementation(async () =>
      makeJsonResponse({ ticket: 't1' }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { states, unmount } = renderProbe();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(states[states.length - 1].status).toBe('connected');

    // Drive MAX_RECONNECT+1 errors with no intervening successful
    // re-open (FakeES.onopen does fire on each reconnect — we'd need
    // to suppress that to force the cap).
    //
    // Strategy: error → reconnect timer fires → ticket fetch happens
    // → new FakeES → its onopen fires async → state flips back to
    // connected. So a SINGLE onerror naturally recovers. To verify the
    // cap, we have to stop the FakeES onopen from firing after the
    // first.
    //
    // Easiest: swap in a non-opening FakeES variant.
    class NoOpenFakeES {
      static last: NoOpenFakeES | null = null;
      onopen?: () => void;
      onmessage?: (e: MessageEvent) => void;
      onerror?: () => void;
      closed = false;
      constructor(public url: string) {
        NoOpenFakeES.last = this;
        // Fire onerror immediately rather than onopen. This simulates
        // the case where the SSE handshake fails mid-stream after a
        // healthy ticket mint (e.g., backend nodes recycling).
        setTimeout(() => this.onerror?.(), 0);
      }
      close() { this.closed = true; }
    }
    vi.stubGlobal('EventSource', NoOpenFakeES as unknown as typeof EventSource);

    // Now drive the FIRST onerror on the already-connected FakeES.
    act(() => { FakeES.last!.onerror?.(); });

    // Walk forward past every scheduled reconnect. Each NoOpenFakeES
    // construction fires onerror after 0ms, scheduling another retry.
    // After MAX_RECONNECT failed attempts the hook surrenders.
    await act(async () => { await vi.advanceTimersByTimeAsync(120_000); });

    const final = states[states.length - 1];
    expect(final.status).toBe('terminal-error');
    // errorCode is null because EventSource doesn't surface HTTP status.
    expect(final.errorCode).toBeNull();
    expect(final.attempts).toBeGreaterThan(MAX_RECONNECT);

    unmount();
  });

  it('clears the reconnect timer on unmount — no zombie ticket fetches', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('Bad Gateway', { status: 502 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { unmount } = renderProbe();

    // One failed mint → schedules the first 250ms reconnect (post-#53).
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    const callsBeforeUnmount = fetchMock.mock.calls.length;

    unmount();

    // Advance past every plausible backoff. If the cleanup didn't
    // clear the pending timer, we'd see additional fetch calls.
    await act(async () => { await vi.advanceTimersByTimeAsync(120_000); });
    expect(fetchMock.mock.calls.length).toBe(callsBeforeUnmount);
  });

  it('burst of 503s within 250ms produces at most 2 EventSource constructions — issue #53', async () => {
    // Pre-#53 the proxy could return a 503 storm during the
    // pulled→loading status transition (Next.js rewrite-proxy
    // surfacing the upstream socket flip). With the OLD 1s initial
    // backoff a tight burst of failures still ramped reconnect
    // attempts proportional to wall-clock time, so a multi-second
    // proxy blip produced a flood of preflight POSTs and EventSource
    // constructions before settling.
    //
    // Post-#53 contract: the hook MUST honour exp-backoff timing even
    // when a burst of 503s arrives faster than the FIRST 250ms tick
    // could fire. We pin this by stubbing fetch to always return 503
    // and asking the test runner to advance time by ONLY 250ms — that
    // is exactly one INITIAL_BACKOFF_MS interval, which is the window
    // where pre-#53 behaviour would have constructed multiple
    // EventSources back-to-back through the synchronous error path.
    //
    // Note: 503 ticket-mints never construct an EventSource on the
    // post-#52 hook (the EventSource is only constructed AFTER a
    // successful mint). So the assertion is doubly bounded: 0 ES
    // constructions on this path AND <= MAX_RECONNECT+1 fetch calls.
    // The "at most 2" wording from the issue spec refers to the
    // upstream-headroom case where the proxy flips healthy mid-burst
    // and one mint succeeds — we exercise that explicitly below.
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('Service Unavailable', { status: 503 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { unmount } = renderProbe();
    // Flush initial mint (the synchronous 503 schedules a reconnect).
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    // Walk exactly one INITIAL_BACKOFF_MS interval — fires retry #1.
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });

    // Two ticket mints — initial + one retry after the 250ms tick.
    // The hook MUST NOT have synchronously fired more during the burst.
    expect(fetchMock.mock.calls.length).toBeLessThanOrEqual(2);
    // No EventSource constructed on a pure 5xx path.
    expect(FakeES.constructed).toBe(0);

    unmount();
  });

  it('burst of 503s alternating with a successful mint — exactly one EventSource opens', async () => {
    // Companion to the pure-503 burst test: simulate the proxy's
    // status-flip blip resolving on the second attempt. The hook
    // should construct exactly ONE EventSource (after the success
    // mint), and the subsequent successful onopen resets the backoff
    // so a later transient failure starts again at the floor.
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response('Service Unavailable', { status: 503 }))
      .mockResolvedValueOnce(new Response('Service Unavailable', { status: 503 }))
      .mockImplementation(async () => makeJsonResponse({ ticket: 't-recovered' }));
    vi.stubGlobal('fetch', fetchMock);

    const { states, unmount } = renderProbe();
    // Flush initial 503 + advance past first two retry windows
    // (250ms + 500ms = 750ms). Third attempt resolves with a ticket
    // and FakeES.onopen fires after another 0ms tick.
    await act(async () => { await vi.advanceTimersByTimeAsync(800); });

    expect(states[states.length - 1].status).toBe('connected');
    expect(FakeES.constructed).toBe(1);
    // Three fetch calls (two 503s + one success), bounded.
    expect(fetchMock.mock.calls.length).toBe(3);

    unmount();
  });
});
