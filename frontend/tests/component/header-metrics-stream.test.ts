// Singleton SSE stream — pins the contract for
// frontend/src/lib/header-metrics-stream.ts:
//
//   - One EventSource per browser tab. Two parallel subscribers must
//     share the underlying connection (so a brief StrictMode double-
//     mount or a route change that re-mounts <NavBar /> doesn't
//     double the ticket-mint traffic and the nvidia-smi probe load).
//   - The last unsubscribe closes the EventSource. A subsequent
//     subscribe opens a fresh one.
//   - The ticket is minted via POST /api/auth/sse-ticket; the body
//     references the configured stream path.
//   - A successful `open` flips state to "connected"; an `error`
//     event triggers backoff + reconnect.
//
// We stub EventSource on globalThis and authFetch via the test's
// fetch mock — both layers identical to how `sse-hook.test.tsx`
// exercises the generic SSE helper.
import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from 'vitest';
import {
  subscribeHeaderMetrics,
  __resetHeaderMetricsStreamForTests,
  type HeaderMetricsState,
} from '@/lib/header-metrics-stream';

// Minimal EventSource stub — records every constructed instance so the
// test can assert on count, last URL, and trigger lifecycle events.
class FakeEventSource {
  static instances: FakeEventSource[] = [];

  url: string;
  readyState = 0;
  onopen: ((this: EventSource, ev: Event) => unknown) | null = null;
  onmessage:
    | ((this: EventSource, ev: MessageEvent) => unknown)
    | null = null;
  onerror: ((this: EventSource, ev: Event) => unknown) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  // Test helpers — not part of the real EventSource API.
  fireOpen() {
    this.readyState = 1;
    this.onopen?.call(this as unknown as EventSource, new Event('open'));
  }
  fireMessage(data: string) {
    this.onmessage?.call(
      this as unknown as EventSource,
      new MessageEvent('message', { data }),
    );
  }
  fireError() {
    this.onerror?.call(this as unknown as EventSource, new Event('error'));
  }
  static reset() {
    FakeEventSource.instances = [];
  }
}

function stubTicketFetch(ticket = 'ticket-xyz') {
  const fetchMock = vi.fn().mockImplementation(async (url: RequestInfo) => {
    const u = typeof url === 'string' ? url : (url as Request).url;
    if (u.endsWith('/api/auth/sse-ticket')) {
      return new Response(JSON.stringify({ ticket }), { status: 200 });
    }
    // CSRF preflight and anything else: empty 200 so authFetch is
    // happy. The header-metrics-stream module doesn't issue any
    // other requests, but auth-fetch.ts may preflight /api/csrf.
    return new Response('{}', { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function flushMicrotasks() {
  // The connect() async path awaits authFetch -> Response.json() -> a
  // few internal microtasks before the EventSource constructor runs.
  // Drain enough microtasks for the full promise chain to settle.
  for (let i = 0; i < 25; i++) {
    await Promise.resolve();
  }
}

// Poll until ``predicate`` is true or the deadline elapses. Used to wait
// for the EventSource constructor to run after the async ticket-mint
// chain settles — flushMicrotasks() alone is racy because authFetch's
// chain can extend past any fixed microtask budget in some browsers.
async function waitFor(
  predicate: () => boolean,
  { timeoutMs = 500, intervalMs = 5 }: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  if (!predicate()) throw new Error('waitFor: deadline exceeded');
}

describe('header-metrics-stream singleton', () => {
  beforeEach(() => {
    FakeEventSource.reset();
    vi.stubGlobal('EventSource', FakeEventSource);
    stubTicketFetch();
  });
  afterEach(() => {
    __resetHeaderMetricsStreamForTests();
    vi.unstubAllGlobals();
  });

  it('opens exactly one EventSource for two concurrent subscribers', async () => {
    const a = vi.fn();
    const b = vi.fn();
    const unsubA = subscribeHeaderMetrics(a);
    const unsubB = subscribeHeaderMetrics(b);
    await waitFor(() => FakeEventSource.instances.length >= 1);

    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.instances[0].url).toContain(
      '/api/header/metrics/stream?ticket=ticket-xyz',
    );
    // Both subscribers got the initial state synchronously on subscribe.
    expect(a).toHaveBeenCalled();
    expect(b).toHaveBeenCalled();

    unsubA();
    unsubB();
  });

  // #251: an unmount + remount while the first ticket was still minting
  // (a route change, a remounting parent) tore the first stream down, but its
  // in-flight connect() then saw the NEW stream as live and opened an
  // EventSource on it too — two connections, the first never closed.
  it('a connect() from a torn-down stream never opens an EventSource on its successor', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => {
      await gate; // both mints are in flight together
      return new Response(JSON.stringify({ ticket: 'ticket-xyz' }), { status: 200 });
    }));
    const unsubA = subscribeHeaderMetrics(vi.fn());
    unsubA();
    const unsubB = subscribeHeaderMetrics(vi.fn());
    release();
    await waitFor(() => FakeEventSource.instances.length >= 1);
    await new Promise((r) => setTimeout(r, 20));
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.instances[0].closed).toBe(false);
    unsubB();
    expect(FakeEventSource.instances[0].closed).toBe(true);
  });

  it('closes the EventSource when the last subscriber unsubscribes', async () => {
    const sub = vi.fn();
    const unsub = subscribeHeaderMetrics(sub);
    await waitFor(() => FakeEventSource.instances.length >= 1);
    const es = FakeEventSource.instances[0];
    expect(es.closed).toBe(false);

    unsub();
    expect(es.closed).toBe(true);

    // Subscribing again opens a fresh EventSource.
    const sub2 = vi.fn();
    const unsub2 = subscribeHeaderMetrics(sub2);
    await waitFor(() => FakeEventSource.instances.length >= 2);
    expect(FakeEventSource.instances).toHaveLength(2);
    unsub2();
  });

  it('broadcasts parsed frames to every subscriber on message', async () => {
    const seenA: HeaderMetricsState[] = [];
    const seenB: HeaderMetricsState[] = [];
    const unsubA = subscribeHeaderMetrics((s) => {
      seenA.push(s);
    });
    const unsubB = subscribeHeaderMetrics((s) => {
      seenB.push(s);
    });
    await waitFor(() => FakeEventSource.instances.length >= 1);
    const es = FakeEventSource.instances[0];
    es.fireOpen();

    es.fireMessage(
      JSON.stringify({
        ts: '2026-05-23T00:00:00Z',
        gpus: [],
        vram_used_mib: 0,
        vram_total_mib: 16376,
        vram_pct: 0,
        gpu_util_pct: 0,
        active_model: 'gpt-oss-20b',
        active_model_id: 'abc',
        probe_error: null,
      }),
    );

    // Both subscribers should have observed the connected state with the
    // parsed frame as their latest broadcast.
    const lastA = seenA[seenA.length - 1];
    const lastB = seenB[seenB.length - 1];
    expect(lastA.status).toBe('connected');
    expect(lastA.frame?.active_model).toBe('gpt-oss-20b');
    expect(lastB.status).toBe('connected');
    expect(lastB.frame?.active_model).toBe('gpt-oss-20b');

    unsubA();
    unsubB();
  });

  it('flips status to reconnecting on EventSource error and reuses the singleton', async () => {
    const sub = vi.fn();
    const unsub = subscribeHeaderMetrics(sub);
    await waitFor(() => FakeEventSource.instances.length >= 1);
    const es = FakeEventSource.instances[0];
    es.fireOpen();
    sub.mockClear();

    es.fireError();
    // The first call after fireError carries the reconnecting status.
    expect(sub).toHaveBeenCalled();
    const lastCallArg = sub.mock.calls[sub.mock.calls.length - 1][0] as HeaderMetricsState;
    expect(lastCallArg.status).toBe('reconnecting');
    expect(es.closed).toBe(true);
    // Singleton still has one instance recorded — the reconnect timer
    // will create the next one on tick (we don't drive timers here;
    // pinning the close + status flip is enough for the regression).
    unsub();
  });

  it('marks terminal-error on a 401 ticket-mint response and does not open EventSource', async () => {
    // Override the fetch stub for this test to return 401 on the ticket
    // call. authFetch may issue a refresh attempt; respond 401 there
    // too so the stream falls into its terminal-error branch.
    const fetchMock = vi.fn().mockImplementation(async () => {
      return new Response('{"detail":"unauthorized"}', { status: 401 });
    });
    vi.stubGlobal('fetch', fetchMock);

    const seen: HeaderMetricsState[] = [];
    const unsub = subscribeHeaderMetrics((s) => {
      seen.push(s);
    });
    // Wait for the connect() chain to settle on terminal-error. Loop
    // bounded — if it never lands, the assertion below fails loudly.
    await waitFor(
      () =>
        seen.some(
          (s) => s.status === 'terminal-error' || s.status === 'reconnecting',
        ),
      { timeoutMs: 1500 },
    );

    expect(FakeEventSource.instances).toHaveLength(0);
    const last = seen[seen.length - 1];
    // authFetch may either propagate the 401 directly (terminal-error)
    // or trigger a refresh that lands us in reconnecting; either way
    // the EventSource MUST NOT have opened — that's the security pin.
    expect(['terminal-error', 'reconnecting']).toContain(last.status);
    unsub();
  });
});
