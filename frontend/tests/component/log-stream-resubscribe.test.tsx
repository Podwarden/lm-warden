/**
 * Issue #52 / #53 — LogStream does NOT churn its EventSource on every
 * model-status transition.
 *
 * Pre-#52 / commit 7df62a3 (the #50 fix) shipped status-keying:
 *   <LogStream key={`${id}:${data.status}`} modelId={id} />
 * Every status flip therefore tore down EventSource, minted a fresh
 * ticket, opened a new EventSource, and racked up 503s through the
 * Next.js rewrite proxy as the upstream socket teardown was visible
 * to the proxy during the `pulled → loading` window. The hook then
 * drove the 503s into reconnect/backoff while the panel sat blank —
 * which is the user-visible #52 symptom.
 *
 * Empirical evidence from devops: the in-pod SSE path (curl from
 * inside the API container during an active load) delivers live vLLM
 * startup lines as expected; the file grows, the fd tracks, the
 * inode is stable. The 503 storm is a proxy artifact of the
 * status-keyed remount, NOT a backend tail bug.
 *
 * Post-fix page.tsx keys on model id alone and passes the status as
 * a prop:
 *   <LogStream key={id} modelId={id} status={data.status} />
 *
 * This test pins the new contract:
 *   1. A status change on the parent does NOT cause a LogStream remount.
 *      The existing EventSource stays open through pulled → loading →
 *      loaded transitions.
 *   2. A genuine remount (key change — e.g. operator navigates to a
 *      different model's detail page) DOES tear down and re-open with
 *      a fresh ticket. This is the same-modelId-stable behaviour
 *      previously tested as a negative control.
 *   3. A status that is non-log-producing (`registered`) skips the
 *      EventSource entirely — gated by the hook's `enabled` option.
 *   4. A status flip from non-log-producing → producing opens the
 *      EventSource exactly once.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act, cleanup } from '@testing-library/react';
import { LogStream } from '@/components/models/log-stream';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// Mirror the FakeES pattern from log-stream.test.tsx so this test reads
// as a peer rather than introducing a new mock shape.
class FakeES {
  static instances: FakeES[] = [];
  static last: FakeES | null = null;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.instances.push(this);
    FakeES.last = this;
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {
    this.closed = true;
  }
}

describe('LogStream — no re-subscribe on status change (post-#52/#53)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    FakeES.instances = [];
    FakeES.last = null;
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('status change keeps the SAME EventSource instance — no churn through pulled → loading', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    // Distinct ticket per mint matches the post-#51 backend contract.
    // If a regression silently caused a remount, we'd see the second
    // ticket value flow into a second EventSource URL.
    let ticketCounter = 0;
    const fetchMock = vi.fn().mockImplementation(() => {
      ticketCounter += 1;
      return Promise.resolve(
        new Response(JSON.stringify({ ticket: `t-${ticketCounter}` })),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    // Initial mount — parent renders with status="pulled". Both the
    // modelId AND the React key are stable across the status flip
    // below (page.tsx now keys only on id, NOT on status).
    const { rerender } = render(
      <LogStream key="m1" modelId="m1" status="pulled" />,
    );

    // Flush the preflight fetch + the FakeES onopen setTimeout so the
    // first EventSource is actually constructed before we assert.
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(FakeES.instances).toHaveLength(1);
    const firstES = FakeES.instances[0];
    expect(firstES.closed).toBe(false);
    expect(firstES.url).toContain('/api/models/m1/logs/stream');
    expect(firstES.url).toContain('ticket=t-1');

    // Status flip — pulled → loading. This is the exact transition
    // that, pre-fix, would have churned the EventSource. Same key,
    // same modelId, different status prop.
    rerender(<LogStream key="m1" modelId="m1" status="loading" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // Contract: NO new EventSource constructed. NO close on the
    // existing one. The hook's path is unchanged and `enabled` is
    // still true (loading is a log-producing state), so the effect
    // doesn't re-run.
    expect(FakeES.instances).toHaveLength(1);
    expect(firstES.closed).toBe(false);

    // And the ticket-mint fetch was only called once — no extra
    // preflight burned on a no-op transition.
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // One more flip — loading → loaded. Still no churn.
    rerender(<LogStream key="m1" modelId="m1" status="loaded" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(FakeES.instances).toHaveLength(1);
    expect(firstES.closed).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('genuine remount (different React key — e.g. operator switches model) DOES open a new EventSource with a distinct ticket', async () => {
    // This is the "real" remount path — the operator navigates to a
    // different model's detail page. page.tsx's key is the model id, so
    // a different id forces React to unmount/remount the subtree. The
    // hook tears down the old EventSource and opens a new one.
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    let ticketCounter = 0;
    const fetchMock = vi.fn().mockImplementation(() => {
      ticketCounter += 1;
      return Promise.resolve(
        new Response(JSON.stringify({ ticket: `t-${ticketCounter}` })),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const { rerender } = render(
      <LogStream key="m1" modelId="m1" status="loaded" />,
    );
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(FakeES.instances).toHaveLength(1);
    const firstES = FakeES.instances[0];

    // Different key — React unmounts the m1 subtree and mounts m2 fresh.
    // The hook's path also changes (different modelId in the URL), so
    // even on a parent that didn't change keys, the effect-deps would
    // re-run. Belt-and-suspenders: assert BOTH the close and the new
    // construction.
    rerender(<LogStream key="m2" modelId="m2" status="loaded" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(FakeES.instances).toHaveLength(2);
    expect(firstES.closed).toBe(true);
    expect(FakeES.instances[1]).not.toBe(firstES);
    expect(FakeES.instances[1].url).toContain('/api/models/m2/logs/stream');

    // Distinct ticket on the second mint — same #51 contract as before.
    const firstTicket = new URL(firstES.url, 'http://x').searchParams.get('ticket');
    const secondTicket = new URL(
      FakeES.instances[1].url,
      'http://x',
    ).searchParams.get('ticket');
    expect(firstTicket).toBe('t-1');
    expect(secondTicket).toBe('t-2');
    expect(secondTicket).not.toBe(firstTicket);
  });

  it('non-log-producing status ("registered") does NOT open an EventSource at all', async () => {
    // Issue #53 — a freshly-registered model has never been pulled, no
    // log file content exists, opening EventSource just burns a ticket
    // and surfaces a "no log lines yet" placeholder either way. The
    // gate is in log-stream.tsx, plumbed via the hook's `enabled` opt.
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{"ticket":"t-never"}'),
    );
    vi.stubGlobal('fetch', fetchMock);

    render(<LogStream key="m1" modelId="m1" status="registered" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // No EventSource opened. No ticket-mint fetched.
    expect(FakeES.instances).toHaveLength(0);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('status flips into a log-producing state — opens EventSource exactly once', async () => {
    // Complementary to the previous case: when the parent flips status
    // from "registered" to "pulling" (operator clicks Pull), the gate
    // opens and the hook should connect. This tests the `enabled`
    // dependency in the effect-deps array.
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{"ticket":"t-pulling"}'),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { rerender } = render(
      <LogStream key="m1" modelId="m1" status="registered" />,
    );
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(FakeES.instances).toHaveLength(0);

    // Operator clicks Pull — status flips. Same key, same modelId.
    rerender(<LogStream key="m1" modelId="m1" status="pulling" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(FakeES.instances).toHaveLength(1);
    expect(FakeES.instances[0].closed).toBe(false);
  });
});
