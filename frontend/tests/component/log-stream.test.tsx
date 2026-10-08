import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup } from '@testing-library/react';
import { LogStream } from '@/components/models/log-stream';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// Minimal EventSource stub mirroring the pattern in sse-hook.test.tsx —
// gives the test direct access to the last constructed instance so it can
// drive onopen/onmessage from the test body.
class FakeES {
  static last: FakeES;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.last = this;
    // Match the useEventSource contract: it expects a real-event ordering
    // where the connection opens before the first message arrives.
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {
    this.closed = true;
  }
}

describe('LogStream', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('renders lines pushed via SSE', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    // useEventSource mints a ticket via POST /api/auth/sse-ticket before
    // connecting. Return a valid ticket payload.
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('{"ticket":"t1"}'),
    );
    vi.stubGlobal('fetch', fetchMock);

    render(<LogStream modelId="abc" />);

    // Before the SSE opens, the component should surface a connecting
    // placeholder so the operator sees the stream isn't dead. Pin the
    // role="status" semantics so a screen-reader user gets the same
    // signal as a sighted one (aria-live="polite" is implicit on
    // role="status").
    expect(screen.getByText(/connecting/i)).toBeInTheDocument();
    expect(screen.getByRole('status')).toBeInTheDocument();

    // Flush the ticket fetch + the FakeES constructor's onopen setTimeout.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Push a JSON-encoded line — exactly the contract emitted by
    // app/models/routes_logs.py (data: {"line":"hello"}). act() wraps the
    // setState call so React flushes the resulting render synchronously
    // under fake timers (findByText would otherwise spin its retry loop
    // waiting for a real-time tick that never arrives).
    act(() => {
      FakeES.last.onmessage?.(
        new MessageEvent('message', { data: '{"line":"hello world"}' }),
      );
    });

    expect(screen.getByText(/hello world/)).toBeInTheDocument();
    // Once at least one line has been delivered, the connecting placeholder
    // must be gone — otherwise a stuck reconnect would falsely advertise
    // "Connecting…" after data is already flowing.
    expect(screen.queryByText(/connecting/i)).not.toBeInTheDocument();
    // The streaming container is role="log" (implicit aria-live="polite",
    // aria-atomic="false") so subsequent lines are announced as deltas.
    expect(screen.getByRole('log')).toBeInTheDocument();
  });

  it('shows a "(no log lines yet)" placeholder while connected with empty buffer', async () => {
    // Regression for v2026.05.15.5: a model that connects to the SSE
    // stream but has produced no stdout yet (e.g. vLLM mid-load of a
    // 20GB weight set) previously rendered a blank <div role="log">.
    // Operators couldn't tell whether the stream was healthy or stuck.
    // The component now shows an explicit placeholder until the first
    // line arrives.
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('{"ticket":"t1"}'),
    ));

    render(<LogStream modelId="abc" />);

    // Flush the ticket-mint promise and the FakeES onopen setTimeout.
    // After this point sse.status === 'connected' but lines is still
    // empty — exactly the case the placeholder targets.
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(screen.getByText(/no log lines yet/i)).toBeInTheDocument();
    // role="status" (polite live region) — never "alert", because a
    // waiting state is benign. Connecting placeholder is also status,
    // so the transition is announce-quiet for screen readers.
    expect(screen.getByRole('status')).toBeInTheDocument();
    // No log container yet — the role="log" region only mounts once at
    // least one line has been delivered. This is the inverse of the
    // assertion in the happy-path test above.
    expect(screen.queryByRole('log')).not.toBeInTheDocument();

    // Push the first line and verify the placeholder disappears.
    act(() => {
      FakeES.last.onmessage?.(
        new MessageEvent('message', { data: '{"line":"first output"}' }),
      );
    });

    expect(screen.queryByText(/no log lines yet/i)).not.toBeInTheDocument();
    expect(screen.getByText(/first output/)).toBeInTheDocument();
    expect(screen.getByRole('log')).toBeInTheDocument();
  });
});
