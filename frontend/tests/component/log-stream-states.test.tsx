/**
 * Regression for v2026.05.15.2 — LogStream surfaces terminal SSE failures.
 *
 * Pre-fix: a 4xx ticket-mint left the placeholder reading
 * "Connecting to log stream…" forever. The operator had no signal that
 * the stream was inaccessible.
 *
 * Post-fix:
 *   - 401/403 ticket-mint  → red status bar "session expired" + role=alert
 *   - 404 ticket-mint      → red status bar "Log stream not found"        + role=alert
 *   - 502 (after retries)  → red status bar "Stream unavailable (HTTP 502)" + role=alert
 *   - Network drop mid-stream after exhausting MAX_RECONNECT
 *                          → red status bar with retry-count phrasing + role=alert
 *
 * The happy path is already covered by log-stream.test.tsx.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup } from '@testing-library/react';
import { LogStream } from '@/components/models/log-stream';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

class FakeES {
  static last: FakeES | null = null;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.last = this;
    setTimeout(() => this.onopen?.(), 0);
  }
  close() { this.closed = true; }
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

describe('LogStream — SSE error states', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    FakeES.last = null;
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('renders a re-login alert when the ticket-mint returns 401', async () => {
    // R19: the 401 branch of authFetch fires the real
    // window.location.replace('/login'); against jsdom's real location that
    // is the "Not implemented: navigation" error that used to leak into
    // other tests. Stub the location so the redirect is a recorded,
    // deterministic call.
    const replace = stubWindowLocation('/');
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('Unauthorized', { status: 401 }),
    ));

    render(<LogStream modelId="abc" />);

    // Initially "Connecting…" — the hook resolves to terminal-error
    // after the ticket-mint promise settles.
    expect(screen.getByText(/connecting/i)).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // The status bar should now be the auth-specific message and use
    // role="alert" so a screen-reader announces it immediately.
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/session expired/i)).toBeInTheDocument();
    expect(screen.queryByText(/connecting/i)).not.toBeInTheDocument();
    // The session bounce is fired exactly once, de-duped by the
    // loginRedirectInFlight flag.
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith('/login');
  });

  it('renders a not-found message when the ticket-mint returns 404', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('Not Found', { status: 404 }),
    ));

    render(<LogStream modelId="bogus-model" />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/not found/i)).toBeInTheDocument();
  });

  it('renders a generic HTTP-coded message for 5xx after exhausting retries', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('Bad Gateway', { status: 502 }),
    ));

    render(<LogStream modelId="abc" />);

    // Advance well past the cumulative backoff so the hook hits the cap.
    await act(async () => { await vi.advanceTimersByTimeAsync(120_000); });

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/HTTP 502/i)).toBeInTheDocument();
  });

  it('shows a "reconnecting (N/5)" status bar between attempts', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('Bad Gateway', { status: 502 }),
    ));

    render(<LogStream modelId="abc" />);
    // First failed mint → state.reconnecting with attempts=1.
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    // The reconnecting placeholder is role="status" (transient, polite)
    // not role="alert" — that's reserved for terminal failures.
    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText(/retrying \(1\/5\)/i)).toBeInTheDocument();
  });
});
