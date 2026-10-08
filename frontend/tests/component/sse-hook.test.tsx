import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { useEventSource } from '@/lib/sse';
import { setAccessToken } from '@/lib/auth-fetch';

class FakeES {
  static last: FakeES;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) { FakeES.last = this; setTimeout(() => this.onopen?.(), 0); }
  close() { this.closed = true; }
}

describe('useEventSource', () => {
  // A signed-in session: with no JWT in memory, authFetch refreshes before a
  // ticket mint (#251), which would add /api/auth/refresh calls to the count.
  beforeEach(() => { vi.useFakeTimers(); setAccessToken('jwt'); });
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); setAccessToken(null); });

  it('mints a fresh ticket per reconnect', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}'));
    vi.stubGlobal('fetch', fetchMock);

    function Probe() {
      useEventSource('/api/models/abc/logs/stream', { onMessage: () => {} });
      return null;
    }
    const { unmount } = render(<Probe />);

    // Flush initial connect() — authFetch → fetch → setTimeout(onopen, 0)
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeES.last).toBeDefined();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Trigger reconnect; backoff is 2s after the onopen reset
    act(() => FakeES.last.onerror?.());
    await vi.advanceTimersByTimeAsync(2000);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    unmount();
  });
});
