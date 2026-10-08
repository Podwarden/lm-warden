import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { useEventSource } from '@/lib/sse';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.all.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }
  close() { this.closed = true; }
}

function ticketCalls(fetchMock: ReturnType<typeof vi.fn>) {
  return fetchMock.mock.calls.filter(([u]) => u === '/api/auth/sse-ticket');
}

describe('useEventSource query option', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeES.all = [];
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  });
  afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

  it('builds path?query&ticket= and mints the ticket for the bare path', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}'));
    vi.stubGlobal('fetch', fetchMock);
    function Probe() {
      useEventSource('/api/admin/godmode/stream', { onMessage: () => {}, query: { token_ids: 'a,b' } });
      return null;
    }
    render(<Probe />);
    await vi.advanceTimersByTimeAsync(0);

    expect(FakeES.all).toHaveLength(1);
    expect(FakeES.all[0].url).toBe('/api/admin/godmode/stream?token_ids=a%2Cb&ticket=t1');
    const [[, init]] = ticketCalls(fetchMock);
    expect(JSON.parse(String((init as RequestInit).body))).toEqual({ path: '/api/admin/godmode/stream' });
  });

  it('keeps the old URL shape when no query is given', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}')));
    function Probe() {
      useEventSource('/api/models/abc/logs/stream', { onMessage: () => {} });
      return null;
    }
    render(<Probe />);
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeES.all[0].url).toBe('/api/models/abc/logs/stream?ticket=t1');
  });

  it('opens no EventSource when unmounted while the ticket is being minted', async () => {
    // Closing the god-mode dock during the mint must not leave a hidden stream
    // behind (spec D6). Before this fix the EventSource was created after the
    // await even though the effect had already been cleaned up.
    let release!: (r: Response) => void;
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((res) => { release = res; })));
    function Probe() {
      useEventSource('/api/admin/godmode/stream', { onMessage: () => {}, query: { token_ids: 'a' } });
      return null;
    }
    const { unmount } = render(<Probe />);
    await vi.advanceTimersByTimeAsync(0);
    unmount();
    release(new Response('{"ticket":"t1"}'));
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeES.all).toHaveLength(0);
  });

  it('reconnects with the new query when it changes', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}')));
    function Probe({ ids }: { ids: string }) {
      useEventSource('/api/admin/godmode/stream', { onMessage: () => {}, query: { token_ids: ids } });
      return null;
    }
    const { rerender } = render(<Probe ids="a" />);
    await vi.advanceTimersByTimeAsync(0);
    rerender(<Probe ids="a,b" />);
    await vi.advanceTimersByTimeAsync(0);

    expect(FakeES.all).toHaveLength(2);
    expect(FakeES.all[0].closed).toBe(true);
    expect(FakeES.all[1].url).toContain('token_ids=a%2Cb&ticket=');
  });
});
