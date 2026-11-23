// usePromptLensStream batches the stream (Task 14 review I1): every event of an
// animation frame goes through one reducer pass, so the dock renders once a
// frame however fast the deltas come.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';

import { usePromptLensStream, keyPostTicket } from '@/lib/prompt-lens/stream';
import { readKeyToken, storeKeyToken, clearKeyToken } from '@/lib/sessions/key-token';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  constructor(public url: string) {
    FakeES.all.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {}
}

const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'requestAnimationFrame', 'cancelAnimationFrame'] });
  FakeES.all = [];
  setAccessToken('jwt');
  setCsrfToken('csrf');
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  vi.stubGlobal('fetch', vi.fn(async () => json({ ticket: 't' })));
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  clearKeyToken();
});

describe('usePromptLensStream batching', () => {
  it('100 events in one frame cause one store update', async () => {
    let renders = 0;
    const { result } = renderHook(() => {
      renders += 1;
      return usePromptLensStream('admin', { enabled: true, tokenIds: null });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(50);
    });
    expect(result.current.state.conn).toBe(1);
    const before = renders;
    const stateBefore = result.current.state;
    act(() => {
      for (let i = 0; i < 100; i++) {
        FakeES.all[0].onmessage?.({
          data: JSON.stringify({ type: 'request_start', req_id: `r${i}`, ts: 1000 + i, token_id: 't', request_id: null, model: 'm', served_name: 'm', stream: true, prompt: '', session_id: null, turn: null, seq: i + 1 }),
        } as MessageEvent);
      }
    });
    // Nothing applied until the frame.
    expect(result.current.state).toBe(stateBefore);
    expect(renders).toBe(before);
    act(() => {
      vi.advanceTimersToNextFrame();
    });
    expect(result.current.state.order).toHaveLength(100);
    expect(renders).toBe(before + 1);
  });

  it('a hidden tab still flushes (rAF does not run in the background)', async () => {
    const { result } = renderHook(() => usePromptLensStream('admin', { enabled: true, tokenIds: null }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(50);
    });
    const vis = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    act(() => {
      FakeES.all[0].onmessage?.({
        data: JSON.stringify({ type: 'request_start', req_id: 'h1', ts: 1, token_id: 't', request_id: null, model: 'm', served_name: 'm', stream: true, prompt: '', session_id: null, turn: null }),
      } as MessageEvent);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    expect(result.current.state.order).toEqual(['h1']);
    vis.mockRestore();
  });
});

describe('usePromptLensStream flush edges', () => {
  const start = (i: number) =>
    ({
      data: JSON.stringify({ type: 'request_start', req_id: `q${i}`, ts: 1000 + i, token_id: 't', request_id: null, model: 'm', served_name: 'm', stream: true, prompt: '', session_id: null, turn: null, seq: i + 1 }),
    }) as MessageEvent;

  it('2000 queued events flush at once, before any frame', async () => {
    const { result } = renderHook(() => usePromptLensStream('admin', { enabled: true, tokenIds: null }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(50);
    });
    const before = result.current.state;
    act(() => {
      for (let i = 0; i < 1999; i++) FakeES.all[0].onmessage?.(start(i));
    });
    expect(result.current.state).toBe(before);
    act(() => {
      FakeES.all[0].onmessage?.(start(1999));
    });
    // The store keeps the newest requests only; the last one arrived.
    expect(result.current.state.order.at(-1)).toBe('q1999');
  });

  it('a frame pending when the tab goes hidden switches to the timer', async () => {
    const { result } = renderHook(() => usePromptLensStream('admin', { enabled: true, tokenIds: null }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(50);
    });
    // Visible: a frame is scheduled. Then the tab hides, so the frame never
    // runs (the stub stands for a browser that holds frames in a hidden tab).
    const raf = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(() => 4242);
    act(() => {
      FakeES.all[0].onmessage?.(start(0));
    });
    expect(raf).toHaveBeenCalled();
    const vis = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    // Only timers run from here on (a hidden tab runs no frames).
    await act(async () => {
      vi.advanceTimersByTime(300);
    });
    expect(result.current.state.order).toEqual(['q0']);
    vis.mockRestore();
    raf.mockRestore();
  });
});

describe('keyPostTicket', () => {
  it('clears the key sign-in on 401 or 403, like sessionsFetcher', async () => {
    for (const status of [401, 403]) {
      storeKeyToken('tok', 3600);
      vi.stubGlobal('fetch', vi.fn(async () => json({ detail: 'session ended' }, status)));
      const r = await keyPostTicket();
      expect(r.status).toBe(status);
      expect(readKeyToken()).toBeNull();
    }
  });

  it('keeps the sign-in on a 429', async () => {
    storeKeyToken('tok', 3600);
    vi.stubGlobal('fetch', vi.fn(async () => json({ detail: 'too many' }, 429)));
    expect((await keyPostTicket()).status).toBe(429);
    expect(readKeyToken()).toBe('tok');
  });
});
