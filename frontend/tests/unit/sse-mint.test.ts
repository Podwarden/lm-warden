// useEventSource's `mint` option (Task 14): a caller-supplied ticket mint
// (the key holder's POST /api/me/prompt-lens/ticket) instead of the default
// POST /api/auth/sse-ticket. The status classification applies to whatever
// `mint` returns: 4xx (except 429) terminal, 5xx and 429 retried.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';

import { useEventSource } from '@/lib/sse';

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
  close() {
    this.closed = true;
  }
}

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });

const flush = (ms = 0) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });

beforeEach(() => {
  vi.useFakeTimers();
  FakeES.all = [];
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  vi.stubGlobal('fetch', vi.fn(async () => json({ ticket: 'default' })));
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('useEventSource mint option', () => {
  it('uses the custom mint, not /api/auth/sse-ticket, and connects with its ticket', async () => {
    const mint = vi.fn(async () => json({ ticket: 'k1' }));
    const { result } = renderHook(() => useEventSource('/api/me/prompt-lens/stream', { onMessage: () => {}, mint }));
    await flush();
    expect(mint).toHaveBeenCalledWith('/api/me/prompt-lens/stream');
    expect(fetch).not.toHaveBeenCalled();
    expect(FakeES.all.map((e) => e.url)).toEqual(['/api/me/prompt-lens/stream?ticket=k1']);
    expect(result.current.status).toBe('connected');
  });

  it('a custom mint returning 404 ends in terminal-error with errorCode 404', async () => {
    const mint = vi.fn(async () => json({ detail: 'Prompt Lens is off' }, 404));
    const { result } = renderHook(() => useEventSource('/api/me/prompt-lens/stream', { onMessage: () => {}, mint }));
    await flush();
    expect(result.current).toMatchObject({ status: 'terminal-error', errorCode: 404 });
    await flush(60_000);
    expect(mint).toHaveBeenCalledTimes(1);
    expect(FakeES.all).toHaveLength(0);
  });

  it('a custom mint returning 503 retries, then connects with a fresh ticket', async () => {
    const mint = vi
      .fn()
      .mockResolvedValueOnce(json({}, 503))
      .mockResolvedValueOnce(json({ ticket: 'k2' }));
    const { result } = renderHook(() => useEventSource('/api/me/prompt-lens/stream', { onMessage: () => {}, mint }));
    await flush();
    expect(result.current).toMatchObject({ status: 'reconnecting', errorCode: 503, attempts: 1 });
    await flush(1_000);
    expect(mint).toHaveBeenCalledTimes(2);
    expect(FakeES.all.map((e) => e.url)).toEqual(['/api/me/prompt-lens/stream?ticket=k2']);
    expect(result.current.status).toBe('connected');
  });

  it('a 429 waits for Retry-After before minting again', async () => {
    const mint = vi
      .fn()
      .mockResolvedValueOnce(json({}, 429, { 'Retry-After': '7' }))
      .mockResolvedValue(json({ ticket: 'k3' }));
    const { result } = renderHook(() => useEventSource('/api/me/prompt-lens/stream', { onMessage: () => {}, mint }));
    await flush();
    expect(result.current).toMatchObject({ status: 'reconnecting', errorCode: 429 });
    await flush(6_000);
    expect(mint).toHaveBeenCalledTimes(1);
    await flush(1_500);
    expect(mint).toHaveBeenCalledTimes(2);
    expect(result.current.status).toBe('connected');
  });

  it('calls onOpen on every EventSource open, before that connection’s messages', async () => {
    const seen: string[] = [];
    const mint = vi.fn(async () => json({ ticket: 't' }));
    renderHook(() =>
      useEventSource<{ n: number }>('/s', {
        mint,
        onOpen: () => seen.push('open'),
        onMessage: (m) => seen.push(`msg${m.n}`),
      }),
    );
    await flush();
    FakeES.all[0].onmessage?.({ data: '{"n":1}' } as MessageEvent);
    FakeES.all[0].onerror?.();
    await flush(1_000);
    expect(FakeES.all).toHaveLength(2);
    FakeES.all[1].onmessage?.({ data: '{"n":2}' } as MessageEvent);
    expect(seen).toEqual(['open', 'msg1', 'open', 'msg2']);
  });
});
