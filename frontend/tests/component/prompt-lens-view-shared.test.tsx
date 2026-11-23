// The shared Prompt Lens view (Task 14; a full page since 2026-11-23, it was
// a dock): one component for the key holder's stream (/api/me/prompt-lens/stream,
// ticket from POST /api/me/prompt-lens/ticket) and the admin's
// (/api/admin/prompt-lens/stream?token_ids=), with the header, the filter
// line, the session-grouped request list and the request screen.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import { useState, type ReactNode } from 'react';
import { SWRConfig, useSWRConfig } from 'swr';

vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { PromptLensView, usePromptLensEnabled } from '@/components/prompt-lens/prompt-lens-view';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { resetServerClock, serverSkew } from '@/lib/prompt-lens/server-clock';
import { clearKeyToken, storeKeyToken } from '@/lib/sessions/key-token';

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

type Route = (url: string, init?: RequestInit) => Response | Promise<Response> | undefined;
let routes: Route[] = [];
let fetchMock: ReturnType<typeof vi.fn>;
const calls = () => fetchMock.mock.calls.map((c) => [String(c[0]), (c[1] as RequestInit | undefined)?.method ?? 'GET'] as const);

function defaultRoute(url: string, init?: RequestInit): Response | Promise<Response> {
  for (const r of routes) {
    const out = r(url, init);
    if (out) return out;
  }
  if (url === '/api/auth/sse-ticket') return json({ ticket: 'a1' });
  if (url === '/api/me/prompt-lens/ticket') return json({ ticket: 'k1' });
  if (url.startsWith('/api/me/prompt-lens/kept') || url.startsWith('/api/admin/prompt-lens/kept')) return json({ kept: [] });
  if (url === '/api/me/key') return json({ id: 'self', name: 'Dana', keep_captures: false });
  return new Promise<Response>(() => {});
}

const flush = (ms = 0) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
    vi.advanceTimersToNextFrame();
  });

const es = () => FakeES.all[FakeES.all.length - 1];
function emit(ev: unknown, src: FakeES = es()) {
  act(() => {
    src.onmessage?.({ data: JSON.stringify(ev) } as MessageEvent);
    // The stream applies its queue once a frame.
    vi.advanceTimersToNextFrame();
  });
}

let T = 1_800_000_000;
function start(reqId: string, o: Record<string, unknown> = {}) {
  T += 1;
  return {
    type: 'request_start',
    req_id: reqId,
    ts: T,
    token_id: 'tok-a',
    request_id: `rq-${reqId}`,
    model: 'qwen3.8-27b',
    served_name: 'qwen3.8-27b',
    stream: true,
    prompt: `prompt of ${reqId}`,
    session_id: 'sess0001aaaa',
    turn: 1,
    ...o,
  };
}
function end(reqId: string, o: Record<string, unknown> = {}) {
  T += 1;
  return {
    type: 'request_end',
    req_id: reqId,
    ts: T,
    token_id: 'tok-a',
    finish_reason: 'stop',
    prompt_tokens: 1200,
    completion_tokens: 80,
    reasoning_tokens: null,
    queued_s: 0,
    ttft_s: 0.2,
    duration_s: 3.4,
    decode_tps: 50,
    own_reused_tokens: null,
    ...o,
  };
}

interface HarnessProps {
  scope: 'key' | 'admin';
  /** The page is open (the view mounted); false: the reader is elsewhere. */
  initialOpen?: boolean;
  presetModel?: string | null;
  presetKeys?: string[] | null;
  keyNames?: Record<string, string>;
  select?: { requestId: string; nonce: number } | null;
}
function Harness({ scope, initialOpen = true, ...rest }: HarnessProps) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <div data-testid="page" style={{ height: 800 }}>
        {initialOpen && <PromptLensView scope={scope} pageKey={`test-${scope}`} {...rest} />}
      </div>
    </SWRConfig>
  );
}

const bar = () => screen.getByTestId('prompt-lens-header');
const dock = () => screen.getByTestId('prompt-lens-view');
const list = () => screen.getByTestId('prompt-lens-list');
const filterLine = () => screen.getByTestId('prompt-lens-filter-line');
const rowIds = () => within(list()).queryAllByTestId(/^prompt-lens-row-/).map((el) => el.getAttribute('data-req'));

beforeAll(() => {
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
});

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date', 'requestAnimationFrame', 'cancelAnimationFrame'] });
  vi.setSystemTime(new Date((T + 5) * 1000));
  FakeES.all = [];
  routes = [];
  resetServerClock();
  window.localStorage.clear();
  setAccessToken('jwt');
  setCsrfToken('csrf');
  storeKeyToken('key-signin', 3600);
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  fetchMock = vi.fn((u: RequestInfo | URL, init?: RequestInit) => Promise.resolve(defaultRoute(String(u), init)));
  vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  clearKeyToken();
});

// ---------------------------------------------------------------------------

describe('PromptLensView (shared)', () => {
  it('a page without the view mints no ticket and opens no EventSource', async () => {
    render(<Harness scope="key" initialOpen={false} />);
    await flush(30_000);
    expect(FakeES.all).toHaveLength(0);
    expect(calls().filter(([u]) => u.includes('ticket'))).toEqual([]);
  });

  it('key scope mints at /api/me/prompt-lens/ticket and connects to /api/me/prompt-lens/stream', async () => {
    render(<Harness scope="key" />);
    fireEvent.click(bar());
    await flush();
    const mint = fetchMock.mock.calls.find((c) => String(c[0]) === '/api/me/prompt-lens/ticket');
    expect(mint).toBeDefined();
    const init = mint![1] as RequestInit;
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer key-signin');
    expect(calls().some(([u]) => u === '/api/auth/sse-ticket')).toBe(false);
    expect(FakeES.all.map((e) => e.url)).toEqual(['/api/me/prompt-lens/stream?ticket=k1']);
    expect(within(bar()).getByText('Live')).toBeInTheDocument();
    expect(bar()).toHaveTextContent('Prompt Lens');
    expect(bar()).not.toHaveTextContent('all keys');

    // Leaving the page unmounts the stream; nothing reconnects in the background.
    cleanup();
    await flush(30_000);
    expect(FakeES.all[0].closed).toBe(true);
    expect(FakeES.all).toHaveLength(1);
  });

  it('admin Key filter reconnects with token_ids', async () => {
    render(<Harness scope="admin" keyNames={{ 'tok-a': 'Dana', 'tok-b': 'Bot-CI' }} />);
    fireEvent.click(bar());
    await flush();
    expect(bar()).toHaveTextContent(/Prompt Lens.*all keys/);
    expect(FakeES.all.map((e) => e.url)).toEqual(['/api/admin/prompt-lens/stream?ticket=a1']);
    expect(calls()).toContainEqual(['/api/admin/prompt-lens/kept', 'GET']);
    emit(start('k-b', { token_id: 'tok-b' }));
    expect(rowIds()).toEqual(['k-b']);

    fireEvent.click(within(filterLine()).getByRole('button', { name: /^Key/ }));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Dana' }));
    await flush();
    expect(FakeES.all).toHaveLength(2);
    expect(FakeES.all[0].closed).toBe(true);
    expect(FakeES.all[1].url).toBe('/api/admin/prompt-lens/stream?token_ids=tok-a&ticket=a1');
    // The old key set's requests do not stay.
    expect(rowIds()).toEqual([]);
    expect(calls()).toContainEqual(['/api/admin/prompt-lens/kept?token_ids=tok-a', 'GET']);
    // The ticket is still minted for the bare path.
    const bodies = fetchMock.mock.calls
      .filter((c) => String(c[0]) === '/api/auth/sse-ticket')
      .map((c) => JSON.parse(String((c[1] as RequestInit).body)));
    expect(bodies.every((b) => b.path === '/api/admin/prompt-lens/stream')).toBe(true);
  });

  it('admin key details preset: the stream opens with the preset keys', async () => {
    render(<Harness scope="admin" presetKeys={['old1', 'self']} keyNames={{ old1: 'laptop (old)', self: 'laptop' }} />);
    fireEvent.click(bar());
    await flush();
    expect(FakeES.all[0].url).toBe('/api/admin/prompt-lens/stream?token_ids=old1%2Cself&ticket=a1');
  });

  it('key scope has no Key and no Client IP filter, and no retention minutes', async () => {
    render(<Harness scope="key" />);
    fireEvent.click(bar());
    await flush();
    // A stray ring frame (the key stream sends none) must still not show minutes.
    emit({ type: 'ring', oldest_ts: T - 600, events: 10, chars: 10, chars_max: 100 });
    emit(start('r1'));
    const line = filterLine();
    expect(within(line).queryByRole('button', { name: /^Key/ })).toBeNull();
    expect(within(line).queryByRole('button', { name: /^Client IP/ })).toBeNull();
    expect(within(line).getByRole('button', { name: /^Session/ })).toBeInTheDocument();
    expect(within(line).getByRole('button', { name: /^Model/ })).toBeInTheDocument();
    expect(within(line).getByRole('button', { name: /^Finish any/ })).toBeInTheDocument();
    expect(within(line).getByRole('button', { name: /^Has/ })).toBeInTheDocument();
    expect(line).toHaveTextContent('1 of 1 requests · older requests leave server memory');
    expect(line).not.toHaveTextContent(/ring keeps/);
    expect(line).not.toHaveTextContent(/min\b/);
    const search = within(line).getByRole('searchbox');
    expect(search).toHaveAttribute('title', 'searches captured text; long prompts are trimmed in the middle');
    expect(search.getAttribute('placeholder')).toMatch(/captured/);
  });

  it('admin shows "ring keeps ≈ N min" from the ring frame', async () => {
    render(<Harness scope="admin" />);
    fireEvent.click(bar());
    await flush();
    emit({ type: 'ring', oldest_ts: T - 12 * 60, events: 10, chars: 10, chars_max: 100 });
    emit(start('r1'));
    expect(filterLine()).toHaveTextContent('1 of 1 requests · ring keeps ≈ 12 min');
    expect(within(filterLine()).getByRole('button', { name: /^Client IP/ })).toBeInTheDocument();
  });

  it('groups rows by session, live first', async () => {
    render(<Harness scope="key" />);
    fireEvent.click(bar());
    await flush();
    emit(start('a1', { session_id: 'aaaa1111xxxx', turn: 1 }));
    emit(end('a1'));
    emit(start('a2', { session_id: 'aaaa1111xxxx', turn: 2 }));
    emit(start('a3', { session_id: 'aaaa1111xxxx', turn: 3 }));
    emit(end('a3'));
    emit(start('b1', { session_id: 'bbbb2222xxxx', turn: 1 }));
    emit(end('b1'));
    // Group b has the newest row; inside group a the live a2 comes first.
    const groups = within(list()).getAllByTestId('prompt-lens-group');
    expect(groups.map((g) => g.getAttribute('data-session'))).toEqual(['bbbb2222xxxx', 'aaaa1111xxxx']);
    expect(rowIds()).toEqual(['b1', 'a2', 'a3', 'a1']);
    const head = within(groups[1]).getByTestId('prompt-lens-group-head');
    expect(head).toHaveTextContent('session aaaa1111');
    expect(head).toHaveTextContent('3 turns here');
    expect(list()).toHaveTextContent(
      'Older requests left server memory. Their metrics stay in the Requests table on the Stats page.',
    );
  });

  it('select opens the request and sets the Session filter', async () => {
    const { rerender } = render(<Harness scope="admin" initialOpen />);
    await flush();
    emit(start('x1', { session_id: 'sessAAAA1111', turn: 4 }));
    emit(start('x2', { session_id: 'sessBBBB2222', turn: 7, request_id: 'want' }));
    rerender(<Harness scope="admin" initialOpen select={{ requestId: 'want', nonce: 1 }} />);
    await flush();
    expect(within(screen.getByTestId('prompt-lens-screen')).getByTestId('rs-title')).toHaveTextContent('session sessBBBB · turn 7');
    expect(within(filterLine()).getByRole('button', { name: /^Session/ })).toHaveTextContent('sessBBBB');
    expect(rowIds()).toEqual(['x2']);
  });

  it('select before the replay waits for the request to arrive', async () => {
    render(<Harness scope="key" initialOpen select={{ requestId: 'rq-late', nonce: 1 }} />);
    await flush();
    emit(start('late', { request_id: 'rq-late', session_id: 'sessLATE0000', turn: 2 }));
    await flush();
    expect(within(screen.getByTestId('prompt-lens-screen')).getByTestId('rs-title')).toHaveTextContent('session sessLATE · turn 2');
  });

  it('select of an evicted request says it left server memory', async () => {
    render(<Harness scope="key" initialOpen select={{ requestId: 'rq-gone', nonce: 1 }} />);
    await flush();
    emit(start('other1'));
    await flush(5_000);
    expect(screen.getByTestId('prompt-lens-screen')).toHaveTextContent(
      'This request left server memory. Its metrics stay in the Requests table on the Stats page.',
    );
  });

  it('Follow newest selects a new live request; Pause freezes the order', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    const follow = screen.getByRole('button', { name: 'Follow newest' });
    expect(follow).toHaveAttribute('aria-pressed', 'true');
    const title = () => within(screen.getByTestId('prompt-lens-screen')).getByTestId('rs-title');
    emit(start('f1', { session_id: 'sessF0000001', turn: 1 }));
    expect(title()).toHaveTextContent('turn 1');
    emit(start('f2', { session_id: 'sessF0000001', turn: 2 }));
    expect(title()).toHaveTextContent('turn 2');
    expect(rowIds()).toEqual(['f2', 'f1']);

    fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
    expect(screen.getByRole('button', { name: 'Pause' })).toHaveAttribute('aria-pressed', 'true');
    emit(start('f3', { session_id: 'sessF0000001', turn: 3 }));
    emit(end('f1'));
    expect(rowIds()).toEqual(['f2', 'f1']);
    expect(title()).toHaveTextContent('turn 2');
    const pill = screen.getByRole('button', { name: '1 new' });
    fireEvent.click(pill);
    expect(rowIds()).toEqual(['f3', 'f2', 'f1']);

    // A row picked by hand stops following.
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
    fireEvent.click(screen.getByTestId('prompt-lens-row-f1'));
    expect(screen.getByRole('button', { name: 'Follow newest' })).toHaveAttribute('aria-pressed', 'false');
    emit(start('f4', { session_id: 'sessF0000001', turn: 4 }));
    expect(title()).toHaveTextContent('turn 1');
  });

  it('the view fills its page: header, one-row filter line, list and screen side by side', async () => {
    render(<Harness scope="key" />);
    await flush();
    expect(dock()).toHaveClass('flex', 'h-full', 'flex-col');
    expect(bar()).toHaveClass('h-10');
    expect(screen.getByTestId('prompt-lens-filter-line')).toHaveClass('h-[38px]', 'overflow-x-auto', 'whitespace-nowrap');
    expect(screen.getByTestId('prompt-lens-list')).toBeInTheDocument();
    expect(screen.getByTestId('prompt-lens-screen')).toBeInTheDocument();
    expect(screen.queryByTestId('prompt-lens-dock-spacer')).toBeNull();
  });
});

describe('stream wiring', () => {
  it('dispatches connected on every open: a key reconnect replays without duplicating', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    const s1 = start('c1', { seq: 1 });
    emit(s1);
    emit({ ...start('c0'), seq: 2 });
    emit({ ...end('c0'), seq: 3 });
    emit({ type: 'delta', req_id: 'c1', ts: T + 1, token_id: 'tok-a', channel: 'content', text: 'hello', seq: 4 });
    es().onerror?.();
    await flush(1_000);
    expect(FakeES.all).toHaveLength(2);
    // The new connection renumbers from 1 (c1's window only): without
    // `connected` the reducer would take seq 3 as already seen.
    emit({ ...s1, seq: 1 });
    emit({ type: 'delta', req_id: 'c1', ts: T + 1, token_id: 'tok-a', channel: 'content', text: 'hello', seq: 2 });
    emit({ type: 'delta', req_id: 'c1', ts: T + 2, token_id: 'tok-a', channel: 'content', text: ' world', seq: 3 });
    fireEvent.click(screen.getByTestId('prompt-lens-row-c1'));
    expect(rowIds()).toEqual(['c1', 'c0']);
    expect(screen.getByTestId('prompt-lens-screen')).toHaveTextContent('hello world');
    expect(screen.getByTestId('prompt-lens-screen')).not.toHaveTextContent('hellohello');
  });

  it('notes the server time of every event', async () => {
    vi.setSystemTime(new Date((T + 100) * 1000));
    render(<Harness scope="admin" initialOpen />);
    await flush();
    emit({ ...start('n1'), ts: Date.now() / 1000 - 30 });
    expect(serverSkew()).toBeCloseTo(30, 0);
  });

  it('a lagged frame says some events were dropped', async () => {
    render(<Harness scope="admin" initialOpen />);
    await flush();
    emit(start('l1'));
    emit({ type: 'lagged' });
    expect(screen.getByTestId('prompt-lens-view')).toHaveTextContent(/fell behind/);
  });

  it('a key mint 429 backs off and says why', async () => {
    let n = 0;
    routes.push((u) => {
      if (u !== '/api/me/prompt-lens/ticket') return undefined;
      n += 1;
      return n === 1 ? json({ detail: 'too many' }, 429, { 'Retry-After': '20' }) : json({ ticket: 'k9' });
    });
    render(<Harness scope="key" initialOpen />);
    await flush();
    expect(screen.getByTestId('prompt-lens-status')).toHaveTextContent(/Too many live streams or tickets for this key/);
    await flush(15_000);
    expect(FakeES.all).toHaveLength(0);
    await flush(6_000);
    expect(FakeES.all.map((e) => e.url)).toEqual(['/api/me/prompt-lens/stream?ticket=k9']);
  });

  it('a revoked key or ended sign-in shows a plain message and stops', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    routes.push((u) => (u === '/api/me/prompt-lens/ticket' ? json({ detail: 'session ended' }, 401) : undefined));
    es().onerror?.();
    await flush(1_000);
    expect(screen.getByTestId('prompt-lens-status')).toHaveTextContent(
      'The live stream ended: your sign-in ended or expired, or the key was revoked, rotated or paused. Sign in again to watch.',
    );
    await flush(60_000);
    expect(FakeES.all).toHaveLength(1);
  });
});

describe('kept captures', () => {
  it('lists the key’s kept captures with a kept mark; opening one fetches it and its prompt', async () => {
    routes.push((u) => {
      if (u === '/api/me/prompt-lens/kept')
        return json({
          kept: [
            {
              req_id: 'kept1',
              request_id: 'rq-k1',
              model: 'qwen3.8-27b',
              stream: true,
              session_id: 'sessKEPT0001',
              turn: 3,
              started_ts: T - 4000,
              ended_ts: T - 3990,
              finish_reason: 'stop',
              prompt_tokens: 1500,
              completion_tokens: 42,
              kept: true,
            },
          ],
        });
      if (u === '/api/me/prompt-lens/kept/kept1')
        return json({
          req_id: 'kept1',
          kept: true,
          events: [
            { ...start('kept1', { session_id: 'sessKEPT0001', turn: 3 }), ts: T - 4000 },
            { ...end('kept1', { completion_tokens: 42 }), ts: T - 3990 },
          ],
          sections: { thinking: { chars: 0, dropped_chars: 0 }, reply: { chars: 11, dropped_chars: 0 }, tool_calls: { count: 0 }, prompt: { total_chars: 20 } },
        });
      if (u === '/api/me/prompt-lens/requests/kept1/section/reply')
        return json({ kind: 'reply', seq: null, live: false, dropped_chars: 0, offset: 0, text: 'kept answer', kept: true });
      if (u === '/api/me/prompt-lens/requests/kept1/section/prompt')
        return json({
          kind: 'prompt',
          seq: null,
          live: false,
          dropped_chars: 0,
          offset: 0,
          messages: [{ i: 0, role: 'user', chars: 12, text: 'kept question', elided: false }],
          older: { older: { from: 0, roles: '', chars: [] }, collapsed: 0, omitted: 0, total_chars: 12 },
          kept: true,
        });
      if (u === '/api/me/key') return json({ id: 'self', name: 'Dana', keep_captures: true });
      return undefined;
    });
    render(<Harness scope="key" initialOpen />);
    await flush();
    const row = screen.getByTestId('prompt-lens-row-kept1');
    expect(within(row).getByText('kept')).toBeInTheDocument();
    // The summary's token counts show before the capture is opened (G-R15).
    expect(row).toHaveTextContent('1.5k in · 42 out');
    fireEvent.click(row);
    await flush();
    await flush();
    expect(calls()).toContainEqual(['/api/me/prompt-lens/kept/kept1', 'GET']);
    expect(calls()).toContainEqual(['/api/me/prompt-lens/requests/kept1/section/prompt', 'GET']);
    expect(calls()).toContainEqual(['/api/me/prompt-lens/requests/kept1/section/reply', 'GET']);
    expect(calls().some(([u]) => u.endsWith('/section/thinking'))).toBe(false);
    const scr = screen.getByTestId('prompt-lens-screen');
    expect(scr).toHaveTextContent('kept answer');
    expect(within(scr).getByTestId('rs-chip')).toHaveTextContent('stop');
    // keep_captures from GET /api/me/key: the privacy line says they are on disk.
    expect(scr).toHaveTextContent(/kept on disk/);
    // The prompt (kept starts carry no prompt_parts) arrives from section/prompt.
    const prompt = within(scr).getByTestId('section-prompt');
    fireEvent.click(within(prompt).getByRole('button', { name: /^Show / }));
    expect(within(scr).getByTestId('section-prompt')).toHaveTextContent('kept question');
  });

  it('keep_captures from GET /api/me/key feeds the request screen’s privacy line', async () => {
    routes.push((u) => (u === '/api/me/key' ? json({ id: 'self', name: 'Dana', keep_captures: true }) : undefined));
    render(<Harness scope="key" initialOpen />);
    await flush();
    emit(start('live1'));
    await flush();
    expect(screen.getByTestId('prompt-lens-screen')).toHaveTextContent(/the last 50 captures are kept on disk for this key/);
  });

  it('without keep_captures the privacy line says nothing is on disk', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    emit(start('live2'));
    await flush();
    expect(screen.getByTestId('prompt-lens-screen')).toHaveTextContent(/not on disk/);
  });

  it('admin lists kept captures for the Key filter', async () => {
    render(<Harness scope="admin" initialOpen presetKeys={['tok-a']} />);
    await flush();
    expect(calls()).toContainEqual(['/api/admin/prompt-lens/kept?token_ids=tok-a', 'GET']);
    expect(calls().some(([u]) => u === '/api/me/key')).toBe(false);
  });
});

describe('scope isolation in the rendered text', () => {
  it('the key dock shows nothing about other keys and never says cobs2', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    emit({ ...start('i1'), token_label: 'SECRET-OTHER-LABEL', client_ip: '10.9.9.9' });
    emit({
      type: 'kvsig',
      req_id: 'i1',
      ts: T + 1,
      token_id: 'tok-a',
      session_id: 'sess0001aaaa',
      turn: 1,
      available: true,
      profile: 'cobs2',
      conv8: 'abcd1234',
      parts: [
        { p: 'sys', from: -1, to: -1, chars: 100, roles: {}, exact8: 'aaaaaaaa', loose8: 'bbbbbbbb', state: 'other' },
        { p: 'nodes', from: 0, to: 0, chars: 100, roles: { user: 1 }, exact8: 'cccccccc', loose8: 'dddddddd', state: 'exact' },
      ],
    });
    const text = dock().textContent ?? '';
    expect(text).not.toMatch(/cobs2/i);
    expect(text).not.toContain('SECRET-OTHER-LABEL');
    expect(text).not.toContain('10.9.9.9');
    expect(text).not.toMatch(/another key/i);
    const mini = within(screen.getByTestId('prompt-lens-row-i1')).getByTestId('prompt-lens-mini');
    expect(mini.querySelector('[data-share="other"]')).toBeNull();
    expect(mini.querySelector('[data-share="exact"]')).not.toBeNull();
  });

  it('the admin mini bar draws the other-key share', async () => {
    render(<Harness scope="admin" initialOpen keyNames={{ 'tok-a': 'Dana' }} />);
    await flush();
    emit({ ...start('i2'), client_ip: '10.1.1.54' });
    emit({
      type: 'kvsig',
      req_id: 'i2',
      ts: T + 1,
      token_id: 'tok-a',
      session_id: 'sess0001aaaa',
      turn: 1,
      available: true,
      parts: [{ p: 'sys', from: -1, to: -1, chars: 100, roles: {}, exact8: 'aaaaaaaa', loose8: 'bbbbbbbb', state: 'other' }],
    });
    const mini = within(screen.getByTestId('prompt-lens-row-i2')).getByTestId('prompt-lens-mini');
    expect(mini.querySelector('[data-share="other"]')).not.toBeNull();
    const head = within(list()).getByTestId('prompt-lens-group-head');
    expect(head).toHaveTextContent('Dana');
    expect(head).toHaveTextContent('10.1.1.54');
  });
});

describe('review round 1', () => {
  it('M1: an admin Key change discards a kept capture still being fetched', async () => {
    let release: (r: Response) => void = () => {};
    routes.push((u) => {
      if (u === '/api/admin/prompt-lens/kept')
        return json({
          kept: [
            { req_id: 'old1', request_id: 'rq-old1', token_id: 'tok-a', model: 'm', stream: true, session_id: 'sessOLD00001', turn: 1, started_ts: T - 50, ended_ts: T - 40, finish_reason: 'stop', kept: true },
          ],
        });
      if (u === '/api/admin/prompt-lens/kept/old1') return new Promise<Response>((res) => (release = res));
      return undefined;
    });
    render(<Harness scope="admin" initialOpen keyNames={{ 'tok-a': 'Dana', 'tok-b': 'Bot-CI' }} />);
    await flush();
    fireEvent.click(screen.getByTestId('prompt-lens-row-old1'));
    await flush();
    fireEvent.click(within(filterLine()).getByRole('button', { name: /^Key/ }));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Bot-CI' }));
    await flush();
    expect(rowIds()).toEqual([]);
    release(
      json({
        req_id: 'old1',
        token_id: 'tok-a',
        kept: true,
        events: [{ ...start('old1', { session_id: 'sessOLD00001' }), ts: T - 50 }, { ...end('old1'), ts: T - 40 }],
        sections: {},
      }),
    );
    await flush();
    await flush();
    expect(rowIds()).toEqual([]);
    expect(calls().some(([u]) => u.includes('/requests/old1/section/'))).toBe(false);
  });

  it('M3: opening a request that other filters hide clears them, with a visible note', async () => {
    window.localStorage.setItem('vw.prompt-lens.filters.test-admin', JSON.stringify({ phase: 'live', models: ['other-model'] }));
    const { rerender } = render(<Harness scope="admin" initialOpen />);
    await flush();
    emit(start('h1', { request_id: 'want-h1', session_id: 'sessHID00001' }));
    emit(end('h1'));
    expect(rowIds()).toEqual([]);
    rerender(<Harness scope="admin" initialOpen select={{ requestId: 'want-h1', nonce: 1 }} />);
    await flush();
    expect(rowIds()).toEqual(['h1']);
    expect(screen.getByTestId('prompt-lens-filter-note')).toHaveTextContent(
      'Other filters hid this request, so they were cleared.',
    );
    expect(within(filterLine()).getByRole('button', { name: 'All' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('M3: a request the filters already show keeps them, with no note', async () => {
    const { rerender } = render(<Harness scope="admin" initialOpen presetModel="qwen3.8-27b" />);
    await flush();
    emit(start('v1', { request_id: 'want-v1', session_id: 'sessVIS00001' }));
    rerender(<Harness scope="admin" initialOpen presetModel="qwen3.8-27b" select={{ requestId: 'want-v1', nonce: 1 }} />);
    await flush();
    expect(rowIds()).toEqual(['v1']);
    expect(screen.queryByTestId('prompt-lens-filter-note')).toBeNull();
    expect(within(filterLine()).getByRole('button', { name: /^Model/ })).toHaveTextContent('qwen3.8-27b');
  });

  it('M4: a stream that never connects says so and offers a retry', async () => {
    routes.push((u) => (u === '/api/me/prompt-lens/ticket' ? new Promise<Response>(() => {}) : undefined));
    render(<Harness scope="key" initialOpen select={{ requestId: 'rq-x', nonce: 1 }} />);
    await flush(14_000);
    expect(screen.queryByText(/can't connect/)).toBeNull();
    await flush(2_000);
    expect(screen.getByTestId('prompt-lens-status')).toHaveTextContent("Prompt Lens can't connect");
    expect(screen.getByTestId('prompt-lens-screen')).not.toHaveTextContent('Looking for the request');
    const mints = () => calls().filter(([u]) => u === '/api/me/prompt-lens/ticket').length;
    expect(mints()).toBe(1);
    routes.length = 0;
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await flush();
    expect(mints()).toBe(2);
    expect(FakeES.all).toHaveLength(1);
    expect(screen.queryByTestId('prompt-lens-status')).toBeNull();
  });

  it('M4: a mint that hangs while reconnecting also says it can\'t connect and offers a retry', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    expect(es()).toBeDefined();
    // From now on every mint hangs; the stream drops.
    routes.push((u) => (u === '/api/me/prompt-lens/ticket' ? new Promise<Response>(() => {}) : undefined));
    act(() => {
      es().onerror?.();
    });
    await flush(1_000);
    expect(screen.getByTestId('prompt-lens-status')).toHaveTextContent('reconnecting');
    // 15 s from the last attempt (the 1 s backoff counts).
    await flush(13_000);
    expect(screen.queryByText(/can't connect/)).toBeNull();
    await flush(2_000);
    expect(screen.getByTestId('prompt-lens-status')).toHaveTextContent("Prompt Lens can't connect");
    routes.length = 0;
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await flush();
    expect(screen.queryByTestId('prompt-lens-status')).toBeNull();
  });

  it('M4: a terminal error also offers a retry, and a pending select stops waiting', async () => {
    routes.push((u) => (u === '/api/me/prompt-lens/ticket' ? json({ detail: 'off' }, 404) : undefined));
    render(<Harness scope="key" initialOpen select={{ requestId: 'rq-x', nonce: 1 }} />);
    await flush();
    expect(screen.getByTestId('prompt-lens-status')).toHaveTextContent('Prompt Lens is off on this server.');
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.getByTestId('prompt-lens-screen')).toHaveTextContent("Can't open the request: Prompt Lens is not connected.");
  });

  it('M5: filter menus take focus, move with arrows and close on Escape and Tab', async () => {
    render(<Harness scope="admin" initialOpen keyNames={{ 'tok-a': 'Dana', 'tok-b': 'Bot-CI' }} />);
    await flush();
    const btn = within(filterLine()).getByRole('button', { name: /^Key/ });
    fireEvent.click(btn);
    const items = screen.getAllByRole('menuitemcheckbox');
    expect(document.activeElement).toBe(items[0]);
    fireEvent.keyDown(items[0], { key: 'ArrowDown' });
    expect(document.activeElement).toBe(items[1]);
    fireEvent.keyDown(items[1], { key: 'End' });
    expect(document.activeElement).toBe(items[items.length - 1]);
    fireEvent.keyDown(document.activeElement!, { key: 'ArrowDown' });
    expect(document.activeElement).toBe(items[0]);
    fireEvent.keyDown(items[0], { key: 'ArrowUp' });
    expect(document.activeElement).toBe(items[items.length - 1]);
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(btn);
    fireEvent.click(btn);
    fireEvent.keyDown(screen.getAllByRole('menuitemcheckbox')[0], { key: 'Tab' });
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('M5: an open menu follows its button when the filter line scrolls', async () => {
    render(<Harness scope="admin" initialOpen />);
    await flush();
    const btn = within(filterLine()).getByRole('button', { name: /^Session/ });
    let left = 100;
    btn.getBoundingClientRect = () => ({ left, right: left + 80, top: 10, bottom: 36, width: 80, height: 26, x: left, y: 10, toJSON: () => ({}) }) as DOMRect;
    fireEvent.click(btn);
    expect(screen.getByRole('menu')).toHaveStyle({ left: '100px', top: '40px' });
    left = 40;
    fireEvent.scroll(filterLine());
    expect(screen.getByRole('menu')).toHaveStyle({ left: '40px' });
  });

  it('M5: the bar names Live and the counts; the mini bar has a name', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    emit(start('a1'));
    emit({
      type: 'kvsig', req_id: 'a1', ts: T + 1, token_id: 'tok-a', session_id: 'sess0001aaaa', turn: 1, available: true,
      parts: [
        { p: 'sys', from: -1, to: -1, chars: 300, roles: {}, exact8: 'aaaaaaaa', loose8: 'bbbbbbbb', state: 'exact' },
        { p: 'nodes', from: 0, to: 0, chars: 100, roles: { user: 1 }, exact8: 'cccccccc', loose8: 'dddddddd', state: 'other' },
      ],
    });
    expect(bar()).toHaveAccessibleName('Prompt Lens, Live, 1 requests, 1 running');
    const mini = within(screen.getByTestId('prompt-lens-row-a1')).getByTestId('prompt-lens-mini');
    expect(mini).toHaveAttribute('role', 'img');
    expect(mini).toHaveAccessibleName('kvsig: 75% exact, 0% loose');
  });

  it('M5: the admin mini bar names the other-key share', async () => {
    render(<Harness scope="admin" initialOpen />);
    await flush();
    emit(start('a2'));
    emit({
      type: 'kvsig', req_id: 'a2', ts: T + 1, token_id: 'tok-a', session_id: 'sess0001aaaa', turn: 1, available: true,
      parts: [{ p: 'sys', from: -1, to: -1, chars: 100, roles: {}, exact8: 'aaaaaaaa', loose8: 'bbbbbbbb', state: 'other' }],
    });
    expect(within(screen.getByTestId('prompt-lens-row-a2')).getByTestId('prompt-lens-mini')).toHaveAccessibleName(
      "kvsig: 0% exact, 0% loose, 100% another key's",
    );
  });
});

// ---------------------------------------------------------------------------
// Moved from the retired tokens/detail/prompt-lens-dock.test.tsx (Task 15): the
// cases that still hold for the shared dock.

// Prompt Lens shows (the menu entry, the eye) only when the status endpoint says enabled; loading or any
// error hides it, including an error on a later refetch, which SWR reports
// beside the last good `data` (#251).
describe('usePromptLensEnabled (moved)', () => {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{children}</SWRConfig>
  );
  function useProbe() {
    return { enabled: usePromptLensEnabled(), mutate: useSWRConfig().mutate };
  }
  beforeEach(() => {
    vi.useRealTimers();
  });

  it('is false while loading', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})));
    const { result } = renderHook(useProbe, { wrapper });
    expect(result.current.enabled).toBe(false);
  });

  it('is false when the status fetch fails', async () => {
    const f = vi.fn().mockResolvedValue(json({ detail: 'boom' }, 500));
    vi.stubGlobal('fetch', f);
    const { result } = renderHook(useProbe, { wrapper });
    await waitFor(() => expect(f).toHaveBeenCalled());
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(result.current.enabled).toBe(false);
  });

  it('is true for {enabled: true} and false for {enabled: false}', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ enabled: true })));
    const on = renderHook(useProbe, { wrapper });
    await waitFor(() => expect(on.result.current.enabled).toBe(true));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ enabled: false })));
    const off = renderHook(useProbe, { wrapper });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(off.result.current.enabled).toBe(false);
  });

  it('turns false when a refetch fails, although SWR still holds the last {enabled: true}', async () => {
    const f = vi.fn().mockResolvedValue(json({ enabled: true }));
    vi.stubGlobal('fetch', f);
    const { result } = renderHook(useProbe, { wrapper });
    await waitFor(() => expect(result.current.enabled).toBe(true));
    f.mockResolvedValue(json({ detail: 'boom' }, 500));
    await act(async () => {
      await result.current.mutate('/api/admin/prompt-lens/status').catch(() => {});
    });
    expect(result.current.enabled).toBe(false);
  });
});
