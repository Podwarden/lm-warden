// A dock that connects while a request is already running (Task 17 concern 1):
// the server replays that request's start, marked `restored`, before the
// window's deltas. The dock must show the prompt, the live phase and a live
// reply box (Jump to latest), not a "non-streamed" row with no prompt; and
// the /stats Model preset (a model row id) must keep it visible, its chip
// naming the served model. Harness as in prompt-lens-dock-shared.test.tsx, with
// the fake section-box layout of prompt-lens-section-box.test.tsx.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import { useState, type ReactNode } from 'react';
import { SWRConfig, useSWRConfig } from 'swr';

vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { PromptLensView } from '@/components/prompt-lens/prompt-lens-view';
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
  /** Kept from the dock days: the view is always open on its page. */
  initialOpen?: boolean;
  presetModel?: string | null;
  presetKeys?: string[] | null;
  keyNames?: Record<string, string>;
  select?: { requestId: string; nonce: number } | null;
}
function Harness({ scope, initialOpen: _open, ...rest }: HarnessProps) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <div style={{ height: 800 }}>
        <PromptLensView scope={scope} pageKey={`test-${scope}`} {...rest} />
      </div>
    </SWRConfig>
  );
}

const bar = () => screen.getByRole('button', { name: /^Prompt Lens/ });
const dock = () => screen.getByTestId('prompt-lens-dock');
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


// jsdom has no layout: a section box is 200 px high, 20 px per 100 chars.
const CLIENT = 200;
const tops = new WeakMap<Element, number>();
const isBox = (el: Element) => (el.getAttribute('data-testid') ?? '').startsWith('section-box-');
const contentHeight = (el: Element) => Math.ceil((el.textContent ?? '').length / 100) * 20;
const clientOf = (el: Element) => Math.min(CLIENT, contentHeight(el));
const maxTop = (el: Element) => Math.max(clientOf(el), contentHeight(el)) - clientOf(el);
const PROPS = ['scrollTop', 'scrollHeight', 'clientHeight'] as const;
const savedProps: Record<string, PropertyDescriptor | undefined> = {};
beforeEach(() => {
  for (const p of PROPS) savedProps[p] = Object.getOwnPropertyDescriptor(Element.prototype, p);
  Object.defineProperty(Element.prototype, 'scrollHeight', {
    configurable: true,
    get(this: Element) {
      return isBox(this) ? Math.max(clientOf(this), contentHeight(this)) : 0;
    },
  });
  Object.defineProperty(Element.prototype, 'clientHeight', {
    configurable: true,
    get(this: Element) {
      return isBox(this) ? clientOf(this) : 0;
    },
  });
  Object.defineProperty(Element.prototype, 'scrollTop', {
    configurable: true,
    get(this: Element) {
      return tops.get(this) ?? 0;
    },
    set(this: Element, v: number) {
      tops.set(this, Math.max(0, Math.min(v, isBox(this) ? maxTop(this) : 0)));
    },
  });
});
afterEach(() => {
  for (const p of PROPS) {
    const d = savedProps[p];
    if (d) Object.defineProperty(Element.prototype, p, d);
    else delete (Element.prototype as unknown as Record<string, unknown>)[p];
  }
});

function delta(reqId: string, text: string, seq?: number) {
  T += 1;
  return { type: 'delta', req_id: reqId, ts: T, token_id: 'tok-a', channel: 'content', text, ...(seq ? { seq } : {}) };
}

describe('a dock that connects mid-request', () => {
  it('key holder: the restored start gives prompt, phase and a live reply box', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    // The window holds only the latest deltas; the start comes back restored.
    emit({ ...start('mid', { model: 'm2', served_name: 'qwen' }), restored: true, seq: 1 });
    emit(delta('mid', 'x'.repeat(3000), 2));
    fireEvent.click(screen.getByTestId('prompt-lens-row-mid'));
    const scr = screen.getByTestId('prompt-lens-screen');
    // The prompt ("prompt of mid", 13 chars) is there, collapsed by role.
    expect(scr).toHaveTextContent('Prompt by role13 chars');
    expect(scr).not.toHaveTextContent(/non-streamed/i);
    expect(within(scr).getByTestId('rs-chip')).toHaveTextContent('answering');
    expect(within(list()).queryByText('non-streamed')).toBeNull();
    // Live: scrolled up, new text offers Jump to latest.
    const box = within(scr).getByTestId('section-box-reply');
    box.scrollTop = 0;
    fireEvent.scroll(box);
    emit(delta('mid', 'y'.repeat(500), 3));
    expect(within(scr).getByRole('button', { name: /Jump to latest/ })).toBeVisible();
  });

  it('without the restored start the same request is a prompt-less non-streamed row (the defect)', async () => {
    render(<Harness scope="key" initialOpen />);
    await flush();
    emit(delta('mid', 'x'.repeat(300), 1));
    fireEvent.click(screen.getByTestId('prompt-lens-row-mid'));
    const scr = screen.getByTestId('prompt-lens-screen');
    expect(scr).not.toHaveTextContent('Prompt by role13 chars');
    expect(scr).toHaveTextContent(/non-streamed/i);
  });

  it('admin /stats: the Model preset (row id) keeps a restored request and the chip names the served model', async () => {
    render(<Harness scope="admin" initialOpen presetModel="m2" />);
    await flush();
    emit(delta('mid', 'abc'));
    emit({ ...start('mid', { model: 'm2', served_name: 'qwen' }), restored: true });
    expect(rowIds()).toEqual(['mid']);
    const chip = within(filterLine()).getByRole('button', { name: /^Model/ });
    expect(chip).toHaveTextContent('qwen');
    expect(chip).not.toHaveTextContent('m2');
  });
});
