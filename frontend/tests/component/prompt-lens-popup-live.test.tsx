// The section pop-up opened from Prompt Lens keeps following the live stream
// (owner bug 2026-11-23: "if i expand this panel, streaming stops - no new
// text appears"). Root cause: a store that joined mid-request counts from a
// later start and is usually already past the server's copy when the whole
// section arrives; the pop-up could not align the two and pinned the store's
// end to the fetched end on every delta, so nothing was appended. Harness as
// in prompt-lens-dock-midrequest.test.tsx.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { SWRConfig } from 'swr';

vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { PromptLensView } from '@/components/prompt-lens/prompt-lens-view';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { resetServerClock } from '@/lib/prompt-lens/server-clock';
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



function tdelta(reqId: string, channel: 'reasoning' | 'content', text: string) {
  T += 1;
  return { type: 'delta', req_id: reqId, ts: T, token_id: 'tok-a', channel, text };
}

const popupBody = () => screen.getByTestId('popup-body');

describe('the section pop-up keeps streaming', () => {
  for (const [kind, channel, title] of [
    ['thinking', 'reasoning', 'Thinking'],
    ['reply', 'content', 'Reply'],
  ] as const) {
    it(`${kind}: text that arrives after the pop-up opened appears in it`, async () => {
      routes.push((url) =>
        url.startsWith(`/api/admin/prompt-lens/requests/live1/section/${kind}`)
          ? json({ kind, seq: 3, live: true, dropped_chars: 0, offset: 0, text: 'alpha beta ', dropped_calls: 0, kept: false })
          : undefined,
      );
      render(<Harness scope="admin" initialOpen />);
      await flush();
      emit(start('live1'));
      emit(tdelta('live1', channel, 'alpha beta '));
      fireEvent.click(screen.getByTestId('prompt-lens-row-live1'));
      fireEvent.click(screen.getByRole('button', { name: `Open ${title} in a large view` }));
      await flush();
      await flush();
      expect(popupBody()).toHaveTextContent('alpha beta');
      emit(tdelta('live1', channel, 'gamma delta '));
      await flush();
      expect(popupBody().textContent).toContain('alpha beta gamma delta ');
      emit(tdelta('live1', channel, 'epsilon '));
      await flush();
      expect(popupBody().textContent).toContain('alpha beta gamma delta epsilon ');
      expect(screen.getByRole('button', { name: 'Following' })).toBeVisible();
    });

    it(`${kind}: a dock that joined mid-request, already past the server's copy, keeps streaming`, async () => {
      // The ring held only the latest deltas: the store counts from "gamma";
      // the server's copy is older than the store's newest delta.
      routes.push((url) =>
        url.startsWith(`/api/admin/prompt-lens/requests/mid1/section/${kind}`)
          ? json({ kind, seq: 9, live: true, dropped_chars: 0, offset: 0, text: 'alpha beta gamma ', dropped_calls: 0, kept: false })
          : undefined,
      );
      render(<Harness scope="admin" initialOpen />);
      await flush();
      emit({ ...start('mid1'), restored: true });
      emit(tdelta('mid1', channel, 'gamma delta '));
      fireEvent.click(screen.getByTestId('prompt-lens-row-mid1'));
      fireEvent.click(screen.getByRole('button', { name: `Open ${title} in a large view` }));
      await flush();
      await flush();
      expect(popupBody().textContent).toContain('alpha beta gamma delta ');
      emit(tdelta('mid1', channel, 'epsilon '));
      await flush();
      expect(popupBody().textContent).toContain('alpha beta gamma delta epsilon ');
      emit(tdelta('mid1', channel, 'zeta '));
      await flush();
      expect(popupBody().textContent).toContain('alpha beta gamma delta epsilon zeta ');
    });
  }
});
