// Prompt Lens as a page (owner request 2026-11-23: "a standalone page, with
// menu, breadcrumbs, etc. (not a slider from the bottom of the page) - we'd
// benefit from it take full screen from start"). Admin /prompt-lens (every
// key, or ?key= preset), key holder /key/prompt-lens (own key, only when the
// platform makes it available), ?req= deep links, the menu entry and the
// breadcrumb.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));
vi.mock('@/components/header-metrics', () => ({ HeaderMetrics: () => null }));

const nav = vi.hoisted(() => ({ path: '/prompt-lens', search: '' }));
vi.mock('next/navigation', () => ({
  usePathname: () => nav.path,
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
}));

import AdminPromptLensPage from '@/app/prompt-lens/page';
import KeyPromptLensPage from '@/app/key/prompt-lens/page';
import { NavBar } from '@/components/nav-bar';
import { promptLensHref } from '@/components/prompt-lens/prompt-lens-page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { buildBreadcrumbs } from '@/lib/breadcrumbs';
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

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

interface Opts {
  enabled?: boolean;
  keyAvailable?: boolean;
  keepCaptures?: boolean;
}
let urls: string[] = [];
function stub(o: Opts = {}) {
  urls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input);
      urls.push(u);
      if (u === '/api/admin/prompt-lens/status') return json({ enabled: o.enabled ?? true });
      if (u === '/api/auth/sse-ticket') return json({ ticket: 'a1' });
      if (u === '/api/me/prompt-lens/ticket') return json({ ticket: 'k1' });
      if (u.includes('/prompt-lens/kept')) return json({ kept: [] });
      if (u === '/api/me/key')
        return json({
          id: 'tok-self', name: 'farm-07', prefix: 'vw_k2x7a', created_at: '2026-09-12 10:00:00', expires_at: null,
          is_revoked: false, is_expired: false, is_paused: false, keep_captures: false,
          prompt_lens_view_available: o.keyAvailable ?? true,
        });
      if (u.startsWith('/api/tokens/'))
        return json({
          id: 'tok-self', name: 'opencode-laptop', keep_captures: o.keepCaptures ?? false,
          lineage: [
            { id: 'tok-old', name: 'opencode-laptop (old)', is_self: false },
            { id: 'tok-self', name: 'opencode-laptop', is_self: true },
          ],
        });
      if (u === '/api/version') return json({ version: 'v1', sha: 'abc' });
      return json({});
    }),
  );
}

const wrap = (ui: React.ReactNode) =>
  render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{ui}</SWRConfig>);

async function settle(ms = 20) {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ms));
  });
}

beforeEach(() => {
  FakeES.all = [];
  nav.path = '/prompt-lens';
  nav.search = '';
  window.localStorage.clear();
  setAccessToken('jwt');
  setCsrfToken('csrf');
  storeKeyToken('key-signin', 3600);
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  clearKeyToken();
  document.documentElement.style.overflow = '';
});

describe('admin /prompt-lens', () => {
  it('fills the screen from the start: no dock, the list and the request screen side by side', async () => {
    stub();
    wrap(<AdminPromptLensPage />);
    const page = await screen.findByTestId('prompt-lens-page');
    expect(page).toHaveClass('w-screen');
    expect(page.style.height).not.toBe('');
    expect(within(page).getByTestId('prompt-lens-view')).toHaveClass('h-full');
    expect(within(page).getByTestId('prompt-lens-list')).toBeInTheDocument();
    expect(within(page).getByTestId('prompt-lens-screen')).toBeInTheDocument();
    expect(screen.queryByTestId('prompt-lens-dock')).toBeNull();
    // The document does not scroll under it; leaving gives the scroll back.
    expect(document.documentElement.style.overflow).toBe('hidden');
    await waitFor(() => expect(FakeES.all.map((e) => e.url)).toEqual(['/api/admin/prompt-lens/stream?ticket=a1']));
    cleanup();
    expect(document.documentElement.style.overflow).toBe('');
  });

  it('?req= opens that request (it waits for it to arrive)', async () => {
    stub();
    nav.search = 'req=rq-42';
    wrap(<AdminPromptLensPage />);
    await screen.findByTestId('prompt-lens-view');
    await waitFor(() => expect(screen.getByTestId('prompt-lens-screen')).toHaveTextContent('Looking for the request'));
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    act(() => {
      FakeES.all[0].onopen?.();
      FakeES.all[0].onmessage?.({
        data: JSON.stringify({
          type: 'request_start', req_id: 'g42', ts: Date.now() / 1000, token_id: 'tok-a', request_id: 'rq-42', model: 'm',
          served_name: 'm', stream: true, prompt: 'hello', session_id: 'sess0001aaaa', turn: 1, seq: 1,
        }),
      } as MessageEvent);
    });
    await waitFor(() => expect(screen.getByTestId('prompt-lens-row-g42')).toHaveAttribute('aria-current', 'true'));
  });

  it('?key= presets the Key filter (the key and its earlier keys) and names them', async () => {
    stub();
    nav.search = 'key=tok-old,tok-self';
    wrap(<AdminPromptLensPage />);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    const u = new URL(FakeES.all[0].url, 'http://x');
    expect(u.searchParams.get('token_ids')).toBe('tok-old,tok-self');
    await waitFor(() =>
      expect(screen.getByTestId('prompt-lens-header')).toHaveTextContent('opencode-laptop (old), opencode-laptop'),
    );
    expect(urls.some((x) => x.startsWith('/api/me/'))).toBe(false);
  });

  it('picking a request writes ?req= into the address without a history entry', async () => {
    stub();
    nav.search = 'key=tok-self';
    window.history.replaceState(null, '', '/ui/prompt-lens/?key=tok-self');
    const len = window.history.length;
    wrap(<AdminPromptLensPage />);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    act(() => {
      FakeES.all[0].onopen?.();
      FakeES.all[0].onmessage?.({
        data: JSON.stringify({
          type: 'request_start', req_id: 'g7', ts: Date.now() / 1000, token_id: 'tok-self', request_id: 'rq-7', model: 'm',
          served_name: 'm', stream: true, prompt: 'hello', session_id: 'sess0001aaaa', turn: 1, seq: 1,
        }),
      } as MessageEvent);
    });
    await settle(40);
    fireEvent.click(await screen.findByTestId('prompt-lens-row-g7'));
    expect(window.location.pathname + window.location.search).toBe('/ui/prompt-lens/?key=tok-self&req=rq-7');
    expect(window.history.length).toBe(len);
  });

  it('says it is off, and where to turn it on, when the platform switch is off', async () => {
    stub({ enabled: false });
    wrap(<AdminPromptLensPage />);
    const off = await screen.findByTestId('prompt-lens-off');
    expect(off).toHaveTextContent('Prompt Lens is off');
    expect(within(off).getByRole('link', { name: /Settings, General, Privacy/ })).toHaveAttribute('href', '/settings');
    await settle();
    expect(FakeES.all).toHaveLength(0);
  });

  it('while off, a ?key= that keeps captures still opens (its kept captures are readable)', async () => {
    stub({ enabled: false, keepCaptures: true });
    nav.search = 'key=tok-self';
    wrap(<AdminPromptLensPage />);
    await screen.findByTestId('prompt-lens-view');
    await waitFor(() => expect(urls.some((x) => x.startsWith('/api/admin/prompt-lens/kept'))).toBe(true));
  });
});

describe('key holder /key/prompt-lens', () => {
  beforeEach(() => {
    nav.path = '/key/prompt-lens';
  });

  it('own key only: the key stream, the area nav with Prompt Lens current, "Home › Prompt Lens"', async () => {
    stub();
    wrap(<KeyPromptLensPage />);
    const page = await screen.findByTestId('prompt-lens-page');
    await waitFor(() => expect(FakeES.all.map((e) => e.url)).toEqual(['/api/me/prompt-lens/stream?ticket=k1']));
    expect(urls.some((x) => x.startsWith('/api/admin/'))).toBe(false);
    const keyNav = within(page).getByRole('navigation', { name: 'Key' });
    expect(within(keyNav).getByRole('link', { name: 'Prompt Lens' })).toHaveAttribute('aria-current', 'page');
    const crumb = within(page).getByTestId('key-breadcrumb');
    expect(crumb).toHaveTextContent('Home›Prompt Lens');
    expect(within(crumb).getByRole('link', { name: 'Home' })).toHaveAttribute('href', '/key');
    // Nothing about other keys: no Key filter.
    expect(within(screen.getByTestId('prompt-lens-filter-line')).queryByRole('button', { name: /^Key/ })).toBeNull();
  });

  it('is not shown, and opens no stream, when the platform does not make it available', async () => {
    stub({ keyAvailable: false });
    wrap(<KeyPromptLensPage />);
    expect(await screen.findByTestId('prompt-lens-off')).toHaveTextContent('Prompt Lens is not available');
    await settle();
    expect(FakeES.all).toHaveLength(0);
    expect(screen.queryByRole('link', { name: 'Prompt Lens' })).toBeNull();
  });
});

describe('menu, breadcrumb and links', () => {
  it('the admin menu lists Prompt Lens only while it is on', async () => {
    nav.path = '/models';
    stub({ enabled: true });
    wrap(<NavBar />);
    await settle();
    fireEvent.click(screen.getByRole('button', { name: 'Open menu' }));
    expect(screen.getByRole('menuitem', { name: 'Prompt Lens' })).toHaveAttribute('href', '/prompt-lens');
    cleanup();
    stub({ enabled: false });
    wrap(<NavBar />);
    await settle();
    fireEvent.click(screen.getByRole('button', { name: 'Open menu' }));
    expect(screen.queryByRole('menuitem', { name: 'Prompt Lens' })).toBeNull();
    expect(screen.getByRole('menuitem', { name: 'Stats' })).toBeInTheDocument();
  });

  it('the breadcrumb reads Home › Prompt Lens, also on the static export\'s trailing-slash path', () => {
    for (const p of ['/prompt-lens', '/prompt-lens/']) {
      expect(buildBreadcrumbs(p).map((c) => [c.label, c.navigable])).toEqual([
        ['Home', true],
        ['Prompt Lens', true],
      ]);
    }
  });

  it('deep links', () => {
    expect(promptLensHref('admin', { req: 'rq 1' })).toBe('/prompt-lens?req=rq+1');
    expect(promptLensHref('admin', { keys: ['a', 'b'] })).toBe('/prompt-lens?key=a%2Cb');
    // A key holder's link never carries a key.
    expect(promptLensHref('key', { req: 'r1', keys: ['a'] })).toBe('/key/prompt-lens?req=r1');
  });
});
