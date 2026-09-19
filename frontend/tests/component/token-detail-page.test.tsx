import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, waitFor, renderHook, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { NOW, tokenDetail, seriesFixture } from './token-fixtures';

const nav = vi.hoisted(() => ({ replace: vi.fn(), push: vi.fn(), search: '', sp: new URLSearchParams() }));
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: nav.push, refresh: vi.fn() }),
  useSearchParams: () => {
    if (nav.sp.toString() !== nav.search) nav.sp = new URLSearchParams(nav.search);
    return nav.sp;
  },
  usePathname: () => '/tokens/tok-self',
}));

import TokenDetailPage from '@/app/tokens/[id]/page';
import { useDockOpen } from '@/components/tokens/detail/godmode-dock';
import { NavStackProvider } from '@/lib/nav-stack';
import { BreadcrumbHeader } from '@/components/breadcrumb-header';

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void; onmessage?: (e: MessageEvent) => void; onerror?: () => void;
  closed = false;
  constructor(public url: string) { FakeES.all.push(this); }
  close() { this.closed = true; }
}

function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = 'fulfilled';
  p.value = value;
  return p;
}

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

function installFetch(
  o: { tokenStatus?: number; godmode?: unknown; godmodeStatus?: number; token?: Parameters<typeof tokenDetail>[0] } = {},
) {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
    if (url === '/api/csrf') return json({ csrf: 'csrf' });
    if (url === '/api/admin/godmode/status') return json(o.godmode ?? { enabled: true }, o.godmodeStatus ?? 200);
    if (url.startsWith('/api/tokens/tok-self/series')) return json(seriesFixture());
    if (url === '/api/tokens/tok-self') {
      if (init?.method === 'PATCH') return json(tokenDetail(JSON.parse(String(init.body))));
      return o.tokenStatus ? json({ detail: 'token not found' }, o.tokenStatus) : json(tokenDetail(o.token));
    }
    return json({ detail: 'unexpected ' + url }, 404);
  });
  vi.stubGlobal('fetch', mock);
  return mock;
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div>loading</div>}>
        <TokenDetailPage params={syncResolved({ id: 'tok-self' })} />
      </Suspense>
    </SWRConfig>,
  );
}

// The history strip's own requests start at the fixture lineage's oldest
// created_at (it no longer waits for a measured width, so it fetches under
// jsdom too, and again whenever the clock crosses a minute). The page's
// preset/custom series requests are everything else.
const STRIP_FROM = Date.UTC(2026, 8, 5, 12, 16) / 1000;
const seriesCalls = (m: ReturnType<typeof vi.fn>) =>
  m.mock.calls.map(([u]) => String(u)).filter((u) => u.includes('/series?'))
    .filter((u) => Number(new URL(u, 'http://x').searchParams.get('from')) !== STRIP_FROM);

// The server window a request asks for: `[floor(from/60), ceil(to/60))` in minutes.
const serverMinutes = (q: URLSearchParams) => Math.ceil(Number(q.get('to')) / 60) - Math.floor(Number(q.get('from')) / 60);

describe('token details page', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    nav.replace.mockReset();
    nav.push.mockReset();
    nav.search = '';
    FakeES.all = [];
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  });
  // Belt-and-suspenders: several tests below flip on fake timers and/or stub
  // `HTMLElement.prototype.clientWidth` inside their own try/finally. If one
  // of those tests times out, Vitest moves on to the next test while the
  // timed-out test's body (and its `finally`) may still be executing in the
  // background — real timeouts/microtasks don't get cancelled — so a
  // straggler can flip real timers back to fake, or delete the clientWidth
  // stub, WHILE the next test is mid-flight. That's exactly how one slow
  // test (a timeout) cascades into "Unable to find role=..." failures in
  // unrelated tests later in the file. This afterEach always restores the
  // shared globals to their default state before the next test starts, so a
  // single timeout can never leave the suite in a broken state.
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    Reflect.deleteProperty(HTMLElement.prototype, 'clientWidth');
  });

  // The page names its crumb (and the back button's label, once the user
  // moves on) after the token, through useBreadcrumb — the app-wide strip
  // replaced the header's own "API tokens / <name>" row.
  it('titles the breadcrumb with the token name', async () => {
    installFetch();
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <NavStackProvider>
          <BreadcrumbHeader />
          <Suspense fallback={<div>loading</div>}>
            <TokenDetailPage params={syncResolved({ id: 'tok-self' })} />
          </Suspense>
        </NavStackProvider>
      </SWRConfig>,
    );
    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' });
    await waitFor(() => {
      const current = crumbs.querySelector('ol [aria-current="page"]');
      expect(current).toHaveTextContent('opencode-ip-macbook');
    });
    expect(crumbs.querySelector('ol')).toHaveTextContent(/^HomeAPI tokensopencode-ip-macbook$/);
    // One trail only: the page itself draws no second breadcrumb.
    expect(screen.getAllByRole('navigation', { name: 'Breadcrumb' })).toHaveLength(1);
  });

  it('titles the breadcrumb with the id when the token is gone', async () => {
    installFetch({ tokenStatus: 404 });
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <NavStackProvider>
          <BreadcrumbHeader />
          <Suspense fallback={<div>loading</div>}>
            <TokenDetailPage params={syncResolved({ id: 'tok-self' })} />
          </Suspense>
        </NavStackProvider>
      </SWRConfig>,
    );
    expect(await screen.findByText('Token not found')).toBeInTheDocument();
    const crumbs = screen.getByRole('navigation', { name: 'Breadcrumb' });
    await waitFor(() => expect(crumbs.querySelector('ol [aria-current="page"]')).toHaveTextContent('tok-self'));
  });

  it('shows "Token not found" with a link back on 404', async () => {
    installFetch({ tokenStatus: 404 });
    renderPage();
    expect(await screen.findByText('Token not found')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Back to API tokens' })).toHaveAttribute('href', '/tokens');
  });

  // #251: a key deleted elsewhere (from another tab, or by another admin)
  // made every later token, series and strip poll 404 forever. Once the token
  // itself 404s, the page shows "Token not found" and stops asking.
  it('stops polling the token, series and strip once the token 404s', async () => {
    vi.useFakeTimers();
    try {
      let gone = false;
      const m = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
        if (url === '/api/admin/godmode/status') return json({ enabled: true });
        if (gone && url.startsWith('/api/tokens/tok-self')) return json({ detail: 'token not found' }, 404);
        if (url.startsWith('/api/tokens/tok-self/series')) return json(seriesFixture());
        if (url === '/api/tokens/tok-self') return json(tokenDetail());
        return json({ detail: 'unexpected ' + url }, 404);
      });
      vi.stubGlobal('fetch', m);
      // 1h: the fastest series cadence (10 s). It also re-renders the page
      // every 10 s (useNow), which used to restart — and so starve — the
      // token's own 10 s poll: the 404 below would never have been seen.
      nav.search = 'range=1h';
      renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      expect(screen.getByRole('heading', { level: 1, name: 'opencode-ip-macbook' })).toBeInTheDocument();

      gone = true;
      await act(async () => { await vi.advanceTimersByTimeAsync(11_000); }); // the next token poll 404s
      expect(screen.getByText('Token not found')).toBeInTheDocument();

      const tokenCalls = () => m.mock.calls.filter(([u]) => String(u).startsWith('/api/tokens/')).length;
      const before = tokenCalls();
      // Error retries (SWR's back-off), token polls, series and strip polls,
      // and focus revalidation would all have fired within five minutes.
      await act(async () => { await vi.advanceTimersByTimeAsync(5 * 60_000); });
      window.dispatchEvent(new Event('focus'));
      document.dispatchEvent(new Event('visibilitychange'));
      await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
      expect(tokenCalls()).toBe(before);
      expect(screen.getByText('Token not found')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  }, 20_000);

  // Fix round 1 (#251 review): a refreshInterval of 0 while the tab was
  // hidden ended SWR's polling loop for good — it only reschedules a
  // non-zero interval — so a tab hidden once never polled the token again.
  it('keeps polling the token after the tab was hidden and shown again', async () => {
    vi.useFakeTimers();
    let hidden = false;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (hidden ? 'hidden' : 'visible') });
    try {
      const m = installFetch();
      renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      expect(screen.getByRole('heading', { level: 1, name: 'opencode-ip-macbook' })).toBeInTheDocument();
      const tokenGets = () => m.mock.calls.filter(([u, init]) =>
        String(u) === '/api/tokens/tok-self' && (init as RequestInit | undefined)?.method !== 'PATCH').length;

      hidden = true;
      await act(async () => { await vi.advanceTimersByTimeAsync(25_000); }); // past a tick while hidden
      hidden = false;                                                        // shown again, no focus event
      const before = tokenGets();
      await act(async () => { await vi.advanceTimersByTimeAsync(10_500); });
      expect(tokenGets()).toBeGreaterThan(before);
    } finally {
      vi.useRealTimers();
      Reflect.deleteProperty(document, 'hidden');
      Reflect.deleteProperty(document, 'visibilityState');
    }
  }, 20_000);

  it('defaults to 7d with earlier keys and asks the server for ≤360 bins', async () => {
    const m = installFetch();
    renderPage();
    expect(await screen.findByRole('heading', { level: 1, name: 'opencode-ip-macbook' })).toBeInTheDocument();
    await waitFor(() => expect(seriesCalls(m).length).toBeGreaterThan(0));
    const u = new URL(seriesCalls(m)[0], 'http://x');
    expect(u.searchParams.get('chain')).toBe('1');
    expect(u.searchParams.get('max_bins')).toBe('360');
    expect(serverMinutes(u.searchParams)).toBe(7 * 1440);
    expect(await screen.findByText('last 7 days · 30 min bins · 337 points')).toBeInTheDocument();
  });

  it('a preset writes ?range= to the URL; the URL drives the window', async () => {
    const m = installFetch();
    const { rerender } = renderPage();
    fireEvent.click(await screen.findByRole('button', { name: '24h' }));
    expect(nav.replace).toHaveBeenCalledWith('/tokens/tok-self?range=24h', { scroll: false });

    nav.search = 'range=24h';
    rerender(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <Suspense fallback={<div>loading</div>}>
          <TokenDetailPage params={syncResolved({ id: 'tok-self' })} />
        </Suspense>
      </SWRConfig>,
    );
    await waitFor(() => {
      const last = new URL(seriesCalls(m).at(-1)!, 'http://x');
      expect(serverMinutes(last.searchParams)).toBe(1440);
    });
    expect(screen.getByRole('button', { name: '24h' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('a custom URL opens with the custom row filled in', async () => {
    nav.search = `from=${Date.UTC(2026, 8, 12, 9, 0) / 1000}&to=${Date.UTC(2026, 8, 14, 18, 30) / 1000}`;
    installFetch();
    renderPage();
    expect(await screen.findByLabelText('From')).toHaveValue('2026-09-12T09:00');
    expect(screen.getByLabelText('To')).toHaveValue('2026-09-14T18:30');
    expect(screen.getByRole('button', { name: 'Custom' })).toHaveAttribute('aria-pressed', 'true');
  });

  // A custom range does not poll, so Apply with the times already in the URL
  // used to do nothing at all: no URL change, no request, no feedback.
  it('Apply with unchanged custom times refetches the series and the strip', async () => {
    const from = Date.UTC(2026, 8, 12, 9, 0) / 1000;
    const to = Date.UTC(2026, 8, 14, 18, 30) / 1000;
    nav.search = `from=${from}&to=${to}`;
    const m = installFetch();
    renderPage();
    expect(await screen.findByLabelText('From')).toHaveValue('2026-09-12T09:00');
    const stripCalls = () => m.mock.calls.filter(([u]) => String(u).includes('/series?')).length - seriesCalls(m).length;
    await screen.findByText(/30 min bins · 337 points/);
    await waitFor(() => expect(stripCalls()).toBe(2)); // own + chain
    const before = seriesCalls(m).length;
    expect(before).toBe(1);

    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

    await waitFor(() => expect(seriesCalls(m).length).toBe(before + 1));
    const q = new URL(seriesCalls(m).at(-1)!, 'http://x').searchParams;
    expect([Number(q.get('from')), Number(q.get('to'))]).toEqual([from, to]);
    await waitFor(() => expect(stripCalls()).toBe(4));
    expect(nav.replace).not.toHaveBeenCalled(); // nothing to write: the URL already says this
  });

  it('Apply with new custom times writes the URL at once', async () => {
    nav.search = `from=${Date.UTC(2026, 8, 12, 9, 0) / 1000}&to=${Date.UTC(2026, 8, 14, 18, 30) / 1000}`;
    installFetch();
    renderPage();
    fireEvent.change(await screen.findByLabelText('From'), { target: { value: '2026-09-13T09:00' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(nav.replace).toHaveBeenCalledWith(
      `/tokens/tok-self?from=${Date.UTC(2026, 8, 13, 9, 0) / 1000}&to=${Date.UTC(2026, 8, 14, 18, 30) / 1000}`,
      { scroll: false },
    );
  });

  // Controller ruling (fix round 1): a preset's SWR key must stay stable
  // between polls (from/to are resolved inside the fetcher, at fetch time,
  // not baked into the key), and `refreshInterval` must be the ONLY thing
  // driving the re-fetch cadence — so 30 s of 1h polling (10 s cadence)
  // produces exactly 3 more fetches, never a growing cache from a key that
  // changes every tick.
  it('polls a preset on a stable SWR key: 30 s of fake time on 1h yields exactly 3 more series fetches', async () => {
    vi.useFakeTimers();
    try {
      const m = installFetch();
      nav.search = 'range=1h';
      renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      expect(screen.getByRole('heading', { level: 1, name: 'opencode-ip-macbook' })).toBeInTheDocument();
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      const before = seriesCalls(m).length;
      expect(before).toBeGreaterThan(0);

      await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
      const fired = seriesCalls(m).slice(before);
      expect(fired).toHaveLength(3);
      // Every poll asks for a fresh 1h window (60 server minutes) even
      // though the underlying SWR key never changed.
      for (const u of fired) {
        const q = new URL(u, 'http://x').searchParams;
        expect(q.get('chain')).toBe('1');
        expect(q.get('max_bins')).toBe('360');
        expect(serverMinutes(q)).toBe(60);
      }
    } finally {
      vi.useRealTimers();
    }
  });

  // Controller ruling (fix round 2): the strip's SWR key carries only the
  // key's identity and the own/chain distinction — never the window — so it
  // never changes as the clock moves, and `refreshInterval: 60_000` is the
  // sole re-fetch driver (own AND chain each refresh once a minute). Before
  // this fix the key baked in a minute-floored `to`, so SWR's (never-
  // evicting) cache grew a new strip entry every minute the page stayed open.
  it('the strip SWR cache does not grow across polls; each variant refreshes once a minute', async () => {
    vi.useFakeTimers();
    try {
      const cache = new Map();
      const m = installFetch();
      render(
        <SWRConfig value={{ provider: () => cache, dedupingInterval: 0 }}>
          <Suspense fallback={<div>loading</div>}>
            <TokenDetailPage params={syncResolved({ id: 'tok-self' })} />
          </Suspense>
        </SWRConfig>,
      );
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      expect(screen.getByRole('heading', { level: 1, name: 'opencode-ip-macbook' })).toBeInTheDocument();

      const stripCalls = (calls: ReturnType<typeof vi.fn>['mock']['calls']) =>
        calls.filter(([u]) => {
          const s = String(u);
          if (!s.includes('/series?')) return false;
          return Number(new URL(s, 'http://x').searchParams.get('from')) === STRIP_FROM;
        });

      const keysBefore = cache.size;
      const before = stripCalls(m.mock.calls).length;
      expect(before).toBeGreaterThan(0); // both variants (own + chain) fetched once already

      await act(async () => { await vi.advanceTimersByTimeAsync(3 * 60_000); });

      expect(cache.size).toBe(keysBefore); // no new SWR cache entries after 3 more minutes
      expect(stripCalls(m.mock.calls).length - before).toBe(3 * 2); // 3 minutes x 2 variants
    } finally {
      vi.useRealTimers();
    }
  });

  // Controller ruling (fix round 2): `bounds` (the strip's drawn axis) comes
  // from the strip response's own `from_minute`/`to_minute`, resolved at
  // fetch time with an UNFLOORED `to`, exactly like a preset's own window —
  // so the two can never drift apart and the selection brush can never hang
  // off the strip's right edge, however long the clock has been advancing.
  // Explicit 20s timeout (default is 5s): the simulated minute crossing
  // below wakes several SWR polls (10s token poll, 60s series/strip polls)
  // that re-render the whole page, which is slow on a loaded CI runner.
  it("the strip's right edge never sits left of the preset window's end (no overhang)", async () => {
    // A key ~10 minutes old, so the strip's whole-life span is short enough
    // that a several-second gap between the strip's bounds and the preset
    // window's end is a large, easily-asserted fraction of the strip's
    // width (on the mockup's 13-day fixture the same gap is sub-pixel).
    // The clock starts 37 s past a whole minute — under the old page-local,
    // minute-floored `bounds.to` this alone put the brush past the strip's
    // right edge.
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 800 });
    vi.useFakeTimers();
    const start = NOW * 1000 + 37_000;
    vi.setSystemTime(new Date(start));
    const created = new Date(start - 10 * 60_000).toISOString().slice(0, 19).replace('T', ' ');
    try {
      const m = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
        if (url === '/api/csrf') return json({ csrf: 'csrf' });
        if (url === '/api/admin/godmode/status') return json({ enabled: true });
        if (url.startsWith('/api/tokens/tok-self/series')) {
          const q = new URL(url, 'http://x').searchParams;
          const from = Number(q.get('from'));
          const to = Number(q.get('to'));
          // Mirrors the real server's [floor(from/60), ceil(to/60)) window.
          return json(seriesFixture({ from_minute: Math.floor(from / 60), to_minute: Math.ceil(to / 60) }));
        }
        if (url === '/api/tokens/tok-self') {
          return json(tokenDetail({
            created_at: created,
            lineage: [
              { id: 'tok-self', name: 'opencode-ip-macbook', created_at: created, rotated_at: null, is_revoked: false, in_grace: false, is_self: true },
            ],
          }));
        }
        return json({ detail: 'unexpected ' + url }, 404);
      });
      vi.stubGlobal('fetch', m);

      renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      await act(async () => { await vi.advanceTimersByTimeAsync(0); });
      expect(screen.getByRole('heading', { level: 1, name: 'opencode-ip-macbook' })).toBeInTheDocument();

      const noOverhang = () => {
        const slider = screen.getByRole('slider');
        const left = parseFloat(slider.style.left);
        const width = parseFloat(slider.style.width);
        expect(left + width).toBeLessThanOrEqual(800 + 0.5);
      };
      noOverhang();

      // One minute-boundary crossing is enough to prove the invariant holds
      // on a fresh poll, not just at mount — a longer run (the original
      // version of this test advanced several more minutes) added runtime
      // (every extra minute wakes the page's 10 s/60 s pollers, each of
      // which re-fetches and re-renders the whole page) but no extra
      // coverage: the fix resolves `bounds.to` at fetch time from the same
      // unfloored `now` as the preset window, every time, so the invariant
      // either holds on every poll or none of them.
      await act(async () => { await vi.advanceTimersByTimeAsync(65_000); });
      noOverhang();
    } finally {
      vi.useRealTimers();
      Reflect.deleteProperty(HTMLElement.prototype, 'clientWidth');
    }
  }, 20_000);

  // Controller ruling (fix round 1): the strip covers the key's whole life;
  // for a key created inside the current polling minute, `lineageStart` can
  // land ON `stripTo`, so the request must fall back to `stripTo - 60 s`
  // rather than asking the server for `from === to` (which 422s).
  it('a key created within the current minute keeps the strip request from < to', async () => {
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 800 });
    try {
      const created = new Date().toISOString().slice(0, 19).replace('T', ' ');
      const m = installFetch({
        token: {
          created_at: created,
          lineage: [
            { id: 'tok-self', name: 'opencode-ip-macbook', created_at: created, rotated_at: null, is_revoked: false, in_grace: false, is_self: true },
          ],
        },
      });
      renderPage();
      await screen.findByRole('heading', { level: 1, name: 'opencode-ip-macbook' });
      const stripCall = await waitFor(() => {
        // The strip's request is the one over the key's life (here one
        // minute), not the 7d preset; it asks for the server's finest bins.
        const u = m.mock.calls.map(([x]) => String(x)).find((x) => {
          if (!x.includes('/series?')) return false;
          const p = new URL(x, 'http://x').searchParams;
          return Number(p.get('to')) - Number(p.get('from')) < 86_400;
        });
        expect(u).toBeDefined();
        return u!;
      });
      const q = new URL(stripCall, 'http://x').searchParams;
      expect(Number(q.get('from'))).toBeLessThan(Number(q.get('to')));
      expect(q.get('max_bins')).toBe('360');
    } finally {
      Reflect.deleteProperty(HTMLElement.prototype, 'clientWidth');
    }
  });

  // #251: the strip draws token volume only, so its (whole-life) requests
  // ask the server to skip the latency timings -- and the per-model split
  // (0036); the charts' requests do not.
  it('the history strip asks for timings=0 and by_model=0; the main series does not', async () => {
    const m = installFetch();
    renderPage();
    await screen.findByText('last 7 days · 30 min bins · 337 points');
    const all = () => m.mock.calls.map(([u]) => String(u)).filter((u) => u.includes('/series?'));
    const isStrip = (u: string) => Number(new URL(u, 'http://x').searchParams.get('from')) === STRIP_FROM;
    await waitFor(() => expect(all().filter(isStrip)).toHaveLength(2)); // own + chain
    for (const u of all().filter(isStrip)) {
      const q = new URL(u, 'http://x').searchParams;
      expect(q.get('timings')).toBe('0');
      expect(q.get('by_model')).toBe('0');
    }
    const main = seriesCalls(m);
    expect(main.length).toBeGreaterThan(0);
    for (const u of main) {
      const q = new URL(u, 'http://x').searchParams;
      expect(q.has('timings')).toBe(false);
      expect(q.has('by_model')).toBe(false);
    }
  });

  it('shows the usage-by-model card between the summary and the charts', async () => {
    installFetch();
    renderPage();
    const card = await screen.findByRole('heading', { name: 'Usage by model' });
    const summary = screen.getByText('prefill tokens');
    const charts = screen.getByRole('heading', { name: 'Tokens per minute' });
    expect(summary.compareDocumentPosition(card) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(card.compareDocumentPosition(charts) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByText('No per-model data in this period yet.')).toBeInTheDocument();
  });

  it('turning off Include earlier keys asks for chain=0', async () => {
    const m = installFetch();
    renderPage();
    fireEvent.click(await screen.findByLabelText('Include earlier keys'));
    await waitFor(() => expect(new URL(seriesCalls(m).at(-1)!, 'http://x').searchParams.get('chain')).toBe('0'));
  });

  it('Save changes sends ONE PATCH with only the priority', async () => {
    const m = installFetch();
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'P5' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    await waitFor(() => expect(screen.getByText('Limits saved')).toBeInTheDocument());
    const patches = m.mock.calls.filter(([, init]) => (init as RequestInit | undefined)?.method === 'PATCH');
    expect(patches).toHaveLength(1);
    expect(JSON.parse(String((patches[0][1] as RequestInit).body))).toEqual({ priority: 5 });
  });

  // #251: a 2xx PATCH whose body isn't JSON used to reject onRename /
  // onSaveLimits with no message at all.
  it('a 2xx PATCH with an unreadable body says so, for Save changes and for rename', async () => {
    const m = installFetch();
    const base = m.getMockImplementation()!;
    m.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) =>
      init?.method === 'PATCH' ? new Response('<html>oops</html>', { status: 200 }) : base(input, init));
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'P5' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
    const msg = "the server's reply could not be read (HTTP 200)";
    expect(await screen.findByRole('alert')).toHaveTextContent(msg);

    fireEvent.click(screen.getByRole('button', { name: 'Rename token' }));
    const input = screen.getByRole('textbox', { name: 'Token name' });
    fireEvent.change(input, { target: { value: 'laptop key' } });
    fireEvent.submit(input.closest('form')!);
    await waitFor(() => expect(screen.getByText(`Rename failed: ${msg}`)).toBeInTheDocument());
    expect(screen.getByRole('textbox', { name: 'Token name' })).toBeInTheDocument(); // still editing
  });

  it('renders the dock closed, with no EventSource, when god mode is enabled', async () => {
    installFetch({ godmode: { enabled: true } });
    renderPage();
    const bar = await screen.findByRole('button', { name: /God mode/ });
    expect(bar).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('region', { name: 'God mode for this token' })).toBeInTheDocument();
    expect(FakeES.all).toHaveLength(0);
  });

  it.each([
    ['disabled', { godmode: { enabled: false } }],
    ['status fetch failed', { godmodeStatus: 500 }],
  ])('has no dock when god mode is %s', async (_label, opts) => {
    installFetch(opts);
    renderPage();
    await screen.findByRole('heading', { level: 1, name: 'opencode-ip-macbook' });
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByRole('region', { name: 'God mode for this token' })).toBeNull();
    expect(screen.queryByRole('button', { name: /God mode/ })).toBeNull();
  });

  // Operator's hard rule: opening the dock must always be a deliberate click.
  // If god mode flips off (admin disables it, or the status fetch starts
  // failing) while the dock happens to be open, `open` must drop to false —
  // otherwise a later flip back to true remounts <GodModeDock> already open
  // and a stream starts with no click.
  it('useDockOpen resets to closed when god mode status flips off, and stays closed when it flips back on', () => {
    const { result, rerender } = renderHook(({ enabled }) => useDockOpen(enabled), {
      initialProps: { enabled: true },
    });
    act(() => result.current[1](true));
    expect(result.current[0]).toBe(true);

    rerender({ enabled: false });
    expect(result.current[0]).toBe(false);

    rerender({ enabled: true });
    expect(result.current[0]).toBe(false);
  });
});
