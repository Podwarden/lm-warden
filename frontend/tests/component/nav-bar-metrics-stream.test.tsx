// #251: NavBar minted TWO metrics-stream SSE tickets on every page load, in
// production too (not a StrictMode artefact). Root cause: on a fresh load
// the access token lives only in memory, so it is null, and NavBar (outside
// SessionGate) opens the header-metrics stream at once. authFetch's eager
// refresh skipped every `/api/auth/*` path, including the JWT-protected
// `/api/auth/sse-ticket`, so the first ticket POST went out with no bearer,
// got a 401, and was replayed after a refresh: two POSTs, one stream.
//
// This mounts the REAL NavBar + HeaderMetrics + shared stream against a
// backend that enforces the bearer on the ticket endpoint, exactly like the
// production page load captured in the #251 notes.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, waitFor, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { NavBar } from '@/components/nav-bar';
import { setAccessToken } from '@/lib/auth-fetch';
import { __resetHeaderMetricsStreamForTests } from '@/lib/header-metrics-stream';

vi.mock('next/navigation', () => ({
  usePathname: () => '/tokens/tok-self',
  useRouter: () => ({ replace: vi.fn() }),
}));

class FakeES {
  static all: FakeES[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { FakeES.all.push(this); }
  close() { this.closed = true; }
}

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

function installBackend() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const auth = (init?.headers as Record<string, string> | undefined)?.Authorization;
    // A real round trip, so concurrent callers overlap the way they do in a browser.
    await new Promise((r) => setTimeout(r, 5));
    if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
    if (url === '/api/auth/sse-ticket') {
      return auth === 'Bearer jwt' ? json({ ticket: `t${Math.random()}` }) : json({ detail: 'unauthorized' }, 401);
    }
    if (url === '/api/version') return auth ? json({ version: 'v1', sha: 'abc' }) : json({ detail: 'x' }, 401);
    return json({ detail: 'unexpected ' + url }, 404);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const tickets = (m: ReturnType<typeof installBackend>) =>
  m.mock.calls.filter(([u]) => String(u) === '/api/auth/sse-ticket');

describe('NavBar header-metrics stream on a fresh page load (#251)', () => {
  beforeEach(() => {
    setAccessToken(null); // a hard reload: the JWT is only ever in memory
    FakeES.all = [];
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    __resetHeaderMetricsStreamForTests();
    setAccessToken(null);
    vi.unstubAllGlobals();
  });

  it('mints ONE ticket and opens ONE EventSource per mount', async () => {
    const m = installBackend();
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <NavBar />
      </SWRConfig>,
    );
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    // Let any straggling replay land before counting.
    await act(async () => { await new Promise((r) => setTimeout(r, 50)); });

    expect(tickets(m)).toHaveLength(1);
    const [, init] = tickets(m)[0];
    expect(JSON.parse(String(init?.body))).toEqual({ path: '/api/header/metrics/stream' });
    expect((init?.headers as Record<string, string>).Authorization).toBe('Bearer jwt');
    expect(FakeES.all).toHaveLength(1);
    expect(FakeES.all[0].url).toMatch(/^\/api\/header\/metrics\/stream\?ticket=t/);
    expect(FakeES.all[0].closed).toBe(false);
    // One refresh, shared by the ticket mint and the version fetch.
    expect(m.mock.calls.filter(([u]) => String(u) === '/api/auth/refresh')).toHaveLength(1);
  });
});
