// /router/stats → "Activity" (router page redesign, plan §2.4 / §4.8 / Task 4).
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { RouterStats } from '@/components/router/router-stats';
import type { components } from '@/lib/api-types.generated';

const nav = vi.hoisted(() => ({ search: '', replace: vi.fn() }));
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => '/router/stats',
}));

import RouterStatsPage from '@/app/router/stats/page';

type Stats = components['schemas']['RouterStatsOut'];
type Decision = components['schemas']['DecisionOut'];

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

function stats(o: Partial<Stats> = {}): Stats {
  return {
    enabled: true,
    since: '2026-10-04T10:00:00Z',
    totals: { local: 11, passthrough: 22, fallback: 3, refused: 4, error: 5 },
    by_reason: { upstream_5xx: 2, timeout: 1 },
    error_reasons: { status_400: 3 },
    rules: [{
      rule_id: 'r1', pattern: 'claude-haiku*', target_model_id: 'm1', target_served_name: 'qwen3.8-27b',
      local: 11, fallback: 3, refused: 4,
      latency_ms: { p50: 120, p95: 900 }, ttfb_ms: { p50: null, p95: null },
    }],
    targets: [
      { model_id: 'm1', served_name: 'qwen3.8-27b', breaker: 'open', consecutive_failures: 3,
        open_until: '2099-01-01T00:00:00Z', last_reason: 'timeout', failures: 7 },
      { model_id: 'm2', served_name: 'other', breaker: 'closed', consecutive_failures: 0,
        open_until: null, last_reason: null, failures: 0 },
      { model_id: 'm3', served_name: null, breaker: 'half_open', consecutive_failures: 1,
        open_until: null, last_reason: 'upstream_5xx', failures: 1 },
    ],
    passthrough: {
      requests: 22, errors: 1, latency_ms: { p50: 800, p95: null }, ttfb_ms: { p50: 50, p95: 70 },
      input_tokens: 1234, output_tokens: 56,
    },
    ...o,
  };
}

// Route mix: 0 fallback, 1 passthrough, 2 refused, 3 error, rest local.
const ROUTES = ['fallback', 'passthrough', 'refused', 'error'];
function decisions(n: number): Decision[] {
  return Array.from({ length: n }, (_, i) => ({
    ts: '2026-10-04T10:00:00Z', path: '/v1/messages', model_in: `claude-haiku-${i}`,
    model_out: i === 1 ? null : 'qwen3.8-27b',
    rule_id: 'r1', route: ROUTES[i] ?? 'local', reason: i === 0 ? 'timeout' : null, status: 200,
    latency_ms: 100 + i, ttfb_ms: i === 1 ? null : 20, stream: i !== 2, token_name: 'laptop',
  }));
}

function install(s: Stats, d: Decision[]) {
  const calls: Array<[string, string]> = [];
  const state = { failStats: false };
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    calls.push([method, url]);
    if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
    if (url === '/api/csrf') return json({ csrf: 'csrf' });
    if (url === '/api/router/stats' && method === 'GET') return state.failStats ? json({ detail: 'boom' }, 500) : json(s);
    if (url === '/api/router/stats/reset' && method === 'POST') return new Response(null, { status: 204 });
    if (url.startsWith('/api/router/decisions')) return json({ decisions: d });
    return json({ detail: 'unexpected ' + url }, 404);
  });
  vi.stubGlobal('fetch', mock);
  return { mock, calls, state };
}

const wrap = (ui: React.ReactNode) => (
  <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{ui}</SWRConfig>
);

function renderIt(props: Parameters<typeof RouterStats>[0] = {}) {
  return render(wrap(<RouterStats {...props} />));
}

const rowsOf = (table: HTMLElement) => within(table).getAllByRole('row').slice(1); // minus header

describe('RouterStats (Activity)', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); nav.search = ''; nav.replace.mockReset(); });
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });

  it('heads the page "Router activity" with ← Router, the routing pill and Since', async () => {
    install(stats(), []);
    renderIt();
    expect(screen.getByRole('heading', { level: 1, name: 'Router activity' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /← Router/ })).toHaveAttribute('href', '/router');
    expect(await screen.findByText('Routing on')).toBeInTheDocument();
    expect(screen.getByText(/^Since .+ · counters are per process; a restart zeroes them\.$/)).toBeInTheDocument();
  });

  it('says "Routing off" in the head pill when the router is off', async () => {
    install(stats({ enabled: false }), []);
    renderIt();
    expect(await screen.findByText('Routing off')).toBeInTheDocument();
  });

  it('renders four KPI cards — refused and errors share one', async () => {
    install(stats(), decisions(3));
    renderIt();
    await screen.findByTestId('router-stats-rules');
    const tiles = screen.getAllByTestId('stat-card');
    expect(tiles.map((t) => t.textContent)).toEqual([
      expect.stringContaining('Local11'),
      expect.stringContaining('Anthropic (pass-through)22'),
      expect.stringContaining('Fell back3'),
      expect.stringContaining('Refused · errors4 · 5'),
    ]);
  });

  it('shows error reasons under the Refused · errors card, not in Fallback reasons', async () => {
    install(stats({ error_reasons: { status_400: 3, stream_interrupted: 1 } }), []);
    renderIt();
    await screen.findByTestId('router-stats-rules');
    const card = screen.getAllByTestId('stat-card')[3];
    expect(card.textContent).toContain('engine/upstream errors: status_400 3, stream_interrupted 1');
    const reasons = screen.getByRole('region', { name: 'Fallback reasons' });
    expect(reasons.textContent).not.toContain('status_400');
  });

  it('lists fallback reasons when there are some', async () => {
    install(stats(), []);
    renderIt();
    const reasons = await screen.findByRole('region', { name: 'Fallback reasons' });
    expect(within(reasons).getByText('upstream_5xx')).toBeInTheDocument();
    expect(within(reasons).getByText('timeout')).toBeInTheDocument();
  });

  it('by rule: counts, p50 / p95 pairs and the breaker looked up by target model', async () => {
    install(stats(), []);
    renderIt();
    const rules = within(await screen.findByTestId('router-stats-rules'));
    expect(rules.getByText('claude-haiku*')).toBeInTheDocument();
    expect(rules.getByText('qwen3.8-27b')).toBeInTheDocument();
    expect(rules.getByText('120 / 900 ms')).toBeInTheDocument();
    // both TTFB percentiles null → one em dash, not "— / — ms"
    expect(rules.getAllByText('—').length).toBeGreaterThanOrEqual(1);
    expect(rules.getByRole('columnheader', { name: 'Breaker' })).toBeInTheDocument();
    const breaker = rules.getByText('open');
    expect(breaker).toHaveClass('text-vw-danger-fg');
  });

  it('by rule: a deleted target and a target missing from the breaker list', async () => {
    install(stats({
      rules: [{
        rule_id: 'r9', pattern: 'claude-opus*', target_model_id: 'gone', target_served_name: null,
        local: 0, fallback: 0, refused: 0, latency_ms: { p50: null, p95: null }, ttfb_ms: { p50: null, p95: null },
      }],
    }), []);
    renderIt();
    const rules = within(await screen.findByTestId('router-stats-rules'));
    expect(rules.getByText('model deleted')).toBeInTheDocument();
    expect(rules.getAllByText('—').length).toBeGreaterThanOrEqual(3); // latency, TTFB, breaker
  });

  it('local models and breakers: tone pills (text first), failures, open-until, last reason', async () => {
    install(stats(), []);
    renderIt();
    const targets = within(await screen.findByTestId('router-stats-targets'));
    expect(targets.getByText('open')).toHaveClass('text-vw-danger-fg');
    expect(targets.getByText('closed')).toHaveClass('text-vw-ok-fg');
    expect(targets.getByText('half_open')).toHaveClass('text-vw-amber-fg');
    expect(targets.getByText('model deleted')).toBeInTheDocument();
    expect(targets.getByText('timeout')).toBeInTheDocument();
    expect(targets.getByText('7')).toBeInTheDocument(); // total failures
    expect(targets.getByText(/^in \d+ h$/)).toBeInTheDocument(); // open until
  });

  it('pass-through is the last by-rule row, with requests, errors and tokens in / out', async () => {
    install(stats(), []);
    renderIt();
    const table = await screen.findByTestId('router-stats-rules');
    const pt = screen.getByTestId('router-stats-passthrough');
    expect(table).toContainElement(pt);
    expect(rowsOf(table).at(-1)).toBe(pt);
    const row = within(pt);
    expect(row.getByText('Pass-through to Anthropic')).toBeInTheDocument();
    expect(row.getByText('22')).toBeInTheDocument();
    expect(row.getByText('1')).toBeInTheDocument();
    expect(row.getByText('1,234')).toBeInTheDocument();
    expect(row.getByText('56')).toBeInTheDocument();
    expect(row.getByText('800 / — ms')).toBeInTheDocument();
    expect(row.getByText('50 / 70 ms')).toBeInTheDocument();
  });

  it('renders 100 recent decisions with route words, path and stream', async () => {
    install(stats(), decisions(100));
    renderIt();
    const table = await screen.findByTestId('router-decisions');
    await waitFor(() => expect(within(table).getAllByRole('row')).toHaveLength(101)); // + header
    expect(within(table).getAllByText('fell back').length).toBe(1);
    expect(within(table).getAllByText('Anthropic').length).toBe(1);
    expect(within(table).getAllByText('claude-haiku-0 → qwen3.8-27b').length).toBe(1);
    // a pass-through decision has no model_out: only the asked-for model
    expect(within(table).getByText('claude-haiku-1')).toBeInTheDocument();
    expect(within(table).getAllByText(/\/v1\/messages/).length).toBe(100);
    expect(within(table).getAllByText(/non-stream/).length).toBe(1);
  });

  it('route chips count, filter the rows and report the choice', async () => {
    const onRouteChange = vi.fn();
    install(stats(), decisions(10));
    renderIt({ onRouteChange });
    const table = await screen.findByTestId('router-decisions');
    await waitFor(() => expect(rowsOf(table)).toHaveLength(10));
    const chips = within(screen.getByRole('group', { name: 'Filter by route' }));
    expect(chips.getByRole('button', { name: 'All 10' })).toHaveAttribute('aria-pressed', 'true');
    expect(chips.getByRole('button', { name: 'Local 6' })).toBeInTheDocument();
    expect(chips.getByRole('button', { name: 'Anthropic 1' })).toBeInTheDocument();
    expect(chips.getByRole('button', { name: 'Refused · errors 2' })).toBeInTheDocument();
    fireEvent.click(chips.getByRole('button', { name: 'Fell back 1' }));
    expect(chips.getByRole('button', { name: 'Fell back 1' })).toHaveAttribute('aria-pressed', 'true');
    expect(rowsOf(table)).toHaveLength(1);
    expect(within(table).getByText('timeout')).toBeInTheDocument();
    expect(onRouteChange).toHaveBeenLastCalledWith('fallback');
    fireEvent.click(chips.getByRole('button', { name: 'Refused · errors 2' }));
    expect(rowsOf(table)).toHaveLength(2);
    fireEvent.click(chips.getByRole('button', { name: 'All 10' }));
    expect(rowsOf(table)).toHaveLength(10);
    expect(onRouteChange).toHaveBeenLastCalledWith(null);
  });

  it('a filter with no matches says so', async () => {
    install(stats(), decisions(1).map((d) => ({ ...d, route: 'local', reason: null })));
    renderIt({ route: 'fallback' });
    expect(await screen.findByText('No fell back decisions in the last 1.')).toBeInTheDocument();
  });

  it('one empty state when since is null: no tiles, no tables, no "None."', async () => {
    install(stats({ since: null }), []);
    renderIt();
    expect(await screen.findByText('No requests yet.')).toBeInTheDocument();
    expect(screen.getByText(/Counters start with the first request after routing is on/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open the router page' })).toHaveAttribute('href', '/router');
    expect(screen.queryAllByTestId('stat-card')).toHaveLength(0);
    expect(screen.queryByTestId('router-stats-rules')).toBeNull();
    expect(screen.queryByTestId('router-decisions')).toBeNull();
    expect(screen.queryByText('None.')).toBeNull();
    expect(screen.getByText(/^No requests yet · counters are per process/)).toBeInTheDocument();
  });

  it('the empty state offers "Turn routing on" when the router is off', async () => {
    install(stats({ since: null, enabled: false }), []);
    renderIt();
    expect(await screen.findByRole('link', { name: 'Turn routing on' })).toHaveAttribute('href', '/router');
  });

  it('reset opens a confirm dialog; Confirm POSTs and re-fetches', async () => {
    const { calls } = install(stats(), []);
    renderIt();
    await screen.findByTestId('router-stats-rules');
    const before = calls.filter(([m, u]) => m === 'GET' && u === '/api/router/stats').length;
    fireEvent.click(screen.getByRole('button', { name: 'Reset counters' }));
    const dialog = within(await screen.findByRole('dialog'));
    expect(dialog.getByText(/recent-decisions list/)).toBeInTheDocument();
    fireEvent.click(dialog.getByRole('button', { name: 'Reset' }));
    await waitFor(() => expect(calls).toContainEqual(['POST', '/api/router/stats/reset']));
    await waitFor(() =>
      expect(calls.filter(([m, u]) => m === 'GET' && u === '/api/router/stats').length).toBeGreaterThan(before));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('reset Cancel sends nothing', async () => {
    const { calls } = install(stats(), []);
    renderIt();
    await screen.findByTestId('router-stats-rules');
    fireEvent.click(screen.getByRole('button', { name: 'Reset counters' }));
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(calls.some(([m]) => m === 'POST')).toBe(false);
  });

  it('shows an error when the stats request fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
      return json({ detail: 'boom' }, 500);
    }));
    renderIt();
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load router stats');
  });

  it('keeps the last good data next to a refetch error', async () => {
    const { state } = install(stats(), []);
    renderIt();
    await screen.findByTestId('router-stats-rules');
    state.failStats = true;
    // Retry lives in the alert; force a refetch through the reset path instead:
    fireEvent.click(screen.getByRole('button', { name: 'Reset counters' }));
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Reset' }));
    expect(await screen.findByText(/showing the last good data/)).toBeInTheDocument();
    expect(screen.getByTestId('router-stats-rules')).toBeInTheDocument();
    state.failStats = false;
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(screen.queryByText(/showing the last good data/)).toBeNull());
  });
});

describe('/router/stats page — ?route= sync', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); nav.replace.mockReset(); });
  afterEach(() => { cleanup(); });

  it('reads ?route=fallback and writes the chip choice back to the URL', async () => {
    nav.search = 'route=fallback';
    install(stats(), decisions(10));
    render(wrap(<RouterStatsPage />));
    const table = await screen.findByTestId('router-decisions');
    const chips = within(screen.getByRole('group', { name: 'Filter by route' }));
    await waitFor(() => expect(chips.getByRole('button', { name: 'Fell back 1' })).toHaveAttribute('aria-pressed', 'true'));
    expect(rowsOf(table)).toHaveLength(1);
    expect(nav.replace).not.toHaveBeenCalled();
    fireEvent.click(chips.getByRole('button', { name: 'Local 6' }));
    expect(nav.replace).toHaveBeenLastCalledWith('/router/stats?route=local', { scroll: false });
    fireEvent.click(chips.getByRole('button', { name: 'All 10' }));
    expect(nav.replace).toHaveBeenLastCalledWith('/router/stats', { scroll: false });
  });

  it('ignores an unknown ?route=', async () => {
    nav.search = 'route=bogus';
    install(stats(), decisions(3));
    render(wrap(<RouterStatsPage />));
    await screen.findByTestId('router-decisions');
    expect(screen.getByRole('button', { name: 'All 3' })).toHaveAttribute('aria-pressed', 'true');
  });
});

describe('Activity — review fixes', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); nav.search = ''; nav.replace.mockReset(); });
  afterEach(() => { cleanup(); });

  it('?route=error is an alias for the Refused · errors filter', async () => {
    nav.search = 'route=error';
    install(stats(), decisions(10));
    render(wrap(<RouterStatsPage />));
    const table = await screen.findByTestId('router-decisions');
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Refused · errors 2' })).toHaveAttribute('aria-pressed', 'true'),
    );
    expect(rowsOf(table)).toHaveLength(2);
  });

  it('stats failing on first load still shows the decisions that did load', async () => {
    const { state } = install(stats(), decisions(4));
    state.failStats = true;
    renderIt();
    expect(await screen.findByText(/Could not load router stats/)).toBeInTheDocument();
    const table = await screen.findByTestId('router-decisions');
    expect(rowsOf(table)).toHaveLength(4);
    expect(screen.queryByTestId('router-stats-rules')).toBeNull();
  });
});
