import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

let mockPath = '/sessions';
const mockReplace = vi.fn();
vi.mock('next/navigation', () => ({
  usePathname: () => mockPath,
  useRouter: () => ({ replace: mockReplace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

import SessionsPage from '@/app/sessions/page';
import { setAccessToken } from '@/lib/auth-fetch';
import { S } from '@/lib/sessions/strings';

const T0 = 1_760_000_400;
const ANAT = { cached: 870, dead: 90, recomputed: 0, new: 40, unknown: 0 };

function item(i: number, extra: Record<string, unknown> = {}) {
  return {
    session_id: 's_' + String(i).padStart(32, 'a'),
    token_id: 'tok1',
    token_name: 'alice',
    models: ['qwen'],
    started_at: T0 + i * 60,
    last_at: T0 + i * 60 + 1860,
    duration_s: 1860,
    requests: 10 + i,
    conversations: 4,
    prefill_tokens: 1000,
    cached_tokens: 870,
    cached_share: 0.87,
    relayed_requests: 0,
    generated_tokens: 50,
    compactions: 2,
    active: false,
    harness: 'claude_code',
    where: 'local',
    prompt_tokens: 1000,
    anatomy: ANAT,
    dead_weight_share: 0.09,
    dead_weight_tokens: 90,
    dead_weight_measured: true,
    saved_vs_vanilla_s: 2460,
    replicas: 3,
    recomputed_s: 1,
    cache_served_s: 2,
    cache_known: true,
    cache_share: 0.87,
    evicted_resumes: 0,
    evicted_lost_s: null,
    ...extra,
  };
}

function group(key: string, label: string, items: unknown[], extra: Record<string, unknown> = {}) {
  return {
    key,
    label,
    harnesses: ['claude_code'],
    where: { local: items.length, relayed: 0, hybrid: 0 },
    sessions: items.length,
    duration_s: 1_483_200,
    prompt_tokens: 9_100_000_000,
    cached_share: 0.95,
    anatomy: ANAT,
    dead_weight_share: 0.09,
    saved_vs_vanilla_s: 111_600,
    evicted_resumes: 0,
    evicted_lost_s: null,
    items,
    more: 0,
    ...extra,
  };
}

const PAUSES = {
  resumes: 5,
  evicted: 2,
  unassessed: 1,
  lost_tokens: 9000,
  lost_s: 540,
  buckets: [
    { label: '1-5m', min_s: 60, max_s: 300, resumes: 3, evicted: 0, lost_tokens: 0, lost_s: null },
    { label: '5-15m', min_s: 300, max_s: 900, resumes: 1, evicted: 1, lost_tokens: 4000, lost_s: 240 },
    { label: '15-60m', min_s: 900, max_s: 3600, resumes: 1, evicted: 1, lost_tokens: 5000, lost_s: 300 },
    { label: '1-4h', min_s: 3600, max_s: 14400, resumes: 0, evicted: 0, lost_tokens: 0, lost_s: null },
    { label: '4h+', min_s: 14400, max_s: null, resumes: 0, evicted: 0, lost_tokens: 0, lost_s: null },
  ],
  disk_tier_hint: { keep_s: 3600, recovered_s: 540 },
};

const VIEWS = { all: 812, running: 6, biggest: 50, dead_weight: 37, cache_trouble: 21, long_running: 44, evicted: 3 };

function hist(sessions = [2, 4, 0, 1]) {
  return {
    bin_s: 3600,
    start: T0,
    bins: sessions.map((n, i) => ({ t: T0 + i * 3600, sessions: n, over_dw: n > 2 ? 1 : 0 })),
  };
}

function resp(groups: unknown[] | null, extra: Record<string, unknown> = {}) {
  return {
    sessions: [],
    next_cursor: null,
    truncated_before: null,
    oldest_data_at: null,
    total: 812,
    views: VIEWS,
    histogram: hist(),
    pause_summary: PAUSES,
    groups,
    ...extra,
  };
}

const EMPTY = resp([], {
  total: 0,
  views: { all: 0, running: 0, biggest: 0, dead_weight: 0, cache_trouble: 0, long_running: 0, evicted: 0 },
});

/** Routes by the first matching substring, in insertion order. */
function stub(pages: [string, unknown][], status = 200) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (status !== 200) return new Response('{"detail":"sessions unavailable"}', { status });
    const hit = pages.find(([k]) => url.includes(k));
    return new Response(JSON.stringify(hit ? hit[1] : EMPTY), { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** The sessions list's own calls (the Score column reads Prompt Lens' scored sessions apart). */
function urls(f: ReturnType<typeof stub>): string[] {
  return f.mock.calls.map((c) => String(c[0])).filter((u) => !u.includes('/prompt-lens/'));
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <SessionsPage />
    </SWRConfig>,
  );
}

function chart(): SVGSVGElement {
  const svg = screen.getByRole('img', { name: S.list.chartAria }) as unknown as SVGSVGElement;
  svg.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 1000, height: 70, right: 1000, bottom: 70, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  return svg;
}

function drag(svg: Element, x0: number, x1: number) {
  fireEvent.mouseDown(svg, { clientX: x0, button: 0 });
  fireEvent.mouseMove(svg, { clientX: x1 });
  fireEvent.mouseUp(svg, { clientX: x1 });
}

const TWO = resp([
  group('tok9', 'gleb', [item(1), item(2), item(3)], { more: 211 }),
  group('tok2', 'alice', [item(4), item(5), item(6)], { more: 2 }),
]);

describe('sessions list page: Prompt Lens score column', () => {
  beforeEach(() => {
    mockPath = '/sessions';
    sessionStorage.clear();
    localStorage.clear();
    setAccessToken('admin-token');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    setAccessToken(null);
  });

  it('shows the overall of a scored session with a way into its scorecard, and a dash for the rest', async () => {
    const scored = {
      session_id: item(1).session_id, model: 'qwen', route: 'local', harness: 'claude_code', turns: 12,
      started: T0, last_at: T0 + 100, overall: 62, e: 48, c: 71, r: 74, confidence: 'medium', reasons: {}, score_version: 1,
    };
    stub([
      ['/api/admin/prompt-lens/sessions', { sessions: [scored], next_cursor: null }],
      ['/api/sessions?', resp([group('tok9', 'gleb', [item(1), item(2)])])],
    ]);
    renderPage();
    expect(await screen.findByRole('columnheader', { name: S.list.cols.score })).toBeTruthy(); // one group: open
    const rows = await screen.findAllByTestId('session-row');
    await waitFor(() => expect(within(rows[0]).getByTestId('session-score')).toHaveTextContent('62'));
    const link = within(rows[0]).getByRole('link', { name: S.list.openInLens });
    expect(link.getAttribute('href')).toBe(`/prompt-lens?view=sessions&session=${item(1).session_id}`);
    expect(within(rows[1]).getByTestId('score-pending')).toHaveAttribute('title', S.list.notScored);
    expect(within(rows[1]).queryByTestId('session-score')).toBeNull();
  });
});

describe('sessions list page', () => {
  beforeEach(() => {
    mockPath = '/sessions';
    sessionStorage.clear();
    localStorage.clear();
    setAccessToken('admin-token');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    setAccessToken(null);
  });

  it('renders the chart head, views with counts and groups with anatomy bars', async () => {
    const f = stub([['/api/sessions?', TWO]]);
    renderPage();
    expect(await screen.findByText(S.list.chartHead(7, S.list.rangeText['24h']))).toBeTruthy();
    expect(screen.getByText(S.list.chartHint)).toBeTruthy();
    const all = screen.getByTestId('view-all');
    expect(all.textContent).toContain(S.list.views.all.label);
    expect(all.textContent).toContain('812');
    expect(screen.getByTestId('view-dead_weight').textContent).toContain(S.list.views.dead_weight.small(37));
    expect(screen.getByTestId('view-biggest').textContent).toContain(S.list.views.biggest.small(50));
    const rows = screen.getAllByTestId('group-row');
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain('gleb');
    expect(rows[0].textContent).toContain(S.harness.claude_code);
    expect(rows[0].textContent).toContain('412 h total');
    expect(rows[0].textContent).toContain('9.1 B · 95%');
    expect(rows[0].querySelectorAll('[data-seg]')).toHaveLength(3);
    expect(screen.getByRole('columnheader', { name: S.list.firstCol.key })).toBeTruthy();
    expect(screen.queryAllByTestId('session-row')).toHaveLength(0);
    const u = urls(f)[0];
    expect(u).toContain('group=key');
    expect(u).toContain('tz_offset_min=');
  });

  it("expands a group, and Show N more fetches group_key with offset and appends", async () => {
    const f = stub([
      ['group_key=tok2', resp([group('tok2', 'alice', [item(7), item(8)], { more: 0 })])],
      ['/api/sessions?', TWO],
    ]);
    renderPage();
    const rows = await screen.findAllByTestId('group-row');
    fireEvent.click(within(rows[1]).getByRole('button', { name: /alice/ }));
    expect(screen.getAllByTestId('session-row')).toHaveLength(3);
    const row = screen.getAllByTestId('session-row')[0];
    expect(within(row).getByRole('link').getAttribute('href')).toContain('/sessions/detail?id=s_');
    expect(row.textContent).toContain('31 min · 4 agents · 2 compactions');
    fireEvent.click(screen.getByRole('button', { name: S.list.showMore(2, 'alice') }));
    await waitFor(() => expect(screen.getAllByTestId('session-row')).toHaveLength(5));
    const more = urls(f).find((x) => x.includes('group_key=tok2'))!;
    expect(more).toContain('offset=3');
    expect(more).toContain('limit=50');
    expect(screen.queryByRole('button', { name: S.list.showMore(2, 'alice') })).toBeNull();
  });

  it('brushing the chart refetches with since/until; Clear selection drops them', async () => {
    const f = stub([['/api/sessions?', TWO]]);
    renderPage();
    await screen.findAllByTestId('group-row');
    drag(chart(), 300, 600);
    await waitFor(() => expect(urls(f).some((x) => x.includes('since='))).toBe(true));
    const u = urls(f).find((x) => x.includes('since='))!;
    expect(u).toContain(`since=${T0 + 3600}`);
    expect(u).toContain(`until=${T0 + 3 * 3600}`);
    const n = f.mock.calls.length;
    fireEvent.click(screen.getByRole('button', { name: S.list.clearSelection }));
    await waitFor(() => expect(f.mock.calls.length).toBeGreaterThan(n));
    expect(urls(f).at(-1)).not.toContain('since=');
  });

  it('switching Group by / Where / a view refetches with that param', async () => {
    const f = stub([['/api/sessions?', TWO]]);
    renderPage();
    await screen.findAllByTestId('group-row');
    fireEvent.click(screen.getByRole('radio', { name: S.list.groupBy.harness }));
    await waitFor(() => expect(urls(f).some((x) => x.includes('group=harness'))).toBe(true));
    expect(screen.getByRole('columnheader', { name: S.list.firstCol.harness })).toBeTruthy();
    fireEvent.click(screen.getByRole('radio', { name: S.list.where.relayed }));
    await waitFor(() => expect(urls(f).some((x) => x.includes('where=relayed'))).toBe(true));
    fireEvent.click(screen.getByTestId('view-dead_weight'));
    await waitFor(() => expect(urls(f).some((x) => x.includes('view=dead_weight'))).toBe(true));
    fireEvent.change(screen.getByRole('combobox', { name: S.list.sortLabel }), { target: { value: 'duration' } });
    await waitFor(() => expect(urls(f).some((x) => x.includes('sort=duration'))).toBe(true));
  });

  it('debounces the search box before refetching with q', async () => {
    const f = stub([['/api/sessions?', TWO]]);
    renderPage();
    await screen.findAllByTestId('group-row');
    vi.useFakeTimers();
    fireEvent.change(screen.getByPlaceholderText(S.list.searchPlaceholder), { target: { value: 'gl' } });
    fireEvent.change(screen.getByPlaceholderText(S.list.searchPlaceholder), { target: { value: 'gleb' } });
    act(() => vi.advanceTimersByTime(299));
    expect(urls(f).some((x) => x.includes('q='))).toBe(false);
    act(() => vi.advanceTimersByTime(1));
    vi.useRealTimers();
    await waitFor(() => expect(urls(f).some((x) => x.includes('q=gleb'))).toBe(true));
    expect(urls(f).some((x) => x.includes('q=gl&'))).toBe(false);
  });

  it('opens the single group for a key holder and hides nothing else', async () => {
    setAccessToken(null);
    mockPath = '/sessions';
    sessionStorage.setItem('vw-sessions-token', 'jwt');
    sessionStorage.setItem('vw-sessions-exp', String(Date.now() + 60_000));
    const f = stub([['/api/sessions?', resp([group('tok1', 'alice', [item(1), item(2)])])]]);
    renderPage();
    expect(await screen.findAllByTestId('session-row')).toHaveLength(2);
    expect(screen.getByRole('img', { name: S.list.chartAria })).toBeTruthy();
    expect(screen.getByTestId('view-all')).toBeTruthy();
    expect(screen.getByText(S.pauses.surviveTitle)).toBeTruthy();
    const init = f.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer jwt');
  });

  it('shows the empty state with Clear selection when the brush has no sessions', async () => {
    const f = stub([
      ['since=', EMPTY],
      ['/api/sessions?', TWO],
    ]);
    renderPage();
    await screen.findAllByTestId('group-row');
    drag(chart(), 500, 520);
    expect(await screen.findByText(S.list.emptyFiltered)).toBeTruthy();
    expect(screen.getByTestId('view-all').textContent).toContain('0');
    const clears = screen.getAllByRole('button', { name: S.list.clearSelection });
    fireEvent.click(clears[clears.length - 1]);
    await waitFor(() => expect(screen.getAllByTestId('group-row')).toHaveLength(2));
    expect(screen.queryByTestId('chart-brush')).toBeNull();
    expect(screen.queryByText(S.list.emptyFiltered)).toBeNull();
    expect(urls(f).filter((x) => x.includes('since='))).toHaveLength(1);
  });

  it('shows the plain empty state when the range has no sessions at all', async () => {
    stub([['/api/sessions?', { ...EMPTY, histogram: hist([0, 0, 0, 0]), oldest_data_at: 1_700_000_000 }]]);
    renderPage();
    expect(await screen.findByText(S.list.empty)).toBeTruthy();
    expect(screen.getByText(/oldest/i)).toBeTruthy();
  });

  it('shows the pause summary collapsed under the chart, titled How long caches survive a pause', async () => {
    stub([['/api/sessions?', TWO]]);
    renderPage();
    const title = await screen.findByText(S.pauses.surviveTitle);
    const details = title.closest('details')!;
    expect(details).toBeTruthy();
    expect(details.open).toBe(false);
    const table = within(details).getByRole('table');
    expect(within(table).getByText(S.pauses.bucket['5-15m'])).toBeTruthy();
    expect(within(table).getAllByRole('row')).toHaveLength(6);
    expect(details.textContent).toContain(S.pauses.hint(3600, 540));
  });

  it('shows — relayed for relayed-only groups, — with the cause when unknown, and ≈ duration for saved', async () => {
    stub([
      [
        '/api/sessions?',
        resp([
          group('tok9', 'gleb', [item(1, { where: 'relayed', saved_vs_vanilla_s: null })], {
            where: { local: 0, relayed: 180, hybrid: 0 },
            saved_vs_vanilla_s: null,
            evicted_resumes: 12,
            evicted_lost_s: 540,
          }),
          group('tok7', 'farm-07', [item(2)], { harnesses: ['opencode'] }),
          group('tok8', 'laptop', [item(3, { saved_vs_vanilla_s: null })], {
            saved_vs_vanilla_s: null,
            where: { local: 3, relayed: 180, hybrid: 31 },
          }),
        ]),
      ],
    ]);
    renderPage();
    const rows = await screen.findAllByTestId('group-row');
    expect(rows[0].textContent).toContain(S.list.savedRelayed);
    expect(rows[2].textContent).toContain('180 relayed, 31 hybrid, 3 local');
    expect(rows[0].textContent).toContain(S.list.groupEvicted(12, 540));
    expect(rows[1].textContent).toContain('≈ 31 h');
    expect(rows[1].textContent).toContain(S.harness.opencode);
    const unknown = within(rows[2]).getByTitle(S.list.savedUnknown);
    expect(unknown.textContent).toBe('—');
    fireEvent.click(within(rows[0]).getByRole('button', { name: /gleb/ }));
    const row = screen.getAllByTestId('session-row')[0];
    expect(row.textContent).toContain(S.list.relayed);
    expect(row.textContent).toContain(S.list.savedRelayed);
  });

  it('renders a flat list with Group by None', async () => {
    const f = stub([
      ['group=none', resp(null, { sessions: [item(1), item(2)], total: 2 })],
      ['/api/sessions?', TWO],
    ]);
    renderPage();
    await screen.findAllByTestId('group-row');
    fireEvent.click(screen.getByRole('radio', { name: S.list.groupBy.none }));
    await waitFor(() => expect(screen.getAllByTestId('session-row')).toHaveLength(2));
    expect(screen.queryAllByTestId('group-row')).toHaveLength(0);
    expect(screen.getByRole('columnheader', { name: S.list.firstCol.none })).toBeTruthy();
    expect(urls(f).some((x) => x.includes('group=none'))).toBe(true);
  });

  it('shows the share served from cache, not the anatomy, and no % when the cache is unknown', async () => {
    stub([
      [
        '/api/sessions?',
        resp([
          group('tok1', 'mlx', [item(1, { cache_known: false, cache_share: null, cached_share: null })], {
            cached_share: null,
          }),
          group('tok2', 'alice', [item(2, { cache_share: 0.9 })]),
        ]),
      ],
    ]);
    renderPage();
    const rows = await screen.findAllByTestId('group-row');
    expect(rows[0].textContent).toContain('9.1 B');
    expect(rows[0].textContent).not.toContain('%');
    fireEvent.click(within(rows[0]).getByRole('button', { name: /mlx/ }));
    fireEvent.click(within(rows[1]).getByRole('button', { name: /alice/ }));
    const srows = screen.getAllByTestId('session-row');
    expect(srows[0].textContent).not.toContain('%');
    expect(srows[1].textContent).toContain('1 k · 90%');
  });

  it('keeps a typed trailing space in the search box', async () => {
    const f = stub([['/api/sessions?', TWO]]);
    renderPage();
    await screen.findAllByTestId('group-row');
    const box = screen.getByPlaceholderText(S.list.searchPlaceholder) as HTMLInputElement;
    vi.useFakeTimers();
    fireEvent.change(box, { target: { value: 'farm ' } });
    act(() => vi.advanceTimersByTime(300));
    vi.useRealTimers();
    await waitFor(() => expect(urls(f).some((x) => x.includes('q=farm'))).toBe(true));
    expect(box.value).toBe('farm ');
  });

  it("shows the API's 503 as unavailable without crashing", async () => {
    stub([], 503);
    renderPage();
    expect(await screen.findByText(S.list.unavailable)).toBeTruthy();
  });
});

describe('sessions list paging (verified finding #59)', () => {
  beforeEach(() => {
    mockPath = '/sessions';
    sessionStorage.clear();
    localStorage.clear();
    setAccessToken('admin-token');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    setAccessToken(null);
  });

  it('never shows a session twice when a later page overlaps the refreshed first page', async () => {
    stub([
      ['group_key=tok2', resp([group('tok2', 'alice', [item(6), item(7)], { more: 0 })])],
      ['/api/sessions?', TWO],
    ]);
    renderPage();
    const rows = await screen.findAllByTestId('group-row');
    fireEvent.click(within(rows[1]).getByRole('button', { name: /alice/ }));
    fireEvent.click(screen.getByRole('button', { name: S.list.showMore(2, 'alice') }));
    // item(6) is already on the first page: it must appear once
    await waitFor(() => expect(screen.getAllByTestId('session-row')).toHaveLength(4));
    const ids = screen.getAllByTestId('session-row').map((r) => r.getAttribute('data-sid'));
    expect(new Set(ids).size).toBe(ids.length);
  });
});
