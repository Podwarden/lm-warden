import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import type { TokenItem } from '@/components/tokens/token-row';
import {
  DEFAULT_LIST_PARAMS,
  SEARCH_DEBOUNCE_MS,
  listApiUrl,
  listParamsToSearch,
  parseListParams,
} from '@/components/tokens/list-params';

// replace() moves the fake URL like the real router does, so the page sees
// its own writes come back through useSearchParams.
const nav = vi.hoisted(() => ({ replace: vi.fn(), search: '', sp: new URLSearchParams() }));
function trackReplace() {
  nav.replace.mockImplementation((href: string) => {
    nav.search = href.includes('?') ? href.slice(href.indexOf('?') + 1) : '';
  });
}
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => {
    if (nav.sp.toString() !== nav.search) nav.sp = new URLSearchParams(nav.search);
    return nav.sp;
  },
  usePathname: () => '/tokens',
}));

import TokensPage from '@/app/tokens/page';

function item(i: number, name = `key-${i}`): TokenItem {
  return {
    id: `t${i}`, name, prefix: `vw_${String(i).padStart(5, '0')}`, preview: 'vw_',
    created_at: '2026-01-01 00:00:00', last_used_at: null, expires_at: null,
    rotated_at: null, rotated_from: null, successor_id: null, successor_deleted: false,
    is_expired: false, is_near_expiry: false, revoked_at: null, is_revoked: false,
    paused_at: null, is_paused: false, priority: 5,
    usage_24h: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
}

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

/** A fake GET /api/tokens over `server.total` rows (names from `names` when
 *  given, filtered by q; `near_expiry=1` keeps the first `nearExpiry`), plus
 *  DELETE, which shrinks the list by one. `hold(offset)` makes that page's
 *  response wait until the returned release() is called. */
const server = { total: 0, nearExpiry: 0, names: null as string[] | null };
const held = new Map<number, Promise<void>>();
function hold(offset: number): () => void {
  let release = () => {};
  held.set(offset, new Promise<void>((r) => { release = r; }));
  return () => release();
}

function installFetch() {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
    if (url === '/api/csrf') return json({ csrf: 'csrf' });
    if (init?.method === 'DELETE') {
      server.total -= 1;
      return new Response(null, { status: 204 });
    }
    if (url.startsWith('/api/tokens?')) {
      const q = new URL(url, 'http://x').searchParams;
      const limit = Number(q.get('limit'));
      const offset = Number(q.get('offset'));
      const needle = (q.get('q') ?? '').toLowerCase();
      if (held.has(offset)) await held.get(offset);
      if (q.get('near_expiry') === '1') {
        const n = server.nearExpiry;
        const items = Array.from({ length: Math.max(0, Math.min(limit, n - offset)) },
          (_, k) => ({ ...item(offset + k, `soon-${offset + k}`), is_near_expiry: true }));
        return json({ items, total: n, limit, offset, near_expiry: n });
      }
      const all = server.names
        ? server.names.map((n, i) => item(i, n)).filter((it) => it.name.toLowerCase().includes(needle))
        : null;
      const total = all ? all.length : server.total;
      const items = all
        ? all.slice(offset, offset + limit)
        : Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, k) => item(offset + k));
      return json({ items, total, limit, offset, near_expiry: server.nearExpiry });
    }
    return json({ detail: 'unexpected ' + url }, 404);
  });
  vi.stubGlobal('fetch', mock);
  return mock;
}

const listCalls = (m: ReturnType<typeof vi.fn>) =>
  m.mock.calls.map(([u]) => String(u)).filter((u) => u.startsWith('/api/tokens?'));

const lastListParams = (m: ReturnType<typeof vi.fn>) => {
  const calls = listCalls(m);
  return new URL(calls[calls.length - 1], 'http://x').searchParams;
};

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <TokensPage />
    </SWRConfig>,
  );
}

const header = (name: string) => screen.getByRole('columnheader', { name: new RegExp(`^${name}`) });
// Scoped queries: a page renders 50 rows of action buttons, and a
// document-wide getByRole('button') computes the accessibility of every one of
// them on each call -- cheap locally, tens of seconds on a loaded CI runner.
const sortButton = (name: string) =>
  within(screen.getByTestId('token-table').querySelector('thead') as HTMLElement).getByRole('button', { name });
const pagerButton = (name: string) =>
  within(screen.getByRole('navigation', { name: 'Token list pages' })).getByRole('button', { name });

describe('token list params', () => {
  it('omits every default from the URL and round-trips the rest', () => {
    expect(listParamsToSearch(DEFAULT_LIST_PARAMS)).toBe('');
    const p = {
      sort: 'usage_24h' as const, dir: 'asc' as const, page: 3, size: 25, q: 'bot', expiring: true,
    };
    const qs = listParamsToSearch(p);
    expect(qs).toBe('sort=usage_24h&dir=asc&page=3&size=25&q=bot&expiring=1');
    expect(parseListParams(new URLSearchParams(qs))).toEqual(p);
  });

  it('falls back to defaults for anything the server would 422', () => {
    const junk = new URLSearchParams('sort=hash&dir=up&page=0&size=7&q=' + 'x'.repeat(80));
    expect(parseListParams(junk)).toEqual({ ...DEFAULT_LIST_PARAMS, q: 'x'.repeat(64) });
    expect(parseListParams(new URLSearchParams('page=2.5')).page).toBe(1);
  });

  it('builds an explicit API request from the page number', () => {
    expect(listApiUrl({ ...DEFAULT_LIST_PARAMS, page: 3, size: 25 }))
      .toBe('/api/tokens?sort=created&dir=desc&limit=25&offset=50');
    expect(listApiUrl({ ...DEFAULT_LIST_PARAMS, q: 'a b' }))
      .toBe('/api/tokens?sort=created&dir=desc&limit=50&offset=0&q=a+b');
    expect(listApiUrl({ ...DEFAULT_LIST_PARAMS, expiring: true }))
      .toBe('/api/tokens?sort=created&dir=desc&limit=50&offset=0&near_expiry=1');
  });
});

// Explicit 20s per-test timeout (default is 5s): each test renders the full
// page with 50 rows, several times over, and a loaded CI runner has been seen
// taking 5-9s for what runs in ~100ms locally.
describe('tokens page: sort, paging, search', { timeout: 20_000 }, () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    nav.replace.mockReset();
    trackReplace();
    nav.search = '';
    held.clear();
    server.total = 120;
    server.nearExpiry = 0;
    server.names = null;
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('asks for the default page and marks Created as the descending sort', async () => {
    const f = installFetch();
    renderPage();
    await screen.findByText('key-0');
    expect(listCalls(f)[0]).toBe('/api/tokens?sort=created&dir=desc&limit=50&offset=0');
    expect(header('Created')).toHaveAttribute('aria-sort', 'descending');
    expect(within(header('Created')).getByTestId('sort-indicator').textContent).toBe('▼');
    expect(header('Name')).toHaveAttribute('aria-sort', 'none');
    expect(within(header('Name')).queryByTestId('sort-indicator')).toBeNull();
    // Actions is not sortable.
    const actions = screen.getByRole('columnheader', { name: 'Actions' });
    expect(actions).not.toHaveAttribute('aria-sort');
    expect(within(actions).queryByRole('button')).toBeNull();
    // Nothing to write back: the URL already says "defaults".
    expect(nav.replace).not.toHaveBeenCalled();
  });

  it('sorts by a clicked header, then flips it on a second click', async () => {
    const f = installFetch();
    renderPage();
    await screen.findByText('key-0');

    fireEvent.click(sortButton('Name'));
    await waitFor(() => expect(lastListParams(f).get('sort')).toBe('name'));
    expect(lastListParams(f).get('dir')).toBe('asc');
    expect(header('Name')).toHaveAttribute('aria-sort', 'ascending');
    expect(within(header('Name')).getByTestId('sort-indicator').textContent).toBe('▲');
    expect(header('Created')).toHaveAttribute('aria-sort', 'none');
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens?sort=name&dir=asc', { scroll: false });

    fireEvent.click(sortButton('Name'));
    await waitFor(() => expect(lastListParams(f).get('dir')).toBe('desc'));
    expect(header('Name')).toHaveAttribute('aria-sort', 'descending');
    expect(within(header('Name')).getByTestId('sort-indicator').textContent).toBe('▼');
    // desc is the default direction, so only the column stays in the URL.
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens?sort=name', { scroll: false });

    fireEvent.click(sortButton('Last 24h'));
    await waitFor(() => expect(lastListParams(f).get('sort')).toBe('usage_24h'));
    expect(lastListParams(f).get('dir')).toBe('desc');
  });

  it('pages with Previous / Next and shows a localized range', async () => {
    server.total = 12345;
    const f = installFetch();
    renderPage();
    await screen.findByText('key-0');
    expect(screen.getByTestId('token-page-range').textContent).toBe('1–50 of 12,345');
    expect(pagerButton('Previous')).toBeDisabled();
    expect(pagerButton('Next')).toBeEnabled();

    fireEvent.click(pagerButton('Next'));
    await screen.findByText('key-50');
    expect(lastListParams(f).get('offset')).toBe('50');
    expect(screen.getByTestId('token-page-range').textContent).toBe('51–100 of 12,345');
    expect(pagerButton('Previous')).toBeEnabled();
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens?page=2', { scroll: false });

    fireEvent.click(pagerButton('Previous'));
    await waitFor(() => expect(lastListParams(f).get('offset')).toBe('0'));
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens', { scroll: false });
  });

  it('disables Next on the last page', async () => {
    server.total = 12345;
    nav.search = 'page=247';
    installFetch();
    renderPage();
    await screen.findByText('key-12300');
    expect(screen.getByTestId('token-page-range').textContent).toBe('12,301–12,345 of 12,345');
    expect(pagerButton('Next')).toBeDisabled();
  });

  it('changes the page size and keeps the first row in view', async () => {
    server.total = 500;
    nav.search = 'page=3';
    const f = installFetch();
    renderPage();
    await screen.findByText('key-100');
    fireEvent.click(pagerButton('Rows per page'));
    fireEvent.mouseDown(screen.getByRole('option', { name: '25' }));
    await waitFor(() => expect(lastListParams(f).get('limit')).toBe('25'));
    // Row 101 was first on page 3 of 50; it is on page 5 of 25.
    expect(lastListParams(f).get('offset')).toBe('100');
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens?page=5&size=25', { scroll: false });
  });

  it('reads sort, direction, page, size and search from the URL', async () => {
    server.names = Array.from({ length: 80 }, (_, i) => `bot-${i}`);
    nav.search = 'sort=usage_24h&dir=asc&page=3&size=25&q=bot';
    const f = installFetch();
    renderPage();
    await screen.findByText('bot-50');
    expect(listCalls(f)[0]).toBe('/api/tokens?sort=usage_24h&dir=asc&limit=25&offset=50&q=bot');
    expect(header('Last 24h')).toHaveAttribute('aria-sort', 'ascending');
    expect(screen.getByLabelText('Search tokens')).toHaveValue('bot');
    expect(screen.getByTestId('token-page-range').textContent).toBe('51–75 of 80');
    expect(nav.replace).not.toHaveBeenCalled();
  });

  it('steps back a page when the last row of the last page is deleted', async () => {
    server.total = 51;
    nav.search = 'page=2';
    const f = installFetch();
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderPage();
    await screen.findByText('key-50');
    expect(screen.getByTestId('token-page-range').textContent).toBe('51–51 of 51');

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    // The revalidation of page 2 comes back empty; the page steps back to 1.
    await screen.findByText('key-0');
    expect(lastListParams(f).get('offset')).toBe('0');
    expect(screen.getByTestId('token-page-range').textContent).toBe('1–50 of 50');
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens', { scroll: false });
    vi.mocked(window.confirm).mockRestore();
  });

  it('revalidates the page in view in place after a change', async () => {
    server.total = 120;
    nav.search = 'page=2';
    const f = installFetch();
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    renderPage();
    await screen.findByText('key-50');
    const before = listCalls(f).length;
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]);
    await waitFor(() => expect(listCalls(f).length).toBeGreaterThan(before));
    expect(lastListParams(f).get('offset')).toBe('50');
    await waitFor(() =>
      expect(screen.getByTestId('token-page-range').textContent).toBe('51–100 of 119'));
    vi.mocked(window.confirm).mockRestore();
  });

  it('shows the expiry banner from near_expiry and filters to those keys on request', async () => {
    server.nearExpiry = 7;
    const f = installFetch();
    renderPage();
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toMatch(/expiring soon \(7\)/i);
    fireEvent.click(within(alert).getByRole('button', { name: /show them/i }));
    await screen.findByText('soon-0');
    expect(lastListParams(f).get('near_expiry')).toBe('1');
    expect(screen.getByTestId('token-page-range').textContent).toBe('1–7 of 7');
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens?expiring=1', { scroll: false });
    // A visible, removable chip; the banner stops offering the filter it shows.
    expect(screen.getByTestId('filter-chip').textContent).toMatch(/expiring within 30 days/i);
    expect(within(screen.getByRole('alert')).queryByRole('button')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /remove filter/i }));
    await screen.findByText('key-0');
    // The unfiltered page was cached from the first load; it shows at once
    // and is revalidated.
    await waitFor(() => expect(lastListParams(f).get('near_expiry')).toBeNull());
    expect(screen.queryByTestId('filter-chip')).toBeNull();
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens', { scroll: false });
  });

  it('says so when nothing expires soon under the filter, and clears it', async () => {
    server.nearExpiry = 0;
    nav.search = 'expiring=1';
    const f = installFetch();
    renderPage();
    await screen.findByText('No tokens expire within 30 days.');
    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
    await screen.findByText('key-0');
    await waitFor(() => expect(lastListParams(f).get('near_expiry')).toBeNull());
  });

  it('keeps the table and pager mounted, and focus on Next, while the next page loads', async () => {
    server.total = 500;
    const f = installFetch();
    renderPage();
    await screen.findByText('key-0');
    const release = hold(50);
    const next = pagerButton('Next');
    next.focus();
    fireEvent.click(next);
    await waitFor(() => expect(lastListParams(f).get('offset')).toBe('50'));
    // Still the old rows, dimmed and marked busy -- not a skeleton.
    const table = screen.getByTestId('token-table');
    expect(table).toHaveAttribute('aria-busy', 'true');
    expect(table.className).toMatch(/opacity-60/);
    expect(screen.getByText('key-0')).toBeInTheDocument();
    expect(pagerButton('Next')).toBe(next);
    expect(document.activeElement).toBe(next);

    release();
    await screen.findByText('key-50');
    expect(pagerButton('Next')).toBe(next);
    expect(document.activeElement).toBe(next);
    await waitFor(() => expect(screen.getByTestId('token-table')).toHaveAttribute('aria-busy', 'false'));
    expect(screen.getByTestId('token-table').className).not.toMatch(/opacity-60/);
  });

  it('follows the URL when it changes under it (the nav "Tokens" link resets the list)', async () => {
    const f = installFetch();
    const view = renderPage();
    await screen.findByText('key-0');
    fireEvent.click(sortButton('Name'));
    await waitFor(() => expect(lastListParams(f).get('sort')).toBe('name'));
    fireEvent.change(screen.getByLabelText('Search tokens'), { target: { value: 'key' } });

    // Navigating to a plain /tokens.
    nav.search = '';
    view.rerender(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <TokensPage />
      </SWRConfig>,
    );
    await waitFor(() => expect(lastListParams(f).get('sort')).toBe('created'));
    expect(lastListParams(f).get('dir')).toBe('desc');
    expect(header('Created')).toHaveAttribute('aria-sort', 'descending');
    expect(screen.getByLabelText('Search tokens')).toHaveValue('');
  });

  it('hides the banner when near_expiry is 0', async () => {
    installFetch();
    renderPage();
    await screen.findByText('key-0');
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('debounces search, resets to page 1 and writes q to the URL', async () => {
    server.names = ['prod-bot', 'staging-bot', 'other', ...Array.from({ length: 60 }, (_, i) => `x-${i}`)];
    nav.search = 'page=2';
    const f = installFetch();
    renderPage();
    await screen.findByText('x-47');
    const input = screen.getByLabelText('Search tokens');
    expect(input).toHaveAttribute('placeholder', 'Search by name');

    vi.useFakeTimers({ shouldAdvanceTime: false });
    fireEvent.change(input, { target: { value: 'b' } });
    fireEvent.change(input, { target: { value: 'bo' } });
    fireEvent.change(input, { target: { value: 'bot' } });
    await act(async () => { vi.advanceTimersByTime(SEARCH_DEBOUNCE_MS - 50); });
    expect(listCalls(f).some((u) => u.includes('q='))).toBe(false);
    await act(async () => { vi.advanceTimersByTime(60); });
    vi.useRealTimers();

    await screen.findByText('prod-bot');
    const withQ = listCalls(f).filter((u) => u.includes('q='));
    expect(withQ).toEqual(['/api/tokens?sort=created&dir=desc&limit=50&offset=0&q=bot']);
    expect(screen.getByTestId('token-page-range').textContent).toBe('1–2 of 2');
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens?q=bot', { scroll: false });
  });

  it('says so when a search matches nothing, and clears it', async () => {
    server.names = ['prod-bot', 'staging-bot'];
    nav.search = 'q=zzz';
    const f = installFetch();
    renderPage();
    await screen.findByText('No tokens match “zzz”.');
    expect(screen.queryByText(/no tokens yet/i)).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Clear search' }));
    await screen.findByText('prod-bot');
    expect(lastListParams(f).get('q')).toBeNull();
    expect(screen.getByLabelText('Search tokens')).toHaveValue('');
    expect(nav.replace).toHaveBeenLastCalledWith('/tokens', { scroll: false });
  });

  it('keeps the "no tokens yet" state for a truly empty list', async () => {
    server.total = 0;
    installFetch();
    renderPage();
    await screen.findByText(/no tokens yet/i);
    expect(screen.queryByText(/no tokens match/i)).toBeNull();
    expect(screen.queryByTestId('token-page-range')).toBeNull();
  });
});
