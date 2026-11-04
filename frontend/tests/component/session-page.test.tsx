import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

let mockId = 's_' + 'f'.repeat(32);
const mockReplace = vi.fn();
vi.mock('next/navigation', () => ({
  usePathname: () => '/sessions/detail',
  useRouter: () => ({ replace: mockReplace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(`id=${mockId}`),
}));

import Page from '@/app/sessions/detail/page';
import { setAccessToken } from '@/lib/auth-fetch';
import { S } from '@/lib/sessions/strings';
import { DEAD_WEIGHT, fixture, PAUSES, REPLAY } from './session-fixture';
import type { SessionResponse } from '@/lib/sessions/types';

function stub(body: unknown, status = 200) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Page />
    </SWRConfig>,
  );
}

async function ready(s: SessionResponse = fixture()) {
  stub(s);
  renderPage();
  await screen.findByTestId('session-timeline');
}

function fact(label: string): HTMLElement {
  return screen.getByTestId(`fact-${label}`);
}

describe('session page', () => {
  beforeEach(() => {
    mockId = 's_' + 'f'.repeat(32);
    mockReplace.mockReset();
    sessionStorage.clear();
    setAccessToken('admin');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    setAccessToken(null);
  });

  it('selecting a block selects its conversation in the tree and its row in the table', async () => {
    await ready();
    fireEvent.click(document.querySelector('rect[data-req="s1a"]')!);
    const tree = screen.getByTestId('conversation-tree');
    expect(within(tree).getByText('subagent #1').closest('[aria-selected="true"]')).toBeTruthy();
    const table = screen.getByTestId('turn-table');
    expect(table.querySelector('tr[data-req="s1a"][aria-selected="true"]')).toBeTruthy();
    expect(document.querySelector('rect[data-req="s1a"][data-selected="true"]')).toBeTruthy();
  });

  it('selecting a tree row filters the turn table and the tools panel to that conversation', async () => {
    await ready();
    fireEvent.click(within(screen.getByTestId('conversation-tree')).getByText('subagent #2'));
    const rows = screen.getByTestId('turn-table').querySelectorAll('tbody tr[data-req]');
    expect(Array.from(rows).map((r) => r.getAttribute('data-req'))).toEqual(['s2a', 's2b']);
    expect(screen.getByTestId('tools-panel').textContent).toContain(S.tools.none);
  });

  it('expands a turn to its message structure with the break index and cause', async () => {
    await ready();
    fireEvent.click(within(screen.getByTestId('conversation-tree')).getByText('main'));
    const row = screen.getByTestId('turn-table').querySelector('tr[data-req="m3"]')!;
    expect(row.textContent).toContain('62% · rewrite');
    fireEvent.click(within(row as HTMLElement).getByRole('button'));
    expect(screen.getByTestId('turn-shape').textContent).toContain(S.turn.broke(3, 'rewrite'));
  });

  it('labels inferred parents and relayed turns', async () => {
    await ready();
    expect(within(screen.getByTestId('conversation-tree')).getAllByText(S.tree.inferred).length).toBe(3);
    expect(screen.getAllByText(S.timeline.inferred).length).toBeGreaterThan(0);
    fireEvent.click(within(screen.getByTestId('conversation-tree')).getByText('main after compaction'));
    const row = screen.getByTestId('turn-table').querySelector('tr[data-req="p2"]')!;
    expect(row.textContent).toContain(S.turn.upstream);
  });

  it('shows the six facts with the three time figures', async () => {
    await ready();
    const head = screen.getByTestId('session-header');
    expect(head.textContent).toContain(S.header.title('alice', S.harness.opencode));
    expect(head.textContent).toContain('on 3 replicas');
    expect(fact('requests').textContent).toContain(S.header.subs.requests(5, 1));
    expect(fact('prompt').textContent).toContain(S.header.subs.prompt(91));
    expect(fact('served').textContent).toContain('≈ 1 h 48 min');
    expect(fact('served').textContent).toContain(S.header.subs.served);
    expect(screen.getByTestId('saved-vs-vanilla').textContent).toBe('≈ 19 min');
    expect(fact('saved').textContent).toContain(S.header.subs.saved);
    expect(fact('recomputed').textContent).toContain('≈ 6 min');
    expect(fact('recomputed').textContent).toContain(S.header.subs.recomputed);
  });

  it("shows — with the reason for a relayed-only session's saved figure", async () => {
    const s = fixture();
    s.totals = { ...s.totals, where: 'relayed', saved_vs_vanilla_s: null, cache_served_s: null, recomputed_s: null };
    await ready(s);
    const slot = screen.getByTestId('saved-vs-vanilla');
    expect(slot.textContent).toBe('—');
    expect(slot.getAttribute('title')).toBe(S.header.unknown.relayed);
  });

  it('names the cause of an unknown time figure: no cold rate, or a model no longer configured', async () => {
    const s = fixture();
    s.totals = { ...s.totals, saved_vs_vanilla_s: null, cache_served_s: null };
    await ready(s);
    expect(screen.getByTestId('saved-vs-vanilla').getAttribute('title')).toBe(S.header.unknown.rate);
    cleanup();
    s.totals = { ...s.totals, replicas: null };
    await ready(s);
    expect(screen.getByTestId('saved-vs-vanilla').getAttribute('title')).toBe(S.header.unknown.model);
  });

  it('shows the dead-weight headline, the four kinds, and — for an unknown kind', async () => {
    await ready();
    const panel = screen.getByTestId('dead-weight-panel');
    expect(within(panel).getByRole('heading', { name: S.deadWeight.title })).toBeTruthy();
    expect(panel.textContent).toContain('22%');
    expect(panel.textContent).toContain(S.deadWeight.of('2.4 k', '11 k'));
    expect(panel.textContent).toContain(S.deadWeight.kinds.unused_tools.sub(25, 31));
    for (const k of ['dup', 'boil', 'reason'] as const) expect(panel.textContent).toContain(S.deadWeight.kinds[k].label);
    const reason = within(panel).getByTestId('dw-reason');
    expect(reason.textContent).toContain('—');
    expect(within(panel).getByTestId('dw-unused_tools').textContent).toContain('≈ 1.5 k');
    expect(within(panel).getByTestId('dw-unused_tools').textContent).toContain('14%');
    expect(panel.textContent).not.toContain(S.deadWeight.measured(13, 13));
  });

  it('says measured on N of M turns when some turns lack dead weight', async () => {
    const s = fixture();
    s.dead_weight = { ...DEAD_WEIGHT, turns_measured: 9, prompt_tokens: 8000 };
    await ready(s);
    const panel = screen.getByTestId('dead-weight-panel');
    expect(panel.textContent).toContain(S.deadWeight.measured(9, 13));
    // the tokens line is over the measured turns, the same base as the share
    expect(panel.textContent).toContain(S.deadWeight.of('2.4 k', '8 k'));
  });

  it('shows dead weight as not measured, never 0%, when no turn was measured', async () => {
    const s = fixture();
    s.dead_weight = {
      ...DEAD_WEIGHT, total_tokens: null, share: null, per_turn: null, turns_measured: 0,
      kinds: { ...DEAD_WEIGHT.kinds, dup: { tokens: null, share: null, per_turn: null }, boil: { tokens: null, share: null, per_turn: null } },
    };
    await ready(s);
    const panel = screen.getByTestId('dead-weight-panel');
    expect(panel.textContent).toContain(S.deadWeight.notMeasured);
    expect(panel.textContent).not.toContain('0%');
  });

  it('lists five engines, marks LM Warden measured, notes disk loads are excluded', async () => {
    await ready();
    const panel = screen.getByTestId('engine-comparison');
    expect(within(panel).getByRole('heading', { name: S.engines.title })).toBeTruthy();
    const rows = within(panel).getAllByTestId(/^engine-/);
    expect(rows.map((r) => r.getAttribute('data-testid'))).toEqual([
      'engine-lm_warden', 'engine-vllm', 'engine-llamacpp', 'engine-mlx_lm', 'engine-omlx',
    ]);
    expect(rows[0].textContent).toContain(S.engines.names.lm_warden);
    expect(rows[0].textContent).toContain(S.engines.measured);
    expect(rows[0].textContent).toContain('≈ 6 min');
    expect(rows[2].textContent).toContain(S.engines.slots(4));
    expect(rows[3].textContent).toContain('≈ 1 h 39 min');
    expect(rows[4].textContent).toContain(S.engines.omlxSmall);
    expect(panel.textContent).toContain(S.engines.note);
    expect(panel.textContent).toContain(S.engines.scope);
    const widths = rows.map((r) => (r.querySelector('[data-bar]') as HTMLElement).style.width);
    expect(widths[3]).toBe('100%');
  });

  it('shows unknown for LM Warden on a cache-unknown session', async () => {
    const s = fixture();
    s.turns = s.turns.map((t) => ({ ...t, cached_tokens: null, cache_known: false }));
    s.totals = { ...s.totals, cache_served_s: null, saved_vs_vanilla_s: null, recomputed_s: null, cached_share: null, cache_share: null };
    s.conversations = s.conversations.map((c) => ({ ...c, cached_tokens: 0 }));
    s.engine_replay = {
      ...REPLAY,
      engines: REPLAY.engines.map((e) => (e.engine === 'lm_warden' ? { ...e, recompute_s: null, recompute_tokens: null } : e)),
    };
    await ready(s);
    expect(screen.getByTestId('engine-lm_warden').textContent).toContain(S.engines.unknown);
    expect(screen.getByTestId('engine-vllm').textContent).toContain('≈ 25 min');
    expect(fact('prompt').textContent).toContain(S.header.subs.promptUnknown);
    expect(fact('prompt').textContent).not.toContain('%');
    const tree = screen.getByTestId('conversation-tree');
    expect(within(tree).getByText('main').closest('li')!.textContent).not.toContain('%');
    const served = fact('served').querySelector('[title]')!;
    expect(served.textContent).toBe('—');
    expect(served.getAttribute('title')).toBe(S.header.unknown.cache);
  });

  it('shows pauses: summary, five buckets, and the disk-tier hint', async () => {
    await ready();
    const panel = screen.getByTestId('pauses-panel');
    expect(within(panel).getByRole('heading', { name: S.pauses.title })).toBeTruthy();
    expect(panel.textContent).toContain(S.pauses.summary(14, 5, 180));
    expect(within(panel).getAllByRole('row')).toHaveLength(6);
    expect(panel.textContent).toContain(S.pauses.hint(14_400, 120));
    expect(panel.textContent).toContain(S.pauses.unassessed(2));
  });

  it('says so when the session had no pause', async () => {
    const s = fixture();
    s.pauses = { ...PAUSES, resumes: 0, evicted: 0, unassessed: 0, lost_s: null, disk_tier_hint: null };
    await ready(s);
    const panel = screen.getByTestId('pauses-panel');
    expect(panel.textContent).toContain(S.pauses.none);
    expect(within(panel).queryByRole('table')).toBeNull();
  });

  it('draws a notch on evicted blocks and says evicted after a 23 min pause in the turn table', async () => {
    await ready();
    const notches = document.querySelectorAll('path[data-notch]');
    expect(notches).toHaveLength(1);
    expect(notches[0].getAttribute('data-req')).toBe('m4');
    fireEvent.click(within(screen.getByTestId('conversation-tree')).getByText('main'));
    expect(screen.getByTestId('turn-table').textContent).toContain(S.turn.evicted(1380));
    expect(S.turn.evicted(1380)).toBe('evicted after a 23 min pause');
  });

  it('expands a turn to its dead-weight breakdown', async () => {
    await ready();
    fireEvent.click(within(screen.getByTestId('conversation-tree')).getByText('main'));
    const table = screen.getByTestId('turn-table');
    fireEvent.click(within(table.querySelector('tr[data-req="m3"]') as HTMLElement).getByRole('button'));
    const shape = screen.getByTestId('turn-shape');
    expect(shape.textContent).toContain(S.turn.deadWeight({ unused_tools: 1200, dup: 600, boil: 200, reason: 100, total: 2100, share: 0.14 }));
    expect(S.turn.deadWeight({ unused_tools: 1200, dup: 600, boil: 200, reason: null, total: 2000, share: 0.14 })).toBe(
      'Dead weight ≈ 2.0k tokens (14%): unused tools ≈ 1.2k · repeated ≈ 600 · boilerplate ≈ 200',
    );
    fireEvent.click(within(table.querySelector('tr[data-req="m1"]') as HTMLElement).getByRole('button'));
    expect(screen.getByTestId('turn-shape').textContent).toContain(S.turn.deadWeightUnknown);
  });

  it('draws an anatomy bar on each conversation row and turn row', async () => {
    await ready();
    const tree = screen.getByTestId('conversation-tree');
    const convRows = tree.querySelectorAll('li[role="option"]');
    // "Whole session" + five conversations; every conversation row has a bar
    expect(Array.from(convRows).filter((li) => li.querySelector('[data-seg]'))).toHaveLength(5);
    const turnRows = screen.getByTestId('turn-table').querySelectorAll('tbody tr[data-req]');
    expect(Array.from(turnRows).every((r) => r.querySelector('[data-seg]'))).toBe(true);
    expect(screen.getByRole('columnheader', { name: S.turn.cols.anatomy })).toBeTruthy();
  });

  it('shows the unused-tools headline; a row shows its name when given, else family and #hash', async () => {
    await ready();
    const panel = screen.getByTestId('tools-panel');
    expect(panel.textContent).toContain('offered 31, called 6; 25 unused');
    expect(panel.textContent).toContain('browser_navigate');
    expect(panel.textContent).toContain('#ef56ab78');
    expect(panel.textContent).toContain(S.tools.costNote);
  });

  it('replaces an r_ ref with the canonical s_ id', async () => {
    mockId = 'r_' + 'a'.repeat(32);
    stub(fixture());
    renderPage();
    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/sessions/detail?id=s_' + 'f'.repeat(32)));
  });

  it('shows notRecordedYet for that 404, notFound for others', async () => {
    stub({ detail: 'not recorded yet' }, 404);
    renderPage();
    expect(await screen.findByText(S.page.notRecordedYet)).toBeTruthy();
    cleanup();
    stub({ detail: 'session not found' }, 404);
    renderPage();
    expect(await screen.findByText(S.page.notFound)).toBeTruthy();
  });
});

describe('session page — relayed style (final review I2)', () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    setAccessToken(null);
  });

  it('draws relayed blocks hatched (no share fill class) and badges a relayed conversation', async () => {
    setAccessToken('admin');
    const s = fixture();
    s.conversations = s.conversations.map((c) => (c.id === 'c-comp' ? { ...c, relayed: true } : c));
    stub(s);
    renderPage();
    await screen.findByTestId('session-timeline');
    const rel = document.querySelector('rect[data-req="p2"]')!;
    expect(rel.getAttribute('fill')).toBe('url(#relayed-hatch)');
    expect(rel.getAttribute('class') ?? '').not.toMatch(/fill-(emerald|amber|rose|slate)-/);
    const local = document.querySelector('rect[data-req="m1"]')!;
    expect(local.getAttribute('class')).toContain('fill-emerald-500');
    const tree = screen.getByTestId('conversation-tree');
    expect(within(tree).getByText('main after compaction').closest('li')!.textContent).toContain(S.timeline.relayed);
  });
});

describe('session page polling (verified finding #60)', () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
    setAccessToken(null);
  });

  it('keeps refreshing a quiet session slowly instead of stopping', async () => {
    setAccessToken('admin');
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const f = stub(fixture()); // last_at = 47 s after epoch: long quiet
    renderPage();
    await screen.findByTestId('session-timeline');
    const n = f.mock.calls.length;
    await vi.advanceTimersByTimeAsync(31_000);
    expect(f.mock.calls.length).toBeGreaterThan(n);
  });
});
