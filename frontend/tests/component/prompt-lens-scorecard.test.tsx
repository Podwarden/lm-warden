// The session scorecard and its pieces (analysis Task 2.5, spec 12.1; mockup
// scorecard.html): tiles with range, confidence, turns, the meaning of 100 and
// an inputs toggle; "not scored: <reason>" with no number (A-7); the model
// review apart from the overall (A-8); top 3 fixes; the turn strip; findings by
// layer; two tiles a row on a phone; key copy never says "cached"; no "cobs2".

import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

import { Scorecard } from '@/components/prompt-lens/sessions/scorecard';
import { ScoreTile } from '@/components/prompt-lens/sessions/score-tile';
import {
  inputsText,
  parseView,
  rangeText,
  savingText,
  scoreLine,
  sessionListUrl,
  scorecardUrl,
} from '@/lib/prompt-lens/scorecard';

import { mutedSession, notScored, scored, scorecardFixture, shortSession } from './prompt-lens-scorecard-fixture';

const wontfix = vi.hoisted(() => ({ set: vi.fn(async (_scope: string, _o: Record<string, unknown>) => true) }));
vi.mock('@/lib/prompt-lens/trends', async (orig) => ({ ...(await orig<typeof import('@/lib/prompt-lens/trends')>()), setWontfix: wontfix.set }));

afterEach(cleanup);

describe('Scorecard', () => {
  it('shows the facts, the four tiles with ranges and confidence, and the review apart', () => {
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={vi.fn()} />);
    const facts = screen.getByTestId('scorecard-facts');
    expect(facts).toHaveTextContent('fake-model');
    expect(facts).toHaveTextContent('24 turns');
    expect(facts).toHaveTextContent('18 min');
    const overall = screen.getByTestId('score-tile-overall');
    expect(within(overall).getByTestId('score-value')).toHaveTextContent('62');
    expect(overall).toHaveTextContent('range 55-68');
    expect(overall).toHaveTextContent('0.45 × Efficiency + 0.15 × Construction + 0.4 × Response');
    expect(within(screen.getByTestId('score-tile-E')).getByTestId('score-value')).toHaveTextContent('48');
    expect(screen.getByTestId('score-tile-E')).toHaveTextContent('40-57');
    expect(screen.getByTestId('score-tile-C')).toHaveTextContent('100 = no dead weight');
    expect(screen.getByTestId('score-tile-R')).toHaveTextContent('medium');
    // The review is its own tile, "a model's opinion", and not among the overall's inputs (A-8).
    const review = screen.getByTestId('score-tile-review');
    expect(review).toHaveTextContent("a model's opinion");
    expect(review).toHaveClass('border-dashed');
    expect(screen.getByTestId('score-from')).toHaveTextContent('from E, C, R');
    expect(overall.textContent).toContain('Model review not included');
  });

  it('a not-scored tile reads "not scored: <reason>" and shows no number (A-7)', () => {
    render(<Scorecard data={shortSession()} scope="key" onShowTurn={vi.fn()} />);
    const e = screen.getByTestId('score-tile-E');
    expect(e).toHaveAttribute('data-scored', 'false');
    expect(e).toHaveTextContent('not scored: 2 of 3 continuing turns');
    expect(within(e).queryByTestId('score-value')).toBeNull();
    expect(e.textContent).not.toMatch(/\d+\s*\/\s*100|\b0\b/);
    expect(screen.getByTestId('score-tile-overall')).toHaveTextContent('not scored: 1 of 2 layers scored');
    expect(within(screen.getByTestId('score-tile-overall')).queryByTestId('score-value')).toBeNull();
    expect(screen.getByTestId('score-tile-R')).toHaveTextContent('not scored: 3 of 5 assistant turns');
    expect(within(screen.getByTestId('score-tile-C')).getByTestId('score-value')).toHaveTextContent('90');
    expect(screen.getByTestId('no-fixes')).toBeInTheDocument();
    // Findings by layer say it too.
    expect(within(screen.getByTestId('findings-E')).getByText(/not scored: 2 of 3 continuing turns/)).toBeInTheDocument();
  });

  it('the inputs toggle shows the numbers a score is computed from', () => {
    render(<ScoreTile layer="E" score={scored('E', 48, { inputs: { ideal: 185000, lost: 96200 } })} scope="admin" />);
    expect(screen.queryByTestId('score-inputs')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'inputs' }));
    expect(screen.getByTestId('score-inputs')).toHaveTextContent('lost 96k of 185k reusable tokens · 24 turns');
    fireEvent.click(screen.getByRole('button', { name: 'hide inputs' }));
    expect(screen.queryByTestId('score-inputs')).toBeNull();
  });

  it('lists the top 3 fixes with rank, rule, saving per session and per 100, and the fix', () => {
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={vi.fn()} />);
    const fixes = screen.getAllByTestId('fix-card');
    expect(fixes.map((f) => f.getAttribute('data-rule'))).toEqual(['E-01', 'C-01', 'R-04']);
    expect(fixes[0]).toHaveTextContent('System prompt line 2 changes every');
    expect(screen.getAllByTestId('fix-saving')[0]).toHaveTextContent('61k tokens · 11 s per session');
    expect(screen.getAllByTestId('fix-saving')[0]).toHaveTextContent('6.1M tokens');
    expect(fixes[0]).toHaveTextContent('make the value stable');
    expect(fixes[2]).toHaveTextContent('search_docs');
    expect(fixes[2]).toHaveTextContent('medium confidence');
  });

  it('"Show turns" opens the request of that finding\'s first turn', () => {
    const open = vi.fn();
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={open} />);
    fireEvent.click(within(screen.getAllByTestId('fix-card')[0]).getByTestId('show-turns'));
    expect(open).toHaveBeenCalledWith('req-fake-0002');
    fireEvent.click(screen.getAllByTestId('turn-bar')[8]);
    expect(open).toHaveBeenLastCalledWith('req-fake-0009');
  });

  it('draws one bar per turn with the layer of its findings', () => {
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={vi.fn()} />);
    const bars = screen.getAllByTestId('turn-bar');
    expect(bars).toHaveLength(24);
    expect(bars[8]).toHaveTextContent('E');
    expect(bars[8]).toHaveTextContent('R');
    expect(bars[0].title).toContain('turn 1');
  });

  it('lists the findings by layer, collapsible, notices not counted', () => {
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={vi.fn()} />);
    const e = screen.getByTestId('findings-E');
    expect(within(e).getAllByTestId('finding-row').map((r) => r.getAttribute('data-rule'))).toEqual(['E-01', 'E-08']);
    fireEvent.click(within(e).getByRole('button', { name: /Efficiency/ }));
    expect(within(e).queryAllByTestId('finding-row')).toHaveLength(0);
    expect(within(screen.getByTestId('findings-R')).getAllByTestId('finding-row')).toHaveLength(1);
  });

  it('pins a secret in the prompt above the fixes as a notice (R-09)', () => {
    const n = {
      rule: 'R-09', layer: 'R' as const, turns: [6, 11], counts: { values: 3, rules: 1, info: 1 }, tool: null,
      cause: 'secret_in_tool_results', saving: { tokens: 0, prefill_s: null, decode_s: null, money: null },
      confidence: 'low' as const, muted: false, part: null, guard: { rules: ['fake_rule'], where: ['tool_result'] },
    };
    render(<Scorecard data={scorecardFixture({ notice: n })} scope="admin" onShowTurn={vi.fn()} />);
    expect(screen.getByTestId('secret-notice')).toHaveTextContent('Secret in the prompt');
  });

  it('the key holder never reads "cached" or "cobs2"; a card shows the money and its date', () => {
    const card = { provider: 'upstream.example', model: 'big', currency: 'USD', effective_from: '2026-11-21' };
    const data = scorecardFixture({
      cost: { ...scorecardFixture().cost, card, actual_input: 0.0166, saving: 0.0126, saving_pct: 76 },
    });
    data.top[0].saving.money = 0.0126;
    const { container } = render(<Scorecard data={data} scope="key" onShowTurn={vi.fn()} />);
    expect(container.textContent).not.toMatch(/cached/i);
    expect(container.textContent).not.toMatch(/cobs2/i);
    expect(container.textContent).toContain('reused from');
    expect(screen.getAllByTestId('fix-saving')[0]).toHaveTextContent('0.0126 USD');
    expect(screen.getByTestId('scorecard-cost')).toHaveTextContent('saves ≈ 76 %');
    expect(screen.getByTestId('scorecard-cost')).toHaveTextContent('effective 2026-11-21');
  });

  it('phone: two tiles a row, the overall and the review spanning both', () => {
    render(<Scorecard data={scorecardFixture()} scope="key" onShowTurn={vi.fn()} />);
    const grid = screen.getByTestId('score-tiles');
    expect(grid).toHaveClass('grid-cols-2');
    expect(screen.getByTestId('score-tile-overall')).toHaveClass('col-span-2');
    expect(screen.getByTestId('score-tile-review')).toHaveClass('col-span-2');
    expect(grid.className).not.toMatch(/(^|\s)grid-cols-[3-9](\s|$)/);
  });
});

describe('scorecard helpers', () => {
  it('scoreLine never turns a missing number into a zero (A-7)', () => {
    expect(scoreLine(scored('E', 48))).toBe('48');
    expect(scoreLine(notScored('E', '2 of 3 continuing turns'))).toBe('not scored: 2 of 3 continuing turns');
    expect(scoreLine({ layer: 'E', scored: true, turns: 1 })).toBe('not scored: not enough data');
    expect(scoreLine(undefined)).toBe('not scored');
    expect(rangeText(notScored('E', 'x'))).toBeNull();
  });

  it('builds the urls for the scope, and only the admin list takes key=', () => {
    expect(scorecardUrl('key', 's_x')).toBe('/api/me/prompt-lens/sessions/s_x/scorecard');
    expect(scorecardUrl('admin', 's_x')).toBe('/api/admin/prompt-lens/sessions/s_x/scorecard');
    expect(sessionListUrl('admin', { key: 'tok1' })).toContain('key=tok1');
    expect(sessionListUrl('key', { key: 'tok1' })).not.toContain('key=');
  });

  it('words for inputs and savings; unknown views fall back to requests', () => {
    expect(inputsText(scored('R', 74, { inputs: { turns: 24, bad: 5, completion: 4800, waste_out: 430 } }), 'admin')).toBe(
      '5 of 24 turns failed · 430 of 4.8k output tokens wasted',
    );
    expect(savingText({ saving: { tokens: 2100, prefill_s: null, decode_s: null, money: null } } as never, null)).toBe('2.1k tokens');
    expect(parseView('sessions')).toBe('sessions');
    expect(parseView('nope')).toBe('requests');
    expect(parseView(null)).toBe('requests');
  });
});


describe('won\'t fix on the scorecard (spec 11.4, D-3)', () => {
  it('a muted finding is dimmed, says who marked it and offers Undo; the tile says what it left out', () => {
    render(<Scorecard data={mutedSession()} scope="key" onShowTurn={vi.fn()} onChanged={vi.fn()} />);
    const row = within(screen.getByTestId('findings-E')).getAllByTestId('finding-row').find((r) => r.getAttribute('data-rule') === 'E-01')!;
    expect(row).toHaveAttribute('data-muted', 'true');
    expect(row).toHaveClass('opacity-60');
    expect(within(row).getByTestId('wontfix-muted')).toHaveTextContent("won't fix · by key holder");
    expect(within(row).getByTestId('wontfix-undo')).toBeInTheDocument();
    // Never in the top 3.
    expect(screen.getAllByTestId('fix-card').map((f) => f.getAttribute('data-rule'))).not.toContain('E-01');
    const tile = screen.getByTestId('score-tile-E');
    expect(within(tile).getByTestId('wontfix-note')).toHaveTextContent("1 finding marked won't fix is not counted · with them: 48");
    expect(within(screen.getByTestId('score-tile-overall')).getByTestId('wontfix-note')).toHaveTextContent('with them: 62');
    // A tile whose score no mark moved says nothing.
    expect(within(screen.getByTestId('score-tile-C')).queryByTestId('wontfix-note')).toBeNull();
  });

  it('says "by admin" when the admin marked it', () => {
    render(<Scorecard data={mutedSession('admin')} scope="key" onShowTurn={vi.fn()} onChanged={vi.fn()} />);
    expect(screen.getAllByTestId('wontfix-muted')[0]).toHaveTextContent('by admin');
  });

  it('Undo clears the mark for every version and this one, then reloads', async () => {
    wontfix.set.mockClear();
    const changed = vi.fn();
    render(<Scorecard data={mutedSession()} scope="key" onShowTurn={vi.fn()} onChanged={changed} />);
    fireEvent.click(screen.getAllByTestId('wontfix-undo')[0]);
    await waitFor(() => expect(changed).toHaveBeenCalled());
    expect(wontfix.set.mock.calls.map((c) => c[1])).toEqual([
      { rule: 'E-01', on: false, version: null, key: 'tok-fake-0001' },
      { rule: 'E-01', on: false, version: 'aaaa1111', key: 'tok-fake-0001' },
    ]);
  });

  it('marks a finding for every version, or for this version only', async () => {
    wontfix.set.mockClear();
    const changed = vi.fn();
    render(<Scorecard data={mutedSession()} scope="admin" onShowTurn={vi.fn()} onChanged={changed} />);
    const row = within(screen.getByTestId('findings-R')).getAllByTestId('finding-row')[0];
    fireEvent.click(within(row).getByTestId('wontfix-open'));
    expect(within(row).getByRole('button', { name: "Won't fix for every version" })).toBeInTheDocument();
    fireEvent.click(within(row).getByRole('button', { name: 'for v4 only' }));
    await waitFor(() => expect(changed).toHaveBeenCalledTimes(1));
    expect(wontfix.set).toHaveBeenLastCalledWith('admin', { rule: 'R-04', on: true, version: 'aaaa1111', key: 'tok-fake-0001' });
    fireEvent.click(within(row).getByTestId('wontfix-open'));
    fireEvent.click(within(row).getByRole('button', { name: "Won't fix for every version" }));
    await waitFor(() => expect(changed).toHaveBeenCalledTimes(2));
    expect(wontfix.set).toHaveBeenLastCalledWith('admin', { rule: 'R-04', on: true, version: null, key: 'tok-fake-0001' });
  });

  it('offers no "this version" choice for a session without a version, and Cancel closes the choices', () => {
    const data = scorecardFixture();
    data.session.key_id = 'tok-fake-0001';
    render(<Scorecard data={data} scope="admin" onShowTurn={vi.fn()} onChanged={vi.fn()} />);
    const row = within(screen.getByTestId('findings-C')).getAllByTestId('finding-row')[0];
    fireEvent.click(within(row).getByTestId('wontfix-open'));
    expect(within(row).queryByRole('button', { name: /only$/ })).toBeNull();
    fireEvent.click(within(row).getByRole('button', { name: 'Cancel' }));
    expect(within(row).queryByTestId('wontfix-choices')).toBeNull();
  });

  it('a secret in the prompt (R-09) cannot be marked; the admin needs the key; a read-only card has no menu', () => {
    const n = {
      rule: 'R-09', layer: 'R' as const, turns: [6], counts: { values: 3, rules: 1, info: 1 }, tool: null, cause: null,
      saving: { tokens: 0, prefill_s: null, decode_s: null, money: null }, confidence: 'low' as const, muted: false, part: null,
      guard: { rules: ['fake_rule'], where: ['tool_result'] },
    };
    const withNotice = scorecardFixture({ notice: n, findings: [...scorecardFixture().findings, n] });
    withNotice.session.key_id = 'tok-fake-0001';
    const { unmount } = render(<Scorecard data={withNotice} scope="admin" onShowTurn={vi.fn()} onChanged={vi.fn()} />);
    const r09 = screen.getAllByTestId('finding-row').find((r) => r.getAttribute('data-rule') === 'R-09')!;
    expect(within(r09).queryByTestId('wontfix-open')).toBeNull();
    unmount();
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={vi.fn()} onChanged={vi.fn()} />);
    expect(screen.queryAllByTestId('wontfix-open')).toHaveLength(0); // no key id on the card
    cleanup();
    render(<Scorecard data={scorecardFixture()} scope="key" onShowTurn={vi.fn()} />);
    expect(screen.queryAllByTestId('wontfix-open')).toHaveLength(0); // nothing to reload: read only
  });
});
