// The Trends view (analysis Task 3.4, spec 13.3; mockup trends.html): the chart
// with version markers and a gap (never a 0) for a bucket under three scored
// sessions, the "Did the last change help?" card with its verdict copy, findings
// by version, and no fleet figure for a key holder (D-4).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

import { CompareCard } from '@/components/prompt-lens/trends/compare-card';
import { TrendChart } from '@/components/prompt-lens/trends/trend-chart';
import { TrendsView } from '@/components/prompt-lens/trends/trends-view';
import { clearKeyToken, storeKeyToken } from '@/lib/sessions/key-token';
import {
  bucketLabel,
  compareUrl,
  diffText,
  ruleStatus,
  trendsUrl,
  verdictText,
  verdictTone,
  type Compare,
  type Trends,
  type VersionInfo,
  type Verdict,
} from '@/lib/prompt-lens/trends';

const DAY = 86_400;
const T0 = 1_700_438_400; // a Monday

const med = (o: number | null) => ({ overall: o, E: o === null ? null : o - 3, C: o === null ? null : o + 5, R: o === null ? null : o + 8 });
const ver = (id: string, label: string, since: number, n: number, rules: Record<string, number>, overall: number, extra: Partial<VersionInfo> = {}): VersionInfo => ({
  id, label, since, last_at: since + DAY, n, medians: med(overall), rules, resolved: [], new: [], ...extra,
});

function trends(over: Partial<Trends> = {}): Trends {
  return {
    by: 'week',
    model: 'fake-model',
    models: ['fake-model', 'other-model'],
    points: [
      { bucket: '2023-11-20', n: 9, medians: med(47), band: [40, 55], start: T0 },
      { bucket: '2023-11-27', n: 7, medians: med(52), band: [45, 60], start: T0 + 7 * DAY },
      { bucket: '2023-12-04', n: 2, medians: med(null), band: null, start: T0 + 14 * DAY },
      { bucket: '2023-12-11', n: 12, medians: med(62), band: [55, 68], start: T0 + 21 * DAY },
    ],
    versions: [
      ver('aaaa1111', 'v1', T0, 9, { 'E-01': 9, 'E-02': 3, 'C-01': 9 }, 47),
      ver('bbbb2222', 'v2', T0 + 7 * DAY, 7, { 'E-01': 6, 'C-01': 7 }, 52, { resolved: ['E-02'] }),
      ver('cccc3333', 'v3', T0 + 21 * DAY, 12, { 'E-01': 11, 'C-01': 4, 'R-04': 5 }, 62, { new: ['R-04'] }),
    ],
    weights: { E: 0.45, C: 0.15, R: 0.4 },
    score_version: 1,
    ...over,
  };
}

function compare(verdict: Verdict, over: Partial<Compare> = {}): Compare {
  return {
    model: 'fake-model', a: 'bbbb2222', b: 'cccc3333', a_label: 'v2', b_label: 'v3', n_a: 31, n_b: 23,
    diffs: { overall: 11, E: 3, C: 24, R: 6 }, verdict, like_for_like: true, note: null, resolved: ['E-02'], new: [], p: 0.004, ...over,
  };
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
let urls: string[] = [];
function stub(t: Trends | null = trends(), c: Compare | null = compare('better_clear')) {
  urls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input);
      urls.push(u);
      if (u.includes('/trends')) return t ? json(t) : json({ detail: 'x' }, 500);
      if (u.includes('/compare')) return c ? json(c) : json({ detail: 'none' }, 404);
      return json({}, 404);
    }),
  );
}

beforeEach(() => storeKeyToken('fake-key-token', 3600));
afterEach(() => {
  cleanup();
  clearKeyToken();
  vi.unstubAllGlobals();
});

describe('TrendChart', () => {
  it('marks each version after the first with a dashed vertical marker and its label', () => {
    const t = trends();
    render(<TrendChart points={t.points} versions={t.versions} by="week" />);
    const marks = screen.getAllByTestId('version-marker');
    expect(marks.map((m) => m.getAttribute('data-version'))).toEqual(['bbbb2222', 'cccc3333']);
    expect(marks[0]).toHaveTextContent('v2');
    expect(marks[1]).toHaveTextContent('v3');
    expect(screen.getByTestId('trend-chart')).toHaveAttribute('aria-label', expect.stringContaining('overall 47 at the start, 62 at the end'));
  });

  it('an unscored bucket is a gap with a "not scored" note, never a zero', () => {
    const t = trends();
    render(<TrendChart points={t.points} versions={t.versions} by="week" />);
    const gap = screen.getByTestId('trend-gap');
    expect(gap).toHaveAttribute('data-bucket', '2023-12-04');
    expect(gap.querySelector('title')?.textContent).toBe('4 Dec: not scored (2 sessions, 3 needed)');
    const overall = screen.getByTestId('trend-line-overall');
    // Three scored points, and the line is broken at the gap: a run of two, then a lone point.
    expect(overall.querySelectorAll('circle')).toHaveLength(3);
    expect(overall.querySelectorAll('path')).toHaveLength(1);
    const ys = Array.from(overall.querySelectorAll('circle')).map((c) => Number(c.getAttribute('cy')));
    expect(Math.min(...ys)).toBeGreaterThan(16);
    expect(Math.max(...ys)).toBeLessThan(264); // none sits on the zero line
  });

  it('draws the 25th-75th band for runs of scored points, and the four lines', () => {
    const t = trends();
    render(<TrendChart points={t.points} versions={t.versions} by="week" />);
    expect(screen.getAllByTestId('trend-band')).toHaveLength(1);
    for (const m of ['overall', 'E', 'C', 'R']) expect(screen.getByTestId(`trend-line-${m}`)).toBeInTheDocument();
  });

  it('end labels of lines that finish close together do not overlap', () => {
    const t = trends();
    render(<TrendChart points={t.points} versions={t.versions} by="week" />);
    const ys = ['overall', 'E', 'C', 'R']
      .map((m) => Number(screen.getByTestId(`trend-line-${m}`).querySelector('text')?.getAttribute('y')))
      .sort((a, b) => a - b);
    for (let i = 1; i < ys.length; i++) expect(ys[i] - ys[i - 1]).toBeGreaterThanOrEqual(14);
  });

  it('a bucket that is not a date is labelled as it is, never "NaN"', () => {
    const t = trends();
    const odd = t.points.map((p, i) => ({ ...p, bucket: `w${i}` }));
    const { container } = render(<TrendChart points={odd} versions={t.versions} by="week" />);
    expect(container.textContent).not.toMatch(/NaN|undefined/);
    expect(container.textContent).toContain('w0');
  });

  it('by version has no boundary markers (each bucket is a version)', () => {
    const t = trends();
    render(<TrendChart points={t.points} versions={t.versions} by="version" />);
    expect(screen.queryAllByTestId('version-marker')).toHaveLength(0);
  });
});

describe('verdict copy', () => {
  const cases: [Verdict, string, string][] = [
    ['better_clear', 'Better, clear · 23 vs 31 sessions', 'ok'],
    ['better_unclear', 'Better, not yet clear · 23 vs 31 sessions', 'wait'],
    ['no_change', 'No change · 23 vs 31 sessions', 'wait'],
    ['worse_unclear', 'Worse, not yet clear · 23 vs 31 sessions', 'wait'],
    ['worse_clear', 'Worse, clear · 23 vs 31 sessions', 'bad'],
    ['not_enough', 'Not enough sessions yet · 23 vs 31 sessions', 'wait'],
  ];
  it.each(cases)('%s reads "%s"', (v, text, tone) => {
    expect(verdictText({ verdict: v, n_a: 31, n_b: 23 })).toBe(text);
    expect(verdictTone(v)).toBe(tone);
  });

  it('shows the verdict, the median differences and what stopped firing on the card', async () => {
    stub();
    render(<CompareCard scope="key" keyId={null} trends={trends()} />);
    const v = await screen.findByTestId('verdict');
    expect(v).toHaveTextContent('Better, clear · 23 vs 31 sessions');
    expect(v).toHaveAttribute('data-verdict', 'better_clear');
    const diffs = screen.getByTestId('compare-diffs');
    expect(diffs).toHaveTextContent('Overall+1152 → 62');
    expect(diffs).toHaveTextContent('Construction+24');
    expect(screen.getByTestId('compare-rules')).toHaveTextContent('Stopped firing: E-02 Tools sent in a different order');
    expect(urls.some((u) => u.includes('a=bbbb2222') && u.includes('b=cccc3333') && !u.includes('key='))).toBe(true);
  });

  it('says so when the sides are not like for like, and shows no numbers with too few sessions (A-7)', async () => {
    stub(trends(), compare('not_enough', { n_b: 4, diffs: { overall: null, E: null, C: null, R: null }, like_for_like: false, note: '4 of 5 scored sessions on one side; the sessions come from different harnesses' }));
    render(<CompareCard scope="key" keyId={null} trends={trends()} />);
    expect(await screen.findByTestId('verdict')).toHaveTextContent('Not enough sessions yet');
    expect(screen.queryByTestId('compare-diffs')).toBeNull();
    expect(screen.getByTestId('compare-note')).toHaveTextContent('Not like for like: 4 of 5 scored sessions on one side; the sessions come from different harnesses.');
  });

  it('with one version there is nothing to compare', () => {
    stub();
    render(<CompareCard scope="key" keyId={null} trends={trends({ versions: [trends().versions[0]] })} />);
    expect(screen.getByTestId('compare-none')).toBeInTheDocument();
    expect(urls).toHaveLength(0);
  });

  it('lets the reader pick two versions', async () => {
    stub();
    render(<CompareCard scope="key" keyId={null} trends={trends()} />);
    await screen.findByTestId('verdict');
    fireEvent.change(screen.getByLabelText('Earlier version'), { target: { value: 'aaaa1111' } });
    await waitFor(() => expect(urls.some((u) => u.includes('a=aaaa1111') && u.includes('b=cccc3333'))).toBe(true));
  });
});

describe('TrendsView', () => {
  it('renders the chart, the card and the findings by version with their status', async () => {
    stub();
    render(<TrendsView scope="key" />);
    expect(await screen.findByTestId('trend-chart')).toBeInTheDocument();
    expect(screen.getByTestId('compare-card')).toBeInTheDocument();
    const rows = screen.getAllByTestId('version-rule-row');
    const by = Object.fromEntries(rows.map((r) => [r.getAttribute('data-rule'), r.textContent ?? '']));
    expect(by['E-01']).toContain('100 %'); // 9 of 9 on v1
    expect(by['E-02']).toContain('33 %');
    expect(by['E-02']).toContain('resolved in v2');
    expect(by['R-04']).toContain('42 %'); // 5 of 12 on v3
    expect(by['R-04']).toContain('open');
    expect(by['C-01']).toContain('improved'); // 100 % then 33 %
    expect(screen.getByTestId('findings-by-version')).toHaveTextContent('share of sessions where the rule fired');
  });

  it('counts the sessions and says how many sit in buckets too small to score', async () => {
    stub();
    render(<TrendsView scope="key" />);
    await screen.findByTestId('trend-chart');
    expect(screen.getByText(/28 scored sessions · 2 in buckets too small to score/)).toBeInTheDocument();
  });

  it('switches by day, week and version, and the model', async () => {
    stub();
    render(<TrendsView scope="key" />);
    await screen.findByTestId('trend-chart');
    fireEvent.click(screen.getByRole('button', { name: 'day' }));
    await waitFor(() => expect(urls.some((u) => u.includes('by=day'))).toBe(true));
    fireEvent.click(screen.getByRole('button', { name: 'version' }));
    await waitFor(() => expect(urls.some((u) => u.includes('by=version'))).toBe(true));
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'other-model' } });
    await waitFor(() => expect(urls.some((u) => u.includes('model=other-model'))).toBe(true));
  });

  it('an empty answer says no session is scored yet', async () => {
    stub(trends({ points: [], versions: [] }));
    render(<TrendsView scope="key" />);
    expect(await screen.findByTestId('trends-empty')).toHaveTextContent('No scored sessions yet');
  });

  it('a failed load says so', async () => {
    stub(null);
    render(<TrendsView scope="key" />);
    expect(await screen.findByTestId('trends-error')).toBeInTheDocument();
  });

  it('a key holder reads only their own route: no key=, no fleet call, no rank or leaderboard text', async () => {
    stub();
    const { container } = render(<TrendsView scope="key" />);
    await screen.findByTestId('trend-chart');
    await screen.findByTestId('verdict');
    expect(urls.every((u) => u.startsWith('/api/me/prompt-lens/') && !u.includes('key=') && !u.includes('fleet'))).toBe(true);
    expect(container.textContent).not.toMatch(/\brank|fleet|leaderboard|cobs2|other keys/i);
    expect(container.textContent).toContain('Only your key');
  });

  it('the admin reads one key: with none picked it asks for one and fetches nothing', () => {
    stub();
    render(<TrendsView scope="admin" keyId={null} />);
    expect(screen.getByTestId('trends-pick-key')).toHaveTextContent('Trends are per key');
    expect(urls).toHaveLength(0);
  });
});

describe('trends helpers', () => {
  it('builds the urls: only the admin names a key', () => {
    expect(trendsUrl('key', { by: 'week', tz: 60 })).toBe('/api/me/prompt-lens/trends?by=week&tz=60&since=0');
    expect(trendsUrl('key', { by: 'week', key: 'tok1' })).not.toContain('key=');
    expect(trendsUrl('admin', { by: 'day', key: 'tok1', model: 'm' })).toBe('/api/admin/prompt-lens/trends?key=tok1&by=day&model=m&tz=0&since=0');
    expect(compareUrl('admin', { key: 'tok1', a: 'x', b: 'y' })).toContain('/api/admin/prompt-lens/compare?key=tok1&a=x&b=y');
    expect(compareUrl('key', { key: 'tok1', a: 'x', b: 'y' })).not.toContain('key=');
  });

  it('words for differences, buckets and rule status', () => {
    expect(diffText(11)).toBe('+11');
    expect(diffText(-3)).toBe('-3');
    expect(diffText(0)).toBe('0');
    expect(diffText(null)).toBe('—');
    expect(bucketLabel({ bucket: '2023-11-20', n: 1, medians: med(1), start: 0 }, 'week')).toBe('20 Nov');
    expect(bucketLabel({ bucket: 'aaaa1111', label: 'v4', n: 1, medians: med(1), start: 0 }, 'version')).toBe('v4');
    const vs = trends().versions;
    expect(ruleStatus('E-02', vs)).toEqual({ text: 'resolved in v2', tone: 'ok' });
    expect(ruleStatus('R-04', vs)).toEqual({ text: 'open', tone: 'bad' });
    expect(ruleStatus('C-01', vs).text).toBe('improved');
    expect(within(document.body).queryByText('x')).toBeNull();
  });
});
