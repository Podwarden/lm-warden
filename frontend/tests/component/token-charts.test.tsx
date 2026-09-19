import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react';
import { niceMax, yTicks, xTicks, nearestBin, hatchSpec } from '@/components/tokens/detail/time-chart';
import { UsageCharts } from '@/components/tokens/detail/usage-charts';
import { fillBins } from '@/lib/token-series';
import { NOW, seriesFixture } from './token-fixtures';

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 600 });
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
});
afterAll(() => {
  delete (HTMLElement.prototype as unknown as { clientWidth?: number }).clientWidth;
  vi.unstubAllGlobals();
});
afterEach(() => cleanup());

describe('chart helpers (mockup lineChart())', () => {
  it('niceMax / ticks', () => {
    expect([0, 0.9, 1.9, 2.1, 3, 120, 7_600].map(niceMax)).toEqual([1, 1, 2, 2.5, 5, 200, 10_000]);
    expect(yTicks(200)).toEqual([0, 50, 100, 150, 200]);
    expect(xTicks(0, 100)).toEqual([0, 20, 40, 60, 80, 100]);
  });
  it('nearestBin picks the closest bin centre', () => {
    const bins = fillBins(seriesFixture());
    const b = nearestBin(bins, 30, bins[3].minute + 14);
    expect(b?.minute).toBe(bins[3].minute);
    expect(nearestBin([], 30, 5)).toBeNull();
  });
  it('hatch covers the part of the period before latency_since', () => {
    const since = Date.UTC(2026, 8, 18, 0, 0) / 1000;
    expect(hatchSpec(since / 60 - 100, since / 60 + 50, since, NOW)).toEqual({ toMin: since / 60, label: 'Not recorded before Sep 18' });
    expect(hatchSpec(since / 60 + 1, since / 60 + 50, since, NOW)).toBeNull();
    // never recorded: the whole period is hatched, dated today
    expect(hatchSpec(10, 20, null, NOW)).toEqual({ toMin: 20, label: 'Not recorded before Sep 18' });
  });
});

// Explicit 20s per-test timeout (default is 5s): each test mounts four
// recharts panels in jsdom, which a loaded CI runner has taken ~4.4s for.
describe('UsageCharts', { timeout: 20_000 }, () => {
  it('renders the four panels with the mockup copy', () => {
    render(<UsageCharts series={seriesFixture()} nowSec={NOW} />);
    for (const t of ['Tokens per minute', 'Requests per minute', 'Queue wait', 'Latency']) {
      expect(screen.getByRole('heading', { name: t })).toBeInTheDocument();
    }
    expect(screen.getByText('Average per minute within each bin, idle minutes counted as zero. Hover for the busiest minute.')).toBeInTheDocument();
    expect(screen.getAllByText('Per-request timings are kept from Sep 18 onwards, for 30 days.')).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Prefill' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'First token' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Median')).toBeInTheDocument();
    expect(screen.getByText('95th percentile')).toBeInTheDocument();
  });

  it('does not clip the line to the exact plot rect (mockup: a value-0 bin draws full stroke width)', () => {
    render(<UsageCharts series={seriesFixture()} nowSec={NOW} />);
    const panel = screen.getByTestId('chart-tokens');
    const rect = panel.querySelector('.recharts-line clipPath rect');
    expect(rect).not.toBeNull();
    // Plot height is 210 - 8 - 24 = 178px; the YAxis must not clip <Line> to
    // that exact rect (allowDataOverflow on YAxis would), or a 1.8px stroke
    // sitting on the y=0 baseline is drawn at half width.
    expect(Number(rect!.getAttribute('height'))).not.toBe(178);
  });

  it('still draws the y axis and grid for a period wholly before latency_since (mockup ?custom)', () => {
    // Sep 12 09:00 → Sep 14 18:30: every timing value is null.
    const from = Date.UTC(2026, 8, 12, 9, 0) / 60000;
    const to = Date.UTC(2026, 8, 14, 18, 30) / 60000;
    const s = seriesFixture({ from_minute: from, to_minute: to, bin_minutes: 10, bins: [] });
    render(<UsageCharts series={s} nowSec={NOW} />);
    for (const name of ['queue', 'latency']) {
      const panel = screen.getByTestId(`chart-${name}`);
      const texts = [...panel.querySelectorAll('text')].map((t) => t.textContent);
      // mockup niceMax(0) = 1 → secs() ticks
      for (const t of ['0 ms', '250 ms', '500 ms', '750 ms', '1.0 s']) expect(texts).toContain(t);
      // (recharts' ReferenceArea — the hatch — does not render under jsdom
      // at all, with or without data; the hatch is checked in the browser.)
      expect(panel.querySelectorAll('.recharts-reference-line').length).toBe(5);
      // No fake data:
      const curves = panel.querySelectorAll('.recharts-line-curve');
      expect(curves.length).toBe(2);                        // p95 + p50 are mounted…
      for (const c of curves) expect(c.getAttribute('d') ?? '').toBe('');   // …but draw nothing
    }
  });

  it('uses the other footnote once the whole period is recorded', () => {
    const s = seriesFixture();
    render(<UsageCharts series={{ ...s, latency_since: s.from_minute * 60 - 3600 }} nowSec={NOW} />);
    expect(screen.getAllByText('Median solid, 95th percentile dashed. Bins with no requests are left empty.')).toHaveLength(2);
  });

  it('shows the sampling line only when stride > 1', () => {
    const { unmount } = render(<UsageCharts series={seriesFixture()} nowSec={NOW} />);
    expect(screen.queryByText(/^Timings from every/)).toBeNull();
    unmount();
    render(<UsageCharts series={seriesFixture({ timing_sample: { total: 99_624, used: 49_812, stride: 2 } })} nowSec={NOW} />);
    expect(screen.getAllByText('Timings from every 2nd request (49,812 of 99,624).')).toHaveLength(2);
  });

  it('switches series and shows the mockup tooltip rows on hover', () => {
    render(<UsageCharts series={seriesFixture()} nowSec={NOW} />);
    fireEvent.click(screen.getByRole('button', { name: 'Generation' }));
    expect(screen.getByRole('button', { name: 'Generation' })).toHaveAttribute('aria-pressed', 'true');

    const panel = screen.getByTestId('chart-tokens');
    const hit = within(panel).getByTestId('chart-hit');
    vi.spyOn(hit, 'getBoundingClientRect').mockReturnValue({ left: 52, top: 8, width: 540, height: 178, right: 592, bottom: 186, x: 52, y: 8, toJSON: () => ({}) } as DOMRect);
    fireEvent.pointerMove(hit, { clientX: 591 });
    const tip = within(panel).getByRole('tooltip');
    expect(tip).toHaveTextContent('Generation / min');
    expect(tip).toHaveTextContent('Busiest minute');
    expect(tip.textContent).toContain(' – ');
    fireEvent.pointerLeave(hit);
    expect(within(panel).queryByRole('tooltip')).toBeNull();
  });

  it('timing tooltips show the request count n', () => {
    render(<UsageCharts series={seriesFixture()} nowSec={NOW} />);
    const panel = screen.getByTestId('chart-queue');
    const hit = within(panel).getByTestId('chart-hit');
    vi.spyOn(hit, 'getBoundingClientRect').mockReturnValue({ left: 52, top: 8, width: 540, height: 178, right: 592, bottom: 186, x: 52, y: 8, toJSON: () => ({}) } as DOMRect);
    fireEvent.pointerMove(hit, { clientX: 300 });
    const tip = within(panel).getByRole('tooltip');
    expect(tip).toHaveTextContent('Median');
    expect(tip).toHaveTextContent('95th pct');
    expect(tip).toHaveTextContent('Requests');
  });
});
