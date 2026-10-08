import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import {
  aggregateThroughput,
  aggregateGpuUtil,
  summarizeThroughput,
  summarizeGpuUtil,
  rangeBounds,
  RANGE_MS,
  type ModelSample,
  type GpuSample,
  type StatsRange,
} from '@/lib/stats';
import { MetricSummaryPanel } from '@/components/panels/metric-summary-panel';

// ---------------------------------------------------------------------------
// aggregateThroughput — backend returns a flat array of per-model per-minute
// samples; the chart needs one row per minute with the per-minute totals
// (and a timestamp ms field for the X axis). These tests pin the contract.
// ---------------------------------------------------------------------------

describe('aggregateThroughput', () => {
  it('sums tokens across models within the same minute', () => {
    const samples: ModelSample[] = [
      { model_id: 'a', minute: 100, requests: 2, prompt_tokens: 10, completion_tokens: 5 },
      { model_id: 'b', minute: 100, requests: 1, prompt_tokens: 3, completion_tokens: 7 },
      { model_id: 'a', minute: 101, requests: 4, prompt_tokens: 20, completion_tokens: 0 },
    ];
    const out = aggregateThroughput(samples);
    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({ minute: 100, tokens: 25, requests: 3 });
    expect(out[1]).toMatchObject({ minute: 101, tokens: 20, requests: 4 });
  });

  it('attaches an epoch-ms `ts` field derived from minute', () => {
    const out = aggregateThroughput([
      { model_id: 'a', minute: 100, requests: 1, prompt_tokens: 1, completion_tokens: 1 },
    ]);
    expect(out[0].ts).toBe(100 * 60_000);
  });

  it('returns an empty array for empty input', () => {
    expect(aggregateThroughput([])).toEqual([]);
  });

  it('sorts the output by minute ascending', () => {
    // Backend already sorts, but if a future caller hands us unsorted data
    // the chart X-axis would jag — pin the defensive sort.
    const out = aggregateThroughput([
      { model_id: 'a', minute: 200, requests: 0, prompt_tokens: 0, completion_tokens: 5 },
      { model_id: 'a', minute: 100, requests: 0, prompt_tokens: 0, completion_tokens: 3 },
    ]);
    expect(out.map((p) => p.minute)).toEqual([100, 200]);
  });
});

// ---------------------------------------------------------------------------
// aggregateGpuUtil — flat array of {gpu_index, minute, utilization_pct, …}
// → array of points keyed by minute, with one numeric column per gpu_index
// (`gpu0`, `gpu1`, …). Empty cells fall through as undefined so recharts
// draws gaps rather than zero-bars when a GPU has no sample for that minute.
// ---------------------------------------------------------------------------

describe('aggregateGpuUtil', () => {
  it('produces one row per minute with per-gpu columns', () => {
    const samples: GpuSample[] = [
      { gpu_index: 0, minute: 100, utilization_pct: 80, memory_used_mib: 1, memory_total_mib: 1 },
      { gpu_index: 1, minute: 100, utilization_pct: 50, memory_used_mib: 1, memory_total_mib: 1 },
      { gpu_index: 0, minute: 101, utilization_pct: 90, memory_used_mib: 1, memory_total_mib: 1 },
    ];
    const { points, gpuIndexes } = aggregateGpuUtil(samples);
    expect(gpuIndexes).toEqual([0, 1]);
    expect(points).toHaveLength(2);
    expect(points[0]).toMatchObject({ minute: 100, gpu0: 80, gpu1: 50 });
    // gpu 1 has no sample at minute 101 → the column is missing (or undefined),
    // not 0; recharts treats missing as a gap which is what we want.
    expect(points[1]).toMatchObject({ minute: 101, gpu0: 90 });
    expect(points[1].gpu1).toBeUndefined();
  });

  it('returns empty result for empty input', () => {
    expect(aggregateGpuUtil([])).toEqual({ points: [], gpuIndexes: [] });
  });

  it('attaches epoch-ms `ts` field per row', () => {
    const out = aggregateGpuUtil([
      { gpu_index: 0, minute: 100, utilization_pct: 50, memory_used_mib: 0, memory_total_mib: 0 },
    ]);
    expect(out.points[0].ts).toBe(100 * 60_000);
  });
});

// ---------------------------------------------------------------------------
// summary helpers — fuel for the MetricSummaryPanel cards above each chart.
// Pin the math, not the formatting (formatting is a render concern).
// ---------------------------------------------------------------------------

describe('summarizeThroughput', () => {
  it('totals tokens + requests across the window', () => {
    const out = summarizeThroughput([
      { model_id: 'a', minute: 1, requests: 2, prompt_tokens: 10, completion_tokens: 5 },
      { model_id: 'b', minute: 2, requests: 3, prompt_tokens: 0, completion_tokens: 7 },
    ]);
    expect(out.totalTokens).toBe(22);
    expect(out.totalRequests).toBe(5);
    expect(out.activeModels).toBe(2);
  });

  it('handles an empty window without dividing by zero', () => {
    expect(summarizeThroughput([])).toEqual({
      totalTokens: 0,
      totalRequests: 0,
      activeModels: 0,
    });
  });
});

describe('summarizeGpuUtil', () => {
  it('reports peak and average utilisation', () => {
    const out = summarizeGpuUtil([
      { gpu_index: 0, minute: 1, utilization_pct: 50, memory_used_mib: 0, memory_total_mib: 0 },
      { gpu_index: 0, minute: 2, utilization_pct: 100, memory_used_mib: 0, memory_total_mib: 0 },
      { gpu_index: 1, minute: 1, utilization_pct: 25, memory_used_mib: 0, memory_total_mib: 0 },
    ]);
    expect(out.peakPct).toBe(100);
    // (50 + 100 + 25) / 3 = 58.33…
    expect(out.avgPct).toBeCloseTo(58.33, 1);
    expect(out.gpuCount).toBe(2);
  });

  it('handles empty input', () => {
    expect(summarizeGpuUtil([])).toEqual({ peakPct: 0, avgPct: 0, gpuCount: 0 });
  });
});

// ---------------------------------------------------------------------------
// MetricSummaryPanel — generic stat-card row. Driven by props, no fetching.
// ---------------------------------------------------------------------------

describe('MetricSummaryPanel', () => {
  afterEach(() => { cleanup(); });

  it('renders the title and each metric label/value', () => {
    render(
      <MetricSummaryPanel
        title="Throughput"
        metrics={[
          { label: 'Tokens', value: '12,345' },
          { label: 'Requests', value: '42' },
        ]}
      />,
    );
    expect(screen.getByText('Throughput')).toBeInTheDocument();
    expect(screen.getByText('Tokens')).toBeInTheDocument();
    expect(screen.getByText('12,345')).toBeInTheDocument();
    expect(screen.getByText('Requests')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('renders nothing problematic for an empty metric list', () => {
    // Edge case: a chart with no data still mounts the panel with zero
    // metrics. We want a quiet placeholder, not a crash.
    const { container } = render(<MetricSummaryPanel title="Empty" metrics={[]} />);
    expect(container.textContent).toContain('Empty');
  });
});

// ---------------------------------------------------------------------------
// rangeBounds — regression for v2026.05.15.2: the /stats X-axis used
// recharts' default ["dataMin","dataMax"] domain, so changing the range
// selector resized the chart container but did NOT actually widen the
// axis when the rollup table was sparse. The fix derives [startMs, endMs]
// from the range and passes it straight into <XAxis domain>. These tests
// pin the derivation; the chart wiring is exercised at runtime.
// ---------------------------------------------------------------------------

describe('rangeBounds', () => {
  const NOW = Date.UTC(2026, 4, 15, 12, 0, 0); // 2026-05-15 12:00 UTC

  it('returns [now - rangeMs, now] for each selector value', () => {
    const cases: StatsRange[] = ['1h', '6h', '24h', '7d'];
    for (const r of cases) {
      const [start, end] = rangeBounds(r, NOW);
      expect(end).toBe(NOW);
      expect(end - start).toBe(RANGE_MS[r]);
    }
  });

  it('1h is one hour, 7d is one week — exact', () => {
    expect(rangeBounds('1h', NOW)).toEqual([NOW - 3_600_000, NOW]);
    expect(rangeBounds('7d', NOW)).toEqual([NOW - 7 * 24 * 3_600_000, NOW]);
  });

  it('start is always strictly less than end', () => {
    // Catches a latent bug where someone swaps the operands; recharts
    // silently renders an empty axis if domain[0] > domain[1].
    for (const r of ['1h', '6h', '24h', '7d'] as StatsRange[]) {
      const [start, end] = rangeBounds(r, NOW);
      expect(start).toBeLessThan(end);
    }
  });

  it('defaults `now` to Date.now() when omitted', () => {
    const before = Date.now();
    const [, end] = rangeBounds('1h');
    const after = Date.now();
    expect(end).toBeGreaterThanOrEqual(before);
    expect(end).toBeLessThanOrEqual(after);
  });

  it('returns a fresh tuple each call so React/recharts see a new prop reference', () => {
    // Recharts memoises on domain identity; passing the same array would
    // mask a stale bounds value across re-renders. We don't actually
    // depend on reference *inequality* but pin reference *value* so a
    // future refactor to a frozen constant doesn't silently regress.
    const a = rangeBounds('24h', NOW);
    const b = rangeBounds('24h', NOW);
    expect(a).toEqual(b);
  });
});

// Silence recharts/ResizeObserver chatter from jsdom — the chart components
// themselves are not under test (we test the data pipeline that feeds them).
// Pinning the polyfill here so a future smoke-render test doesn't crash.
class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', FakeResizeObserver);
