import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { DpRoutingCard } from '@/components/models/dp-routing-card';
import { stickyShare, type DpRoutingResponse } from '@/lib/dp-routing';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

function rank(i: number, over: Record<string, unknown> = {}) {
  return {
    rank: i,
    in_flight: i,
    assigned_sessions: i,
    placed: 0,
    sticky: 10 * (i + 1),
    spilled_in: i,
    client_pinned: 0,
    requests_running: i + 0.0,
    requests_waiting: 0.0,
    kv_cache_usage_perc: 0.41,
    prefix_cache_hits: 1234.0,
    prefix_cache_queries: 1500.0,
    prefix_cache_hit_rate: 0.8227,
    ...over,
  };
}

function payload(over: Partial<DpRoutingResponse> = {}): DpRoutingResponse {
  return {
    model_id: 'abc',
    data_parallel_size: 7,
    affinity_enabled: true,
    spill_threshold: 8,
    spill_threshold_source: 'auto',
    since: '2026-10-04T06:40:00Z',
    totals: { in_flight: 12, sticky: 962, spilled: 38, client_pinned: 0, balanced: 3, unrouted: 0 },
    ranks: Array.from({ length: 7 }, (_, i) => rank(i)),
    engine_metrics: { available: true, error: null, scraped_at: '2026-10-04T06:41:02Z' },
    ...over,
  } as DpRoutingResponse;
}

function stub(body: unknown, status = 200) {
  const m = vi.fn(async (..._a: unknown[]) => new Response(JSON.stringify(body), { status }));
  vi.stubGlobal('fetch', m);
  return m;
}

async function renderCard(props: { dp: number; status?: string }) {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <DpRoutingCard
        modelId="abc"
        dataParallelSize={props.dp}
        status={(props.status ?? 'loaded') as 'loaded'}
      />
    </SWRConfig>,
  );
}

describe('stickyShare', () => {
  it('is sticky / (sticky + spilled), null with no routed traffic', () => {
    expect(stickyShare({ sticky: 962, spilled: 38 } as never)).toBeCloseTo(0.962);
    expect(stickyShare({ sticky: 0, spilled: 0 } as never)).toBeNull();
  });
});

describe('DpRoutingCard', () => {
  beforeEach(() => {
    setAccessToken('t');
    setCsrfToken('c');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders nothing and does not fetch for dp = 1', async () => {
    const f = stub(payload());
    await renderCard({ dp: 1 });
    expect(screen.queryByText('Data-parallel routing')).toBeNull();
    expect(f).not.toHaveBeenCalled();
  });

  it('renders one row per rank with the numbers and the sticky share', async () => {
    const f = stub(payload());
    await renderCard({ dp: 7 });
    const table = await screen.findByTestId('dp-rank-table');
    expect(f.mock.calls[0][0]).toBe('/api/models/abc/dp-routing');
    for (let i = 0; i < 7; i++) {
      expect(within(table).getByTestId(`dp-rank-row-${i}`)).toBeInTheDocument();
    }
    const r3 = screen.getByTestId('dp-rank-row-3');
    expect(r3).toHaveTextContent('40'); // sticky
    expect(r3).toHaveTextContent('41.0 %'); // kv used
    expect(r3).toHaveTextContent('82.3 %'); // prefix hit rate
    expect(r3).toHaveTextContent('3.0 / 0.0'); // running / waiting
    expect(screen.getByText('Sticky 96.2 %')).toBeInTheDocument();
    expect(screen.getByText('In flight 12')).toBeInTheDocument();
    expect(screen.getByText('Affinity on')).toBeInTheDocument();
    expect(screen.getByText('Spill threshold 8 (auto)')).toBeInTheDocument();
    expect(screen.getByText(/Counters since/)).toBeInTheDocument();
  });

  it('shows the setting source and affinity off', async () => {
    stub(payload({ affinity_enabled: false, spill_threshold: 5, spill_threshold_source: 'setting' }));
    await renderCard({ dp: 7 });
    expect(await screen.findByText('Affinity off')).toBeInTheDocument();
    expect(screen.getByText('Spill threshold 5 (setting)')).toBeInTheDocument();
  });

  it('renders a dash for null metrics and for an empty sticky share', async () => {
    stub(
      payload({
        since: null,
        totals: { in_flight: 0, placed: 0, sticky: 0, spilled: 0, client_pinned: 0, balanced: 0, unrouted: 0 },
        ranks: Array.from({ length: 7 }, (_, i) =>
          rank(i, {
            in_flight: 0,
            sticky: 0,
            spilled_in: 0,
            requests_running: null,
            requests_waiting: null,
            kv_cache_usage_perc: null,
            prefix_cache_hits: null,
            prefix_cache_queries: null,
            prefix_cache_hit_rate: null,
          }),
        ),
        engine_metrics: { available: false, error: 'model not loaded', scraped_at: null },
      }),
    );
    await renderCard({ dp: 7 });
    const row = await screen.findByTestId('dp-rank-row-0');
    expect(row.textContent).toContain('—');
    expect(screen.getByText('Sticky —')).toBeInTheDocument();
    expect(screen.getByText(/no requests yet/i)).toBeInTheDocument();
    expect(screen.getByTestId('dp-engine-metrics-unavailable')).toHaveTextContent('model not loaded');
  });

  it('explains missing per-replica series when the error is null', async () => {
    stub(payload({ engine_metrics: { available: false, error: null, scraped_at: null } }));
    await renderCard({ dp: 7 });
    expect(await screen.findByTestId('dp-engine-metrics-unavailable')).toHaveTextContent(
      /does not report per-replica metrics/i,
    );
  });

  it('degrades quietly on 404 (older API) and on a server error', async () => {
    stub({ detail: 'nope' }, 404);
    await renderCard({ dp: 7 });
    expect(await screen.findByTestId('dp-routing-unavailable')).toBeInTheDocument();
    expect(screen.queryByTestId('dp-rank-table')).toBeNull();
    cleanup();
    stub({ detail: 'boom' }, 500);
    await renderCard({ dp: 7 });
    expect(await screen.findByTestId('dp-routing-unavailable')).toBeInTheDocument();
  });

  it('polls at 2000 ms only while loaded', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const f = stub(payload());
      await renderCard({ dp: 7, status: 'loaded' });
      await screen.findByTestId('dp-rank-table');
      const n = f.mock.calls.length;
      await vi.advanceTimersByTimeAsync(2100);
      expect(f.mock.calls.length).toBeGreaterThan(n);
      cleanup();
      const g = stub(payload());
      await renderCard({ dp: 7, status: 'failed' });
      await screen.findByTestId('dp-rank-table');
      const m = g.mock.calls.length;
      await vi.advanceTimersByTimeAsync(6000);
      expect(g.mock.calls.length).toBe(m);
    } finally {
      vi.useRealTimers();
    }
  });
});
