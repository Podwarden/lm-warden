import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  parseRange, rangeQuery, resolveWindow, pollIntervalMs, spanWords, binLabel, seriesUrl,
  fillBins, fetchSeries, SeriesError, ordinal, timingSampleLine, type RangeSel, type TokenSeries,
} from '@/lib/token-series';
import { setAccessToken } from '@/lib/auth-fetch';

const sp = (q: string) => new URLSearchParams(q);

describe('URL range state', () => {
  it('defaults to 7d, reads presets and custom periods', () => {
    expect(parseRange(null)).toEqual({ kind: 'preset', preset: '7d' });
    expect(parseRange(sp(''))).toEqual({ kind: 'preset', preset: '7d' });
    expect(parseRange(sp('range=24h'))).toEqual({ kind: 'preset', preset: '24h' });
    expect(parseRange(sp('range=lol'))).toEqual({ kind: 'preset', preset: '7d' });
    expect(parseRange(sp('from=100&to=200'))).toEqual({ kind: 'custom', from: 100, to: 200 });
    expect(parseRange(sp('from=200&to=100'))).toEqual({ kind: 'preset', preset: '7d' });
    expect(parseRange(sp('from=1.5&to=200&range=1h'))).toEqual({ kind: 'preset', preset: '1h' });
  });
  it.each<RangeSel>([
    { kind: 'preset', preset: '1h' }, { kind: 'preset', preset: '6h' },
    { kind: 'preset', preset: '24h' }, { kind: 'preset', preset: '7d' },
    { kind: 'custom', from: 1789000000, to: 1789200000 },
  ])('round-trips %o through the query string', (sel) => {
    expect(parseRange(sp(rangeQuery(sel)))).toEqual(sel);
  });
  it('resolves windows and poll cadence', () => {
    // A preset's server window `[floor(from/60), ceil(to/60))` is exactly the
    // preset's length and includes the current (partial) minute; `to` never
    // runs ahead of now (the route 422s a future `to`).
    expect(resolveWindow({ kind: 'preset', preset: '24h' }, 100_000)).toEqual({ from: 100_020 - 86_400, to: 100_000 });
    for (const [preset, span] of [['1h', 60], ['6h', 360], ['24h', 1440], ['7d', 10_080]] as const) {
      for (const now of [100_000, 100_059, 100_001, 100_020 /* exactly on a minute */]) {
        const w = resolveWindow({ kind: 'preset', preset }, now);
        const fromMin = Math.floor(w.from / 60), toMin = Math.ceil(w.to / 60);
        expect(toMin - fromMin).toBe(span);
        expect(toMin * 60).toBeGreaterThanOrEqual(now);           // every elapsed second is covered…
        if (now % 60) expect(Math.floor(now / 60)).toBeLessThan(toMin);   // …incl. the current partial minute
        expect(w.to).toBeLessThanOrEqual(now);
      }
    }
    expect(resolveWindow({ kind: 'custom', from: 5, to: 9 }, 100_000)).toEqual({ from: 5, to: 9 });
    expect(pollIntervalMs({ kind: 'preset', preset: '1h' })).toBe(10_000);
    expect(pollIntervalMs({ kind: 'preset', preset: '6h' })).toBe(10_000);
    expect(pollIntervalMs({ kind: 'preset', preset: '24h' })).toBe(60_000);
    expect(pollIntervalMs({ kind: 'preset', preset: '7d' })).toBe(60_000);
    expect(pollIntervalMs({ kind: 'custom', from: 1, to: 2 })).toBe(0);
    expect(spanWords({ kind: 'preset', preset: '1h' })).toBe('last hour');
    expect(spanWords({ kind: 'preset', preset: '7d' })).toBe('last 7 days');
    expect(spanWords({ kind: 'custom', from: 1, to: 2 })).toBeNull();
  });
});

describe('bin label', () => {
  it('names every ladder width like the mockup', () => {
    expect([1, 2, 5, 10, 15, 30, 60, 120, 360, 720, 1440, 10080].map(binLabel)).toEqual([
      '1 min bins', '2 min bins', '5 min bins', '10 min bins', '15 min bins', '30 min bins',
      '1 h bins', '2 h bins', '6 h bins', '12 h bins', '1 day bins', '1 week bins',
    ]);
  });
});

describe('timing sample line (spec amendment d913ddd)', () => {
  it('ordinals', () => {
    expect([1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111].map(ordinal)).toEqual(
      ['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '23rd', '101st', '111th']);
  });
  it('is shown only when stride > 1', () => {
    expect(timingSampleLine(undefined)).toBeNull();
    expect(timingSampleLine({ total: 50_000, used: 50_000, stride: 1 })).toBeNull();
    expect(timingSampleLine({ total: 99_624, used: 49_812, stride: 2 }))
      .toBe('Timings from every 2nd request (49,812 of 99,624).');
    expect(timingSampleLine({ total: 150_001, used: 37_501, stride: 4 }))
      .toBe('Timings from every 4th request (37,501 of 150,001).');
  });
});

describe('series', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('builds the series URL', () => {
    expect(seriesUrl('t 1', { from: 10.7, to: 20, chain: true, maxBins: 360 }))
      .toBe('/api/tokens/t%201/series?from=10&to=20&chain=1&max_bins=360');
  });

  it('adds timings=0 only when timings are explicitly turned off', () => {
    expect(seriesUrl('t', { from: 10, to: 20, chain: false, maxBins: 360, timings: false }))
      .toBe('/api/tokens/t/series?from=10&to=20&chain=0&max_bins=360&timings=0');
    expect(seriesUrl('t', { from: 10, to: 20, chain: false, maxBins: 360, timings: true }))
      .toBe('/api/tokens/t/series?from=10&to=20&chain=0&max_bins=360');
  });

  it('adds by_model=0 only when the per-model split is explicitly turned off', () => {
    expect(seriesUrl('t', { from: 10, to: 20, chain: false, maxBins: 360, timings: false, byModel: false }))
      .toBe('/api/tokens/t/series?from=10&to=20&chain=0&max_bins=360&timings=0&by_model=0');
    expect(seriesUrl('t', { from: 10, to: 20, chain: false, maxBins: 360, byModel: true }))
      .toBe('/api/tokens/t/series?from=10&to=20&chain=0&max_bins=360');
  });

  it('fills sparse bins into a dense, aligned list', () => {
    const s = {
      token_ids: ['a'], from_minute: 61, to_minute: 181, bin_minutes: 30, latency_since: null,
      timing_sample: { total: 0, used: 0, stride: 1 },
      totals: { requests: 3, prompt_tokens: 0, completion_tokens: 0 },
      bins: [{ minute: 90, requests_per_min: 0.1, prompt_per_min: 0, completion_per_min: 0, peak_prompt: 0,
        peak_completion: 0, requests: 3, n: 0, queue_p50: null, queue_p95: null, ttft_p50: null,
        ttft_p95: null, duration_p50: null, duration_p95: null }],
      by_model_since: null, by_model: [], model_bins: [],
    } satisfies TokenSeries;
    const bins = fillBins(s);
    expect(bins.map((b) => b.minute)).toEqual([60, 90, 120, 150, 180]);
    expect(bins[1].requests).toBe(3);
    expect(bins[0]).toMatchObject({ requests: 0, requests_per_min: 0, queue_p50: null, n: 0 });
  });

  it('surfaces a 422 detail as SeriesError', async () => {
    setAccessToken('jwt');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'from must be before to' }), { status: 422 }),
    ));
    const err = await fetchSeries('/api/tokens/a/series?from=2&to=1&chain=1&max_bins=360').catch((e) => e);
    expect(err).toBeInstanceOf(SeriesError);
    expect(err.status).toBe(422);
    expect(err.detail).toBe('from must be before to');
  });

  it('joins pydantic validation messages', async () => {
    setAccessToken('jwt');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: [{ msg: 'Input should be a valid integer' }, { msg: 'too long' }] }), { status: 422 },
    )));
    const err = await fetchSeries('/x').catch((e) => e);
    expect(err.detail).toBe('Input should be a valid integer; too long');
  });
});
