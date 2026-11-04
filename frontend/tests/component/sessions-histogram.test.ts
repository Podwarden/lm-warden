import { describe, it, expect } from 'vitest';

import { barLayout, brushToRange, rangeToBrush } from '@/lib/sessions/histogram';
import type { Histogram } from '@/lib/sessions/types';

const H: Histogram = {
  bin_s: 3600,
  start: 1_000_000,
  bins: [
    { t: 1_000_000, sessions: 2, over_dw: 1 },
    { t: 1_003_600, sessions: 4, over_dw: 0 },
    { t: 1_007_200, sessions: 0, over_dw: 0 },
    { t: 1_010_800, sessions: 1, over_dw: 1 },
  ],
};

describe('activity histogram', () => {
  it('lays out one bar per bin with the light top as the over-10% share', () => {
    const bars = barLayout(H, 400, 80);
    expect(bars).toHaveLength(4);
    expect(bars.map((b) => b.x)).toEqual([0, 100, 200, 300]);
    expect(bars.every((b) => b.w === 100)).toBe(true);
    expect(bars.map((b) => b.h)).toEqual([40, 80, 0, 20]);
    expect(bars.map((b) => b.hTop)).toEqual([20, 0, 0, 20]);
    expect(bars[1].t).toBe(1_003_600);
  });

  it('lays out an all-empty histogram as zero-height bars', () => {
    const empty = { ...H, bins: H.bins.map((b) => ({ ...b, sessions: 0, over_dw: 0 })) };
    expect(barLayout(empty, 400, 80).map((b) => b.h)).toEqual([0, 0, 0, 0]);
  });

  it('maps a drag to bin-aligned since/until, either direction, clamped', () => {
    expect(brushToRange(130, 250, H, 400)).toEqual({ since: 1_003_600, until: 1_010_800 });
    expect(brushToRange(250, 130, H, 400)).toEqual({ since: 1_003_600, until: 1_010_800 });
    expect(brushToRange(-50, 999, H, 400)).toEqual({ since: 1_000_000, until: 1_014_400 });
    // a click (no width) still selects the bin under it
    expect(brushToRange(150, 150, H, 400)).toEqual({ since: 1_003_600, until: 1_007_200 });
  });

  it('maps a range back to the brush box, clamped to the chart', () => {
    expect(rangeToBrush({ since: 1_003_600, until: 1_010_800 }, H, 400)).toEqual({ x0: 100, x1: 300 });
    expect(rangeToBrush({ since: 0, until: 9e9 }, H, 400)).toEqual({ x0: 0, x1: 400 });
  });
});
