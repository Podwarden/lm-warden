import { describe, it, expect } from 'vitest';

import { ANATOMY_CLASS, ANATOMY_ORDER, anatomyWidths } from '@/lib/sessions/anatomy';

describe('prompt anatomy widths', () => {
  it('turns an anatomy into widths summing to 100 in the fixed order', () => {
    const w = anatomyWidths({ cached: 71, dead: 22, recomputed: 3, new: 4, unknown: 0 });
    expect(w.map((s) => s.key)).toEqual(['cached', 'dead', 'recomputed', 'new']);
    expect(w.reduce((a, s) => a + s.pct, 0)).toBeCloseTo(100, 6);
    expect(w[0].pct).toBeCloseTo(71, 6);
  });

  it('keeps the fixed order whatever the sizes', () => {
    const w = anatomyWidths({ cached: 1, dead: 0, recomputed: 0, new: 50, unknown: 49 });
    expect(w.map((s) => s.key)).toEqual(['cached', 'new', 'unknown']);
  });

  it('returns no segments for an all-zero bar', () => {
    expect(anatomyWidths({ cached: 0, dead: 0, recomputed: 0, new: 0, unknown: 0 })).toEqual([]);
  });

  it('has a class for every segment, the hatched one for dead weight', () => {
    expect(ANATOMY_ORDER.map((k) => ANATOMY_CLASS[k])).toEqual([
      'bg-emerald-500',
      'anat-dead',
      'bg-rose-500',
      'bg-stone-500',
      'bg-slate-400',
    ]);
  });
});
