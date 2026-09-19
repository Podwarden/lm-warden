// Unit tests for the god-mode per-request color function.
//
// The color is what makes the interleaved live stream legible — one session's
// tokens are trackable by eye only because the same req_id always renders the
// same color and adjacent requests render distinguishable ones. These
// properties are load-bearing, so they're pinned here.

import { describe, it, expect } from 'vitest';
import { reqColor, reqColorHue } from '@/components/godmode/req-color';

describe('reqColor', () => {
  it('is deterministic — the same req_id yields the same color across calls', () => {
    const a1 = reqColor('req-abc-123');
    const a2 = reqColor('req-abc-123');
    expect(a1.hue).toBe(a2.hue);
    expect(a1.accent).toBe(a2.accent);
    expect(a1.headerBg).toBe(a2.headerBg);
  });

  it('spreads distinct req_ids across the hue wheel (few collisions)', () => {
    // Twelve unrelated ids should mostly land on different hues. Allow the
    // odd 360-bucket collision but demand a broad spread so interleaved
    // sessions stay visually separable.
    const ids = Array.from({ length: 12 }, (_, i) => `session-${i}-${(i * 7).toString(36)}`);
    const hues = new Set(ids.map((id) => Math.round(reqColor(id).hue)));
    expect(hues.size).toBeGreaterThanOrEqual(10);
  });

  it('golden-angle index path separates adjacent requests by a wide arc', () => {
    // The viewer keys color on a per-session request index; index 0 and 1
    // must be far apart on the wheel regardless of req_id.
    const h0 = reqColorHue('whatever', 0);
    const h1 = reqColorHue('whatever', 1);
    const raw = Math.abs(h0 - h1);
    const circular = Math.min(raw, 360 - raw);
    expect(circular).toBeGreaterThan(60);
  });

  it('the index path ignores the id so color is stable even if the id string changes representation', () => {
    // Same session index → same hue, independent of the id passed alongside.
    expect(reqColorHue('a', 3)).toBe(reqColorHue('b', 3));
  });

  it('uses the mockup palette: hue = 40 + i·137.508, accent 70%/60%, header 45%/18%/.55, text 85%/75%', () => {
    expect(reqColorHue('x', 0)).toBe(40);
    const c = reqColor('x', 1);
    expect(c.hue).toBeCloseTo(177.508, 3);
    expect(c.accent).toBe('hsl(177.5 70% 60%)');
    expect(c.headerBg).toBe('hsl(177.5 45% 18% / 0.55)');
    expect(c.text).toBe('hsl(177.5 85% 75%)');
  });
});
