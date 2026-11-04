// The activity chart's geometry: bars per hour bin, and the brush <-> time range mapping (snapped to bin edges).

import type { Histogram } from "@/lib/sessions/types";

export interface Bar {
  x: number;
  w: number;
  h: number;
  /** Height of the light top: the share of the bin's sessions with dead weight over 10%. */
  hTop: number;
  t: number;
}

export function barLayout(h: Histogram, width: number, height: number): Bar[] {
  const n = h.bins.length;
  if (n === 0) return [];
  const w = width / n;
  const max = Math.max(1, ...h.bins.map((b) => b.sessions));
  return h.bins.map((b, i) => {
    const bh = (b.sessions / max) * height;
    return { x: i * w, w, h: bh, hTop: b.sessions > 0 ? (b.over_dw / b.sessions) * bh : 0, t: b.t };
  });
}

function end(h: Histogram): number {
  return h.start + h.bins.length * h.bin_s;
}

export function brushToRange(x0: number, x1: number, h: Histogram, width: number): { since: number; until: number } {
  const n = Math.max(1, h.bins.length);
  const bw = width / n;
  const lo = Math.min(Math.max(Math.min(x0, x1), 0), width);
  const hi = Math.min(Math.max(Math.max(x0, x1), 0), width);
  const i0 = Math.min(Math.floor(lo / bw), n - 1);
  const i1 = Math.max(Math.ceil(hi / bw), i0 + 1);
  return { since: h.start + i0 * h.bin_s, until: Math.min(h.start + i1 * h.bin_s, end(h)) };
}

export function rangeToBrush(r: { since: number; until: number }, h: Histogram, width: number): { x0: number; x1: number } {
  const span = end(h) - h.start;
  const x = (t: number) => (span > 0 ? Math.min(Math.max(((t - h.start) / span) * width, 0), width) : 0);
  return { x0: x(r.since), x1: x(r.until) };
}
