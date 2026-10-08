// The forest's token bars (panels/TokenBars): the left axis' tick labels are distinct, in order and fit their width
// (the coordinator's report: they read "100.0k, 150.0k, 100.0k, 150.0k", the leading digits clipped).
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TokenBars, formatAxis, niceTicks } from "@/components/forest/panels/TokenBars";
import type { ForestState, Tree, Turn } from "@/lib/forest/types";

const TRAITS = { sys0: 1, ctx_gen: 1, mean_gen: 1, thrash: 0, fail: 0, fanout: 0 };
/** One turn per minute for 20 minutes, prompts up to ~600k a minute, generated up to ~8k. */
function state(): ForestState {
  const turns: Turn[] = Array.from({ length: 20 }, (_, i) => [i * 60 + 5, 30_000 * (i + 1), 400 * (i + 1), 20_000 * (i + 1), [], [], "stop", 1]);
  const tree: Tree = { id: "a", key: "k", start: 5, end: 1145, last: 1145, last_at: 0, traits: TRAITS, sessions: [{ id: "s", variant: null, turns, children: [] }] };
  return { t0: 0, now: 1200, range: "1h", trees: new Map([["a", tree]]), flowers: [], models: {}, cursor: 0 };
}

describe("token bars axis", () => {
  it("niceTicks: 0 and even steps of 1, 2, 2.5 or 5 × 10^k, covering the top, at most 5 ticks", () => {
    expect(niceTicks(600_000)).toEqual([0, 200_000, 400_000, 600_000]);
    expect(niceTicks(530_000)).toEqual([0, 200_000, 400_000, 600_000]);
    expect(niceTicks(7_200)).toEqual([0, 2_000, 4_000, 6_000, 8_000]);
    expect(niceTicks(7_000)).toEqual([0, 2_000, 4_000, 6_000, 8_000]);
    expect(niceTicks(9_000)).toEqual([0, 2_500, 5_000, 7_500, 10_000]);
    expect(niceTicks(0)).toEqual([0, 1]);
    // token counts are integers: small maxima get whole steps (re-review 2, R6)
    expect(niceTicks(3)).toEqual([0, 1, 2, 3]);
    expect(niceTicks(1)).toEqual([0, 1]);
    expect(niceTicks(6)).toEqual([0, 2, 4, 6]);
    for (const m of [1, 2, 3, 5, 7, 9]) expect(niceTicks(m).every(Number.isInteger)).toBe(true);
    for (const m of [3, 97, 1234, 45_678, 2_345_678]) {
      const t = niceTicks(m);
      expect(t[0]).toBe(0);
      expect(t.at(-1)!).toBeGreaterThanOrEqual(m);
      expect(t.length).toBeLessThanOrEqual(5);
    }
  });

  it("formatAxis: short, distinct, no clipped digits", () => {
    expect([0, 150_000, 300_000, 450_000, 600_000].map(formatAxis)).toEqual(["0", "150k", "300k", "450k", "600k"]);
    expect([2_500, 5_000, 7_500].map(formatAxis)).toEqual(["2.5k", "5k", "7.5k"]);
    expect([1_000_000, 1_500_000].map(formatAxis)).toEqual(["1M", "1.5M"]);
    for (const n of [999, 12_345, 987_654, 3_210_000]) expect(formatAxis(n).length).toBeLessThanOrEqual(5);
  });

  it("the rendered left ticks are distinct and increase from bottom to top", () => {
    render(<TokenBars state={state()} atRel={1200} />);
    const box = screen.getByTestId("token-bars");
    const ticks = [...box.querySelectorAll('[data-axis-tick="left"]')].map((t) => ({ y: Number(t.getAttribute("y")), label: t.textContent ?? "" }));
    expect(ticks.length).toBeGreaterThanOrEqual(3);
    const labels = ticks.map((t) => t.label);
    expect(new Set(labels).size).toBe(labels.length);
    const parse = (s: string) => Number.parseFloat(s) * (s.endsWith("M") ? 1e6 : s.endsWith("k") ? 1e3 : 1);
    const byHeight = [...ticks].sort((a, b) => b.y - a.y).map((t) => parse(t.label)); // bottom (large y) first
    for (let i = 1; i < byHeight.length; i++) expect(byHeight[i]).toBeGreaterThan(byHeight[i - 1]);
    // linear (re-review N4): starts at 0 and the values are evenly spaced, so it never reads as a log scale
    expect(byHeight[0]).toBe(0);
    const steps = byHeight.slice(1).map((v, i) => v - byHeight[i]);
    for (const d of steps) expect(d).toBeCloseTo(steps[0], 6);
    // and evenly spaced on screen
    const ys = [...ticks].sort((a, b) => b.y - a.y).map((t) => t.y);
    const gaps = ys.slice(1).map((y, i) => ys[i] - y);
    for (const g of gaps) expect(Math.abs(g - gaps[0])).toBeLessThan(1);
    // every label lies inside the chart (re-review 2, R4: the bottom "0" was cut to its top arc): a 9 px label drawn
    // with its baseline at y + 3 spans y − 4 … y + 3
    const svg = box.querySelector("svg.recharts-surface")!;
    const h = Number(svg.getAttribute("height"));
    for (const t of box.querySelectorAll('[data-axis-tick]')) {
      const y = Number(t.getAttribute("y")) + Number(t.getAttribute("dy") ?? 0);
      expect(y + 2).toBeLessThanOrEqual(h);
      expect(y - 7).toBeGreaterThanOrEqual(0);
    }
    for (const l of labels) expect(l.length).toBeLessThanOrEqual(5);
  });
});
