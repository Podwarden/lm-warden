// frontend/tests/component/forest-pixel-ratio.test.ts
// Adaptive render resolution (rulings R26, R28): thresholds, hysteresis, floor and ceiling of the pure governor.
import { describe, expect, it } from "vitest";
import { PIXEL_RATIO, PixelRatioGovernor } from "@/components/forest/engine/frame";

/** Runs frames for `ms` of wall time from `t0`; `cost(ratio, t)` gives each frame's interval. Returns the end time. */
function run(g: PixelRatioGovernor, t0: number, ms: number, cost: (r: number, t: number) => number, log?: number[]) {
  let t = t0;
  while (t < t0 + ms) {
    const dt = cost(g.ratio, t);
    t += dt;
    const r = g.frame(t, dt);
    if (r !== null) log?.push(r);
  }
  return t;
}

const HZ60 = 1000 / 60, HZ120 = 1000 / 120;

describe("pixel ratio governor", () => {
  it("thresholds: down above 20 ms, up at ≤ 17.5 ms (60 Hz frames) held 5 s, steps of 0.25 between 1.0 and 1.5", () => {
    expect(PIXEL_RATIO).toMatchObject({
      DOWN_P90_MS: 20, UP_P90_MS: 17.5, UP_HOLD_MS: 5000, STEP: 0.25, WINDOW_MS: 2000, FLOOR: 1, MAX: 1.5, STALL_MS: 250,
    });
  });
  it("starts at min(devicePixelRatio, 1.5); compact stays ≤ 1", () => {
    expect(new PixelRatioGovernor(2).ratio).toBe(1.5);
    expect(new PixelRatioGovernor(1.25).ratio).toBe(1.25);
    expect(new PixelRatioGovernor(1).ratio).toBe(1);
    const c = new PixelRatioGovernor(2, true);
    expect(c.ratio).toBe(1);
    run(c, 0, 20_000, () => 40); // slow, but never below 1 and never above 1
    expect(c.ratio).toBe(1);
  });
  it("steps down 0.25 per full 2 s window while p90 > 20 ms, to a floor of 1.0", () => {
    const g = new PixelRatioGovernor(2), log: number[] = [];
    let t = run(g, 0, 1900, () => 33, log);
    expect(log).toEqual([]); // no decision before a full window
    t = run(g, t, 400, () => 33, log);
    expect(log).toEqual([1.25]);
    t = run(g, t, 2400, () => 33, log);
    expect(log).toEqual([1.25, 1]);
    run(g, t, 10_000, () => 33, log);
    expect(log).toEqual([1.25, 1]); // floor
  });
  it("the 17.5–20 ms dead band holds the ratio either way", () => {
    const g = new PixelRatioGovernor(2), log: number[] = [];
    const t = run(g, 0, 2030, () => 25, log); // → 1.25 at the first full window (then the slow frames stop)
    expect(log).toEqual([1.25]);
    run(g, t, 120_000, () => 19, log);
    expect(log).toEqual([1.25]);
  });
  it("60 Hz: after a slow patch, steady 16.7 ms frames step back up (once the 30 s back-off has passed)", () => {
    const g = new PixelRatioGovernor(2), log: number[] = [];
    let t = run(g, 0, 2300, () => 30, log); // a loaded 2 s patch → 1.25
    t = run(g, t, 20_000, () => HZ60, log);
    expect(log).toEqual([1.25]); // 1.5 is not retried within 30 s of proving too slow
    run(g, t, 20_000, () => HZ60, log);
    expect(log).toEqual([1.25, 1.5]);
  });
  it("60 Hz from the floor: 1.0 → 1.25 after 5 s of 16.7 ms frames, not before", () => {
    const g = new PixelRatioGovernor(1.25), log: number[] = [];
    let t = run(g, 0, 2300, () => 30, log); // → 1.0; 1.25 blocked for 30 s
    t = run(g, t, 30_000, () => 19, log); // wait out the back-off in the dead band
    t = run(g, t, 6000, () => HZ60, log); // the window clears of 19 ms frames after ~2 s: ~4 s good so far
    expect(log).toEqual([1]);
    run(g, t, 2000, () => HZ60, log);
    expect(log).toEqual([1, 1.25]);
  });
  it("120 Hz: 8.3 ms frames step back up just the same", () => {
    const g = new PixelRatioGovernor(2), log: number[] = [];
    const t = run(g, 0, 2300, () => 25, log);
    run(g, t, 40_000, () => HZ120, log);
    expect(log).toEqual([1.25, 1.5]);
  });
  it("one interruption of the good streak restarts the 5 s count", () => {
    const g = new PixelRatioGovernor(1.25), log: number[] = [];
    let t = run(g, 0, 2300, () => 30, log);
    t = run(g, t, 31_000, () => 19, log);
    t = run(g, t, 4000, () => HZ60, log);
    t = run(g, t, 2500, () => 19, log); // a dead-band patch long enough to lift the window's p90
    t = run(g, t, 4000, () => HZ60, log);
    expect(log).toEqual([1]);
    run(g, t, 3500, () => HZ60, log); // now 5 s good in a row
    expect(log).toEqual([1, 1.25]);
  });
  it("never flips back and forth when the start ratio is just too slow (hysteresis)", () => {
    // fill-bound: 1.5 costs 22 ms (too slow), 1.25 meets 60 Hz
    const g = new PixelRatioGovernor(2), log: number[] = [];
    run(g, 0, 10 * 60_000, (r) => (r > 1.3 ? 22 : HZ60), log);
    // down once; each retry of 1.5 doubles the back-off (30 s, 60 s, 120 s, 240 s, 480 s …): few changes in 10 min
    expect(log[0]).toBe(1.25);
    expect(log.length).toBeLessThanOrEqual(9);
    for (let i = 1; i < log.length; i++) expect(Math.abs(log[i] - log[i - 1])).toBe(0.25);
  });
  it("a stall (interval > 250 ms) is dropped: it neither resets the window nor counts toward a step down", () => {
    const g = new PixelRatioGovernor(2), log: number[] = [];
    // slow frames around a stall still step down once a full window has passed
    let t = run(g, 0, 1500, () => 30, log);
    t += 5000;
    expect(g.frame(t, 5000)).toBeNull();
    run(g, t, 600, () => 30, log);
    expect(log).toEqual([1.25]);
    // a scene of fast frames with a stall every second never steps down
    const h = new PixelRatioGovernor(2), lh: number[] = [];
    run(h, 0, 30_000, (_r, t2) => (Math.floor(t2 / 1000) !== Math.floor((t2 + HZ60) / 1000) ? 400 : HZ60), lh);
    expect(lh).toEqual([]);
  });
  it("pinned at both limits (start ≤ floor): it keeps nothing and never changes", () => {
    const g = new PixelRatioGovernor(1), log: number[] = [];
    run(g, 0, 20_000, () => 40, log);
    run(g, 20_000, 20_000, () => 5, log);
    expect(log).toEqual([]);
    expect((g as unknown as { samples: unknown[] }).samples).toHaveLength(0);
  });
});
