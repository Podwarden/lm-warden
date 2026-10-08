// "Bigger trees when far" (user decision, 2026-10-06): trees and flora are drawn larger as the camera pulls back, at
// render time about their bases, capped so no scaled crown overlaps a neighbour.
import { describe, expect, it } from "vitest";
import { FAR_D0, FAR_MAX, REF_VIEW_H, crownCaps, effectiveDistance, farScale } from "@/components/forest/engine/farscale";

describe("farScale", () => {
  it("is 1.0 up to FAR_D0 (every 1h, 6h and 24h view), then grows: continuous, monotone, capped", () => {
    // measured: full window 1h 42, 6h ≤ 69, 24h ≤ 146; card 24h ≤ 73 u, i.e. ≤ 205 normalised
    for (const d of [0, 42, 69, 146, 205, FAR_D0]) expect(farScale(d)).toBe(1);
    let prev = 1;
    for (let d = FAR_D0; d <= 4000; d += 0.5) {
      const s = farScale(d);
      expect(s).toBeGreaterThanOrEqual(prev - 1e-12); // monotone
      expect(s - prev).toBeLessThan(0.01); // continuous: no step at 0.5 u
      prev = s;
    }
    expect(farScale(1e6)).toBe(FAR_MAX);
    // beyond the blend the apparent size holds: s = d / FAR_D0
    expect(farScale(2 * FAR_D0)).toBeCloseTo(2, 9);
  });

  it("the distance is normalised to the full window's height (the card's 320 px canvas is a third of it)", () => {
    expect(effectiveDistance(100, REF_VIEW_H)).toBe(100);
    expect(effectiveDistance(100, REF_VIEW_H / 2)).toBe(200);
  });

  it("crownCaps: no two scaled crowns overlap; trees already touching stay at 1; a lone tree is not limited", () => {
    const trees = [
      { id: "a", x: 0, z: 0, r: 3 },
      { id: "b", x: 12, z: 0, r: 3 }, // 12 apart: up to 2×
      { id: "c", x: 12, z: 4.4, r: 2.6 }, // a lane over from b, touching at 1×
      { id: "d", x: 500, z: 0, r: 3 },
    ];
    const caps = crownCaps(trees);
    expect(caps.get("a")).toBeCloseTo(2, 9);
    expect(caps.get("b")).toBe(1);
    expect(caps.get("c")).toBe(1);
    expect(caps.get("d")).toBe(FAR_MAX);
    for (const p of trees)
      for (const q of trees) {
        if (p === q) continue;
        const s = Math.min(caps.get(p.id)!, FAR_MAX), t = Math.min(caps.get(q.id)!, FAR_MAX);
        const dist = Math.hypot(p.x - q.x, p.z - q.z);
        if (dist >= p.r + q.r) expect(s * p.r + t * q.r).toBeLessThanOrEqual(dist + 1e-9);
      }
  });
});

describe("far views build the live front first, and every tree they can (re-review 3, N3)", () => {
  it("selectBuilt: with the view mid-span, the newest trees are built before the rest", async () => {
    const { selectBuilt } = await import("@/components/forest/engine/frame");
    const c = Array.from({ length: 82 }, (_, i) => ({ id: "t" + i, x: i * 5 }));
    const keep = selectBuilt(c, 205, 60, Infinity);
    expect(keep.size).toBe(60);
    for (let i = 62; i < 82; i++) expect(keep.has("t" + i), "t" + i).toBe(true); // the front's 20
    expect(keep.has("t41")).toBe(true); // then outwards from the view
  });
  it("treeBudget: a far view raises the build budget to FAR_BUILT (every tree of a 7 d week), a near one keeps its own", async () => {
    const { FAR_BUILT, treeBudget } = await import("@/components/forest/engine/farscale");
    expect(treeBudget(60, false)).toBe(60);
    expect(treeBudget(30, false)).toBe(30);
    expect(treeBudget(60, true)).toBe(FAR_BUILT);
    expect(treeBudget(12, true)).toBe(FAR_BUILT);
    expect(FAR_BUILT).toBeGreaterThanOrEqual(82);
  });
  it("final review I2: FAR_BUILT and the frozen cache hold a realistic week (200); the cubes cap the far build count", async () => {
    const { FAR_BUILT, FAR_CUBES_PER_TREE, treeBudget } = await import("@/components/forest/engine/farscale");
    const { FROZEN_CACHE } = await import("@/components/forest/engine/trees");
    expect(FAR_BUILT).toBe(200);
    expect(FROZEN_CACHE).toBe(200);
    expect(FAR_CUBES_PER_TREE).toBe(70);
    // the card: (cube cap − flora) / 70, so the budget never runs out of coarser levels
    expect(treeBudget(30, true, 15_000 - 400)).toBe(200); // 208 fit: FAR_BUILT
    expect(treeBudget(30, true, 15_000 - 6_000)).toBe(Math.floor(9_000 / 70));
    expect(treeBudget(30, true, 1_000)).toBe(30); // never below the view's own budget
    expect(treeBudget(30, false, 1_000)).toBe(30);
    expect(treeBudget(60, true, 120_000 - 400)).toBe(200); // the full window: the cubes never bind
  });
});

describe("caps change smoothly (re-review 3, N4)", () => {
  it("easeScale moves toward its target with a 0.25 s time constant: no jump when a neighbour is built", async () => {
    const { easeScale } = await import("@/components/forest/engine/farscale");
    let s = 3.4, prev = 3.4, maxStep = 0;
    for (let i = 0; i < 120; i++) {
      s = easeScale(s, 2.2, 1 / 60);
      maxStep = Math.max(maxStep, Math.abs(s - prev));
      prev = s;
    }
    expect(Math.abs(s - 2.2)).toBeLessThan(0.01); // there within 2 s
    expect(maxStep).toBeLessThan(0.1); // ≤ 1.2 · (1 − e^(−1/15)) per frame
    expect(easeScale(1, 1, 1 / 60)).toBe(1);
  });
});

describe("flora stays on the grass when far", () => {
  it("floraRoom: a bed at 0.85 of the half-width may grow until it reaches the grass edge, never less than 1", async () => {
    const { floraRoom } = await import("@/components/forest/engine/farscale");
    // half-width 15, bed at z 12.75 with radius 1.9 (squashed 0.85 across): room 2.25 / 1.615
    expect(floraRoom(15, 12.75, 1.9)).toBeCloseTo(2.25 / (1.9 * 0.85), 6);
    expect(floraRoom(15, 14.9, 1.9)).toBe(1);
    expect(floraRoom(15, 0, 0.5)).toBeGreaterThan(4);
  });
});
