// frontend/tests/component/forest-live-time.test.ts
// Final-review fix wave (Plan 2): the shown time runs on the browser's clock with the server's `now` used only for the
// skew (C1); one late rule for turns and flora, with known turns tracked across trees and a history floor (I1, I2, M10);
// flower beds keep their x while shown (M2).
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { LOOK, SKEW, SkewEstimator } from "@/components/forest/engine/clock";
import { FrameCore, sceneTimeline, treeInfo, type CameraLike } from "@/components/forest/engine/frame";
import { FLOWER_GROW_S, FloraRegistry, createGround } from "@/components/forest/engine/ground";
import { canEmerge } from "@/components/forest/engine/trees";
import { applyResponse } from "@/lib/forest/client";
import type { ForestResponse, ForestState, Session, Tree, Turn } from "@/lib/forest/types";

const TRAITS = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
const r1 = (a: number, t0: number) => Math.round((a - t0) * 10) / 10;
const turnAt = (a: number, t0: number): Turn => [r1(a, t0), 37000, 150, 30000, [], ["read"], "tool_calls", 1];
const ses = (id: string, abs: number[], t0: number): Session => ({ id, variant: "v1", children: [], turns: abs.map((a) => turnAt(a, t0)) });
const treeOf = (id: string, sessions: Session[]): Tree => {
  const all = sessions.flatMap((s) => s.turns.map((t) => t[0]));
  return { id, key: "k", start: Math.min(...all), end: Math.max(...all), last: Math.max(...all), last_at: 0, traits: TRAITS, sessions };
};
const resp = (t0: number, nowAbs: number, trees: Tree[], o: Partial<ForestResponse> = {}): ForestResponse => ({
  range: "48h", t0, now: nowAbs, models: {}, trees, flowers: [], ids: trees.map((t) => t.id), full: true, cursor: nowAbs, ...o,
});
const fakeCam = (): CameraLike => ({ step: () => false, pose: () => ({ pos: { x: 0, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 } }), paused: false, setEvents: vi.fn() });
const timesOf = (s: ForestState) => {
  const out = new Map<string, number[]>();
  const go = (x: Session) => (out.set(x.id, x.turns.map((t) => t[0])), x.children.forEach(go));
  for (const t of s.trees.values()) t.sessions.forEach(go);
  return out;
};

// ---------------- C1: the shown time runs on the local clock ----------------

describe("C1: shown time advances at 1× on the browser's clock; the server's now only sets the skew", () => {
  it("cached responses whose now lags by up to 3 s, polled every 2 s: the shown time runs at 1.00 ± 0.05×, no stall", () => {
    const core = new FrameCore(fakeCam());
    const T0 = 1_000_000, TTL = 3, POLL_MS = 2000, LATENCY_MS = 40, FRAME_MS = 1000 / 60;
    /** The server's real time at local ms `l`; its cache answers with the build time (up to 3 s old). */
    const server = (l: number) => T0 + l / 1000;
    let built = -Infinity;
    const respond = (l: number) => {
      if (server(l) - built >= TTL) built = server(l);
      return resp(T0 - 48 * 3600, built, []);
    };
    let held: ForestState | null = null, nextPoll = 0, prevRel: number | null = null, stillMs = 0, worstStall = 0;
    const rates: number[] = [];
    for (let l = 0; l <= 180_000; l += FRAME_MS) {
      if (l >= nextPoll) {
        held = applyResponse(held, respond(nextPoll)).state; // the response was built when the request reached the server
        core.intake(held, l);
        nextPoll += POLL_MS + LATENCY_MS;
      }
      const out = core.frame(l, FRAME_MS / 1000)!;
      if (prevRel !== null && l > 10_000) {
        const rate = (out.rel - prevRel) / (FRAME_MS / 1000);
        rates.push(rate);
        stillMs = out.rel === prevRel ? stillMs + FRAME_MS : 0;
        worstStall = Math.max(worstStall, stillMs);
      }
      prevRel = out.rel;
    }
    expect(Math.min(...rates)).toBeGreaterThanOrEqual(0.95 - 1e-6);
    expect(Math.max(...rates)).toBeLessThanOrEqual(1.05 + 1e-6);
    expect(worstStall).toBeLessThan(100);
  });

  it("the skew is the median of the last 5 samples, followed at most 5 % fast; a jump over 10 s is one step", () => {
    const k = new SkewEstimator();
    expect(k.sample(100, 0)).toBe(true); // the first sample is a step
    expect(k.now(0)).toBe(100);
    // one outlier among five does not move the median
    for (const [s, l] of [[102, 1000], [102, 2000], [160, 3000], [104, 4000]] as const) expect(k.sample(s, l)).toBe(false);
    expect(Math.abs(k.now(4000) - 104)).toBeLessThanOrEqual(0.1); // skew about 100 throughout (the median moved by 0.5 at most)
    // the server's clock is 3 s ahead from now on: the estimate gains it at 5 % (60 s), never faster
    for (let l = 5000; l <= 9000; l += 1000) k.sample(100 + l / 1000 + 3, l);
    let prev = k.now(9000);
    for (let l = 9100; l <= 80_000; l += 100) {
      const n = k.now(l);
      expect((n - prev) / 0.1).toBeLessThanOrEqual(1 + SKEW.MAX_RATE + 1e-9);
      expect((n - prev) / 0.1).toBeGreaterThanOrEqual(1 - 1e-9);
      prev = n;
    }
    expect(k.now(80_000)).toBeCloseTo(100 + 80 + 3, 6);
    // a jump of 60 s: discontinuous once the median has moved
    const steps = [81_000, 82_000, 83_000].map((l) => k.sample(100 + l / 1000 + 63, l));
    expect(steps).toEqual([false, false, true]);
    expect(k.now(83_000)).toBeCloseTo(100 + 83 + 63, 6);
    expect(SKEW).toMatchObject({ SAMPLES: 5, MAX_RATE: 0.05, STEP_S: 10 });
  });

  it("a skew jump over 10 s raises the history floor (a discontinuous clock step)", () => {
    const core = new FrameCore(fakeCam());
    const tr = (abs: number[]) => [treeOf("a", [ses("S", abs, 0)])];
    core.intake(applyResponse(null, resp(0, 2000, tr([1500]))).state, 0);
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    // the server's clock moves 120 s ahead; three samples move the median
    for (const [l, now] of [[10_000, 2130], [12_000, 2132], [14_000, 2134]]) core.intake(applyResponse(null, resp(0, now, tr([1500]))).state, l);
    core.frame(14_000, 0.1);
    // the skipped stretch (−20, 104] is history
    expect(core.isHistory(2134 - 2000 - LOOK - 0.1)).toBe(true);
    expect(core.isHistory(-19.9)).toBe(true);
    expect(core.isHistory(-25)).toBe(false); // before the jump: not skipped
    // a turn from the skipped stretch is history: it keeps its own time
    const s = core.intake(applyResponse(null, resp(0, 2136, tr([1500, 2050]))).state, 16_000).state;
    expect(timesOf(s).get("S")).toEqual([-500, 50]);
  });
});

// ---------------- I1: known turns across trees ----------------

describe("I1: a tree id change keeps every turn's effective time", () => {
  it("the same sessions under a new tree id: no turn is late, the late turn keeps its arrival, nothing regrows or emerges", () => {
    const core = new FrameCore(fakeCam());
    const A = [1500, 1600], B = [1700, 1800];
    core.intake(applyResponse(null, resp(1000, 2000, [treeOf("a", [ses("SA", A, 1000)]), treeOf("b", [ses("SB", B, 1000)])])).state, 0);
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1); // rel ≈ 30
    const arrival = core.clock!.rel;
    const late = arrival - 40 + 2000;
    const s2 = core.intake(applyResponse(null, resp(1000, 2060, [treeOf("a", [ses("SA", [...A, late], 1000)]), treeOf("b", [ses("SB", B, 1000)])])).state, 60_000).state;
    const before = timesOf(s2);
    expect(before.get("SA")!.at(-1)).toBe(arrival);
    for (let k = 601; k <= 900; k++) core.frame(k * 100, 0.1);
    // a day turns busy: both spells regroup into one day tree with a new id
    const day = treeOf("day", [ses("SA", [...A, late], 1000), ses("SB", B, 1000)]);
    const s3 = core.intake(applyResponse(null, resp(1000, 2090, [day])).state, 90_000).state;
    expect(timesOf(s3)).toEqual(before);
    expect(core.lateCount()).toBe(1); // only the one really late turn, still keyed by its session
    // built fully grown: the new id's turns were all known, so it never grows out of the ground
    expect(core.inherited.has("day")).toBe(true);
    const tl = sceneTimeline([...s3.trees.values()], s3.now, core.H);
    const info = treeInfo(s3.trees.get("day")!, tl.places("day"), core.inherited.has("day"));
    expect(canEmerge(core.clock!.rel, -30, true, info)).toBe(false);
    // a tree that really is new may still emerge
    const fresh = treeInfo(treeOf("n", [ses("SN", [2085], 1000)]), [{ from: 85, x: 0, z: 0, lane: 0 }]);
    expect(canEmerge(core.clock!.rel, -30, true, fresh)).toBe(true);
  });

  it("a window slide that renames a tree makes nothing late", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeOf("x", [ses("S1", [1950], 1000), ses("S2", [1960], 1000)])])).state, 0);
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1);
    const s = core.intake(applyResponse(null, resp(1000, 2060, [treeOf("y", [ses("S2", [1960], 1000)])])).state, 60_000).state;
    expect(timesOf(s).get("S2")).toEqual([-40]);
    expect(core.lateCount()).toBe(0);
  });
});

// ---------------- I2: the hidden tab ----------------

describe("I2: showing the tab again does not replay the gap", () => {
  it("hidden 10 min, then a poll with 300 turns from the gap: zero late turns", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeOf("a", [ses("S", [1500], 1000)])])).state, 0);
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    // hidden from 10 s to 610 s: no frames, no polls
    core.resumeAfterHidden(610_000);
    expect(core.clock!.rel).toBeCloseTo(610 - LOOK, 1);
    expect(core.isHistory(610 - LOOK - 0.1)).toBe(true);
    const gap = Array.from({ length: 300 }, (_, i) => 2010 + i * 1.9); // up to 2578 (abs) = scene 578
    const s = core.intake(applyResponse(null, resp(1000, 2610, [treeOf("a", [ses("S", [1500, ...gap], 1000)])])).state, 610_050).state;
    expect(core.lateCount()).toBe(0);
    expect(timesOf(s).get("S")!.slice(1)).toEqual(gap.map((a) => r1(a, 1000) - 1000));
  });

  it("a turn that really is late (its row arrives after the shown time passed its start) still grows from its arrival", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeOf("a", [ses("S", [1500], 1000)])])).state, 0);
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    core.resumeAfterHidden(610_000);
    for (let k = 1; k <= 200; k++) core.frame(610_000 + k * 100, 0.1); // 20 s after the show
    const rel = core.clock!.rel; // ≈ 600
    const s = core.intake(applyResponse(null, resp(1000, 2630, [treeOf("a", [ses("S", [1500, rel + 2000 - 12], 1000)])])).state, 630_000).state;
    expect(timesOf(s).get("S")!.at(-1)).toBe(rel);
    expect(core.lateCount()).toBe(1);
  });

  it("the first data is history: a turn at or below the floor is never late", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeOf("a", [ses("S", [1500], 1000)])])).state, 0);
    expect(core.isHistory(-LOOK)).toBe(true);
    expect(core.isHistory(-LOOK + 0.1)).toBe(false);
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    // a request in flight at load: it started before the shown time at load (−35 < −30), its row arrives now
    const s = core.intake(applyResponse(null, resp(1000, 2010, [treeOf("a", [ses("S", [1500, 1965], 1000)])])).state, 10_000).state;
    expect(timesOf(s).get("S")).toEqual([-500, -35]);
    expect(core.lateCount()).toBe(0);
  });
});

// ---------------- N1, N2: only the skipped stretch of a jump is history ----------------

describe("N1/N2: a resume or a stall makes only the skipped stretch history", () => {
  const start = () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeOf("a", [ses("S", [1500], 1000)])])).state, 0);
    return core;
  };
  /** One poll: `abs` turns in session S (after its first at 1500), `long` a separate session's long request. */
  const poll = (core: FrameCore, abs: number[], nowAbs: number, ms: number, long?: number) => {
    const sessions = [ses("S", [1500, ...abs], 1000), ...(long === undefined ? [] : [ses("L", [long], 1000)])];
    const m = timesOf(core.intake(applyResponse(null, resp(1000, nowAbs, [treeOf("a", sessions)])).state, ms).state);
    return { S: m.get("S")!, L: m.get("L")?.[0] };
  };

  it("a 1 s hide is no jump: a 40 s request finishing after it grows from its arrival", () => {
    const core = start();
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1); // rel ≈ 30
    core.resumeAfterHidden(61_000); // hidden 60 s → 61 s: one second behind, caught up by the clock
    expect(core.isHistory(core.clock!.rel - 1)).toBe(false);
    for (let k = 1; k <= 90; k++) core.frame(61_000 + k * 100, 0.1);
    const rel = core.clock!.rel;
    // started 40 s before the shown time, finished now
    const t = poll(core, [], 2070, 70_000, rel - 40 + 2000);
    expect(t.L).toBe(rel);
    expect(core.lateCount()).toBe(1);
  });

  it("a 10 min hide: the gap's turns are history; a request started before the hide and finished after grows", () => {
    const core = start();
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1); // rel ≈ −20
    core.resumeAfterHidden(610_000); // jumps (−20, 580]
    const gap = Array.from({ length: 300 }, (_, i) => 2010 + i * 1.9);
    const t = poll(core, gap, 2610, 610_050, 1975); // 1975: scene −25, before the jump
    expect(t.L).toBe(core.clock!.rel); // grows from its arrival
    expect(t.S.slice(1)).toEqual(gap.map((a) => r1(a, 1000) - 1000)); // history
    expect(core.lateCount()).toBe(1);
  });

  it("a 12 s stall: the skipped stretch is history, a request started before it grows from its arrival (N2)", () => {
    const core = start();
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1); // rel ≈ 30
    const from = core.clock!.rel;
    core.frame(72_000, 0.1); // a 12 s frame stall: the clock jumps to the bound
    expect(core.clock!.rel - from).toBeGreaterThan(10);
    const t = poll(core, [from + 5 + 2000], 2072, 72_050, from - 10 + 2000);
    expect(t.L).toBe(core.clock!.rel); // started before the stall: late
    expect(t.S[1]).toBeCloseTo(from + 5, 1); // inside the skipped stretch: history
    expect(core.lateCount()).toBe(1);
  });
});

// ---------------- M10: flora late rule ----------------

describe("M10: a flower or mushroom that arrives after its start was shown grows from its arrival", () => {
  const flowersAt = (core: FrameCore, reg: FloraRegistry, t0: number, now: number, abs: [number, number, number][], ms: number) => {
    const s = core.intake(applyResponse(null, resp(t0, now, [], { flowers: abs.map(([a, c, g]) => [r1(a, t0), c, g]) })).state, ms).state;
    return reg.items(s, (t) => t, core.lateFor());
  };
  it("a 45 s request's flower arrives late: its time is the arrival; history and re-sent ones keep theirs", () => {
    const core = new FrameCore(fakeCam()), reg = new FloraRegistry();
    const a = flowersAt(core, reg, 1000, 2000, [[1500, 900, 300]], 0);
    expect(a.map((f) => f.time)).toEqual([-500]);
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1);
    const arrival = core.clock!.rel; // ≈ 30
    const b = flowersAt(core, reg, 1000, 2060, [[1500, 900, 300], [arrival - 45 + 2000, 800, 120], [arrival - 50 + 2000, 400, 0]], 60_000);
    expect(b.map((f) => [f.kind, f.time])).toEqual([["flower", -500], ["flower", arrival], ["mushroom", arrival]].sort((x, y) => (x[1] as number) - (y[1] as number)));
    // grows from nothing at its arrival
    const bed = b.find((f) => f.gen === 120)!;
    expect(bed.time).toBe(arrival);
    // the next poll (and a re-anchor) keeps the effective time
    for (let k = 601; k <= 700; k++) core.frame(k * 100, 0.1);
    const c = flowersAt(core, reg, 1299.97, 2070, [[1500, 900, 300], [arrival - 45 + 2000, 800, 120], [arrival - 50 + 2000, 400, 0]], 70_000);
    expect(c.find((f) => f.gen === 120)!.time).toBe(arrival);
    expect(c.find((f) => f.id === bed.id)).toBeDefined();
    expect(FLOWER_GROW_S).toBe(90);
  });
  it("flora after a hide: 1 s is no jump (late flora grows); 10 min skips only the gap", () => {
    const core = new FrameCore(fakeCam()), reg = new FloraRegistry();
    flowersAt(core, reg, 1000, 2000, [], 0);
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1);
    core.resumeAfterHidden(61_000);
    for (let k = 1; k <= 90; k++) core.frame(61_000 + k * 100, 0.1);
    const rel = core.clock!.rel;
    const a = flowersAt(core, reg, 1000, 2070, [[rel - 45 + 2000, 900, 300]], 70_000);
    expect(a.map((f) => f.time)).toEqual([rel]);
    // then a 10 min hide: a call from before it grows, the gap's calls are history
    core.resumeAfterHidden(670_000);
    const gap: [number, number, number][] = [[2200, 500, 20], [2400, 600, 30]];
    const b = flowersAt(core, reg, 1000, 2670, [[rel - 45 + 2000, 900, 300], [rel - 5 + 2000, 700, 0], ...gap], 670_050);
    const at = core.clock!.rel;
    expect(b.find((f) => f.ctx === 700)!.time).toBe(at);
    expect(b.filter((f) => f.ctx === 500 || f.ctx === 600).map((f) => f.time)).toEqual([1200, 1400].map((x) => r1(x + 1000, 1000) - 1000));
  });

  it("flora from a hidden gap is history, not late", () => {
    const core = new FrameCore(fakeCam()), reg = new FloraRegistry();
    flowersAt(core, reg, 1000, 2000, [], 0);
    core.resumeAfterHidden(600_000);
    const fl = Array.from({ length: 50 }, (_, i) => [2010 + i * 10, 500 + i, 20] as [number, number, number]);
    const items = flowersAt(core, reg, 1000, 2600, fl, 600_050);
    expect(items.map((f) => f.time)).toEqual(fl.map(([a]) => r1(a, 1000) - 1000));
  });
});

// ---------------- M2: beds keep their x ----------------

describe("M2: flower beds never jump on a poll", () => {
  it("a bed keeps its x (and z) while shown when X(t) changes under it", () => {
    const g = createGround(new THREE.Scene(), { shadows: false });
    const st = (flowers: [number, number, number][]): ForestState => ({ t0: 0, now: 1000, range: "48h", trees: new Map(), flowers, models: {}, cursor: 1000 });
    const w = () => 15;
    g.setData(st([[100, 900, 300], [200, 400, 50]]), (t) => t * 0.1, w);
    g.tick(500, 0);
    const before = g.placed();
    expect(before.map((p) => p.x)).toEqual([10, 20]);
    // new activity smooths X over the last 10 min: the same calls now map elsewhere, and a new one arrives
    g.setData(st([[100, 900, 300], [200, 400, 50], [400, 300, 30]]), (t) => t * 0.1 + 3, w);
    g.tick(500, 0);
    const after = g.placed();
    for (const p of before) expect(after.find((q) => q.id === p.id)).toEqual(p);
    expect(after.find((q) => !before.some((p) => p.id === q.id))!.x).toBe(43); // the new bed stands at X(t) now
    g.dispose();
  });
});
