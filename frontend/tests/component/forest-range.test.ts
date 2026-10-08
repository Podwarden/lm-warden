// frontend/tests/component/forest-range.test.ts
// Forest follow-up C (brief 2026-10-06-forest-camera-range): the 1h / 6h / 24h / 7d range switch, pure parts.
// - the client: every range is its own state (a switch is one full fetch), the full window's default is 24 h
// - the hand-off carries the range; an older payload without one stays valid
// - scene time stays on the fixed origin: a switch never makes a known turn late, keeps a late turn's arrival, and the
//   trees newly in range are history (no emergence, no late growth); the trees near the front keep their places
// - the land reaches back to the range's start; the follow fit frames the range's span; the switch glides (≤ 3 s)
// - the card's tree budget: 12 for 1h / 6h, 30 for 24h / 7d
import { describe, expect, it, vi } from "vitest";
import { CameraController, framePoints, type V3 } from "@/components/forest/engine/camera";
import {
  FrameCore, type CameraLike, FAR_MIN, FOG_DENSITY, MAX_ZOOM_OUT, depthStep, SceneLayout, zoomOutLimit, cardTreeBudget, viewDepth, planTrees, sceneTimeline, treeInfo, type ReqRecord, type TreeInfo,
} from "@/components/forest/engine/frame";
import { decodeHandoff, encodeHandoff } from "@/components/forest/engine/handoff";
import { LX0, MAX_LAND_COLS, SEA_TILE, TAPER, groundHeight, landSpan, seaFit, taper } from "@/components/forest/engine/land";
import { canEmerge } from "@/components/forest/engine/trees";
import { buildTreeGeometry } from "@/lib/forest/geometry";
import { FOREST_RANGE, applyResponse, forestUrl, pollForest } from "@/lib/forest/client";
import { buildTimeline } from "@/lib/forest/timeline";
import { isForestTokenCacheKey } from "@/lib/forest/token";
import { FOREST_RANGES, RANGE_S, type ForestResponse, type ForestState, type Range, type Tree, type Turn } from "@/lib/forest/types";

const TRAITS = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
const H = 3600;
/** One tree, one session, turns at absolute times `abs`, sent against `t0` (rounded to 0.1 s, as the server does). */
const treeAt = (id: string, abs: number[], t0: number, ses = "S-" + id): Tree => {
  const r = (a: number) => Math.round((a - t0) * 10) / 10;
  const turns = abs.map((a): Turn => [r(a), 37000, 150, 30000, [], ["read"], "tool_calls", 1]);
  return { id, key: "k", start: turns[0][0], end: turns.at(-1)![0], last: turns.at(-1)![0], last_at: t0 + turns.at(-1)![0], traits: TRAITS,
    sessions: [{ id: ses, variant: "v1", children: [], turns }] };
};
const resp = (range: Range, t0: number, nowAbs: number, trees: Tree[], o: Partial<ForestResponse> = {}): ForestResponse => ({
  range, t0, now: nowAbs, models: {}, trees, flowers: [], ids: trees.map((t) => t.id), full: true, cursor: nowAbs, ...o,
});
const fakeCam = (): CameraLike => ({
  step: () => false, pose: () => ({ pos: { x: 0, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 } }), paused: false, setEvents: vi.fn(),
});
const times = (s: ForestState, id: string) => s.trees.get(id)!.sessions[0].turns.map((t) => t[0]);

/** The data of one server: absolute turn times per tree; a window keeps the turns finished inside it. */
const NOW = 1_800_000_000;
const DATA: Record<string, number[]> = {
  // only in 24 h: a spell 20 h ago
  old: [NOW - 20 * H, NOW - 20 * H + 300, NOW - 20 * H + 900],
  // one session across the 6 h edge: two turns before it, two inside
  edge: [NOW - 7 * H, NOW - 6.5 * H, NOW - 2 * H, NOW - 1 * H],
  // recent
  near: [NOW - 900, NOW - 600, NOW - 300],
  // after the first data: shown live in a 1 h view, then out of it an hour later
  mid: [NOW + 600, NOW + 900, NOW + 1200],
};
/** The window [nowAbs − range, nowAbs] of DATA (+ extra turns), as a response anchored at its start. */
const windowOf = (range: Range, nowAbs: number, extra: Record<string, number[]> = {}) => {
  const t0 = nowAbs - RANGE_S[range];
  const trees: Tree[] = [];
  for (const [id, abs] of Object.entries(DATA)) {
    const all = [...abs, ...(extra[id] ?? [])].filter((a) => a >= t0 && a <= nowAbs).sort((a, b) => a - b);
    if (all.length) trees.push(treeAt(id, all, t0));
  }
  return applyResponse(null, resp(range, t0, nowAbs, trees)).state;
};
const run = (core: FrameCore, fromMs: number, seconds: number) => {
  for (let k = 1; k <= seconds * 10; k++) core.frame(fromMs + k * 100, 0.1);
  return fromMs + seconds * 1000;
};

describe("client: one state per range", () => {
  it("the full window's default range is 24 h; the forest ranges are 1h / 6h / 24h / 7d", () => {
    expect(FOREST_RANGE).toBe("24h");
    expect(forestUrl()).toBe("/api/stats/forest?range=24h");
    expect([...FOREST_RANGES]).toEqual(["1h", "6h", "24h", "7d"]);
    expect(RANGE_S["1h"]).toBe(3600);
  });

  it("a state held for another range is not polled with its cursor: one full fetch of the new range, no loop", async () => {
    const s6 = windowOf("6h", NOW);
    const get = vi.fn(async (url: string) => resp("24h", NOW - 24 * H, NOW + 2, [treeAt("near", DATA.near, NOW - 24 * H)]));
    const s = await pollForest(s6, get, "24h");
    expect(get.mock.calls.map((c) => c[0])).toEqual(["/api/stats/forest?range=24h"]);
    expect(s.range).toBe("24h");
  });

  it("the forest-token SWR entries of every range are cleared with the token", () => {
    expect(isForestTokenCacheKey("forest-view:forest-token")).toBe(true);
    expect(isForestTokenCacheKey("forest-view:forest-token:6h")).toBe(true);
    expect(isForestTokenCacheKey("forest-view:session:6h")).toBe(false);
    expect(isForestTokenCacheKey("forest-view:session")).toBe(false);
  });
});

describe("hand-off carries the range", () => {
  const V = { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "follow" as const, frontRel: true as const };
  it("encodes and decodes the card's range", () => {
    expect(decodeHandoff(encodeHandoff({ ...V, range: "6h" }, 1000), 2000)).toEqual({ ...V, range: "6h" });
  });
  it("an older payload without a range stays valid (no range); an unknown range is dropped, the view kept", () => {
    const old = JSON.stringify({ pos: V.pos, target: V.target, mode: "follow", frontRel: true, at: 1000 });
    expect(decodeHandoff(old, 2000)).toEqual(V);
    const odd = JSON.stringify({ pos: V.pos, target: V.target, mode: "follow", frontRel: true, at: 1000, range: "99d" });
    expect(decodeHandoff(odd, 2000)).toEqual(V);
  });
});

describe("frame core: a range switch never regrows what is shown", () => {
  it("6h → 24h: a held tree keeps its turn list (no prepended history), nothing is late, trees newly in range are history", () => {
    const core = new FrameCore(fakeCam());
    const a = core.intake(windowOf("6h", NOW), 0).state;
    const ms = run(core, 0, 60);
    const before = { edge: times(a, "edge"), near: times(a, "near") };
    const b = core.intake(windowOf("24h", NOW + 60), ms).state;
    expect(core.lateCount()).toBe(0);
    expect(times(b, "near")).toEqual(before.near);
    // a held tree's turn list never changes at its old end: the two older turns a trimming window would add are not
    // prepended (its shape stays as shown)
    expect(times(b, "edge")).toEqual(before.edge);
    expect(times(b, "old")).toEqual(DATA.old.map((x) => x - core.origin!));
    expect(core.isHistory(NOW - 20 * H - core.origin!)).toBe(true);
  });

  it("1h watched for 2 h, then 6h: a tree that slid out of the 1 h window comes back as history, not late", () => {
    const core = new FrameCore(fakeCam());
    core.intake(windowOf("1h", NOW), 0);
    const ms = run(core, 0, 2 * 3600);
    const b = core.intake(windowOf("6h", NOW + 2 * H), ms).state;
    expect(times(b, "mid")).toEqual(DATA.mid.map((x) => x - core.origin!));
    expect(core.lateCount()).toBe(0);
  });

  it("a turn growing late keeps its arrival across 6h → 24h", () => {
    const core = new FrameCore(fakeCam());
    core.intake(windowOf("6h", NOW), 0);
    let ms = run(core, 0, 60);
    const arrival = core.clock!.rel;
    const late = arrival - 40 + core.origin!;
    const s1 = core.intake(windowOf("6h", NOW + 60, { edge: [late] }), ms).state;
    expect(times(s1, "edge").at(-1)).toBe(arrival);
    ms = run(core, ms, 5);
    const s2 = core.intake(windowOf("24h", NOW + 65, { edge: [late] }), ms).state;
    expect(times(s2, "edge")).toHaveLength(3);
    expect(times(s2, "edge").at(-1)).toBe(arrival);
    expect(core.lateCount()).toBe(1);
  });

  it("24h → 6h: trees outside the new range go; a tree straddling the edge keeps its whole turn list; nothing is late", () => {
    const core = new FrameCore(fakeCam());
    const a = core.intake(windowOf("24h", NOW), 0).state;
    const ms = run(core, 0, 60);
    const b = core.intake(windowOf("6h", NOW + 60), ms).state;
    expect([...b.trees.keys()].sort()).toEqual(["edge", "near"]);
    expect(times(b, "near")).toEqual(times(a, "near"));
    expect(times(b, "edge")).toEqual(times(a, "edge"));
    expect(core.lateCount()).toBe(0);
  });

  it("no regrowth: an unchanged tree is not rebuilt, and a tree newly in range is built fully grown at once (no emergence)", () => {
    const core = new FrameCore(fakeCam());
    const a = core.intake(windowOf("6h", NOW), 0).state;
    const ms = run(core, 0, 60);
    const rel = core.clock!.rel;
    const infos = (s: ForestState) => {
      const tl = sceneTimeline([...s.trees.values()], s.now, core.H);
      return new Map([...s.trees].map(([id, t]) => [id, treeInfo(t, tl.places(id), core.inherited.has(id))])) as Map<string, TreeInfo>;
    };
    const ia = infos(a);
    const reqs = new Map<string, ReqRecord>([...ia].map(([id, i]) => [id, { cut: rel, times: i.times }]));
    const b = core.intake(windowOf("24h", NOW + 60), ms).state;
    const ib = infos(b);
    const plan = planTrees(ib, reqs, rel, 0, 60, Infinity);
    const builds = new Map(plan.builds.map((x) => [x.id, x]));
    expect(builds.has("near")).toBe(false);
    // first build: at its freeze (30 min after its last turn, long past): every turn in it, fully grown
    const old = ib.get("old")!, cut = builds.get("old")!.cut;
    expect(cut).toBe(Math.min(rel, old.freezeAt));
    expect(old.times.every((t) => t <= cut && t < rel - 3600)).toBe(true);
    expect(canEmerge(rel, -30, true, ib.get("old")!)).toBe(false);
  });
});

describe("C1: a held tree's wood never reshapes (a trimming window, belt and braces)", () => {
  const knobs = { girth: 1.15, tall: 0.8, el: 0.48 };
  const norm = { len: 0.15, foliage: 0.15, models: {} };
  const build = (t: Tree, cut: number) => buildTreeGeometry(t, cut, knobs, norm, { x: 0, z: 0, s: 1 }, TRAITS);
  /** Every turn a minute apart for 3 h; the last one at `NOW`, then a turn per poll. */
  const LONG = Array.from({ length: 180 }, (_, k) => NOW - (179 - k) * 60);
  /** What a server that cut trees at the window's edge would send: the turns of the last hour only. */
  const trimmed = (range: Range, nowAbs: number) => {
    const t0 = nowAbs - RANGE_S[range];
    const abs = [...LONG, ...Array.from({ length: 40 }, (_, k) => NOW + 120 * (k + 1))].filter((a) => a >= t0 && a <= nowAbs);
    const tr = treeAt("long", abs, t0);
    return applyResponse(null, resp(range, t0, nowAbs, [{ ...tr, traits: TRAITS }])).state;
  };
  const same = (a: ReturnType<typeof build>, b: ReturnType<typeof build>) => {
    // the voxels' inputs: wood centrelines with their girth schedules, foliage points, blossom tips
    expect([...b.vox.wood]).toEqual([...a.vox.wood]);
    expect([...b.vox.sched]).toEqual([...a.vox.sched]);
    expect([...b.vox.foliage]).toEqual([...a.vox.foliage]);
    expect([...b.vox.tips]).toEqual([...a.vox.tips]);
    // the cells drawn: identical
    expect([...b.voxels.coords]).toEqual([...a.voxels.coords]);
    expect([...b.voxels.born]).toEqual([...a.voxels.born]);
    expect([...b.voxels.kind]).toEqual([...a.voxels.kind]);
    expect([...b.voxels.shade]).toEqual([...a.voxels.shade]);
  };

  it("a 1h window sliding across a long session for 30 polls: the held turns' wood, foliage, blossoms and cells stay identical", () => {
    const core = new FrameCore(fakeCam());
    const first = core.intake(trimmed("1h", NOW), 0).state;
    const cut = Math.max(...times(first, "long"));
    const ref = build(first.trees.get("long")!, cut);
    expect(ref.vox.foliage.length + ref.voxels.count).toBeGreaterThan(0);
    let ms = 0, n0 = times(first, "long").length;
    for (let k = 1; k <= 30; k++) {
      ms = run(core, ms, 120); // 2 min a poll: the window's start passes two of the held turns each time
      const s = core.intake(trimmed("1h", NOW + 120 * k), ms).state;
      const ts = times(s, "long");
      expect(ts.slice(0, n0)).toEqual(times(first, "long")); // only appended, never cut at the old end
      same(ref, build(s.trees.get("long")!, cut));
    }
    expect(core.lateCount()).toBe(0);
  }, 60_000);

  it("widening and narrowing across a straddling tree leave it unchanged (6h → 24h → 1h → 6h)", () => {
    const core = new FrameCore(fakeCam());
    const a = core.intake(windowOf("6h", NOW), 0).state;
    const ref = build(a.trees.get("edge")!, Math.max(...times(a, "edge")));
    let ms = 0, t = NOW;
    for (const r of ["24h", "1h", "6h"] as const) {
      ms = run(core, ms, 10);
      t += 10;
      const s = core.intake(windowOf(r, t), ms).state;
      if (!s.trees.has("edge")) continue; // 1h: the edge tree has no turn in the last hour (left as a whole)
      expect(times(s, "edge")).toEqual(times(a, "edge"));
      same(ref, build(s.trees.get("edge")!, Math.max(...times(a, "edge"))));
    }
  });
});

describe("I3: no switch moves a shown tree (places frozen at first sight)", () => {
  it("6h → 24h → 7d → 1h → 6h: every tree present throughout keeps its front-relative place (±0.01) and lane", () => {
    // a crowded front: many trees near now, so a re-layout would reassign lanes
    const extra: Record<string, number[]> = {};
    for (let k = 0; k < 14; k++) extra[`f${k}`] = [NOW - 5 * H + k * 1200, NOW - 1800 + k * 60];
    for (let k = 0; k < 10; k++) extra[`o${k}`] = [NOW - 23 * H + k * 200, NOW - 23 * H + k * 200 + 60];
    const win = (range: Range, nowAbs: number) => {
      const t0 = nowAbs - RANGE_S[range];
      const trees: Tree[] = [];
      for (const [id, abs] of Object.entries({ ...DATA, ...extra })) {
        if (!abs.some((a) => a >= t0 && a <= nowAbs)) continue; // a whole tree, picked by the window (C1)
        trees.push(treeAt(id, abs.filter((a) => a <= nowAbs), t0));
      }
      return applyResponse(null, resp(range, t0, nowAbs, trees)).state;
    };
    const core = new FrameCore(fakeCam()), layout = new SceneLayout();
    let s = core.intake(win("6h", NOW), 0).state, ms = 0;
    const frontRel = () => {
      const tl = layout.update([...s.trees.values()], s.now, core.H, s.now - RANGE_S[s.range] - 3600);
      prevRange = s.range;
      const X = tl.X(core.clock!.rel);
      return new Map([...s.trees.keys()].map((id) => [id, { x: tl.places(id)[0].x - X, z: tl.places(id)[0].z }]));
    };
    let prevRange: Range | undefined;
    const seen: Map<string, { x: number; z: number }>[] = [frontRel()];
    for (const r of ["24h", "7d", "1h", "6h"] as const) {
      ms += 1000;
      s = core.intake(win(r, NOW), ms).state; // the same moment: nothing grew, only the range changed
      seen.push(frontRel());
    }
    const always = [...seen[0].keys()].filter((id) => seen.every((m) => m.has(id)));
    expect(always.length).toBeGreaterThanOrEqual(15);
    let worst = 0;
    for (const id of always) for (const m of seen) {
      worst = Math.max(worst, Math.abs(m.get(id)!.x - seen[0].get(id)!.x));
      expect(m.get(id)!.z).toBe(seen[0].get(id)!.z);
    }
    console.log(`I3: ${always.length} trees through 5 ranges, max front-relative move ${worst}`);
    expect(worst).toBeLessThanOrEqual(0.01);
    // new trees (the 7 d history) were laid out around the held ones: no two places share a lane within a crown
    const tl = layout.timeline!;
    const fresh = [...s.trees.keys()].filter((id) => !always.includes(id));
    for (const a of fresh) for (const b of s.trees.keys()) {
      if (a === b) continue;
      const p = tl.places(a)[0], q = tl.places(b)[0];
      if (p.z === q.z) expect(Math.abs(p.x - q.x), `${a} vs ${b}`).toBeGreaterThanOrEqual(11 - 1e-9);
    }
  });
});

describe("layout: a switch keeps the trees near the front in place", () => {
  it("6h → 24h: the near tree's place is unchanged (±0.01); older trees stand behind the old window, x may go below 0", () => {
    const core = new FrameCore(fakeCam()), layout = new SceneLayout();
    const a = core.intake(windowOf("6h", NOW), 0).state;
    const tl6 = layout.update([...a.trees.values()], a.now, core.H, a.now - 6 * H - 3600);
    const x6 = tl6.places("near")[0].x, start6 = tl6.X(NOW - 6 * H - core.origin!);
    const ms = run(core, 0, 60);
    const rel = core.clock!.rel;
    const b = core.intake(windowOf("24h", NOW + 60), ms).state;
    const tl24 = layout.update([...b.trees.values()], b.now, core.H, b.now - 24 * H - 3600);
    // X older than now − 20 min is frozen; the last 20 min (the shown time among them) may rescale a little
    expect(Math.abs(tl24.X(rel) - tl6.X(rel))).toBeLessThan(0.01);
    expect(Math.abs(tl24.X(rel - 1260) - tl6.X(rel - 1260))).toBeLessThan(1e-9);
    expect(Math.abs(tl24.places("near")[0].x - x6)).toBeLessThan(0.01);
    expect(tl24.places("old")[0].x).toBeLessThan(start6);
    // X is increasing through the older (negative layout time) stretch
    let prev = -Infinity;
    for (let t = NOW - 24 * H; t <= NOW; t += 600) {
      const x = tl24.X(t - core.origin!);
      expect(x).toBeGreaterThanOrEqual(prev);
      prev = x;
    }
  });

  it("buildTimeline without negative times is unchanged by the extension (X(0) = 0, same X)", () => {
    const tr = [treeAt("x", [100, 400, 900], 0), treeAt("y", [2000, 2100], 0)];
    const tl = buildTimeline(tr, 3600);
    expect(tl.X(0)).toBe(0);
    const shifted = buildTimeline(tr, 3600, 5);
    expect(shifted.X(1800) - tl.X(1800)).toBeCloseTo(5, 9);
    expect(shifted.places("x")[0].x - tl.places("x")[0].x).toBeCloseTo(5, 9);
  });
});

describe("land reaches back to the range's start", () => {
  it("landSpan starts at LX0 without a range, else just before the range's start (snapped); the cap keeps the front", () => {
    expect(landSpan(1071)!.lx0).toBe(LX0);
    const at = landSpan(1071, 500)!;
    expect(at.lx0).toBeLessThanOrEqual(500 - 160);
    expect(at.lx0).toBeGreaterThan(500 - 160 - 200);
    const back = landSpan(1071, -2000)!;
    expect(back.lx0).toBeLessThanOrEqual(-2000 - 160);
    expect(back.cols).toBe(Math.ceil((back.lx1 - back.lx0) / 0.5));
    const capped = landSpan(15_000, -20_000)!;
    expect(capped.cols).toBeLessThanOrEqual(MAX_LAND_COLS);
    expect(capped.lx1).toBe(landSpan(15_000)!.lx1);
  });
  it("both ends of the strip taper into the sea (no square end), and the range's start is on full land", () => {
    const s = landSpan(1071, 500)!;
    expect(taper(s.lx0, s.lx0, s.lx1)).toBe(0);
    expect(taper(s.lx1 - 20, s.lx0, s.lx1)).toBe(0);
    expect(taper(500 - 40, s.lx0, s.lx1)).toBe(1); // the range's start (and 40 units before it) is untouched
    expect(taper(1071 + 160, s.lx0, s.lx1)).toBe(1); // the front and a transplant's landing too
    let prev = 0;
    for (let x = s.lx0; x <= s.lx0 + TAPER; x += 1) {
      const k = taper(x, s.lx0, s.lx1);
      expect(k).toBeGreaterThanOrEqual(prev);
      prev = k;
    }
  });
  it("groundHeight is land from the strip's start on", () => {
    const w = () => 6;
    expect(groundHeight(-500, 0, w)).toBe(-0.3);
    expect(groundHeight(-500, 0, w, -1000)).toBeGreaterThan(0);
  });
});

describe("the follow fit frames the range's span", () => {
  const W = () => 15;
  const tree = (x: number, active: boolean) => ({ x, active, points: [{ x, y: 0, z: 0 }, { x, y: 9, z: 0 }] });
  it("with a span start: every built tree in the span counts, active or not, and the shore reaches the span's start", () => {
    const trees = [tree(-400, false), tree(-150, false), tree(90, true), tree(100, false)];
    const P = framePoints(trees, 110, W, -200);
    const xs = P.map((p) => p.x);
    expect(Math.min(...xs)).toBeLessThanOrEqual(-200);
    expect(xs).toContain(-150); // an idle tree inside the span
    expect(xs).not.toContain(-400); // outside it
    expect(Math.max(...P.map((p) => p.y))).toBe(9);
  });
  it("without a span: unchanged (the active trees near the front)", () => {
    const trees = [tree(-150, false), tree(90, true)];
    expect(framePoints(trees, 110, W).map((p) => p.x)).not.toContain(-150);
  });

  it("a switch glides to the new fit within 3 s with no frame step > d/30", () => {
    const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
    const cam = new CameraController({ fov: 36, aspect: 2.5 });
    cam.setMode("follow");
    cam.follow([{ x: 80, y: 0, z: -15 }, { x: 116, y: 9, z: 15 }], S); // 1 h
    cam.cutToFit();
    for (let i = 1; i <= 120; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 16.7 });
    cam.follow([{ x: -400, y: 0, z: -15 }, { x: 116, y: 9, z: 15 }], S); // 24 h
    cam.glideToFit();
    const goal = cam.followGoal()!, d0 = Math.hypot(cam.pose().pos.x - goal.x, cam.pose().pos.y - goal.y, cam.pose().pos.z - goal.z);
    let prev: V3 = cam.pose().pos, maxStep = 0, back: number | null = null;
    for (let i = 1; i <= 5 * 60; i++) {
      cam.step(1 / 60, { rel: 0, nowMs: 2000 + i * 16.7 });
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, Math.hypot(p.x - prev.x, p.y - prev.y, p.z - prev.z));
      if (back === null && Math.hypot(p.x - goal.x, p.y - goal.y, p.z - goal.z) < 0.05) back = i / 60;
      prev = p;
    }
    expect(d0).toBeGreaterThan(50);
    expect(back).not.toBeNull();
    expect(back!).toBeLessThanOrEqual(3);
    expect(maxStep).toBeLessThanOrEqual(d0 / 30);
  });
});

describe("the switch's glide keeps its 3 s deadline; a hand-off settles gently", () => {
  const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
  const d = (a: V3, b: V3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  const settled = () => {
    const cam = new CameraController({ fov: 36, aspect: 2.5 });
    cam.setMode("follow");
    cam.follow([{ x: 80, y: 0, z: -15 }, { x: 116, y: 9, z: 15 }], S);
    cam.cutToFit();
    for (let i = 1; i <= 120; i++) cam.step(1 / 60, { rel: 0, nowMs: i * 16.7 });
    return cam;
  };

  it("a re-fit after the glide ended but inside the 3 s (trees still landing) is a planned return, never a lurch", () => {
    const cam = settled();
    cam.follow([{ x: -100, y: 0, z: -15 }, { x: 116, y: 9, z: 15 }], S);
    cam.glideToFit();
    const d0 = d(cam.pose().pos, cam.followGoal()!);
    let prev = cam.pose().pos, maxStep = 0, at: number | null = null, d1 = 0, lateStep = 0;
    for (let i = 1; i <= 6 * 60; i++) {
      if (i === 156) {
        cam.follow([{ x: -100, y: 0, z: -15 }, { x: 116, y: 16, z: 15 }], S); // 2.6 s: a tall tree landed
        d1 = d(cam.pose().pos, cam.followGoal()!);
      }
      cam.step(1 / 60, { rel: 0, nowMs: 3000 + i * 16.7 });
      const p = cam.pose().pos;
      maxStep = Math.max(maxStep, d(p, prev));
      if (i > 156) lateStep = Math.max(lateStep, d(p, prev));
      prev = p;
      if (i >= 156 && at === null && d(p, cam.followGoal()!) < 0.05) at = i / 60;
    }
    expect(at).not.toBeNull();
    expect(at!).toBeLessThanOrEqual(2.6 + 2.4 + 1 / 60); // a full RETURN_S return from the re-fit
    expect(maxStep).toBeLessThanOrEqual(d0 / 30);
    expect(lateStep).toBeLessThanOrEqual(Math.max(0.1, d1 / 30)); // the late re-fit's own move: no lurch
  });

  it("settleToFit: from a hand-off 120 units off the fit, ≤ 0.1 per frame for 30 frames, there within 10 s, never fast", () => {
    const cam = settled();
    const goal = cam.followGoal()!, p0 = { x: goal.x - 90, y: goal.y - 20, z: goal.z - 76 };
    cam.placeInitial(p0, { x: p0.x + 5, y: 0, z: p0.z - 40 });
    cam.follow([{ x: 80, y: 0, z: -15 }, { x: 116, y: 9, z: 15 }], S); // the receiving scene's fit
    cam.settleToFit();
    let prev = cam.pose().pos, first30 = 0, maxStep = 0, at: number | null = null;
    for (let i = 1; i <= 12 * 60; i++) {
      if (i === 180) cam.follow([{ x: 80, y: 0, z: -15 }, { x: 118, y: 11, z: 15 }], S); // the fit moves a little
      cam.step(1 / 60, { rel: 0, nowMs: 3000 + i * 16.7 });
      const p = cam.pose().pos, s = d(p, prev);
      if (i <= 30) first30 = Math.max(first30, s);
      maxStep = Math.max(maxStep, s);
      prev = p;
      if (at === null && d(p, cam.followGoal()!) < 0.5) at = i / 60;
    }
    expect(first30).toBeLessThanOrEqual(0.1);
    expect(at).not.toBeNull();
    expect(at!).toBeLessThanOrEqual(10);
    expect(maxStep).toBeLessThanOrEqual(0.5);
  });
});

describe("the card's tree budget", () => {
  it("12 trees for 1h and 6h, 30 for 24h and 7d", () => {
    expect(["1h", "6h", "24h", "7d"].map((r) => cardTreeBudget(r as Range))).toEqual([12, 12, 30, 30]);
  });
});

describe("a far fit stays visible (review I2)", () => {
  it("near, far and fog are unchanged near the forest", () => {
    expect(viewDepth(60)).toEqual({ near: 0.1, far: FAR_MIN, fog: FOG_DENSITY });
    expect(viewDepth(150)).toEqual({ near: 0.1, far: FAR_MIN, fog: FOG_DENSITY });
  });
  it.each([650, 1600, 3000])("at %i u: depth precision at the target as good as at 150 u today, haze kept, the far plane fogged out", (d) => {
    const v = viewDepth(d), ref = depthStep(150, 0.1, FAR_MIN);
    console.log(`viewDepth(${d}): near ${v.near.toFixed(2)}, far ${v.far}, fog ${v.fog.toExponential(3)}, depth step at target ${depthStep(d, v.near, v.far).toFixed(4)} (ref ${ref.toFixed(4)})`);
    expect(depthStep(d, v.near, v.far)).toBeLessThanOrEqual(ref * 1.0001);
    expect(v.near).toBeLessThanOrEqual(d / 4);
    expect(v.far).toBeGreaterThanOrEqual(5 * d);
    expect(Math.exp(-((v.far * v.fog) ** 2))).toBeLessThan(0.03); // anything at the far plane is fog
    expect(Math.exp(-((d * v.fog) ** 2))).toBeGreaterThan(0.85); // the forest at the target keeps its contrast
    expect(Math.exp(-((d * v.fog) ** 2))).toBeLessThan(0.99); // and still some haze
  });
  it("the sea reaches past the far plane from the camera, whole wave tiles, centred under the camera", () => {
    for (const [x, z, reach] of [[1234.5, 1550, 8000], [-3000, 40, 900], [10, -20, 100]] as const) {
      const f = seaFit(x, z, reach);
      const half = f.size / 2;
      expect(half).toBeGreaterThanOrEqual(Math.max(600, reach));
      expect(Math.abs(f.x - x)).toBeLessThanOrEqual(SEA_TILE / 2 + 1e-9);
      expect(Math.abs(f.z - z)).toBeLessThanOrEqual(SEA_TILE / 2 + 1e-9);
      // every edge is at least `reach` from the camera (beyond the far plane)
      expect(Math.min(f.x + half - x, x - (f.x - half), f.z + half - z, z - (f.z - half))).toBeGreaterThanOrEqual(reach - SEA_TILE / 2);
      expect(Number.isInteger(Math.round(f.tiles * 1e9) / 1e9)).toBe(true);
    }
  });
});

describe("zoom-out limit (review M2)", () => {
  it("is bounded by the fit's distance, not the camera's: repeated wheel-outs cannot ratchet it", () => {
    expect(zoomOutLimit(null)).toBe(MAX_ZOOM_OUT);
    expect(zoomOutLimit(60)).toBe(MAX_ZOOM_OUT);
    expect(zoomOutLimit(1000)).toBe(1500);
    // the scene's rule at each drag start: max(limit, min(current, previous max))
    let max = MAX_ZOOM_OUT, cur = 200;
    for (let k = 0; k < 50; k++) {
      max = Math.max(zoomOutLimit(400), Math.min(cur, max));
      cur = max; // the user zooms out as far as allowed
    }
    expect(max).toBe(600);
  });
});
