// frontend/tests/component/forest-frame.test.ts
// The scene's per-frame orchestration and data intake (engine/frame.ts), without three.js: clock, camera step and pose,
// scene time across t0 re-anchors, setEvents only on change, and the rebuild schedule.
import { describe, expect, it, vi } from "vitest";
import { CameraController, FOLLOW_PAUSE_MS } from "@/components/forest/engine/camera";
import {
  FrameCore, PREBUILD_S, cutOnJump, nextEnd, planTrees, sceneTimeline, staleBuild, treeInfo, type CameraLike, type ReqRecord, type TreeInfo,
} from "@/components/forest/engine/frame";
import { applyResponse } from "@/lib/forest/client";
import type { ForestResponse, ForestState, Tree, Turn } from "@/lib/forest/types";
import { buildShown } from "@/lib/forest/geometry/worker";
import { buildTreeGeometry } from "@/lib/forest/geometry";
import { CHOSEN } from "@/lib/forest/voxel";
import { GIRTH_RATE, SCALE_IN_S, girthOf, scaleIn, topOf } from "@/components/forest/engine/voxels";
import { TREE_SCALE, buildTimeline } from "@/lib/forest/timeline";

const TRAITS = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
/** A tree whose turns happened at absolute times `abs`, sent against anchor `t0` (rel rounded to 0.1 s, as the server does). */
const treeAt = (id: string, abs: number[], t0: number): Tree => {
  const r = (a: number) => Math.round((a - t0) * 10) / 10;
  const turns = abs.map((a): Turn => [r(a), 37000, 150, 30000, [], ["read"], "tool_calls", 1]);
  return { id, key: "k", start: turns[0][0], end: turns.at(-1)![0], last: turns.at(-1)![0], last_at: 0, traits: TRAITS,
    sessions: [{ id: "S-" + id, variant: "v1", children: [], turns }] };
};
const resp = (t0: number, nowAbs: number, trees: Tree[], o: Partial<ForestResponse> = {}): ForestResponse => ({
  range: "48h", t0, now: nowAbs, models: {}, trees, flowers: [], ids: trees.map((t) => t.id),
  full: true, cursor: nowAbs, ...o,
});
const fakeCam = (): CameraLike & { setEvents: ReturnType<typeof vi.fn> } => ({
  step: () => false, pose: () => ({ pos: { x: 0, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 } }), paused: false, setEvents: vi.fn(),
});
/** Tree infos of a scene-time state (the timeline is laid out in layout time = scene + H). */
const infosOf = (s: ForestState, H: number): Map<string, TreeInfo> => {
  const tl = sceneTimeline([...s.trees.values()], s.now, H);
  return new Map([...s.trees].map(([id, t]) => [id, treeInfo(t, tl.places(id))]));
};
const rec = (cut: number, times: readonly number[]) => ({ cut, times });

describe("frame core: camera", () => {
  it("steps the camera every frame: a drag pause ends 9 s after the drag, and the pose is applied every frame again", () => {
    // above the ground: a held camera below it is lifted (and that pose applied) at once (review M1)
    const cam = new CameraController({ fov: 36, aspect: 1.5, pos: { x: 0, y: 10, z: 30 }, target: { x: 0, y: 1, z: 0 } });
    const core = new FrameCore(cam);
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", [1500, 1600, 1950], 1000)])).state, 0);
    cam.setDragging(true, 0);
    cam.setDragging(false, 1000);
    const poses: [number, boolean][] = [];
    for (let t = 1000; t <= 12_000; t += 100) poses.push([t, core.frame(t, 0.1)!.pose !== null]);
    for (const [t, has] of poses) expect(has, `t=${t}`).toBe(t > 1000 + FOLLOW_PAUSE_MS);
    expect(cam.paused).toBe(false);
  });
});

describe("frame core: scene time (t0 re-anchor)", () => {
  const abs = [1500.04, 1530.66, 1561.27, 1950.95];
  it("a re-anchor by 299.97 s changes no scene time, no place and schedules no build", () => {
    const cam = fakeCam(), core = new FrameCore(cam);
    const a = core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", abs, 1000)])).state, 0).state;
    const timesA = a.trees.get("a")!.sessions[0].turns.map((t) => t[0]);
    for (let k = 0; k < 600; k++) core.frame(k * 100, 0.1);
    const relA = core.clock!.rel, infosA = infosOf(a, core.H);
    const plan = planTrees(infosA, new Map(), relA, 0);
    const req = new Map(plan.builds.map((b) => [b.id, rec(b.cut, infosA.get(b.id)!.times)]));
    // a full refresh 60 s later, anchored on a float t0
    const t0b = 1299.97;
    const b = core.intake(applyResponse(null, resp(t0b, 2060, [treeAt("a", abs, t0b)])).state, 60_000).state;
    expect(b.trees.get("a")!.sessions[0].turns.map((t) => t[0])).toEqual(timesA);
    expect(b.trees.get("a")!.start).toBe(a.trees.get("a")!.start);
    expect(b.now).toBeCloseTo(a.now + 60, 1);
    expect(core.clock!.rel).toBe(relA); // the clock did not move on intake
    const infosB = infosOf(b, core.H);
    expect(infosB.get("a")!.places).toEqual(infosA.get("a")!.places);
    expect(planTrees(infosB, req, relA, 0).builds).toEqual([]);
    expect(cam.setEvents).toHaveBeenCalledTimes(1);
  });
  it("the clock keeps scene time across the re-anchor (no jump back, no 300 s hold)", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", abs, 1000)])).state, 0);
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    const before = core.clock!.rel;
    core.intake(applyResponse(null, resp(1299.97, 2010, [treeAt("a", abs, 1299.97)])).state, 10_000);
    const out = core.frame(10_100, 0.1)!;
    expect(out.rel - before).toBeGreaterThan(0.05);
    expect(out.rel - before).toBeLessThan(0.2);
    // the API speaks absolute time: scene time + the origin (the first state's now, 2000)
    expect(core.shownAbs()).toBeCloseTo(out.rel + 2000, 9);
    expect(core.origin).toBe(2000);
  });
});

describe("frame core: setEvents only on data change", () => {
  it("same object, an equal copy and a re-anchored copy do not replan; a new turn does", () => {
    const cam = fakeCam(), core = new FrameCore(cam);
    const s1 = applyResponse(null, resp(1000, 2000, [treeAt("a", [1500, 1600], 1000)])).state;
    core.intake(s1, 0);
    core.intake(s1, 10);
    core.intake(structuredClone(s1), 20);
    core.intake(applyResponse(null, resp(1299.97, 2001, [treeAt("a", [1500, 1600], 1299.97)])).state, 30);
    expect(cam.setEvents).toHaveBeenCalledTimes(1);
    core.intake(applyResponse(null, resp(1000, 2002, [treeAt("a", [1500, 1600, 1990], 1000)])).state, 40);
    expect(cam.setEvents).toHaveBeenCalledTimes(2);
    expect(cam.setEvents.mock.calls[1][0].map((e: { time: number }) => e.time)).toEqual([-500, -400, -10]); // scene time: absolute − the first now (2000)
  });
});

describe("frame core: rebuild schedule", () => {
  const tree: Tree = treeAt("a", [1010, 1050, 1090], 1000);
  const info = (dx = 0): Map<string, TreeInfo> =>
    new Map([["a", treeInfo(tree, [{ from: 10, x: 5 + dx, z: 0, lane: 0 }])]]);

  it("builds when first seen, then exactly at each turn and every 20 s, requested 2 s early", () => {
    const req = new Map<string, ReqRecord>(), got: [number, number][] = [];
    for (let rel = 10; rel <= 120; rel += 0.25) {
      for (const b of planTrees(info(), req, rel, 0).builds) {
        got.push([rel, b.cut]);
        req.set(b.id, rec(b.cut, info().get(b.id)!.times));
      }
    }
    expect(got.map(([, c]) => c)).toEqual([10, 30, 50, 70, 90, 110]);
    expect(PREBUILD_S).toBe(2);
    for (const [rel, cut] of got.slice(1)) expect(rel).toBe(cut - 2);
  });
  it("a place move schedules no build", () => {
    const req = new Map([["a", rec(30, [10, 50, 90])]]);
    expect(planTrees(info(), req, 40, 0).builds).toEqual([]);
    expect(planTrees(info(3.7), req, 40, 0).builds).toEqual([]);
  });
  it("a build is on screen until the next hand-over (the next turn or 20 s); no morph toward it", () => {
    const [b] = planTrees(info(), new Map(), 31, 0).builds;
    expect(b).toEqual({ id: "a", cut: 31 });
    expect(nextEnd(info().get("a")!.times, 31, 1e9)).toBe(50);
  });
});

describe("frame core: late turns grow from their arrival", () => {
  const knobs = { girth: 1.15, tall: 0.8, el: 0.48 };
  const norm = { len: 0.15, foliage: 0.15, models: {} };
  /** The tree's cells at `cut` (a turn's own cells are born at its effective time). */
  const grown = (t: Tree, cut: number) => buildTreeGeometry(t, cut, knobs, norm, { x: 0, z: 0, s: 1 }, t.traits).voxels;

  it("a turn arriving 40 s after its time grows over the normal 10 s from its arrival; rebuilds keep it; the camera sees it then", () => {
    const cam = fakeCam(), core = new FrameCore(cam);
    const base = [1500, 1600, 1700];
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", base, 1000)])).state, 0);
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1); // 60 s on: rel ≈ 1030 (scene time = abs − 1000)
    const arrival = core.clock!.rel;
    // the request started 40 s before the shown time; its row arrives only now
    const late = arrival - 40 + 2000; // absolute (scene time + the origin, 2000)
    const s2 = core.intake(applyResponse(null, resp(1000, 2060, [treeAt("a", [...base, late], 1000)])).state, 60_000).state;
    const turns = s2.trees.get("a")!.sessions[0].turns;
    expect(turns.map((t) => t[0])).toEqual([-500, -400, -300, arrival]);
    // the camera's event for that turn is at its arrival
    const ev = cam.setEvents.mock.calls.at(-1)![0] as { time: number; sessionId: string }[];
    expect(ev.at(-1)).toMatchObject({ time: arrival, sessionId: "S-a" });

    // its growth is born at the arrival (the voxelizer's inputs: its wood and foliage), and so is every cell it claims:
    // nothing of them shows then, and they are full grown 0.8 s later
    const b = grown(s2.trees.get("a")!, arrival), at = Math.fround(arrival);
    const vin = buildTreeGeometry(s2.trees.get("a")!, arrival, knobs, norm, { x: 0, z: 0, s: 1 }, TRAITS).vox;
    const wood = [...vin.wood].filter((x, i) => i % 6 === 3 && x === at), fol = [...vin.foliage].filter((x, i) => i % 6 === 4 && x === at);
    expect(wood.length + fol.length).toBeGreaterThan(0);
    const own = [...b.born].filter((x) => x === at);
    expect(scaleIn(arrival - at)).toBeLessThan(1e-3);
    expect(scaleIn(arrival + SCALE_IN_S / 2 - at)).toBeGreaterThan(0.4);
    expect(scaleIn(arrival + SCALE_IN_S - at)).toBe(1);
    // (a wood block's outer cells are born one scale-in after it: they grow from the inside out)
    expect([...b.born].every((x) => x <= Math.fround(at + SCALE_IN_S))).toBe(true);

    // later polls (and a float re-anchor) keep the same effective time, so a rebuild reproduces the same born
    for (let k = 601; k <= 900; k++) core.frame(k * 100, 0.1);
    const t0b = 1299.97;
    const s3 = core.intake(applyResponse(null, resp(t0b, 2090, [treeAt("a", [...base, late], t0b)])).state, 90_000).state;
    expect(s3.trees.get("a")!.sessions[0].turns.map((t) => t[0])).toEqual([-500, -400, -300, arrival]);
    const b2 = grown(s3.trees.get("a")!, arrival + 30);
    expect([...b2.born].filter((x) => x === at).length).toBe(own.length);
  });
  it("on the first data nothing is late: history keeps its times", () => {
    const core = new FrameCore(fakeCam());
    const s = core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", [1500, 1990], 1000)])).state, 0).state;
    expect(s.trees.get("a")!.sessions[0].turns.map((t) => t[0])).toEqual([-500, -10]);
    // the origin is the first now: scene time starts at 0, the clock 30 s behind (float32 resolution is best there)
    expect(core.origin).toBe(2000);
    expect(core.clock!.rel).toBe(-30);
  });
});

describe("frame core: a turn inside the prebuild window (N1)", () => {
  const knobs = { girth: 1.15, tall: 0.8, el: 0.48 };
  const norm = { len: 0.15, foliage: 0.15, models: {} };
  const mk = (times: number[]) => {
    const t = treeAt("a", times.map((x) => x + 1000), 1000);
    return new Map([["a", treeInfo(t, [{ from: 0, x: 5, z: 0, lane: 0 }])]]);
  };
  it("a late turn at 29 after a prebuild to 30 is built at once and grows from its effective born; the stale build is not shown", () => {
    const before = mk([0, 10]);
    // the prebuild for the 20 s hand-over at 30 went out at 28.5 with turns [0, 10]
    const [pre] = planTrees(before, new Map([["a", rec(10, [0, 10])]]), 28.5, 0).builds;
    expect(pre).toMatchObject({ cut: 30 });
    const req = new Map([["a", rec(30, before.get("a")!.times)]]);
    // the late turn arrives: its effective time is the arrival, 29
    const after = mk([0, 10, 29]);
    const [b] = planTrees(after, req, 29, 0).builds;
    expect(b).toEqual({ id: "a", cut: 29 });
    // the waiting build for 30 lacks it: stale, never shown; the new one has it
    expect(staleBuild(before.get("a")!.times, after.get("a")!.times, 30)).toBe(true);
    expect(staleBuild(after.get("a")!.times, after.get("a")!.times, 29)).toBe(false);
    // and it grows from zero at 29
    const g = buildShown({ tree: after.get("a")!.tree, cut: b.cut, knobs, norm, place: { x: 0, z: 0, s: 1 }, traits: TRAITS }).levels[0];
    const own = [...g.born].filter((x) => x === 29);
    expect(own.length).toBeGreaterThan(0);
    expect(scaleIn(29 - own[0])).toBe(0);
    // once built, nothing more is due until the next hand-over
    req.set("a", rec(29, after.get("a")!.times));
    expect(planTrees(after, req, 29.25, 0).builds).toEqual([]);
  });
  it("a turn arriving 1 s ahead of its time inside the window is built at its own time", () => {
    const req = new Map([["a", rec(30, [0, 10])]]);
    expect(planTrees(mk([0, 10, 29.5]), req, 28.6, 0).builds).toEqual([{ id: "a", cut: 29.5 }]);
  });
});

describe("frame core: long sessions (server LOD is prefix-stable)", () => {
  it("a session growing past 90 turns never changes the times of its earlier turns", () => {
    const cam = fakeCam(), core = new FrameCore(cam);
    const at = (n: number) => Array.from({ length: n }, (_, i) => 1000 + i * 10); // absolute
    // shown time at load: 1871 − 30 = 1841, before the 86th turn (1850); the clock is not advanced here
    core.intake(applyResponse(null, resp(1000, 1871, [treeAt("a", at(85), 1000)])).state, 0);
    let prev = core.state!.trees.get("a")!.sessions[0].turns.map((t) => t[0]);
    for (let n = 86; n <= 100; n++) {
      // each new turn reaches the data ahead of the shown time (now − 30 s)
      const s = core.intake(applyResponse(null, resp(1000, 1000 + (n - 1) * 10 + 31, [treeAt("a", at(n), 1000)])).state, n * 10_000).state;
      const times = s.trees.get("a")!.sessions[0].turns.map((t) => t[0]);
      expect(times.slice(0, prev.length), `n=${n}`).toEqual(prev);
      expect(times.at(-1)).toBeCloseTo(at(n).at(-1)! - core.origin!, 6); // not pinned as late
      prev = times;
    }
    expect(prev).toHaveLength(100);
  });
});

describe("frame core: late turns under the server's block LOD (M1)", () => {
  /** The server's prefix-stable LOD (forest.py merge_turns) at test scale: `cap` singles, then closed blocks of 8. */
  const lod = (abs: number[], cap: number, block = 8) => {
    if (abs.length <= cap) return abs;
    const rest = abs.slice(cap), full = rest.length - (rest.length % block), out = abs.slice(0, cap);
    for (let i = 0; i < full; i += block) out.push(rest[i + block - 1]); // a block's time is its last turn's
    return [...out, ...rest.slice(full)];
  };
  it("a closing block never leaves a turn pinned to another turn's arrival, and times never go backwards", () => {
    const core = new FrameCore(fakeCam());
    const abs = (j: number) => 1000 + 10 * j;
    let ms = 0;
    for (let n = 2; n <= 22; n++) {
      // the newest turn's row arrives 15 s after the shown time passed its start: late
      const all = Array.from({ length: n }, (_, j) => abs(j));
      const s = core.intake(applyResponse(null, resp(1000, abs(n - 1) + 45, [treeAt("a", lod(all, 3), 1000)])).state, ms).state;
      const times = s.trees.get("a")!.sessions[0].turns.map((t) => t[0]);
      expect(times, `n=${n}`).toHaveLength(lod(all, 3).length);
      for (let i = 1; i < times.length; i++) expect(times[i], `n=${n} i=${i}`).toBeGreaterThanOrEqual(times[i - 1]);
      // a merged block keeps a time at or after its own last turn's start (never an older turn's arrival)
      const raw = lod(all, 3).map((a) => a - core.origin!);
      times.forEach((t, i) => expect(t, `n=${n} i=${i}`).toBeGreaterThanOrEqual(raw[i] - 0.15));
      ms += 20_000;
      core.frame(ms, 20); // the shown time jumps to the bound
    }
  });
});

describe("tree size comes only from growth (R25)", () => {
  it("a tree crossing 60 requests keeps its trunk girth and height: no step beyond the per-frame growth", () => {
    const abs = Array.from({ length: 70 }, (_, j) => 1000 + 2 * j);
    const tree = treeAt("a", abs, 1000), times = tree.sessions[0].turns.map((t) => t[0]), freezeAt = times.at(-1)! + 1800;
    // the timeline's places carry no size: the build is always at scale 1, before and after 60 requests
    // a neighbour planted 3 s later, right next to it: the two must not share a lane, before or after "a" passes 60
    const b = treeAt("b", [1003, 1050, 1100], 1000);
    const before = buildTimeline([treeAt("a", abs.slice(0, 60), 1000), b], 200), after = buildTimeline([tree, b], 200);
    expect(before.places("a")[0]).not.toHaveProperty("s");
    expect(TREE_SCALE).toBe(1);
    expect(Math.abs(before.places("b")[0].x - before.places("a")[0].x)).toBeLessThan(5.5 * 2); // they overlap on x
    expect(before.places("b")[0].z).not.toBe(before.places("a")[0].z);
    for (const id of ["a", "b"]) expect(after.places(id)[0].z, `lane of ${id}`).toBe(before.places(id)[0].z);
    const knobs = { girth: 1.15, tall: 0.8, el: 0.48 }, norm = { len: 0.15, foliage: 0.15, models: {} };
    const job = (cut: number) => buildShown({ tree, cut, knobs, norm, place: { x: 0, z: 0, s: TREE_SCALE }, traits: TRAITS }).levels[0];
    const e = CHOSEN.size;
    // hand-overs around the 60th, 61st and 62nd request (turns 2 s apart: every turn is a swap)
    for (const c of [times[58], times[59], times[60], times[61]]) {
      const old = job(c), T = nextEnd(times, c, freezeAt), nu = job(T);
      expect(Math.abs(girthOf(old, T, e) - girthOf(nu, T, e)), `girth at ${c}→${T}`).toBeLessThan(1e-9);
      expect(Math.abs(topOf(old, T, e) - topOf(nu, T, e)), `height at ${c}→${T}`).toBeLessThan(1e-9);
      // and within one build, a 60 fps frame moves the girth only as fast as a cube scales in, never back
      for (let k = 1; k <= (T - c) * 60; k++) {
        const d = girthOf(old, c + k / 60, e) - girthOf(old, c + (k - 1) / 60, e);
        expect(d).toBeGreaterThanOrEqual(-1e-12);
        expect(d).toBeLessThanOrEqual(GIRTH_RATE / 60 + 1e-9);
      }
    }
  });
});

describe("frame core: pause and resume (the Stats card offscreen)", () => {
  /** Turns every `every` s in absolute time from 1000 up to `until` (abs), sent against t0 = 1000. */
  const live = (every: number, until: number) => treeAt("a", Array.from({ length: Math.floor((until - 1000) / every) + 1 }, (_, i) => 1000 + i * every), 1000);
  const turnTimes = (core: FrameCore) => core.state!.trees.get("a")!.sessions[0].turns.map((t) => t[0]);

  it("card resumes without burst after offscreen: the queued state is applied, the clock jumps, nothing is late", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [live(20, 2000)])).state, 0); // scene time: abs − 2000
    expect(core.paused).toBe(false);
    core.pause();
    expect(core.paused).toBe(true);
    const n0 = turnTimes(core).length;
    // an hour of turns arrives while paused: only queued, not applied
    const r = core.intake(applyResponse(null, resp(1000, 5600, [live(20, 5600)])).state, 3_600_000);
    expect(r.changed).toBe(false);
    expect(turnTimes(core)).toHaveLength(n0);
    core.resume(3_600_000);
    expect(core.paused).toBe(false);
    expect(turnTimes(core).length).toBeGreaterThan(n0);
    expect(core.lateCount()).toBe(0);
    expect(core.clock!.rel).toBeCloseTo(3600 - 30, 0);
    // every turn keeps its own time: the hour is history, no burst of turns growing from the resume
    expect(turnTimes(core).at(-1)).toBeCloseTo(3600, 1);
    expect(core.isHistory(3600 - 30 - 1)).toBe(true);
    expect(core.takeStep().continuous).toBe(false);
  });

  it("card resumes after a long pause with nothing queued: the clock jumps, then the hour's poll is history, not late", () => {
    // the card polls nothing while paused, so resume() has no queued state: the jump comes from the server-now estimate
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [live(20, 2000)])).state, 0);
    const prev = applyResponse(null, resp(1000, 2000, [live(20, 2000)])).state;
    const n0 = turnTimes(core).length;
    core.pause();
    expect(core.resume(3_600_000)).toBeNull(); // nothing queued
    expect(core.paused).toBe(false);
    expect(turnTimes(core)).toHaveLength(n0);
    expect(core.clock!.rel).toBeCloseTo(3600 - 30, 0);
    expect(core.takeStep().continuous).toBe(false);
    // the first poll after resume brings the whole hour (≤ 2 s later)
    const r = core.intake(applyResponse(prev, resp(1000, 5602, [live(20, 5600)], { full: false })).state, 3_602_000);
    expect(r.changed).toBe(true);
    expect(turnTimes(core).length).toBeGreaterThan(n0);
    expect(core.lateCount()).toBe(0); // no burst: the hour fell inside the skipped stretch
    expect(core.isHistory(3600 - 30 - 1)).toBe(true);
    expect(core.isHistory(1800)).toBe(true);
  });

  it("paused core does not advance", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [live(20, 2000)])).state, 0);
    for (let k = 1; k <= 50; k++) core.frame(k * 100, 0.1);
    const t = core.clock!.rel;
    core.pause();
    expect(core.frame(10_000, 5)).toBeNull();
    expect(core.clock!.rel).toBe(t);
  });

  it("a 5 s pause is no jump: no history band, and a late turn queued during it grows from its arrival", () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", [1500], 1000)])).state, 0);
    for (let k = 1; k <= 600; k++) core.frame(k * 100, 0.1); // rel ≈ 30
    core.takeStep();
    const arrival = core.clock!.rel;
    core.pause();
    // a request that started 40 s before the shown time finishes while paused
    core.intake(applyResponse(null, resp(1000, 2063, [treeAt("a", [1500, arrival - 40 + 2000], 1000)])).state, 63_000);
    expect(core.lateCount()).toBe(0); // queued, not applied
    core.resume(65_000);
    expect(turnTimes(core).at(-1)).toBe(arrival); // grows from its arrival
    expect(core.lateCount()).toBe(1);
    // the floor was not raised: the paused stretch is not history; the clock catches up continuously
    expect(core.clock!.rel).toBe(arrival);
    for (const t of [arrival + 0.1, arrival + 2.5, arrival + 5]) expect(core.isHistory(t)).toBe(false);
    for (let k = 1; k <= 100; k++) core.frame(65_000 + k * 100, 0.1);
    expect(core.takeStep().continuous).toBe(true);
    expect(core.clock!.rel).toBeCloseTo(65 + 10 - 30, 1); // caught up with the bound
  });
});

describe("front-relative hand-off: the card's 6 h layout and the full window's 48 h one agree near the front", () => {
  it("the same tree's x − X(shown) is equal (±0.1) in both, whatever second of the minute each window's t0 falls on", () => {
    const NOW = 1_800_000_000.4;
    const all: number[][] = [];
    for (let k = 0; k < 120; k++) {
      const s = NOW - 47 * 3600 + k * 1370 + (k % 7) * 97;
      if (s > NOW - 100) break;
      all.push(Array.from({ length: 3 + (k % 9) }, (_, j) => Math.min(NOW - 40, s + j * (35 + (k % 5) * 20))));
    }
    // the full window opens 23.6 s after the card's data: its window (and t0) is offset by a non-whole minute
    const fullNow = NOW + 23.6, cardNow = NOW;
    const t48 = fullNow - 48 * 3600, t6 = cardNow - 6 * 3600;
    const at = (core: FrameCore, t0: number, now: number, trees: [string, number[]][]) => {
      core.intake(applyResponse(null, resp(t0, now, trees.map(([id, abs]) => treeAt(id, abs, t0)))).state, 0);
      return sceneTimeline([...core.state!.trees.values()], core.state!.now, core.H);
    };
    const full = new FrameCore(fakeCam()), card = new FrameCore(fakeCam());
    const tlFull = at(full, t48, fullNow, all.map((a, k) => [`t${k}`, a]));
    // the card sees only the last 6 h; trees in its first 30 min (the window's edge) are left out of the comparison
    const inCard = all.filter((a) => a[0] >= t6);
    const tlCard = at(card, t6, cardNow, inCard.map((a) => [`t${all.indexOf(a)}`, a]));
    const shown = NOW - 30; // the same absolute shown time on both surfaces
    const rel = (core: FrameCore, tl: ReturnType<typeof sceneTimeline>, id: string) => tl.places(id)[0].x - tl.X(shown - core.origin!);
    const near = inCard.filter((a) => a[0] >= t6 + 1800);
    expect(near.length).toBeGreaterThan(8);
    for (const a of near) {
      const id = `t${all.indexOf(a)}`;
      expect(Math.abs(rel(card, tlCard, id) - rel(full, tlFull, id)), id).toBeLessThanOrEqual(0.1);
    }
  });
});

describe("frame core: a discontinuous step asks the scene to cut (review I1)", () => {
  const start = () => {
    const core = new FrameCore(fakeCam());
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", [1500], 1000)])).state, 0);
    for (let k = 1; k <= 100; k++) expect(core.frame(k * 100, 0.1)!.cut, `k=${k}`).toBe(false);
    return core;
  };
  it("an hour's pause, then resume: the first frame after it says cut, the next ones do not", () => {
    const core = start();
    core.pause();
    core.resume(3_610_000);
    expect(core.frame(3_610_016, 0.016)!.cut).toBe(true);
    expect(core.frame(3_610_033, 0.016)!.cut).toBe(false);
  });
  it("a hidden tab of 10 min and a 12 s stall cut; a 5 s pause and a 1 s hide do not", () => {
    const a = start();
    a.resumeAfterHidden(610_000);
    expect(a.frame(610_016, 0.016)!.cut).toBe(true);
    const b = start();
    b.frame(22_000, 0.1); // a 12 s frame stall: the clock jumps
    expect(b.frame(22_016, 0.016)!.cut).toBe(false); // reported once, by the stalled frame itself
    const c = start();
    c.pause();
    c.resume(15_000);
    expect(c.frame(15_016, 0.016)!.cut).toBe(false);
    const d = start();
    d.resumeAfterHidden(11_000);
    expect(d.frame(11_016, 0.016)!.cut).toBe(false);
  });
  it("the stalled frame itself reports the cut", () => {
    const core = start();
    expect(core.frame(22_000, 0.1)!.cut).toBe(true);
  });
});

describe("an hour's pause, then resume, in follow mode (review I1)", () => {
  it("the first frame after resume puts the camera at the fit for the new time (the scene's sequence)", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setMode("follow");
    const core = new FrameCore(cam);
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", [1500, 1990], 1000)])).state, 0);
    const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
    const fitAt = (x: number) => cam.follow([{ x: x - 20, y: 0, z: -8 }, { x, y: 8, z: 8 }], S);
    fitAt(0);
    expect(cam.cutToFit()).toBe(true);
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    core.pause();
    core.resume(3_610_000);
    // the scene's frame: the core's step reports the cut, the scene re-fits at the new front (22 u further on, the
    // idle front's hour) and cuts
    const out = core.frame(3_610_016, 0.016)!;
    expect(out.cut).toBe(true);
    fitAt(22);
    expect(cam.cutToFit()).toBe(true);
    expect(dist3(cam.pose().pos, cam.followGoal()!)).toBeLessThan(1e-9);
    // and it stays there (no glide left over)
    const p = cam.pose().pos;
    for (let k = 1; k <= 60; k++) core.frame(3_610_016 + k * 16, 0.016);
    expect(dist3(cam.pose().pos, p)).toBeLessThan(1e-6);
  });
});
const dist3 = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

describe("the scene's cut decision, cutOnJump (re-review N3: the wiring scene.ts runs every frame)", () => {
  const S = { l: -0.9, r: 0.9, t: 0.9, b: -0.9 };
  const rig = (mode: "follow" | "cinema") => {
    const cam = new CameraController({ fov: 36, aspect: 1.6 });
    cam.setMode("follow");
    const core = new FrameCore(cam);
    core.intake(applyResponse(null, resp(1000, 2000, [treeAt("a", [1500, 1990], 1000)])).state, 0);
    const fitAt = (x: number) => cam.follow([{ x: x - 20, y: 0, z: -8 }, { x, y: 8, z: 8 }], S);
    fitAt(0);
    cam.cutToFit();
    if (mode === "cinema") cam.setMode("cinema");
    for (let k = 1; k <= 100; k++) core.frame(k * 100, 0.1);
    return { cam, core, fitAt };
  };
  // the scene's per-frame call, as in scene.ts: out = core.frame(...); cutOnJump(out, ctl, refollow)
  const sceneFrame = (core: FrameCore, cam: CameraController, refit: () => void, ms: number) => cutOnJump(core.frame(ms, 0.016), cam, refit);

  it("follow: a resume after an hour cuts to the fit at the new time in that first frame; later frames do not cut", () => {
    const { cam, core, fitAt } = rig("follow");
    core.pause();
    core.resume(3_610_000);
    let refits = 0;
    expect(sceneFrame(core, cam, () => (refits++, fitAt(22)), 3_610_016)).toBe(true);
    expect(refits).toBe(1);
    expect(dist3(cam.pose().pos, cam.followGoal()!)).toBeLessThan(1e-9);
    expect(sceneFrame(core, cam, () => refits++, 3_610_033)).toBe(false);
    expect(refits).toBe(1);
  });
  it("cinema: the same resume does not cut, and does not re-fit", () => {
    const { cam, core } = rig("cinema");
    core.pause();
    core.resume(3_610_000);
    const before = cam.pose().pos;
    let refits = 0;
    expect(sceneFrame(core, cam, () => refits++, 3_610_016)).toBe(false);
    expect(refits).toBe(0);
    expect(dist3(cam.pose().pos, before)).toBeLessThan(1); // its own (capped) motion only, no cut
  });
  it("follow: a 5 s pause (no jump) does not cut", () => {
    const { cam, core } = rig("follow");
    core.pause();
    core.resume(15_000);
    let refits = 0;
    expect(sceneFrame(core, cam, () => refits++, 15_016)).toBe(false);
    expect(refits).toBe(0);
  });
});
