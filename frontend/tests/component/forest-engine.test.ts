// frontend/tests/component/forest-engine.test.ts
// Pure helpers of the rendering engine only: jsdom has no WebGL, so nothing here creates a renderer or a texture.
import { describe, expect, it } from "vitest";
import { hillsIn, terraceHeight } from "@/components/forest/engine/land";
import * as THREE from "three";
import { VOXEL_PATCH, type VoxelMaterial, createMaterials } from "@/components/forest/engine/materials";
import {
  BuildQueue,
  CUBE_CAP,
  FreezeTable,
  createTreeLayer,
  normFor,
  type BuildJob,
  type BuildReply,
  type Builder,
} from "@/components/forest/engine/trees";
import { createTrail } from "@/components/forest/engine/flights";
import { treeInfo } from "@/components/forest/engine/frame";
import { createDirectBuilder } from "@/components/forest/engine/scene";
import { FADE_S, LOD_D, SCALE_IN_S, type LevelCells } from "@/components/forest/engine/voxels";
import { emergePose, flightPose, shouldFly } from "@/components/forest/engine/flights";
import { buildShown, handleBuild } from "@/lib/forest/geometry/worker";
import { planTrees, rebuildDue, selectBuilt } from "@/components/forest/engine/frame";
import { decodeHandoff, encodeHandoff } from "@/components/forest/engine/handoff";
import { buildTreeGeometry } from "@/lib/forest/geometry";
import { KNOBS } from "@/lib/forest/species";
import type { ForestState, Tree, Traits } from "@/lib/forest/types";

describe("materials", () => {
  it("cubes scale in from their born (smoothstep over 0.8 s) and are tinted cyan while they grow (fading over 12 s)", () => {
    expect(VOXEL_PATCH.beginVertex).toContain("float age=uNow-iborn;float gs=clamp(age/0.8,0.0,1.0);gs=gs*gs*(3.0-2.0*gs)");
    expect(VOXEL_PATCH.beginVertex).toContain("vGlow=step(0.0,age)*clamp(1.0-age/12.0,0.0,1.0)");
    expect(VOXEL_PATCH.color).toContain("mix(diffuseColor.rgb,vec3(0.3,0.75,1.0),0.6*vGlow)");
    // no morph: no per-vertex or per-instance deltas anywhere
    expect(Object.values(VOXEL_PATCH).join()).not.toMatch(/dpos|iDelta|imk|bh/);
  });
  it("the cel material: toon with a 4-step ramp, vertex colours (face shading) times the instance colour", () => {
    const m = createMaterials();
    expect(m.voxel).toBeInstanceOf(THREE.MeshToonMaterial);
    expect(m.voxel.vertexColors).toBe(true);
    expect(m.voxel.fade.uFade.value).toBe(1); // at rest: no dither
    const a = m.fading(true), b = m.fading(false);
    expect([a.fade.uFade.value, a.fade.uFadeIn.value, b.fade.uFade.value, b.fade.uFadeIn.value]).toEqual([0, 1, 0, 0]);
    expect(a.fade).not.toBe(b.fade);
    // the cube's faces: top lit, sides darker, bottom dark (mockup shadeBox)
    const c = m.cube.getAttribute("color"), n = m.cube.getAttribute("normal"), k = new Set<string>();
    for (let i = 0; i < c.count; i++) k.add(`${n.getY(i)}:${c.getX(i).toFixed(2)}`);
    expect(k).toContain("1:1.16");
    expect(k).toContain("-1:0.55");
    m.dispose();
  });
});

const TRAITS: Traits = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
const tree = (traits: Traits, n = 6, id = "t1"): Tree => ({
  id, key: "k", start: 0, end: n * 30, last: n * 30, last_at: n * 30, traits,
  sessions: [{ id: "S-" + id, variant: "v1", children: [],
    turns: Array.from({ length: n }, (_, i) => [i * 30, 37000 + i * 500, 150, 35000, [], ["read"], "tool_calls", 1]) }],
});
const state = (trees: Tree[], models: ForestState["models"]): ForestState => ({
  t0: 0, now: 1000, range: "48h", trees: new Map(trees.map((t) => [t.id, t])), flowers: [], models, cursor: 1000,
});
const M1 = { v1: { prefill_tps: 20000, decode_tps: 1000, max_model_len: 262144 } };
const M2 = { v1: { prefill_tps: 2000, decode_tps: 50, max_model_len: 8192 } };

/** A worker result for `t` at `cut`. */
const built = (t: Tree, cut: number, s: ForestState) =>
  buildShown({ tree: t, cut, knobs: KNOBS.oak, norm: normFor(s), place: { x: 0, z: 0, s: 1 }, traits: t.traits });

/** A builder the test answers by hand. */
function fakeBuilder() {
  const jobs: BuildJob[] = [];
  const b: Builder & { jobs: BuildJob[]; reply(r: BuildReply): void } = {
    jobs,
    onReply: null,
    post: (j) => void jobs.push(j),
    dispose: () => {},
    reply(r) { b.onReply?.(r); },
  };
  return b;
}

describe("trees: per-tree freeze (R17)", () => {
  it("species, knobs, traits and normalization are fixed at first sight", () => {
    const ft = new FreezeTable();
    const a = ft.get(tree(TRAITS), state([tree(TRAITS)], M1));
    // the server's whole-spell traits drift (now cactus-like) and the model stats change: nothing moves
    const drifted = tree({ ...TRAITS, fail: 0.9, sys0: 90000 }, 9);
    const b = ft.get(drifted, state([drifted], M2));
    expect(b).toBe(a);
    expect(b.species).toBe(a.species);
    expect(b.species).not.toBe("cactus");
    expect(b.traits).toEqual(TRAITS);
    expect(b.norm.models.v1).toEqual(M1.v1);
    // a different tree gets its own
    expect(ft.get(tree(TRAITS, 6, "t2"), state([], M2)).norm.models.v1).toEqual(M2.v1);
  });
  it("the build job of a drifted tree carries the frozen inputs, so the geometry is identical", () => {
    const fb = fakeBuilder();
    const q = new BuildQueue(fb, () => {});
    const t0 = tree(TRAITS);
    q.request("t1", t0, state([t0], M1), 200, 1);
    fb.reply({ id: fb.jobs[0].id, buffers: built(t0, 200, state([t0], M1)) });
    const drifted = { ...t0, traits: { ...TRAITS, fail: 0.9, ctx_gen: 3 } };
    q.request("t1", drifted, state([drifted], M2), 200, 1);
    const [j1, j2] = fb.jobs;
    expect(j2.knobs).toEqual(j1.knobs);
    expect(j2.norm).toEqual(j1.norm);
    expect(j2.traits).toEqual(j1.traits);
    const g = (j: BuildJob) => buildTreeGeometry(j.tree, j.cut, j.knobs, j.norm, j.place, j.traits);
    expect(Array.from(g(j2).vox.wood)).toEqual(Array.from(g(j1).vox.wood));
    expect(Array.from(g(j2).voxels.coords)).toEqual(Array.from(g(j1).voxels.coords));
  });
  it("normalization reference is the median turn of the window", () => {
    const n = normFor(state([tree(TRAITS)], M1));
    expect(n.len).toBeCloseTo(150 / 1000, 9);
    expect(n.foliage).toBeGreaterThan(0);
    expect(n.models).toEqual(M1);
    // no turns: a neutral reference, never 0 or NaN
    const e = normFor(state([], {}));
    expect(e.len).toBeGreaterThan(0);
    expect(e.foliage).toBeGreaterThan(0);
  });
});

describe("trees: worker job bookkeeping", () => {
  it("one job in flight per tree; later requests coalesce into one follow-up with the newest input", () => {
    const fb = fakeBuilder(), got: number[] = [];
    const q = new BuildQueue(fb, (r) => got.push(r.job.cut));
    const t = tree(TRAITS), s = state([t], M1);
    q.request("t1", t, s, 100, 1);
    q.request("t1", t, s, 110, 1);
    q.request("t1", t, s, 120, 1);
    expect(fb.jobs.map((j) => j.cut)).toEqual([100]);
    expect(q.busy("t1")).toBe(true);
    fb.reply({ id: fb.jobs[0].id, buffers: built(t, 100, s) });
    expect(got).toEqual([100]);
    expect(fb.jobs.map((j) => j.cut)).toEqual([100, 120]);
  });
  it("a stale result (cancelled or superseded job id) is dropped", () => {
    const fb = fakeBuilder(), got: number[] = [];
    const q = new BuildQueue(fb, (r) => got.push(r.job.id));
    const t = tree(TRAITS), s = state([t], M1);
    q.request("t1", t, s, 100, 1);
    const stale = fb.jobs[0].id;
    q.cancel("t1"); // the tree left the view
    q.request("t1", t, s, 130, 1); // and came back: a fresh job goes out at once
    expect(fb.jobs).toHaveLength(2);
    const fresh = fb.jobs[1].id;
    const buffers = built(t, 100, s);
    fb.reply({ id: stale, buffers });
    expect(got).toEqual([]);
    fb.reply({ id: fresh, buffers });
    expect(got).toEqual([fresh]);
    fb.reply({ id: fresh, buffers }); // a duplicate is stale too
    expect(got).toEqual([fresh]);
  });
  it("a failed build frees the slot and is reported (the layer does not retry it until new data)", () => {
    const fb = fakeBuilder(), failed: string[] = [];
    const q = new BuildQueue(fb, () => {}, (k) => failed.push(k));
    const t = tree(TRAITS), s = state([t], M1);
    q.request("t1", t, s, 100, 1);
    fb.reply({ id: fb.jobs[0].id, error: "boom" });
    expect(q.busy("t1")).toBe(false);
    expect(failed).toEqual(["t1"]);
  });
  it("the worker transfers every typed array of the result, and sends no bark, leaves, blossoms or morph", () => {
    const t = tree(TRAITS), s = state([t], M1);
    const { reply, transfer } = handleBuild({
      id: 7, key: "t1", tree: t, cut: 100, knobs: KNOBS.oak, norm: normFor(s), place: { x: 0, z: 0, s: 1 }, traits: TRAITS,
    });
    expect(reply.id).toBe(7);
    if (!("buffers" in reply)) throw new Error("no buffers");
    const b = reply.buffers;
    const arrays = [b.points, ...b.levels.flatMap((l) => [l.coords, l.kind, l.shade, l.born, l.shown])];
    for (const a of arrays) expect(transfer).toContain(a.buffer);
    expect(new Set(transfer).size).toBe(transfer.length);
    for (const k of ["bark", "leaves", "blooms", "morph", "vox", "voxels"]) expect(k in b, k).toBe(false);
    expect(b.levels.map((l) => l.level)).toEqual([0, 1, 2]);
    // the camera frames the voxel extent: the points are its box's corners
    expect(b.points.length).toBe(8 * 3);
    expect(Math.max(...Array.from(b.points).filter((_, i) => i % 3 === 1))).toBe(b.crown.top);
  });
});

describe("trees: rebuild rule", () => {
  it("rebuilds exactly when a turn lands or 20 s pass (requested 2 s early), never after the freeze", () => {
    const times = [10, 50, 90];
    expect(rebuildDue(10, 12, times, 1890)).toBeNull();
    expect(rebuildDue(10, 28, times, 1890)).toBe(30); // 20 s, requested early for its exact time
    expect(rebuildDue(30, 47, times, 1890)).toBeNull();
    expect(rebuildDue(30, 48, times, 1890)).toBe(50); // the turn at 50
    expect(rebuildDue(1890, 4000, times, 1890)).toBeNull(); // frozen: cached for good
    expect(rebuildDue(1885, 4000, times, 1890)).toBe(1890); // one last build at the freeze time
    expect(rebuildDue(10, 60, times, 1890)).toBe(60); // late (data arrived after its time): at once
  });
  it("builds at most 60 trees, nearest the view, none more than 320 units behind it", () => {
    const c = Array.from({ length: 100 }, (_, i) => ({ id: "t" + i, x: i * 5 }));
    const keep = selectBuilt(c, 495);
    expect(keep.size).toBe(60);
    expect(keep.has("t99")).toBe(true);
    expect(keep.has("t30")).toBe(false);
    const few = selectBuilt([{ id: "far", x: 0 }, { id: "near", x: 400 }], 400);
    expect([...few]).toEqual(["near"]);
  });
  it("planTrees asks for the newest trees' builds first: a range switch's first burst fills the live front (final review I2)", () => {
    const infos = new Map(
      Array.from({ length: 30 }, (_, i) => {
        const t = tree(TRAITS, 5, "p" + i);
        return [t.id, treeInfo(t, [{ from: -100, x: i * 5, z: 0, lane: 0 }])] as const;
      }),
    );
    const plan = planTrees(infos, new Map(), 1000, 70, 30, Infinity);
    const xs = plan.builds.map((b) => infos.get(b.id)!.places[0].x);
    expect(xs.length).toBe(30);
    expect(xs).toEqual([...xs].sort((a, b) => b - a));
  });
});

describe("flights and emergence", () => {
  it("a tree flies only forward, in ordinary time steps", () => {
    expect(shouldFly(0, 1, { from: 100, to: 101, continuous: true })).toBe(true);
    expect(shouldFly(1, 0, { from: 100, to: 101, continuous: true })).toBe(false);
    expect(shouldFly(0, 1, { from: 100, to: 4000, continuous: false })).toBe(false);
    expect(shouldFly(undefined, 1, { from: 100, to: 101, continuous: true })).toBe(false);
  });
  it("the flight stays above ground, moves only toward its new spot and lands exactly", () => {
    for (const dx of [-12, -80]) {
      const f = { dx, dz: 4, s: 1, far: Math.abs(dx) > 30, t0: 0, dur: 3200 };
      let prev = -Infinity;
      for (let ms = 0; ms <= 3300; ms += 50) {
        const p = flightPose(f, ms);
        expect(p.lift).toBeGreaterThanOrEqual(0);
        // trees are built about their base, which is the group origin: the bank pivots there
        expect(p.y).toBeGreaterThanOrEqual(0); // never underground
        expect(p.y).toBeCloseTo(p.lift, 9);
        expect(p.x).toBeGreaterThanOrEqual(prev - 1e-9); // only forward
        expect(p.x).toBeLessThanOrEqual(1e-9);
        prev = p.x;
      }
      const end = flightPose(f, 3300);
      expect(end.done).toBe(true);
      expect([end.x, end.y, end.z, end.rotZ]).toEqual([0, 0, 0, 0]);
    }
  });
  it("a new tree scales up from its base over 2.5 s, ease-out", () => {
    const a = emergePose(0, 0), m = emergePose(0, 1250), z = emergePose(0, 2600);
    expect(a.s).toBeLessThan(0.01);
    expect(m.s).toBeGreaterThan(0.5);
    expect(z).toEqual({ s: 1, done: true });
  });
});

// ---------------- ground flora (flower beds, mushrooms), live girth, stable camera ----------------
import { DAISY_CUBES, FLOWER_GROW_S, MUSH_CUBES, bedFlowers, classifyFlora, createGround, floraItems, mushrooms } from "@/components/forest/engine/ground";
import { applyResponse } from "@/lib/forest/client";
import { buildTimeline } from "@/lib/forest/timeline";
import { CameraController } from "@/components/forest/engine/camera";
import type { ForestResponse } from "@/lib/forest/types";

describe("ground flora", () => {
  it("a one-shot call with text out is a flower; prompt in, nothing out (embeddings) is a mushroom", () => {
    expect(classifyFlora([10, 1200, 300])).toBe("flower");
    expect(classifyFlora([10, 1200, 1])).toBe("flower");
    expect(classifyFlora([10, 1200, 0])).toBe("mushroom");
  });
  it("items stand at X(t) of their time, across the spit (|f| ≤ 0.85), sorted by time", () => {
    const tl = buildTimeline([tree(TRAITS)], 4000);
    const s = { ...state([], {}), t0: 1_700_000_000, flowers: [[3000, 900, 120], [500, 400, 0], [1200, 800, 40]] as [number, number, number][] };
    const it_ = floraItems(s, tl.X);
    expect(it_.map((i) => i.time)).toEqual([500, 1200, 3000]);
    expect(it_.map((i) => i.kind)).toEqual(["mushroom", "flower", "flower"]);
    for (const i of it_) {
      expect(i.x).toBeCloseTo(tl.X(i.time), 9);
      expect(Math.abs(i.f)).toBeLessThanOrEqual(0.85);
    }
  });
  it("a since-poll re-sending recent flowers does not duplicate them, and their layout survives a t0 rebase", () => {
    const resp = (o: Partial<ForestResponse>): ForestResponse => ({
      range: "48h", t0: 1000, now: 1100, models: {}, trees: [], flowers: [], ids: [], full: true, cursor: 1090, ...o,
    });
    const X = (t: number) => t * 0.01;
    const s1 = { ...applyResponse(null, resp({ flowers: [[10, 100, 5], [80, 200, 0], [85, 300, 7]] })).state, cursor: 1650 };
    const a = floraItems(s1, X);
    // the server resends everything newer than since − 600; t0 moved by 10 s
    const s2 = applyResponse(s1, resp({ t0: 1010, full: false, cursor: 1655, flowers: [[70, 200, 0], [75, 300, 7], [90, 50, 9]] })).state;
    const b = floraItems(s2, X);
    expect(b).toHaveLength(4);
    expect(new Set(b.map((i) => i.id)).size).toBe(4);
    for (const i of a) {
      const j = b.find((k) => k.id === i.id)!;
      expect(j).toBeDefined();
      expect([j.f, j.seed, j.time + s2.t0]).toEqual([i.f, i.seed, i.time + s1.t0]);
    }
  });
  it("a bed spreads and rises from nothing over 90 s, with a fixed flower count", () => {
    const [bed] = floraItems({ ...state([], {}), flowers: [[100, 900, 350]] }, (t) => t);
    const z = bedFlowers(bed, 2, 0), m = bedFlowers(bed, 2, 0.5), f = bedFlowers(bed, 2, 1);
    expect(z.length).toBe(f.length);
    expect(f.length).toBe(Math.round(8 + (350 / 700) * 18)); // the mockup's voxel bed: 8 … 26 daisies
    expect(Math.max(...z.map((p) => p.s0))).toBe(0);
    expect(Math.max(...m.map((p) => p.s0))).toBeGreaterThan(0);
    for (const p of f) expect(Math.hypot(p.x - bed.x, (p.z - 2) / 0.85)).toBeLessThanOrEqual(0.8 + 0.5 * 1.4 + 1e-9);
    // voxel sprites turn in quarter turns only (they stay on the grid's axes)
    for (const p of f) expect((p.a / (Math.PI / 2)) % 1).toBeCloseTo(0, 9);
    expect(FLOWER_GROW_S).toBe(90);
  });
  it("flora is drawn as the mockup's voxel sprites; their boxes count as cubes against the budget", () => {
    expect(DAISY_CUBES).toBe(5);
    expect(MUSH_CUBES).toBe(10);
    const g = createGround(new THREE.Scene(), { shadows: false });
    const st = { ...state([], {}), flowers: [[100, 900, 350], [120, 50_000, 0]] as [number, number, number][] };
    g.setData(st, (t) => t, () => 15);
    g.tick(500, 0);
    expect(g.count()).toBe(Math.round(8 + (350 / 700) * 18) * DAISY_CUBES + 3 * MUSH_CUBES);
    g.dispose();
  });
  it("mushrooms come up over 60 s; one to three per call by prompt size", () => {
    const items = floraItems({ ...state([], {}), flowers: [[100, 500, 0], [200, 50_000, 0]] }, (t) => t);
    expect(mushrooms(items[0], 100).every((m) => m.s === 0)).toBe(true);
    expect(mushrooms(items[0], 160).every((m) => m.s > 0)).toBe(true);
    expect(mushrooms(items[0], 1000)).toHaveLength(1);
    expect(mushrooms(items[1], 1000)).toHaveLength(3);
  });
});

describe("ghost ground: voxel soil and stone blocks, no smooth disc (final review: old-look remnant)", () => {
  it("a ghost's ground is axis-aligned toon cubes on the voxel grid, in the soil and stone colours, seeded", async () => {
    const { ghostGround } = await import("@/components/forest/engine/flights");
    const { GHOST_SOIL, GHOST_STONE } = await import("@/components/forest/engine/ghostColors");
    const mats = createMaterials();
    const look = (g: THREE.Group) => {
      const out: string[] = [];
      const kinds = new Set<string>();
      g.traverse((o) => {
        if (!(o instanceof THREE.Mesh)) return;
        expect(o).toBeInstanceOf(THREE.InstancedMesh); // no CircleGeometry patch
        expect(o.geometry).toBe(mats.cube);
        expect(o.material).toBeInstanceOf(THREE.MeshToonMaterial);
        const m = o as THREE.InstancedMesh, M = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
        const c = new THREE.Color();
        expect(m.count).toBeGreaterThanOrEqual(10);
        expect(m.count).toBeLessThanOrEqual(40);
        for (let i = 0; i < m.count; i++) {
          m.getMatrixAt(i, M);
          M.decompose(p, q, sc);
          expect(Math.abs(q.w)).toBeCloseTo(1, 9); // no tilt, no turn
          for (const v of [sc.x, sc.y, sc.z, p.x, p.z]) expect((v / 0.125) % 1).toBeCloseTo(0, 6);
          m.getColorAt(i, c);
          const hex = "#" + c.getHexString();
          expect([...GHOST_SOIL, ...GHOST_STONE]).toContain(hex);
          kinds.add(GHOST_SOIL.includes(hex as never) ? "soil" : "stone");
          out.push(`${p.toArray()}|${sc.toArray()}|${hex}`);
        }
      });
      expect(kinds).toEqual(new Set(["soil", "stone"]));
      return out;
    };
    const a = look(ghostGround("t1", 1, 1, mats)), b = look(ghostGround("t1", 1, 1, mats)), c = look(ghostGround("t1", 2, 1, mats));
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    mats.dispose();
  });
});

describe("tree layer: voxel cubes, level of detail, budget, cross-fade, picking", () => {
  /** n trees of 40 turns, 14 u apart along x, a layer over them with a direct builder, every tree built at `rel`. */
  async function forest(n: number, cap?: number) {
    const trees = Array.from({ length: n }, (_, i) => tree(TRAITS, 40, "t" + i));
    const s = state(trees, M1), scene = new THREE.Scene(), mats = createMaterials();
    const infos = new Map(trees.map((t, i) => [t.id, treeInfo(t, [{ from: -100, x: i * 14, z: 0, lane: 0 }])]));
    const layer = createTreeLayer(scene, mats, createTrail(scene), {
      shadows: false, builder: createDirectBuilder(), onChange: () => {}, maxTrees: 100, cubeCap: cap === undefined ? undefined : () => cap,
    });
    layer.setData(s, infos);
    const rel = 1300;
    layer.tick(rel, { continuous: false, dt: 0 } as never, n * 7);
    await layer.settle();
    layer.animate(0, 0, rel);
    return { layer, scene, mats, trees, rel, at: (i: number) => ({ x: i * 14, z: 0 }) };
  }
  const meshes = (scene: THREE.Scene) => {
    const out: THREE.InstancedMesh[] = [];
    scene.traverse((o) => o instanceof THREE.InstancedMesh && o.userData.kind === "tree" && out.push(o));
    return out;
  };

  it("a tree stands on the terrace under its place; a place glide keeps its base on the ground under it (final review I1)", async () => {
    // a 3-tier terrace: the tree's place on its top tier, then the timeline nudges the place off the terrace
    const h = hillsIn(0, 3000).find((q) => q.tiers.length === 3)!, top = h.tiers[2];
    const x = (top.x0 + top.x1) / 2, z = (top.z0 + top.z1) / 2, off = h.tiers[0].x1 + 2;
    expect(terraceHeight(x, z)).toBe(1.5);
    expect(terraceHeight(off, z)).toBe(0);
    const t = tree(TRAITS, 30, "tb"), s = state([t], M1), scene = new THREE.Scene(), mats = createMaterials();
    const layer = createTreeLayer(scene, mats, createTrail(scene), {
      shadows: false, builder: createDirectBuilder(), onChange: () => {}, maxTrees: 10, baseAt: (px, pz) => terraceHeight(px, pz),
    });
    const group = () => {
      let g: THREE.Object3D | null = null;
      scene.traverse((o) => o instanceof THREE.InstancedMesh && o.userData.kind === "tree" && (g = o.parent));
      return g as THREE.Object3D | null;
    };
    layer.setData(s, new Map([[t.id, treeInfo(t, [{ from: -100, x, z, lane: 0 }])]]));
    layer.tick(800, { continuous: false, dt: 0 } as never, x);
    await layer.settle();
    layer.animate(0, 0.016, 800);
    expect(group()!.position.y).toBe(1.5);
    expect(layer.cameraInputs(800).crowns.get(t.id)!.top).toBeGreaterThan(1.5 + 1);
    const ys: number[] = [], xs: number[] = [];
    // new data moves the place off the terrace (a glide): x, z and the base glide together, the base on the ground
    layer.setData(s, new Map([[t.id, treeInfo(t, [{ from: -100, x: off, z, lane: 0 }])]]));
    const cells0 = layer.cellsAt(t.id);
    for (let k = 0; k < 400; k++) {
      layer.tick(800 + k, { continuous: true, dt: 1 } as never, x);
      if (k < 120) await layer.settle(); // the tree regrows (new turns land): new builds swap in at the new place
      layer.animate(k * 16, 0.016, 800 + k);
      const p = group()!.position;
      xs.push(p.x);
      ys.push(p.y);
      // every frame: the base sits on the ground under where the tree is drawn
      expect(Math.abs(p.y - terraceHeight(p.x, p.z))).toBeLessThanOrEqual(0.05);
    }
    expect(layer.cellsAt(t.id)).toBeGreaterThan(cells0!); // it did swap in new builds
    // continuous: the base only changes where the ground under the glide does (a tier edge, one 0.5 step at a time)
    for (let k = 1; k < ys.length; k++) {
      expect(Math.abs(ys[k] - ys[k - 1])).toBeLessThanOrEqual(0.5 + 1e-9);
      if (ys[k] !== ys[k - 1]) expect(terraceHeight(xs[k], z)).not.toBe(terraceHeight(xs[k - 1], z));
    }
    expect(new Set(ys)).toEqual(new Set([1.5, 1, 0.5, 0])); // it stepped down every tier: never floating, never sunk
    // the place is frozen: the base is that place's ground height
    expect(xs.at(-1)).toBeCloseTo(off, 3);
    expect(ys.at(-1)).toBe(0);
    layer.dispose();
  });

  it("one instanced mesh of cubes per tree; its count is the cells drawn at the tree's level", async () => {
    const { layer, scene, rel } = await forest(3);
    layer.setView({ x: 14, y: 20, z: 40 }, 900, 0, 0); // near: level 0
    layer.animate(1000, 0, rel); // any fade from the first level ends
    const ms = meshes(scene);
    expect(ms.length).toBe(3);
    for (const m of ms) {
      expect(m.userData.lc.level).toBe(0);
      expect(m.count).toBe(m.userData.lc.shown.length);
      expect(m.count).toBeGreaterThan(50);
    }
    expect(layer.stats().cubes).toBe(ms.reduce((a, m) => a + m.count, 0));
    layer.dispose();
  });

  it("a level switch cross-fades over 0.3 s (both levels drawn, complementary), with no switch back at the threshold", async () => {
    const { layer, scene, rel } = await forest(1);
    const cam = (d: number) => ({ x: 0, y: d * 0.6, z: d * 0.8 });
    layer.setView(cam(20), 900, 0, 0);
    layer.animate(500, 0, rel);
    expect(layer.levelOf("t0")).toBe(0);
    layer.setView(cam(LOD_D[0] * 1.3), 900, 0, 1000); // far enough: level 1
    expect(layer.levelOf("t0")).toBe(1);
    const both = meshes(scene);
    expect(both.map((m) => m.userData.lc.level).sort()).toEqual([0, 1]);
    layer.animate(1000 + FADE_S * 500, 0, rel); // half way: both still drawn, fades at 0.5
    const fades = both.map((m) => (m.material as unknown as VoxelMaterial).fade);
    expect(fades.map((f) => f.uFade.value)).toEqual([0.5, 0.5]);
    expect(fades.map((f) => f.uFadeIn.value).sort()).toEqual([0, 1]);
    layer.animate(1000 + FADE_S * 1000 + 1, 0, rel);
    expect(meshes(scene).map((m) => m.userData.lc.level)).toEqual([1]);
    // hovering just under the threshold (inside the band) keeps level 1
    for (const k of [0.97, 1.0, 1.02, 0.95]) layer.setView(cam(LOD_D[0] * k * (900 / 900)), 900, 0, 2000);
    expect(layer.levelOf("t0")).toBe(1);
    layer.dispose();
  });

  it("Q2: a fade reversed half way runs back from where it stands (no pop), and a third level waits for its end", async () => {
    const { layer, scene, rel } = await forest(1);
    const cam = (d: number) => ({ x: 0, y: d * 0.6, z: d * 0.8 });
    layer.setView(cam(20), 900, 0, 0);
    layer.animate(500, 0, rel);
    layer.setView(cam(LOD_D[0] * 1.3), 900, 0, 1000); // 0 → 1
    layer.animate(1000 + FADE_S * 500, 0, rel); // half way
    const fadeOf = (m: THREE.InstancedMesh) => (m.material as unknown as VoxelMaterial).fade.uFade.value;
    const [m1, m2] = meshes(scene);
    layer.setView(cam(20), 900, 0, 1000 + FADE_S * 500); // back to 0, mid-fade
    expect(layer.levelOf("t0")).toBe(0);
    expect(meshes(scene)).toEqual([m1, m2]); // the same two meshes, nothing disposed or rebuilt
    layer.animate(1000 + FADE_S * 750, 0, rel);
    expect(fadeOf(m1)).toBeCloseTo(0.25, 6); // running back from 0.5
    layer.animate(1000 + FADE_S * 1000 + 1, 0, rel);
    const end = meshes(scene);
    expect(end.length).toBe(1);
    expect(end[0].userData.lc.level).toBe(0);
    // a third level during a fade: the fade finishes first, then the next one starts
    layer.setView(cam(LOD_D[0] * 1.3), 900, 0, 5000); // 0 → 1
    layer.setView(cam(LOD_D[1] * 1.3), 900, 0, 5050); // asks for 2 mid-fade
    expect(meshes(scene).map((m) => m.userData.lc.level).sort()).toEqual([0, 1]);
    layer.animate(5000 + FADE_S * 1000 + 1, 0, rel);
    expect(meshes(scene).map((m) => m.userData.lc.level).sort()).toEqual([1, 2]);
    layer.dispose();
  });

  it("over the cube budget the farthest trees go coarser first; the count fits and every tree stays drawn", async () => {
    const { layer, scene, rel } = await forest(6);
    layer.setView({ x: 0, y: 12, z: 16 }, 900, 0, 0);
    layer.animate(1000, 0, rel);
    const all0 = layer.stats().cubes;
    layer.dispose();
    const tight = await forest(6, Math.floor(all0 * 0.5));
    tight.layer.setView({ x: 0, y: 12, z: 16 }, 900, 0, 0); // tree t0 nearest, t5 farthest
    tight.layer.animate(1000, 0, tight.rel);
    const lv = tight.trees.map((t) => tight.layer.levelOf(t.id)!);
    expect(tight.layer.stats().cubes).toBeLessThanOrEqual(all0 * 0.5);
    expect(lv[5]).toBeGreaterThan(0);
    for (let i = 1; i < lv.length; i++) expect(lv[i]).toBeGreaterThanOrEqual(lv[i - 1]);
    expect(meshes(tight.scene).length).toBe(6);
    expect(scene.children.length).toBeGreaterThanOrEqual(0);
    tight.layer.dispose();
  });

  it("picking: the cube under the ray maps back to the tree, the session and the turn that grew it", async () => {
    const { layer, scene, rel, at } = await forest(2);
    layer.setView({ x: 14, y: 20, z: 40 }, 900, 0, 0);
    layer.animate(1000, 0, rel);
    scene.updateMatrixWorld(true);
    const m = meshes(scene).find((x) => x.userData.treeId === "t1")!, lc = m.userData.lc as LevelCells;
    let checked = 0;
    for (let k = 0; k < lc.shown.length && checked < 12; k += Math.ceil(lc.shown.length / 40)) {
      const i = lc.shown[k], c = [0, 1, 2].map((j) => (lc.coords[i * 3 + j] + 0.5) * lc.size);
      const p = new THREE.Vector3(c[0] + at(1).x, c[1], c[2] + at(1).z);
      const ray = new THREE.Raycaster(p.clone().add(new THREE.Vector3(0, 300, 0)), new THREE.Vector3(0, -1, 0));
      const hit = ray.intersectObject(m, false)[0];
      if (!hit || lc.shown[hit.instanceId!] !== i) continue; // another cube above it
      const r = layer.pick(ray);
      expect(r?.result.kind).toBe("turn");
      if (r?.result.kind !== "turn") continue;
      expect(r.result.treeId).toBe("t1");
      expect(r.result.sessionId).toBe("S-t1");
      expect(r.result.time).toBe(lc.born[i]);
      expect(r.result.turn).toBe(Math.round(lc.born[i] / 30)); // the tree's turns are 30 s apart
      checked++;
    }
    expect(checked).toBeGreaterThan(3);
    layer.dispose();
  });

  it("girthAt and heightAt follow the cells on screen: non-decreasing as new cubes scale in", async () => {
    const { layer, rel } = await forest(1);
    const g0 = layer.girthAt("t0", rel)!, h0 = layer.heightAt("t0", rel)!;
    expect(g0).toBeGreaterThan(0);
    expect(h0).toBeGreaterThan(1);
    expect(layer.girthAt("t0", rel + SCALE_IN_S)!).toBeGreaterThanOrEqual(g0);
    expect(layer.cellsAt("t0")).toBeGreaterThan(100);
    expect(CUBE_CAP).toBe(120_000);
    layer.dispose();
  });
});

describe("ground flora: identity across re-anchors (I2)", () => {
  it("a float re-anchor (Δt0 = 299.97) on full and delta responses keeps every flower's id, seed, place and time", async () => {
    const { FrameCore } = await import("@/components/forest/engine/frame");
    const { FloraRegistry } = await import("@/components/forest/engine/ground");
    const abs = [1010.04, 1080.0, 1086.66, 1290.55]; // absolute call times; the server rounds rel to 0.1 per anchor
    const fl = (t0: number) => [[abs[0], 100, 5], [abs[1], 200, 0], [abs[2], 300, 7], [abs[3], 300, 7]]
      .map(([a, c, g]) => [Math.round((a - t0) * 10) / 10, c, g] as [number, number, number]);
    const resp = (o: Partial<ForestResponse>): ForestResponse => ({
      range: "48h", t0: 1000, now: 1400, models: {}, trees: [], flowers: [], ids: [], full: true, cursor: 1400, ...o,
    });
    const cam = { step: () => false, pose: () => ({ pos: { x: 0, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 } }), paused: false, setEvents: () => {} };
    const X = (t: number) => t * 0.01;
    for (const full of [true, false]) {
      const core = new FrameCore(cam), reg = new FloraRegistry();
      const s1 = applyResponse(null, resp({ flowers: fl(1000) })).state;
      const a = reg.items(core.intake(s1, 0).state, X);
      const t0b = 1000 + 299.97;
      // delta: the client keeps (and rebases) what is older than since − 600 = 1100; the server resends the rest
      const r2 = resp({ t0: t0b, full, cursor: 1705, now: 1700 /* absolute on the wire */, flowers: full ? fl(t0b) : fl(t0b).slice(3) });
      const s2 = applyResponse({ ...s1, cursor: 1700 }, r2).state;
      const b = reg.items(core.intake(s2, 300_000).state, X);
      expect(b, `full=${full}`).toHaveLength(4);
      for (const i of a) {
        const j = b.find((k) => k.id === i.id);
        expect(j, `${i.id} full=${full}`).toBeDefined();
        expect([j!.f, j!.seed, j!.time, j!.x, j!.hue]).toEqual([i.f, i.seed, i.time, i.x, i.hue]);
      }
    }
  });
});

describe("scene camera stability", () => {
  it("reset moves the one controller to a new pose without replacing it", () => {
    const c = new CameraController({ fov: 36, aspect: 1.5 });
    c.setMode("follow");
    c.reset({ x: 100, y: 12, z: 40 }, { x: 100, y: 3, z: 0 });
    const p = c.pose();
    expect([p.pos.x, p.pos.y, p.pos.z]).toEqual([100, 12, 40]);
    expect(p.target.x).toBeCloseTo(100, 6);
    expect(c.cameraMode).toBe("follow");
    expect(c.destination).toBeNull();
  });
});

describe("land: a bad time never allocates a giant strip", () => {
  it("landSpan sizes the strip from the front and clamps it to MAX_LAND_COLS", async () => {
    const { landSpan, MAX_LAND_COLS } = await import("@/components/forest/engine/land");
    const ok = landSpan(1071)!; // 48 h at XS = 0.0062
    expect(ok.cols).toBe(Math.ceil((ok.lx1 + 30) / 0.5));
    expect(ok.lx1).toBe(1071 + 80 + 240);
    // an absolute unix time read as relative: X ≈ 1.1e7, which once asked for a 2 GB Float32Array
    const huge = landSpan(1.79e9 * 0.0062)!;
    expect(huge.cols).toBe(MAX_LAND_COLS);
    expect(MAX_LAND_COLS * 57 * 8 * 4).toBeLessThan(64 * 2 ** 20); // position + uv + colour stay well under 64 MB
  });
  it.each([NaN, Infinity, -Infinity, -1])("landSpan(%s) is a no-op (null)", async (x) => {
    const { landSpan } = await import("@/components/forest/engine/land");
    expect(landSpan(x)).toBeNull();
  });
});

describe("view hand-off (Stats card ↔ forest page)", () => {
  it("handoff round-trips and expires after 10 s", () => {
    const v = { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "follow" as const, frontRel: true as const };
    expect(decodeHandoff(encodeHandoff(v, 1000), 5000)).toEqual(v);
    expect(decodeHandoff(encodeHandoff(v, 1000), 12_000)).toBeNull();
    expect(decodeHandoff("garbage", 1000)).toBeNull();
  });
  it("a wrong shape is rejected", () => {
    const at = 1000;
    for (const bad of [
      { pos: { x: 1, y: 2 }, target: { x: 0, y: 1, z: 0 }, mode: "follow", at },
      { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "orbit", at },
      { pos: { x: 1, y: 2, z: "3" }, target: { x: 0, y: 1, z: 0 }, mode: "cinema", at },
      { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "cinema" },
      // the old (world) format, without frontRel: ignored
      { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "cinema", at },
      { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "cinema", at, frontRel: false },
      null, 42, [],
    ]) expect(decodeHandoff(JSON.stringify(bad), at), JSON.stringify(bad)).toBeNull();
    expect(decodeHandoff(encodeHandoff({ pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "cinema", frontRel: true }, at), at)).toEqual({
      pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "cinema", frontRel: true,
    });
  });
  it("importView sets the pose at rest and the mode, without animation; the next frame does not jump", () => {
    for (const mode of ["follow", "cinema"] as const) {
      const cam = new CameraController({ fov: 36, aspect: 1.5, pos: { x: 0, y: 20, z: 60 }, target: { x: 0, y: 0, z: 0 } });
      // the camera is flying fast toward a far fit when the view is imported
      cam.setMode("follow");
      cam.follow([{ x: 400, y: 0, z: 0 }, { x: 420, y: 10, z: 5 }], { l: -1, r: 1, t: 1, b: -1 });
      for (let k = 0; k < 60; k++) cam.step(1 / 60, { rel: k / 60, nowMs: k * 16 });
      const modes: string[] = [];
      cam.onModeChange = (m) => modes.push(m);
      const v = { pos: { x: 10, y: 12, z: 40 }, target: { x: 5, y: 3, z: 0 }, mode };
      cam.importView(v.pos, v.target, v.mode);
      expect(cam.cameraMode).toBe(mode);
      expect(modes).toEqual(mode === "follow" ? [] : ["cinema"]);
      const p = cam.pose();
      for (const k of ["x", "y", "z"] as const) {
        expect(p.pos[k]).toBeCloseTo(v.pos[k], 9);
        expect(p.target[k]).toBeCloseTo(v.target[k], 9);
      }
      cam.step(1 / 60, { rel: 1, nowMs: 1000 });
      const q = cam.pose();
      const moved = Math.hypot(q.pos.x - p.pos.x, q.pos.y - p.pos.y, q.pos.z - p.pos.z);
      expect(moved, mode).toBeLessThan(0.1); // at rest: one frame moves a few centimetres, not to the old flight
      const d0 = { x: p.target.x - p.pos.x, y: p.target.y - p.pos.y, z: p.target.z - p.pos.z };
      const d1 = { x: q.target.x - q.pos.x, y: q.target.y - q.pos.y, z: q.target.z - q.pos.z };
      const cos = (d0.x * d1.x + d0.y * d1.y + d0.z * d1.z) / (Math.hypot(d0.x, d0.y, d0.z) * Math.hypot(d1.x, d1.y, d1.z));
      expect(Math.acos(Math.min(1, cos)), mode).toBeLessThan(0.03); // the look turns ≤ 1.2 rad/s
    }
  });
});

describe("trees: waiting for outstanding builds (snapshot)", () => {
  it("whenIdle resolves once every key busy at the call has replied, including its coalesced follow-up; a new request later does not hold it", async () => {
    const fb = fakeBuilder();
    const q = new BuildQueue(fb, () => {}, () => {});
    const t = tree(TRAITS), s = state([t], M1);
    q.request("a", t, s, 100, 1);
    q.request("a", t, s, 110, 1); // coalesced
    q.request("b", t, s, 100, 1);
    let done = false;
    const p = q.whenIdle([...q.busyKeys(), "idle-key"]).then(() => void (done = true));
    fb.reply({ id: fb.jobs[0].id, buffers: built(t, 100, s) }); // a's first: its follow-up goes out, a still busy
    await Promise.resolve();
    expect(done).toBe(false);
    fb.reply({ id: fb.jobs[1].id, error: "boom" }); // b failed: idle
    fb.reply({ id: fb.jobs[2].id, buffers: built(t, 110, s) }); // a's follow-up: idle
    q.request("a", t, s, 130, 1); // a new build after the idle moment
    await p;
    expect(done).toBe(true);
    // nothing busy: resolves at once; a cancelled key counts as idle; dispose releases every waiter
    await q.whenIdle(["a-none"]);
    const c = q.whenIdle(["a"]);
    q.cancel("a");
    await c;
    q.request("z", t, s, 1, 1);
    const d = q.whenIdle(["z"]);
    q.dispose();
    await d;
  });
});

describe("view hand-off: import before the first data, and during a drag (review I1, M3)", () => {
  const v = { pos: { x: 10, y: 12, z: 40 }, target: { x: 5, y: 3, z: 0 }, mode: "follow" as const };
  const front = { pos: { x: 100, y: 30, z: 80 }, target: { x: 90, y: 3, z: 0 } };
  const near = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) =>
    (["x", "y", "z"] as const).forEach((k) => expect(a[k]).toBeCloseTo(b[k], 9));

  it("a view imported before the first data wins over the front placement; the placement still resets the shot", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.5 });
    expect(cam.importView(v.pos, v.target, v.mode)).toBe(true);
    cam.placeInitial(front.pos, front.target); // the scene's first data
    near(cam.pose().pos, v.pos);
    near(cam.pose().target, v.target);
    expect(cam.cameraMode).toBe("follow");
    // used once: a later placement (none happens in the scene, but the rule is one-shot) goes to the front
    cam.placeInitial(front.pos, front.target);
    near(cam.pose().pos, front.pos);
  });

  it("without an import the first placement is the front", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.5 });
    cam.placeInitial(front.pos, front.target);
    near(cam.pose().pos, front.pos);
    near(cam.pose().target, front.target);
  });

  it("an import during a live drag is ignored: the user's pose, mode and hold stay", () => {
    const cam = new CameraController({ fov: 36, aspect: 1.5, pos: front.pos, target: front.target });
    cam.setDragging(true, 0);
    expect(cam.importView(v.pos, v.target, "follow")).toBe(false);
    near(cam.pose().pos, front.pos);
    expect(cam.cameraMode).toBe("cinema");
    expect(cam.paused).toBe(true);
    cam.placeInitial(front.pos, front.target); // nothing was held for the first data either
    near(cam.pose().pos, front.pos);
    // after the drag ends, an import is taken again
    cam.setDragging(false, 100);
    expect(cam.importView(v.pos, v.target, "follow")).toBe(true);
    near(cam.pose().pos, v.pos);
  });
});
