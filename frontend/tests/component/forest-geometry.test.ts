import { describe, expect, it } from "vitest";
import { buildTreeGeometry, type TreeBuffers } from "@/lib/forest/geometry";
import { KNOBS } from "@/lib/forest/species";
import type { Tree, Turn } from "@/lib/forest/types";

const T = (n: number) => ({
  id: "tok:spell:1", key: "farm", start: 0, end: n * 30, last: n * 30, last_at: n * 30,
  traits: { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 },
  sessions: [{ id: "S", variant: "v1", children: [],
    turns: Array.from({ length: n }, (_, i) => [i * 30, 37000 + i * 500, 150, 35000, [], ["read"], "tool_calls", 1]) }],
}) as never;
const norm = { len: 0.2, foliage: 0.15, models: { v1: { prefill_tps: 20000, decode_tps: 1000, max_model_len: 262144 } } } as never;
const place = { x: 0, z: 0, s: 1 };

const turn = (t: number, o: Partial<{ ctx: number; gen: number; cached: number | null; out: string[] }> = {}): Turn => [
  t, o.ctx ?? 37000, o.gen ?? 150, o.cached === undefined ? 35000 : o.cached, [["read", 900, false]],
  o.out ?? ["read"], (o.out ?? ["read"]).length ? "tool_calls" : "stop", 1,
];

/** Two top-level sessions (the second starts at t=200 and becomes the leader) plus a subagent and an answer turn. */
const TWO = (): Tree => ({
  id: "tok:spell:2", key: "farm", start: 0, end: 400, last: 400, last_at: 400,
  traits: { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0.1, fanout: 0.5 },
  sessions: [
    {
      id: "A", variant: "v1",
      turns: [turn(0), turn(30), turn(60, { out: [] }), turn(90, { cached: null }), turn(120)],
      children: [{ id: "A.sub", variant: "v1", at: 1, children: [], turns: [turn(40), turn(70), turn(100, { out: [] })] }],
    },
    { id: "B", variant: "v1", children: [], turns: [turn(200), turn(230), turn(260), turn(290)] },
  ],
});

/** Every typed array a build yields: the voxelizer's inputs, the cells and the camera points. */
const all = (b: TreeBuffers): ArrayLike<number>[] => [
  b.vox.origin, b.vox.foliage, b.vox.tips, b.vox.wood, b.vox.lines, b.vox.sched,
  b.voxels.coords, b.voxels.kind, b.voxels.shade, b.voxels.born, b.voxels.buried, b.points,
];
const same = (a: TreeBuffers, b: TreeBuffers, what = "") =>
  all(a).forEach((arr, i) => expect(Array.from(all(b)[i]), `${what} array ${i}`).toEqual(Array.from(arr)));

const sub3 = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = (a: number[]) => Math.hypot(a[0], a[1], a[2]);
/** Rows of `stride` from a flat array whose born sits at `bornAt`, for rows born before `before`. */
const rows = (a: Float32Array, stride: number, bornAt: number, before: number) => {
  const out: number[][] = [];
  for (let i = 0; i < a.length; i += stride) if (a[i + bornAt] < before) out.push(Array.from(a.subarray(i, i + stride)));
  return out;
};
const PINE = (n: number) => T(n) as unknown as Tree;
/** A parent whose subagent grows turn by turn too (children are cut into shoots the same way). */
const withSub = (n: number) => {
  const t = T(n) as unknown as Tree;
  const sub = { id: "S.sub", variant: "v1", at: 0, children: [],
    turns: Array.from({ length: n }, (_, i) => turn(i * 30 + 5, { out: i % 3 ? ["edit"] : [] })) };
  return { ...t, sessions: [{ ...t.sessions[0], turns: t.sessions[0].turns.slice(0, 1), children: [sub] }] } as Tree;
};

// The old look's bark tubes, leaf instances and their shader growth (anchors, swelling, LEAFK) are gone (Plan 4): a
// build yields the voxelizer's inputs, the cells, the turn nodes and the camera points. The voxelizer's own rules are in
// forest-voxel.test.ts.
describe("geometry", () => {
  it("is deterministic", () => {
    same(buildTreeGeometry(T(10), 400, KNOBS.oak, norm, place), buildTreeGeometry(T(10), 400, KNOBS.oak, norm, place));
    same(buildTreeGeometry(TWO(), 400, KNOBS.pine, norm, place), buildTreeGeometry(TWO(), 400, KNOBS.pine, norm, place));
  });

  it("a later turn never moves earlier wood", () => {
    // R4: cut 280 for both builds; T(11) adds a turn at t=300, after the cut.
    same(buildTreeGeometry(T(10), 280, KNOBS.oak, norm, place), buildTreeGeometry(T(11), 280, KNOBS.oak, norm, place));
  });

  it("is prefix-stable: turn n+1 never moves the wood, foliage or blossoms of turns 1..n (n = 2..14)", () => {
    for (let n = 2; n <= 14; n++) {
      const cut = n * 30 - 15; // after turn n (t = (n−1)·30), before turn n+1 (t = n·30)
      for (const [name, k] of [["oak", KNOBS.oak], ["pine", KNOBS.pine], ["round", KNOBS.round]] as const)
        same(buildTreeGeometry(T(n), cut, k, norm, place), buildTreeGeometry(T(n + 1), cut, k, norm, place), `${name} n=${n}`);
      same(buildTreeGeometry(withSub(n), cut, KNOBS.oak, norm, place), buildTreeGeometry(withSub(n + 1), cut, KNOBS.oak, norm, place), `sub n=${n}`);
    }
  });

  it("prefix-stable after turn n+1 grows: joints, twig tips, foliage points and blossom tips of turns ≤ n stay put", () => {
    const key = (n: { kind: string; sessionId: string; turn: number }) => `${n.kind}:${n.sessionId}:${n.turn}`;
    for (let n = 2; n <= 14; n++) {
      const cut = n * 30 + 15; // after turn n+1 (t = n·30)
      for (const [name, mk] of [["oak", T], ["sub", withSub]] as const) {
        const a = buildTreeGeometry(mk(n), cut, KNOBS.oak, norm, place);
        const b = buildTreeGeometry(mk(n + 1), cut, KNOBS.oak, norm, place);
        const nb = new Map(b.nodes.map((x) => [key(x), x.p]));
        expect(a.nodes.length, `${name} n=${n}`).toBeGreaterThan(0);
        for (const x of a.nodes) expect(nb.get(key(x)), `${name} n=${n} ${key(x)}`).toEqual(x.p);
        expect(rows(b.vox.foliage, 6, 4, n * 30), `${name} n=${n} foliage`).toEqual(rows(a.vox.foliage, 6, 4, n * 30));
        expect(rows(b.vox.tips, 4, 3, n * 30), `${name} n=${n} blossoms`).toEqual(rows(a.vox.tips, 4, 3, n * 30));
      }
    }
  });

  it("pine: every twig is bounded and none points back along its limb (45 turns)", () => {
    const b = buildTreeGeometry(PINE(45), 5000, KNOBS.pine, norm, place);
    const J = new Map<number, number[]>(), W = new Map<number, number[]>();
    for (const nd of b.nodes) (nd.kind === "joint" ? J : W).set(nd.turn, nd.p);
    expect(W.size).toBe(45);
    for (let i = 0; i < 45; i++) {
      const tw = sub3(W.get(i)!, J.get(i)!);
      expect(len3(tw), `twig ${i}`).toBeLessThanOrEqual(1.5 * place.s);
      // limb direction within its shoot of 5 (a shoot starts on its parent, not on the previous turn's joint)
      const lo = i % 5 === 0 ? i : i - 1, hi = i % 5 === 4 || i === 44 ? i : i + 1;
      if (lo === hi) continue;
      const d = sub3(J.get(hi)!, J.get(lo)!);
      // cone twigs leave square to the limb (y −0.08), so the dot is ≈ 0; backwards would be clearly negative
      expect(dot3(tw, d) / len3(tw) / len3(d), `twig ${i}`).toBeGreaterThanOrEqual(-0.05);
    }
  });

  it("a subagent's wood starts at the joint of the turn that spawned it", () => {
    const b = buildTreeGeometry(TWO(), 400, KNOBS.oak, norm, place);
    const joint1 = b.nodes.find((x) => x.kind === "joint" && x.sessionId === "A" && x.turn === 1)!.p;
    // the centreline whose first sample is born at the subagent's first turn (t = 40)
    const { wood, lines } = b.vox, starts: number[][] = [];
    for (let l = 0; l + 1 < lines.length; l++) if (wood[lines[l] * 6 + 3] === 40) starts.push(Array.from(wood.subarray(lines[l] * 6, lines[l] * 6 + 3)));
    expect(starts).toHaveLength(1);
    starts[0].forEach((v, k) => expect(v).toBeCloseTo(joint1[k] - (k === 0 ? place.x : k === 2 ? place.z : 0), 5));
  });

  it("habit follows the traits passed in, not the tree's live traits", () => {
    const frozenTraits = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0, fanout: 0 };
    const drifted = { ...(T(10) as unknown as Tree), traits: { ...frozenTraits, sys0: 4000, ctx_gen: 3, mean_gen: 900 } };
    const a = buildTreeGeometry(T(10), 400, KNOBS.round, norm, place);
    same(a, buildTreeGeometry(drifted, 400, KNOBS.round, norm, place, frozenTraits));
    expect(Array.from(buildTreeGeometry(drifted, 400, KNOBS.round, norm, place).vox.wood)).not.toEqual(Array.from(a.vox.wood));
  });

  it("every wood sample, foliage point and blossom tip is born at a turn time, never after the cut", () => {
    const tree = TWO();
    const times = new Set<number>();
    const walk = (s: Tree["sessions"][number]) => {
      s.turns.forEach((t) => times.add(t[0]));
      s.children.forEach(walk);
    };
    tree.sessions.forEach(walk);
    for (const cut of [50, 130, 220, 400]) {
      const b = buildTreeGeometry(tree, cut, KNOBS.oak, norm, place);
      for (const [a, stride, at] of [[b.vox.wood, 6, 3], [b.vox.foliage, 6, 4], [b.vox.tips, 4, 3]] as const)
        for (let i = at; i < a.length; i += stride) {
          expect(a[i]).toBeLessThanOrEqual(cut);
          expect(times.has(a[i])).toBe(true);
        }
    }
  });

  it("blossoms only on turns that called no tools", () => {
    expect(buildTreeGeometry(T(10), 400, KNOBS.oak, norm, place).vox.tips.length).toBe(0);
    expect(buildTreeGeometry(TWO(), 400, KNOBS.oak, norm, place).vox.tips.length / 4).toBe(2);
  });

  it("a median turn gets the mockup's median foliage and segment length", () => {
    const points = (folRef: number, lenRef: number) =>
      buildTreeGeometry(T(1), 400, KNOBS.oak, { ...(norm as object), foliage: folRef, len: lenRef } as never, place).vox.foliage.length / 6;
    // turn 0: ctx 37000 on a 262144 window = 0.1411 → reference equal to it means compress() = 1 → mockup ctx 37000
    // 1 tool point + round(7 + 2.6·log2(1 + 37000/2000)) = 1 + 18 foliage points
    expect(points(37000 / 262144, 150 / 1000)).toBe(1 + 18);
    expect(points(1e9, 1e9)).toBe(1 + 7);
  });

  it("subagents fork from their parent turn and add nodes", () => {
    const b = buildTreeGeometry(TWO(), 400, KNOBS.oak, norm, place);
    expect(b.nodes.some((n) => n.sessionId === "A.sub")).toBe(true);
    expect(b.nodes.every((n) => n.time <= 400)).toBe(true);
    expect(b.points.length % 3).toBe(0);
    expect(b.crown.r).toBeGreaterThanOrEqual(1.5);
    expect(b.crown.top).toBeGreaterThanOrEqual(1);
  });

  it("no NaN in any buffer, for every species", () => {
    for (const k of Object.values(KNOBS)) {
      for (const tree of [T(10), TWO()]) {
        const b = buildTreeGeometry(tree, 400, k, norm, { x: 3, z: -4, s: 0.8 });
        expect(all(b).map((arr) => Array.from(arr).every(Number.isFinite))).not.toContain(false);
        expect(b.voxels.count).toBeGreaterThan(0);
      }
    }
  });

  it("does not mutate its input", () => {
    const tree = TWO();
    const before = JSON.stringify(tree);
    buildTreeGeometry(tree, 400, KNOBS.oak, norm, place);
    expect(JSON.stringify(tree)).toBe(before);
  });
});
