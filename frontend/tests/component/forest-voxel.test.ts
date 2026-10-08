import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { buildTreeGeometry, type TreeBuffers } from "@/lib/forest/geometry";
import { handleBuild } from "@/lib/forest/geometry/worker";
import { KNOBS, type Knobs } from "@/lib/forest/species";
import type { Session, Tree, Turn } from "@/lib/forest/types";
import { TRUNK_STEP } from "@/lib/forest/geometry/tree";
import { CHOSEN, KIND, VOXEL_CAP, type VoxelCells, type VoxelOpts, voxelizeTree } from "@/lib/forest/voxel";
import { LOD_CAP, shiftTree } from "@/components/forest/engine/frame";
import { SaturationShader, createMaterials, VOXEL_SHADER_PATCH } from "@/components/forest/engine/materials";
import {
  BudgetLevels, FADE_S, GIRTH_RATE, LOD_D, LOD_HYST, SCALE_IN_S, TINT_S, cubeColors, fadeKeep, fitBudget, girthOf, levelCells, levelFor, pickTurn,
  scaleIn, tintAt, topOf, voxelMesh,
} from "@/components/forest/engine/voxels";

const norm = { len: 0.2, foliage: 0.15, models: { v1: { prefill_tps: 20000, decode_tps: 1000, max_model_len: 262144 } } } as never;
const place = { x: 3, z: -2, s: 1 };
const traits = { sys0: 37000, ctx_gen: 280, mean_gen: 134, thrash: 0, fail: 0.1, fanout: 0.5 };

/** Turn i of a session: tools on most turns, an answer (no tools: a blossom) every 4th, a failed call every 7th. */
const turn = (t: number, i: number): Turn => {
  const answer = i % 4 === 3;
  const out = answer ? [] : i % 3 ? ["read", "edit"] : ["shell"];
  const ins: Turn[4] = out.map((f, j) => [f, 400 + 300 * j, i % 7 === 5 && j === 0] as Turn[4][number]);
  return [t, 37000 + i * 500, 150, 35000, ins, out, answer ? "stop" : "tool_calls", 1];
};
const ses = (id: string, n: number, t0 = 0, dt = 30, at?: number, children: Session[] = []): Session =>
  ({ id, variant: "v1", at, children, turns: Array.from({ length: n }, (_, i) => turn(t0 + i * dt, i)) }) as Session;
const tree = (id: string, sessions: Session[]): Tree => {
  const end = Math.max(0, ...sessions.flatMap(function all(s): number[] { return [...s.turns.map((u) => u[0]), ...s.children.flatMap(all)]; }));
  return { id, key: "farm", start: 0, end, last: end, last_at: end, traits, sessions } as Tree;
};
/** One session of n turns plus a subagent forked at turn 1 that grows alongside. */
const T = (n: number) => tree("tok:spell:1", [ses("S", n, 0, 30, undefined, n > 1 ? [ses("S.sub", Math.ceil(n / 2), 35, 60, 1)] : [])]);

const build = (t: Tree, cut: number, k: Knobs = KNOBS.oak) => buildTreeGeometry(t, cut, k, norm, place, traits);
const vox = (b: TreeBuffers, o: Partial<VoxelOpts> = {}) => voxelizeTree(b, { ...CHOSEN, ...o }, "tok:spell:1");

type Cell = { kind: number; shade: number; born: number; buried: number };
const cellMap = (v: VoxelCells) => {
  const m = new Map<string, Cell>();
  for (let i = 0; i < v.count; i++)
    m.set(`${v.coords[i * 3]},${v.coords[i * 3 + 1]},${v.coords[i * 3 + 2]}`, { kind: v.kind[i], shade: v.shade[i], born: v.born[i], buried: v.buried[i] });
  return m;
};
/** `a`'s cells all in `b` with identical attributes, and `a`'s arrays a prefix of `b`'s (append-only). */
const expectGrowsFrom = (a: VoxelCells, b: VoxelCells, what: string) => {
  expect(b.count, what).toBeGreaterThanOrEqual(a.count);
  const mb = cellMap(b);
  for (const [k, c] of cellMap(a)) expect(mb.get(k), `${what} ${k}`).toEqual(c);
  expect(Array.from(b.coords.subarray(0, a.count * 3)), what).toEqual(Array.from(a.coords.subarray(0, a.count * 3)));
  expect(Array.from(b.kind.subarray(0, a.count)), what).toEqual(Array.from(a.kind.subarray(0, a.count)));
  expect(Array.from(b.shade.subarray(0, a.count)), what).toEqual(Array.from(a.shade.subarray(0, a.count)));
  expect(Array.from(b.born.subarray(0, a.count)), what).toEqual(Array.from(a.born.subarray(0, a.count)));
  expect(Array.from(b.buried.subarray(0, a.count)), what).toEqual(Array.from(a.buried.subarray(0, a.count)));
};
const arrays = (v: VoxelCells) => [Array.from(v.coords), Array.from(v.kind), Array.from(v.shade), Array.from(v.born), Array.from(v.buried), v.count];
/** Every buried cell has all 6 neighbours in the cell set, kept no later than itself (a wood block's outer cell is born
 * one scale-in after its claim): nothing can see into the solid. */
const expectEnclosed = (v: VoxelCells, what: string) => {
  const m = cellMap(v);
  let n = 0;
  for (const [k, c] of m) {
    if (!c.buried) continue;
    n++;
    const [x, y, z] = k.split(",").map(Number);
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      const nb = m.get(`${x + dx},${y + dy},${z + dz}`);
      expect(nb, `${what}: buried ${k} open toward ${dx},${dy},${dz}`).toBeDefined();
      expect(nb!.born, `${what}: ${k}`).toBeLessThanOrEqual(Math.fround(c.born + 0.8));
    }
  }
  return n;
};

/** forest.py `merge_turns` (LOD_CAP 2000, blocks of 8): the server's level of detail for long sessions. */
const mergeTurns = (turns: Turn[], cap = 2000, block = 8): Turn[] => {
  if (turns.length <= cap) return turns;
  const rest = turns.slice(cap), full = rest.length - (rest.length % block), out = turns.slice(0, cap);
  for (let i = 0; i < full; i += block) {
    const g = rest.slice(i, i + block), cached = g.map((t) => t[3]).filter((x): x is number => x != null);
    out.push([g[block - 1][0], Math.max(...g.map((t) => t[1])), g.reduce((a, t) => a + t[2], 0),
      cached.length ? Math.floor(cached.reduce((a, x) => a + x, 0) / cached.length) : null,
      g.flatMap((t) => t[4]).slice(0, 6), g.flatMap((t) => t[5]).slice(0, 6), g[block - 1][6], g.reduce((a, t) => a + t[7], 0)]);
  }
  return [...out, ...rest.slice(full)];
};
/** forest.py `merged_index`. */
const mergedIndex = (i: number, n: number, cap = 2000, block = 8) => {
  if (i < cap) return i;
  const full = (n - cap) - ((n - cap) % block), j = i - cap;
  return cap + (j < full ? Math.floor(j / block) : full / block + (j - full));
};

describe("voxelizer", () => {
  it("is deterministic, bit for bit", () => {
    const a = vox(build(T(24), 800)), b = vox(build(T(24), 800));
    expect(a.count).toBeGreaterThan(50);
    expect(arrays(b)).toEqual(arrays(a));
    // and through the geometry builder (the worker path)
    expect(arrays(build(T(24), 800).voxels)).toEqual(arrays(build(T(24), 800).voxels));
  });

  it("blossoms sit on the crown: never stacked on another blossom", () => {
    const v = vox(build(T(300), 9000));
    const m = cellMap(v);
    let n = 0;
    for (const [k, c] of m) {
      if (c.kind !== 1) continue;
      n++;
      const [x, y, z] = k.split(",").map(Number);
      expect(m.get(`${x},${y - 1},${z}`)?.kind, k).not.toBe(1);
    }
    expect(n).toBeGreaterThan(5);
  });

  it("has every kind: wood, leaves, failed tools and blossoms (blossoms off: none)", () => {
    const v = vox(build(T(24), 800));
    const kinds = new Set(Array.from(v.kind.subarray(0, v.count)));
    expect([...kinds].sort()).toEqual([0, 1, 2, 3]);
    const off = vox(build(T(24), 800), { blossoms: false });
    expect(Array.from(off.kind.subarray(0, off.count))).not.toContain(1);
  });

  it("is prefix-stable over 2..40 turns: turn k's cells unchanged after k+1, which only adds cells", () => {
    for (const [name, k] of [["oak", KNOBS.oak], ["pine", KNOBS.pine], ["palm", KNOBS.palm], ["bamboo", KNOBS.bamboo], ["willow", KNOBS.willow], ["round", KNOBS.round]] as const) {
      let prev: VoxelCells | null = null;
      for (let n = 2; n <= 40; n++) {
        // after turn n (t = (n−1)·30), before turn n+1; the next tree also moves the cut past its new turn
        const cur = vox(build(T(n), n * 30 - 15, k));
        if (prev) expectGrowsFrom(prev, cur, `${name} n=${n}`);
        prev = cur;
      }
      expect(prev!.count, name).toBeGreaterThan(40);
    }
  });

  it("cut sweep: the cells at cut c are a subset of the cells at c + 10 s, attributes identical", () => {
    const t = tree("tok:spell:2", [
      ses("A", 30, 0, 30, undefined, [ses("A.sub", 12, 40, 30, 1), ses("A.sub2", 6, 400, 25, 8)]),
      ses("B", 25, 200, 33),
    ]);
    for (const [name, k] of [["oak", KNOBS.oak], ["palm", KNOBS.palm], ["baobab", KNOBS.baobab], ["banyan", KNOBS.banyan]] as const) {
      let prev: VoxelCells | null = null;
      for (let c = 0; c <= 1300; c += 10) {
        const cur = voxelizeTree(build(t, c, k), CHOSEN, t.id);
        if (prev) expectGrowsFrom(prev, cur, `${name} cut=${c}`);
        prev = cur;
      }
    }
  });

  it("girth grows outward: the trunk gains wood cells over time and keeps every one it had", () => {
    const t = T(120);
    const wood = (v: VoxelCells) => {
      let n = 0;
      for (let i = 0; i < v.count; i++) if (v.kind[i] === 3 && v.coords[i * 3 + 1] <= 1) n++;
      return n;
    };
    const early = vox(build(t, 300)), late = vox(build(t, 3600));
    expectGrowsFrom(early, late, "girth");
    expect(wood(late)).toBeGreaterThan(wood(early));
  });

  it("a wood cell is never taken over by leaves", () => {
    for (let n = 3; n <= 30; n += 3) {
      const a = vox(build(T(n), n * 30 - 15)), b = vox(build(T(n + 3), n * 30 + 75));
      const mb = cellMap(b);
      for (const [k, c] of cellMap(a)) if (c.kind === 3) expect(mb.get(k)?.kind, `n=${n} ${k}`).toBe(3);
    }
  });

  it("2000 turns: stays under the per-tree cap, and no cell is removed as it grows", () => {
    const t = T(2000);
    let prev: VoxelCells | null = null;
    for (const c of [3000, 15000, 30000, 45000, 60000, 60030]) {
      const cur = vox(build(t, c));
      expect(cur.count).toBeLessThanOrEqual(VOXEL_CAP);
      if (prev) expectGrowsFrom(prev, cur, `cut=${c}`);
      prev = cur;
    }
    // the ramified crown saturates (shoots shrink with depth): a 2000-turn session is far below the cap
    expect(prev!.count).toBeGreaterThan(200);
  });

  it("buried cells are enclosed by kept cells; drawn cells form the shell", () => {
    for (const [name, k] of [["oak", KNOBS.oak], ["palm", KNOBS.palm], ["baobab", KNOBS.baobab]] as const) {
      const v = vox(build(T(60), 1800, k));
      const n = expectEnclosed(v, name);
      expect(n, `${name}: some interior`).toBeGreaterThan(0);
      expect(n, name).toBeLessThan(v.count);
    }
  });

  it("a capped tree has no hole next to a drawn cell: cells the cap drops never count as solid", () => {
    const full = vox(build(T(60), 1800), { cap: 1e9 }).count;
    for (const cap of [60, 250, 700, Math.floor(full * 0.9)]) {
      const v = vox(build(T(60), 1800), { cap });
      expect(v.count).toBe(cap);
      expectEnclosed(v, `cap ${cap}`);
    }
    // the widest kind of tree, at its real cap
    const wide = tree("tok:wide", Array.from({ length: 40 }, (_, k) => ses(`W${k}`, 30, k * 2000)));
    const w = voxelizeTree(build(wide, 1e6), CHOSEN, wide.id);
    expectEnclosed(w, "wide");
  });

  it("float32 times: two subagents whose turns land 0.03 s apart (one float32 at 7 d) stay stable at every cut", () => {
    const T0 = 600000; // float32 spacing here is 0.0625 s: t + 0.002 and t + 0.03 round to the same value
    // the parent's fork turns (1 at T0 − 60, 2 at T0 − 30) come before the subagents' first turns, as in real data
    const t = tree("tok:f32", [ses("P", 4, T0 - 90, 30, undefined, [ses("P.a", 8, T0 + 10.002, 30, 1), ses("P.b", 8, T0 + 10.03, 30, 2)])]);
    expect(Math.fround(T0 + 10.002)).toBe(Math.fround(T0 + 10.03));
    let prev: VoxelCells | null = null, newCells = 0;
    // cuts between each pair of near-simultaneous turns, and around them
    const cuts = Array.from({ length: 8 }, (_, k) => [-0.5, 0.001, 0.01, 0.02, 0.031, 0.06, 0.2, 5].map((x) => T0 + 10 + k * 30 + x)).flat();
    for (const c of cuts) {
      const cur = voxelizeTree(build(t, c), CHOSEN, t.id);
      if (prev) {
        expectGrowsFrom(prev, cur, `cut=${c}`);
        newCells += cur.count - prev.count;
      }
      prev = cur;
    }
    expect(newCells).toBeGreaterThan(100);
  });

  it("server LOD past 2000 turns: the client keeps held singles over a closing block, so cells are only added", () => {
    // the server's view: one long session (and a subagent forked past the cap), merged by forest.py's rule
    const N = 2052, raw = Array.from({ length: N }, (_, i) => turn(i * 30, i));
    // the subagent's turns up to the parent's newest turn (none arrives behind a cut already built: no late turns here)
    const subRaw = (upTo: number) => Array.from({ length: 60 }, (_, i) => turn(2010 * 30 + 7 + i * 40, i)).filter((u) => u[0] <= upTo);
    const serve = (n: number): Tree => {
      const sub = { id: "L.sub", variant: "v1", at: mergedIndex(2010, n), children: [], turns: subRaw((n - 1) * 30) };
      return tree("tok:lod", [{ id: "L", variant: "v1", children: n > 2011 ? [sub] : [], turns: mergeTurns(raw.slice(0, n)) } as Session]);
    };
    // polls: one turn at a time, then gaps that skip part of a block (closing over held and unseen turns alike)
    const polls = [1995, 1999, 2000, 2001, 2003, 2004, 2007, 2008, 2009, 2012, 2016, 2017, 2023, 2030, 2033, 2040, 2041, 2048, 2052];
    let held: Tree | undefined, prev: VoxelCells | null = null, prevRaw: Set<string> | null = null, lost = 0;
    for (const n of polls) {
      const sent = serve(n), cut = (n - 1) * 30 + 5;
      held = shiftTree(sent, 0, held);
      const cur = voxelizeTree(build(held, cut), CHOSEN, held.id);
      if (prev) expectGrowsFrom(prev, cur, `n=${n}`);
      prev = cur;
      // control: taking the server's list as sent moves or drops held turns' voxel inputs (foliage points, with their
      // born) whenever a block closes; the solid crown may hide it, the inputs show it
      const f = build(sent, cut).vox.foliage, asSent = new Set<string>();
      for (let i = 0; i < f.length; i += 6) asSent.add(Array.from(f.subarray(i, i + 6)).join(","));
      if (prevRaw) for (const k of prevRaw) if (!asSent.has(k)) lost++;
      prevRaw = asSent;
    }
    // held singles plus the merged entries that stood only for turns never seen one by one
    const ht = held!.sessions[0].turns;
    expect(ht.length).toBeGreaterThan(LOD_CAP + 30);
    expect(ht.some((u) => u[7] > 1), "some block accepted merged (its turns were never held)").toBe(true);
    for (let k = 1; k < ht.length; k++) expect(ht[k][0]).toBeGreaterThan(ht[k - 1][0]);
    expect(lost, "the merged list as sent does lose grown inputs (the test has teeth)").toBeGreaterThan(0);
  });

  it("a small cap: new turns add only up to the cap, at the highest density priority; nothing grown is removed", () => {
    let prev: VoxelCells | null = null;
    for (let n = 2; n <= 40; n += 2) {
      const b = build(T(n), n * 30 - 15), cur = vox(b, { cap: 60 }), full = vox(b, { cap: 1e9 });
      expect(cur.count).toBeLessThanOrEqual(60);
      if (prev) expectGrowsFrom(prev, cur, `cap n=${n}`);
      // the capped cells are exactly the first ones of the uncapped order (instant, then density priority)
      // (burial differs: the capped tree never counts the dropped cells as solid)
      expect(arrays(cur).filter((_, k) => k !== 4), `cap n=${n}`).toEqual(arrays({ ...full, coords: full.coords.slice(0, cur.count * 3),
        kind: full.kind.slice(0, cur.count), shade: full.shade.slice(0, cur.count), born: full.born.slice(0, cur.count),
        buried: full.buried.slice(0, cur.count), count: cur.count }).filter((_, k) => k !== 4));
      prev = cur;
    }
    expect(prev!.count).toBe(60);
  });

  it("1000 turns over 20 subagent sessions: deterministic, prefix-stable, within the cap, no NaN", () => {
    const subs = Array.from({ length: 20 }, (_, j) => ses(`M.sub${j}`, 45, 100 + j * 600, 20, (j * 3) % 40));
    const t = tree("tok:spell:3", [ses("M", 100, 0, 120, undefined, subs)]);
    const t0 = performance.now();
    const b = build(t, 13000);
    const t1 = performance.now();
    const v = voxelizeTree(b, CHOSEN, t.id);
    const t2 = performance.now();
    console.log(`1000 turns / 20 subagents: ${v.count} cells; geometry ${(t1 - t0).toFixed(0)} ms, voxelize ${(t2 - t1).toFixed(0)} ms`);
    expect(v.count).toBeLessThanOrEqual(VOXEL_CAP);
    expect(arrays(voxelizeTree(build(t, 13000), CHOSEN, t.id))).toEqual(arrays(v));
    expectGrowsFrom(voxelizeTree(build(t, 6000), CHOSEN, t.id), v, "1000 turns");
    expect(v.born.subarray(0, v.count).every((x) => Number.isFinite(x))).toBe(true);
  });

  it("the density hash is order-independent: shuffled foliage gives the same cells", () => {
    const b = build(T(30), 900);
    const n = b.vox.foliage.length / 6, perm = Array.from({ length: n }, (_, i) => (i * 7919) % n);
    expect(new Set(perm).size).toBe(n);
    const shuffled = new Float32Array(b.vox.foliage.length);
    perm.forEach((p, i) => shuffled.set(b.vox.foliage.subarray(p * 6, p * 6 + 6), i * 6));
    const a = vox(b), s = vox({ ...b, vox: { ...b.vox, foliage: shuffled } });
    expect(arrays(s)).toEqual(arrays(a));
    // thinning picks the points the crown is fitted to: another density, another (solid) crown
    expect(arrays(vox(b, { density: 0.1 }))).not.toEqual(arrays(a));
    expect(arrays(vox(b, { density: 1 }))).not.toEqual(arrays(a));
  });

  it("shade and dark flag come from the tree id and the coordinate: same cell, same shade; dark share ≈ dark", () => {
    const v = vox(build(T(40), 1300));
    let dark = 0;
    for (let i = 0; i < v.count; i++) {
      expect(v.shade[i] & 0x7f).toBeLessThan(CHOSEN.shades);
      if (v.shade[i] & 0x80) dark++;
    }
    expect(dark / v.count).toBeGreaterThan(0.2);
    expect(dark / v.count).toBeLessThan(0.48);
    const other = voxelizeTree(build(T(40), 1300), CHOSEN, "tok:other");
    expect(Array.from(other.shade.subarray(0, other.count))).not.toEqual(Array.from(v.shade.subarray(0, v.count)));
  });

  it("no NaN, coordinates in the tree's local grid, for every species", () => {
    for (const [name, k] of Object.entries(KNOBS)) {
      const v = vox(build(T(30), 900, k));
      expect(v.count, name).toBeGreaterThan(0);
      for (let i = 0; i < v.count; i++) expect(Number.isNaN(v.born[i]), name).toBe(false);
      for (let i = 0; i < v.count * 3; i++) expect(Number.isInteger(v.coords[i]), name).toBe(true);
      // local grid: the tree stands at its own origin, not at place (3, −2)
      let sx = 0;
      for (let i = 0; i < v.count; i++) sx += v.coords[i * 3];
      expect(Math.abs(sx / v.count), name).toBeLessThan(12);
      for (let i = 0; i < v.count; i++) expect(v.coords[i * 3 + 1], name).toBeGreaterThanOrEqual(0);
    }
  });

  it("the worker voxelizes once and returns each level's drawn cells (Q4: no level work on the main thread)", () => {
    const t = T(10);
    const { reply, transfer } = handleBuild({ id: 1, key: t.id, tree: t, cut: 290, knobs: KNOBS.oak, norm, place, traits });
    if (!("buffers" in reply)) throw new Error(reply.error);
    const full = build(t, 290).voxels, ls = reply.buffers.levels;
    expect(reply.buffers.cells).toBe(full.count);
    for (const L of [0, 1, 2]) {
      const want = levelCells(full, L, 290), got = ls[L];
      expect(got.count).toBe(want.shown.length);
      expect(Array.from(got.born)).toEqual(Array.from(want.shown, (i) => want.born[i]));
      expect(Array.from(got.coords)).toEqual(Array.from(want.shown).flatMap((i) => Array.from(want.coords.subarray(i * 3, i * 3 + 3))));
      for (const a of [got.coords, got.kind, got.shade, got.born, got.shown]) expect(transfer).toContain(a.buffer);
    }
    // the voxelizer's inputs stay in the worker
    expect("vox" in reply.buffers).toBe(false);
  });

  it("R5: the crown is one solid, round ellipsoid crown (a union of the per-turn fits), lifted off the ground", () => {
    for (const k of [KNOBS.oak, KNOBS.round]) {
      const v = build(T(40), 1300, k).voxels, m = cellMap(v);
      const leaf = [...m].filter(([, c]) => c.kind === KIND.leaf);
      expect(leaf.length).toBeGreaterThan(300);
      // solid: most crown cells are enclosed (a fitted solid, not a scatter of spheres around the twigs)
      const inside = leaf.filter(([key]) => {
        const [x, y, z] = key.split(",").map(Number);
        return [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].every(([a, b2, c]) => m.has(`${x + a},${y + b2},${z + c}`));
      });
      expect(inside.length / leaf.length).toBeGreaterThan(0.45);
      // lifted: no crown cell within 0.75 u of the ground
      const low = Math.min(...leaf.map(([key]) => Number(key.split(",")[1])));
      expect(low * CHOSEN.size).toBeGreaterThanOrEqual(0.75);
      // round: its footprint is about as deep as it is wide
      const xs = leaf.map(([key]) => Number(key.split(",")[0])), zs = leaf.map(([key]) => Number(key.split(",")[2]));
      const w = Math.max(...xs) - Math.min(...xs), d = Math.max(...zs) - Math.min(...zs);
      expect(Math.min(w, d) / Math.max(w, d)).toBeGreaterThan(0.6);
    }
  });

  it("L2: a short blocky trunk shows under the crown: at least 2 × 2 blocks of 0.5 u at the base", () => {
    for (const k of [KNOBS.oak, KNOBS.pine, KNOBS.round]) {
      const v = build(T(20), 700, k).voxels, m = cellMap(v);
      for (let x = -2; x <= 1; x++) for (let z = -2; z <= 1; z++) expect(m.get(`${x},0,${z}`)?.kind, `${x},0,${z}`).toBe(KIND.wood);
      // wood under the crown's lowest cell: the trunk is seen
      const crownLow = Math.min(...[...m].filter(([, c]) => c.kind !== KIND.wood).map(([key]) => Number(key.split(",")[1])));
      expect([...m].some(([key, c]) => c.kind === KIND.wood && Number(key.split(",")[1]) < crownLow && !c.buried)).toBe(true);
    }
  });

  it("L3: a failed tool is one cube on the crown (leaf wins ties): a few percent of the crown at most", () => {
    const b = build(T(40), 1300), v = vox(b), F = b.vox.foliage;
    let failedPts = 0;
    for (let i = 0; i < F.length; i += 6) if (F[i + 5] > 0) failedPts++;
    const kinds = Array.from(v.kind.subarray(0, v.count));
    const failed = kinds.filter((k) => k === KIND.failed).length, crown = kinds.filter((k) => k !== KIND.wood).length;
    expect(failedPts).toBeGreaterThan(0);
    expect(failed).toBeLessThanOrEqual(failedPts);
    expect(failed / crown).toBeLessThan(0.03);
  });

  it("cell edge is the mockup's chosen crown cube, 0.25 u", () => {
    expect(CHOSEN.size).toBe(0.25);
  });

  it("palm: old crowns stay where they formed (lower tiers), the newest tier sits at the trunk top, append-only", () => {
    const palm = tree("tok:palm", [ses("A", 12, 0), ses("B", 12, 2000), ses("C", 12, 4000)]);
    let prev: VoxelCells | null = null;
    for (const cut of [10, 100, 400, 1000, 2010, 2100, 2500, 4010, 4100, 4500, 6000]) {
      const b = build(palm, cut, KNOBS.palm), v = b.voxels;
      if (prev) expectGrowsFrom(prev, v, `palm cut=${cut}`);
      prev = v;
      // the voxel inputs: each session's base sample sits on the trunk axis at its own tier; the trunk is the last line
      const L = b.vox.lines, W = b.vox.wood, nl = L.length - 1, ox = b.vox.origin[0], oz = b.vox.origin[1];
      const trunkTop = W[(L[nl] - 1) * 6 + 1];
      const bases: { y: number; born: number }[] = [];
      for (let l = 0; l < nl - 1; l++) {
        const s0 = L[l] * 6;
        if (Math.hypot(W[s0] - ox, W[s0 + 2] - oz) < 0.15 && W[s0 + 1] > 0.3) bases.push({ y: W[s0 + 1], born: W[s0 + 3] });
      }
      bases.sort((p, q) => p.born - q.born);
      expect(bases.length, `cut=${cut}`).toBeGreaterThan(0);
      for (let k = 1; k < bases.length; k++) expect(bases[k].y, `tiers rise, cut=${cut}`).toBeGreaterThanOrEqual(bases[k - 1].y);
      const newest = bases[bases.length - 1];
      expect(trunkTop, `trunk reaches the newest tier, cut=${cut}`).toBeGreaterThanOrEqual(newest.y - TRUNK_STEP);
      // right after a session's first turn (the newest event), its tier IS the trunk top
      if (cut % 2000 === 10) expect(Math.abs(trunkTop - newest.y), `newest tier at the top, cut=${cut}`).toBeLessThanOrEqual(TRUNK_STEP);
      // older tiers stay lower down the trunk
      if (bases.length > 1) expect(bases[0].y, `cut=${cut}`).toBeLessThan(newest.y);
      // the camera's crown bounds are the voxel extent
      let topCell = 0;
      for (let i = 0; i < v.count; i++) topCell = Math.max(topCell, (v.coords[i * 3 + 1] + 1) * CHOSEN.size);
      expect(b.crown.top, `cut=${cut}`).toBe(Math.max(1, topCell));
    }
  });
});

// ---------------- Task 2: the renderer (engine/voxels.ts): levels, instances, swaps, budget, picking ----------------

const SEED = "tok:spell:1";
/** A level's shown cells by coordinate: their instance attributes (centre, edge, colour, born). */
const instancesOf = (m: THREE.InstancedMesh) => {
  const out = new Map<string, { m: number[]; c: number[]; born: number }>();
  const born = (m.geometry.getAttribute("iborn") as THREE.InstancedBufferAttribute).array as Float32Array;
  const M = m.instanceMatrix.array as Float32Array, C = m.instanceColor!.array as Float32Array;
  for (let i = 0; i < m.count; i++) {
    const s = M[i * 16], key = [12, 13, 14].map((k) => Math.round(M[i * 16 + k] / s - 0.5)).join(",");
    out.set(key, { m: Array.from(M.subarray(i * 16, i * 16 + 16)), c: Array.from(C.subarray(i * 3, i * 3 + 3)), born: born[i] });
  }
  return out;
};

describe("voxel renderer", () => {
  const mats = createMaterials();

  it("the instance count matches the cells: one cube per shown cell, buried cells (enclosed by grown cells) not drawn", () => {
    for (const cut of [100, 400, 1200]) {
      const b = build(T(40), cut), v = b.voxels;
      const lc = levelCells(v, 0, cut);
      expect(lc.count, `cut ${cut}`).toBe(v.count);
      const set = cellMap(v);
      // drawn = not enclosed by six neighbours that are fully grown at the cut
      let expectShown = 0;
      for (const [k] of set) {
        const [x, y, z] = k.split(",").map(Number);
        const enclosed = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].every(([dx, dy, dz]) => {
          const n = set.get(`${x + dx},${y + dy},${z + dz}`);
          return n !== undefined && n.born <= cut - SCALE_IN_S;
        });
        if (!enclosed) expectShown++;
      }
      expect(lc.shown.length, `cut ${cut}`).toBe(expectShown);
      expect(lc.shown.length).toBeLessThan(v.count); // a solid crown has an inside
      const mesh = voxelMesh(lc, cubeColors(lc, SEED), mats.voxel, mats.cube);
      expect(mesh.count).toBe(lc.shown.length);
      const inst = instancesOf(mesh);
      expect(inst.size).toBe(lc.shown.length);
      for (const i of lc.shown) {
        const k = `${lc.coords[i * 3]},${lc.coords[i * 3 + 1]},${lc.coords[i * 3 + 2]}`, c = inst.get(k)!;
        expect(c, k).toBeDefined();
        expect(c.m[0]).toBeCloseTo(CHOSEN.size, 9); // a cube of the cell's edge, centred in the cell
        expect(c.m[13]).toBeCloseTo((lc.coords[i * 3 + 1] + 0.5) * CHOSEN.size, 6);
        expect(c.born).toBe(lc.born[i]);
      }
      mesh.geometry.dispose();
    }
  });

  it("macro-cells (levels 1, 2): the union of their fine cells, kind/shade/born of the earliest, solid with no holes", () => {
    const v = build(T(40), 1200).voxels, fine = cellMap(v);
    for (const L of [1, 2]) {
      const lc = levelCells(v, L, 1200), k = 1 << L;
      expect(lc.size).toBe(CHOSEN.size * k);
      const macro = new Map<string, number>();
      for (let i = 0; i < lc.count; i++) macro.set(`${lc.coords[i * 3]},${lc.coords[i * 3 + 1]},${lc.coords[i * 3 + 2]}`, i);
      // every fine cell lies in a drawn-or-enclosed macro-cell; no macro-cell without a fine cell
      const members = new Map<string, Cell[]>();
      for (const [key, c] of fine) {
        const [x, y, z] = key.split(",").map(Number), mk = `${Math.floor(x / k)},${Math.floor(y / k)},${Math.floor(z / k)}`;
        expect(macro.has(mk), `L${L} ${key}`).toBe(true);
        (members.get(mk) ?? members.set(mk, []).get(mk)!).push(c);
      }
      expect(members.size).toBe(lc.count);
      // kind, shade and born: its earliest fine cell's (the first in the voxelizer's order), fixed once it exists (Q3)
      const first = new Map<string, number>();
      for (let f = 0; f < v.count; f++) {
        const mk = `${v.coords[f * 3] >> L},${v.coords[f * 3 + 1] >> L},${v.coords[f * 3 + 2] >> L}`;
        if (!first.has(mk)) first.set(mk, f);
      }
      for (const [mk, i] of macro) {
        const f = first.get(mk)!;
        expect(lc.born[i]).toBe(v.born[f]);
        expect(lc.kind[i], `L${L} ${mk}`).toBe(v.kind[f]);
        expect(lc.shade[i]).toBe(v.shade[f]);
      }
      // crowns stay solid: every macro-cell not drawn is enclosed by six grown macro-cells
      const shown = new Set(lc.shown);
      for (const [mk, i] of macro) {
        if (shown.has(i)) continue;
        const [x, y, z] = mk.split(",").map(Number);
        for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
          const j = macro.get(`${x + dx},${y + dy},${z + dz}`);
          expect(j, `L${L} hidden ${mk} open`).toBeDefined();
          expect(lc.born[j!]).toBeLessThanOrEqual(1200 - SCALE_IN_S);
        }
      }
      expect(lc.shown.length).toBeLessThan(levelCells(v, L - 1, 1200).shown.length);
    }
  });

  it("a rebuild swap keeps the cubes identical: no morph; new cells start at scale 0 and scale in over 0.8 s", () => {
    const t = T(30), times = [...new Set(t.sessions.flatMap(function all(s: Session): number[] { return [...s.turns.map((u) => u[0]), ...s.children.flatMap(all)]; }))].sort((a, b) => a - b);
    for (const L of [0, 1, 2]) {
      for (const [c1, c2] of [[100, 120], [120, 140], [300, 305], [600, 620], [620, 635]]) {
        // the hand-over is at c2 (a turn lands, or 20 s pass): the old build and the new one, both at the swap instant
        const a = build(t, c1).voxels, b = build(t, c2).voxels;
        const la = levelCells(a, L, c1), lb = levelCells(b, L, c2);
        const ma = voxelMesh(la, cubeColors(la, SEED), mats.voxel, mats.cube), mb = voxelMesh(lb, cubeColors(lb, SEED), mats.voxel, mats.cube);
        const ia = instancesOf(ma), ib = instancesOf(mb);
        const bornB = new Map<string, number>();
        for (let i = 0; i < lb.count; i++) bornB.set(`${lb.coords[i * 3]},${lb.coords[i * 3 + 1]},${lb.coords[i * 3 + 2]}`, lb.born[i]);
        for (const [k, x] of ia) {
          const y = ib.get(k);
          if (y) {
            // the same cube: place, edge and born; at level 0 the same colour (a macro-cell's majority may turn)
            expect(y.m, `L${L} ${c1}→${c2} ${k}`).toEqual(x.m);
            expect(y.born).toBe(x.born);
            if (L === 0) expect(y.c).toEqual(x.c);
          } else {
            // no longer drawn: enclosed by cells already grown at c2 (it was hidden from every side anyway)
            expect(bornB.get(k), `L${L} ${k} dropped`).toBeDefined();
          }
        }
        // the new cubes: born after the old cut, and at the swap they show at scaleIn(c2 − born)
        for (const [k, y] of ib) {
          if (ia.has(k) || (bornB.get(k)! <= c1 && L > 0)) continue;
          if (!ia.has(k) && y.born <= c1) continue; // an old cell that is newly exposed: fully grown either way
          expect(y.born, `L${L} new ${k}`).toBeGreaterThan(c1);
          if (times.includes(c2)) expect(scaleIn(c2 - y.born), `L${L} new ${k} at the swap`).toBeLessThanOrEqual(scaleIn(c2 - (c1 + 1e-6)));
        }
        ma.geometry.dispose();
        mb.geometry.dispose();
      }
    }
  });

  it("scale-in is a smoothstep from 0 to 1 over 0.8 s; the cyan tint lasts ~12 s and fades out; the shaders say so", () => {
    expect(SCALE_IN_S).toBe(0.8);
    expect(scaleIn(-1)).toBe(0);
    expect(scaleIn(0)).toBe(0);
    expect(scaleIn(0.4)).toBeCloseTo(0.5, 9);
    expect(scaleIn(0.8)).toBe(1);
    expect(scaleIn(5)).toBe(1);
    let prev = 0;
    for (let a = 0; a <= 1; a += 1 / 600) {
      const s = scaleIn(a);
      expect(s).toBeGreaterThanOrEqual(prev);
      expect(s - prev).toBeLessThanOrEqual((1.5 / SCALE_IN_S) * (1 / 600) + 1e-9); // max slope of a smoothstep
      prev = s;
    }
    expect(tintAt(-0.1)).toBe(0);
    expect(tintAt(0)).toBe(1);
    expect(tintAt(6)).toBeGreaterThan(0.4);
    expect(tintAt(TINT_S)).toBe(0);
    expect(TINT_S).toBe(12);
    const src = VOXEL_SHADER_PATCH;
    expect(src).toContain(`/${SCALE_IN_S.toFixed(1)}`);
    expect(src).toContain("transformed*=max(gs,0.0001)");
    expect(src).toContain("vec3(0.3,0.75,1.0)"); // the mockup's cyan tint
    expect(src).toContain(`/${TINT_S.toFixed(1)}`);
    expect(SaturationShader.uniforms.uSat.value).toBe(1.4); // one saturation pass over the whole frame
    expect(src).not.toContain("1.4");
    expect(mats.voxel).toBeInstanceOf(THREE.MeshToonMaterial);
    // 4 cel steps: 0.4, 0.6, 0.8, 1.0
    const g = mats.voxel.gradientMap!.image.data as Uint8Array;
    expect([...new Set(g)]).toEqual([102, 153, 204, 255]);
  });

  it("level of detail by distance, with hysteresis: a camera hovering at a threshold does not flicker", () => {
    expect(levelFor(0)).toBe(0);
    expect(levelFor(LOD_D[0] * 1.2)).toBe(1);
    expect(levelFor(LOD_D[1] * 1.2)).toBe(2);
    // around the first threshold: from 0 it stays 0 until past the band, from 1 it stays 1 until below the band
    for (const d of [LOD_D[0] * (1 - LOD_HYST / 2), LOD_D[0], LOD_D[0] * (1 + LOD_HYST / 2)]) {
      expect(levelFor(d, 0), `${d} from 0`).toBe(0);
      expect(levelFor(d, 1), `${d} from 1`).toBe(1);
    }
    expect(levelFor(LOD_D[0] * (1 + LOD_HYST * 1.01), 0)).toBe(1);
    expect(levelFor(LOD_D[0] * (1 - LOD_HYST * 1.01), 1)).toBe(0);
    // a jump far out goes straight to 2
    expect(levelFor(LOD_D[1] * 3, 0)).toBe(2);
    // the cross-fade: the two levels' dithers are complementary at every fade (no pixel twice, none missing)
    for (const f of [0, 0.3, 0.5, 0.9, 1])
      for (let h = 0; h < 1; h += 0.01) expect(Number(fadeKeep(h, f, true)) + Number(fadeKeep(h, f, false))).toBe(1);
    expect(FADE_S).toBe(0.3);
  });

  it("the budget picks coarser levels for the farthest trees first, never dropping a cell: the count fits", () => {
    const counts = [[5000, 1400, 400], [4000, 1100, 300], [3000, 900, 250], [2000, 600, 200]];
    const trees = counts.map((c, i) => ({ id: "t" + i, d: 50 + 40 * i, level: 0, count: (L: number) => c[L] }));
    expect(fitBudget(trees, 20_000)).toEqual(new Map(trees.map((t) => [t.id, 0]))); // fits: nothing changes
    const f = fitBudget(trees, 8000);
    const total = trees.reduce((a, t) => a + t.count(f.get(t.id)!), 0);
    expect(total).toBeLessThanOrEqual(8000);
    // the farthest went coarser first, and no nearer tree is coarser than a farther one
    for (let i = 1; i < trees.length; i++) expect(f.get(trees[i].id)!).toBeGreaterThanOrEqual(f.get(trees[i - 1].id)!);
    expect(f.get("t3")).toBeGreaterThan(0);
    // natural levels are a floor
    const g = fitBudget(trees.map((t) => ({ ...t, level: 2 })), 1);
    for (const t of trees) expect(g.get(t.id)).toBe(2);
  });

  it("Q1: the budget's choice has per-tree hysteresis: two trees at the same distance never trade levels", () => {
    const B = new BudgetLevels(0.92), count = (L: number) => [1000, 300, 80][L];
    let swaps = 0, prev: string | null = null;
    for (let f = 0; f < 2000; f++) {
      // A and B swap order by a hair every frame; a camera hovering over the row
      const eps = f % 2 ? 0.01 : -0.01;
      const lv = B.levels([{ id: "A", d: 500 + eps, level: 0, count }, { id: "B", d: 500 - eps, level: 0, count }, { id: "C", d: 100, level: 0, count }], 2500, f * 100);
      const key = `${lv.get("A")}${lv.get("B")}`;
      expect(lv.get("C")).toBe(0);
      expect(key === "10" || key === "01").toBe(true); // one of the two goes coarser: 1300 + 1000 ≤ 2300
      if (prev !== null && key !== prev) swaps++;
      prev = key;
    }
    expect(swaps).toBe(0);
    // a budget that moves by under 10 % keeps the choice; a big move re-ranks at once
    const B2 = new BudgetLevels(0.92), trees = (d: number) => [{ id: "A", d, level: 0, count }, { id: "B", d: 300, level: 0, count }];
    expect(B2.levels(trees(400), 1500, 0).get("A")).toBe(1);
    expect(B2.levels(trees(200), 1450, 10).get("A")).toBe(1); // A is nearer now, but nothing moved by 10 %: held
    expect(B2.levels(trees(200), 1450, 6000).get("B")).toBe(1); // 5 s on: re-ranked, B (farther) goes coarser...
    expect(B2.levels(trees(200), 1450, 6010).get("A")).toBe(0); // ...and A is back at its own level
    expect(B2.levels(trees(200), 5000, 6020).get("B")).toBe(0); // the cap grew past the total: released
  });

  it("far scale with cubes: a scaled tree's cubes stay on its own grid, and its crown radius is the voxel extent", () => {
    const b = build(T(40), 1200), lc = levelCells(b.voxels, 1, 1200);
    const mesh = voxelMesh(lc, cubeColors(lc, SEED), mats.voxel, mats.cube);
    const g = new THREE.Group();
    g.position.set(7, 0, -3);
    g.scale.setScalar(2.37);
    g.add(mesh);
    g.updateMatrixWorld(true);
    const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
    for (let i = 0; i < mesh.count; i++) {
      mesh.getMatrixAt(i, m);
      m.premultiply(mesh.matrixWorld).decompose(p, q, s);
      const e = lc.size * 2.37;
      expect(s.x).toBeCloseTo(e, 6);
      for (const [w, o] of [[p.x, 7], [p.y, 0], [p.z, -3]] as const) {
        const u = (w - o) / e - 0.5;
        expect(Math.abs(u - Math.round(u))).toBeLessThan(1e-5);
      }
    }
    // crown radius: every cell corner within crown.r of the place, top within crown.top
    let r = 0;
    for (let i = 0; i < b.voxels.count; i++) {
      const x = b.voxels.coords[i * 3] * CHOSEN.size, z = b.voxels.coords[i * 3 + 2] * CHOSEN.size;
      r = Math.max(r, Math.hypot(Math.max(Math.abs(x), Math.abs(x + CHOSEN.size)), Math.max(Math.abs(z), Math.abs(z + CHOSEN.size))));
    }
    expect(b.crown.r).toBeCloseTo(Math.max(1.5, r), 9);
    mesh.geometry.dispose();
  });

  it("picking: a ray onto a cube maps back to the tree's turn that grew the cell (session and turn), at every level", () => {
    const t = T(40), b = build(t, 1200);
    for (const L of [0, 1, 2]) {
      const lc = levelCells(b.voxels, L, 1200), mesh = voxelMesh(lc, cubeColors(lc, SEED), mats.voxel, mats.cube);
      mesh.updateMatrixWorld(true);
      // straight down onto a column: the first cube hit is the column's top cell
      const tops = new Map<string, number>();
      for (const i of lc.shown) {
        const k = `${lc.coords[i * 3]},${lc.coords[i * 3 + 2]}`, j = tops.get(k);
        if (j === undefined || lc.coords[i * 3 + 1] > lc.coords[j * 3 + 1]) tops.set(k, i);
      }
      let n = 0;
      const kinds = new Set<number>();
      const cols = [...tops.values()], every = Math.max(1, Math.floor(cols.length / 24));
      for (const i of cols.filter((_, k) => k % every === 0)) {
        const x = (lc.coords[i * 3] + 0.5) * lc.size, z = (lc.coords[i * 3 + 2] + 0.5) * lc.size;
        const ray = new THREE.Raycaster(new THREE.Vector3(x, 500, z), new THREE.Vector3(0, -1, 0));
        const hit = ray.intersectObject(mesh, false)[0];
        expect(hit, `L${L} column ${x},${z}`).toBeDefined();
        const cell = lc.shown[hit.instanceId!];
        expect(cell).toBe(i);
        const turn = pickTurn(b.nodes, lc.born[cell], hit.point);
        expect(turn, `L${L}`).not.toBeNull();
        // the turn that claimed the cell: its time is the cell's born, and it belongs to a session of this tree
        expect(turn!.time).toBe(lc.born[cell]);
        expect(["S", "S.sub"]).toContain(turn!.sessionId);
        const src = (turn!.sessionId === "S" ? t.sessions[0] : t.sessions[0].children[0]).turns[turn!.turn];
        expect(Math.fround(src[0])).toBe(turn!.time);
        kinds.add(lc.kind[cell]);
        n++;
      }
      expect(n).toBeGreaterThan(5);
      expect(kinds.size).toBeGreaterThan(0);
      mesh.geometry.dispose();
    }
  });

  it("girth is the wood cell cross-section at the trunk base: non-decreasing and smooth (scale-in), identical across a swap", () => {
    // one session of 110 turns 30 s apart: a build at every turn (the hand-overs), each on screen until the next
    const t = tree("tok:spell:1", [ses("S", 110, 0, 30)]), cuts = Array.from({ length: 110 }, (_, k) => k * 30);
    let a = build(t, cuts[0]).voxels, prev = girthOf(a, cuts[0], CHOSEN.size);
    const g0 = prev;
    for (let j = 0; j + 1 < cuts.length; j++) {
      const c1 = cuts[j], c2 = cuts[j + 1];
      // 60 fps over the first 1.2 s of the build (its new cells scale in there), then at the hand-over
      for (const now of [...Array.from({ length: 73 }, (_, k) => c1 + k / 60), c2]) {
        const g = girthOf(a, now, CHOSEN.size), dt = now - c1 <= 1.2 ? 1 / 60 : 1;
        expect(g, `${now}`).toBeGreaterThanOrEqual(prev - 1e-12);
        expect(g - prev, `${now}`).toBeLessThanOrEqual(GIRTH_RATE * dt + 1e-9);
        prev = g;
      }
      // the new build at the swap instant shows the same girth (its new cells start at 0)
      const b = build(t, c2).voxels;
      expect(Math.abs(girthOf(b, c2, CHOSEN.size) - prev), `swap at ${c2}`).toBeLessThan(1e-9);
      a = b;
    }
    expect(prev, "the trunk base thickened").toBeGreaterThan(g0);
    expect(topOf(a, cuts.at(-1)! + 1, CHOSEN.size)).toBeGreaterThan(1);
  });
});
