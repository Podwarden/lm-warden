/**
 * The ground flora (README §4.1): one-shot question/answer sessions bloom as flower beds in the grass instead of
 * growing a tree; embeddings calls (prompt in, no text out) come up as mushrooms. Ported from forest-real.html (`FL`,
 * `updateFlowers`, `updateMush`) and the style mockup's voxel sprites (`DAI`, `MSH`).
 * - Voxel sprites (the style mockup's `DAI` and `MSH` boxes): a bed is a patch of voxel daisies (green stem, a cross
 *   of white petals, a yellow centre), a longer answer makes a wider, denser bed; it spreads and rises from nothing over
 *   90 s of history after its time. Mushrooms (cream stem, red cap with white spots) come up over 60 s. Sprites turn in
 *   quarter turns only, so they stay on the voxel grid's axes.
 * - Placement: x = X(t) on the timeline, z = a fixed fraction (±0.85) of the spit's half-width at x.
 * - Identity (`FloraRegistry`): the server rounds relative times to 0.1 s against each t0 anchor, so after a re-anchor
 *   the same call can come back up to 0.05 s off. An entry with the same kind and token counts within 0.15 s of a
 *   known one is that one: it keeps its id, its time and its seed (seeded by the id, made once from the absolute time
 *   rounded to the second plus the tokens), so a since-poll or a t0 move never moves, re-colours or duplicates a bed.
 * - Late flora (final review M10): a call whose entry reaches the data after the shown time passed its start (a long
 *   request) grows from its arrival, by the turns' rule (frame.ts): new to the registry, outside the history bands,
 *   before the shown time. Its effective time is kept with its identity.
 * - A bed or mushroom keeps the x and z it was first drawn at for as long as it stays drawn (final review M2): X(t) is
 *   smoothed over ±10 min, so new activity moves it; the flora never jumps on a poll.
 * - Instanced meshes; their boxes count against the scene's cube budget (the trees get what is left).
 */
import * as THREE from "three";
import { effectiveDistance, farScale, floraRoom } from "./farscale";
import { Rng, fnv } from "@/lib/forest/geometry/random";
import type { ForestState } from "@/lib/forest/types";
import { boxes } from "./voxels";

export const FLOWER_GROW_S = 90;
export const MUSH_GROW_S = 60;
/** Most daisies in one bed (mockup `updateFlowers`: 8 + 18 by the answer's length). */
export const PATCH = 26;
/** The style mockup's voxel daisy: [w, h, d, x, y, z, colour] boxes. */
export const DAISY = [
  [0.07, 0.32, 0.07, 0, 0.16, 0, "#2f9e2f"], [0.16, 0.05, 0.07, 0.09, 0.1, 0, "#3cb83a"], [0.44, 0.07, 0.15, 0, 0.34, 0, "#ffffff"],
  [0.15, 0.07, 0.44, 0, 0.34, 0, "#ffffff"], [0.15, 0.1, 0.15, 0, 0.37, 0, "#ffd21f"],
] as const;
/** The style mockup's voxel mushroom: cream stem, red cap with white spots. */
export const MUSHROOM = [
  [0.2, 0.3, 0.2, 0, 0.15, 0, "#fff2d6"], [0.58, 0.16, 0.58, 0, 0.36, 0, "#e8302c"], [0.36, 0.1, 0.36, 0, 0.49, 0, "#e8302c"],
  [0.13, 0.02, 0.13, 0, 0.545, 0, "#ffffff"], [0.11, 0.02, 0.11, 0.2, 0.445, -0.18, "#ffffff"], [0.11, 0.02, 0.11, -0.2, 0.445, 0.18, "#ffffff"],
  [0.02, 0.09, 0.13, 0.295, 0.36, 0.08, "#ffffff"], [0.02, 0.09, 0.13, -0.295, 0.36, -0.1, "#ffffff"], [0.13, 0.09, 0.02, 0.06, 0.36, 0.295, "#ffffff"],
  [0.13, 0.09, 0.02, -0.1, 0.36, -0.295, "#ffffff"],
] as const;
/** Cubes per sprite (they count against the scene's cube budget). */
export const DAISY_CUBES = DAISY.length, MUSH_CUBES = MUSHROOM.length;
/** A mushroom sprite's size against the old lathe mushroom's (`mushrooms().s` keeps its scale). */
const MUSH_K = 0.8;
/** Flora cubes drawn at most; the newest beds win. */
export const FLORA_CAP = 24_000;
/** Beds further behind the view than this are not drawn (as for trees). */
const BEHIND = 320;
/** Flower hues (HSL), drawn per bed (the seed's draw order; the voxel daisies are white). */
const HUES = [0.97, 0.14, 0.62, 0.06, 0.79, 0.0, 0.12];

export type FloraKind = "flower" | "mushroom";

/** A wire flower entry `[tRel, prompt_tokens, completion_tokens]`: no text out means an embeddings call. */
export function classifyFlora(f: readonly [number, number, number]): FloraKind {
  return f[2] === 0 ? "mushroom" : "flower";
}

export interface FloraItem {
  /** Stable across polls and t0 rebases: kind, absolute time (ms), tokens, and an occurrence count. */
  id: string;
  kind: FloraKind;
  /** Relative to the state's t0. */
  time: number;
  x: number;
  /** Fraction of the half-width across the spit, −0.85…0.85 (negative: north). */
  f: number;
  ctx: number;
  gen: number;
  seed: number;
  rot: number;
  hue: number;
  ry: number;
}

/** Same call, re-anchored: absolute times within this (s) with the same tokens are one entry. */
export const FLORA_TOLERANCE = 0.15;

/** Stable identities for the window's flowers and mushrooms across polls and t0 re-anchors (one per scene). */
export class FloraRegistry {
  private known = new Map<string, { abs: number; id: string; time: number }[]>();

  /**
   * Every flower and mushroom of `state` (times relative to `state.t0`) at X(t), in time order. `late`: the shown time
   * and the history bands (`FrameCore.lateFor`); a new entry before the shown time and outside the bands grows from the
   * shown time.
   */
  items(state: Pick<ForestState, "t0" | "flowers">, X: (t: number) => number, late?: { rel: number | null; isHistory: (t: number) => boolean }): FloraItem[] {
    const next = new Map<string, { abs: number; id: string; time: number }[]>(), ids = new Set<string>(), rng = new Rng();
    const out = state.flowers.map((f) => {
      const kind = classifyFlora(f), [t, ctx, gen] = f, key = `${kind}:${ctx}:${gen}`, abs = t + state.t0;
      const prev = this.known.get(key)?.find((p) => !ids.has(p.id) && Math.abs(p.abs - abs) <= FLORA_TOLERANCE);
      let id = prev?.id;
      if (!id) {
        const base = `${key}:${Math.round(abs)}`;
        id = base;
        for (let n = 1; ids.has(id) || this.taken(id); n++) id = `${base}#${n}`;
      }
      const time = prev ? prev.time : late && late.rel !== null && t < late.rel && !late.isHistory(t) ? late.rel : t;
      ids.add(id);
      (next.get(key) ?? next.set(key, []).get(key)!).push({ abs: prev ? prev.abs : abs, id, time });
      rng.SEED(id);
      const fr = (rng.R() * 2 - 1) * 0.85, rot = rng.R() * 6.3, hue = HUES[Math.floor(rng.R() * HUES.length)], ry = rng.R();
      return { id, kind, time, x: X(time), f: fr, ctx, gen, seed: fnv(id), rot, hue, ry };
    });
    this.known = next;
    return out.sort((a, b) => a.time - b.time || (a.id < b.id ? -1 : 1));
  }

  private taken(id: string): boolean {
    for (const l of this.known.values()) if (l.some((p) => p.id === id)) return true;
    return false;
  }
}

/** One-off identities (tests, a single state). */
export const floraItems = (state: Pick<ForestState, "t0" | "flowers">, X: (t: number) => number) => new FloraRegistry().items(state, X);

export interface BedFlower {
  x: number;
  z: number;
  /** Daisy scale across, its height factor, and its turn about the vertical (quarter turns). */
  s0: number;
  h: number;
  a: number;
}

/** The daisies of a bed: mockup `updateFlowers` (voxel) at growth `grow` (0…1) around (x, z0). The count is fixed. */
export function bedFlowers(F: FloraItem, z0: number, grow: number): BedFlower[] {
  const gen = Math.min(700, F.gen), cnt = Math.round(8 + (gen / 700) * (PATCH - 8)), rad = 0.8 + (gen / 700) * 1.4;
  let sd = Math.floor(F.rot * 1e5) + 7;
  const r = () => (sd = (sd * 16807) % 2147483647) / 2147483647;
  const out: BedFlower[] = [];
  for (let k = 0; k < cnt; k++) {
    const a = r() * 6.283, rr = Math.sqrt(r()) * rad * grow;
    const x = F.x + Math.cos(a) * rr, z = z0 + Math.sin(a) * rr * 0.85, s0 = (1.25 + r() * 0.6) * grow;
    out.push({ x, z, s0, h: s0 * (0.8 + r() * 0.5), a: Math.floor(r() * 4) * (Math.PI / 2) });
  }
  return out;
}

export interface Mushroom {
  dx: number;
  dz: number;
  s: number;
  /** Turn about the vertical (radians; drawn in quarter turns). */
  ry: number;
}

/** Mockup `updateMush`: a call's mushrooms (1–3 by prompt size) at history time `rel`; they come up over 60 s. */
export function mushrooms(M: FloraItem, rel: number): Mushroom[] {
  const g = Math.min(1, Math.max(0, (rel - M.time) / MUSH_GROW_S));
  const n = 1 + (M.ctx > 2000 ? 1 : 0) + (M.ctx > 20_000 ? 1 : 0);
  const sc = Math.min(3.5, 2 + 0.5 * Math.log10(1 + M.ctx / 1000));
  const rng = new Rng();
  rng.SEED(M.id, "mush");
  const out: Mushroom[] = [];
  for (let k = 0; k < n; k++) {
    const ry = k ? rng.R() : M.ry, a = rng.R() * 6.283, d = k ? 0.35 + rng.R() * 0.3 : 0;
    out.push({ dx: Math.cos(a) * d, dz: Math.sin(a) * d, s: g * sc * (k ? 0.55 + rng.R() * 0.2 : 1), ry: ry * Math.PI * 2 });
  }
  return out;
}

export type FloraPick = { kind: "flower"; time: number; ctx: number; gen: number } | { kind: "mushroom"; time: number; ctx: number };

/**
 * `toon`: the look's toon material (materials.ts; a plain one by default). `baseAt`: the ground's height under each
 * daisy and mushroom (land.ts terraces, a pure function of the world position: flat 0 by default).
 */
export function createGround(
  scene: THREE.Scene,
  opts: { shadows: boolean; cap?: number; toon?: (o: THREE.MeshToonMaterialParameters) => THREE.Material; baseAt?: (x: number, z: number) => number },
) {
  const baseAt = opts.baseAt ?? (() => 0);
  const geo = { daisy: boxes(DAISY), mush: boxes(MUSHROOM) };
  const mat = opts.toon?.({ vertexColors: true }) ?? new THREE.MeshToonMaterial({ vertexColors: true });
  const root = new THREE.Group();
  scene.add(root);
  const registry = new FloraRegistry();
  let items: FloraItem[] = [];
  let width: (x: number, side: -1 | 1) => number = () => 15;
  /** The drawn beds and mushrooms, their z0 and instance ranges; rebuilt when the selection changes. */
  let shown: { F: FloraItem; z0: number; at: number; n: number; far?: number }[] = [];
  let sig = "", parts = 0, heads: THREE.InstancedMesh | null = null, caps: THREE.InstancedMesh | null = null;
  let headOf: FloraItem[] = [], capOf: FloraItem[] = [];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), s3 = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0), QUARTER = Math.PI / 2;

  const clear = () => {
    for (const im of [heads, caps]) if (im) (im.removeFromParent(), im.dispose());
    heads = caps = null;
  };

  /** Writes one item's instances at history time `rel`. */
  const write = (it: (typeof shown)[number], rel: number) => {
    const { F, z0, at } = it;
    const k0 = it.far ?? 1; // "bigger when far" (farscale.ts): about the bed's or the mushroom's own spot
    if (F.kind === "flower") {
      bedFlowers(F, z0, Math.min(1, Math.max(0, (rel - F.time) / FLOWER_GROW_S))).forEach((b, k) => {
        q.setFromAxisAngle(up, b.a);
        const x = F.x + (b.x - F.x) * k0, z = z0 + (b.z - z0) * k0;
        heads!.setMatrixAt(at + k, m4.compose(v.set(x, baseAt(x, z), z), q, s3.set(Math.max(0.001, b.s0 * k0), Math.max(0.001, b.h * k0), Math.max(0.001, b.s0 * k0))));
      });
    } else {
      mushrooms(F, rel).forEach((m, k) => {
        q.setFromAxisAngle(up, Math.round(m.ry / QUARTER) * QUARTER);
        const mx = F.x + m.dx * k0, mz = z0 + m.dz * k0;
        m4.compose(v.set(mx, baseAt(mx, mz), mz), q, s3.setScalar(Math.max(0.001, m.s * MUSH_K * k0)));
        caps!.setMatrixAt(at + k, m4);
      });
    }
  };

  const flag = (im: THREE.InstancedMesh | null, from?: number, n?: number) => {
    if (!im) return;
    if (from !== undefined && n !== undefined) im.instanceMatrix.addUpdateRange(from * 16, n * 16);
    im.instanceMatrix.needsUpdate = true;
  };

  const rebuild = (rel: number) => {
    clear();
    let nf = 0, nm = 0;
    headOf = [];
    capOf = [];
    shown = shown.map((s) => {
      const n = s.F.kind === "flower" ? bedFlowers(s.F, 0, 0).length : mushrooms(s.F, rel).length;
      const at = s.F.kind === "flower" ? nf : nm;
      for (let k = 0; k < n; k++) (s.F.kind === "flower" ? headOf : capOf).push(s.F);
      if (s.F.kind === "flower") nf += n;
      else nm += n;
      return { ...s, at, n };
    });
    const mk = (g: THREE.BufferGeometry, n: number) => {
      const im = new THREE.InstancedMesh(g, mat, n);
      im.castShadow = opts.shadows;
      im.frustumCulled = false;
      root.add(im);
      return im;
    };
    if (nf) heads = mk(geo.daisy, nf);
    if (nm) caps = mk(geo.mush, nm);
    for (const s of shown) write(s, rel);
    flag(heads);
    flag(caps);
    for (const im of [heads, caps]) im?.computeBoundingSphere(); // picking
    parts = nf * DAISY_CUBES + nm * MUSH_CUBES;
  };

  return {
    setData(state: ForestState, X: (t: number) => number, w: (x: number, side: -1 | 1, cut?: number) => number, late?: { rel: number | null; isHistory: (t: number) => boolean }) {
      items = registry.items(state, X, late);
      width = (x, side) => w(x, side);
      sig = ""; // new or late entries: re-select at the next tick (drawn ones keep their places)
    },

    /** Chooses what to draw (born by `rel`, not > 320 behind the view, newest first within the cap); true if changed. */
    tick(rel: number, viewX: number): boolean {
      const cap = opts.cap ?? FLORA_CAP, sel: FloraItem[] = [];
      let used = 0;
      for (let i = items.length - 1; i >= 0; i--) {
        const F = items[i];
        if (F.time > rel || F.x < viewX - BEHIND) continue;
        const cost = F.kind === "flower" ? DAISY_CUBES * Math.round(8 + (Math.min(700, F.gen) / 700) * (PATCH - 8)) : MUSH_CUBES * 3;
        if (used + cost > cap) break;
        used += cost;
        sel.push(F);
      }
      sel.reverse();
      const s = sel.length ? `${sel.length}:${sel[0].id}:${sel[sel.length - 1].id}` : "";
      if (s === sig) return false;
      sig = s;
      const was = new Map(shown.map((x) => [x.F.id, x]));
      shown = sel.map((F) => {
        const o = was.get(F.id); // drawn already: it stays where it stands
        return o ? { F: { ...F, x: o.F.x }, z0: o.z0, at: 0, n: 0 } : { F, z0: F.f * width(F.x, F.f < 0 ? -1 : 1), at: 0, n: 0 };
      });
      rebuild(rel);
      return true;
    },

    /**
     * Grows the beds and mushrooms still coming up at `rel` (in time order, so only the tail is visited; finished
     * mushrooms between growing beds are skipped, not a stop). `bounds`: also refresh the picking bounds. True if any.
     */
    /**
     * "Bigger when far" (farscale.ts): each bed and mushroom drawn larger about its own spot for a camera at `cam` over a
     * canvas `viewH` px high; rewritten only when its scale moved by more than 2 %. True when anything was rewritten.
     */
    setView(cam: { x: number; y: number; z: number }, viewH: number, rel: number): boolean {
      let any = false;
      for (const s of shown) {
        // a bed's radius (bedFlowers: 0.8 … 2.2 by its answer's length), a mushroom cluster's ~0.7; capped to the grass
        const radius = s.F.kind === "flower" ? 0.8 + (Math.min(700, s.F.gen) / 700) * 1.4 : 0.7;
        const room = floraRoom(width(s.F.x, s.z0 < 0 ? -1 : 1), s.z0, radius);
        const f = Math.min(room, farScale(effectiveDistance(Math.hypot(cam.x - s.F.x, cam.y, cam.z - s.z0), viewH)));
        if (Math.abs(f - (s.far ?? 1)) <= 0.02 * (s.far ?? 1) && !(f === 1 && (s.far ?? 1) !== 1)) continue;
        s.far = f;
        write(s, rel);
        flag(s.F.kind === "flower" ? heads : caps, s.at, s.n);
        any = true;
      }
      if (any) for (const im of [heads, caps]) im?.computeBoundingSphere();
      return any;
    },

    grow(rel: number, bounds = false): boolean {
      let any = false;
      for (let i = shown.length - 1; i >= 0; i--) {
        const s = shown[i], T = s.F.kind === "flower" ? FLOWER_GROW_S : MUSH_GROW_S;
        if (rel - s.F.time > FLOWER_GROW_S + 1) break; // older than the longest growth: everything before is done
        if (rel - s.F.time > T + 1) continue;
        write(s, rel);
        flag(s.F.kind === "flower" ? heads : caps, s.at, s.n);
        any = true;
      }
      if (any && bounds) for (const im of [heads, caps]) im?.computeBoundingSphere();
      return any;
    },

    /** Cubes drawn (the sprites' boxes; counted against the scene's cube budget). */
    count: () => parts,

    /** Where each drawn bed and mushroom stands (tests). */
    placed: () => shown.map((x) => ({ id: x.F.id, x: x.F.x, z: x.z0 })),

    pick(ray: THREE.Raycaster): { result: FloraPick; distance: number } | null {
      const hit = ray.intersectObjects([heads, caps].filter((m): m is THREE.InstancedMesh => m !== null), false)[0];
      if (!hit || hit.instanceId === undefined) return null;
      if (hit.object === heads) {
        const F = headOf[hit.instanceId];
        return F ? { result: { kind: "flower", time: F.time, ctx: F.ctx, gen: F.gen }, distance: hit.distance } : null;
      }
      const M = capOf[hit.instanceId];
      return M ? { result: { kind: "mushroom", time: M.time, ctx: M.ctx }, distance: hit.distance } : null;
    },

    dispose() {
      clear();
      Object.values(geo).forEach((g) => g.dispose());
      mat.dispose();
      root.removeFromParent();
    },
  };
}

export type Ground = ReturnType<typeof createGround>;
