/**
 * What the voxelizer (`lib/forest/voxel.ts`) reads from one tree build: foliage points, blossom tips and the wood
 * centrelines with a girth *schedule* per centreline sample.
 *
 * Everything here is prefix-stable by construction, which the cut-dependent bark mesh is not:
 * - Points are the limbs' joints and seeded bow points, the twig foliage and the twig tips — none of them move when
 *   a turn arrives or the cut advances (palm crowns are placed per turn: see `shiftAt`).
 * - A foliage point's radius is its leaf size WITHOUT `LEAFK` (which grows with the tree's work), so the cells a turn
 *   claims never change after it.
 * - Girth is a step schedule `(time, radius)`: the radius a sample has once every turn up to `time` has landed, from
 *   the pipe-model load counted without `RAMP` and kept as a running maximum. Its value at a past time depends only on
 *   turns up to that time, so a later cut only appends entries, and the voxelizer can say exactly when a wood cell
 *   was first covered (its `born`).
 */
import { V3 } from "./tube";

/** A load event: a turn (or a limb's base) adds `w` to every centreline point it lies beyond, from time `t`. */
export type Ev = [t: number, w: number];

/** One wood centreline sample: position, the time it exists from, and its girth schedule. */
export interface WoodSample {
  p: V3;
  born: number;
  /** Ascending times and the (non-decreasing) radius from each one on. */
  st: number[];
  sr: number[];
}

/** The voxel inputs gathered during one build (see `VoxelInput` for the packed form). */
export interface VoxRec {
  /**
   * While set, the offset added to a point recorded for a turn at time `t`. Palm crowns ride the trunk top, which
   * rises with the cut; the voxels place each turn's wood, foliage and tip at the top as it stood when that turn
   * landed, so its cells never move. Old turns stay where they formed (lower frond tiers: a layered palm) and the
   * newest sit at the current top.
   */
  shiftAt: ((t: number) => V3) | null;
  /** x, y, z, radius, born, failed (0/1). */
  foliage: number[];
  /** x, y, z, born. */
  tips: number[];
  lines: WoodSample[][];
  /** The trunk's centreline (an index into `lines`), −1 for a tree without one. */
  trunk: number;
}

export const newVoxRec = (): VoxRec => ({ shiftAt: null, foliage: [], tips: [], lines: [], trunk: -1 });

/** Smallest girth increase kept in a schedule (world units): finer steps never change a cell at any grid size used. */
export const SCHED_EPS = 0.004;

/**
 * The girth schedule from load events: `radius(load, t)` at every event time from `from` on, kept as a running
 * maximum and only where it grew by at least `SCHED_EPS` (the decision only looks back, so it is prefix-stable).
 */
export function schedule(ev: Ev[], from: number, radius: (load: number, t: number) => number): { st: number[]; sr: number[] } {
  const e = [...ev].sort((a, b) => a[0] - b[0]);
  const st: number[] = [], sr: number[] = [];
  let load = 0, best = 0;
  for (let i = 0; i < e.length; i++) {
    load += e[i][1];
    if (i + 1 < e.length && e[i + 1][0] === e[i][0]) continue; // one value per instant
    const r = Math.max(best, radius(load, e[i][0]));
    if (!sr.length || r >= sr[sr.length - 1] + SCHED_EPS) {
      st.push(Math.max(from, e[i][0]));
      sr.push(r);
    }
    best = r;
  }
  return { st, sr };
}

/** The step value of a schedule at time `t` (0 before its first entry). */
export function stepAt(st: ArrayLike<number>, sr: ArrayLike<number>, t: number, o = 0, n = st.length): number {
  let v = 0;
  for (let k = o; k < o + n && st[k] <= t; k++) v = sr[k];
  return v;
}

/** The schedule a bow point between two joints carries: the joints' radii mixed `f` of the way, at every change. */
export function lerpSched(a: { st: number[]; sr: number[] }, b: { st: number[]; sr: number[] }, f: number): { st: number[]; sr: number[] } {
  const ts = [...new Set([...a.st, ...b.st])].sort((x, y) => x - y);
  const st: number[] = [], sr: number[] = [];
  for (const t of ts) {
    const r = stepAt(a.st, a.sr, t) * (1 - f) + stepAt(b.st, b.sr, t) * f;
    if (!sr.length || r > sr[sr.length - 1]) {
      st.push(t);
      sr.push(r);
    }
  }
  return { st, sr };
}

/** A point recorded for a turn at time `t`, with the current shift applied (a copy). */
export const shifted = (rec: VoxRec, p: V3, t: number): V3 => (rec.shiftAt ? p.clone().add(rec.shiftAt(t)) : p.clone());

/** Packed, transferable voxel inputs (`TreeBuffers.vox`). */
export interface VoxelInput {
  /** The tree's place (x, z): the origin of its local grid. */
  origin: Float32Array;
  /** 6 per point: x, y, z, radius, born, failed (0/1). */
  foliage: Float32Array;
  /** 4 per answered turn: x, y, z of its twig tip, born. */
  tips: Float32Array;
  /** 6 per centreline sample: x, y, z, born, first schedule entry, entry count. */
  wood: Float32Array;
  /** First sample of each centreline, plus a final end index (length = lines + 1). */
  lines: Uint32Array;
  /** 2 per schedule entry: time, radius. */
  sched: Float32Array;
  /** The trunk's centreline index in `lines`, −1 if none. */
  trunk: number;
}

export function packVox(rec: VoxRec, origin: { x: number; z: number }): VoxelInput {
  const lines = new Uint32Array(rec.lines.length + 1);
  let ns = 0, ne = 0;
  rec.lines.forEach((l, i) => {
    lines[i] = ns;
    ns += l.length;
    for (const s of l) ne += s.st.length;
  });
  lines[rec.lines.length] = ns;
  const wood = new Float32Array(ns * 6), sched = new Float32Array(ne * 2);
  let i = 0, j = 0;
  for (const l of rec.lines)
    for (const s of l) {
      wood.set([s.p.x, s.p.y, s.p.z, s.born, j, s.st.length], i * 6);
      for (let k = 0; k < s.st.length; k++, j++) sched.set([s.st[k], s.sr[k]], j * 2);
      i++;
    }
  return { origin: new Float32Array([origin.x, origin.z]), foliage: new Float32Array(rec.foliage), tips: new Float32Array(rec.tips), wood, lines, sched, trunk: rec.trunk };
}
