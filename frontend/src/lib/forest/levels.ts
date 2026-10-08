/**
 * The renderer's levels of detail (Plan 4, Task 2, ruling R4), built in the geometry worker from a tree's voxel cells
 * (voxel.ts). Pure TypeScript, no three.js.
 *
 * - Level 0 is the voxelizer's own cells; levels 1 and 2 are macro-cells of 2³ and 4³ fine cells at `coord >> L`, on
 *   the tree's own grid. A macro-cell exists when any fine cell in it does (a crown stays solid). Its kind, shade and
 *   born are its earliest-claimed fine cell's: the first in emit order (the claim instant `at`), not always the
 *   earliest born, since a 2-cell wood block's outer cells are born after `at`. Fixed once the macro-cell exists, so a
 *   coarse cube never changes colour as it fills (review Q3).
 * - Shown: a cell is drawn unless its six neighbours at its level are all present and fully grown at the build's cut
 *   (born ≤ cut − SCALE_IN_S): nothing can see it then, and nothing scaling in can uncover it.
 * - `shownLevels` is what the worker sends: per level only the drawn cells (the main thread does no O(cells) work).
 */
import { CHOSEN, type VoxelCells } from "./voxel";

/** New cells scale in from 0 over this long (s of history). */
export const SCALE_IN_S = 0.8;
/** Macro-cells of 2^L fine cells a side, L ≤ LOD_MAX. */
export const LOD_MAX = 2;

/** One level of a tree's cells, in emit order (as the voxelizer emits them: by claim instant). */
export interface LevelCells {
  level: number;
  /** Cell edge (world units, before the far scale). */
  size: number;
  coords: Int32Array;
  kind: Uint8Array;
  shade: Uint8Array;
  born: Float32Array;
  count: number;
  /** Indices of the drawn cells (not enclosed by grown cells), ascending: instance i is cell shown[i]. */
  shown: Uint32Array;
}

/** Coordinates fit in ±2047 cells (voxel.ts). */
const OFF = 2048, SPAN = 4096;
const keyOf = (x: number, y: number, z: number) => ((x + OFF) * SPAN + (y + OFF)) * SPAN + (z + OFF);

/** Level `level` of cells `v`, as drawn from a build at `cut` (e: the fine cell edge). */
export function levelCells(v: Pick<VoxelCells, "coords" | "kind" | "shade" | "born" | "count">, level: number, cut: number, e = CHOSEN.size): LevelCells {
  let coords: Int32Array, kind: Uint8Array, shade: Uint8Array, born: Float32Array, count: number;
  if (level <= 0) ({ coords, kind, shade, born, count } = v);
  else {
    const at = new Set<number>(), cs: number[] = [], bs: number[] = [], ks: number[] = [], ss: number[] = [];
    for (let i = 0; i < v.count; i++) {
      const x = v.coords[i * 3] >> level, y = v.coords[i * 3 + 1] >> level, z = v.coords[i * 3 + 2] >> level, k = keyOf(x, y, z);
      if (at.has(k)) continue; // founded already, by an earlier (or equal) born
      at.add(k);
      cs.push(x, y, z);
      bs.push(v.born[i]);
      ks.push(v.kind[i]);
      ss.push(v.shade[i]);
    }
    count = bs.length;
    coords = new Int32Array(cs);
    born = new Float32Array(bs);
    kind = new Uint8Array(ks);
    shade = new Uint8Array(ss);
  }
  // drawn unless enclosed by six cells fully grown at the cut
  const grownBy = Math.fround(cut - SCALE_IN_S), grown = new Set<number>();
  for (let i = 0; i < count; i++) if (born[i] <= grownBy) grown.add(keyOf(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]));
  const shown: number[] = [];
  for (let i = 0; i < count; i++) {
    const k = keyOf(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
    const enclosed = grown.has(k + SPAN * SPAN) && grown.has(k - SPAN * SPAN) && grown.has(k + SPAN) && grown.has(k - SPAN) &&
      grown.has(k + 1) && grown.has(k - 1);
    if (!enclosed) shown.push(i);
  }
  return { level: Math.max(0, level), size: e * (1 << Math.max(0, level)), coords, kind, shade, born, count, shown: new Uint32Array(shown) };
}

/** Only the drawn cells of a level (shown = 0 … count − 1): what the worker transfers. */
export function compactLevel(lc: LevelCells): LevelCells {
  const n = lc.shown.length, coords = new Int32Array(n * 3), kind = new Uint8Array(n), shade = new Uint8Array(n), born = new Float32Array(n);
  lc.shown.forEach((i, k) => {
    coords.set(lc.coords.subarray(i * 3, i * 3 + 3), k * 3);
    kind[k] = lc.kind[i];
    shade[k] = lc.shade[i];
    born[k] = lc.born[i];
  });
  return { level: lc.level, size: lc.size, coords, kind, shade, born, count: n, shown: Uint32Array.from({ length: n }, (_, k) => k) };
}

/** Every level's drawn cells for a build at `cut`. */
export const shownLevels = (v: VoxelCells, cut: number, e = CHOSEN.size): LevelCells[] =>
  Array.from({ length: LOD_MAX + 1 }, (_, L) => compactLevel(levelCells(v, L, cut, e)));
