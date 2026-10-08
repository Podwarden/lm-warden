/**
 * The voxel renderer's cells (Plan 4, Task 2): a tree's `VoxelCells` (lib/forest/voxel.ts) drawn as one instanced mesh
 * of cubes, at a level of detail chosen by the camera's distance.
 *
 * - **Levels** (R4: distance LOD, not thinning). Level 0 is the voxelizer's own 0.25 u cells; levels 1 and 2 are
 *   macro-cells of 2³ and 4³ fine cells. A macro-cell exists when any fine cell in it does (so a crown stays solid, with
 *   no holes); its kind, shade and born are those of its earliest-claimed fine cell (the first in emit order;
 *   lib/forest/levels.ts). Coordinates are floor(fine / 2^L): a macro-cell sits on the tree's own grid.
 * - **Shown.** A cell is drawn unless its six neighbours (at the same level) are all present and fully grown at the
 *   build's cut (born ≤ cut − SCALE_IN_S): nothing can see it then, and nothing scaling in can uncover it.
 * - **Append-only swaps.** Cells only append (voxel.ts), so a rebuild shows the same cubes plus new ones, which scale in
 *   from 0 over SCALE_IN_S (a smoothstep on `born` in the shader). There is no morph.
 * - **Growth tint.** A cell glows toward the mockup's cyan for TINT_S after its born, fading out.
 * - **Cross-fade.** A level switch draws both levels for FADE_S with complementary screen-space dithers (`fadeKeep`):
 *   every pixel comes from exactly one of the two, so nothing doubles, holes or needs sorting.
 * - **Budget** (`fitBudget`): the farthest trees go one level coarser first until the cubes fit.
 * - **Picking** (`pickTurn`): a cube's born is the time of the turn that claimed it; the turn is the tree's node at that
 *   time (the nearest to the hit when several turns share it).
 */
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { NodeRef } from "@/lib/forest/geometry";
import { type LevelCells, LOD_MAX, SCALE_IN_S } from "@/lib/forest/levels";
import { CHOSEN, DARK, KIND, cellJitter } from "@/lib/forest/voxel";

export { type LevelCells, LOD_MAX, SCALE_IN_S, compactLevel, levelCells, shownLevels } from "@/lib/forest/levels";

/** Growth tint: the cyan glow of a new cell lasts this long, fading out (s of history). */
export const TINT_S = 12;
/** A level switch cross-fades over this long (s of wall time). */
export const FADE_S = 0.3;
/**
 * Normalised camera distance (farscale.ts `effectiveDistance`) from which level 1, then level 2, is drawn.
 * - A 0.25 u cell at distance d in the full window is ≈ 346 / d px: level 0 is drawn down to ≈ 3 px a cube (the 1 h and
 *   6 h views, d ≤ 90), level 1 beyond (the 24 h view, d ≈ 150, and the card's 24 h, ≈ 165 normalised).
 * - The 7 d fit of a realistic week (~155 trees) stands past the level-2 threshold: 7 d draws level 2, measured for
 *   every tree in the full window and the card alike (the e2e perf matrix; about 57 cubes a tree). Level 2 is also the
 *   budget's last resort (`fitBudget`), which is why a far view builds at most (cubes left) / 70 trees (farscale.ts).
 */
export const LOD_D: readonly [number, number] = [100, 1000];
/** Hysteresis: a switch happens LOD_HYST past a threshold, and back LOD_HYST before it. */
export const LOD_HYST = 0.1;
/** Saturation of the chosen look (applied in the cube and flora shaders). */
export const SATURATION = 1.4;
/** Cel shading steps of the chosen look. */
export const CEL_STEPS = 4;

const smooth = (x: number) => x * x * (3 - 2 * x);

/** A cell's scale `age` s after its born: a smoothstep from 0 to 1 over SCALE_IN_S (the shader's rule). */
export const scaleIn = (age: number) => (age <= 0 ? 0 : age >= SCALE_IN_S ? 1 : smooth(age / SCALE_IN_S));

/** The growth tint's weight `age` s after born: 1 at birth, fading linearly to 0 at TINT_S (the shader's rule). */
export const tintAt = (age: number) => (age < 0 ? 0 : Math.max(0, 1 - age / TINT_S));

/** The cross-fade's dither: the level fading in keeps the pixels whose hash is below the fade, the other the rest. */
export const fadeKeep = (h: number, fade: number, fadingIn: boolean) => (fadingIn ? h < fade : h >= fade);

/** The level for normalised distance `d`, moving off `prev` only LOD_HYST past a threshold. */
export function levelFor(d: number, prev?: number): number {
  if (prev === undefined) {
    let L = 0;
    while (L < LOD_MAX && d >= LOD_D[L]) L++;
    return L;
  }
  let L = Math.max(0, Math.min(LOD_MAX, prev));
  while (L < LOD_MAX && d >= LOD_D[L] * (1 + LOD_HYST)) L++;
  while (L > 0 && d < LOD_D[L - 1] * (1 - LOD_HYST)) L--;
  return L;
}

/**
 * Fits the trees' cubes into `cap`: from each tree's own level (a floor), the farthest trees go one level coarser
 * first, pass after pass, until the total fits or every tree is at LOD_MAX. Nothing is dropped: a coarser level covers
 * every cell of the finer one.
 */
export function fitBudget(trees: readonly { id: string; d: number; level: number; count: (L: number) => number }[], cap: number): Map<string, number> {
  const lv = new Map(trees.map((t) => [t.id, t.level]));
  let total = trees.reduce((a, t) => a + t.count(t.level), 0);
  const far = [...trees].sort((a, b) => b.d - a.d || (a.id < b.id ? -1 : 1));
  for (let changed = true; total > cap && changed; ) {
    changed = false;
    for (const t of far) {
      if (total <= cap) break;
      const L = lv.get(t.id)!;
      if (L >= LOD_MAX) continue;
      total += t.count(L + 1) - t.count(L);
      lv.set(t.id, L + 1);
      changed = true;
    }
  }
  return lv;
}

/** The budget re-ranks the trees only when it moved by more than this share, or every BUDGET_RERANK_MS. */
export const BUDGET_MOVE = 0.1;
export const BUDGET_RERANK_MS = 5000;
/** At a re-rank, a tree the budget held coarser counts this much farther (it stays first in line: no swaps). */
const STICKY = 1.1;

/**
 * The budget's level choice with per-tree hysteresis (review Q1). Over the cap, `fitBudget` coarsens the farthest
 * trees first; the result is kept (each tree at least at its own distance level) until the budget moves by more than
 * BUDGET_MOVE (the natural total or the cap), the kept choice no longer fits, or BUDGET_RERANK_MS pass. At a re-rank a
 * tree already held coarser counts STICKY farther, so two trees at about the same distance never trade levels. Once
 * held, it lets go only when the natural total fits `release` of the cap.
 */
export class BudgetLevels {
  private held: Map<string, number> | null = null;
  private at = { natural: 0, cap: 0, t: -Infinity };

  constructor(private release = 0.92) {}

  levels(trees: readonly { id: string; d: number; level: number; count: (L: number) => number }[], cap: number, nowMs: number): Map<string, number> {
    const natural = trees.reduce((a, t) => a + t.count(t.level), 0);
    const out = new Map(trees.map((t) => [t.id, t.level]));
    if (this.held ? natural <= cap * this.release : natural <= cap) return void (this.held = null), out;
    const lv = (t: (typeof trees)[number]) => Math.max(t.level, this.held?.get(t.id) ?? t.level);
    const moved = (a: number, b: number) => Math.abs(a - b) > BUDGET_MOVE * Math.max(1, b);
    let rerank = !this.held || moved(natural, this.at.natural) || moved(cap, this.at.cap) || nowMs - this.at.t >= BUDGET_RERANK_MS;
    if (!rerank && trees.reduce((a, t) => a + t.count(lv(t)), 0) > cap) rerank = true;
    if (rerank) {
      const prev = this.held;
      this.held = fitBudget(trees.map((t) => ({ ...t, d: (prev?.get(t.id) ?? t.level) > t.level ? t.d * STICKY : t.d })), cap * this.release);
      this.at = { natural, cap, t: nowMs };
    }
    for (const t of trees) out.set(t.id, lv(t));
    return out;
  }

  /** Some tree is held coarser than its distance level. */
  get active(): boolean {
    return this.held !== null;
  }
}

// ---------------- colours (the mockup's voxel crowns) ----------------

/** Mockup `CROWNS[ti][0]`: a crown's base green, picked per tree. */
const CROWNS = ["#3cbf3a", "#2fb04a", "#56c232", "#37b83f"];
/** Wood: the mockup's trunk brown a step darker, so it reads brown (not orange) after the whole-frame saturation. */
const WOOD = new THREE.Color("#8f5a2c"), FAILED = new THREE.Color("#b8902f"), BLOSSOM = new THREE.Color("#ff7fbf");
const WOOD_SALT = 0x5bd1e995;

/** Mockup `shadesFor(ti)` with the chosen shade count, and its dark cell colour (the last shade × 0.7). */
function palette(seed: string): { shades: THREE.Color[]; dark: THREE.Color } {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (Math.imul(h, 31) + seed.charCodeAt(i)) >>> 0;
  const c0 = new THREE.Color(CROWNS[h % CROWNS.length]), hsl = { h: 0, s: 0, l: 0 };
  c0.getHSL(hsl);
  const n = Math.max(1, CHOSEN.shades), shades: THREE.Color[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0.35 : i / (n - 1);
    shades.push(new THREE.Color().setHSL(hsl.h + (t - 0.5) * 0.025, Math.min(1, hsl.s * (1.04 - 0.08 * t)), Math.min(0.8, hsl.l * (1.22 - 0.48 * t))));
  }
  return { shades, dark: shades[shades.length - 1].clone().multiplyScalar(0.7) };
}

/** Linear RGB per drawn cell (3 each): leaves by shade and dark flag, wood brown with a jitter, failed olive, blossoms pink. */
export function cubeColors(lc: LevelCells, seed: string): Float32Array {
  const { shades, dark } = palette(seed), out = new Float32Array(lc.shown.length * 3), c = new THREE.Color();
  lc.shown.forEach((i, k) => {
    const kd = lc.kind[i], sh = lc.shade[i];
    if (kd === KIND.wood) {
      const j = cellJitter(seed, lc.coords[i * 3], lc.coords[i * 3 + 1], lc.coords[i * 3 + 2], WOOD_SALT + lc.level);
      c.copy(WOOD).multiplyScalar(0.92 + 0.16 * j);
    } else if (kd === KIND.blossom) c.copy(BLOSSOM);
    else if (kd === KIND.failed) c.copy(FAILED);
    else c.copy(sh & DARK ? dark : shades[(sh & ~DARK) % shades.length]);
    out.set([c.r, c.g, c.b], k * 3);
  });
  return out;
}

// ---------------- meshes ----------------

/** Mockup `shadeBox`: flat face shading baked into vertex colours (top light, sides darker, bottom dark). */
function shadeBox(g0: THREE.BufferGeometry, col: THREE.Color): THREE.BufferGeometry {
  const g = g0.index ? g0.toNonIndexed() : g0, n = g.attributes.normal, a: number[] = [];
  for (let i = 0; i < n.count; i++) {
    const ny = n.getY(i), nx = Math.abs(n.getX(i)), k = ny > 0.5 ? 1.16 : ny < -0.5 ? 0.55 : nx > 0.5 ? 0.78 : 0.9;
    a.push(col.r * k, col.g * k, col.b * k);
  }
  g.setAttribute("color", new THREE.Float32BufferAttribute(a, 3));
  if (g.attributes.uv) g.deleteAttribute("uv");
  return g;
}

/** Mockup `boxes(list)`: boxes [w, h, d, x, y, z, colour] merged into one shaded geometry (a voxel sprite). */
export function boxes(list: readonly (readonly [number, number, number, number, number, number, string])[]): THREE.BufferGeometry {
  const parts = list.map(([w, h, d, x, y, z, c]) => shadeBox(new THREE.BoxGeometry(w, h, d), new THREE.Color(c)).translate(x, y, z));
  const g = mergeGeometries(parts, false);
  parts.forEach((p) => p.dispose());
  if (!g) throw new Error("voxel sprite merge failed");
  return g;
}

/** The unit cube every cell is drawn with (white, face-shaded). */
export const cubeGeometry = () => boxes([[1, 1, 1, 0, 0, 0, "#ffffff"]]);

/** A level's drawn cells as one instanced mesh of `cube`: per instance its offset and edge (the instance matrix), its
 * colour and `iborn`. */
export function voxelMesh(lc: LevelCells, colors: Float32Array, mat: THREE.Material, cube: THREE.BufferGeometry): THREE.InstancedMesh {
  const n = lc.shown.length, s = lc.size, M = new Float32Array(n * 16), born = new Float32Array(n);
  lc.shown.forEach((i, k) => {
    M.set([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, (lc.coords[i * 3] + 0.5) * s, (lc.coords[i * 3 + 1] + 0.5) * s, (lc.coords[i * 3 + 2] + 0.5) * s, 1], k * 16);
    born[k] = lc.born[i];
  });
  const g = cube.clone();
  g.setAttribute("iborn", new THREE.InstancedBufferAttribute(born, 1));
  const m = new THREE.InstancedMesh(g, mat, n);
  m.instanceMatrix = new THREE.InstancedBufferAttribute(M, 16);
  m.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
  m.computeBoundingSphere();
  return m;
}

// ---------------- picking, girth, height ----------------

/** The turn whose time is `born` (the nearest in time), nearest to `p` (the tree's own frame) among equals. */
export function pickTurn(nodes: readonly NodeRef[], born: number, p: { x: number; y: number; z: number }): NodeRef | null {
  let best: NodeRef | null = null, bt = Infinity, bd = Infinity;
  for (const n of nodes) {
    const dt = Math.abs(n.time - born), d = (n.p[0] - p.x) ** 2 + (n.p[1] - p.y) ** 2 + (n.p[2] - p.z) ** 2;
    if (dt < bt - 1e-9 || (Math.abs(dt - bt) <= 1e-9 && d < bd)) (best = n), (bt = dt), (bd = d);
  }
  return best;
}

/** How fast `girthOf` moves at most (u/s): one cell edge per scale-in. */
export const GIRTH_RATE = CHOSEN.size / SCALE_IN_S;

type Cells = Pick<LevelCells, "coords" | "kind" | "born" | "count">;

/**
 * The trunk base's girth on screen at `now` (S1): how far the bottom layer's wood reaches out from the trunk axis, as a
 * front that crosses one ring of cells (Chebyshev, the blocks' own metric) per scale-in. Ring k starts when its first
 * cell is born, but never before ring k − 1 has finished, and the front crosses it linearly over SCALE_IN_S (the
 * drawn cubes follow a smoothstep of the same span). Non-decreasing, continuous, at most one edge per SCALE_IN_S; it
 * stops at the first missing ring (wood not joined to the trunk, a low limb, is ignored).
 */
export function girthOf(v: Cells, now: number, e = CHOSEN.size): number {
  const inner = (i: number) => (i >= 0 ? i : -i - 1);
  const first: number[] = [];
  for (let i = 0; i < v.count; i++) {
    if (v.coords[i * 3 + 1] !== 0 || v.kind[i] !== KIND.wood) continue;
    const k = Math.max(inner(v.coords[i * 3]), inner(v.coords[i * 3 + 2]));
    first[k] = Math.min(first[k] ?? Infinity, v.born[i]);
  }
  let ext = 0, start = -Infinity;
  for (let k = 0; k < first.length && first[k] !== undefined; k++) {
    start = Math.max(first[k], start + SCALE_IN_S);
    if (now <= start) break;
    const u = Math.min(1, (now - start) / SCALE_IN_S);
    ext = (k + u) * e;
    if (u < 1) break;
  }
  return ext;
}

/** The top of the tree's fully grown cells at `now` (units above its base), 0 if none. */
export function topOf(v: Cells, now: number, e = CHOSEN.size): number {
  let top = 0;
  for (let i = 0; i < v.count; i++) if (now - v.born[i] >= SCALE_IN_S) top = Math.max(top, (v.coords[i * 3 + 1] + 1) * e);
  return top;
}
