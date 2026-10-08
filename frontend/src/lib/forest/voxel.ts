/**
 * The voxelizer: one tree build → a set of grid cells (the voxel look, Plan 4). Pure TypeScript with no three.js, so
 * it runs in the geometry Web Worker.
 *
 * **Prefix-stable and append-only — under these conditions.** Given the same tree id, traits, knobs, norm and place
 * scale, a cell present at cut c is present with the same coordinate, kind, shade, `born` and `buried` flag at every
 * later cut, and the output arrays at c are a prefix of the arrays at any later cut, provided that:
 * 1. A session's turns are only appended: a turn already held keeps its index and its time. The client guarantees
 *    this at intake (engine/frame.ts `shiftSession`): held turns are append-only across range switches, and a server
 *    LOD block (past 2000 turns) that closes over singles the client already holds is ignored for those singles.
 * 2. No turn lands at or before an earlier cut: the late-turn rule (`LateTurns`, engine/frame.ts) grows a late turn
 *    from its arrival, effective born = max(t, arrival).
 * 3. Turn times are float32 values (`Math.fround` in `adaptTurn`), so the cut filter and the recorded `born` agree:
 *    two turns that share a float32 time are always in or out of a build together.
 * 4. A subagent forks from a parent turn no later than its own first turn (its fork joint exists when it starts), as
 *    the server's `at` (the spawning turn) guarantees.
 * How, given those:
 * - The inputs (`TreeBuffers.vox`) never move and never change for a turn already landed (see `geometry/voxin.ts`).
 * - A cell's `born` is the minimum `born` of everything that covers it: the first crown ellipsoid that holds it, or a
 *   wood sample from the moment the sample's girth schedule first reaches the cell. Anything that starts covering it later
 *   is later than every present cell, so the minimum never changes.
 * - Claims resolve by their claim instant: the earliest claim wins, then the earlier born, then the stronger kind
 *   (wood, failed, leaf). A later claim never changes a cell (a leaf cell the wood later grows into stays a leaf).
 * - Cells are kept instant by instant in `born` order (then by density priority), so a later cut only appends.
 * - Shade, dark flag, density thinning and cap priority are hashes of (tree id, cell coordinate), never of input or
 *   iteration order.
 *
 * **Crown** (R5, the mockup's `crownBuild`). One solid, round crown per tree: an ellipsoid fitted to the kept foliage
 * points (mean and spread of the points thinned by `density` with a hash of the point's own cell; failed tools always
 * kept), its rim roughened, cut off below a floor so a blocky trunk shows under it. At every instant the ellipsoid of
 * the points born by then is filled, so the crown is the union of the per-instant ellipsoids and a cell's born is the
 * first instant whose ellipsoid holds it.
 *
 * **Wood.** Each limb and the trunk are stepped along their centrelines in half-cell steps (≤ 0.5 u); thick wood fills
 * 2-cell blocks (the trunk at least 2 × 2), growing from the inside out. Girth growth adds cells outward and never
 * re-snaps one. Wood that would poke out of the upper crown is clipped.
 *
 * **Single cubes.** A blossom (an answered turn) or a failed tool is one cube projected onto its turn's crown
 * ellipsoid (onto the upper surface when that lands under the floor), climbing to the first free cell; never stacked
 * on another single cube, and leaf wins an occupied cell.
 *
 * **Buried.** Every kept cell is output (the full solid, for Task 2's macro-cell LOD); `buried` flags a cell whose six
 * neighbours were all kept by the end of its own instant (the instant's cells after the cap cut, before its blossoms).
 * The kept set only grows, so a buried cell stays enclosed for good and need not be drawn at full detail; a cell
 * dropped by the cap is never counted as solid, so the cap leaves no hole into a hollow interior.
 *
 * **Cap.** At most `VOXEL_CAP` kept cells per tree (`opts.cap` to override). When an instant's new cells would pass
 * it, that instant contributes only up to the cap, lowest priority hash first; cells already kept are never removed,
 * and later instants add nothing.
 */
import type { VoxelInput } from "./geometry/voxin";

export interface VoxelOpts {
  /** Cell edge, world units. */
  size: number;
  /** Share of foliage points kept (by a hash of the point's cell). */
  density: number;
  /** Colour shades per crown. */
  shades: number;
  /** Share of cells flagged dark. */
  dark: number;
  blossoms: boolean;
  /** Per-tree cell cap; `VOXEL_CAP` when left out. */
  cap?: number;
}

export interface VoxelCells {
  /** ix, iy, iz per cell, in the tree's local grid (origin at the tree's place on the ground; cell = floor(p / size)). */
  coords: Int32Array;
  /** 0 leaf, 1 blossom, 2 failed, 3 wood. */
  kind: Uint8Array;
  /** 0..shades−1, plus a dark flag in the high bit (`DARK`). */
  shade: Uint8Array;
  /** The turn time that claimed the cell (history seconds). */
  born: Float32Array;
  /** 1: enclosed by kept cells since its own instant (see "Buried"); not drawn at full detail. */
  buried: Uint8Array;
  count: number;
}

/**
 * The look the user picked in the style mockup (chosen-style.json). Its crown cubes are 0.5 · size(0.5) = 0.25 u, so
 * the cell edge here is 0.25 u.
 */
export const CHOSEN: VoxelOpts = { size: 0.25, density: 0.35, shades: 2, dark: 0.34, blossoms: true };

export const KIND = { leaf: 0, blossom: 1, failed: 2, wood: 3 } as const;
export const DARK = 0x80;

/**
 * Per-tree cap on kept cells (drawn + buried). The crown is solid, so a big tree keeps tens of thousands of cells; the
 * worker sends only the drawn ones (levels.ts), and the cube budget is the renderer's distance LOD (macro-cells).
 */
export const VOXEL_CAP = 60_000;

/** Blossom climb limit (cells). */
const CLIMB = 8;
const SALT = { density: 0x9e3779b9, priority: 0x85ebca6b, shade: 0xc2b2ae35, dark: 0x27d4eb2f, rim: 0x165667b1, bloom: 0xd3a2646c } as const;

/** 32-bit avalanche mix (lowbias32). */
function mix(h: number): number {
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
}

/** FNV-1a of the tree id. */
function fnv(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

/** A hash of (tree, cell, salt) in [0, 1). */
const cellHash = (seed: number, ix: number, iy: number, iz: number, salt: number) =>
  mix(mix(mix(mix(seed ^ salt) ^ ix) ^ iy) ^ iz) / 4294967296;

/** A cell's shade and dark flag: a hash of (tree, cell) only. `level` > 0: a macro-cell of the renderer's LOD. */
function shadeHash(sd: number, x: number, y: number, z: number, opts: Pick<VoxelOpts, "shades" | "dark">, level = 0): number {
  const sh = Math.max(1, Math.min(127, Math.round(opts.shades))), lv = Math.imul(level, 0x632be5ab);
  return Math.min(sh - 1, Math.floor(cellHash(sd, x, y, z, SALT.shade ^ lv) * sh)) | (cellHash(sd, x, y, z, SALT.dark ^ lv) < opts.dark ? DARK : 0);
}

/** A hash of (tree, cell, salt) in [0, 1), for the renderer's colour jitter. */
export const cellJitter = (seed: string, x: number, y: number, z: number, salt: number) => cellHash(fnv(seed), x, y, z, salt);

/** Grid coordinates fit in ±2047 cells (±512 u at 0.25 u). */
const OFF = 2048, SPAN = 4096;
const keyOf = (ix: number, iy: number, iz: number) => ((ix + OFF) * SPAN + (iy + OFF)) * SPAN + (iz + OFF);
const inGrid = (ix: number, iy: number, iz: number) => ix > -OFF && ix < OFF && iy >= 0 && iy < OFF && iz > -OFF && iz < OFF;

/** A crown ellipsoid (mockup `ell`): centre, semi-axes, the instant it was fitted at, and its floor. */
interface Ell {
  t: number;
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  floor: number;
}

/** The crown's bottom is at least this share of its top above the ground, so a short trunk shows under it. */
const CROWN_FLOOR = 0.3;
/** The crown's floor is at least this high (u; up to 0.6 of its top). */
const TRUNK_SHOW = 1.25;
/** Wood thicker than this (radius, u) is blocky: 2-cell (0.5 u) blocks, as the mockup's `woodCells`. */
const WOOD_FINE_R = 0.19;
/** The trunk's least radius on the 2-cell grid: 2 × 2 blocks (1 u across). */
const TRUNK_MIN_R = 0.5;
/** Beyond its 2 × 2 blocks the trunk is this many times its girth radius thick (the blocky look reads thicker). */
const TRUNK_K = 3;
/** A 2-cell wood block's outer cells are born this much after its inner ones (the cubes' scale-in time, s). */
const WOOD_STAGGER = 0.8;

interface Cand {
  ix: number;
  iy: number;
  iz: number;
  key: number;
  born: number;
  kind: number;
  /** The instant of the claim (= born, except a wood block's outer cells, born a scale-in later). */
  at: number;
}

export function voxelizeTree(buffers: { vox: VoxelInput }, opts: VoxelOpts, seed: string): VoxelCells {
  const V = buffers.vox, e = opts.size, cap = opts.cap ?? VOXEL_CAP, sd = fnv(seed);
  const ox = V.origin[0], oz = V.origin[1];
  const cands = new Map<number, Cand>();
  // kind codes double as the tie rank at one instant: wood 3 > failed 2 > leaf 0
  // the earliest claim instant wins; at one instant the earliest born, then the stronger kind (a later claim never
  // changes a cell, even one whose born lies a scale-in after its claim)
  const claim = (ix: number, iy: number, iz: number, born: number, kind: number, at = born) => {
    if (!inGrid(ix, iy, iz)) return;
    const key = keyOf(ix, iy, iz), c = cands.get(key);
    if (!c) cands.set(key, { ix, iy, iz, key, born, kind, at });
    else if (at < c.at || (at === c.at && (born < c.born || (born === c.born && kind > c.kind)))) {
      c.born = born;
      c.kind = kind;
      c.at = at;
    }
  };

  // crown (mockup `crownBuild`, crown "tree"): one solid ellipsoid fitted to the kept foliage points. Prefix-stable:
  // at every instant the ellipsoid of the points born by then is filled, so the crown is the union of the per-instant
  // ellipsoids, and a cell's born is the first instant whose ellipsoid holds it
  const F = V.foliage, pts: { x: number; y: number; z: number; r: number; born: number; fail: boolean; key: number }[] = [];
  for (let i = 0; i < F.length; i += 6) {
    const x = F[i] - ox, y = F[i + 1], z = F[i + 2] - oz, fail = F[i + 5] > 0;
    const cx = Math.floor(x / e), cy = Math.floor(y / e), cz = Math.floor(z / e);
    if (!fail && cellHash(sd, cx, cy, cz, SALT.density) >= opts.density) continue;
    pts.push({ x, y, z, r: F[i + 3], born: F[i + 4], fail, key: inGrid(cx, cy, cz) ? keyOf(cx, cy, cz) : -1 });
  }
  // a fixed order, so the running sums (floating point) never depend on the input order
  pts.sort((p, q) => p.born - q.born || p.x - q.x || p.y - q.y || p.z - q.z || p.r - q.r || Number(p.fail) - Number(q.fail));
  const ells: Ell[] = [];
  let n = 0, sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, mr = 0;
  for (let i = 0; i < pts.length; ) {
    const t = pts[i].born;
    for (; i < pts.length && pts[i].born === t; i++) {
      const p = pts[i];
      n++;
      (sx += p.x), (sy += p.y), (sz += p.z), (sxx += p.x * p.x), (syy += p.y * p.y), (szz += p.z * p.z);
      mr = Math.max(mr, p.r);
    }
    const cx = sx / n, cy = sy / n, cz = sz / n, sd2 = (s2: number, c: number) => Math.sqrt(Math.max(0, s2 / n - c * c));
    const E = { t, cx, cy, cz, ex: sd2(sxx, cx) * 1.7 + mr, ey: sd2(syy, cy) * 1.6 + mr * 0.8, ez: sd2(szz, cz) * 1.7 + mr, floor: 0 };
    // the crown's floor: the ellipsoid's own bottom, never below CROWN_FLOOR of its top nor below TRUNK_SHOW (up to half
    // the top): the trunk shows under it, from the first turn on (early, low crowns stay part of the union)
    const top = cy + E.ey;
    E.floor = Math.max(0.2, cy - E.ey, CROWN_FLOOR * top, Math.min(TRUNK_SHOW, 0.6 * top));
    ells.push(E);
    for (let ix = Math.floor((cx - E.ex) / e); ix <= Math.floor((cx + E.ex) / e); ix++)
      for (let iy = Math.max(0, Math.floor(E.floor / e)); iy <= Math.floor((cy + E.ey) / e); iy++)
        for (let iz = Math.floor((cz - E.ez) / e); iz <= Math.floor((cz + E.ez) / e); iz++) {
          const px = (ix + 0.5) * e, py = (iy + 0.5) * e, pz = (iz + 0.5) * e;
          if (py <= E.floor || !inGrid(ix, iy, iz)) continue;
          const q = ((px - cx) / E.ex) ** 2 + ((py - cy) / E.ey) ** 2 + ((pz - cz) / E.ez) ** 2;
          // the mockup's rough rim: the outer shell loses cells by a hash
          if (q > 1 - (q > 0.55 ? 0.3 * cellHash(sd, ix, iy, iz, SALT.rim) : 0)) continue;
          if (!cands.has(keyOf(ix, iy, iz))) claim(ix, iy, iz, t, KIND.leaf);
        }
  }
  /** The crown ellipsoid as it stood at time `t` (null before the first foliage). */
  const ellAt = (t: number): Ell | null => {
    let lo = 0, hi = ells.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ells[mid].t <= t) lo = mid + 1;
      else hi = mid;
    }
    return lo ? ells[lo - 1] : null;
  };
  /** Mockup `woodCells`: wood that would poke out of the upper crown (as it stood at the wood's born) is left out. */
  const clipped = (x: number, y: number, z: number, born: number) => {
    const E = ellAt(born);
    return !!E && y > E.cy - E.ey * 0.5 && ((x - E.cx) / E.ex) ** 2 + ((y - E.cy) / E.ey) ** 2 + ((z - E.cz) / E.ez) ** 2 > 0.8;
  };

  // wood (mockup `woodCells`): steps of ≤ 1/4 cell along every centreline. Where a schedule entry's radius exceeds
  // WOOD_FINE_R the wood is blocky, on the 2-cell grid (0.5 u); the trunk is at least TRUNK_MIN_R thick on that grid
  // (2 × 2 blocks). A cell is covered from the first entry whose radius reaches it
  const W = V.wood, S = V.sched, step = e / 4;
  const sched = (s: number) => ({ o: W[s * 6 + 4], n: W[s * 6 + 5] });
  const valueAt = (o: number, n: number, t: number) => {
    let v = 0;
    for (let k = 0; k < n && S[(o + k) * 2] <= t; k++) v = S[(o + k) * 2 + 1];
    return v;
  };
  /**
   * A wood block of g × g × g cells. A 2-cell block grows from the inside out: its cells farther (across) from the
   * centreline than the block's centre are born SCALE_IN later, so the wood's extent widens by one cell per scale-in
   * (the trunk's girth moves smoothly; S1).
   */
  const woodCell = (g: number, gx: number, gy: number, gz: number, born: number, px: number, pz: number) => {
    const ge = g * e, bx = (gx + 0.5) * ge, bz = (gz + 0.5) * ge;
    if (clipped(bx, (gy + 0.5) * ge, bz, born)) return;
    const d0 = Math.hypot(bx - px, bz - pz);
    for (let a = 0; a < g; a++)
      for (let b2 = 0; b2 < g; b2++)
        for (let c = 0; c < g; c++) {
          const x = gx * g + a, z = gz * g + c, out = g > 1 && Math.hypot((x + 0.5) * e - px, (z + 0.5) * e - pz) > d0 + 1e-9;
          claim(x, gy * g + b2, z, out ? Math.fround(born + WOOD_STAGGER) : born, KIND.wood, born);
        }
  };
  const stepPoint = (x: number, y: number, z: number, born: number, ts: number[], rs: number[], trunk: boolean) => {
    // the entries on each grid: (time, reach)
    const fine: [number, number][] = [], coarse: [number, number][] = [];
    if (trunk) coarse.push([born, TRUNK_MIN_R]);
    rs.forEach((r, j) => {
      const t = Math.max(born, ts[j]);
      // the trunk: 2 × 2 blocks at least, widened with its girth on the fine grid beyond them
      if (trunk) fine.push([t, TRUNK_K * r]);
      else if (r > WOOD_FINE_R) coarse.push([t, Math.max(r, e)]);
      else fine.push([t, Math.max(r, e / 2)]);
    });
    if (!trunk) woodCell(1, Math.floor(x / e), Math.floor(y / e), Math.floor(z / e), born, x, z); // always its own cell
    for (const [g, list] of [[1, fine], [2, coarse]] as const) {
      if (!list.length) continue;
      const ge = g * e, rmax = list[list.length - 1][1];
      for (let ix = Math.floor((x - rmax) / ge); ix <= Math.floor((x + rmax) / ge); ix++)
        for (let iy = Math.floor((y - rmax) / ge); iy <= Math.floor((y + rmax) / ge); iy++)
          for (let iz = Math.floor((z - rmax) / ge); iz <= Math.floor((z + rmax) / ge); iz++) {
            const dx = (ix + 0.5) * ge - x, dy = (iy + 0.5) * ge - y, dz = (iz + 0.5) * ge - z, d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > rmax * rmax) continue;
            let j = 0;
            while (list[j][1] * list[j][1] < d2) j++;
            woodCell(g, ix, iy, iz, list[j][0], x, z);
          }
    }
  };
  for (let l = 0; l + 1 < V.lines.length; l++) {
    const s0 = V.lines[l], s1 = V.lines[l + 1], trunk = l === V.trunk;
    for (let s = s0; s < s1; s++) {
      const a = s * 6, A = sched(s);
      if (s + 1 === s1) {
        // the line's last sample on its own
        const ts: number[] = [], rs: number[] = [];
        for (let k = 0; k < A.n; k++) {
          ts.push(S[(A.o + k) * 2]);
          rs.push(S[(A.o + k) * 2 + 1]);
        }
        stepPoint(W[a] - ox, W[a + 1], W[a + 2] - oz, W[a + 3], ts, rs, trunk);
        continue;
      }
      const b = a + 6, Bs = sched(s + 1);
      const times = new Set<number>();
      for (let k = 0; k < A.n; k++) times.add(S[(A.o + k) * 2]);
      for (let k = 0; k < Bs.n; k++) times.add(S[(Bs.o + k) * 2]);
      const ts = [...times].sort((p, q) => p - q);
      const va = ts.map((t) => valueAt(A.o, A.n, t)), vb = ts.map((t) => valueAt(Bs.o, Bs.n, t));
      const dx = W[b] - W[a], dy = W[b + 1] - W[a + 1], dz = W[b + 2] - W[a + 2];
      const n2 = Math.max(1, Math.ceil(Math.hypot(dx, dy, dz) / step));
      for (let k = 0; k < n2; k++) {
        const u = k / n2, rs = va.map((v, j) => v + (vb[j] - v) * u);
        // the segment toward the next sample grows with that sample's turn
        stepPoint(W[a] + dx * u - ox, W[a + 1] + dy * u, W[a + 2] + dz * u - oz, k ? W[b + 3] : W[a + 3], ts, rs, trunk);
      }
    }
  }

  // emit instant by instant (born order), then by density priority; blossoms of an instant after its cells
  const list = [...cands.values()].map((c) => ({ c, pri: cellHash(sd, c.ix, c.iy, c.iz, SALT.priority) }));
  list.sort((p, q) => p.c.at - q.c.at || p.pri - q.pri || p.c.key - q.c.key);
  // single cubes set on the crown: blossoms (answered turns' tips) and failed tools (their foliage points), each on the
  // outside of the crown as it stood at its turn (mockup `bloomAt`), then up to the first free cell
  const T = V.tips, tips: { x: number; y: number; z: number; born: number; kind: number }[] = [];
  const onCrown = (x: number, y: number, z: number, born: number, kind: number) => {
    const E = ellAt(born);
    if (E) {
      const vx = x - E.cx, vy = y - E.cy, vz = z - E.cz, q = Math.sqrt((vx / E.ex) ** 2 + (vy / E.ey) ** 2 + (vz / E.ez) ** 2) || 1;
      (x = E.cx + vx / q), (y = E.cy + vy / q), (z = E.cz + vz / q);
      // under the crown's floor (the lower surface is cut away): onto the upper surface above it instead
      if (y <= E.floor + e) y = E.cy + E.ey * Math.sqrt(Math.max(0, 1 - ((x - E.cx) / E.ex) ** 2 - ((z - E.cz) / E.ez) ** 2));
    }
    tips.push({ x, y, z, born, kind });
  };
  if (opts.blossoms)
    for (let i = 0; i < T.length; i += 4) {
      const x = T[i] - ox, y = T[i + 1], z = T[i + 2] - oz;
      // mockup: blossoms thinned like the leaves
      if (cellHash(sd, Math.floor(x / e), Math.floor(y / e), Math.floor(z / e), SALT.bloom) < Math.max(opts.density, 0.35)) onCrown(x, y, z, T[i + 3], KIND.blossom);
    }
  for (const p of pts) if (p.fail) onCrown(p.x, p.y, p.z, p.born, KIND.failed);
  tips.sort((p, q) => p.born - q.born || p.kind - q.kind || p.y - q.y || p.x - q.x || p.z - q.z);

  /** Kept cells (drawn or buried) by key: their kind. */
  const kept = new Map<number, number>();
  const enclosed = (key: number) =>
    kept.has(key + SPAN * SPAN) && kept.has(key - SPAN * SPAN) && kept.has(key + SPAN) && kept.has(key - SPAN) &&
    kept.has(key + 1) && kept.has(key - 1);
  const oc: number[] = [], ok: number[] = [], ob: number[] = [], oq: number[] = [];
  const keep = (ix: number, iy: number, iz: number, kind: number, born: number) => {
    kept.set(keyOf(ix, iy, iz), kind);
    oc.push(ix, iy, iz);
    ok.push(kind);
    ob.push(born);
    oq.push(0);
  };
  const buryFrom = (k0: number) => {
    for (let k = k0; k < ok.length; k++) oq[k] = enclosed(keyOf(oc[k * 3], oc[k * 3 + 1], oc[k * 3 + 2])) ? 1 : 0;
  };
  let i = 0, j = 0;
  while ((i < list.length || j < tips.length) && ok.length < cap) {
    const t = Math.min(i < list.length ? list[i].c.at : Infinity, j < tips.length ? tips[j].born : Infinity);
    // this instant's cells up to the cap; then burial against what was kept (cells the cap dropped are not solid)
    const k0 = ok.length;
    for (; i < list.length && list[i].c.at === t; i++) {
      const c = list[i].c;
      if (ok.length < cap && !kept.has(c.key)) keep(c.ix, c.iy, c.iz, c.kind, c.born);
    }
    buryFrom(k0);
    for (; j < tips.length && tips[j].born === t; j++) {
      const p = tips[j];
      const ix = Math.floor(p.x / e), iz = Math.floor(p.z / e);
      let iy = Math.max(0, Math.floor(p.y / e));
      // climb over crown and wood only: a column already topped by a single cube (blossom or failed) gets no second
      let open = true;
      for (let k = 0; k < CLIMB && kept.has(keyOf(ix, iy, iz)) && open; k++) {
        const kd = kept.get(keyOf(ix, iy++, iz));
        open = kd !== KIND.blossom && kd !== KIND.failed;
      }
      const below = iy > 0 ? kept.get(keyOf(ix, iy - 1, iz)) : undefined;
      if (below === KIND.blossom || below === KIND.failed) open = false; // never stacked on another single cube
      if (open && ok.length < cap && inGrid(ix, iy, iz) && !kept.has(keyOf(ix, iy, iz))) {
        keep(ix, iy, iz, p.kind, p.born);
        buryFrom(ok.length - 1);
      }
    }
  }

  const count = ok.length, shade = new Uint8Array(count);
  for (let k = 0; k < count; k++) shade[k] = shadeHash(sd, oc[k * 3], oc[k * 3 + 1], oc[k * 3 + 2], opts);
  return { coords: new Int32Array(oc), kind: new Uint8Array(ok), shade, born: new Float32Array(ob), buried: new Uint8Array(oq), count };
}
