/**
 * README §4.3 — the spit: history runs left → right on a non-linear X(t), trees take lanes across the spit,
 * resumed sessions transplant the tree forward, and the spit widens only where many trees stand together.
 * Ported from forest-real.html (`XS`/`XSTEP`/`XTAB`/`X`, `LANES`, moves, places, `widths`/`wAt`) with the same
 * constants. The mockup worked on globals (NOW, TREES); here everything comes in as arguments.
 */
import type { Session, Tree } from "./types";

/** World units per second at normal speed, and the XTAB bucket size in seconds. */
export const XS = 0.0062;
export const XSTEP = 60;
export const LANES = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5] as const;
export const LANE_W = 4.4;
export const BASE_W = 15;
/**
 * Quiet hours (coordinator ruling 2026-10-06): a stretch of minutes with no tree active adds at most this many units to
 * X, however long it is (about 21 min of the normal speed: shorter pauses are unchanged, a night is a short step).
 * Global, in the speed table: it never depends on the selected range. Transplants are still decided on the
 * uncompressed X (a session resumed after a night moves, as it always did).
 */
export const IDLE_CAP_U = 8;

/** "6 h quiet": a compressed stretch's label (the marker's tooltip). */
export function quietLabel(seconds: number): string {
  const m = Math.round(seconds / 60);
  if (m < 60) return `${m} min quiet`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h quiet`;
  const d = Math.floor(h / 24), r = h - d * 24;
  return `${d} d${r ? ` ${r} h` : ""} quiet`;
}

/** A quiet stretch the speed table compressed: [from, to) in the timeline's time. */
export interface QuietGap {
  from: number;
  to: number;
}

/** The layout reaches at most this many minutes before time 0 (31 days: past the widest range). */
export const MAX_BACK_MIN = 31 * 24 * 60;
/** A session's next turn after this gap (s) may transplant the tree… */
export const MOVE_GAP_S = 1800;
/** …if it lands more than this many units further along. */
export const MOVE_MIN_UNITS = 20;

/**
 * The absolute time a layout starts at for a window anchored at `t0`: the minute boundary at or before it. XTAB's
 * XSTEP buckets then fall on whole minutes for every window, so two layouts of the same data (the Stats card's 6 h and
 * the full window's 48 h) give the same X(now) − X(t) for any t well inside both (a front-relative hand-off).
 */
export const layoutOrigin = (t0: number): number => Math.floor(t0 / XSTEP) * XSTEP;

export interface Place {
  /** Relative time this spot was taken (tree start, or the resumed turn). */
  from: number;
  x: number;
  z: number;
  /** Lane number (0, ±1, ±2 …); z = lane × LANE_W. */
  lane: number;
}

/**
 * Every tree is laid out (and built) at scale 1 (ruling R25). The mockup's `tr.s` (to_forest.py: 0.8 up to 60 raw
 * requests, then 1.0) is dropped: in the live view it rescaled all grown wood by 1.25 in one frame when a tree passed
 * 60 requests. A tree's size now comes only from its growth.
 */
export const TREE_SCALE = 1;

export interface Timeline {
  X(t: number): number;
  /** Lane number of the tree's first place (0 for an unknown tree). */
  lane(treeId: string): number;
  /** Spit half-width at x on the north (−1, −z) or south (+1, +z) side, as of `cut` (default: `now`). */
  width(x: number, side: -1 | 1, cut?: number): number;
  /** Every spot the tree stands on, in time order ([] for an unknown tree). */
  places(treeId: string): Place[];
  /** The quiet stretches compressed (IDLE_CAP_U), in time order. */
  gaps(): QuietGap[];
}

const walk = (sessions: Session[], f: (s: Session) => void) => {
  const go = (s: Session) => {
    f(s);
    s.children.forEach(go);
  };
  sessions.forEach(go);
};

/**
 * Ports mockup `XTAB`: speed per minute from the number of trees active in the last 10 min, smoothed ±10 min. Bucket i
 * is the minute `i + k0`: k0 = 0 unless a turn lies before time 0 (a range switch brought older history in; follow-up
 * C), in which case the table reaches back to that turn's minute, or `from` is (the window's start: flowers there).
 */
function buildXTab(trees: Tree[], now: number, from = 0): { X: Float64Array; raw: Float64Array; k0: number; gaps: QuietGap[] } {
  let k0 = Math.min(0, Math.floor(from / XSTEP));
  for (const tr of trees) walk(tr.sessions, (s) => s.turns.forEach((t) => (k0 = Math.min(k0, Math.floor(t[0] / XSTEP)))));
  // bounded (a bad time must not allocate gigabytes): older turns sit at the table's start
  k0 = Number.isFinite(k0) ? Math.max(k0, -MAX_BACK_MIN) : 0;
  const N = Math.ceil(Math.max(0, now) / XSTEP) + 2 - k0;
  const act = new Float64Array(N);
  for (const tr of trees) {
    const seen = new Set<number>();
    walk(tr.sessions, (s) => {
      for (const t of s.turns)
        for (let k = 0; k <= 10; k++) {
          const i = Math.floor(t[0] / XSTEP) + k - k0;
          if (i >= 0 && i < N && !seen.has(i)) {
            seen.add(i);
            act[i]++;
          }
        }
    });
  }
  const sp = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const a = act[i];
    sp[i] = a === 0 ? 1 : a === 1 ? 0.12 : a === 2 ? 0.3 : Math.min(1, 0.3 + 0.25 * (a - 2));
  }
  const sm = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0, c = 0;
    for (let k = -10; k <= 10; k++) {
      const j = i + k;
      if (j >= 0 && j < N) {
        s += sp[j];
        c++;
      }
    }
    sm[i] = s / c;
  }
  // the table before the quiet hours are compressed: transplants are decided on it (a session resumed after a night
  // still moves, as it always did), while places use the compressed one
  const raw = new Float64Array(N);
  for (let i = 1; i < N; i++) raw[i] = raw[i - 1] + sm[i - 1] * XS * XSTEP;
  // quiet hours: every maximal run of idle minutes is scaled down to add at most IDLE_CAP_U (speeds stay > 0: X stays
  // monotone). Working time is never compressed (re-review N1).
  const gaps: QuietGap[] = [];
  const cap = (quiet: (a: number) => boolean, max: number, mark: boolean): void => {
    for (let i = 0; i < N; ) {
      if (!quiet(act[i])) {
        i++;
        continue;
      }
      let j = i, len = 0;
      while (j < N && quiet(act[j])) len += sm[j++] * XS * XSTEP;
      if (len > max) {
        const f = max / len;
        for (let k = i; k < j; k++) sm[k] *= f;
        // marked only between activity: the empty stretch before the table's first activity is no quiet hour
        if (mark && i > 0) gaps.push({ from: (i + k0) * XSTEP, to: (j + k0) * XSTEP });
      }
      i = j;
    }
  };
  cap((a) => a === 0, IDLE_CAP_U, true);
  const X = new Float64Array(N);
  for (let i = 1; i < N; i++) X[i] = X[i - 1] + sm[i - 1] * XS * XSTEP;
  return { X, raw, k0, gaps };
}

const cmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

interface TreeInfo {
  tree: Tree;
  start: number;
  places: Place[];
}

/**
 * The layout of `trees` (times ≥ 0, or older history: see buildXTab) up to `now`. X(0) = `xoff` (0 by default): the
 * scene shifts a new layout by `xoff` so the trees near the front keep their places across a range switch (frame.ts
 * `keepFront`); the spit's shore is evaluated in those shifted coordinates. `from`: X reaches back at least to this
 * time (the shown window's start, where its flowers stand), not only to the oldest turn. `fixed`: the places of trees
 * already shown (review I3), kept as they are (only a later transplant is added); the other trees are laid out around
 * them. `frozen`: X as already shown, kept (only time after it follows this table).
 */
/** X as already shown (re-review N2): samples every XSTEP s from `t0` (the timeline's time). */
export interface FrozenX {
  t0: number;
  xs: readonly number[];
}

export function buildTimeline(
  trees: Tree[], now: number, xoff = 0, from = 0, fixed?: ReadonlyMap<string, readonly Place[]>, frozen?: FrozenX,
): Timeline {
  const { X: XTAB, raw: RAW, k0, gaps } = buildXTab(trees, now, Number.isFinite(from) ? from : 0);
  const base = XTAB[-k0];
  // Ports mockup `X`: linear interpolation in XTAB (beyond `now`, the last segment is extrapolated).
  const interp = (T: Float64Array, t: number) => {
    const f = Math.max(0, t / XSTEP - k0);
    const i = Math.min(T.length - 2, Math.floor(f));
    const w = f - i;
    return T[i] * (1 - w) + T[i + 1] * w;
  };
  const Xc = (t: number) => interp(XTAB, t) - base + xoff;
  // X already shown is kept: inside the frozen samples X is theirs; after them it continues by this table's own
  // advance, before them it reaches back by it (older history a wider range brings in)
  const fz = frozen && frozen.xs.length ? frozen : null;
  const X = !fz
    ? Xc
    : (t: number) => {
        const n = fz.xs.length, t1 = fz.t0 + (n - 1) * XSTEP;
        if (t >= t1) return fz.xs[n - 1] + Xc(t) - Xc(t1);
        if (t <= fz.t0) return fz.xs[0] - (Xc(fz.t0) - Xc(t));
        const f = (t - fz.t0) / XSTEP, i = Math.min(n - 2, Math.floor(f)), w = f - i;
        return fz.xs[i] * (1 - w) + fz.xs[i + 1] * w;
      };
  /** X without the quiet-hour compression: only for the transplant rule. */
  const Xraw = (t: number) => interp(RAW, t);

  // Mockup: tr.start = earliest turn across sessions and subagents (wire `start` if the tree has no turns);
  // then TREES.sort by start. Ties break by tree id, so the result never depends on input (or Map) order.
  const infos: TreeInfo[] = trees.map((tree) => {
    let s0 = Infinity;
    walk(tree.sessions, (s) => s.turns.forEach((t) => (s0 = Math.min(s0, t[0]))));
    return { tree, start: Number.isFinite(s0) ? s0 : tree.start, places: [] };
  });
  infos.sort((a, b) => a.start - b.start || cmp(a.tree.id, b.tree.id));

  // Ports mockup moves: the first turn of a session after a > 30 min gap, if it lands > 20 units past the
  // tree's current spot. Candidates are in time order, so a tree only ever moves forward.
  const held = (info: TreeInfo) => {
    const f = fixed?.get(info.tree.id);
    return f && f.length ? f : null;
  };
  const moves = infos.map((info) => {
    const cand: number[] = [];
    walk(info.tree.sessions, (s) => {
      for (let i = 1; i < s.turns.length; i++) if (s.turns[i][0] - s.turns[i - 1][0] > MOVE_GAP_S) cand.push(s.turns[i][0]);
    });
    cand.sort((a, b) => a - b);
    const out: number[] = [];
    const f = held(info);
    let at = f ? f[f.length - 1].from : info.start;
    for (const t of cand.filter((c) => !f || c > at))
      if (Xraw(t) - Xraw(at) > MOVE_MIN_UNITS) {
        out.push(t);
        at = t;
      }
    return out;
  });

  // Ports mockup places: every spot a tree ever stands on, in time order, gets a lane; a vacated spot keeps it.
  // A first spot sits 2 units past X(start); a transplant lands 16 units ahead so it settles on the frame's right.
  const spots: { info: TreeInfo; from: number; first: boolean }[] = [];
  const placed: Place[] = [];
  infos.forEach((info, k) => {
    const f = held(info);
    if (f) for (const p of f) (placed.push({ ...p }), info.places.push({ ...p }));
    else spots.push({ info, from: info.start, first: true });
    for (const t of moves[k]) spots.push({ info, from: t, first: false });
  });
  spots.sort((a, b) => a.from - b.from || cmp(a.info.tree.id, b.info.tree.id) || Number(b.first) - Number(a.first));
  for (const sp of spots) {
    const x = X(sp.from) + (sp.first ? 2 : 16);
    let lane = 0;
    for (const ln of LANES) {
      const z = ln * LANE_W;
      if (!placed.some((o) => o.z === z && Math.abs(o.x - x) < 5.5 * 2 * TREE_SCALE)) {
        lane = ln;
        break;
      }
    }
    const p: Place = { from: sp.from, x, z: lane * LANE_W, lane };
    placed.push(p);
    sp.info.places.push(p);
  }

  const byId = new Map<string, TreeInfo>();
  for (const info of infos) if (!byId.has(info.tree.id)) byId.set(info.tree.id, info);

  // Ports mockup `placeIdx`: the latest place taken at or before `cut`.
  const placeAt = (info: TreeInfo, cut: number) => {
    let k = 0;
    for (let i = 0; i < info.places.length; i++) if (info.places[i].from <= cut) k = i;
    return info.places[k];
  };

  // Ports mockup `widths` + `wAt`, evaluated directly at x instead of tabulated on a 0.5-unit grid. The spit only
  // grows where more than 10 trees stand together: half a lane per extra tree, rising over its first 5 minutes.
  const width = (x: number, side: -1 | 1, cut = now) => {
    let cnt = 0;
    for (const info of infos) {
      if (info.start > cut) continue;
      const pl = placeAt(info, cut);
      const span = 8 * TREE_SCALE + 4, dx = x - pl.x;
      if (dx < -span || dx > span) continue;
      cnt += (0.5 + 0.5 * Math.cos((Math.PI * dx) / span)) * Math.min(1, (cut - info.start) / 300);
    }
    const extra = Math.max(0, cnt - 10) * LANE_W * 0.5;
    return side < 0
      ? BASE_W + extra + 1.2 * Math.sin(x * 0.05) + 0.6 * Math.sin(x * 0.17 + 1)
      : BASE_W + extra + 1.2 * Math.sin(x * 0.06 + 2) + 0.6 * Math.sin(x * 0.19);
  };

  return {
    X,
    lane: (id) => byId.get(id)?.places[0]?.lane ?? 0,
    width,
    places: (id) => byId.get(id)?.places.slice() ?? [],
    gaps: () => gaps.map((g) => ({ ...g })),
  };
}
