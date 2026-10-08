/**
 * The scene's per-frame orchestration and data intake, without three.js (unit-tested with fakes):
 * - **Scene time.** The scene keeps one time origin for its lifetime: the server's now (absolute seconds) in the first
 *   state it sees, so scene times start near 0 (history is negative). Every incoming state is converted:
 *   `sceneRel = stateRel + (state.t0 − origin)`. The server re-anchors t0 on every full refresh (about every 5 min) and
 *   rounds relative times to 0.1 s against each anchor, so the same turn can come back up to 0.05 s off; a turn time
 *   within 0.15 s of the one already known for that session and index keeps the known value (`shiftTree`). Built
 *   buffers (the cells' born), places and the clock therefore never move on a t0 change.
 * - **Float32 resolution** of GPU times (iborn, uNow) at scene time T is 2^(⌊log2 T⌋ − 23):
 *   0.0005 s at 1 h, 0.008 s at 1 day (about half a 60 Hz frame), 0.016 s at 2 days, 0.031 s at 4 days, 0.0625 s at
 *   7 days. No re-origin is done: over a 7-day tab the shader's history time then advances in steps of at most
 *   1/16 s, i.e. a cube's 0.8 s scale-in moves in ≤ 8 % steps (a few frames) and its 12 s tint fade in ≤ 0.5 %
 *   steps. A
 *   re-origin would need every tree rebuilt and swapped in one frame (all times are baked into the buffers).
 * - **Layout time.** The timeline (lib/forest/timeline) integrates X(t) from t = 0 and clamps negative times, so it is
 *   fed `layout = scene + H`: layout time = absolute − `layoutOrigin(first t0)` (the minute at or before it), ≥ 0 for
 *   everything in the window, with XTAB's minute buckets on whole minutes so a 6 h and a 48 h layout agree near the
 *   front (the card's front-relative hand-off); `sceneTimeline` hides the offset.
 * - **Late turns.** One rule (final review I1, I2): a turn is late only if it is newer than everything the scene has
 *   already accounted for. Late, it grows from its arrival (`LateTurns`). Known turns are tracked by session id and
 *   turn index across all trees, so a tree id change (a day turning busy, a window slide) never makes them late, and a
 *   new tree id holding sessions already shown under other ids is `inherited` (built fully grown, no emergence).
 *   History bands: every discontinuous clock step (the first data, a resume after a hidden tab of more than
 *   CATCH_UP_MAX, a skew jump or a frame stall over 10 s) marks only the stretch it skipped, (from, to], as history
 *   (the first data: everything up to the first shown time). A turn inside a band is never late; one that started
 *   before a jump and finishes after it still grows from its arrival (re-review N1, N2). A short hide is no jump: the
 *   clock catches up. Flora follows the same rule (`lateFor`, ground.ts).
 * - **Range switch** (follow-up C). The origin never changes. At a switch, everything the old range no longer covered
 *   then becomes a history band (the trees a wider range brings in are never late), and a session's turn indices are
 *   aligned with its previous copy by their times (`indexShift`). A held session's turn list never changes at its old
 *   end (review C1): older turns are not prepended and cut-off ones are kept, so turns are only appended, known turns
 *   keep their times and indices, and a late turn keeps its arrival. The server sends trees whole anyway; this is the
 *   client's guarantee.
 * - **Clock.** The scene owns the `ForestClock`, in scene time. The shown time runs on the browser's monotonic clock;
 *   the server's now in each state only feeds the skew estimate (`SkewEstimator`, clock.ts; final review C1).
 * - **Pause** (the Stats card scrolled offscreen): `frame` is a no-op and `intake` only queues the latest state.
 *   `resume` applies it, then follows the hidden-tab rule (`resumeAfterHidden`): a pause longer than CATCH_UP_MAX is
 *   one jump whose stretch is history; a shorter one is no jump (the clock catches up, a late turn grows from its
 *   arrival).
 * - **Camera.** The controller is stepped every frame (its step is what ends a drag pause); its pose is applied every
 *   frame unless the user holds it. `setEvents` runs only when the trees changed.
 * - **Rebuild scheduling** (`planTrees`): a tree is built when first seen, then exactly at its hand-over times
 *   (`nextEnd`: its next turn, 20 s later, or its freeze), requested PREBUILD_S early. Its cells only append, so a swap
 *   needs no morph: the new cells scale in from their born. A turn that
 *   reaches the data at or before a requested cut but was not in that request's data (a late turn, or one arriving
 *   inside the prebuild window) is built at once: at its own time if that is still ahead, else now; the stale request's
 *   result is dropped (trees.ts). A place move or a t0 move never schedules a build.
 */
import { type FrozenX, type Place, type Timeline, XSTEP, buildTimeline, layoutOrigin } from "@/lib/forest/timeline";
import { type ForestState, RANGE_S, type Range, type Session, type Tree } from "@/lib/forest/types";
import type { CamEvent, CameraMode, V3 } from "./camera";
import { CATCH_UP_MAX, ForestClock, type ClockStep, SkewEstimator } from "./clock";

export const MAX_BUILT = 60;
export const BEHIND = 320;
export const REBUILD_S = 20;
export const FREEZE_S = 1800;
/** A scheduled rebuild is requested this long (s of history) before it is shown, so it swaps on time. */
export const PREBUILD_S = 2;
/** Same turn, re-anchored: times within this (s) are one time. */
export const TIME_TOLERANCE = 0.15;
/** History bands kept (one per discontinuous step); older ones merge. */
const MAX_BANDS = 64;

// ---------------- scene time ----------------

const stick = (v: number, prev: number | undefined) => (prev !== undefined && Math.abs(v - prev) <= TIME_TOLERANCE ? prev : v);

/**
 * Late turns. A turn's time is its request start, but its row reaches the data only when the request finishes, often
 * after the shown time has passed it. Such a turn grows from its arrival instead: effective time = max(t, arrival),
 * recorded per (session, turn index) — across trees, so a tree id change keeps it — so every later copy and rebuild
 * uses the same value (born, RAMP, the camera's event). A turn is late only when it is unknown (no earlier scene copy of
 * its session has that index, in any tree), outside the history bands, and before the shown time `rel` (null on the
 * first intake: everything then is history).
 */
export interface LateTurns {
  /** Per (session, index): the turn's own scene time when it was found late, and its effective time. */
  map: Map<string, { raw: number; eff: number }>;
  rel: number | null;
  /** True for a time inside a stretch a discontinuous clock step skipped: history, never late. */
  isHistory?: (t: number) => boolean;
  /** The previous scene copy of every session, by id, across all trees. */
  known?: ReadonlyMap<string, Session>;
}

const lateKey = (ses: string, i: number) => `${ses}\u0001${i}`;

/**
 * How far the session's turn indices moved against the previous copy (follow-up C): the window's start can add older
 * turns before the known ones (a wider range) or drop the oldest (a narrower range, or the window sliding). Positive:
 * this many turns were prepended (new index i is the previous i − off); `shiftSession` then keeps the held list. Matched on the first turns' own times (a late
 * turn's raw time, not its arrival).
 */
function indexShift(s: Session, d: number, prev: Session | undefined, late?: LateTurns): number {
  if (!prev?.turns.length || !s.turns.length) return 0;
  const prevRaw = (i: number) => late?.map.get(lateKey(s.id, i))?.raw ?? prev.turns[i][0];
  const p0 = prevRaw(0), n0 = s.turns[0][0] + d;
  let off = 0;
  if (n0 < p0 - TIME_TOLERANCE) while (off < s.turns.length && s.turns[off][0] + d < p0 - TIME_TOLERANCE) off++;
  else if (p0 < n0 - TIME_TOLERANCE) while (-off < prev.turns.length && prevRaw(-off) < n0 - TIME_TOLERANCE) off--;
  return off;
}

/** forest.py `LOD_CAP`: a session's first turns sent one by one; later ones in closing blocks of `LOD_BLOCK`. */
export const LOD_CAP = 2000;

/**
 * Held singles win over a closing LOD block (Task 1 fix 2, I1). Past `LOD_CAP` the server sends a filling block's
 * turns one by one and then one merged turn for the whole block. Replacing singles the client already grew with that
 * merged turn would re-lay their limb, foliage and blossoms (cells vanishing). So, against the previous scene copy:
 * every held turn stays as held (its time, index and data), a merged entry whose time is within the held turns is
 * ignored (it covers held singles), and only server entries newer than the last held turn are appended — a merged block
 * is accepted only for turns the client never saw as singles. Indices after the first merged block then count the
 * client's held list; a child's fork index (`at`) is mapped to the client's turn at the same time. Only for an
 * unshifted window (no range-switch index shift): a window that re-cuts a long session re-forms every block anyway.
 */
function keepHeldSingles(s: Session, d: number, prev: Session | undefined, late?: LateTurns): Session {
  const pt = prev?.turns, st = s.turns;
  if (!pt || pt.length <= LOD_CAP || st.length <= LOD_CAP || indexShift(s, d, prev, late) !== 0) return s;
  const prevRaw = (i: number) => late?.map.get(lateKey(s.id, i))?.raw ?? pt[i][0];
  const n = Math.min(pt.length, st.length);
  let j = LOD_CAP;
  while (j < n && Math.abs(st[j][0] + d - prevRaw(j)) <= TIME_TOLERANCE) j++;
  if (j === pt.length) return s; // every held turn came back as sent: plain appends
  const lastHeld = prevRaw(pt.length - 1);
  const fresh = st.slice(j).filter((t) => t[0] + d > lastHeld + TIME_TOLERANCE);
  const turns = [...st.slice(0, j), ...pt.slice(j).map((t, k) => [prevRaw(j + k) - d, ...t.slice(1)] as typeof t), ...fresh];
  // a fork index past the first merged entry: the client's turn at that time (held), or its place among the appended
  const mapAt = (at: number) => {
    if (at < j || at >= st.length) return at;
    const T = st[at][0] + d;
    if (T > lastHeld + TIME_TOLERANCE) return pt.length + fresh.indexOf(st[at]);
    let k = j;
    while (k + 1 < pt.length && prevRaw(k + 1) <= T + TIME_TOLERANCE) k++;
    return k;
  };
  return { ...s, turns, children: s.children.map((c) => (c.at === undefined ? c : { ...c, at: mapAt(c.at) })) };
}

/**
 * Server LOD (forest.py `merge_turns`): past the first 2000 turns a session is sent in fixed blocks of 8; while a block
 * fills, its turns come one by one, and when it closes they become one turn (its time: the block's last start), so
 * the indices after it shift down. A late entry is therefore kept only while its index still names the same turn: it is
 * dropped when the index is past the session's length, or when the turn there no longer has the recorded time
 * (± TIME_TOLERANCE). Times within a session never go backwards (a merged block never sits before the turn ahead).
 */
function shiftSession(s: Session, d: number, prev: Session | undefined, late?: LateTurns): Session {
  prev ??= late?.known?.get(s.id);
  s = keepHeldSingles(s, d, prev, late);
  const pc = new Map((prev?.children ?? []).map((c) => [c.id, c]));
  // C1: a held session's turn list never changes at its old end. Older turns a window brings in (off > 0) are not
  // prepended; turns a window cut off (off < 0) are kept from the previous copy. Turns are only ever appended, so the
  // indices of everything shown (shoots, limbs, child `at`) stay as they were.
  const off = indexShift(s, d, prev, late);
  const keep = off < 0 && prev ? prev.turns.slice(0, -off) : [];
  const from = off > 0 ? off : 0;
  const shiftAt = (c: Session): Session => (c.at === undefined ? c : { ...c, at: Math.max(0, c.at - off) });
  const n = keep.length + s.turns.length - from;
  if (late) {
    const pre = lateKey(s.id, 0).slice(0, -1);
    for (const k of [...late.map.keys()]) if (k.startsWith(pre) && Number(k.slice(pre.length)) >= n) late.map.delete(k);
  }
  let last = keep.length ? keep[keep.length - 1][0] : -Infinity;
  const fresh = s.turns.slice(from).map((t, j) => {
    const i = j + keep.length;
    const raw = t[0] + d;
    const was = prev?.turns[i];
    let v = stick(raw, was?.[0]);
    if (late) {
      const k = lateKey(s.id, i);
      let known = late.map.get(k);
      if (known && Math.abs(known.raw - raw) > TIME_TOLERANCE) (late.map.delete(k), (known = undefined));
      if (known) v = known.eff;
      else if (late.rel !== null && v < late.rel && !late.isHistory?.(v) && !was) {
        late.map.set(k, { raw, eff: late.rel });
        v = late.rel;
      }
    }
    v = last = Math.max(v, last);
    return v === t[0] ? t : ([v, ...t.slice(1)] as typeof t);
  });
  const children = s.children.map((c) => shiftSession(off === 0 ? c : shiftAt(c), d, pc.get(c.id), late));
  // a held child the window no longer sends (it ended before the cut) stays, as shown
  const sent = new Set(s.children.map((c) => c.id));
  for (const c of prev?.children ?? []) if (!sent.has(c.id) && off < 0) children.push(c);
  return { ...s, turns: keep.length ? [...keep, ...fresh] : fresh, children };
}

/**
 * A tree in scene time (`+d`); times within TIME_TOLERANCE of the previous scene copy keep that copy's values, and a
 * turn that arrived after the shown time had passed it takes its arrival time (`late`).
 */
export function shiftTree(t: Tree, d: number, prev?: Tree, late?: LateTurns): Tree {
  const ps = new Map((prev?.sessions ?? []).map((s) => [s.id, s]));
  const sessions = t.sessions.map((s) => shiftSession(s, d, ps.get(s.id), late));
  // C1: a held root session the window no longer sends, older than everything it does send, stays as shown
  if (prev && late) {
    const sent = new Set(t.sessions.map((s) => s.id));
    const first = Math.min(...sessions.map((s) => s.turns[0]?.[0] ?? Infinity));
    const kept = prev.sessions.filter((s) => !sent.has(s.id) && (s.turns.at(-1)?.[0] ?? Infinity) < first);
    if (kept.length) sessions.unshift(...kept);
  }
  const start = stick(t.start + d, prev?.start);
  return {
    ...t,
    start: prev && late ? Math.min(start, prev.start) : start,
    end: stick(t.end + d, prev?.end),
    last: stick(t.last + d, prev?.last),
    sessions,
  };
}

/** Every turn as a camera event, ascending by time (mockup `events()`). */
export function cameraEvents(state: ForestState): CamEvent[] {
  const out: CamEvent[] = [];
  for (const tree of state.trees.values()) {
    const go = (s: Session) => {
      for (const t of s.turns) out.push({ time: t[0], sessionId: s.id, treeId: tree.id });
      s.children.forEach(go);
    };
    tree.sessions.forEach(go);
  }
  return out.sort((a, b) => a.time - b.time || (a.treeId < b.treeId ? -1 : a.treeId > b.treeId ? 1 : 0));
}

// ---------------- rebuild scheduling ----------------

export interface TreeInfo {
  tree: Tree;
  start: number;
  /** Every turn time, ascending. */
  times: number[];
  freezeAt: number;
  places: Place[];
  /** Its sessions were already on screen under other tree ids (a regroup, a window slide): never grows out of the ground. */
  inherited?: boolean;
}

export function treeInfo(tree: Tree, places: Place[], inherited = false): TreeInfo {
  const times: number[] = [];
  const go = (s: Session) => (s.turns.forEach((t) => times.push(t[0])), s.children.forEach(go));
  tree.sessions.forEach(go);
  times.sort((a, b) => a - b);
  return { tree, start: times.length ? times[0] : tree.start, times, freezeAt: (times.at(-1) ?? tree.start) + FREEZE_S, places, inherited };
}

export const placeIdxAt = (places: readonly Place[], cut: number) => {
  let k = 0;
  for (let i = 0; i < places.length; i++) if (places[i].from <= cut) k = i;
  return k;
};

/** First turn time > `cut` in an ascending list (Infinity if none). */
function nextTime(times: readonly number[], cut: number): number {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] > cut) hi = mid;
    else lo = mid + 1;
  }
  return lo < times.length ? times[lo] : Infinity;
}

/** Where a build at `cut` hands over to the next one: the next turn, 20 s on, or the freeze (= cut once frozen). */
export function nextEnd(times: readonly number[], cut: number, freezeAt: number): number {
  if (cut >= freezeAt) return cut;
  return Math.min(nextTime(times, cut), cut + REBUILD_S, freezeAt);
}

/**
 * The next build of a tree last requested at `reqCut`, if one is due at `rel`: at the next hand-over time with the
 * current data (a turn, 20 s, the freeze), requested PREBUILD_S early. A turn that reached the data only after its time
 * (rel already past it) is built at `rel`. Null: nothing due (or frozen).
 */
export function rebuildDue(reqCut: number, rel: number, times: readonly number[], freezeAt: number): number | null {
  if (reqCut >= freezeAt) return null;
  const target = nextEnd(times, reqCut, freezeAt);
  if (rel < target - PREBUILD_S) return null;
  return target > rel ? target : Math.min(rel, freezeAt);
}

/** ≤ 60 trees: the newest third first (the live front), then the nearest the view; none more than `behind` back. */
export function selectBuilt(
  c: readonly { id: string; x: number }[], viewX: number, max = MAX_BUILT, behind = BEHIND, front = Math.ceil(max / 3),
): Set<string> {
  const near = c.filter((t) => t.x >= viewX - behind);
  // the live front first (the newest trees: the furthest along the spit), then outwards from the view (re-review 3, N3)
  const out = new Set(
    [...near].sort((a, b) => b.x - a.x || (a.id < b.id ? -1 : 1)).slice(0, Math.min(front, max)).map((t) => t.id),
  );
  for (const t of [...near].sort((a, b) => Math.abs(a.x - viewX) - Math.abs(b.x - viewX) || (a.id < b.id ? -1 : 1))) {
    if (out.size >= max) break;
    out.add(t.id);
  }
  return out;
}

export interface BuildPlan {
  keep: Set<string>;
  builds: { id: string; cut: number }[];
}

/** A tree's last build request: its cut and the turn times the data had then. */
export interface ReqRecord {
  cut: number;
  times: readonly number[];
}

/** How many of the ascending `times` are ≤ `cut`. */
export function countUpTo(times: readonly number[], cut: number): number {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] > cut) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** The earliest turn time ≤ `cut` in `now` that `had` lacked (both ascending, as multisets); null if none. */
export function missedUpTo(had: readonly number[], now: readonly number[], cut: number): number | null {
  const n = countUpTo(now, cut);
  if (n <= countUpTo(had, cut)) return null;
  let j = 0;
  for (let i = 0; i < n; i++) {
    while (j < had.length && had[j] < now[i]) j++;
    if (j < had.length && had[j] === now[i]) j++;
    else return now[i];
  }
  return null;
}

/**
 * A build made at `cut` from data whose turn times were `built` is stale once the data `now` has more turns at or
 * before that cut (a late turn, or one arriving inside the prebuild window): it must not be shown over them.
 */
export function staleBuild(built: readonly number[], now: readonly number[], cut: number): boolean {
  return countUpTo(built, cut) < countUpTo(now, cut);
}

/** One tick: which trees to keep built and which to (re)build now. `reqs` holds each kept tree's last request. */
export function planTrees(
  infos: ReadonlyMap<string, TreeInfo>, reqs: ReadonlyMap<string, ReqRecord>, rel: number, viewX: number, max = MAX_BUILT, behind = BEHIND,
): BuildPlan {
  const cands: { id: string; x: number }[] = [];
  for (const [id, info] of infos) if (info.start <= rel) cands.push({ id, x: info.places[placeIdxAt(info.places, rel)].x });
  const keep = selectBuilt(cands, viewX, max, behind);
  const builds: BuildPlan["builds"] = [];
  // the newest trees' builds are asked for first (the furthest along the spit): a range switch's first burst of
  // worker builds fills the live front before the old end (final review I2)
  const xOf = new Map(cands.map((c) => [c.id, c.x]));
  const order = [...keep].sort((a, b) => xOf.get(b)! - xOf.get(a)! || (a < b ? -1 : 1));
  for (const id of order) {
    const info = infos.get(id)!, rc = reqs.get(id);
    let cut: number | null;
    if (rc === undefined) cut = Math.min(rel, info.freezeAt);
    else {
      // a turn the last request did not have, at or before its cut: build it now (at its own time if still ahead)
      const miss = missedUpTo(rc.times, info.times, rc.cut);
      cut = miss === null ? rebuildDue(rc.cut, rel, info.times, info.freezeAt) : miss > rel ? miss : Math.min(rel, info.freezeAt);
    }
    if (cut !== null) builds.push({ id, cut });
  }
  return { keep, builds };
}

/**
 * The timeline for scene-time trees: built in layout time (scene + H; ≥ 0 for the first window, older history a wider
 * range brings in is below 0) and handed back in scene time (X, places, width all take and give scene times). `xoff`
 * shifts the layout (`keepFront`); `from` (scene time) is how far back X must reach at least (the window's start).
 */
export function sceneTimeline(
  trees: Tree[], now: number, H: number, xoff = 0, from = 0, fixed?: ReadonlyMap<string, readonly Place[]>,
  frozen?: FrozenX,
): Timeline {
  const fx = fixed && new Map([...fixed].map(([id, ps]) => [id, ps.map((p) => ({ ...p, from: p.from + H }))]));
  const fz = frozen && { t0: frozen.t0 + H, xs: frozen.xs.slice() }; // a copy: the layout extends its own
  const tl = buildTimeline(trees.map((t) => shiftTree(t, H)), now + H, xoff, from + H, fx, fz);
  return {
    X: (t) => tl.X(t + H),
    lane: (id) => tl.lane(id),
    width: (x, side, cut) => tl.width(x, side, cut === undefined ? undefined : cut + H),
    places: (id) => tl.places(id).map((p) => ({ ...p, from: p.from - H })),
    gaps: () => tl.gaps().map((g) => ({ from: g.from - H, to: g.to - H })),
  };
}

/**
 * The scene's layout over its lifetime (reviews I3, C1): every tree's places are frozen when it is first laid out, so
 * neither new data nor a range switch ever moves a tree that is shown; new trees are laid out around them. X itself is
 * frozen as shown for every time older than X_FREEZE_S (re-review N2), so a wider range's older activity cannot move
 * the front either. A tree that leaves the data is forgotten.
 */
/** X is frozen for every time older than this before now (s): only the last 20 min may still rescale (re-review N2). */
export const X_FREEZE_S = 1200;

export class SceneLayout {
  readonly pinned = new Map<string, Place[]>();
  timeline: Timeline | null = null;
  /** X as shown, every XSTEP s of scene time from `t0`, up to X_FREEZE_S before the latest now (re-review N2). */
  frozen: { t0: number; xs: number[] } | null = null;

  update(trees: Tree[], now: number, H: number, from: number): Timeline {
    const tl = sceneTimeline(trees, now, H, 0, from, this.pinned, this.frozen ?? undefined);
    // freeze X back to the window's start and up to now − X_FREEZE_S (both ends only ever extend)
    const lo = Math.floor(from / XSTEP) * XSTEP, hi = Math.floor((now - X_FREEZE_S) / XSTEP) * XSTEP;
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo) {
      const f = (this.frozen ??= { t0: lo, xs: [tl.X(lo)] });
      while (f.t0 > lo) {
        f.t0 -= XSTEP;
        f.xs.unshift(tl.X(f.t0));
      }
      for (let t = f.t0 + f.xs.length * XSTEP; t <= hi; t += XSTEP) f.xs.push(tl.X(t));
    }
    const ids = new Set(trees.map((t) => t.id));
    for (const id of [...this.pinned.keys()]) if (!ids.has(id)) this.pinned.delete(id);
    for (const id of ids) {
      const ps = tl.places(id);
      if (ps.length) this.pinned.set(id, ps);
    }
    return (this.timeline = tl);
  }
}

/** The fog's density at the usual follow distances (sky.ts): light, as the style mockup's day fog. */
export const FOG_DENSITY = 0.0011;
/** The nearest far plane: far enough that the light fog (FOG_DENSITY) still fogs it out. */
export const FAR_MIN = 1800;
/** The far plane is always fogged out: fog ≥ FAR_FOG / far (e^-(FAR_FOG)² ≈ 2 % left at the far plane). */
const FAR_FOG = 0.0022 * 900;
/** Up to this camera distance (from its target) the fog, near and far planes are as they always were. */
export const FOG_FREE_DIST = 150;
/** The near plane at the usual distances. */
export const NEAR = 0.1;
/**
 * The camera planes and fog for a camera this far from its target (follow-up C, review I2): a wide range's fit can stand
 * the camera a thousand units off.
 * - far: 5× the distance (at least FAR_MIN), so the forest and the haze beyond it are drawn;
 * - near: grows with the distance squared, so the depth precision at the target (≈ d²/(near·2²⁴)) stays what it is at
 *   FOG_FREE_DIST with near 0.1 (0.013 units), and never past a quarter of the distance;
 * - fog: thins beyond FOG_FREE_DIST so the forest keeps its contrast, but never below the floor that fogs the far plane
 *   out (e^-(far·fog)² ≈ 2 %): the horizon haze of README §4.4 stays, and no sea or land edge is ever drawn sharp.
 */
export function viewDepth(dist: number): { near: number; far: number; fog: number } {
  const d = Number.isFinite(dist) ? Math.max(0, dist) : 0;
  const far = Math.max(FAR_MIN, 5 * d);
  const near = Math.max(NEAR, Math.min(NEAR * (d / FOG_FREE_DIST) ** 2, d / 4));
  const fog = Math.max(FOG_DENSITY * Math.min(1, FOG_FREE_DIST / Math.max(FOG_FREE_DIST, d)), FAR_FOG / far);
  return { near, far, fog };
}

/** Depth resolution (world units) of a 24-bit depth buffer at distance `z` for these planes. */
export const depthStep = (z: number, near: number, far: number): number => (z * z * (far - near)) / (far * near * 2 ** 24);

/** OrbitControls' usual zoom-out limit (units from the target). */
export const MAX_ZOOM_OUT = 220;
/**
 * How far out the user may zoom (review M2): the usual limit, or 1.5× the follow fit's distance when a wide range
 * stands the fit further out. Bounded by the fit, never by the camera's current distance (wheel-outs do not ratchet it).
 */
export const zoomOutLimit = (fitDist: number | null): number => Math.max(MAX_ZOOM_OUT, 1.5 * (fitDist !== null && Number.isFinite(fitDist) ? fitDist : 0));

/** The Stats card's tree budget (spec §6.3, §6.5): 12 trees for 1h and 6h, 30 for 24h and 7d. */
export const cardTreeBudget = (range: Range): number => ((RANGE_S[range] ?? 0) >= 86400 ? 30 : 12);

// ---------------- the frame core ----------------

export interface CameraLike {
  step(dt: number, ctx: { rel: number; nowMs: number }): boolean;
  pose(): { pos: V3; target: V3 };
  readonly paused: boolean;
  setEvents(events: readonly CamEvent[]): void;
}

export interface FrameOut {
  /** History time shown (scene time). */
  rel: number;
  /** The camera pose to copy onto the scene camera (null while the user holds it). */
  pose: { pos: V3; target: V3 } | null;
  /** The camera moved this frame. */
  moved: boolean;
  /**
   * The shown time made a discontinuous step since the last frame (a resume after a long pause or hidden tab, a stall,
   * a jump): the viewer was not watching it happen, so the scene cuts the follow camera to its fit (review I1).
   */
  cut: boolean;
}

export class FrameCore {
  /** The scene's time origin (absolute seconds): the server's now in the first state seen. */
  origin: number | null = null;
  /** Layout offset: scene time + H = absolute − layoutOrigin(first t0), ≥ 0 for the whole window. */
  H = 0;
  clock: ForestClock | null = null;
  /** The latest state, in scene time. */
  state: ForestState | null = null;
  /** Scene time minus the latest state's time (`state.t0 − origin`). */
  d = 0;
  /** History bands (lo, hi] in scene time, ascending and disjoint: the stretches discontinuous steps skipped. */
  private bands: [number, number][] = [];
  /** Tree ids first seen holding sessions already shown under other ids (built fully grown, no emergence). */
  readonly inherited = new Set<string>();
  private src: ForestState | null = null;
  private skew = new SkewEstimator();
  private trees = new Map<string, { src: Tree; d: number; out: Tree }>();
  private late = new Map<string, { raw: number; eff: number }>();
  private acc: ClockStep | null = null;
  private pausedFlag = false;
  /** A discontinuous step happened since the last frame (`FrameOut.cut`). */
  private jumped = false;
  /** The latest state that arrived while paused, applied on `resume`. */
  private queued: { s: ForestState; nowMs: number } | null = null;

  constructor(private cam: CameraLike) {}

  /** A new state: converted to scene time; the camera's events are replaced only when the trees changed. */
  intake(s: ForestState, nowMs: number): { state: ForestState; changed: boolean } {
    if (s === this.src && this.state) return { state: this.state, changed: false };
    if (this.pausedFlag && this.state) {
      this.queued = { s, nowMs };
      return { state: this.state, changed: false };
    }
    const prevRange = this.src?.range;
    this.src = s;
    if (this.origin === null) {
      this.origin = s.t0 + s.now;
      this.H = s.now + (s.t0 - layoutOrigin(s.t0));
    }
    const d = (this.d = s.t0 - this.origin);
    // A range switch (follow-up C): whatever the old range no longer covered at this moment is history, so the trees a
    // wider range brings back (older than the old window, or slid out of it while watching) are never late. Not a jump:
    // the shown time does not move.
    if (prevRange !== undefined && prevRange !== s.range && this.clock) {
      const edge = s.now + d - (RANGE_S[prevRange] ?? 0);
      this.addBand(-Infinity, Math.min(edge, this.clock.rel));
    }
    let changed = !this.state || this.trees.size !== s.trees.size;
    const next = new Map<string, { src: Tree; d: number; out: Tree }>();
    const trees = new Map<string, Tree>();
    // every session's previous scene copy, across trees: a tree id change never makes a known turn late
    const known = new Map<string, Session>();
    const index = (x: Session) => (known.set(x.id, x), x.children.forEach(index));
    for (const c of this.trees.values()) c.out.sessions.forEach(index);
    const late: LateTurns = { map: this.late, rel: this.clock ? this.clock.rel : null, isHistory: this.isHistory, known };
    const current = new Set<string>();
    for (const [id, t] of s.trees) {
      const c = this.trees.get(id);
      let out: Tree;
      if (c && c.src === t && c.d === d) out = c.out;
      else {
        out = shiftTree(t, d, c?.out, late);
        if (!c || !sameTree(c.out, out)) changed = true;
      }
      // its wood is already on screen under another id: never grow it out of the ground again
      if (!c && this.state && t.sessions.some((x) => known.has(x.id))) this.inherited.add(id);
      next.set(id, { src: t, d, out });
      trees.set(id, out);
      const walk = (x: Session) => (current.add(x.id), x.children.forEach(walk));
      t.sessions.forEach(walk);
    }
    this.trees = next;
    for (const id of [...this.inherited]) if (!s.trees.has(id)) this.inherited.delete(id);
    for (const k of [...this.late.keys()]) if (!current.has(k.slice(0, k.indexOf("\u0001")))) this.late.delete(k);
    const flowers = s.flowers.map((f) => [f[0] + d, f[1], f[2]] as [number, number, number]);
    this.state = { ...s, t0: this.origin, now: s.now + d, trees, flowers };
    this.skew.sample(this.state.now, nowMs);
    if (!this.clock) {
      this.clock = new ForestClock(this.state.now);
      this.bands = [[-Infinity, this.clock.rel]]; // the first data is history
    }
    if (changed) this.cam.setEvents(cameraEvents(this.state));
    return { state: this.state, changed };
  }

  /** The late-turn context for flora at this moment (ground.ts `FloraRegistry.items`): the shown time and the bands. */
  lateFor(): { rel: number | null; isHistory: (t: number) => boolean } {
    return { rel: this.clock ? this.clock.rel : null, isHistory: this.isHistory };
  }

  /** Scene time `t` lies in a stretch a discontinuous step skipped (or before the first shown time). */
  readonly isHistory = (t: number): boolean => this.bands.some(([lo, hi]) => t > lo && t <= hi);

  /** How many turns are currently held late (tests and the debug hooks). */
  lateCount(): number {
    return this.late.size;
  }

  /** The server's now in scene time at local monotonic time `nowMs` (the skew estimate; never the raw state now). */
  sceneNow(nowMs: number): number {
    return this.state ? this.skew.now(nowMs) : 0;
  }

  /**
   * The tab is shown again. Behind the bound by more than CATCH_UP_MAX: one jump to it, the skipped stretch is history.
   * Otherwise nothing: the clock catches up as after any short lag (re-review N1).
   */
  resumeAfterHidden(nowMs: number): void {
    const c = this.clock;
    if (!c || this.pausedFlag) return;
    const now = this.sceneNow(nowMs);
    c.setNow(now);
    if (c.bound - c.rel <= CATCH_UP_MAX) return;
    c.resumeAfterHidden(now);
    this.skipped(c.lastStep);
  }

  /** Paused: no frame advances the clock or the camera; new data is only queued (the latest). */
  get paused(): boolean {
    return this.pausedFlag;
  }

  pause(): void {
    this.pausedFlag = true;
  }

  /**
   * Leaves the pause: the queued state is applied first (its late turns measured against the shown time as it was,
   * as for polls during a hidden tab), then the clock follows `resumeAfterHidden`. Returns the intake of the queued
   * state (null: nothing queued, or not paused).
   */
  resume(nowMs: number): { state: ForestState; changed: boolean } | null {
    if (!this.pausedFlag) return null;
    this.pausedFlag = false;
    const q = this.queued;
    this.queued = null;
    const r = q ? this.intake(q.s, q.nowMs) : null;
    this.resumeAfterHidden(nowMs);
    return r;
  }

  /** A discontinuous step: its stretch (from, to] becomes a history band (merged with its neighbours). */
  private skipped(st: ClockStep): void {
    if (st.continuous) return;
    this.jumped = true;
    this.acc = { from: this.acc?.from ?? st.from, to: st.to, continuous: false };
    this.addBand(Math.min(st.from, st.to), Math.max(st.from, st.to));
  }

  /** (lo, hi] becomes a history band, merged with its neighbours. */
  private addBand(lo: number, hi: number): void {
    if (!(hi > lo)) return;
    const out: [number, number][] = [];
    let cur: [number, number] = [lo, hi];
    for (const b of this.bands) {
      if (b[1] < cur[0] || b[0] > cur[1]) out.push(b);
      else cur = [Math.min(b[0], cur[0]), Math.max(b[1], cur[1])];
    }
    out.push(cur);
    out.sort((a, b) => a[0] - b[0]);
    // bounded: past MAX_BANDS the two oldest merge (the stretch between them counts as history too)
    while (out.length > MAX_BANDS) out.splice(0, 2, [out[0][0], out[1][1]]);
    this.bands = out;
  }

  /** One frame: advance the clock, step the camera (always: its step ends a drag pause), hand out the pose. */
  frame(nowMs: number, dt: number): FrameOut | null {
    if (!this.clock || !this.state || this.pausedFlag) return null;
    this.clock.setNow(this.sceneNow(nowMs));
    this.clock.tick(dt);
    const st = this.clock.lastStep;
    this.acc = this.acc ? { from: this.acc.from, to: st.to, continuous: this.acc.continuous && st.continuous } : { ...st };
    this.skipped(st);
    const rel = this.clock.rel;
    const moved = this.cam.step(dt, { rel, nowMs });
    const cut = this.jumped;
    this.jumped = false;
    // under a user hold the pose is the user's, unless the step lifted it off the ground (review M1)
    return { rel, pose: this.cam.paused && !moved ? null : this.cam.pose(), moved, cut };
  }

  /** The clock's steps since the last call, merged (discontinuous if any was). */
  takeStep(): ClockStep {
    const rel = this.clock?.rel ?? 0;
    const s = this.acc ?? { from: rel, to: rel, continuous: true };
    this.acc = null;
    return s;
  }

  /** The shown history time, absolute (unix seconds). */
  shownAbs(): number {
    return (this.clock?.rel ?? 0) + (this.origin ?? 0);
  }

  /** Show absolute time `abs` now, as one discontinuous jump (never past nowRel − LOOK). */
  jumpAbs(abs: number): void {
    const c = this.clock;
    if (!c || this.origin === null) return;
    const to = Math.min(abs - this.origin, c.bound);
    if (to === c.rel) return;
    c.lastStep = { from: c.rel, to, continuous: false };
    c.rel = to;
    this.skipped(c.lastStep);
  }

}

/**
 * The scene's response to a frame's discontinuous step (review I1, N3): in follow mode it re-fits at the new time
 * (`refit`), then cuts the camera to that fit; cinema keeps its own rules. True when it cut (the scene applies the pose).
 */
export function cutOnJump(
  out: FrameOut | null,
  cam: { readonly cameraMode: CameraMode; cutToFit(): boolean },
  refit: () => void,
): boolean {
  if (!out?.cut || cam.cameraMode !== "follow") return false;
  refit();
  return cam.cutToFit();
}

/** Same turn times, counts and ids (the scene copy did not change). */
function sameTree(a: Tree, b: Tree): boolean {
  if (a.sessions.length !== b.sessions.length || a.start !== b.start || a.end !== b.end) return false;
  const same = (x: Session, y: Session): boolean =>
    x.id === y.id && x.turns.length === y.turns.length &&
    x.turns.every((t, i) => t.every((v, k) => typeof v === "object" || v === y.turns[i][k])) &&
    x.children.length === y.children.length && x.children.every((c, i) => same(c, y.children[i]));
  return a.sessions.every((s, i) => same(s, b.sessions[i]));
}

// ---------------- adaptive render resolution (rulings R26, R28) ----------------

/**
 * The scene is GPU fill-bound at Retina sizes (Task 9: p99 35 ms at DPR 2). The render pixel ratio starts at
 * min(devicePixelRatio, 1.5) (≤ 1 in compact mode) and adapts to the frame time:
 * - frame intervals are kept over a rolling 2 s window; a decision needs a full window since the last change, and the
 *   window's p90 is computed at most every 250 ms (never per frame);
 * - p90 above 20 ms: one step down (0.25), never below 1.0 (or the start, if lower);
 * - p90 at most 17.5 ms (frames meeting 60 Hz; R28: with vsync a 60 Hz display never reports less than 16.7 ms)
 *   for 5 s in a row: one step up, never above the start;
 * - hysteresis: the 17.5–20 ms dead band, plus a back-off: after stepping down from a ratio, it is not tried again for
 *   30 s, doubling on every further step down (to 10 min), so a scene sitting near the edge never flips back and forth;
 * - an interval over 250 ms is a stall (a GC, a tab switch the scene did not see): that sample is dropped, it neither
 *   resets the window nor counts toward a step down;
 * - with the start at or below the floor (nothing to adapt), it does nothing at all.
 */
export const PIXEL_RATIO = {
  MAX: 1.5,
  FLOOR: 1,
  STEP: 0.25,
  WINDOW_MS: 2000,
  DOWN_P90_MS: 20,
  UP_P90_MS: 17.5,
  UP_HOLD_MS: 5000,
  BACKOFF_MS: 30_000,
  BACKOFF_MAX_MS: 600_000,
  STALL_MS: 250,
  EVAL_MS: 250,
} as const;

export class PixelRatioGovernor {
  ratio: number;
  readonly ceiling: number;
  readonly floor: number;
  private samples: { t: number; ms: number }[] = [];
  private since: number | null = null;
  private lastEval = -Infinity;
  private goodSince: number | null = null;
  private blockedFrom = Infinity;
  private blockedUntil = -Infinity;
  private backoff: number = PIXEL_RATIO.BACKOFF_MS;

  constructor(devicePixelRatio: number, compact = false) {
    this.ceiling = Math.min(devicePixelRatio || 1, compact ? 1 : PIXEL_RATIO.MAX);
    this.floor = Math.min(PIXEL_RATIO.FLOOR, this.ceiling);
    this.ratio = this.ceiling;
  }

  private restart(t: number) {
    this.samples = [];
    this.since = t;
    this.goodSince = null;
  }

  /** One frame interval `ms`, ending at wall time `t` (ms). Returns the new ratio when it changes, else null. */
  frame(t: number, ms: number): number | null {
    if (this.floor >= this.ceiling) return null; // pinned at both limits: nothing to adapt
    if (ms > PIXEL_RATIO.STALL_MS) return null; // a stall: dropped, not counted
    this.since ??= t;
    this.samples.push({ t, ms });
    if (t - this.since < PIXEL_RATIO.WINDOW_MS || t - this.lastEval < PIXEL_RATIO.EVAL_MS) return null;
    this.lastEval = t;
    let k = 0;
    while (k < this.samples.length && this.samples[k].t < t - PIXEL_RATIO.WINDOW_MS) k++;
    if (k) this.samples.splice(0, k);
    const sorted = this.samples.map((s) => s.ms).sort((a, b) => a - b);
    const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))];
    if (p90 > PIXEL_RATIO.DOWN_P90_MS) {
      this.goodSince = null;
      if (this.ratio <= this.floor) return null;
      this.blockedFrom = this.ratio;
      this.blockedUntil = t + this.backoff;
      this.backoff = Math.min(this.backoff * 2, PIXEL_RATIO.BACKOFF_MAX_MS);
      this.ratio = Math.max(this.floor, this.ratio - PIXEL_RATIO.STEP);
      this.restart(t);
      return this.ratio;
    }
    if (p90 > PIXEL_RATIO.UP_P90_MS) return void (this.goodSince = null), null;
    this.goodSince ??= t;
    if (t - this.goodSince < PIXEL_RATIO.UP_HOLD_MS || this.ratio >= this.ceiling) return null;
    const next = Math.min(this.ceiling, this.ratio + PIXEL_RATIO.STEP);
    if (next >= this.blockedFrom && t < this.blockedUntil) return null;
    this.ratio = next;
    this.restart(t);
    return this.ratio;
  }
}
