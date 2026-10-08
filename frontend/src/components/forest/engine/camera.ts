/**
 * README §4.5 / §4.7, spec §10.6 — the forest cameras (live view only). Ported from forest-real.html (`safeRect`,
 * `setCamGoal`, `LOOK`, `events`, `aimFor`, `damp`, `crownOf`, `stepCinema`, `stepCamera`). No three.js here: `V3` is a
 * plain {x, y, z} and the projection is a pinhole camera written out; the scene copies `pose()` onto its camera.
 *
 * Two cameras share one state (position, velocity, target, target velocity), so the cinema ↔ follow toggle carries
 * velocity and nothing ever jumps:
 * - cinema (default): the scene is shown LOOK s late, so every turn in the next LOOK s is known and the path is planned
 *   (`planShot`): one or two legs (tree, arrive, leave) that show the most growth the camera can reach in time. It
 *   commits to a leg (no change in transit, a visit lasts ≥ MIN_VISIT_S unless its tree runs dry, a better plan must
 *   win by REPLAN_MARGIN, replans ≤ 1 per REPLAN_INTERVAL_S), and after WATCH_MAX_S on one tree other busy trees earn a
 *   bonus so both are seen. It orbits the chosen tree outside its crown, steered by accelerations integrated in fixed
 *   substeps (speed ≤ V_MAX, acceleration ≤ A_MAX at any frame rate). Crowns in the way are cleared by a soft lift
 *   inside the spring (the height goal rises; the approach slows until the camera is above).
 * - follow: the wide fit of the growing front; it damps target, distance, azimuth and elevation separately, so it
 *   sweeps around instead of cutting through the scene.
 * A user drag pauses whichever is active until 9 s after the drag ends.
 *
 * Ground (follow-up A): after every update (a step in any mode, a planned return, a drag or zoom pose, a hand-off
 * import, a reset) the camera is held ≥ ground + GROUND_CLEARANCE and its target ≥ ground (`clampToGround`, with the
 * land's own height function from `setGround`). The clamp only lifts, straight up, so it is continuous: a path that
 * moves ≤ s per frame stays within s plus the ground's rise under it. "Reset view" (`resetView`, follow-up B) ends any
 * user override and returns, by the ≤ RETURN_S planned glide, to the follow fit or to the cinematic planner's shot.
 */
import { LOOK } from "./clock";

export { ForestClock, LOOK, CATCH_UP_MAX } from "./clock";
export type { ClockStep } from "./clock";

export interface V3 {
  x: number;
  y: number;
  z: number;
}

/** A tree's crown as the camera sees it: the axis at its current place (x, z), horizontal reach r and top height. */
export interface Crown {
  x: number;
  z: number;
  r: number;
  top: number;
}

/** One turn (mockup `EV` entries). */
export interface CamEvent {
  time: number;
  sessionId: string;
  treeId: string;
  /** How much growth the turn shows (e.g. its turn budget); default 1. */
  weight?: number;
}

/** Safe area in NDC (−1…1, y up), as the mockup's `safeRect`. */
export interface SafeRect {
  l: number;
  r: number;
  t: number;
  b: number;
}

/** A panel's client rect in CSS px (DOMRect-compatible). */
export interface PanelRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export type CameraMode = "cinema" | "follow";

/** The ground's height at (x, z) (land.ts `groundHeight` over the shown spit). */
export type GroundFn = (x: number, z: number) => number;
/** The camera stays at least this far above the ground (units). */
export const GROUND_CLEARANCE = 0.5;
/** Before the scene hands over its land: a flat ground at the grass's top. */
const FLAT_GROUND: GroundFn = () => 0.04;

/**
 * The ground clamp: `p` lifted straight up to ground + `clearance` if it is below (a copy; `p` unchanged otherwise).
 * Lifting only (never sideways) keeps it continuous: two points `s` apart clamp to points at most `s` plus the ground's
 * rise between them apart.
 */
export function clampToGround(p: Readonly<V3>, ground: GroundFn, clearance = GROUND_CLEARANCE): V3 {
  const g = ground(p.x, p.z) + clearance;
  return { x: p.x, y: Number.isFinite(g) && p.y < g ? g : p.y, z: p.z };
}

/** Orbit radius = crown radius + this (± breathing). */
export const CROWN_CLEARANCE = 3.2;
/** A drag (or wheel, click) pauses the active camera until this long after it ends. */
export const FOLLOW_PAUSE_MS = 9000;
/** Lingers this long on what just grew. */
export const LINGER_MS = 4000;
/** Cinema: speed and acceleration limits (units/s, units/s²). 5.5 u/s is < 0.1 units per frame at 60 fps. */
export const V_MAX = 5.5;
export const A_MAX = 6;
/** A new shot plan replaces the committed one only if it shows this much more growth (fraction). */
export const REPLAN_MARGIN = 0.5;
/** Planning: cruise speed for travel-time estimates (the radial approach cap), plus this for speeding up and
 * slowing down (s). */
export const TRAVEL_PAD = 2;
/** Planning: after a leg's last watched turn, stay this long (s) to see it grow before moving on. */
export const LEAVE_AFTER = 4;
/** Fairness: after the camera has watched one tree this long (s), growth on other trees earns a bonus… */
export const WATCH_MAX_S = 90;
/** …rising smoothly over this long (s)… */
export const FAIR_RAMP_S = 30;
/** …to this weight multiplier: enough that a far tree's few reachable turns beat a full window here by REPLAN_MARGIN.
 * A tree that is the only one growing keeps the camera indefinitely (nothing else earns the bonus). */
export const FAIR_BOOST_MAX = 6;
/** Once arrived, a visit lasts at least this long (s), unless its tree has no turn left in the window and another
 * tree has growth. Planned legs never leave earlier either. */
export const MIN_VISIT_S = 30;
/** At most one replan per this long (s). */
export const REPLAN_INTERVAL_S = 1;
/**
 * A focused session (an in-flight row click) is released once it has had no activity in the last LOOK + FOCUS_IDLE_S
 * of the server's time: no turn from 10 s before the shown time onward, and the click itself (counted as activity at
 * the server's time it happened, shown time + LOOK) more than LOOK + 10 s ago. So a click holds at least LOOK + 10 s
 * (the clicked row's own request is still in flight and has no turn yet), and longer while the session grows.
 */
export const FOCUS_IDLE_S = 10;
/** Planning: only the trees with the most growth in the window are considered. */
const PLAN_TREES = 6;
/** Arrived = within the crown's widest orbit radius (crown.r + 3.2 + 1.5 breathing) plus this. */
export const ARRIVE_MARGIN = 2;
/** The look direction turns at most this fast (rad/s, ≈ 69°/s), easing in over LOOK_TAU. */
export const LOOK_RATE = 1.2;
/** Speed above V_MAX (carried in from the follow view) decays with this time constant plus A_MAX: ≤ 0.5 s from 200 u/s. */
export const BRAKE_TAU = 0.08;
/** Slightly isometric, from the south (mockup `ISO`). */
export const ISO: Readonly<V3> = Object.freeze(norm({ x: 0.1, y: 0.22, z: 1 }));

// cinema tuning
const ORBIT_RATE = 0.07; // rad/s
const V_RADIAL = 4.5; // approach speed toward the orbit radius
const T_RADIAL = 2;
const V_RISE = 2; // climb speed
const T_RISE = 1.2;
const TAU_TAN = 0.8; // tangential speed settles with this time constant
const LIFT_SOFT = 1.2; // a crown's protected radius: r + this
const LIFT_CLEAR = 1; // pass this far above a crown's top
const CROWN_RAMP = 6; // a reset glide's lift eases in over this many units outside a crown's protected radius
const LOOKAHEAD = 15; // units of path checked for crowns
const ORBIT_LOOK = 8; // units of orbit arc checked for crowns
const LOOK_TAU = 0.4;
const SUBSTEP = 1 / 120;
const MAX_DT = 0.5;
// follow tuning
const FOLLOW_T = 1.2;
const AZ_RATE = 1; // rad/s
const EL_RATE = 0.5; // rad/s
const DIST_RATE = 40; // units/s
/**
 * A return to the follow fit (the 9 s hold after a pan ends, the Cinematic → Wide toggle, re-fits just after a cut)
 * arrives in this many seconds whatever the distance: a cubic Hermite path from the camera's state (position and
 * velocity) to the fit at rest, in the follow's own coordinates (the target moves straight; azimuth, elevation and
 * distance about it), so it sweeps around instead of cutting through the trees, never steps (position and velocity are
 * continuous) and does not overshoot from rest. Its speed scales with the distance (peak 1.5·d/RETURN_S). Steady live
 * motion keeps the V_MAX / A_MAX limits (0.1 per frame).
 */
export const RETURN_S = 2.4;
/** After a cut, re-fits during this long (s) are returns too (trees built at the new time land and move the fit). */
const CUT_ARM_S = 4;
/** A fit that moves less than this (units) during a return does not re-plan it. */
const REPLAN_EPS = 0.5;
/**
 * A re-plan (the fit moved during a return) keeps the return's deadline: it runs over the time left, at least this
 * long (s), and never past RETURN_LIMIT_S from the return's start, however often the fit moves.
 */
const REPLAN_FLOOR_S = 0.4;
export const RETURN_LIMIT_S = 3;
/** A hand-off's settle onto the receiving fit takes at most this long (s; `settleToFit`). */
export const SETTLE_MAX_S = 9.6;

/**
 * The start rate of one channel of a Hermite path from `a` to `b` over `T`, clamped so the path never passes `b`: a
 * cubic Hermite ending at rest overshoots iff it starts toward `b` faster than 3·(b − a)/T. A rate away from `b` is
 * kept (the path turns back, then approaches; it does not pass `b`).
 */
export function noOvershootRate(a: number, b: number, v: number, T: number): number {
  const lim = (3 * (b - a)) / T;
  return lim >= 0 ? Math.min(v, lim) : Math.max(v, lim);
}

const v3 = (x: number, y: number, z: number): V3 => ({ x, y, z });
const copy = (a: Readonly<V3>): V3 => ({ x: a.x, y: a.y, z: a.z });
const sub = (a: V3, b: V3): V3 => v3(a.x - b.x, a.y - b.y, a.z - b.z);
const add = (a: V3, b: V3): V3 => v3(a.x + b.x, a.y + b.y, a.z + b.z);
const mul = (a: Readonly<V3>, k: number): V3 => v3(a.x * k, a.y * k, a.z * k);
const dot = (a: V3, b: V3) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: V3, b: V3): V3 => v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const len = (a: V3) => Math.hypot(a.x, a.y, a.z);
const lerp = (a: V3, b: V3, k: number): V3 => v3(a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, a.z + (b.z - a.z) * k);
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const wrapPi = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
function norm(a: V3): V3 {
  const l = Math.hypot(a.x, a.y, a.z) || 1;
  return { x: a.x / l, y: a.y / l, z: a.z / l };
}
const set = (o: V3, a: V3) => {
  o.x = a.x;
  o.y = a.y;
  o.z = a.z;
};

/** Mockup `damp`: one axis of a critically damped spring (Game Programming Gems 4 smooth-damp). */
function damp1(cur: number, goal: number, v: number, T: number, dt: number): [number, number] {
  const w = 2 / T, x = w * dt, ex = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const ch = cur - goal, tmp = (v + ch * w) * dt;
  return [goal + (ch + tmp) * ex, (v - tmp * w) * ex];
}

/** `damp1` that settles at no more than `vmax`: the goal is held at most vmax·T away (Unity SmoothDamp's maxSpeed). */
function dampLim(cur: number, goal: number, v: number, T: number, dt: number, vmax: number): [number, number] {
  const D = vmax * T;
  return damp1(cur, cur + clamp(goal - cur, -D, D), v, T, dt);
}

/**
 * Follow's hard limits (the cinema's V_MAX and A_MAX): the step from `from` to the spring's `to` becomes a move at
 * ≤ V_MAX, changing velocity by ≤ A_MAX·dt from `v0`, and braking in time for `goal` (speed ≤ √(2·A_MAX·distance
 * left)), so a far re-fit never jumps and never overshoots. Returns the new point and velocity.
 */
export function limitMove(from: V3, to: V3, v0: V3, goal: V3 | null, dt: number, vmax = V_MAX, amax = A_MAX): [V3, V3] {
  if (!(dt > 0)) return [copy(from), copy(v0)];
  let v = mul(sub(to, from), 1 / dt);
  const dv = sub(v, v0), a = len(dv) / dt;
  if (a > amax) v = add(v0, mul(dv, amax / a));
  const cap = goal ? Math.min(vmax, Math.sqrt(2 * amax * len(sub(goal, from)))) : vmax;
  const sp = len(v);
  if (sp > cap) v = mul(v, cap / sp);
  return [add(from, mul(v, dt)), v];
}

/** Mockup `damp`: moves `cur` toward `goal` with velocity `vel` (both updated in place). No overshoot from rest. */
export function damp(cur: V3, goal: V3, vel: V3, T: number, dt: number): void {
  [cur.x, vel.x] = damp1(cur.x, goal.x, vel.x, T, dt);
  [cur.y, vel.y] = damp1(cur.y, goal.y, vel.y, T, dt);
  [cur.z, vel.z] = damp1(cur.z, goal.z, vel.z, T, dt);
}

/** Index of the first item with key > t in an ascending list (length if none). */
function firstAfter<T>(items: readonly T[], t: number, key: (x: T) => number): number {
  let lo = 0, hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(items[mid]) > t) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** Where a tree stands at a time: its place and scale (Timeline `Place`). */
export type BaseOf = (treeId: string, time: number) => { x: number; z: number; s: number } | undefined;

export interface Aim {
  p: V3;
  /** Spring time: ~45 % of the time left (≥ 1.5 s), so the camera arrives as growth starts. */
  T: number;
  treeId: string;
  sessionId: string;
  time: number;
}

/**
 * Mockup `stepCinema` look-ahead + `aimFor`: the next event within LOOK s of `rel` (events ascending by time). The aim
 * is the session's last tip, or the tree base (place, 2.5 × scale high) for a new tree. With `treeId`, only that
 * tree's events count.
 */
export function lookAheadAim(
  events: readonly CamEvent[],
  rel: number,
  tips: ReadonlyMap<string, V3>,
  baseOf: BaseOf,
  treeId: string,
): Aim | null {
  let i = firstAfter(events, rel, (e) => e.time);
  while (i < events.length && events[i].time - rel < LOOK && events[i].treeId !== treeId) i++;
  const nx = events[i];
  if (!nx || nx.time - rel >= LOOK) return null;
  let p = tips.get(nx.sessionId);
  if (!p) {
    const b = baseOf(nx.treeId, nx.time);
    if (!b) return null;
    p = v3(b.x, 2.5 * b.s, b.z);
  }
  return { p: copy(p), T: Math.max(1.5, (nx.time - rel) * 0.45), treeId: nx.treeId, sessionId: nx.sessionId, time: nx.time };
}

/** Pinhole projection to NDC (x, y in −1…1, y up) for a camera at `pos` looking at `target` with world up +y (−z when
 * looking straight down or up). `depth` is the distance along the view axis (≤ 0: behind the camera). */
export function project(p: V3, pos: V3, target: V3, fovDeg: number, aspect: number): { x: number; y: number; depth: number } {
  const f = norm(sub(target, pos));
  let r = cross(f, v3(0, 1, 0));
  if (len(r) < 1e-9) r = cross(f, v3(0, 0, -1));
  r = norm(r);
  const u = cross(r, f);
  const d = sub(p, pos);
  const depth = dot(d, f), th = Math.tan((fovDeg * Math.PI) / 360);
  return { x: dot(d, r) / (depth * th * aspect), y: dot(d, u) / (depth * th), depth };
}

/**
 * Mockup `safeRect`, without the DOM: the viewport minus the open panels (rects in CSS px), with a margin, in NDC.
 * Each panel cuts the one side (left, right, top or bottom) that leaves the largest free area.
 */
export function safeRectFromPanels(width: number, height: number, panels: readonly PanelRect[], margin = 28): SafeRect {
  let L = margin, R = width - margin, T = margin, B = height - margin;
  for (const p of panels) {
    if (p.right <= L || p.left >= R || p.bottom <= T || p.top >= B) continue; // outside the free area already
    const cands = [
      { L, R: Math.min(R, p.left - margin), T, B },
      { L: Math.max(L, p.right + margin), R, T, B },
      { L, R, T: Math.max(T, p.bottom + margin), B },
      { L, R, T, B: Math.min(B, p.top - margin) },
    ];
    let best = null as (typeof cands)[number] | null, area = 0;
    for (const c of cands) {
      const a = Math.max(0, c.R - c.L) * Math.max(0, c.B - c.T);
      if (a > area) [best, area] = [c, a];
    }
    if (best) ({ L, R, T, B } = best);
  }
  return { l: -1 + (2 * L) / width, r: -1 + (2 * R) / width, t: 1 - (2 * T) / height, b: 1 - (2 * B) / height };
}

/** A tree as the follow framing sees it (mockup `TREEBOX`). */
export interface FrameTree {
  x: number;
  active: boolean;
  points: readonly V3[];
}

/**
 * Mockup `setCamGoal` point selection: the trees growing near the front (`xc` = X(cut)); in a lull the newest one or
 * two near the front (a long lull frames the empty front, not a tree hours back); plus a stretch of shore.
 * With `spanX0` (follow-up C: X of the range's start, now − range) the fit frames the whole range instead: every given
 * tree standing in [spanX0, front], growing or not, and the shore from spanX0 to the front.
 */
export function framePoints(trees: readonly FrameTree[], xc: number, width: (x: number, side: -1 | 1) => number, spanX0?: number): V3[] {
  const span = spanX0 !== undefined && Number.isFinite(spanX0) && spanX0 < xc;
  let use = span ? trees.filter((t) => t.x >= spanX0 && t.x <= xc + 20) : trees.filter((t) => t.active && t.x > xc - 110);
  if (!span && !use.length) use = trees.filter((t) => t.x <= xc + 2 && t.x > xc - 60).sort((a, b) => b.x - a.x).slice(0, 2);
  const P: V3[] = [];
  for (const t of use) for (const p of t.points) P.push(copy(p));
  const xl = Math.min(span ? spanX0 : xc - 12, xc - 12, ...use.map((t) => t.x - 6));
  for (const x of [xl, (xl + xc) / 2, xc, xc + 6]) P.push(v3(x, 0, -width(x, -1)), v3(x, 0, width(x, 1)));
  return P;
}

/**
 * Horizontal path from P0 to P1 against a crown's protected circle (axis A, radius R): the distance along the path
 * at which it enters; 0 if it starts inside and heads further in; Infinity if it never enters (or is leaving).
 */
function entryAlong(p0x: number, p0z: number, p1x: number, p1z: number, A: Crown, R: number): number {
  const dx = p1x - p0x, dz = p1z - p0z, L = Math.hypot(dx, dz);
  const fx = p0x - A.x, fz = p0z - A.z;
  if (fx * fx + fz * fz < R * R) return dx * fx + dz * fz < 0 ? 0 : Infinity;
  if (L < 1e-9) return Infinity;
  const a = dx * dx + dz * dz, b = 2 * (fx * dx + fz * dz), c = fx * fx + fz * fz - R * R;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  const u = (-b - Math.sqrt(disc)) / (2 * a);
  return u >= 0 && u <= 1 ? u * L : Infinity;
}

/** One leg of a shot plan: watch `treeId` from `arriveBy` until `leaveAt`, seeing `events`. */
export interface Leg {
  treeId: string;
  arriveBy: number;
  leaveAt: number;
  events: CamEvent[];
}

export interface ShotPlan {
  legs: Leg[];
  /** Total weight of the growth the plan shows. */
  score: number;
}

/** Resolves a tree to the crown the camera orbits (or its base, for a tree not built yet). */
export type CrownOf = (treeId: string, time: number) => Crown | undefined;

export interface PlanOptions {
  /** Only plans whose first leg is this tree (the result is the best of those). */
  firstTree?: string;
  /** The tree the camera is orbiting: no travel to it, and its visit so far counts toward MIN_VISIT_S. */
  local?: string;
  /** How long the camera has been at `local` (s). */
  visitedFor?: number;
  /** Multiplies a tree's turn weights (the fairness bonus). */
  boost?: (treeId: string) => number;
}

/** First index with times[i] ≥ t (or > t when `strict`). */
function bound(times: number[], t: number, strict: boolean): number {
  let lo = 0, hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (strict ? times[mid] > t : times[mid] >= t) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/**
 * The shot planner: the best plan overall and the best plan whose first leg is `firstTree ?? local`, in one pass.
 * Per tree it keeps the window's turn times and prefix sums of their (boosted) weights, so any interval's growth is
 * two binary searches.
 */
function planCore(events: readonly CamEvent[], rel: number, from: V3, crownOf: CrownOf, o: PlanOptions) {
  const boost = o.boost ?? (() => 1);
  const local = o.local;
  const head = o.firstTree ?? local;
  const end = rel + LOOK;
  const i0 = firstAfter(events, rel, (e) => e.time), i1 = firstAfter(events, end, (e) => e.time);
  type T = { id: string; c: Crown; evs: CamEvent[]; times: number[]; W: number[] };
  const byTree = new Map<string, T>();
  for (let i = i0; i < i1; i++) {
    const e = events[i];
    let t = byTree.get(e.treeId);
    if (!t) {
      const c = crownOf(e.treeId, e.time);
      if (!c) continue;
      byTree.set(e.treeId, (t = { id: e.treeId, c, evs: [], times: [], W: [0] }));
    }
    t.evs.push(e);
    t.times.push(e.time);
    t.W.push(t.W[t.W.length - 1] + (e.weight ?? 1) * boost(e.treeId));
  }
  const total = (t: T) => t.W[t.W.length - 1];
  const trees = [...byTree.values()].sort((a, b) => total(b) - total(a)).slice(0, PLAN_TREES);
  for (const id of [head, local]) {
    const t = id === undefined ? undefined : byTree.get(id);
    if (t && !trees.includes(t)) trees.push(t);
  }
  // growth of tree t in [a, b]: [weight, count, lo, hi]
  const inRange = (t: T, a: number, b: number) => {
    const lo = bound(t.times, a, false), hi = bound(t.times, b, true);
    return { w: t.W[hi] - t.W[lo], n: hi - lo, lo, hi };
  };
  const orbitR = (c: Crown) => c.r + CROWN_CLEARANCE;
  const travel = (x: number, z: number, c: Crown) =>
    Math.max(0, Math.hypot(x - c.x, z - c.z) - orbitR(c) - ARRIVE_MARGIN) / V_RADIAL + TRAVEL_PAD;
  type Cand = { score: number; legs: { t: T; a: number; leave: number; lo: number; hi: number }[] };
  let best: Cand | null = null, bestHead: Cand | null = null;
  const better = (p: Cand, q: Cand | null) =>
    !q || p.score > q.score + 1e-9 || (Math.abs(p.score - q.score) <= 1e-9 && p.legs.length < q.legs.length);
  const offer = (p: Cand) => {
    if (better(p, best)) best = p;
    if (p.legs[0].t.id === head && better(p, bestHead)) bestHead = p;
  };
  for (const t1 of trees) {
    const isLocal = t1.id === local;
    const a1 = isLocal ? rel : rel + travel(from.x, from.z, t1.c);
    const stay = isLocal ? Math.max(0, MIN_VISIT_S - (o.visitedFor ?? 0)) : MIN_VISIT_S;
    const r1 = inRange(t1, a1, end);
    if (r1.n) offer({ score: r1.w, legs: [{ t: t1, a: a1, leave: end, lo: r1.lo, hi: r1.hi }] });
    for (const t2 of trees) {
      if (t2 === t1) continue;
      const hop = Math.max(
        TRAVEL_PAD,
        travel(t1.c.x, t1.c.z, t2.c) - Math.min(orbitR(t1.c), Math.hypot(t1.c.x - t2.c.x, t1.c.z - t2.c.z)) / V_RADIAL,
      );
      let prevLeave = -Infinity;
      for (let k = r1.lo; k < r1.hi; k++) {
        const leave = Math.max(t1.times[k] + LEAVE_AFTER, a1 + stay);
        if (leave === prevLeave || leave >= end) continue;
        prevLeave = leave;
        const w1 = inRange(t1, a1, leave), w2 = inRange(t2, leave + hop, end);
        if (!w2.n) continue;
        offer({
          score: w1.w + w2.w,
          legs: [
            { t: t1, a: a1, leave, lo: w1.lo, hi: w1.hi },
            { t: t2, a: leave + hop, leave: end, lo: w2.lo, hi: w2.hi },
          ],
        });
      }
    }
  }
  const toPlan = (c: Cand | null): ShotPlan =>
    c
      ? { score: c.score, legs: c.legs.map((l) => ({ treeId: l.t.id, arriveBy: l.a, leaveAt: l.leave, events: l.t.evs.slice(l.lo, l.hi) })) }
      : { legs: [], score: 0 };
  return { best: toPlan(best), head: toPlan(bestHead) };
}

/**
 * The cinematic shot plan (README §4.5: the scene is shown LOOK s late, so every turn in (rel, rel + LOOK] is known
 * and the path is planned instead of reacting turn by turn). Candidates are one leg (watch one tree for the whole
 * window) or two legs (watch T1 until after one of its turns, at least MIN_VISIT_S, then move once to T2). A leg sees
 * only the turns after the camera can get there at V_RADIAL (plus TRAVEL_PAD); unreachable turns count for nothing.
 * The plan that shows the most growth wins; ties go to fewer legs. Interleaved turns in far-apart trees therefore
 * become one tree watched for a stretch, then one move: a shuttle can never win, because each move forfeits the
 * turns during travel. With `firstTree`, the best plan starting there.
 */
export function planShot(events: readonly CamEvent[], rel: number, from: V3, crownOf: CrownOf, opts: PlanOptions = {}): ShotPlan {
  const r = planCore(events, rel, from, crownOf, opts);
  return opts.firstTree !== undefined ? r.head : r.best;
}

export interface StepContext {
  /** History time shown (ForestClock.rel). */
  rel: number;
  /** Wall clock, ms (performance.now()). */
  nowMs: number;
}

/** A view as a centre, a unit direction from it to the camera, and a distance (the follow fit's form). */
interface Goal {
  c: V3;
  dist: number;
  dir: V3;
}

export interface CameraOptions {
  fov: number;
  aspect: number;
  pos?: V3;
  target?: V3;
}

/**
 * The camera. Holds its own position, target and velocities; the scene sets the inputs (crowns, tips, events, the
 * newest growth point, the follow points) and applies `pose()` every frame.
 */
export class CameraController {
  private fov: number;
  private aspect: number;
  private mode: CameraMode = "cinema";
  private pos: V3;
  private vel = v3(0, 0, 0);
  private target: V3;
  private tvel = v3(0, 0, 0);
  private crowns: ReadonlyMap<string, Crown> = new Map();
  private tips: ReadonlyMap<string, V3> = new Map();
  private events: readonly CamEvent[] = [];
  private sessionTree = new Map<string, string>();
  /** Each session's newest event time. */
  private sessionLast = new Map<string, number>();
  private baseOf: BaseOf = () => undefined;
  private grow: { p: V3; since: number } | null = null;
  private focus: string | null = null;
  /** The shown time at the first step after the focusing click (null: not stepped yet). */
  private focusAt: number | null = null;
  private pausedFlag = false;
  private dragging = false;
  /** A return in progress (`RETURN_S`): its start state in follow coordinates, its start rates, the fit it aims at. */
  private glide: {
    /** Time into this segment, its length, and the time since the return started (re-plans keep its deadline). */
    t: number; T: number; elapsed: number;
    /** The return's planned length and its hard deadline from its start (RETURN_S / RETURN_LIMIT_S, or a settle's). */
    total: number; limit: number;
    tgt0: V3; tv0: V3;
    az0: number; el0: number; d0: number;
    azd0: number; eld0: number; dd0: number;
    aim: { pos: V3; c: V3 };
    /** A fixed destination (the cinematic shot of a reset); null: the live follow fit (`goal`). */
    dest: Goal | null;
  } | null = null;
  /** "Reset view" in cinematic mode: the glide to the planner's shot starts at the next step (it needs the shown time). */
  private resetPending = false;
  private ground: GroundFn = FLAT_GROUND;
  /** Seconds left in which a re-fit is a return (after a cut). */
  private armed = 0;
  /** A range switch's window (s left, follow-up C): a re-fit after its glide ended but inside it starts a return. */
  private glideUntil = 0;
  /** A view imported before the scene's first data: `placeInitial` starts there. */
  private imported: { pos: V3; target: V3 } | null = null;
  private lastUser = -Infinity;
  private goal: Goal | null = null;
  // cinema (mockup orbT, crownR, camTree)
  private orbT = 0;
  private crownR = 4;
  private camTree: string | null = null;
  private center: { x: number; z: number } | null = null;
  /** orbT when the camera arrived at camTree (null: still travelling). */
  private arrivedAt: number | null = null;
  /** The committed leg of the shot plan, and the window end (event index) it was planned with. */
  private leg: Leg | null = null;
  /** The plan's next leg: taken when `leg` ends (following the plan, not a replan). */
  private nextLeg: Leg | null = null;
  /** Time of the last turn in the window at the last look (robust to the array being pruned at the front). */
  private windowLast = -Infinity;
  private windowGrew = false;
  private lastPlanAt = -Infinity;
  private planRuns = 0;
  /** Where the camera looks (unit); rate-limited toward the target in every mode. */
  private lookDir: V3;
  /** Called whenever the mode changes, by the toggle (`setMode`) or by a row click (`focusSession`). */
  onModeChange: ((mode: CameraMode) => void) | null = null;

  constructor(opts: CameraOptions) {
    this.fov = opts.fov;
    this.aspect = opts.aspect;
    this.pos = copy(opts.pos ?? v3(0, 20, 60));
    this.target = copy(opts.target ?? v3(0, 0, 0));
    this.lookDir = this.dirToTarget();
  }

  /**
   * Jump to a new pose at rest and forget the shot in progress (the scene does this on its first data, so it keeps
   * one controller for its lifetime). Mode, inputs (crowns, tips, events, base) and a user pause are kept.
   */
  reset(pos: V3, target: V3): void {
    this.pos = copy(pos);
    this.target = copy(target);
    this.vel = v3(0, 0, 0);
    this.tvel = v3(0, 0, 0);
    this.lookDir = this.dirToTarget();
    this.goal = null;
    this.grow = null;
    this.crownR = 4;
    this.camTree = null;
    this.center = null;
    this.arrivedAt = null;
    this.leg = null;
    this.nextLeg = null;
    this.windowLast = -Infinity;
    this.windowGrew = true;
    this.lastPlanAt = -Infinity;
    this.glide = null;
    this.armed = 0;
    this.glideUntil = 0;
    this.resetPending = false;
    this.clampGround();
    this.lookDir = this.dirToTarget();
  }

  /** The land's height function (the scene's, over the shown spit); the camera is clamped to it from now on. */
  setGround(ground: GroundFn): void {
    this.ground = ground;
    this.clampGround();
  }

  /**
   * Holds the camera ≥ ground + GROUND_CLEARANCE and the target ≥ ground (straight up only, so continuous). A lift
   * also drops the downward part of that velocity: the springs do not keep pushing into the ground.
   */
  private clampGround(): boolean {
    let lifted = false;
    const p = clampToGround(this.pos, this.ground);
    if (p.y !== this.pos.y) {
      this.pos.y = p.y;
      if (this.vel.y < 0) this.vel.y = 0;
      lifted = true;
    }
    const t = clampToGround(this.target, this.ground, 0);
    if (t.y !== this.target.y) {
      this.target.y = t.y;
      if (this.tvel.y < 0) this.tvel.y = 0;
      lifted = true;
    }
    return lifted;
  }

  /**
   * A discontinuous jump of the shown time (a resume after a long pause or hidden tab, a stall): the viewer was not
   * watching, so a follow camera cuts straight to its fit (at rest), and re-fits in the next CUT_ARM_S (trees built at
   * the new time land) are returns (`RETURN_S`). False (nothing done) outside follow mode (cinema keeps its own rules),
   * under a user hold, or without a fit.
   */
  cutToFit(): boolean {
    const g = this.goal, p = this.followPose();
    if (!g || !p || this.mode !== "follow" || this.focus || this.pausedFlag) return false;
    this.reset(p.pos, p.target);
    this.goal = g;
    this.armed = CUT_ARM_S;
    return true;
  }

  /** The camera about its target: distance, azimuth, elevation and their rates (from the shared velocities). */
  private polar() {
    const r = sub(this.pos, this.target), rv = sub(this.vel, this.tvel);
    const d = Math.max(1e-3, len(r)), rh = Math.max(1e-3, Math.hypot(r.x, r.z));
    const rhd = (r.x * rv.x + r.z * rv.z) / rh;
    return {
      d, az: Math.atan2(r.x, r.z), el: Math.atan2(r.y, rh),
      dd: dot(r, rv) / d, azd: (r.z * rv.x - r.x * rv.z) / (rh * rh), eld: (rv.y * rh - r.y * rhd) / (d * d),
    };
  }

  /**
   * Starts a return to the current fit, arriving in RETURN_S (`replan`: the fit moved during one; the path is re-planned
   * from the current state over the time left, so the deadline holds, ≤ RETURN_LIMIT_S). Every channel's start rate is
   * clamped so its path cannot pass the fit (`noOvershootRate`): a return that starts in motion, or a late re-plan,
   * does not overshoot.
   */
  private startGlide(replan = false, dest: Goal | null = null, plan: { total: number; limit: number } | null = null): void {
    const g = dest ?? this.goal;
    if (!g || (!dest && (this.mode !== "follow" || this.focus))) return;
    const p = { pos: add(g.c, mul(g.dir, g.dist)), target: copy(g.c) };
    const elapsed = replan && this.glide ? this.glide.elapsed : 0;
    // a re-plan keeps the return's own plan (a settle's long one included); a new return takes `plan` or the default
    const { total, limit } = replan && this.glide ? this.glide : (plan ?? { total: RETURN_S, limit: RETURN_LIMIT_S });
    const T = Math.max(1e-3, replan ? Math.max(total - elapsed, Math.min(REPLAN_FLOOR_S, limit - elapsed)) : total);
    const q = this.polar();
    const gAz = q.az + wrapPi(Math.atan2(g.dir.x, g.dir.z) - q.az), gEl = Math.atan2(g.dir.y, Math.hypot(g.dir.x, g.dir.z));
    const tv = this.tvel, tg = this.target;
    this.glide = {
      t: 0, T, elapsed, total, limit,
      tgt0: copy(tg),
      tv0: v3(noOvershootRate(tg.x, g.c.x, tv.x, T), noOvershootRate(tg.y, g.c.y, tv.y, T), noOvershootRate(tg.z, g.c.z, tv.z, T)),
      az0: q.az, el0: q.el, d0: q.d,
      azd0: noOvershootRate(q.az, gAz, q.azd, T), eld0: noOvershootRate(q.el, gEl, q.eld, T), dd0: noOvershootRate(q.d, g.dist, q.dd, T),
      aim: { pos: p.pos, c: p.target },
      dest,
    };
  }

  /** One frame of a return: the Hermite path toward the current fit; at its end the camera is there, at rest. */
  private stepGlide(dt: number): void {
    const G = this.glide!, g = G.dest ?? this.goal!;
    const p0 = copy(this.pos), t0 = copy(this.target);
    const step = Math.min(G.T - G.t, dt);
    G.t += step;
    G.elapsed += step;
    const T = G.T, u = G.t / T, u2 = u * u, u3 = u2 * u;
    const h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = 3 * u2 - 2 * u3;
    const H = (a: number, va: number, b: number) => h00 * a + h10 * T * va + h01 * b;
    const gAz = G.az0 + wrapPi(Math.atan2(g.dir.x, g.dir.z) - G.az0);
    const gEl = Math.atan2(g.dir.y, Math.hypot(g.dir.x, g.dir.z));
    this.target = v3(H(G.tgt0.x, G.tv0.x, g.c.x), H(G.tgt0.y, G.tv0.y, g.c.y), H(G.tgt0.z, G.tv0.z, g.c.z));
    const az = H(G.az0, G.azd0, gAz), el = H(G.el0, G.eld0, gEl), dist = Math.max(1e-3, H(G.d0, G.dd0, g.dist));
    const ce = Math.cos(el);
    this.pos = add(this.target, mul(v3(ce * Math.sin(az), Math.sin(el), ce * Math.cos(az)), dist));
    // a reset's glide to the cinematic shot (review I1): lifted over every other crown on its way, as the cinema's own
    // flight is (the destination's crown is where it arrives, beside it)
    if (G.dest) this.pos.y = Math.max(this.pos.y, this.crownFloor(this.pos.x, this.pos.z, this.camTree));
    this.clampGround();
    if (G.t >= T - 1e-9) {
      this.glide = null;
      this.vel = v3(0, 0, 0);
      this.tvel = v3(0, 0, 0);
    } else if (dt > 0) {
      this.vel = mul(sub(this.pos, p0), 1 / dt);
      this.tvel = mul(sub(this.target, t0), 1 / dt);
    }
    this.turnLook(dt);
  }

  private dirToTarget(): V3 {
    const d = sub(this.target, this.pos);
    return len(d) > 1e-9 ? norm(d) : v3(0, 0, -1);
  }

  setViewport(fov: number, aspect: number): void {
    this.fov = fov;
    this.aspect = aspect;
  }

  /** Cinematic (default) or the wide follow view. Position and velocity carry across the switch. The toggle releases a
   * focused session, in either direction. */
  setMode(mode: CameraMode): void {
    this.focus = null;
    const was = this.mode;
    this.changeMode(mode);
    if (mode === "follow" && was !== "follow") this.startGlide(); // Cinematic → Wide: a ≤ RETURN_S glide
  }

  private changeMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    // a glide belongs to the mode that started it (a follow return, or a cinematic reset)
    this.glide = null;
    this.glideUntil = 0;
    this.resetPending = false;
    this.onModeChange?.(mode);
  }

  /** The session an in-flight row click focused (null: the automatic shot). */
  get focused(): string | null {
    return this.focus;
  }

  /** How many times the shot planner has run (instrumentation; ≤ 1 per REPLAN_INTERVAL_S). */
  get plannerRuns(): number {
    return this.planRuns;
  }

  /** The tree the cinematic camera is committed to (null before the first pick). */
  get destination(): string | null {
    return this.camTree;
  }

  get cameraMode(): CameraMode {
    return this.mode;
  }

  /** Crowns by tree id, at their current place. */
  setCrowns(crowns: ReadonlyMap<string, Crown>): void {
    this.crowns = crowns;
  }

  /** Last grown tip by session id (mockup `tipOf`). */
  setTips(tips: ReadonlyMap<string, V3>): void {
    this.tips = tips;
  }

  /** Every turn, ascending by time (mockup `events()`). Any call marks the window as changed. */
  setEvents(events: readonly CamEvent[]): void {
    this.events = events;
    // new data may land anywhere in the window (or the front may be pruned): look again at the next allowed replan
    this.windowGrew = true;
    this.windowLast = -Infinity;
    this.sessionTree = new Map(events.map((e) => [e.sessionId, e.treeId]));
    this.sessionLast = new Map();
    for (const e of events) this.sessionLast.set(e.sessionId, Math.max(e.time, this.sessionLast.get(e.sessionId) ?? -Infinity));
  }

  /** Where a tree stands at a time; aims at a new tree's base when its session has no tip yet. */
  setBaseOf(baseOf: BaseOf): void {
    this.baseOf = baseOf;
  }

  /** The newest grown point (mockup `growAt`); the camera lingers ~4 s after it changes. */
  setGrowth(p: V3 | null, nowMs: number): void {
    if (!p) this.grow = null;
    else if (!this.grow || len(sub(this.grow.p, p)) > 1e-9) this.grow = { p: copy(p), since: nowMs };
  }

  /** Where the camera is and what it looks at, to copy onto the scene camera. */
  pose(): { pos: V3; target: V3 } {
    const target = add(this.pos, mul(this.lookDir, Math.max(1e-3, len(sub(this.target, this.pos)))));
    return { pos: copy(this.pos), target: clampToGround(target, this.ground, 0) };
  }

  /**
   * Sync during a user drag only: adopts the OrbitControls pose (and zeroes velocity) so the resumed camera starts
   * from it. Ignored (returns false) while the camera is driving itself, so an echoed pose cannot stall the springs.
   */
  setPose(pos: V3, target: V3): boolean {
    if (!this.pausedFlag) return false;
    set(this.pos, pos);
    set(this.target, target);
    this.vel = v3(0, 0, 0);
    this.tvel = v3(0, 0, 0);
    this.clampGround();
    this.lookDir = this.dirToTarget();
    return true;
  }

  /**
   * A view handed over from another surface (the Stats card ↔ the forest page): this pose at rest and this mode, with
   * no animation. The shot in progress and a focused session are dropped and a user hold (the 9 s grace) is released,
   * so the camera continues from exactly this pose on the next step. The pose is also kept for `placeInitial`: imported
   * before the scene's first data, it wins over the front placement. Ignored (returns false) during a live drag: the
   * user's hand on the camera wins, and the drag keeps working.
   */
  importView(pos: V3, target: V3, mode: CameraMode): boolean {
    if (this.dragging) return false;
    this.reset(pos, target);
    this.imported = { pos: copy(pos), target: copy(target) };
    this.focus = null;
    this.pausedFlag = false;
    this.changeMode(mode);
    return true;
  }

  /**
   * The scene's first data: `reset` to the front placement (`pos`, `target`), or to the view imported before it
   * (`importView`), once. Everything else of `reset` (the shot forgotten, at rest) applies either way.
   */
  placeInitial(pos: V3, target: V3): void {
    const v = this.imported;
    this.imported = null;
    this.reset(v?.pos ?? pos, v?.target ?? target);
  }

  /** The follow fit as a pose (camera at the goal, looking at the fit's centre); null before the first fit. */
  followPose(): { pos: V3; target: V3 } | null {
    const g = this.goal;
    return g ? { pos: add(g.c, mul(g.dir, g.dist)), target: copy(g.c) } : null;
  }

  /** Where the follow fit wants the camera (world); null without a fit, or outside follow mode (debug hooks). */
  /** The follow fit's distance from its centre (null without a fit). */
  fitDistance(): number | null {
    return this.goal?.dist ?? null;
  }

  followGoal(): V3 | null {
    return this.mode === "follow" && !this.focus ? this.followPose()?.pos ?? null : null;
  }

  /** True while a user drag (or its 9 s grace) holds the camera. */
  get paused(): boolean {
    return this.pausedFlag;
  }

  /** OrbitControls `start` (true) / `end` (false). No resume while a drag is active, however long it lasts. */
  setDragging(active: boolean, nowMs: number): void {
    this.dragging = active;
    this.lastUser = nowMs;
    if (active) this.pausedFlag = true;
  }

  /** A one-off interaction (wheel, click): pause for FOLLOW_PAUSE_MS from now. */
  userInteracted(nowMs: number): void {
    this.pausedFlag = true;
    this.lastUser = nowMs;
  }

  /**
   * "Reset view": ends any user override (the drag or zoom hold, a focused session) and returns to this mode's default
   * view by the planned ≤ RETURN_S glide (no jump): follow → the fit; cinema → the planner's current shot (the
   * destination tree's orbit, at the camera's bearing from it), started at the next step.
   */
  resetView(): void {
    this.focus = null;
    this.focusAt = null;
    this.pausedFlag = false;
    this.dragging = false;
    this.glide = null;
    this.resetPending = false;
    if (this.mode === "follow") this.startGlide();
    else this.resetPending = true;
  }

  /**
   * A range switch (follow-up C): the fit now frames another span; in follow mode the camera glides there by the
   * planned ≤ RETURN_S return (no jump; re-fits while it runs re-plan it within the same deadline). Nothing under a
   * user hold (the hold's own return goes to the new fit), a focused session, or in cinema (the planner's own rules).
   */
  glideToFit(): void {
    if (this.mode !== "follow" || this.focus || this.pausedFlag) return;
    this.startGlide(this.glide !== null);
    this.glideUntil = this.glide ? this.glide.limit - this.glide.elapsed : 0;
  }

  /**
   * A hand-off landed (follow-up C): the camera continues from the sender's view, whose fit (another surface, another
   * aspect and panels) can be far from this one's. It settles on its own fit by a planned glide whose length grows with
   * the distance (T = 1.1·√(d/2), RETURN_S to SETTLE_MAX_S): from rest it moves ≤ 0.1 per frame over its first half
   * second, and arrives within SETTLE_MAX_S. Re-fits while it runs re-plan it within its own length. Follow mode only.
   */
  settleToFit(): void {
    const p = this.followPose();
    if (!p || this.mode !== "follow" || this.focus || this.pausedFlag) return;
    const T = clamp(1.1 * Math.sqrt(len(sub(p.pos, this.pos)) / 2), RETURN_S, SETTLE_MAX_S);
    this.startGlide(false, null, { total: T, limit: Math.min(T + (RETURN_LIMIT_S - RETURN_S), SETTLE_MAX_S + 0.3) });
  }

  /**
   * The height a reset's glide keeps over the crowns (except `skip`'s): top + LIFT_CLEAR inside a crown's protected
   * radius (r + LIFT_SOFT), easing (smoothstep) to nothing CROWN_RAMP units further out, so the lift is continuous.
   */
  private crownFloor(x: number, z: number, skip: string | null): number {
    let y = -Infinity;
    for (const [id, c] of this.crowns) {
      if (id === skip) continue;
      const R = c.r + LIFT_SOFT, h = Math.hypot(x - c.x, z - c.z);
      if (h >= R + CROWN_RAMP) continue;
      const u = h <= R ? 1 : 1 - (h - R) / CROWN_RAMP, w = u * u * (3 - 2 * u);
      y = Math.max(y, (c.top + LIFT_CLEAR) * w);
    }
    return y;
  }

  /** The cinematic shot a reset returns to: the destination's orbit ring, at the camera's bearing, aimed at its growth. */
  private shotGoal(rel: number): Goal | null {
    this.followPlan(rel);
    const tree = this.camTree;
    const crown = tree ? this.crownOf(tree, rel) : undefined;
    if (!tree || !crown) return null;
    const own = lookAheadAim(this.events, rel, this.tips, this.baseOf, tree);
    const c = clampToGround(own ? own.p : v3(crown.x, Math.max(1, crown.top * 0.6), crown.z), this.ground, 0);
    const R = Math.max(this.crownR, crown.r) + CROWN_CLEARANCE;
    let bx = this.pos.x - crown.x, bz = this.pos.z - crown.z;
    const bl = Math.hypot(bx, bz);
    [bx, bz] = bl > 1e-6 ? [bx / bl, bz / bl] : [0, 1];
    const at = clampToGround(v3(crown.x + bx * R, Math.max(1.2, c.y + 0.8), crown.z + bz * R), this.ground);
    const off = sub(at, c), d = len(off);
    return d > 1e-6 ? { c, dist: d, dir: mul(off, 1 / d) } : null;
  }

  /** The camera toggle pressed while paused: resume now. */
  resume(): void {
    const held = this.pausedFlag;
    this.pausedFlag = false;
    this.dragging = false;
    if (held) this.startGlide(); // back to the fit from where the user left it: a ≤ RETURN_S glide
  }

  /**
   * An in-flight row was clicked: orbit that session's tree, aiming at its tip, in cinematic mode (switching from Wide;
   * reported through `onModeChange`). Null releases it: the automatic shot. Released also when the session goes idle
   * (FOCUS_IDLE_S) and on the mode toggle.
   */
  focusSession(sessionId: string | null): void {
    this.focus = sessionId;
    this.focusAt = null;
    this.resetPending = false;
    if (sessionId !== null) this.changeMode("cinema");
    this.resume();
  }

  /**
   * Mockup `setCamGoal` solve: from the isometric direction, find the distance and centre at which every point fits
   * inside `safe` (NDC), then add 3 %. Call again whenever the points or the panels change.
   */
  follow(points: readonly V3[], safe: SafeRect): void {
    if (!points.length) return;
    const { cc, dist } = this.solveFit(points, safe);
    cc.y = clampToGround(cc, this.ground, 0).y; // the look target never under the ground
    this.goal = { c: cc, dist, dir: copy(ISO) };
    // the fit moved during a return (or just after a cut): re-plan from the current state, continuous in velocity
    const aim = this.glide?.aim, p = this.followPose()!;
    const moved = !aim || len(sub(aim.pos, p.pos)) > REPLAN_EPS || len(sub(aim.c, p.target)) > REPLAN_EPS;
    if (moved && !this.pausedFlag) {
      if (this.glide || this.armed > 0) this.startGlide(this.glide !== null);
      // a range switch's glide ended, its window not yet (trees still landing): this re-fit is a return of its own, a
      // full RETURN_S one (review M3: squeezing it into the time left made a large late move a lurch)
      else if (this.glideUntil > 0) this.startGlide();
    }
  }

  /** The fit's centre and distance for `points` inside one free area (`follow`). */
  private solveFit(points: readonly V3[], safe: SafeRect): { cc: V3; dist: number } {
    const dir = copy(ISO);
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (const p of points) {
      x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
      y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      z0 = Math.min(z0, p.z); z1 = Math.max(z1, p.z);
    }
    const cc = v3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    const extent = (c: V3, dist: number) => {
      const pos = add(c, mul(dir, dist));
      let a = Infinity, b = -Infinity, e = Infinity, f = -Infinity;
      for (const p of points) {
        const q = project(p, pos, c, this.fov, this.aspect);
        if (q.depth <= 0.1) return { x0: -Infinity, x1: Infinity, y0: -Infinity, y1: Infinity }; // behind: no fit
        a = Math.min(a, q.x); b = Math.max(b, q.x); e = Math.min(e, q.y); f = Math.max(f, q.y);
      }
      return { x0: a, x1: b, y0: e, y1: f };
    };
    const inside = (c: V3, dist: number) => {
      const e = extent(c, dist);
      return e.x0 >= safe.l && e.x1 <= safe.r && e.y0 >= safe.b && e.y1 <= safe.t;
    };
    const right = norm(cross(mul(dir, -1), v3(0, 1, 0)));
    const up = norm(cross(right, mul(dir, -1)));
    let dist = 60;
    // the search reaches far enough for a wide range's span (a fixed 400 left the recentring guessing at a wrong scale)
    const reach = Math.max(400, 4 * Math.max(x1 - x0, y1 - y0, z1 - z0));
    for (let it = 0; it < 3; it++) {
      let lo = 2, hi = reach;
      for (let k = 0; k < 22; k++) {
        const mid = (lo + hi) / 2, e = extent(cc, mid);
        if (e.x1 - e.x0 <= safe.r - safe.l && e.y1 - e.y0 <= safe.t - safe.b) hi = mid;
        else lo = mid;
      }
      dist = hi;
      const e = extent(cc, dist);
      const dx = (e.x0 + e.x1 - (safe.l + safe.r)) / 2, dy = (e.y0 + e.y1 - (safe.b + safe.t)) / 2;
      const halfH = Math.tan((this.fov * Math.PI) / 360) * dist, halfW = halfH * this.aspect;
      set(cc, add(cc, add(mul(right, dx * halfW), mul(up, dy * halfH))));
    }
    dist *= 1.03;
    // the recentring is approximate; back off until every point is really inside
    for (let k = 0; k < 200 && !inside(cc, dist); k++) dist *= 1.02;
    return { cc, dist };
  }

  /** Advance one frame. Returns true when the pose changed. */
  step(dt: number, ctx: StepContext): boolean {
    if (this.pausedFlag) {
      if (!this.dragging && ctx.nowMs - this.lastUser > FOLLOW_PAUSE_MS) {
        this.pausedFlag = false;
        this.startGlide(); // the hold after a pan ended: back to the fit in ≤ RETURN_S
      }
      // the user's hold: the pose stays, except that the ground rising under it lifts it (review M1)
      else return this.clampGround();
    }
    const before = this.pose();
    dt = Math.min(dt, MAX_DT);
    if (this.focus !== null) {
      this.focusAt ??= ctx.rel;
      const active = Math.max(this.sessionLast.get(this.focus) ?? -Infinity, this.focusAt + LOOK);
      if (!(active >= ctx.rel - FOCUS_IDLE_S)) this.focus = null;
    }
    if (this.resetPending && this.mode === "cinema" && !this.focus) {
      this.resetPending = false;
      const shot = this.shotGoal(ctx.rel);
      if (shot) this.startGlide(false, shot);
    }
    if (this.glide?.dest && this.mode === "cinema" && !this.focus) {
      // a reset's glide to the planner's shot; the cinema takes over from there, at rest
      this.orbT += dt;
      this.stepGlide(dt);
      if (!this.glide && this.camTree) {
        const c = this.crownOf(this.camTree, ctx.rel);
        if (c) this.center = { x: c.x, z: c.z };
      }
    } else if (this.mode === "cinema" || this.focus) this.stepCinemaAuto(dt, ctx);
    else this.stepFollow(dt);
    this.clampGround();
    return len(sub(before.pos, this.pos)) > 1e-4 || len(sub(before.target, this.target)) > 1e-4;
  }

  /** Turn the look direction toward the target: eases in (LOOK_TAU), never faster than LOOK_RATE. */
  private turnLook(dt: number): void {
    const d = this.dirToTarget(), l = this.lookDir;
    const th = Math.acos(clamp(dot(l, d), -1, 1));
    if (th < 1e-9) return void (this.lookDir = d);
    const stepA = Math.min(LOOK_RATE * dt, th * (1 - Math.exp(-dt / LOOK_TAU)));
    const sn = Math.sin(th);
    if (sn < 1e-6) {
      // opposite: turn about the vertical (or any perpendicular) axis
      let ax = cross(l, v3(0, 1, 0));
      if (len(ax) < 1e-6) ax = cross(l, v3(1, 0, 0));
      const perp = norm(cross(norm(ax), l));
      this.lookDir = norm(add(mul(l, Math.cos(stepA)), mul(perp, Math.sin(stepA))));
      return;
    }
    this.lookDir = norm(add(mul(l, Math.sin(th - stepA) / sn), mul(d, Math.sin(stepA) / sn)));
  }

  /**
   * Mockup `stepCamera` (follow branch). The mockup lerped direction and distance about the target; here the target,
   * distance, azimuth and elevation each follow a critically damped spring (turn rate ≤ AZ_RATE / EL_RATE), derived
   * every frame from the shared position/velocity state. The springs' result is then held to the cinema's hard limits
   * (`limitMove`: ≤ V_MAX, ≤ A_MAX, braking for the fit), for the camera and its target alike, so a far re-fit (a
   * hand-off from the card's 6 h layout to the full window's 48 h one, the toggle from a distant orbit) glides.
   */
  private stepFollow(dt: number): void {
    this.armed = Math.max(0, this.armed - dt);
    this.glideUntil = Math.max(0, this.glideUntil - dt);
    if (this.glide && this.goal) return this.stepGlide(dt);
    const p0 = copy(this.pos), t0 = copy(this.target), v0 = copy(this.vel), tv0 = copy(this.tvel);
    const r = sub(this.pos, this.target), rv = sub(this.vel, this.tvel);
    const d = Math.max(1e-3, len(r)), rh = Math.max(1e-3, Math.hypot(r.x, r.z));
    let az = Math.atan2(r.x, r.z), el = Math.atan2(r.y, rh);
    let dd = dot(r, rv) / d;
    let azd = (r.z * rv.x - r.x * rv.z) / (rh * rh);
    const rhd = (r.x * rv.x + r.z * rv.z) / rh;
    let eld = (rv.y * rh - r.y * rhd) / (d * d);
    const g = this.goal;
    const gc = g ? g.c : copy(this.target);
    const gDist = g ? g.dist : d;
    const gAz = g ? az + wrapPi(Math.atan2(g.dir.x, g.dir.z) - az) : az;
    const gEl = g ? Math.atan2(g.dir.y, Math.hypot(g.dir.x, g.dir.z)) : el;
    damp(this.target, gc, this.tvel, FOLLOW_T, dt);
    let dist: number;
    [dist, dd] = dampLim(d, gDist, dd, FOLLOW_T, dt, DIST_RATE);
    [az, azd] = dampLim(az, gAz, azd, FOLLOW_T, dt, AZ_RATE);
    [el, eld] = dampLim(el, gEl, eld, FOLLOW_T, dt, EL_RATE);
    const ce = Math.cos(el), se = Math.sin(el), ca = Math.cos(az), sa = Math.sin(az);
    const dir = v3(ce * sa, se, ce * ca);
    const dAz = v3(ce * ca, 0, -ce * sa), dEl = v3(-se * sa, ce, -se * ca);
    this.pos = add(this.target, mul(dir, dist));
    [this.target, this.tvel] = limitMove(t0, this.target, tv0, g ? g.c : null, dt);
    [this.pos, this.vel] = limitMove(p0, this.pos, v0, g ? add(g.c, mul(g.dir, g.dist)) : null, dt);
    this.turnLook(dt);
  }

  /**
   * Mockup `stepCinema`: follows the shot plan (`planShot`) and aims at the committed tree's next growth. The camera
   * commits to its leg: it replans only when the leg is done (its time is up, or its tree has no turn left in the
   * window), or when new turns enter the window and a fresh plan shows REPLAN_MARGIN more growth than the best plan
   * that keeps the current tree. While in transit (not yet arrived) the destination never changes. An empty window
   * keeps the last tree. After WATCH_MAX_S on one tree, other trees' growth earns a rising bonus (fairBoost), so with
   * two busy trees the viewer sees both.
   */
  private stepCinemaAuto(dt: number, ctx: StepContext): void {
    const pick = (tree: string | null) => {
      if (tree !== this.camTree) {
        this.camTree = tree;
        this.arrivedAt = null;
      }
    };
    let aim: { p: V3; T: number } | null = null;
    if (this.focus) {
      const tip = this.tips.get(this.focus);
      if (tip) aim = { p: copy(tip), T: 2 };
      pick(this.sessionTree.get(this.focus) ?? this.camTree);
    } else {
      this.followPlan(ctx.rel);
      const own = this.camTree ? lookAheadAim(this.events, ctx.rel, this.tips, this.baseOf, this.camTree) : null;
      if (own) aim = { p: own.p, T: own.T };
    }
    // a tree that is not built yet orbits its place base, so the camera flies there before its first turn
    const crown = this.camTree ? this.crownOf(this.camTree, ctx.rel) ?? null : null;
    // linger on what just grew, if it grew on this tree
    const growHere = this.grow && (!crown || Math.hypot(this.grow.p.x - crown.x, this.grow.p.z - crown.z) < crown.r + 5);
    if (!aim && this.grow && growHere) aim = { p: copy(this.grow.p), T: 3 };
    if (aim && !this.focus && this.grow && growHere && ctx.nowMs - this.grow.since < LINGER_MS) aim.p = lerp(aim.p, this.grow.p, 0.5);
    this.cinema(dt, aim, crown);
    if (crown && this.arrivedAt === null) {
      const arriveR = crown.r + CROWN_CLEARANCE + 1.5 + ARRIVE_MARGIN;
      if (Math.hypot(this.pos.x - crown.x, this.pos.z - crown.z) <= arriveR) this.arrivedAt = this.orbT;
    }
  }

  private crownOf: CrownOf = (id, time) => {
    const c = this.crowns.get(id);
    if (c) return c;
    const b = this.baseOf(id, time);
    return b ? { x: b.x, z: b.z, r: 1.5, top: 1 } : undefined;
  };

  /** Fairness bonus for trees other than the one watched for more than WATCH_MAX_S (smoothstep ramp). */
  private fairBoost(): (treeId: string) => number {
    const watched = this.arrivedAt === null ? 0 : this.orbT - this.arrivedAt;
    const k = clamp((watched - WATCH_MAX_S) / FAIR_RAMP_S, 0, 1);
    const b = 1 + (FAIR_BOOST_MAX - 1) * k * k * (3 - 2 * k);
    const here = this.camTree;
    return (id) => (id === here ? 1 : b);
  }

  private followPlan(rel: number): void {
    const end = firstAfter(this.events, rel + LOOK, (e) => e.time);
    const lastT = end > 0 ? this.events[end - 1].time : -Infinity;
    if (lastT > this.windowLast) this.windowGrew = true;
    this.windowLast = lastT;
    const due = this.orbT - this.lastPlanAt >= REPLAN_INTERVAL_S;
    const plan = (o: PlanOptions) => {
      this.planRuns++;
      this.lastPlanAt = this.orbT;
      this.windowGrew = false;
      return planCore(this.events, rel, this.pos, this.crownOf, o);
    };
    const adopt = (l: Leg, next: Leg | null = null) => {
      if (l.treeId !== this.camTree) {
        this.camTree = l.treeId;
        this.arrivedAt = null;
      }
      this.leg = l;
      this.nextLeg = next;
    };
    if (!this.camTree) {
      if (!due) return;
      const r = plan({});
      if (r.best.legs.length) return adopt(r.best.legs[0], r.best.legs[1] ?? null);
      // nothing reachable: the first turn in the window, else the last one seen
      const i = firstAfter(this.events, rel, (e) => e.time);
      const e = this.events[i] && this.events[i].time <= rel + LOOK ? this.events[i] : this.events[i - 1];
      if (e) this.camTree = e.treeId;
      return;
    }
    if (this.arrivedAt === null) return; // in transit: the destination holds
    // the tree still has growth to show: a turn ahead in the window, or one that played within LEAVE_AFTER (still growing)
    let ownLeft = false;
    for (let i = firstAfter(this.events, rel - LEAVE_AFTER, (e) => e.time); i < end && !ownLeft; i++) ownLeft = this.events[i].treeId === this.camTree;
    // its last turn is still growing: one landed within LEAVE_AFTER and none is ahead. The planner only sees turns
    // ahead, so a replan would leave mid-growth; hold until LEAVE_AFTER has passed (planned legs already leave then)
    let landed = false, ahead = false;
    for (let i = firstAfter(this.events, rel - LEAVE_AFTER, (e) => e.time); i < end; i++)
      if (this.events[i].treeId === this.camTree) {
        if (this.events[i].time <= rel) landed = true;
        else ahead = true;
      }
    const lastGrowing = landed && !ahead;
    const visited = this.orbT - this.arrivedAt;
    // the 30 s floor protects watching growth, not empty time: a dry tree may be left early if another tree grows
    const mayMove = !lastGrowing && (!ownLeft || visited >= MIN_VISIT_S);
    const legDone = !this.leg || rel >= this.leg.leaveAt || !ownLeft;
    // the leg ended as planned: take the plan's next leg, if its tree still has turns ahead
    const nx = this.nextLeg;
    if (legDone && nx && mayMove) {
      for (let i = firstAfter(this.events, rel, (e) => e.time); i < end; i++)
        if (this.events[i].treeId === nx.treeId) return adopt(nx);
    }
    if (!(legDone || this.windowGrew) || !due) return;
    // the orbited tree is local (no travel, its visit so far counts); a move must beat staying by REPLAN_MARGIN and
    // may only happen after MIN_VISIT_S, unless this tree has run dry
    const r = plan({ firstTree: this.camTree, local: this.camTree, visitedFor: visited, boost: this.fairBoost() });
    const go = r.best.legs[0];
    if (mayMove && go && go.treeId !== this.camTree && r.best.score > r.head.score * (1 + REPLAN_MARGIN) + 1e-9)
      adopt(go, r.best.legs[1] ?? null);
    else if (r.head.legs.length) {
      // keeping the same planned move must not postpone it (each new turn would push "leave after its next turn")
      const same = this.leg && this.nextLeg && rel < this.leg.leaveAt && r.head.legs[1]?.treeId === this.nextLeg.treeId;
      if (!same) adopt(r.head.legs[0], r.head.legs[1] ?? null);
    }
    else this.leg = this.nextLeg = null; // no plan: hold the tree; retried at most once per REPLAN_INTERVAL_S
  }

  /**
   * One frame of the crown orbit (mockup `stepCinema` tail). The aim follows a critically damped spring with time
   * `aim.T`. The camera circles the crown axis at max(crownR, crown.r) + 3.2 (± 1.5 breathing, pulled back while the
   * aim travels), at the growth height, 0.07 rad/s, and lifts over any crown in its way (soft, inside the spring).
   * Safe to call directly; `step()` calls it in cinema mode.
   */
  cinema(dt: number, aim: { p: V3; T: number } | null, crown: Crown | null): void {
    dt = Math.min(dt, MAX_DT);
    this.orbT += dt;
    damp(this.target, aim ? aim.p : copy(this.target), this.tvel, aim?.T ?? 3, dt);
    if (crown) {
      this.crownR += (crown.r - this.crownR) * Math.min(1, dt * 0.5);
      this.center = { x: crown.x, z: crown.z };
    } else if (!this.center) this.center = { x: this.target.x, z: this.target.z };
    const n = Math.max(1, Math.ceil(dt / SUBSTEP - 1e-9)), h = dt / n;
    for (let i = 0; i < n; i++) this.cinemaSubstep(h, crown, this.orbT - dt + (i + 1) * h);
    this.clampGround();
    this.turnLook(dt);
  }

  private cinemaSubstep(h: number, crown: Crown | null, t: number): void {
    const C = this.center!;
    const p = this.pos, v = this.vel;
    const dx = p.x - C.x, dz = p.z - C.z, rho = Math.hypot(dx, dz);
    const erx = rho > 1e-9 ? dx / rho : 1, erz = rho > 1e-9 ? dz / rho : 0;
    const epx = -erz, epz = erx;
    const rhod = v.x * erx + v.z * erz, vt = v.x * epx + v.z * epz;
    const Rg = Math.max(this.crownR, crown?.r ?? 0) + CROWN_CLEARANCE + 1.5 * Math.sin(t * 0.09) + Math.min(4, len(this.tvel) * 0.6);
    let hGoal = Math.max(1.2, this.target.y + 0.8 + 1.2 * Math.sin(t * 0.06));

    // crowns in the way: along the radial path to the orbit, and (other trees) along the current velocity
    let vRad = V_RADIAL;
    const reach = Math.min(LOOKAHEAD, Math.abs(Rg - rho));
    const sgn = Rg >= rho ? 1 : -1;
    const r1x = p.x + erx * sgn * reach, r1z = p.z + erz * sgn * reach;
    const vh = Math.hypot(v.x, v.z), vk = vh > 1e-9 ? Math.min(3, LOOKAHEAD / vh) : 0;
    let vtGoal = ORBIT_RATE * Math.min(rho, Rg);
    const phi = Math.atan2(dz, dx), arcSign = vtGoal >= 0 ? 1 : -1;
    for (const c of this.crowns.values()) {
      const R = c.r + LIFT_SOFT, need = c.top + LIFT_CLEAR;
      const eRad = entryAlong(p.x, p.z, r1x, r1z, c, R);
      const isDest = crown !== null && c.x === crown.x && c.z === crown.z;
      const eVel = isDest ? Infinity : entryAlong(p.x, p.z, p.x + v.x * vk, p.z + v.z * vk, c, R);
      // the orbit ring ahead, sampled every unit of arc (speed-independent, so a stopped camera still sees it)
      let eArc = Infinity;
      if (!isDest && rho > 0.5)
        for (let j = 1; j <= ORBIT_LOOK; j++) {
          const a = phi + (arcSign * j) / rho;
          if (Math.hypot(C.x + Math.cos(a) * rho - c.x, C.z + Math.sin(a) * rho - c.z) < R) {
            eArc = j - 1;
            break;
          }
        }
      const inside = Math.hypot(p.x - c.x, p.z - c.z) < R;
      if (inside || eRad < LOOKAHEAD || eVel < LOOKAHEAD || eArc < ORBIT_LOOK) hGoal = Math.max(hGoal, need);
      if (p.y < need - 0.05) {
        // hold back until the climb is done: every path into the crown is speed-gated by the distance left
        const gate = (e: number) => Math.max(0, e - 0.5) * 0.8;
        if (eRad < Infinity) vRad = Math.min(vRad, gate(eRad));
        if (eVel < Infinity) vRad = Math.min(vRad, gate(eVel));
        if (eArc < Infinity) vtGoal = arcSign * Math.min(Math.abs(vtGoal), gate(eArc));
      }
    }

    // spring accelerations (radial, height, tangential), then limits
    const wR = 2 / T_RADIAL, wH = 2 / T_RISE;
    const aR = wR * wR * clamp(Rg - rho, -vRad * T_RADIAL, vRad * T_RADIAL) - 2 * wR * rhod - (rho > 0.5 ? (vt * vt) / rho : 0);
    const aH = wH * wH * clamp(hGoal - p.y, -V_RISE * T_RISE, V_RISE * T_RISE) - 2 * wH * v.y;
    const aT = (vtGoal - vt) / TAU_TAN;
    let a = v3(erx * aR + epx * aT, aH, erz * aR + epz * aT);
    const al = len(a);
    if (al > A_MAX) a = mul(a, A_MAX / al);
    let nv = add(v, mul(a, h));
    const s = len(nv);
    // over the limit (e.g. carried in from the follow view): the excess e decays as e' = −A_MAX − e/BRAKE_TAU, so it
    // is gone within 0.5 s even from 200 u/s, without a snap
    if (s > V_MAX) {
      const e = Math.max(0, (s - V_MAX + A_MAX * BRAKE_TAU) * Math.exp(-h / BRAKE_TAU) - A_MAX * BRAKE_TAU);
      nv = mul(nv, (V_MAX + e) / s);
    }
    this.vel = nv;
    this.pos = add(p, mul(nv, h));
  }
}
