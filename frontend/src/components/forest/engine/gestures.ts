/**
 * Plan 4 Task 4: trackpad gestures, ported from the style mockup (forest-styles.html, "trackpad gestures"). Pure: no
 * three.js and no DOM, so the mapping is unit-tested; the scene (scene.ts) reads the events, applies the pose these
 * functions return through the camera controller (`userInteracted` + `setPose`: the 9 s hold and the ground clamp).
 *
 * | input                                         | action                                         |
 * |-----------------------------------------------|------------------------------------------------|
 * | two-finger slide (wheel, no ctrlKey)          | pan, scaled by the camera's distance           |
 * | pinch (wheel with ctrlKey)                    | zoom toward the pointer (the page never zooms) |
 * | Shift + two-finger slide                      | orbit and tilt, above the horizon              |
 * | Safari twist (gesturechange rotation)         | orbit                                          |
 * | real mouse wheel (line/page deltas, notches)  | zoom toward the pointer                        |
 *
 * The Stats card (`WheelGate`) lets the page scroll over it until it is clicked or tapped into; a pinch zooms the card
 * even before that (the browser would otherwise zoom the whole page).
 */
import type { V3 } from "./camera";

/** The fields of a WheelEvent the mapping reads. */
export interface WheelLike {
  deltaX: number;
  deltaY: number;
  /** 0 pixels, 1 lines, 2 pages (WheelEvent.DOM_DELTA_*). */
  deltaMode: number;
  ctrlKey: boolean;
  shiftKey: boolean;
}

export type GestureIntent =
  | { kind: "pan"; dx: number; dy: number }
  | { kind: "zoom"; factor: number }
  | { kind: "orbit"; dTheta: number; dPhi: number };

/** A line of wheel delta, in px. */
const LINE_PX = 16;
/** Pinch zoom per px of deltaY (exp scale: −10 px → ×0.905). */
const PINCH_K = 0.01;
/** Mouse-wheel zoom per px of deltaY (a 100 px notch → ×1.16). */
const WHEEL_K = 0.0015;
/** Shift-slide orbit per px (rad). */
const ORBIT_K = 0.006;
/** Pan per px, per unit of camera distance. */
const PAN_K = 0.0016;
/** The orbit never tilts lower than this polar angle (just above the horizon), unless it already stands lower. */
export const HORIZON_MAX = Math.PI * 0.49;
/** …nor higher than this (straight down is degenerate). */
const POLAR_MIN = 0.05;

/**
 * A real mouse wheel, not a trackpad (the mockup's heuristic): line or page deltas, or whole-pixel vertical notches of
 * ≥ 50 px with no horizontal part. A pinch (ctrlKey) is never one.
 */
export function isMouseWheel(e: WheelLike): boolean {
  if (e.deltaMode === 1 || e.deltaMode === 2) return true;
  return !e.ctrlKey && e.deltaX === 0 && Number.isInteger(e.deltaY) && Math.abs(e.deltaY) >= 50;
}

/** What a wheel event does. `viewH`: the view's height in px (a page of delta). */
export function wheelIntent(e: WheelLike, viewH: number): GestureIntent {
  const px = e.deltaMode === 1 ? LINE_PX : e.deltaMode === 2 ? Math.max(1, viewH) : 1;
  if (e.ctrlKey) return { kind: "zoom", factor: Math.exp(e.deltaY * px * PINCH_K) };
  if (isMouseWheel(e)) return { kind: "zoom", factor: Math.exp(e.deltaY * px * WHEEL_K) };
  // macOS turns a shifted vertical slide into a horizontal one: either axis orbits
  if (e.shiftKey) return { kind: "orbit", dTheta: -(e.deltaX || 0) * px * ORBIT_K, dPhi: -(e.deltaY || 0) * px * ORBIT_K };
  return { kind: "pan", dx: e.deltaX * px, dy: e.deltaY * px };
}

/** Safari's twist: the change of `GestureEvent.rotation` (degrees) turns the azimuth. */
export function twistOrbit(prevDeg: number, nowDeg: number): { kind: "orbit"; dTheta: number; dPhi: number } {
  return { kind: "orbit", dTheta: (-(nowDeg - prevDeg) * Math.PI) / 180, dPhi: 0 };
}

/** World units moved per px of slide at camera distance `dist`. */
export const panScale = (dist: number): number => dist * PAN_K;

// ---------------- pose math (plain V3) ----------------

const v = (x: number, y: number, z: number): V3 => ({ x, y, z });
const add = (a: V3, b: V3) => v(a.x + b.x, a.y + b.y, a.z + b.z);
const sub = (a: V3, b: V3) => v(a.x - b.x, a.y - b.y, a.z - b.z);
const mul = (a: V3, k: number) => v(a.x * k, a.y * k, a.z * k);
const dot = (a: V3, b: V3) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: V3, b: V3) => v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const len = (a: V3) => Math.hypot(a.x, a.y, a.z);
const norm = (a: V3) => {
  const l = len(a);
  return l > 1e-12 ? mul(a, 1 / l) : v(0, 0, -1);
};

/** The camera's basis (looking from `pos` at `target`, world up +y): forward, right, up. */
function basis(pos: V3, target: V3): { f: V3; r: V3; u: V3 } {
  const f = norm(sub(target, pos));
  let r = cross(f, v(0, 1, 0));
  if (len(r) < 1e-9) r = v(1, 0, 0); // looking straight down or up
  r = norm(r);
  return { f, r, u: cross(r, f) };
}

export interface Pose {
  pos: V3;
  target: V3;
}

/** Two-finger slide: camera and target move together in the screen plane, by `panScale(distance)` per px. */
export function applyPan(pos: V3, target: V3, dx: number, dy: number): Pose {
  const { r, u } = basis(pos, target);
  const k = panScale(len(sub(pos, target)));
  const off = add(mul(r, dx * k), mul(u, -dy * k));
  return { pos: add(pos, off), target: add(target, off) };
}

/**
 * Orbit about the target at the same distance: azimuth by `dTheta`, polar angle by `dPhi`, kept in [POLAR_MIN,
 * HORIZON_MAX]. A camera already lower than HORIZON_MAX (a low cinematic orbit) keeps its own angle as the limit, so it
 * is never snapped up and may only tilt up.
 */
export function applyOrbit(pos: V3, target: V3, dTheta: number, dPhi: number): Pose {
  const off = sub(pos, target), r = len(off);
  if (r < 1e-9) return { pos: { ...pos }, target: { ...target } };
  const phi0 = Math.acos(Math.max(-1, Math.min(1, off.y / r)));
  const theta = Math.atan2(off.x, off.z) + dTheta;
  const phi = Math.max(POLAR_MIN, Math.min(Math.max(HORIZON_MAX, phi0), phi0 + dPhi));
  const s = Math.sin(phi);
  return { pos: add(target, v(r * s * Math.sin(theta), r * Math.cos(phi), r * s * Math.cos(theta))), target: { ...target } };
}

export interface ZoomOptions {
  fovDeg: number;
  aspect: number;
  minDist: number;
  maxDist: number;
}

/**
 * Zoom toward the pointer at NDC (`nx`, `ny`; −1…1, y up): the point under the pointer on the plane through the target
 * (facing the camera) stays where it is on screen; the distance to the target scales by `factor`, within
 * [minDist, maxDist].
 */
export function applyZoom(pos: V3, target: V3, nx: number, ny: number, factor: number, o: ZoomOptions): Pose {
  const d0 = len(sub(target, pos));
  if (d0 < 1e-9 || !Number.isFinite(factor)) return { pos: { ...pos }, target: { ...target } };
  const { f, r, u } = basis(pos, target);
  const th = Math.tan((o.fovDeg * Math.PI) / 360);
  const dir = add(f, add(mul(r, nx * th * o.aspect), mul(u, ny * th)));
  const t = dot(sub(target, pos), f) / Math.max(1e-9, dot(dir, f));
  const P = add(pos, mul(dir, t));
  const d1 = Math.max(o.minDist, Math.min(o.maxDist, d0 * factor));
  const k = d1 / d0;
  return { pos: add(P, mul(sub(pos, P), k)), target: add(P, mul(sub(target, P), k)) };
}

/**
 * Whether a wheel event is the forest's (handled, default prevented) or the page's (it scrolls). The full window always
 * captures. The card (`needsEngage`) captures only after a click or tap into it, until a pointer-down elsewhere; a
 * pinch (ctrlKey) is always the card's, so the browser never zooms the page over it. Esc and a window blur release it
 * too. `onChange` hears every change of engagement (the card shows a focus ring while engaged); the full window, which
 * always captures, is never "engaged".
 */
export class WheelGate {
  private on = false;
  constructor(
    readonly needsEngage: boolean,
    private readonly onChange?: (engaged: boolean) => void,
  ) {}
  get engaged(): boolean {
    return this.on;
  }
  private set(v: boolean): void {
    v &&= this.needsEngage;
    if (v === this.on) return;
    this.on = v;
    this.onChange?.(v);
  }
  /** A pointer-down anywhere: inside the canvas engages, outside releases. */
  pointerDown(inside: boolean): void {
    this.set(inside);
  }
  /** A key pressed anywhere: Esc releases. */
  key(key: string): void {
    if (key === "Escape") this.set(false);
  }
  /** The window lost focus. */
  blur(): void {
    this.set(false);
  }
  release(): void {
    this.set(false);
  }
  captures(e: WheelLike): boolean {
    return !this.needsEngage || this.engaged || e.ctrlKey;
  }
  /** A Safari twist orbits like Shift + two fingers: only once the card is engaged (final review M7). */
  capturesTwist(): boolean {
    return !this.needsEngage || this.engaged;
  }
}
