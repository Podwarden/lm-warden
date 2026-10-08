/**
 * The camera view handed between the Stats card and the forest page (pure; no three.js). The sender stores
 * `encodeHandoff(view, Date.now())` (sessionStorage); the receiver accepts it only for HANDOFF_TTL_MS, so a stale
 * entry never moves a camera later.
 *
 * Front-relative: the card's 6 h layout and the full window's 48 h one put the same forest at different world x, but
 * X(now) − X(t) depends only on the activity in [t, now] (layoutOrigin aligns the minute buckets), so x is sent as
 * x − X(shown time) and mapped back with the receiving scene's own front. y and z are sent as they are. A hand-off
 * without `frontRel: true` (the earlier world-space format) is ignored.
 *
 * Range (follow-up C): the card's range travels with the view and the full window opens with it. It is optional: a
 * payload without one (the format before the range switch) stays valid, and an unknown range is dropped (the view kept).
 */
import { type ForestRange, isForestRange } from "@/lib/forest/types";
import type { CameraMode, V3 } from "./camera";

export interface ForestView {
  /** x relative to the scene's front, X(shown time); y and z as in the scene. */
  pos: V3;
  target: V3;
  mode: CameraMode;
  frontRel: true;
  /** The sender's range (the Stats card's); absent in older payloads. */
  range?: ForestRange;
}

/** A hand-off older than this (ms) is ignored. */
export const HANDOFF_TTL_MS = 10_000;

export function encodeHandoff(v: ForestView, nowMs: number): string {
  return JSON.stringify({ pos: v.pos, target: v.target, mode: v.mode, frontRel: true, at: nowMs, ...(v.range ? { range: v.range } : {}) });
}

const num = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const v3 = (x: unknown): V3 | null => {
  if (!x || typeof x !== "object") return null;
  const o = x as Record<string, unknown>;
  return num(o.x) && num(o.y) && num(o.z) ? { x: o.x, y: o.y, z: o.z } : null;
};

/** The view, or null when `s` is not a hand-off, has the wrong shape, or is older than HANDOFF_TTL_MS at `nowMs`. */
export function decodeHandoff(s: string, nowMs: number): ForestView | null {
  let d: unknown;
  try {
    d = JSON.parse(s);
  } catch {
    return null;
  }
  if (!d || typeof d !== "object" || Array.isArray(d)) return null;
  const o = d as Record<string, unknown>;
  if (!num(o.at) || nowMs - o.at < 0 || nowMs - o.at > HANDOFF_TTL_MS) return null;
  const pos = v3(o.pos), target = v3(o.target);
  if (!pos || !target || (o.mode !== "cinema" && o.mode !== "follow") || o.frontRel !== true) return null;
  return { pos, target, mode: o.mode, frontRel: true, ...(isForestRange(o.range) ? { range: o.range } : {}) };
}

/** The sessionStorage key the hand-off travels under (Stats card → /forest). */
export const HANDOFF_KEY = "forest.handoff";

/**
 * The receiver's side: reads the stored hand-off, deletes it (it is used at most once, valid or not) and returns the
 * view it carries, or null when there is none, it is stale or malformed, or storage is unavailable.
 */
export function takeHandoff(nowMs: number): ForestView | null {
  try {
    const s = sessionStorage.getItem(HANDOFF_KEY);
    sessionStorage.removeItem(HANDOFF_KEY);
    return s === null ? null : decodeHandoff(s, nowMs);
  } catch {
    return null;
  }
}
