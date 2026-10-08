/**
 * "Bigger trees when far" (user decision, 2026-10-06): as the camera pulls back, trees and ground flora are drawn
 * larger, at render time, about their own bases (a group or instance scale: no geometry, place or time changes).
 *
 * - The distance is the camera's distance to the tree, normalised to the full window's height (a card's 320 px canvas
 *   shows a unit at a third of the full window's size, so its distances count ~2.8× as far).
 * - Up to FAR_D0 the scale is 1: every 1h, 6h and 24h view stays true to size (measured on the e2e farm week: their
 *   trees stand 42–146 units off in the full window and 52–73 in the card, i.e. at most 205 normalised). Beyond it the scale blends smoothly into d / FAR_D0, which
 *   keeps a tree's apparent size from shrinking further; it never exceeds FAR_MAX.
 * - Per tree it is capped (`crownCaps`) so no scaled crown overlaps a neighbouring tree's (crowns already touching stay
 *   at 1).
 */

/** Below this (normalised) distance trees are true to size. */
export const FAR_D0 = 230;
/** The largest scale. */
export const FAR_MAX = 4;
/** The full window's canvas height the distances are normalised to (px). */
export const REF_VIEW_H = 900;

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** The distance as the full window would see it: the same apparent size per unit. */
export function effectiveDistance(d: number, viewH: number): number {
  return d * (REF_VIEW_H / Math.max(1, viewH));
}

/** 1 up to FAR_D0, then smoothly into d / FAR_D0 (by 1.5 FAR_D0), capped at FAR_MAX. Continuous and monotone. */
export function farScale(dEff: number): number {
  if (!(dEff > FAR_D0)) return 1;
  const s = 1 + (dEff / FAR_D0 - 1) * smoothstep(FAR_D0, 1.5 * FAR_D0, dEff);
  return Math.min(FAR_MAX, s);
}

/**
 * Per tree, the largest scale at which its crown (radius r about its base) and every neighbour's, all scaled by their
 * own caps, cannot overlap: min over neighbours of distance / (r_i + r_j), at least 1 (crowns already touching keep
 * their true size), at most FAR_MAX.
 */
export function crownCaps(trees: readonly { id: string; x: number; z: number; r: number }[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const p of trees) {
    let cap = FAR_MAX;
    for (const q of trees) {
      if (q === p) continue;
      const rr = p.r + q.r;
      if (!(rr > 0)) continue;
      const d = Math.hypot(p.x - q.x, p.z - q.z);
      if (d >= rr * FAR_MAX) continue;
      cap = Math.min(cap, d / rr);
    }
    out.set(p.id, Math.max(1, cap));
  }
  return out;
}

/** A far view's build budget: every tree of a realistic 7 d week (about 155 trees in the e2e farm; final review I2). */
export const FAR_BUILT = 200;

/** Cubes a far tree may take at the coarsest level (level 2 averages about 57 a tree, with room): the far build count
 * is capped at the cubes left / this, so the budget never runs out of coarser levels (final review I2). */
export const FAR_CUBES_PER_TREE = 70;

/**
 * The build budget: the view's own (60 full window, 12 or 30 card), or in a far view FAR_BUILT, capped by `cubes` (the
 * cubes the trees may use: the card's 15k less its flora) / FAR_CUBES_PER_TREE, never below the view's own.
 */
export function treeBudget(base: number, far: boolean, cubes = Infinity): number {
  if (!far) return base;
  return Math.max(base, Math.min(FAR_BUILT, Math.floor(cubes / FAR_CUBES_PER_TREE)));
}

/** Caps and scales ease toward their target (time constant, s): a neighbour built or dropped never pops a crown. */
export const SCALE_TAU = 0.25;

export function easeScale(cur: number, target: number, dt: number): number {
  if (cur === target) return cur;
  const k = 1 - Math.exp(-Math.max(0, dt) / SCALE_TAU);
  const next = cur + (target - cur) * k;
  return Math.abs(next - target) < 1e-4 ? target : next;
}

/**
 * How much a bed of flowers (or a mushroom cluster) of `radius` at `z0` may grow before it reaches the grass edge at
 * half-width `halfW` (beds are squashed 0.85 across the spit): its far scale is capped by this, never below 1, so a
 * far-scaled bed never spills onto the sand (re-review 3).
 */
export function floraRoom(halfW: number, z0: number, radius: number): number {
  const room = halfW - Math.abs(z0);
  return Math.max(1, room / Math.max(1e-3, radius * 0.85));
}
