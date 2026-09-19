// Per-request color derivation for the god-mode viewer.
//
// God mode interleaves the token streams of many concurrent requests. To
// let an operator track one session by eye through that interleaving, every
// request (`req_id`) gets a STABLE, deterministic color applied to its block
// header + left-edge accent bar. This is display-only — the backend never
// sends a color; it's derived here so there is zero server state.
//
// Two derivation paths:
//   - golden-angle (preferred): keyed on a per-session request index, so
//     adjacent requests land ~137.5° apart on the hue wheel and are always
//     visually distinct. The viewer assigns each new req_id a monotonic
//     session index (never reclaimed), so a request's color is stable for
//     the life of the session even as older events are elided from the ring.
//   - hash fallback: FNV-1a over the req_id → hue, used when no index is
//     available (e.g. a request whose start was elided before we saw it).
//
// Saturation/lightness are fixed at mid values that stay legible on both the
// light and dark themes: the accent is a solid mid-L/mid-S hue that has
// enough contrast against slate-950 and against white, and the header tint is
// alpha-blended so it composits over whichever theme background is behind it.

export interface ReqColor {
  /** Hue in degrees, 0–360. Exposed for tests + any hue-based styling. */
  hue: number;
  /** Solid accent for the left-edge bar and header text keyline. */
  accent: string;
  /** Alpha-blended header background tint (theme-composites). */
  headerBg: string;
  /** Header label text color — lighter/brighter variant of the hue. */
  text: string;
}

// The golden angle keeps successive session indices maximally spread around
// the hue wheel (used the same way d3 / many palette generators do).
const GOLDEN_ANGLE = 137.508;
// Start offset so index 0 isn't pure red (0°) — a small aesthetic nudge.
const HUE_OFFSET = 40;

/** FNV-1a 32-bit hash. `Math.imul` keeps the multiply in exact 32-bit space
 *  so the result is stable across engines. */
function fnv1a(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
  }
  return h >>> 0;
}

/**
 * Resolve a hue for a request. When `index` is supplied (the per-session
 * request counter), use golden-angle stepping for guaranteed adjacent-distinct
 * hues; otherwise fall back to a hash of the id.
 */
export function reqColorHue(reqId: string, index?: number): number {
  if (index !== undefined && index >= 0) {
    return (HUE_OFFSET + index * GOLDEN_ANGLE) % 360;
  }
  return fnv1a(reqId) % 360;
}

/** Full color set for a request block. Deterministic in `(reqId, index)`. */
export function reqColor(reqId: string, index?: number): ReqColor {
  const hue = reqColorHue(reqId, index);
  // Values from the approved token-details mockup (spec 2026-09-18 §4.6).
  return {
    hue,
    accent: `hsl(${hue.toFixed(1)} 70% 60%)`,
    headerBg: `hsl(${hue.toFixed(1)} 45% 18% / 0.55)`,
    text: `hsl(${hue.toFixed(1)} 85% 75%)`,
  };
}
