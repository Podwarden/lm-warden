/**
 * Spec §5 — equal lushness on any hardware. A turn's size is measured in seconds of full-server work on the
 * model variant that served it, so a tree grown on a slow GPU looks as lush as one grown on a fast one.
 */
import type { ModelStats, Turn } from "./types";

/** Fallbacks while a variant has no measured speed (spec §5 "Fallback"). */
export const FALLBACK_DECODE_TPS = 400;
export const FALLBACK_PREFILL_TPS = 4000;
export const FALLBACK_MAX_MODEL_LEN = 32768;
/** A prefix-cache hit costs 1/20 of a computed prompt token. */
export const CACHE_HIT_WEIGHT = 1 / 20;

export interface Budgets {
  /** Seconds of full-server decode: drives segment length. */
  len: number;
  /** Seconds of full-server prefill, cache hits at 1/20: drives girth. */
  girth: number;
  /** Share of the context window used: drives foliage density. */
  foliage: number;
}

export function budgets(turn: Turn, m: ModelStats): Budgets {
  const [, ctx, gen, cached] = turn;
  const hit = cached ?? 0;
  const P = m.prefill_tps ?? FALLBACK_PREFILL_TPS;
  return {
    len: gen / (m.decode_tps ?? FALLBACK_DECODE_TPS),
    girth: Math.max(0, ctx - hit) / P + (hit * CACHE_HIT_WEIGHT) / P,
    foliage: ctx / (m.max_model_len ?? FALLBACK_MAX_MODEL_LEN),
  };
}

/** Log compression `log2(1 + v / vRef)`: one monster session does not flatten the rest. */
export function compress(v: number, vRef: number): number {
  return Math.log2(1 + v / vRef);
}
