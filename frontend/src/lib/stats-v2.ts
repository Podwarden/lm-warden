// Frontend types + tiny helpers for the /api/stats/v2 contract (S7, #124).
//
// The backend (app/stats/routes_api.py:stats_v2_overview / _tokens_per_key)
// returns shapes that the frontend redesign consumes directly — no
// aggregation needed on the client (v2's whole point: do that work in
// SQL once, not in every browser tab). These types pin the contract;
// they intentionally mirror the docstrings on the Python side so a
// shape drift fails typecheck in CI before it ships.
//
// Note on regenerated types: v2 routes return bare dicts (FastAPI
// without `response_model=`), so openapi-typescript can't reach the
// inner shape — it lands as `Json` in api-types.generated.ts. We
// hand-type the response here and lock the contract via the page
// component tests instead.

export type StatsRange = "1h" | "6h" | "24h" | "7d";

export const STATS_RANGES: readonly StatsRange[] = ["1h", "6h", "24h", "7d"];

// ---- /api/stats/v2/overview ----------------------------------------------

/** The GPU probe right now (#255). "failing": nvidia-smi is installed but
 *  did not answer — e.g. "Failed to initialize NVML: Unknown Error" after a
 *  systemd reload revoked the container's GPU grant. "absent": no nvidia-smi
 *  (a CPU-only install). Optional: an API older than this omits it. */
export interface StatsV2GpuProbe {
  state: "unknown" | "ok" | "failing" | "absent";
  error: string | null;
}

export interface StatsV2Current {
  /** The four GPU numbers are null while the probe is failing: the newest
   *  sample predates the failure and must not pass for current (#255). */
  vram_used_mib: number | null;
  vram_total_mib: number | null;
  vram_pct: number | null; // 0..100, rounded
  gpu_util_pct: number | null; // max across GPUs at the most recent minute
  gpu_probe?: StatsV2GpuProbe;
  /** Sum of per-GPU averages over the last minute. `null` when no card
   *  on the host reports power.draw (older or virtualised GPUs). */
  power_w: number | null;
  /** Tokens per second over the last full minute (prompt + completion / 60). */
  tps: number;
}

export interface StatsV2ActiveModel {
  id: string;
  served_model_name: string;
  /**
   * The cards this model occupies. Optional at the type level for ui/api skew.
   *
   * It matters because VRAM, utilisation and power have no model dimension —
   * gpu_samples has no model column — so a per-model reading of them is, and
   * can only be, "the cards this model holds". This list is what lets the
   * selector say which cards a checkbox brings in.
   */
  gpu_indices?: number[];
}

export interface StatsV2VramPoint {
  minute: number;
  used_mib: number;
  total_mib: number;
}

export interface StatsV2UtilPoint {
  minute: number;
  max_pct: number;
}

export interface StatsV2PowerPoint {
  minute: number;
  watts: number;
}

export interface StatsV2TokensPoint {
  minute: number;
  prompt: number;
  completion: number;
}

export interface StatsV2Series {
  vram: StatsV2VramPoint[];
  util: StatsV2UtilPoint[];
  power: StatsV2PowerPoint[];
  tokens: StatsV2TokensPoint[];
}

export interface StatsV2Overview {
  range: StatsRange;
  now_minute: number;
  since_minute: number;
  /**
   * The model selection this response was computed for; `null` when the
   * request carried no `?models=` and the numbers cover the whole deployment.
   *
   * Echoed rather than assumed so the page states what its numbers cover from
   * the RESPONSE, not from checkboxes that may have moved while it was in
   * flight. Null and "all the ids, which happen to be all of them" are
   * different questions about the CARDS, and the difference survives here.
   */
  selected_model_ids: string[] | null;
  /** The cards that selection resolved to; null when unfiltered. */
  selected_gpu_indices: number[] | null;
  current: StatsV2Current;
  /** Every loaded model — NOT narrowed by the selection, because this is the
   *  list the selector itself is built from. */
  active_models: StatsV2ActiveModel[];
  series: StatsV2Series;
}

// ---- /api/stats/v2/tokens-per-key ----------------------------------------

export interface StatsV2TokensPerKeyRow {
  token_id: string;
  /** "(unknown)" for orphan rows where the api_tokens entry was deleted. */
  name: string;
  /** `null` for the orphan case — no api_tokens row to join. */
  prefix: string | null;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StatsV2TokensPerKey {
  range: StatsRange;
  since_minute: number;
  rows: StatsV2TokensPerKeyRow[];
}

// ---- Formatters ----------------------------------------------------------
//
// Kept in this module so the page and the future export-as-CSV path can
// share one definition. All functions are pure.

/** Render an integer MiB count as a human-friendly GiB string, e.g.
 *  `12000` → `"11.7"`. One decimal place — matches header-metrics. */
export function mibToGib(mib: number | null): string {
  if (mib === null) return "—";
  if (!mib) return "0";
  return (mib / 1024).toFixed(1);
}

/** Render a watts value or "—" for missing. */
export function formatWatts(w: number | null | undefined): string {
  if (w === null || w === undefined || Number.isNaN(w)) return "—";
  // Operator-friendly precision: one decimal for sub-100W readings,
  // integer above that — keeps the digit count stable and readable.
  return w < 100 ? w.toFixed(1) : Math.round(w).toString();
}

/** Render TPS — integer once we cross 10, one decimal below. */
export function formatTps(tps: number): string {
  if (!Number.isFinite(tps) || tps <= 0) return "0";
  return tps < 10 ? tps.toFixed(1) : Math.round(tps).toString();
}

/** Attach an epoch-ms `ts` column to a minute-bucketed point so recharts
 *  can use a time-scaled XAxis. Generic over any object with `minute`. */
export function withTs<T extends { minute: number }>(rows: readonly T[]): (T & { ts: number })[] {
  return rows.map((r) => ({ ...r, ts: r.minute * 60_000 }));
}
