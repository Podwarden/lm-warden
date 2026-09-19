// Frontend types + tiny formatters for the two live-stats data planes
// (docs/live-stats-spec.md). Two independent backend shapes:
//
//   - Plane A "engine" — GET /api/stats/live (SSE). Aggregate truth scraped
//     from vLLM /metrics. See LiveEngineFrame.
//   - Plane B "requests" — GET /api/stats/requests (JWT JSON, polled ~1.5s).
//     Per-request truth the warden computes on the forward path. See
//     LiveRequestsSnapshot.
//
// These types pin the contract by hand: the /api/stats/* routes return bare
// dicts (no FastAPI response_model), so openapi-typescript can't reach the
// inner shape — same rationale as stats-v2.ts. A drift here fails typecheck.
//
// Unknown / absent engine metrics arrive as `null` (vLLM 0.25.1 renamed some
// names) — every optional numeric is therefore `number | null`.

// ---- Plane A: GET /api/stats/live (SSE frame) -----------------------------

// Every numeric field here is nullable, and that is a CONTRACT rather than
// defensive typing. Sub-project C added a second backend: llama.cpp publishes no
// KV-usage gauge, no cache_config_info, no preemption counter and no sleep
// state, so those arrive as `null` and must render as ABSENT -- never as 0. A
// zero here says "plenty of KV headroom" or "the engine is idle"; the truth is
// "this engine is silent about that", and an operator who acts on the first has
// been actively misled (design spec §9.3).
//
// The six that used to be non-nullable were widened deliberately, so TypeScript
// would point at every render site that needed fixing. That is the cheapest
// possible audit, and it found three.
export interface LiveEngine {
  num_requests_running: number | null;
  num_requests_waiting: number | null;
  /** Free-form reason → waiting count, e.g. {capacity, deferred}. */
  waiting_by_reason: Record<string, number | null>;
  kv_cache_usage_perc: number | null; // 0..1
  kv_tokens_used: number | null; // derived absolute
  kv_tokens_total: number | null; // block_size * num_gpu_blocks
  engine_sleep_state: number | null; // 0 = awake
  preemptions_total: number | null;
  preemptions_per_s: number | null; // null on the connection's first frame
}

export interface LiveThroughput {
  prompt_tokens_per_s: number | null;
  generation_tokens_per_s: number | null;
  prompt_tokens_total: number | null;
  generation_tokens_total: number | null;
}

export interface LiveCache {
  /** Interval delta hits/queries — null on the first frame. */
  prefix_hit_rate: number | null;
  prefix_hit_rate_cumulative: number | null;
  mm_hit_rate_cumulative: number | null;
  external_prefix_hit_rate_cumulative: number | null;
}

/**
 * One engine histogram, as cumulative-since-engine-start buckets.
 *
 * `le: null` encodes +Inf (JSON has no infinity, and dropping the bucket would
 * discard the whole tail). `counts` are cumulative along the boundaries. A
 * missing histogram is `null` at the parent — llama.cpp reports none at all,
 * and zeros would render as a distribution in which every request was
 * instantaneous.
 */
export interface HistogramBuckets {
  le: (number | null)[];
  counts: number[];
  count: number | null;
  sum: number | null;
}

export interface LatencyBuckets {
  ttft: HistogramBuckets | null;
  itl: HistogramBuckets | null;
  tpot: HistogramBuckets | null;
  e2e: HistogramBuckets | null;
}

export interface LiveLatency {
  ttft_p50: number | null;
  ttft_p90: number | null;
  ttft_p99: number | null;
  ttft_mean: number | null;
  itl_p50: number | null;
  itl_p99: number | null;
  tpot_p50: number | null;
  e2e_p50: number | null;
  e2e_p90: number | null;
  e2e_p99: number | null;
  /** Optional at the type level for ui/api rollout skew. */
  buckets?: LatencyBuckets | null;
}

export interface LiveEngineFrame {
  ts: string;
  /** Served model name, or null when no model is loaded (engine idle). */
  model: string | null;
  model_id: string | null;
  /**
   * Which engine produced this frame — "vllm" | "llamacpp" (sub-project C's one
   * additive field). Present so the dashboard can say WHICH engine does not
   * report a metric rather than showing a blank tile of unknown provenance.
   * Null only on the frames emitted before any model resolves.
   */
  backend: string | null;
  /** From the model row / engine — the denominator for context bars. */
  max_model_len: number | null;
  /**
   * NULL on a null frame, and the type says so. `_null_frame` on the backend
   * emits `engine`/`throughput`/`cache`/`latency`/`finished` as JSON null
   * while keeping `model_id`, so a scrape error produces a block that reaches
   * the per-model render path. The old page's types declared these
   * non-nullable, which was false, and `EngineHero` dereferenced
   * `frame.engine.kv_cache_usage_perc` unguarded — a scrape error could crash
   * the whole page. Widening the types makes TypeScript point at every render
   * site that needs the guard.
   */
  engine: LiveEngine | null;
  throughput: LiveThroughput | null;
  cache: LiveCache | null;
  latency: LiveLatency | null;
  /** request_success_total{finished_reason} → count. Values are null when the
   *  engine does not report that counter; the whole map is null on a null
   *  frame. */
  finished: Record<string, number | null> | null;
  /** Non-null string when the /metrics scrape failed this tick. */
  scrape_error: string | null;
  /**
   * Every loaded model's block, most significant first.
   *
   * Optional at the type level: ui and api ship as separate images, so a
   * bundle newer than its api pod receives a frame with only the top-level
   * fields. `liveFramesOf` below turns both shapes into one list.
   *
   * The top-level fields on THIS interface mirror `models[0]`; they are what
   * an api-newer-than-ui pairing keeps reading. Never read both.
   */
  models?: LiveEngineFrame[];
}

/**
 * Every model block in a frame, however old the API that produced it.
 *
 * A pre-multi-model frame IS one block, so it becomes a one-element list
 * rather than an empty one — otherwise a rollout skew would blank a dashboard
 * whose engine is serving perfectly well.
 */
export function liveFramesOf(
  frame: LiveEngineFrame | null | undefined,
): LiveEngineFrame[] {
  if (!frame) return [];
  if (Array.isArray(frame.models)) return frame.models;
  return frame.model_id === null && frame.model === null ? [] : [frame];
}

// ---- Combining several models' readings -----------------------------------
//
// THE RULE, and it is the whole reason this is a named function with tests
// rather than a `.reduce` at a call site: `null` means "this engine does not
// report it", NEVER zero. llama.cpp publishes no KV gauge, no preemption
// counter and no latency histograms; vLLM publishes all of them. Adding a
// reported number to an unreported one with `?? 0` produces a total that is
// silently short by an unknown amount and looks authoritative — which is
// exactly the bug sub-project C fixed for three panels, and a multi-model
// selection is a fresh chance to reintroduce it once per tile.
//
// So a combined value carries its own provenance: the number, and how many of
// the selected models actually contributed to it. A tile that reads "1 of 2
// models report this" is honest in a way that any single number cannot be.

export interface CombinedMetric {
  /** Sum over the models that REPORT it, or null when none of them do. */
  value: number | null;
  /** How many selected models reported it. */
  reporting: number;
  /** How many models were selected. `reporting < total` means partial. */
  total: number;
}

/**
 * Sum `values`, counting only the ones that are actually reported.
 *
 * `null`/`undefined`/`NaN` entries are SKIPPED, not zeroed. When every entry is
 * absent the result is `null` — "nobody here measures this" — and never `0`,
 * which would read as "measured, and it is zero".
 */
export function sumReported(
  values: readonly (number | null | undefined)[],
): CombinedMetric {
  let sum = 0;
  let reporting = 0;
  for (const v of values) {
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    sum += v;
    reporting += 1;
  }
  return {
    value: reporting > 0 ? sum : null,
    reporting,
    total: values.length,
  };
}

/** True when at least one selected model stayed silent about a reported metric. */
export function isPartial(m: CombinedMetric): boolean {
  return m.reporting > 0 && m.reporting < m.total;
}

export interface CombinedThroughput {
  prompt_tokens_per_s: CombinedMetric;
  generation_tokens_per_s: CombinedMetric;
  prompt_tokens_total: CombinedMetric;
  generation_tokens_total: CombinedMetric;
  running: CombinedMetric;
  waiting: CombinedMetric;
}

/**
 * Fleet-wide throughput across the selected models.
 *
 * Only additive, same-unit quantities are combined. Latency percentiles, KV
 * usage fractions and cache hit rates are deliberately NOT here: a p99 across
 * two engines is not a p99 of anything, and averaging two KV-usage fractions
 * whose denominators are different cache sizes produces a number with no
 * referent. Those stay per-model, one panel each, which is also what the
 * operator needs in order to act on them.
 */
export function combineThroughput(
  frames: readonly LiveEngineFrame[],
): CombinedThroughput {
  return {
    prompt_tokens_per_s: sumReported(
      frames.map((f) => f.throughput?.prompt_tokens_per_s),
    ),
    generation_tokens_per_s: sumReported(
      frames.map((f) => f.throughput?.generation_tokens_per_s),
    ),
    prompt_tokens_total: sumReported(
      frames.map((f) => f.throughput?.prompt_tokens_total),
    ),
    generation_tokens_total: sumReported(
      frames.map((f) => f.throughput?.generation_tokens_total),
    ),
    running: sumReported(frames.map((f) => f.engine?.num_requests_running)),
    waiting: sumReported(frames.map((f) => f.engine?.num_requests_waiting)),
  };
}

// ---- Plane B: GET /api/stats/requests (JSON snapshot) ---------------------

export interface LiveRequestRow {
  id: string;
  /** Opaque token row id. Null for anonymous requests; links the row to the
   *  token's details page (`/tokens/{id}`) when present. */
  token_id: string | null;
  token_name: string | null;
  client_ip: string | null;
  model: string;
  path: string; // /v1/chat/completions | /v1/completions
  prompt_tokens: number;
  completion_tokens: number;
  context_tokens: number; // prompt + completion
  max_model_len: number;
  context_pct: number; // context_tokens / max_model_len, 0..1
  elapsed_s: number;
  phase: string; // "prefill" | "decode"
  orphan: boolean;
}

export interface LiveByToken {
  /** Opaque token row id — the grouping key (names are reused across
   *  rotations, so two distinct tokens can share a name). Null = anonymous. */
  token_id: string | null;
  token_name: string | null;
  requests: number;
  context_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface LiveByIp {
  client_ip: string | null;
  requests: number;
  context_tokens: number;
}

export interface LiveRequestsSnapshot {
  ts: string;
  count: number;
  requests: LiveRequestRow[];
  by_token: LiveByToken[];
  by_ip: LiveByIp[];
}

/**
 * Re-derive the by-token / by-IP rollups from a SUBSET of the rows.
 *
 * The server computes them from every in-flight row (`_aggregate` in
 * app/stats/live_requests.py), which is right for the unfiltered view and
 * wrong the moment the operator narrows to one model: the request list would
 * shrink while the two panels underneath it kept describing the whole box.
 * Two views of the same instant disagreeing about which requests exist is
 * worse than either one alone.
 *
 * The server's aggregation is a plain group-by over the same rows, so
 * recomputing from the filtered rows is exact rather than an approximation —
 * and passing ALL rows here reproduces the server's own output.
 *
 * Grouped by `token_id`, not `token_name` — names are reused across
 * rotations, so two distinct tokens sharing a name must stay separate rows.
 * `token_id === null` is the anonymous bucket.
 */
export function aggregateRequests(rows: readonly LiveRequestRow[]): {
  by_token: LiveByToken[];
  by_ip: LiveByIp[];
} {
  const byToken = new Map<string | null, LiveByToken>();
  const byIp = new Map<string | null, LiveByIp>();
  for (const r of rows) {
    const t = byToken.get(r.token_id) ?? {
      token_id: r.token_id,
      token_name: r.token_name,
      requests: 0,
      context_tokens: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
    };
    t.requests += 1;
    t.context_tokens += r.context_tokens;
    t.prompt_tokens += r.prompt_tokens;
    t.completion_tokens += r.completion_tokens;
    byToken.set(r.token_id, t);

    const p = byIp.get(r.client_ip) ?? {
      client_ip: r.client_ip,
      requests: 0,
      context_tokens: 0,
    };
    p.requests += 1;
    p.context_tokens += r.context_tokens;
    byIp.set(r.client_ip, p);
  }
  const byContext = <T extends { context_tokens: number }>(a: T, b: T) =>
    b.context_tokens - a.context_tokens;
  return {
    by_token: [...byToken.values()].sort(byContext),
    by_ip: [...byIp.values()].sort(byContext),
  };
}

// ---- Pressure scale -------------------------------------------------------
//
// The single visual metaphor that ties the engine-wide KV cache gauge to the
// per-session context bars: both are "how full is this?" meters on the same
// green→amber→red scale. Thresholds are shared so a row that's hot means the
// same thing as a KV gauge that's hot.

export type Pressure = "healthy" | "warm" | "hot";

export const PRESSURE_WARM = 0.6;
export const PRESSURE_HOT = 0.85;

export function pressureOf(fraction: number): Pressure {
  if (fraction >= PRESSURE_HOT) return "hot";
  if (fraction >= PRESSURE_WARM) return "warm";
  return "healthy";
}

// ---- Formatters -----------------------------------------------------------
// Pure. Kept beside the types so the page and any future export path agree.

const DASH = "—";

/** Abbreviate a token/FLOP-style count: 195900 → "195.9k", 1.2e6 → "1.2M". */
export function formatCompact(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return DASH;
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return Math.round(n).toString();
}

/** Full integer with locale separators, or "—" for missing. */
export function formatInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return DASH;
  return Math.round(n).toLocaleString();
}

/** A 0..1 fraction as a whole-percent string: 0.871 → "87%". */
export function formatPct(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return DASH;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Seconds → operator-friendly latency: <1s in ms, else "8.1 s". */
export function formatLatency(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return DASH;
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  return `${s.toFixed(1)} s`;
}

/** Elapsed seconds → "m:ss" (or "s.s" under 10s for a live feel). */
export function formatElapsed(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return DASH;
  if (s < 10) return `${s.toFixed(1)}s`;
  const total = Math.floor(s);
  const m = Math.floor(total / 60);
  const sec = total % 60;
  if (m === 0) return `${sec}s`;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

/** FLOP/s → "1.2 PFLOP/s". */
export function formatFlops(f: number | null | undefined): string {
  if (f === null || f === undefined || Number.isNaN(f)) return DASH;
  if (f >= 1e15) return `${(f / 1e15).toFixed(1)} PFLOP/s`;
  if (f >= 1e12) return `${(f / 1e12).toFixed(1)} TFLOP/s`;
  if (f >= 1e9) return `${(f / 1e9).toFixed(1)} GFLOP/s`;
  return `${Math.round(f)} FLOP/s`;
}
