// Pure data-shape helpers for the /stats page.
//
// The backend (app/stats/routes_api.py) exposes two endpoints that return
// flat arrays of minute-bucketed samples:
//
//   GET /api/stats/models  →  ModelSample[]
//   GET /api/stats/gpus    →  GpuSample[]
//
// `minute` is `int(time.time() // 60)` — epoch-minutes. The chart
// components need data shaped for recharts:
//
//   throughput → one row per minute summing across all models
//   gpu util   → one row per minute with a numeric column per gpu_index
//
// Keeping these transforms in a pure module makes them trivial to test
// without dragging recharts into jsdom.

export interface ModelSample {
  model_id: string;
  minute: number;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface GpuSample {
  gpu_index: number;
  minute: number;
  utilization_pct: number;
  memory_used_mib: number;
  memory_total_mib: number;
}

export interface ThroughputPoint {
  /** epoch-minute (matches backend bucket key) */
  minute: number;
  /** epoch-ms — convenient for Date construction in axis formatters */
  ts: number;
  /** prompt + completion tokens across all models, summed per minute */
  tokens: number;
  /** request count across all models, summed per minute */
  requests: number;
}

export interface GpuUtilPoint {
  minute: number;
  ts: number;
  /** one numeric column per gpu_index, e.g. gpu0, gpu1, gpu2 */
  [gpuColumn: string]: number;
}

export interface GpuUtilAggregate {
  points: GpuUtilPoint[];
  /** stable, ascending list of gpu_index values present in the window */
  gpuIndexes: number[];
}

/**
 * Collapse the per-(model, minute) backend rows into a single per-minute
 * series suitable for a LineChart. Tokens and requests are summed across
 * all models in the same minute bucket.
 *
 * The backend already orders by minute ASC, but we sort defensively so a
 * future caller (e.g. mock fixtures, alternate endpoints) cannot produce
 * a jagged X axis.
 */
export function aggregateThroughput(samples: readonly ModelSample[]): ThroughputPoint[] {
  if (samples.length === 0) return [];

  const byMinute = new Map<number, ThroughputPoint>();
  for (const s of samples) {
    const existing = byMinute.get(s.minute);
    if (existing) {
      existing.tokens += s.prompt_tokens + s.completion_tokens;
      existing.requests += s.requests;
    } else {
      byMinute.set(s.minute, {
        minute: s.minute,
        ts: s.minute * 60_000,
        tokens: s.prompt_tokens + s.completion_tokens,
        requests: s.requests,
      });
    }
  }

  return Array.from(byMinute.values()).sort((a, b) => a.minute - b.minute);
}

/**
 * Pivot the flat GPU samples into one row per minute with a numeric column
 * per gpu_index. Missing cells (a GPU with no sample at a given minute) are
 * left undefined so recharts renders a gap rather than a misleading zero.
 */
export function aggregateGpuUtil(samples: readonly GpuSample[]): GpuUtilAggregate {
  if (samples.length === 0) return { points: [], gpuIndexes: [] };

  const indexes = new Set<number>();
  const byMinute = new Map<number, GpuUtilPoint>();

  for (const s of samples) {
    indexes.add(s.gpu_index);
    const col = `gpu${s.gpu_index}`;
    const existing = byMinute.get(s.minute);
    if (existing) {
      existing[col] = s.utilization_pct;
    } else {
      const row: GpuUtilPoint = {
        minute: s.minute,
        ts: s.minute * 60_000,
      };
      row[col] = s.utilization_pct;
      byMinute.set(s.minute, row);
    }
  }

  return {
    points: Array.from(byMinute.values()).sort((a, b) => a.minute - b.minute),
    gpuIndexes: Array.from(indexes).sort((a, b) => a - b),
  };
}

// ---------------------------------------------------------------------------
// Summary helpers — small headline numbers for the MetricSummaryPanel above
// each chart. Pure math; formatting is a render concern.
// ---------------------------------------------------------------------------

export interface ThroughputSummary {
  totalTokens: number;
  totalRequests: number;
  activeModels: number;
}

export function summarizeThroughput(samples: readonly ModelSample[]): ThroughputSummary {
  let totalTokens = 0;
  let totalRequests = 0;
  const models = new Set<string>();
  for (const s of samples) {
    totalTokens += s.prompt_tokens + s.completion_tokens;
    totalRequests += s.requests;
    models.add(s.model_id);
  }
  return { totalTokens, totalRequests, activeModels: models.size };
}

export interface GpuUtilSummary {
  peakPct: number;
  avgPct: number;
  gpuCount: number;
}

export function summarizeGpuUtil(samples: readonly GpuSample[]): GpuUtilSummary {
  if (samples.length === 0) return { peakPct: 0, avgPct: 0, gpuCount: 0 };
  let peak = 0;
  let sum = 0;
  const indexes = new Set<number>();
  for (const s of samples) {
    if (s.utilization_pct > peak) peak = s.utilization_pct;
    sum += s.utilization_pct;
    indexes.add(s.gpu_index);
  }
  return {
    peakPct: peak,
    avgPct: sum / samples.length,
    gpuCount: indexes.size,
  };
}

// ---------------------------------------------------------------------------
// Range bounds — drives the chart X-axis domain so it follows the selector
// even when the data is sparse. The previous implementation let recharts
// pick `["dataMin", "dataMax"]`, which (a) collapses to a point when a
// single sample is in-window and (b) silently truncates the axis when the
// most recent minute hasn't yet been written by the metrics rollup.
// ---------------------------------------------------------------------------

/** Selector ranges shown in the /stats UI. Keep in sync with RANGES in page.tsx. */
export type StatsRange = "1h" | "6h" | "24h" | "7d";

/** Width of each range in milliseconds. Exported for callers that want the
 *  raw number (e.g. SWR refresh tuning, debug logging). */
export const RANGE_MS: Record<StatsRange, number> = {
  "1h": 60 * 60 * 1000,
  "6h": 6 * 60 * 60 * 1000,
  "24h": 24 * 60 * 60 * 1000,
  "7d": 7 * 24 * 60 * 60 * 1000,
};

/**
 * Derive `[startMs, endMs]` for an axis domain from a range selector value.
 * `endMs` defaults to `Date.now()` (the chart is "now-relative") but is
 * injectable to keep the helper pure for unit tests.
 *
 * The bounds always run forward: `start < end`. Callers should pass the
 * result straight into recharts as `<XAxis domain={bounds} />`.
 */
export function rangeBounds(
  range: StatsRange,
  now: number = Date.now(),
): [number, number] {
  return [now - RANGE_MS[range], now];
}
