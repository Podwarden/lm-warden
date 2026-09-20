// Types + pure helpers for the per-request history endpoints:
//
//   GET /api/stats/v2/requests   one row per completed request in the window
//   GET /api/stats/v2/latency    TTFT / ITL / duration distributions from those rows
//
// Both are served from the request_history table the proxy feeds
// (app/stats/request_history.py). Hand-typed for the same reason as
// stats-v2.ts: the routes return `dict[str, Any]`, so openapi-typescript
// cannot reach the inner shape.
//
// Everything here is DOM-free so the chart's encoding decisions can be
// tested without rendering a chart.

import type { HistogramBuckets } from "./live-stats";
import type { StatsRange } from "./stats-v2";

// ---- wire shapes ------------------------------------------------------------

/** One completed request, as the proxy recorded it. */
export interface RequestHistoryRow {
  id: string;
  /** Epoch SECONDS when the proxy saw the request end. */
  finished_at: number;
  /** models table ROW id — what `?models=` filters on. */
  model_id: string;
  /** Served name — what a human reads. */
  model: string;
  token_name: string | null;
  /** Opaque token row id; links this row to the token's details page. Null
   *  for anonymous requests and for rows written before the token_id column
   *  existed (migration adding it, v2026.09.18.1). */
  token_id: string | null;
  client_ip: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  duration_s: number;
  /** Null when no token ever arrived (abort before the first frame, or a
   *  non-streaming request, which has no first frame to time). */
  ttft_s: number | null;
  /**
   * Seconds spent waiting at the proxy's per-engine admission gate before the
   * request was forwarded (migration 0032).
   *
   * NOT part of `ttft_s` or `duration_s` — the clock both are measured from is
   * read once the slot is held, so this was never inside either. Null means a
   * row written before 0032, i.e. not measured; 0 is a real reading meaning
   * the engine had a free slot.
   */
  queued_s: number | null;
  finish_reason: string | null;
  orphan: boolean;
  started_iso: string;
}

export interface HistoryCoverage {
  /** When history begins, store-wide, epoch seconds. Null when empty. */
  earliest_epoch: number | null;
  retention_days: number;
  max_rows: number;
  /** True only when history reaches back to the window start AND retention
   *  is at least as long as the window. */
  covers_window: boolean;
}

export interface RequestHistoryResponse {
  range: StatsRange;
  since_epoch: number;
  now_epoch: number;
  selected_model_ids: string[] | null;
  /** Rows in the window for the selection, before sampling. */
  total: number;
  /** 1 = every row returned; k = every k-th row, newest first. */
  stride: number;
  requests: RequestHistoryRow[];
  coverage: HistoryCoverage;
}

export interface LatencyDistribution {
  count: number;
  p50: number | null;
  p90: number | null;
  p99: number | null;
  mean: number | null;
  /** Engine-histogram wire shape (cumulative counts, `le: null` = +Inf), so
   *  the existing LatencyHistogram renderer draws it unchanged. */
  buckets: HistogramBuckets;
}

export type LatencyBasis = "window" | "last";

export interface LatencyResponse {
  basis: LatencyBasis;
  range: StatsRange;
  /** Set for basis=last. */
  n: number | null;
  since_epoch: number | null;
  selected_model_ids: string[] | null;
  count: number;
  span_s: number | null;
  oldest_epoch: number | null;
  newest_epoch: number | null;
  ttft: LatencyDistribution;
  /** MEAN inter-token latency PER REQUEST — not the engine's per-token
   *  histogram. Every label that shows it says so. */
  itl: LatencyDistribution;
  duration: LatencyDistribution;
  coverage: HistoryCoverage;
}

/** The fixed count the "last N" basis asks for. */
export const LATENCY_LAST_N = 500;

// ---- the chart's encoding ---------------------------------------------------
//
// One mark per request: time on x, duration on a log y. Colour carries ONE
// categorical field, chosen from the data (below); mark size carries generated
// tokens; finish reason is carried by the mark's SHAPE, never by colour —
// colour is already spoken for. Prompt tokens and TTFT ride in the tooltip
// and the table.

/** How a request ended, folded to the three shapes the chart can draw. */
export type FinishClass = "stop" | "length" | "other";

export function finishClassOf(reason: string | null | undefined): FinishClass {
  if (reason === "stop") return "stop";
  if (reason === "length") return "length";
  return "other";
}

export const FINISH_CLASS_LABEL: Record<FinishClass, string> = {
  stop: "stop (filled)",
  length: "length (hollow)",
  other: "other / none (cross)",
};

/** Which categorical field colour encodes. */
export type ColourField = "client" | "model" | "none";

/** Ordered `text-chat-*` classes for colour groups; `currentColor` marks
 *  inherit them. Six distinct-enough tokens across both themes; the rest of
 *  the categories fold into `OTHER_CLASS`. */
export const GROUP_CLASSES: readonly string[] = [
  "text-chat-accent",
  "text-chat-negative",
  "text-chat-fg",
  "text-chat-muted",
  "text-chat-user",
  "text-chat-accent-strong",
];
export const OTHER_CLASS = "text-chat-dim";
export const OTHER_LABEL = "other";

/** The label of a client for grouping: token name first, IP as fallback. */
export function clientOf(r: Pick<RequestHistoryRow, "token_name" | "client_ip">): string {
  return r.token_name ?? r.client_ip ?? "anonymous";
}

/**
 * Decide what colour should encode, from the data.
 *
 * Colour by client is what the operator asked for, and with one dominant
 * client it earns nothing: a chart where 97% of marks share a colour reads
 * as a single series with noise. So: colour carries CLIENT when there are at
 * least two and the largest is under 95% of rows; otherwise MODEL when there
 * are at least two; otherwise nothing — one accent, and the legend says so.
 */
export function pickColourField(rows: readonly RequestHistoryRow[]): ColourField {
  if (rows.length === 0) return "none";
  const clients = countBy(rows, clientOf);
  if (clients.size >= 2 && topShare(clients, rows.length) < 0.95) return "client";
  const models = countBy(rows, (r) => r.model);
  if (models.size >= 2) return "model";
  return "none";
}

export function groupKeyOf(r: RequestHistoryRow, field: ColourField): string {
  if (field === "client") return clientOf(r);
  if (field === "model") return r.model;
  return "all";
}

export interface ColourGroup {
  key: string;
  count: number;
  /** A `text-chat-*` class. */
  className: string;
}

/**
 * The colour legend: the top groups by count, each with its class, then
 * `other` for everything past the palette. Stable across polls for the same
 * data because the order is (count desc, key asc).
 */
export function colourGroups(
  rows: readonly RequestHistoryRow[],
  field: ColourField,
): ColourGroup[] {
  if (field === "none") {
    return rows.length ? [{ key: "all", count: rows.length, className: GROUP_CLASSES[0] }] : [];
  }
  const counts = countBy(rows, (r) => groupKeyOf(r, field));
  const ordered = [...counts.entries()].sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
  const out: ColourGroup[] = ordered
    .slice(0, GROUP_CLASSES.length)
    .map(([key, count], i) => ({ key, count, className: GROUP_CLASSES[i] }));
  const rest = ordered.slice(GROUP_CLASSES.length);
  if (rest.length > 0) {
    out.push({
      key: OTHER_LABEL,
      count: rest.reduce((s, [, c]) => s + c, 0),
      className: OTHER_CLASS,
    });
  }
  return out;
}

/** Class for one row given the legend, `OTHER_CLASS` past the palette. */
export function classFor(
  groups: readonly ColourGroup[],
  key: string,
): string {
  return groups.find((g) => g.key === key)?.className ?? OTHER_CLASS;
}

/**
 * Mark radius in px from generated tokens: sqrt so AREA tracks the count,
 * 2..6 px so thousands of marks stay legible and a giant response does not
 * eclipse its neighbours.
 */
export function markRadius(generated: number, maxGenerated: number): number {
  if (maxGenerated <= 0 || generated <= 0) return 2;
  return 2 + 4 * Math.sqrt(Math.min(1, generated / maxGenerated));
}

/** Log-scale y needs a strictly positive floor; sub-10ms durations are
 *  clamped rather than dropped so an immediate error still shows. */
export const DURATION_FLOOR_S = 0.01;

/** Nice log ticks (…0.1, 1, 10, 100…) covering [lo, hi]. */
export function logTicks(lo: number, hi: number): number[] {
  if (!(hi > 0)) return [1];
  const start = Math.floor(Math.log10(Math.max(DURATION_FLOOR_S, lo)));
  const end = Math.ceil(Math.log10(hi));
  const out: number[] = [];
  for (let e = start; e <= end; e++) out.push(10 ** e);
  return out;
}

// ---- wording ------------------------------------------------------------------

/** Seconds → "45 s", "3 h 12 m", "6 d 4 h". */
export function formatSpan(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s) || s < 0) return "—";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  if (h < 24) return mm ? `${h} h ${mm} min` : `${h} h`;
  const d = Math.floor(h / 24);
  const hh = h % 24;
  return hh ? `${d} d ${hh} h` : `${d} d`;
}

/**
 * The precise statement of what a window covers, or null when it covers the
 * whole window. Wording carries the WHY: history that has not existed long
 * enough, or a retention shorter than the button — two different facts.
 */
export function coverageNote(
  coverage: HistoryCoverage | undefined,
  sinceEpoch: number | null,
  nowEpoch: number,
  human: string,
): string | null {
  if (!coverage || sinceEpoch === null) return null;
  if (coverage.covers_window) return null;
  const windowS = nowEpoch - sinceEpoch;
  if (coverage.retention_days * 86400 < windowS) {
    return `retention is ${coverage.retention_days} d, shorter than the ${human} — the window cannot be served in full`;
  }
  if (coverage.earliest_epoch === null) {
    return "no requests recorded yet";
  }
  const since = Math.max(0, nowEpoch - coverage.earliest_epoch);
  const covered = Math.min(since, windowS);
  return `history begins ${formatSpan(since)} ago — covers ${formatSpan(covered)} of the ${human}`;
}

// ---- tiny generic helpers ------------------------------------------------------

function countBy<T>(rows: readonly T[], key: (r: T) => string): Map<string, number> {
  const m = new Map<string, number>();
  for (const r of rows) {
    const k = key(r);
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return m;
}

function topShare(counts: Map<string, number>, total: number): number {
  let top = 0;
  for (const c of counts.values()) if (c > top) top = c;
  return total > 0 ? top / total : 0;
}

// ---- /api/stats/v2/throughput ----------------------------------------------

/**
 * Which question the panel is answering.
 *
 * `request`   per-request engine speed. How fast is the rig — idle time is
 *             invisible to it.
 * `wallclock` tokens counted per minute, idle minutes included as zero. How
 *             much work did the box do.
 */
export type ThroughputBasis = "request" | "wallclock";

/** One rate series. `null` means absent, which is NOT the same as 0 tok/s. */
export interface RateSummary {
  count: number;
  avg: number | null;
  max: number | null;
  /** Modal value over ~5%-wide bins. Null when the sample is too small or
   *  nothing repeats — see request_history.mode_relative on the Python side. */
  mode: number | null;
}

export interface ThroughputResponse {
  basis: ThroughputBasis;
  range: StatsRange;
  since_epoch: number;
  selected_model_ids: string[] | null;
  /** Prompt tokens per second. On the request basis this is
   *  `prompt_tokens / ttft`, which INCLUDES queue wait and counts
   *  prefix-cached tokens it never computed. */
  prefill: RateSummary;
  /** Completion tokens per second through the decode phase. */
  generation: RateSummary;
  coverage: HistoryCoverage;
}
