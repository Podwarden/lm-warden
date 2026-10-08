// Pure helpers behind the merged /stats page: windowing the engine's
// cumulative histograms, bucketing minute samples for long ranges, and the
// fixed 24h busy-median/peak reference.
//
// THE MIRROR RULE. Three functions here are client-side mirrors of committed
// backend code, and must stay semantically identical to it:
//
//   bucketDeltas    ← app/stats/live_engine.py::bucket_deltas
//   busyMedian      ← app/stats/history.py::busy_median
//   peakOf          ← app/stats/history.py::peak
//   quantileFromBuckets ← app/stats/prometheus.py::hist_quantile
//
// They are mirrored rather than served because their inputs live on the
// client: the SSE stream hands over cumulative buckets per frame and the ring
// of recent frames only exists in the browser tab, and the 24h reference is
// computed from an overview response the page already fetches. If the backend
// ever serves these directly, delete the mirrors.
//
// Everything here is pure and DOM-free so it can be tested without a page.

import type { HistogramBuckets } from "./live-stats";
import type { StatsRange, StatsV2TokensPoint } from "./stats-v2";

// ---- window metadata -------------------------------------------------------
//
// The window governs HISTORY panels only. Longer windows bucket up — 7d of
// minute samples is 10,080 rows, and one pixel-column per sample is both
// slower and less legible than aggregating — and the panel heading says which
// bucket it drew.

export interface WindowMeta {
  /** Bucket width in minutes for the minute-sampled history series. */
  bucketMinutes: number;
  /** What the heading says, e.g. "5 min buckets". */
  bucketLabel: string;
  /** Human phrase for the span, e.g. "last 24 hours". */
  human: string;
}

// Widths are capped so a ~1,000 px panel keeps a visible gap between bars:
// 7d at the old 30-min width drew 336 bars (~3 px each, no room for the gap
// or the tokens whiskers). The change smooths the GPU util and power charts
// too; their headings already say which width they drew at.
export const WINDOW_META: Record<StatsRange, WindowMeta> = {
  "1h": { bucketMinutes: 1, bucketLabel: "1 min buckets", human: "last hour" },
  "6h": { bucketMinutes: 2, bucketLabel: "2 min buckets", human: "last 6 hours" },
  "24h": { bucketMinutes: 10, bucketLabel: "10 min buckets", human: "last 24 hours" },
  "7d": { bucketMinutes: 60, bucketLabel: "60 min buckets", human: "last 7 days" },
};

/**
 * Fold minute-keyed points into `bucketMinutes`-wide buckets.
 *
 * `fold` collapses the points of one bucket into one point; the bucket is
 * keyed on its FIRST minute so the x-axis stays a real timestamp. A bucket
 * width of 1 returns the input untouched (same identity, no re-allocation) so
 * the short windows pay nothing.
 */
export function bucketPoints<T extends { minute: number }>(
  points: readonly T[],
  bucketMinutes: number,
  fold: (bucket: T[]) => T,
): T[] {
  if (bucketMinutes <= 1) return points as T[];
  const byBucket = new Map<number, T[]>();
  for (const p of points) {
    const key = Math.floor(p.minute / bucketMinutes) * bucketMinutes;
    const arr = byBucket.get(key);
    if (arr) arr.push(p);
    else byBucket.set(key, [p]);
  }
  return [...byBucket.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([key, bucket]) => ({ ...fold(bucket), minute: key }));
}

export const maxOf = (xs: readonly number[]): number =>
  xs.reduce((m, v) => (v > m ? v : m), -Infinity);

export const meanOf = (xs: readonly number[]): number =>
  xs.length === 0 ? 0 : xs.reduce((s, v) => s + v, 0) / xs.length;

// ---- tokens/second buckets ---------------------------------------------------
//
// The dense rollup the Tokens / second chart draws. Unlike bucketPoints it
// emits a row for EVERY bucket in the window (idle stretches are runs of
// zero bars, not gaps the axis jumps over), and each value is a RATE over
// the bucket's complete minutes: a 30-min bucket holding one 600,000-token
// minute reads 600,000 / (30 x 60) = 333 tok/s, not the 10,000 that
// averaging only the minutes that HAVE rows implies. The right bound mirrors
// the backend's _wallclock_rates: the last COMPLETE minute — the current
// minute is still being written and would read artificially low.

export interface TokenRateBucket {
  /** The bucket's start minute (its key) — a real timestamp for the x-axis. */
  minute: number;
  /** Mean prompt tok/s over the bucket's complete minutes. null when the
   *  bucket predates the measured range — "no data", not a measured zero. */
  prompt_tps: number | null;
  completion_tps: number | null;
  /** Total prompt tokens in the bucket — the tooltip's "how many tokens". */
  prompt_tokens: number;
  completion_tokens: number;
  /** Measured cached prompt tokens in the bucket (0 when none were measured). */
  cached_tokens: number;
  /** Requests of the bucket whose engine reported cached tokens, and all of
   *  its requests. measured 0 = nothing claimed about caching (pre-0042 rows,
   *  or an engine that does not report it); unmeasured requests count as
   *  computed. */
  cached_measured_requests: number;
  requests: number;
  /** Mean tok/s of the cached part of the prompt. null unless some request
   *  in the bucket was measured, so an unmeasured bucket draws no cached bar. */
  prompt_cached_tps: number | null;
  /** The busiest minute's prompt rate (max per-minute total / 60), the
   *  whisker on wide bars. null on 1-min buckets (it would equal the bar)
   *  and on buckets with no rows. */
  prompt_peak_tps: number | null;
  completion_peak_tps: number | null;
  /** True when the bucket overlaps [measuredFrom, last complete minute]:
   *  a 0 rate there is a measured zero, worth drawing as a zero-height bar. */
  measured: boolean;
}

/**
 * Dense tokens/second buckets over a minute-keyed token series.
 *
 * `nowMinute` is the minute still being written: a bucket divides by its
 * COMPLETE minutes only, and a trailing bucket with no complete minute is
 * not drawn. A bucket before `measuredFromMinute` is null (the warden kept
 * no samples then); when the API does not send the field (older builds), the
 * first point's minute is the left bound. The bucket that CONTAINS
 * `measuredFromMinute` divides by its minutes from that boundary onward —
 * the minutes before it have no samples by definition, so counting them
 * would read the rate up to 2x low. Rows of the in-progress minute are
 * dropped from the sums, like the throughput endpoint's right bound.
 */
export function tokenRateBuckets(
  points: readonly StatsV2TokensPoint[],
  {
    bucketMinutes,
    sinceMinute,
    nowMinute,
    measuredFromMinute,
  }: {
    bucketMinutes: number;
    sinceMinute: number;
    nowMinute: number;
    /** null/undefined when the API is older than the field: fall back to the
     *  first point's minute (nothing is measured when there are no points). */
    measuredFromMinute: number | null | undefined;
  },
): TokenRateBucket[] {
  const bm = Math.max(1, Math.floor(bucketMinutes));
  // points arrive minute-ascending from the API, so the first is the earliest.
  const measuredFrom =
    measuredFromMinute ?? (points.length > 0 ? points[0].minute : null);

  // Sum per minute first (the busiest-MINUTE peak needs per-minute totals),
  // then fold each minute into its bucket.
  type Sum = {
    prompt: number;
    completion: number;
    cached: number;
    measured: number;
    requests: number;
  };
  const zero = (): Sum => ({ prompt: 0, completion: 0, cached: 0, measured: 0, requests: 0 });
  const perMinute = new Map<number, Sum>();
  for (const p of points) {
    if (p.minute < sinceMinute || p.minute >= nowMinute) continue;
    const acc = perMinute.get(p.minute) ?? zero();
    acc.prompt += p.prompt;
    acc.completion += p.completion;
    acc.cached += p.cached ?? 0;
    acc.measured += p.cached_measured_requests ?? 0;
    acc.requests += p.requests ?? 0;
    perMinute.set(p.minute, acc);
  }
  const sums = new Map<number, Sum>();
  const peaks = new Map<number, { prompt: number; completion: number }>();
  for (const [minute, v] of perMinute) {
    const key = Math.floor(minute / bm) * bm;
    const s = sums.get(key) ?? zero();
    s.prompt += v.prompt;
    s.completion += v.completion;
    s.cached += v.cached;
    s.measured += v.measured;
    s.requests += v.requests;
    sums.set(key, s);
    const pk = peaks.get(key) ?? { prompt: -1, completion: -1 };
    pk.prompt = Math.max(pk.prompt, v.prompt);
    pk.completion = Math.max(pk.completion, v.completion);
    peaks.set(key, pk);
  }

  const rows: TokenRateBucket[] = [];
  const firstKey = Math.floor(sinceMinute / bm) * bm;
  const lastKey = Math.floor(nowMinute / bm) * bm;
  for (let key = firstKey; key <= lastKey; key += bm) {
    // Complete minutes in this bucket: the whole width, except the trailing
    // bucket, which caps at nowMinute (the minute being written is not one).
    const complete = Math.min(key + bm, nowMinute) - key;
    if (complete <= 0) continue; // trailing bucket, no complete minute
    // Of those, the minutes that HAVE SAMPLES: minutes before
    // measuredFromMinute never had a row (it is MIN(minute) over the same
    // table), so counting them would dilute a mid-bucket boundary's rate by
    // up to 2x. Measured minutes without rows stay in: a quiet minute is a
    // measured zero, not missing data.
    const measuredSpan =
      measuredFrom === null
        ? complete
        : Math.min(key + bm, nowMinute) - Math.max(key, measuredFrom);
    const s = sums.get(key) ?? zero();
    const pk = peaks.get(key);
    const measured =
      measuredFrom !== null &&
      Math.max(key, measuredFrom) < Math.min(key + bm, nowMinute);
    rows.push({
      minute: key,
      prompt_tps: measured ? s.prompt / (measuredSpan * 60) : null,
      completion_tps: measured ? s.completion / (measuredSpan * 60) : null,
      prompt_tokens: s.prompt,
      completion_tokens: s.completion,
      cached_tokens: s.cached,
      cached_measured_requests: s.measured,
      requests: s.requests,
      prompt_cached_tps:
        measured && s.measured > 0 ? Math.min(s.cached, s.prompt) / (measuredSpan * 60) : null,
      prompt_peak_tps:
        measured && bucketMinutes > 1 && pk ? pk.prompt / 60 : null,
      completion_peak_tps:
        measured && bucketMinutes > 1 && pk ? pk.completion / 60 : null,
      measured,
    });
  }
  return rows;
}

export interface CacheHitSummary {
  /** cached / prompt over the window, as 0..1. null when no request in the
   *  window reported cached tokens (nothing to claim). */
  pct: number | null;
  cached: number;
  prompt: number;
  measuredRequests: number;
  requests: number;
}

/**
 * Cache hit over the last `minutes` COMPLETE minutes. Requests whose engine
 * did not report cached tokens count as computed, so the figure is a floor
 * when coverage is partial (`measuredRequests` vs `requests` says by how much).
 */
export function cacheHitSummary(
  points: readonly StatsV2TokensPoint[],
  { nowMinute, minutes = 60 }: { nowMinute: number; minutes?: number },
): CacheHitSummary {
  let cached = 0;
  let prompt = 0;
  let measuredRequests = 0;
  let requests = 0;
  for (const p of points) {
    if (p.minute >= nowMinute || p.minute < nowMinute - minutes) continue;
    cached += p.cached ?? 0;
    prompt += p.prompt;
    measuredRequests += p.cached_measured_requests ?? 0;
    requests += p.requests ?? 0;
  }
  return {
    pct: measuredRequests > 0 && prompt > 0 ? Math.min(1, cached / prompt) : null,
    cached,
    prompt,
    measuredRequests,
    requests,
  };
}

// ---- the fixed 24h reference ----------------------------------------------
//
// Dashed = median of BUSY minutes, dotted = peak, both over a FIXED 24 hours
// regardless of the selected window: a reference that moved with the view
// could never answer "is this normal for this box?".

/**
 * Median over the samples where the host was working, or null.
 *
 * Mirror of app/stats/history.py::busy_median — including the deliberate
 * choice of a median over busy minutes rather than a mean over everything
 * (these series are mostly idle with bursts; a mean lands between the two
 * states and describes neither), and including "no busy minutes → null"
 * rather than a baseline invented from idle samples. Like the backend's
 * strict=False zip, a shorter sequence narrows the median instead of
 * throwing.
 */
export function busyMedian(
  values: readonly number[],
  busy: readonly boolean[],
): number | null {
  const n = Math.min(values.length, busy.length);
  const picked: number[] = [];
  for (let i = 0; i < n; i++) if (busy[i]) picked.push(values[i]);
  if (picked.length === 0) return null;
  picked.sort((a, b) => a - b);
  const mid = picked.length >> 1;
  if (picked.length % 2) return picked[mid];
  return (picked[mid - 1] + picked[mid]) / 2;
}

/**
 * Highest sample over the WHOLE period, busy or not — mirror of
 * app/stats/history.py::peak. Deliberately not masked: a spike during an
 * otherwise-quiet minute is still the peak.
 */
export function peakOf(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  return maxOf(values);
}

export interface Reference {
  /** 24h median of busy minutes, or null when the box was never busy. */
  median: number | null;
  /** 24h peak over every sample, or null with no samples. */
  peak: number | null;
  /** How many minutes of the 24h were busy — the caption's tooltip. */
  busyMinutes: number;
}

/**
 * The host-level busy mask, decided ONCE and applied to every series.
 *
 * A minute is busy when the box did work in it: GPU util above a floor, or
 * any completion tokens. Per-series thresholds would be meaningless for VRAM,
 * which does not fall when the box goes quiet — the weights stay resident.
 */
export function hostBusyMinutes(
  util: readonly { minute: number; max_pct: number }[],
  tokens: readonly { minute: number; completion: number }[],
): Set<number> {
  const busy = new Set<number>();
  for (const u of util) if (u.max_pct > 5) busy.add(u.minute);
  for (const t of tokens) if (t.completion > 0) busy.add(t.minute);
  return busy;
}

/** busyMedian + peakOf over one minute-keyed series, against the host mask. */
export function referenceFor<T extends { minute: number }>(
  points: readonly T[],
  value: (p: T) => number,
  busy: ReadonlySet<number>,
): Reference {
  const values = points.map((p) => value(p));
  const mask = points.map((p) => busy.has(p.minute));
  return {
    median: busyMedian(values, mask),
    peak: peakOf(values),
    busyMinutes: mask.filter(Boolean).length,
  };
}

// ---- windowed histogram deltas --------------------------------------------

/**
 * Two cumulative histogram reads into "what happened between them", or null.
 *
 * Mirror of app/stats/live_engine.py::bucket_deltas, every refusal included:
 * no earlier read (the lifetime total under a "last 5 minutes" heading is the
 * mislabelling this page was rebuilt to remove), counters that went backwards
 * (engine restart), boundaries that moved (engine upgrade), and a count that
 * shrank. A window in which nothing happened is NOT null: it is a
 * distribution whose counts are all zero, so the panel can say "no requests
 * in this window" instead of going blank.
 *
 * The returned counts stay CUMULATIVE along the boundaries (a difference of
 * cumulatives is cumulative), which is exactly what quantileFromBuckets
 * expects; per-bar rendering takes successive differences.
 */
export function bucketDeltas(
  newer: HistogramBuckets | null | undefined,
  older: HistogramBuckets | null | undefined,
): HistogramBuckets | null {
  if (!newer || !older) return null;
  const a = newer.le ?? [];
  const b = older.le ?? [];
  if (a.length !== b.length || a.some((le, i) => le !== b[i])) return null;
  const nc = newer.counts ?? [];
  const oc = older.counts ?? [];
  if (nc.length !== oc.length) return null;
  if (nc.some((x, i) => x < oc[i])) return null;
  if (newer.count === null || older.count === null || newer.count < older.count) {
    return null;
  }
  return {
    le: [...a],
    counts: nc.map((x, i) => x - oc[i]),
    count: newer.count - older.count,
    sum: Math.max(0, (newer.sum ?? 0) - (older.sum ?? 0)),
  };
}

/**
 * Sum histograms across models. Only histograms whose boundaries are
 * IDENTICAL are combined — counts of requests are additive, differently-
 * edged bins are not. Returns the combined histogram plus how many of the
 * inputs actually contributed, so the panel can carry its provenance the way
 * every combined number on this page does.
 */
export function combineHistograms(
  hists: readonly (HistogramBuckets | null)[],
): { combined: HistogramBuckets | null; contributing: number; total: number } {
  const present = hists.filter((h): h is HistogramBuckets => h !== null);
  if (present.length === 0) {
    return { combined: null, contributing: 0, total: hists.length };
  }
  const base = present[0];
  const matching = present.filter(
    (h) =>
      h.le.length === base.le.length && h.le.every((le, i) => le === base.le[i]),
  );
  const combined: HistogramBuckets = {
    le: [...base.le],
    counts: base.counts.map((_, i) =>
      matching.reduce((s, h) => s + (h.counts[i] ?? 0), 0),
    ),
    count: matching.reduce((s, h) => s + (h.count ?? 0), 0),
    sum: matching.reduce((s, h) => s + (h.sum ?? 0), 0),
  };
  return { combined, contributing: matching.length, total: hists.length };
}

/**
 * Interpolated quantile from cumulative buckets — mirror of
 * app/stats/prometheus.py::hist_quantile, with `le: null` standing in for
 * +Inf. Returns null for an empty or all-zero histogram; a rank landing in
 * the open-ended top bucket answers with the last finite boundary.
 */
export function quantileFromBuckets(
  hist: HistogramBuckets | null,
  q: number,
): number | null {
  if (!hist || hist.counts.length === 0) return null;
  const total = hist.counts[hist.counts.length - 1];
  if (!total || total <= 0) return null;
  const rank = q * total;
  let prevLe = 0;
  let prevC = 0;
  for (let i = 0; i < hist.counts.length; i++) {
    const le = hist.le[i];
    const c = hist.counts[i];
    if (rank <= c) {
      if (le === null) return prevC > 0 ? prevLe : null;
      if (c <= prevC) return le;
      const frac = (rank - prevC) / (c - prevC);
      return prevLe + frac * (le - prevLe);
    }
    if (le !== null) prevLe = le;
    prevC = c;
  }
  return prevLe;
}

/** Per-bucket increments for drawing bars from cumulative counts. */
export function bucketIncrements(hist: HistogramBuckets): number[] {
  return hist.counts.map((c, i) => (i === 0 ? c : c - hist.counts[i - 1]));
}

// ---- the 5-minute live ring ------------------------------------------------
//
// Keyed on wall time, never on the request set, so idle is a real span of the
// chart rather than a collapsed nothing. One sample lands per SSE frame; the
// evict keeps the span bounded however fast frames arrive.

export const TIMELINE_SPAN_MS = 5 * 60_000;

export interface TimelineSample {
  at: number; // epoch ms of the frame
  /** Combined generation tok/s over the models in scope at APPEND time is
   *  wrong — scope changes at render time — so the sample keeps PER-MODEL
   *  readings and the chart combines the selected ones when it draws. */
  perModel: Record<
    string,
    {
      gen: number | null;
      running: number | null;
      waiting: number | null;
    }
  >;
}

/** Append one sample and evict everything older than the span. Pure. */
export function pushSample(
  ring: readonly TimelineSample[],
  sample: TimelineSample,
): TimelineSample[] {
  const cutoff = sample.at - TIMELINE_SPAN_MS;
  return [...ring.filter((s) => s.at >= cutoff), sample];
}

export interface TimelinePoint {
  at: number;
  /** Sum over selected models that REPORT it; null when none do. */
  gen: number | null;
  running: number | null;
  waiting: number | null;
  /** True when every selected model reports zero running — "nothing
   *  happened here", drawn as an explicit idle band. */
  idle: boolean;
}

/** Combine a ring down to the selected models, null-not-zero throughout. */
export function combineTimeline(
  ring: readonly TimelineSample[],
  selectedIds: readonly string[],
): TimelinePoint[] {
  return ring.map((s) => {
    let gen: number | null = null;
    let running: number | null = null;
    let waiting: number | null = null;
    for (const id of selectedIds) {
      const m = s.perModel[id];
      if (!m) continue;
      if (m.gen !== null) gen = (gen ?? 0) + m.gen;
      if (m.running !== null) running = (running ?? 0) + m.running;
      if (m.waiting !== null) waiting = (waiting ?? 0) + m.waiting;
    }
    return { at: s.at, gen, running, waiting, idle: (running ?? 0) === 0 };
  });
}

// ---- per-model histogram snapshots for the 5-minute delta window -----------

export interface BucketSnapshot {
  at: number; // epoch ms
  buckets: HistogramBuckets;
}

/**
 * Append the newest read and evict snapshots that have aged out of the
 * window (keeping one older-than-window snapshot as the delta baseline, so
 * the delta spans the WHOLE window rather than only what remains inside it).
 */
export function pushSnapshot(
  ring: readonly BucketSnapshot[],
  snap: BucketSnapshot,
  spanMs: number = TIMELINE_SPAN_MS,
): BucketSnapshot[] {
  const next = [...ring, snap];
  const cutoff = snap.at - spanMs;
  // Keep the newest snapshot at-or-before the cutoff as the baseline.
  let firstInside = next.findIndex((s) => s.at > cutoff);
  if (firstInside < 0) firstInside = next.length - 1;
  const baselineIdx = Math.max(0, firstInside - 1);
  return next.slice(baselineIdx);
}

/**
 * The windowed distribution: newest read minus the ring's baseline, with
 * every refusal bucketDeltas encodes. Null until the ring has two reads.
 */
export function windowedDeltas(
  ring: readonly BucketSnapshot[],
): HistogramBuckets | null {
  if (ring.length < 2) return null;
  return bucketDeltas(ring[ring.length - 1].buckets, ring[0].buckets);
}
