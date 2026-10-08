// The pure helpers behind the merged /stats page.
//
// Three of these are MIRRORS of committed backend code and the tests pin the
// mirrored semantics with the backend's own worked examples:
//
//   busyMedian / peakOf     ← app/stats/history.py (test_history.py)
//   bucketDeltas            ← app/stats/live_engine.py (test_latency_buckets.py)
//   quantileFromBuckets     ← app/stats/prometheus.py (test_live_engine.py)

import { describe, it, expect } from "vitest";
import {
  bucketDeltas,
  bucketIncrements,
  bucketPoints,
  busyMedian,
  combineHistograms,
  combineTimeline,
  hostBusyMinutes,
  peakOf,
  pushSample,
  pushSnapshot,
  quantileFromBuckets,
  cacheHitSummary,
  tokenRateBuckets,
  windowedDeltas,
  WINDOW_META,
  TIMELINE_SPAN_MS,
} from "@/lib/live-history";
import type { TokenRateBucket } from "@/lib/live-history";
import type { HistogramBuckets } from "@/lib/live-stats";

const hist = (
  le: (number | null)[],
  counts: number[],
  count: number | null = counts[counts.length - 1] ?? null,
  sum: number | null = 0,
): HistogramBuckets => ({ le, counts, count, sum });

// ---------------------------------------------------------------------------
// busyMedian / peakOf — history.py mirror
// ---------------------------------------------------------------------------

describe("busyMedian", () => {
  it("takes the median over BUSY samples only", () => {
    // Mostly idle with bursts: a mean over everything would land between the
    // two states and describe neither.
    const values = [0, 0, 300, 320, 0, 340, 0];
    const busy = [false, false, true, true, false, true, false];
    expect(busyMedian(values, busy)).toBe(320);
  });

  it("averages the middle pair for an even count", () => {
    expect(busyMedian([10, 20, 30, 40], [true, true, true, true])).toBe(25);
  });

  it("returns null when nothing was busy — no invented baseline", () => {
    expect(busyMedian([5, 6, 7], [false, false, false])).toBeNull();
  });

  it("narrows, not throws, when the sequences differ in length", () => {
    expect(busyMedian([1, 2, 3], [true])).toBe(1);
  });
});

describe("peakOf", () => {
  it("is over the WHOLE period, busy or not", () => {
    // A spike during an otherwise-quiet minute is still the peak — VRAM does
    // not track business at all, and masking would hide the one event worth
    // seeing.
    expect(peakOf([1, 99, 2])).toBe(99);
  });
  it("is null with no samples", () => {
    expect(peakOf([])).toBeNull();
  });
});

describe("hostBusyMinutes", () => {
  it("is decided once at host level: util OR completions", () => {
    const busy = hostBusyMinutes(
      [
        { minute: 1, max_pct: 80 },
        { minute: 2, max_pct: 0 },
        { minute: 3, max_pct: 3 },
      ],
      [
        { minute: 2, completion: 100 },
        { minute: 3, completion: 0 },
      ],
    );
    expect(busy.has(1)).toBe(true); // util
    expect(busy.has(2)).toBe(true); // completions
    expect(busy.has(3)).toBe(false); // neither
  });
});

// ---------------------------------------------------------------------------
// bucketDeltas — live_engine.py mirror, every refusal included
// ---------------------------------------------------------------------------

describe("bucketDeltas", () => {
  const older = hist([0.1, 1, null], [5, 10, 12], 12, 20);
  const newer = hist([0.1, 1, null], [8, 20, 25], 25, 55);

  it("subtracts cumulative reads into a windowed distribution", () => {
    const d = bucketDeltas(newer, older)!;
    expect(d.counts).toEqual([3, 10, 13]);
    expect(d.count).toBe(13);
    expect(d.sum).toBe(35);
  });

  it("refuses when there is no earlier read", () => {
    // The lifetime total under a "last 5 minutes" heading is precisely the
    // mislabelling the merged page removes.
    expect(bucketDeltas(newer, null)).toBeNull();
  });

  it("refuses counters that went backwards (engine restart)", () => {
    expect(bucketDeltas(older, newer)).toBeNull();
  });

  it("refuses boundaries that moved (engine upgrade)", () => {
    const moved = hist([0.2, 1, null], [8, 20, 25], 25, 55);
    expect(bucketDeltas(moved, older)).toBeNull();
  });

  it("keeps an all-zero window as a real distribution, not null", () => {
    // "No requests in this window" is a different and more useful statement
    // than a blank panel.
    const d = bucketDeltas(older, older)!;
    expect(d.counts).toEqual([0, 0, 0]);
    expect(d.count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// quantileFromBuckets — prometheus.py mirror, with le:null as +Inf
// ---------------------------------------------------------------------------

describe("quantileFromBuckets", () => {
  // The backend's own worked example (test_hist_quantile_interpolates).
  const e2e = hist([1, 5, 10, 30, 60, null], [0, 20, 60, 90, 99, 100], 100, 0);

  it("interpolates inside the straddling bucket", () => {
    expect(quantileFromBuckets(e2e, 0.5)).toBeCloseTo(8.75);
    expect(quantileFromBuckets(e2e, 0.9)).toBeCloseTo(30.0);
    expect(quantileFromBuckets(e2e, 0.99)).toBeCloseTo(60.0);
  });

  it("returns null for an empty or all-zero histogram", () => {
    expect(quantileFromBuckets(hist([], []), 0.5)).toBeNull();
    expect(quantileFromBuckets(hist([1, null], [0, 0], 0), 0.5)).toBeNull();
    expect(quantileFromBuckets(null, 0.5)).toBeNull();
  });

  it("answers the open-ended top bucket with the last finite boundary", () => {
    const h = hist([1, null], [50, 100], 100, 0);
    expect(quantileFromBuckets(h, 0.99)).toBe(1);
  });
});

describe("bucketIncrements", () => {
  it("turns cumulative counts into per-bar increments", () => {
    expect(bucketIncrements(hist([1, 2, null], [3, 10, 12]))).toEqual([3, 7, 2]);
  });
});

// ---------------------------------------------------------------------------
// combineHistograms — additive only when the edges are identical
// ---------------------------------------------------------------------------

describe("combineHistograms", () => {
  it("sums matching-edged histograms with provenance", () => {
    const a = hist([1, null], [2, 4], 4, 1);
    const b = hist([1, null], [1, 1], 1, 2);
    const out = combineHistograms([a, b]);
    expect(out.combined?.counts).toEqual([3, 5]);
    expect(out.contributing).toBe(2);
    expect(out.total).toBe(2);
  });

  it("excludes a differently-edged histogram rather than mis-binning it", () => {
    const a = hist([1, null], [2, 4], 4, 1);
    const b = hist([2, null], [1, 1], 1, 2);
    const out = combineHistograms([a, b]);
    expect(out.combined?.counts).toEqual([2, 4]);
    expect(out.contributing).toBe(1);
  });

  it("is null with provenance when nothing reports (llama.cpp)", () => {
    const out = combineHistograms([null, null]);
    expect(out.combined).toBeNull();
    expect(out.contributing).toBe(0);
    expect(out.total).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// WINDOW_META — the bucket width each range draws at
// ---------------------------------------------------------------------------

describe("WINDOW_META", () => {
  it("buckets 1h at 1 min, 6h at 2 min, 24h at 10 min and 7d at 60 min", () => {
    // Wide enough that a ~1,000 px panel keeps a visible gap between bars
    // (the old 6h/24h/7d widths gave 360/288/336 bars, ~3 px each), and wide
    // enough that the tokens/second whiskers stay legible on 24h and 7d.
    expect(WINDOW_META["1h"].bucketMinutes).toBe(1);
    expect(WINDOW_META["1h"].bucketLabel).toBe("1 min buckets");
    expect(WINDOW_META["6h"].bucketMinutes).toBe(2);
    expect(WINDOW_META["6h"].bucketLabel).toBe("2 min buckets");
    expect(WINDOW_META["24h"].bucketMinutes).toBe(10);
    expect(WINDOW_META["24h"].bucketLabel).toBe("10 min buckets");
    expect(WINDOW_META["7d"].bucketMinutes).toBe(60);
    expect(WINDOW_META["7d"].bucketLabel).toBe("60 min buckets");
  });
});

// ---------------------------------------------------------------------------
// bucketPoints — the window's bucket-up
// ---------------------------------------------------------------------------

describe("bucketPoints", () => {
  const pts = [
    { minute: 100, v: 1 },
    { minute: 101, v: 5 },
    { minute: 105, v: 3 },
  ];

  it("returns the input untouched for 1-minute buckets", () => {
    expect(bucketPoints(pts, 1, (b) => b[0])).toBe(pts);
  });

  it("folds points into fixed-width buckets keyed on the first minute", () => {
    const out = bucketPoints(pts, 5, (b) => ({
      minute: b[0].minute,
      v: Math.max(...b.map((p) => p.v)),
    }));
    expect(out).toEqual([
      { minute: 100, v: 5 },
      { minute: 105, v: 3 },
    ]);
  });
});

// ---------------------------------------------------------------------------
// tokenRateBuckets — the dense mean-rate rollup for the Tokens / second chart
// ---------------------------------------------------------------------------

describe("tokenRateBuckets — cached prompt tokens", () => {
  const opts = { bucketMinutes: 5, sinceMinute: 0, nowMinute: 20, measuredFromMinute: 0 };

  it("folds cached, measured and request counts into the bucket", () => {
    const out = tokenRateBuckets(
      [
        { minute: 5, prompt: 6000, completion: 0, cached: 3000, cached_measured_requests: 2, requests: 3 },
        { minute: 7, prompt: 3000, completion: 0, cached: 600, cached_measured_requests: 1, requests: 1 },
      ],
      opts,
    );
    const b = out.find((x) => x.minute === 5)!;
    expect(b.cached_tokens).toBe(3600);
    expect(b.cached_measured_requests).toBe(3);
    expect(b.requests).toBe(4);
    expect(b.prompt_cached_tps).toBeCloseTo(3600 / 300, 6);
  });

  it("claims nothing for buckets with no measured request (pre-migration)", () => {
    const out = tokenRateBuckets([{ minute: 5, prompt: 6000, completion: 0 }], opts);
    const b = out.find((x) => x.minute === 5)!;
    expect(b.cached_tokens).toBe(0);
    expect(b.cached_measured_requests).toBe(0);
    expect(b.prompt_cached_tps).toBeNull();
  });

  it("summarises the last hour and says null when nothing was measured", () => {
    const pts = [
      { minute: 10, prompt: 1000, completion: 0, cached: 400, cached_measured_requests: 1, requests: 2 },
      { minute: 11, prompt: 1000, completion: 0, cached: 100, cached_measured_requests: 1, requests: 1 },
      { minute: 19, prompt: 9999, completion: 0, cached: 9999, cached_measured_requests: 1, requests: 1 },
    ];
    const s = cacheHitSummary(pts, { nowMinute: 12 });
    expect(s.pct).toBeCloseTo(0.25, 6);
    expect(s.measuredRequests).toBe(2);
    expect(s.requests).toBe(3);
    expect(cacheHitSummary([{ minute: 1, prompt: 5, completion: 0 }], { nowMinute: 5 }).pct).toBeNull();
  });
});

describe("tokenRateBuckets", () => {
  const pt = (minute: number, prompt: number, completion: number) => ({
    minute,
    prompt,
    completion,
  });

  it("divides a sparse wide bucket by its FULL width, not its busy minutes", () => {
    // The 7d bug this helper exists to fix: one 600,000-token minute alone in
    // a 30-min bucket is 600,000 / (30 x 60) = 333.3 tok/s, not the 10,000
    // tok/s that averaging only the minutes that HAVE rows implies.
    const out = tokenRateBuckets([pt(65, 600_000, 0)], {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 120,
      measuredFromMinute: 0,
    });
    const bucket = out.find((b) => b.minute === 60)!;
    expect(bucket.prompt_tps).toBeCloseTo(600_000 / 1800, 4);
    expect(bucket.prompt_tps).toBeLessThan(1000); // not the busy-minute mean
    expect(bucket.prompt_tokens).toBe(600_000);
    expect(bucket.completion_tps).toBe(0);
    expect(bucket.measured).toBe(true);
  });

  it("is null before the measured range and a measured zero inside it", () => {
    const out = tokenRateBuckets([], {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 120,
      measuredFromMinute: 60,
    });
    expect(out.map((b) => b.minute)).toEqual([0, 30, 60, 90]);
    for (const b of out.slice(0, 2)) {
      expect(b.prompt_tps).toBeNull();
      expect(b.completion_tps).toBeNull();
      expect(b.prompt_peak_tps).toBeNull();
      expect(b.measured).toBe(false);
    }
    for (const b of out.slice(2)) {
      // No rows inside the measured range is a real 0, not "no data".
      expect(b.prompt_tps).toBe(0);
      expect(b.completion_tps).toBe(0);
      expect(b.prompt_tokens).toBe(0);
      expect(b.measured).toBe(true);
    }

    // measuredFromMinute missing (an older API) falls back to the first
    // point's minute: bucket 60 holds it and is measured, bucket 30 is not.
    const fb = tokenRateBuckets([pt(75, 300, 0)], {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 120,
      measuredFromMinute: undefined,
    });
    expect(fb.find((b) => b.minute === 30)!.measured).toBe(false);
    expect(fb.find((b) => b.minute === 60)!.measured).toBe(true);
  });

  it("divides a mid-bucket measuredFromMinute by its measured minutes only", () => {
    // measuredFromMinute 75 sits INSIDE bucket [60, 90), not on a boundary:
    // minutes 60..74 have no rows at all (it is MIN(minute) over the
    // samples), so the bucket's rate runs over minutes 75..89 only — 15 real
    // minutes of 60 tokens/min (1 tok/s) must read 1 tok/s, not the 0.5 the
    // full 30-min width implies.
    const rows = Array.from({ length: 15 }, (_, i) => pt(75 + i, 60, 30));
    const out = tokenRateBuckets(rows, {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 90,
      measuredFromMinute: 75,
    });
    const bucket = out.find((b) => b.minute === 60)!;
    expect(bucket.measured).toBe(true);
    expect(bucket.prompt_tokens).toBe(15 * 60);
    expect(bucket.prompt_tps).toBeCloseTo((15 * 60) / (15 * 60), 4);
    expect(bucket.completion_tps).toBeCloseTo((15 * 30) / (15 * 60), 4);
    // The buckets before the measured range stay "no data".
    expect(out.find((b) => b.minute === 0)!.prompt_tps).toBeNull();
    expect(out.find((b) => b.minute === 30)!.measured).toBe(false);

    // The same boundary mid-TRAILING-bucket: the divisor is the measured
    // minutes that are also complete (75..79 with nowMinute 80), not width.
    const trailing = tokenRateBuckets(rows.slice(0, 5), {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 80,
      measuredFromMinute: 75,
    }).find((b) => b.minute === 60)!;
    expect(trailing.prompt_tps).toBeCloseTo((5 * 60) / (5 * 60), 4);

    // Controls: measuredFromMinute on the bucket start, or before the whole
    // window, keeps the full width — minutes 60..74 are then measured zeros,
    // not missing data, so the sparse-bucket rule still applies.
    for (const m of [60, 0]) {
      const aligned = tokenRateBuckets(rows, {
        bucketMinutes: 30,
        sinceMinute: 0,
        nowMinute: 90,
        measuredFromMinute: m,
      }).find((b) => b.minute === 60)!;
      expect(aligned.prompt_tps).toBeCloseTo((15 * 60) / (30 * 60), 4);
    }
  });

  it("divides the trailing bucket by its complete minutes only, and omits it with none", () => {
    // nowMinute 100: the trailing bucket [90, 120) has 10 complete minutes
    // (90..99), not 30 — the current minute is still being written.
    const out = tokenRateBuckets([pt(95, 60_000, 0)], {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 100,
      measuredFromMinute: 0,
    });
    expect(out).toHaveLength(4);
    const trailing = out.find((b) => b.minute === 90)!;
    expect(trailing.prompt_tps).toBeCloseTo(60_000 / (10 * 60), 4);

    // nowMinute on a bucket edge: the trailing bucket has NO complete minute
    // and is not drawn.
    const none = tokenRateBuckets([pt(80, 600_000, 0)], {
      bucketMinutes: 30,
      sinceMinute: 60,
      nowMinute: 90,
      measuredFromMinute: 60,
    });
    expect(none).toHaveLength(1);
    expect(none[0].minute).toBe(60);
    expect(none[0].prompt_tps).toBeCloseTo(600_000 / (30 * 60), 4);
  });

  it("sets the busiest-minute peak only for buckets wider than one minute", () => {
    const narrow: TokenRateBucket[] = tokenRateBuckets(
      [pt(10, 600, 120), pt(11, 300, 60)],
      {
        bucketMinutes: 1,
        sinceMinute: 10,
        nowMinute: 12,
        measuredFromMinute: 10,
      },
    );
    expect(narrow).toHaveLength(2);
    for (const b of narrow) {
      // On a 1-min bucket the peak equals the bar, so it is not carried.
      expect(b.prompt_peak_tps).toBeNull();
      expect(b.completion_peak_tps).toBeNull();
    }
    expect(narrow[0].prompt_tps).toBe(600 / 60);

    const wide = tokenRateBuckets(
      [pt(62, 600_000, 6_000), pt(70, 300_000, 3_000)],
      {
        bucketMinutes: 30,
        sinceMinute: 0,
        nowMinute: 120,
        measuredFromMinute: 0,
      },
    );
    const bucket = wide.find((b) => b.minute === 60)!;
    // Mean over the full 30 min; peak over the busiest single minute.
    expect(bucket.prompt_tps).toBeCloseTo(900_000 / 1800, 4);
    expect(bucket.prompt_peak_tps).toBe(600_000 / 60);
    expect(bucket.completion_peak_tps).toBe(6_000 / 60);
    // A measured bucket with no rows has no busiest minute.
    expect(wide.find((b) => b.minute === 90)!.prompt_peak_tps).toBeNull();
  });

  it("is dense: one row per bucket in the window, ascending, rows or not", () => {
    // One stray row must not thin the grid: idle buckets are real zero bars,
    // not gaps the x-axis jumps over.
    const out = tokenRateBuckets([pt(85, 60, 0)], {
      bucketMinutes: 30,
      sinceMinute: 0,
      nowMinute: 90,
      measuredFromMinute: 0,
    });
    expect(out.map((b) => b.minute)).toEqual([0, 30, 60]);
    expect(out[2].prompt_tokens).toBe(60);
    expect(out[0].prompt_tokens).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// the 5-minute rings
// ---------------------------------------------------------------------------

describe("pushSample / combineTimeline", () => {
  it("evicts beyond the span and keeps per-model readings", () => {
    let ring = pushSample([], { at: 0, perModel: { a: { gen: 5, running: 1, waiting: 0 } } });
    ring = pushSample(ring, {
      at: TIMELINE_SPAN_MS + 1000,
      perModel: { a: { gen: 7, running: 2, waiting: 0 } },
    });
    expect(ring).toHaveLength(1);
    expect(ring[0].at).toBe(TIMELINE_SPAN_MS + 1000);
  });

  it("combines selected models null-not-zero", () => {
    const ring = pushSample([], {
      at: 0,
      perModel: {
        a: { gen: 5, running: 1, waiting: 0 },
        b: { gen: null, running: 1, waiting: null },
      },
    });
    const [p] = combineTimeline(ring, ["a", "b"]);
    expect(p.gen).toBe(5); // b's silence contributes nothing, not 0
    expect(p.running).toBe(2);
    expect(p.waiting).toBe(0);
    const [onlyB] = combineTimeline(ring, ["b"]);
    expect(onlyB.gen).toBeNull(); // nobody reports → null, never 0
    expect(onlyB.idle).toBe(false); // running 1
  });

  it("marks idle when every selected model reports zero running", () => {
    const ring = pushSample([], {
      at: 0,
      perModel: { a: { gen: 0, running: 0, waiting: 0 } },
    });
    expect(combineTimeline(ring, ["a"])[0].idle).toBe(true);
  });
});

describe("pushSnapshot / windowedDeltas", () => {
  const at = (s: number) => s * 1000;
  const snap = (s: number, c: number) => ({
    at: at(s),
    buckets: hist([1, null], [c, c], c, 0),
  });

  it("needs two reads before it answers", () => {
    const ring = pushSnapshot([], snap(0, 5));
    expect(windowedDeltas(ring)).toBeNull();
  });

  it("keeps one beyond-window snapshot as the delta baseline", () => {
    let ring = pushSnapshot([], snap(0, 5));
    ring = pushSnapshot(ring, snap(100, 7));
    ring = pushSnapshot(ring, snap(400, 9)); // 0s aged out; 100s is baseline
    expect(ring[0].at).toBe(at(100));
    const d = windowedDeltas(ring)!;
    expect(d.count).toBe(2); // 9 - 7
  });
});
