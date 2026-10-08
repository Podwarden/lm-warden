// R27: the tokens panel is now tokens/SECOND as binned bars: each bar is the
// mean rate over its bucket (bucket token sum / complete minutes x 60), null
// buckets (before the measured range) draw nothing, and wide buckets carry a
// whisker up to their busiest minute. Prompt and completion stay two stacked
// small multiples with INDEPENDENT axes (a prompt-heavy box ran 350,352
// prompt against 36 completion in the same minute), shared X domain, and
// synchronised hover (recharts syncId).
//
// These tests pin: one bar per non-null bucket per panel; the 24h references
// converted to tok/s (the per-minute values divided by 60) and relabelled
// "24h peak minute" / "24h busy-minute median"; no rectangle for a null
// bucket while a measured zero still draws; a busiest-minute whisker on wide
// buckets and none on 1-min buckets; and the per-series axis ceilings from
// the pure `tokensYMax` helper — completion's from completion alone,
// prompt's from the existing ceilingWith rule against the 24h peak / 60.

import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
process.env.TZ = "UTC";
import { render, screen, cleanup, within } from "@testing-library/react";
import { TokensChart, promptSplit, tokensYMax } from "@/components/stats/v2-charts";
import type { Reference, TokenRateBucket } from "@/lib/live-history";

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 600 });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get: () => 300 });
  if (!("PointerEvent" in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
});
afterAll(() => {
  delete (HTMLElement.prototype as unknown as { clientWidth?: number }).clientWidth;
  delete (HTMLElement.prototype as unknown as { clientHeight?: number }).clientHeight;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// R19: the ResizeObserver stub and the getBoundingClientRect spy live in a
// per-test beforeEach (not beforeAll) because vitest's per-test global
// clear (unstubGlobals: true / restoreMocks: true) runs BEFORE each test and
// would otherwise drop a beforeAll-installed stub/spy before the first test
// that needs it.
beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  // recharts' ResponsiveContainer takes its INITIAL size from
  // getBoundingClientRect (the fake ResizeObserver never fires); without a
  // non-zero rect the inner chart renders empty and its bars/axes are absent
  // from the jsdom DOM.
  const rect = {
    left: 0, top: 0, width: 600, height: 300, right: 600, bottom: 300, x: 0, y: 0,
    toJSON: () => ({}),
  } as DOMRect;
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(rect);
});
afterEach(() => cleanup());

// A bucket row, unmeasured ("no data") unless overridden.
function bucket(minute: number, over: Partial<TokenRateBucket> = {}): TokenRateBucket {
  return {
    minute,
    prompt_tps: null,
    completion_tps: null,
    prompt_tokens: 0,
    completion_tokens: 0,
    cached_tokens: 0,
    cached_measured_requests: 0,
    requests: 0,
    prompt_cached_tps: null,
    prompt_peak_tps: null,
    completion_peak_tps: null,
    measured: false,
    ...over,
  };
}

// `count` consecutive complete minutes ending at now - (fromAgo - count + 1).
function recentMinutes(count: number, fromAgo = 5): number[] {
  const now = Math.floor(Date.now() / 60_000);
  return Array.from({ length: count }, (_, i) => now - fromAgo + i);
}

const REFERENCE: Reference = { median: 12_000, peak: 250_000, busyMinutes: 42 };

// Two unmeasured buckets (null) then two measured 1-min buckets, in 1h.
function oneHourBuckets(): TokenRateBucket[] {
  const m = recentMinutes(4, 4);
  return [
    bucket(m[0]),
    bucket(m[1]),
    bucket(m[2], { prompt_tps: 100, completion_tps: 2, prompt_tokens: 6_000, completion_tokens: 120, measured: true }),
    bucket(m[3], { prompt_tps: 50, completion_tps: 1, prompt_tokens: 3_000, completion_tokens: 60, measured: true }),
  ];
}

// Three measured 30-min buckets, each with a busiest minute above the mean.
function wideBuckets(): TokenRateBucket[] {
  const now = Math.floor(Date.now() / 60_000);
  return [120, 90, 60].map((ago, i) =>
    bucket(now - ago, {
      prompt_tps: 10 + i,
      completion_tps: 0.5,
      prompt_tokens: 18_000 + 600 * i,
      completion_tokens: 90,
      prompt_peak_tps: 60 + i,
      completion_peak_tps: 2,
      measured: true,
    }),
  );
}

describe("TokensChart — binned bars in tok/s", { timeout: 20_000 }, () => {
  it("draws one bar per non-null bucket in each panel", () => {
    render(<TokensChart buckets={oneHourBuckets()} bucketMinutes={1} range="1h" reference={REFERENCE} />);
    const prompt = screen.getByTestId("tokens-prompt-chart");
    const completion = screen.getByTestId("tokens-completion-chart");
    // Two charts, not two bars in one: each element holds its own chart.
    expect(prompt.querySelectorAll(".recharts-wrapper").length).toBeGreaterThanOrEqual(1);
    expect(completion.querySelectorAll(".recharts-wrapper").length).toBeGreaterThanOrEqual(1);
    // The two unmeasured buckets draw nothing; the two measured ones, one bar each.
    expect(prompt.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
    expect(completion.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
  });

  it("labels each chart in its series colour (prompt #a78bfa, completion #34d399)", () => {
    render(<TokensChart buckets={oneHourBuckets()} bucketMinutes={1} range="1h" reference={REFERENCE} />);
    const promptLabel = within(screen.getByTestId("tokens-prompt-chart")).getByText("Prompt");
    const completionLabel = within(screen.getByTestId("tokens-completion-chart")).getByText("Completion");
    expect(promptLabel.getAttribute("style") ?? "").toMatch(/#a78bfa|167,\s*139,\s*250/i);
    expect(completionLabel.getAttribute("style") ?? "").toMatch(/#34d399|52,\s*211,\s*153/i);
  });

  it("keeps the 24h reference lines on the PROMPT chart only", () => {
    render(<TokensChart buckets={oneHourBuckets()} bucketMinutes={1} range="1h" reference={REFERENCE} />);
    const prompt = screen.getByTestId("tokens-prompt-chart");
    const completion = screen.getByTestId("tokens-completion-chart");
    expect(prompt.querySelectorAll(".recharts-reference-line").length).toBe(2); // peak + median
    expect(completion.querySelectorAll(".recharts-reference-line").length).toBe(0);
  });

  it("labels the references in tok/s, the per-minute values divided by 60", () => {
    // REFERENCE = { median: 12_000, peak: 250_000 } per minute -> 200 and
    // 4,166.7 tok/s -> "200" and "4.2k" under the axis's k rule.
    render(<TokensChart buckets={oneHourBuckets()} bucketMinutes={1} range="1h" reference={REFERENCE} />);
    expect(screen.getByText("24h peak minute 4.2k tok/s")).toBeInTheDocument();
    expect(screen.getByText("24h busy-minute median 200 tok/s")).toBeInTheDocument();
  });

  it("draws no rectangle for a null bucket, but a bar for a measured zero", () => {
    const m = recentMinutes(3, 3);
    const buckets = [
      bucket(m[0]), // before the measured range: null, nothing drawn
      bucket(m[1], { prompt_tps: 0, completion_tps: 0, measured: true }),
      bucket(m[2], { prompt_tps: 0, completion_tps: 0, measured: true }),
    ];
    render(<TokensChart buckets={buckets} bucketMinutes={1} range="1h" reference={undefined} />);
    const prompt = screen.getByTestId("tokens-prompt-chart");
    const completion = screen.getByTestId("tokens-completion-chart");
    expect(prompt.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
    expect(completion.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
  });

  it("draws a busiest-minute whisker on wide buckets, none on 1-min buckets", () => {
    render(<TokensChart buckets={wideBuckets()} bucketMinutes={30} range="24h" reference={REFERENCE} />);
    const prompt = screen.getByTestId("tokens-prompt-chart");
    const completion = screen.getByTestId("tokens-completion-chart");
    // Every non-null wide bucket carries its whisker; 1-min data carries none.
    expect(prompt.querySelectorAll(".recharts-errorBar")).toHaveLength(3);
    expect(completion.querySelectorAll(".recharts-errorBar")).toHaveLength(3);
    cleanup();
    render(<TokensChart buckets={oneHourBuckets()} bucketMinutes={1} range="1h" reference={REFERENCE} />);
    expect(screen.getByTestId("tokens-prompt-chart").querySelectorAll(".recharts-errorBar")).toHaveLength(0);
    expect(screen.getByTestId("tokens-completion-chart").querySelectorAll(".recharts-errorBar")).toHaveLength(0);
  });

  it("keeps ONE empty-state message for the whole panel when there is no data", () => {
    render(<TokensChart buckets={[]} bucketMinutes={1} range="1h" reference={undefined} />);
    expect(screen.getByText("No token usage in this window.")).toBeInTheDocument();
    expect(screen.queryByTestId("tokens-prompt-chart")).toBeNull();
    expect(screen.queryByTestId("tokens-completion-chart")).toBeNull();
  });
});

describe("tokensYMax — per-series tok/s ceilings", () => {
  it("derives completion's ceiling from completion ALONE, not the prompt scale", () => {
    // 350,352 prompt against 36 completion per minute (the owner's production
    // shape) in tok/s: the old shared axis put completion at ~0.
    const buckets = [bucket(1, { prompt_tps: 350_352 / 60, completion_tps: 36 / 60, measured: true })];
    const { prompt, completion } = tokensYMax(buckets, REFERENCE);
    expect(completion).toBeCloseTo((36 / 60) * 1.08, 6);
    expect(prompt).toBeCloseTo(Math.max(350_352 / 60, 250_000 / 60) * 1.08, 6);
    // and the two ceilings are actually on different scales
    expect(prompt / completion).toBeGreaterThan(100);
  });

  it("prompt's ceiling still includes the 24h peak / 60 above the window (ceilingWith rule)", () => {
    const buckets = [bucket(1, { prompt_tps: 10 / 60, completion_tps: 5 / 60, measured: true })];
    const { prompt } = tokensYMax(buckets, { median: null, peak: 500, busyMinutes: 1 });
    expect(prompt).toBeCloseTo((500 / 60) * 1.08, 6);
  });

  it("prompt's ceiling keeps the window max when it beats the 24h peak", () => {
    const buckets = [bucket(1, { prompt_tps: 10_000 / 60, completion_tps: 2 / 60, measured: true })];
    const { prompt } = tokensYMax(buckets, { median: null, peak: 500, busyMinutes: 1 });
    expect(prompt).toBeCloseTo((10_000 / 60) * 1.08, 6);
  });

  it("null buckets contribute nothing to either ceiling", () => {
    // Window max (10) above the 24h peak / 60 (8.33), so the ceiling comes
    // from the measured bucket — a null bucket leaking in as NaN would break it.
    const buckets = [
      bucket(1),
      bucket(2, { prompt_tps: 10, completion_tps: 0.5, measured: true }),
    ];
    const { prompt, completion } = tokensYMax(buckets, { median: null, peak: 500, busyMinutes: 1 });
    expect(prompt).toBeCloseTo(10 * 1.08, 6);
    expect(completion).toBeCloseTo(0.5 * 1.08, 6);
  });

  it("falls back to 1 for an all-zero series rather than a 0-height axis", () => {
    const { prompt, completion } = tokensYMax([], undefined);
    expect(prompt).toBe(1);
    expect(completion).toBe(1);
  });
});


// ---- cached vs computed prompt ---------------------------------------------

function cachedBucket(minute: number, over: Partial<TokenRateBucket> = {}): TokenRateBucket {
  return bucket(minute, {
    prompt_tps: 100,
    completion_tps: 1,
    prompt_tokens: 6_000,
    completion_tokens: 60,
    cached_tokens: 4_500,
    cached_measured_requests: 3,
    requests: 4,
    prompt_cached_tps: 75,
    measured: true,
    ...over,
  });
}

describe("TokensChart — cached vs computed prompt", { timeout: 20_000 }, () => {
  it("stacks a cached bar under the computed bar and labels both in the legend", () => {
    const m = recentMinutes(2, 3);
    render(
      <TokensChart buckets={[cachedBucket(m[0]), cachedBucket(m[1])]} bucketMinutes={1} range="1h" reference={undefined} />,
    );
    const prompt = screen.getByTestId("tokens-prompt-chart");
    // two buckets x (cached + computed)
    expect(prompt.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(4);
    const legend = within(prompt).getByTestId("tokens-legend");
    expect(legend.textContent).toContain("cached (measured)");
    expect(legend.textContent).toContain("computed");
    // completion stays a single plain bar series
    expect(screen.getByTestId("tokens-completion-chart").querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
  });

  it("draws plain computed bars, no cached bars, when nothing was measured", () => {
    const m = recentMinutes(2, 3);
    const plain = (x: number) =>
      cachedBucket(x, { cached_tokens: 0, cached_measured_requests: 0, prompt_cached_tps: null });
    render(<TokensChart buckets={[plain(m[0]), plain(m[1])]} bucketMinutes={1} range="1h" reference={undefined} />);
    expect(screen.getByTestId("tokens-prompt-chart").querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
  });

  it("shows the cache-hit summary, and 'not measured' without data", () => {
    const m = recentMinutes(1, 3);
    const summary = { pct: 0.73, cached: 73, prompt: 100, measuredRequests: 3, requests: 4 };
    render(<TokensChart buckets={[cachedBucket(m[0])]} bucketMinutes={1} range="1h" reference={undefined} cacheSummary={summary} />);
    const text = screen.getByTestId("cache-hit-summary");
    expect(text.textContent).toContain("73%");
    expect(text.textContent).toContain("last hour");
    expect(text.getAttribute("title")).toMatch(/3 of 4 requests measured/);
    cleanup();
    render(
      <TokensChart
        buckets={[cachedBucket(m[0])]}
        bucketMinutes={1}
        range="1h"
        reference={undefined}
        cacheSummary={{ pct: null, cached: 0, prompt: 0, measuredRequests: 0, requests: 0 }}
      />,
    );
    expect(screen.getByTestId("cache-hit-summary").textContent).toContain("not measured");
  });

  it("explains that tokens are credited when a request finishes", () => {
    const m = recentMinutes(1, 3);
    render(<TokensChart buckets={[cachedBucket(m[0])]} bucketMinutes={1} range="1h" reference={undefined} />);
    expect(screen.getByTestId("tokens-credit-note").getAttribute("aria-label")).toMatch(/FINISHES/);
  });
});

describe("promptSplit — what the tooltip says", () => {
  it("reports prompt, cached, computed and cached %", () => {
    expect(promptSplit(cachedBucket(1))).toEqual({
      prompt: 6_000,
      cached: 4_500,
      computed: 1_500,
      pct: 0.75,
    });
  });

  it("makes no cached claim when no request was measured", () => {
    const sp = promptSplit(cachedBucket(1, { cached_tokens: 0, cached_measured_requests: 0 }));
    expect(sp.cached).toBeNull();
    expect(sp.pct).toBeNull();
    expect(sp.computed).toBe(6_000);
  });

  it("never lets cached exceed the prompt", () => {
    expect(promptSplit(cachedBucket(1, { cached_tokens: 9_000 })).computed).toBe(0);
  });
});
