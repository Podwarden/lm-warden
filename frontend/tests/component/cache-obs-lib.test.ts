import { describe, it, expect } from "vitest";
import { dominantCause, efficiency, fmtPct, fromCache, isEstimated, outcomeLabel, topProblem, type CacheSummary } from "@/lib/cache-obs";

const base: CacheSummary = {
  model_id: "m", model: "m", backend: "vllm", requests: 100, prompt_tokens: 100_000,
  cached_tokens: 40_000, known_prompt_tokens: 100_000, measured_requests: 100, estimated_requests: 0,
  reused_tokens: 40_000, reusable_tokens: 50_000,
  outcomes: { hit: 60, partial: 5, lost: 20, misrouted: 0, diverged: 5, cold: 10, compacted: 0, harness_rewrite: 0 },
  by_rank: [], prefill_saved_s: 20, rate_source: "learned", top_diverging: [],
};

describe("cache-obs helpers", () => {
  it("ratios", () => {
    expect(fromCache(base)).toBeCloseTo(0.4);
    expect(efficiency(base)).toBeCloseTo(0.8);
  });
  it("null, never 0, when nothing is known", () => {
    const s = { ...base, measured_requests: 0, estimated_requests: 0, cached_tokens: 0, known_prompt_tokens: 0, reused_tokens: 0, reusable_tokens: 0 };
    expect(fromCache(s)).toBeNull();
    expect(efficiency(s)).toBeNull();
    expect(fmtPct(null, false)).toBe("—");
  });
  it("estimate marker", () => {
    expect(isEstimated({ ...base, estimated_requests: 3 })).toBe(true);
    expect(fmtPct(0.42, true)).toBe("≈42%");
  });
  it("per-backend lost wording", () => {
    expect(topProblem(base, { lens: "operator" })?.text).toMatch(/LRU eviction/);
    expect(topProblem({ ...base, backend: "llamacpp" }, { lens: "operator" })?.text).toMatch(/slot/);
    expect(topProblem({ ...base, backend: "mlx" }, { lens: "operator" })?.text).toMatch(/prompt cache/);
  });
  it("no hint below 15%", () => {
    const s = { ...base, outcomes: { ...base.outcomes, lost: 10 } };
    expect(topProblem(s, { lens: "operator" })).toBeNull();
  });
  it("own lens speaks to the client developer and never mentions other keys", () => {
    const s = { ...base, outcomes: { ...base.outcomes, lost: 0, diverged: 30 }, top_diverging: undefined };
    const h = topProblem(s, { lens: "own" });
    expect(h?.text).toMatch(/your prompt/i);
    expect(h?.text).not.toMatch(/token /i);
  });
  it("own lens never surfaces a populated top_diverging name", () => {
    const s = {
      ...base,
      outcomes: { ...base.outcomes, lost: 0, diverged: 30 },
      top_diverging: [{ token_id: "tid-9", token_name: "secret-key", requests: 30 }],
    };
    const own = topProblem(s, { lens: "own" });
    expect(own?.text).not.toMatch(/secret-key|tid-9/);
    expect(topProblem(s, { lens: "operator" })?.text).toMatch(/secret-key/);
  });
  it("efficiency divides the reused tokens by the reusable ones of the same rows", () => {
    // 50 requests C=9000,R=0 and 50 requests C=0,R=9000: half reused, not 100%
    // cached_tokens also counts a pre-0045 row (C known, R unknown) that efficiency must skip
    const s = { ...base, cached_tokens: 459_000, reused_tokens: 450_000, reusable_tokens: 900_000 };
    expect(efficiency(s)).toBeCloseTo(0.5);
  });
  it("efficiency is not clamped and is null on an empty denominator", () => {
    expect(efficiency({ ...base, reused_tokens: 60_000, reusable_tokens: 50_000 })).toBeCloseTo(1.2);
    expect(efficiency({ ...base, reused_tokens: 0, reusable_tokens: 0 })).toBeNull();
  });
  it("from cache divides by the prompts of rows with a known C only", () => {
    const s = { ...base, prompt_tokens: 1_000_000, known_prompt_tokens: 100_000, cached_tokens: 40_000 };
    expect(fromCache(s)).toBeCloseTo(0.4);
    expect(fromCache({ ...base, known_prompt_tokens: 0 })).toBeNull();
  });
  it("operator diverged hint names the key, its count and the fix", () => {
    const s = {
      ...base,
      outcomes: { ...base.outcomes, lost: 0, diverged: 30 },
      top_diverging: [{ token_id: "t1", token_name: "batch-job", requests: 27 }],
    };
    expect(topProblem(s, { lens: "operator" })?.text).toBe(
      'Key "batch-job" rewrites the start of its prompt in 27 requests. Move changing data (time, ids) to the end.',
    );
  });
  it("own lens lost hint blames other traffic or pauses", () => {
    expect(topProblem(base, { lens: "own" })?.text).toBe(
      "Your requests often find their cached prefix gone: other traffic or long pauses let the engine evict it.",
    );
  });
  it("cobs2 outcome labels are lowercase like the others", () => {
    expect(outcomeLabel("compacted")).toBe("compacted");
    expect(outcomeLabel("harness_rewrite")).toBe("harness rewrote history");
  });
  it("a summary without the cobs2 keys gives a hint, not NaN shares", () => {
    const { compacted: _c, harness_rewrite: _h, ...older } = base.outcomes;
    const h = topProblem({ ...base, outcomes: older as CacheSummary["outcomes"] }, { lens: "operator" });
    expect(h?.outcome).toBe("lost");
    expect(Number.isNaN(h?.share)).toBe(false);
  });
  it("harness rewrites are a problem with their own hint, in both lenses", () => {
    const s = { ...base, outcomes: { ...base.outcomes, lost: 0, harness_rewrite: 30 } };
    for (const lens of ["operator", "own"] as const) {
      const h = topProblem(s, { lens });
      expect(h?.outcome).toBe("harness_rewrite");
      expect(h?.text).toBe(
        "The client rewrote earlier messages (reminders, dates, dropped reasoning); the cache can't reuse past that point.",
      );
    }
  });
  it("compaction is informational, never the top problem", () => {
    const s = { ...base, outcomes: { ...base.outcomes, lost: 0, diverged: 0, compacted: 60 } };
    expect(topProblem(s, { lens: "operator" })).toBeNull();
    expect(topProblem(s, { lens: "own" })).toBeNull();
  });
  it("dominant cause: the larger known cause, when it covers most broken prefixes", () => {
    const s = { ...base, deep_diverged: 4, outcomes: { ...base.outcomes, diverged: 2 } };
    expect(dominantCause({ ...s, divergence_causes: { compacted: 0, harness_rewrite: 4 } })).toBe("harness_rewrite");
    expect(dominantCause({ ...s, divergence_causes: { compacted: 5, harness_rewrite: 1 } })).toBe("compacted");
    // 2 of 6 is not "most"
    expect(dominantCause({ ...s, divergence_causes: { compacted: 0, harness_rewrite: 2 } })).toBeNull();
    expect(dominantCause({ ...s, divergence_causes: undefined })).toBeNull();
    expect(dominantCause({ ...s, divergence_causes: { compacted: 0, harness_rewrite: 0 } })).toBeNull();
  });
  it("a tie names no cause", () => {
    const s = { ...base, deep_diverged: 4, outcomes: { ...base.outcomes, diverged: 0 } };
    expect(dominantCause({ ...s, divergence_causes: { compacted: 3, harness_rewrite: 3 } })).toBeNull();
  });
  it("compacted hits outside the broken prefixes never make a compaction dominant", () => {
    // 80 hits (15 compacted, no divergence), 20 diverged with no cause, 1 deep-diverged:
    // the server counts causes over the 21 broken rows only, so none is a compaction
    const s = {
      ...base,
      deep_diverged: 1,
      outcomes: { ...base.outcomes, hit: 80, partial: 0, lost: 0, diverged: 20, cold: 0 },
      divergence_causes: { compacted: 0, harness_rewrite: 0 },
    };
    expect(dominantCause(s)).toBeNull();
    expect(topProblem(s, { lens: "operator" })?.text).not.toMatch(/compaction/);
  });
  it("the diverged hint names the dominant cause in one sentence", () => {
    const s = {
      ...base,
      outcomes: { ...base.outcomes, lost: 0, diverged: 30 },
      divergence_causes: { compacted: 0, harness_rewrite: 25 },
    };
    const own = topProblem(s, { lens: "own" })?.text ?? "";
    expect(own).toMatch(/Most of these follow the harness rewriting earlier messages\.$/);
    expect(own.split("\n")).toHaveLength(1);
  });
});
