// The operator Cache card explains a coarse engine cache in plain words
// (2026-11-22: lmwarden.com caches in 1,584-token blocks and MTP drops one
// more, so most "lost"/"partial" requests were in fact full hits).
import { describe, it, expect, afterEach } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { AdminCacheCard } from "@/components/stats/redesign/cache-card";
import type { CacheSummary } from "@/lib/cache-obs";

const row: CacheSummary = {
  model_id: "m", model: "m", backend: "vllm", requests: 10, prompt_tokens: 40_000,
  cached_tokens: 10_000, known_prompt_tokens: 40_000, measured_requests: 10, estimated_requests: 0,
  measured_cached_tokens: 10_000, measured_prompt_tokens: 40_000,
  reused_tokens: 10_000, reusable_tokens: 30_000,
  outcomes: { hit: 8, partial: 0, lost: 1, misrouted: 0, diverged: 0, cold: 1, compacted: 0, harness_rewrite: 0 },
  by_rank: [], prefill_saved_s: 5, rate_source: "learned",
};

afterEach(cleanup);

describe("AdminCacheCard engine block note", () => {
  it("shows the block size with the full explanation as its tooltip", () => {
    render(<AdminCacheCard row={{ ...row, engine_block_tokens: 1584, engine_drops_last_block: true }} learnedRate={2000} />);
    const line = screen.getByText("engine caches in ~1.6k-token blocks");
    expect(line.getAttribute("title")).toMatch(/two blocks before anything is reused/);
  });
  it("shows the match step when it is finer than the block", () => {
    render(<AdminCacheCard row={{ ...row, engine_block_tokens: 1584, engine_match_unit_tokens: 16, engine_drops_last_block: true }} learnedRate={2000} />);
    expect(screen.getByText("engine reuses in 16-token steps").getAttribute("title")).toMatch(/every 16 tokens/);
  });
  it("shows nothing when the block size is unknown", () => {
    render(<AdminCacheCard row={row} learnedRate={2000} />);
    expect(screen.queryByText(/engine caches in/)).toBeNull();
  });
});
