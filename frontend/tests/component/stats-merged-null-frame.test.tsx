// A null frame must not crash the page.
//
// `_null_frame` on the backend emits engine/throughput/cache/latency/finished
// as JSON null while KEEPING model_id, so the block reaches the per-model
// render path. The old /stats/live page's types declared `engine`
// non-nullable — false — and EngineHero dereferenced
// `frame.engine.kv_cache_usage_perc` unguarded, so a scrape error could take
// down the whole page (there was no error boundary either). This pins the
// guard on the merged page, and the honest banner copy: nothing holds a
// "last good" frame — the stream replaces state on every message — so the
// banner must not claim one is being shown.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, cleanup, waitFor } from "@testing-library/react";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";
import type { LiveEngineFrame } from "@/lib/live-stats";

vi.mock("@/lib/live-stats-stream", async (orig) => {
  const actual = await orig<typeof import("@/lib/live-stats-stream")>();
  const { mockStream } = await import("./stats-merged-stream");
  return { ...actual, useLiveStats: () => mockStream.state };
});

import {
  FakeResizeObserver,
  block,
  frameWith,
  installFetchStub,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

/** Exactly what app/stats/live_engine.py::_null_frame emits. */
function nullFrame(name: string): LiveEngineFrame {
  return {
    ts: "2026-09-05T19:00:00Z",
    model: name,
    model_id: `id-${name}`,
    backend: "vllm",
    max_model_len: 8192,
    engine: null,
    throughput: null,
    cache: null,
    latency: null,
    finished: null,
    scrape_error: "connect ECONNREFUSED 127.0.0.1:8001",
  };
}

describe("merged stats — a null frame from a failed scrape", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    installFetchStub();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the page rather than crashing on null engine data", async () => {
    setFrame(frameWith([nullFrame("model-a-8b")]));
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("stats-page")).toBeInTheDocument(),
    );
    // The model's row exists, attributed, with an explicit no-data state —
    // never a fabricated "0%" reading.
    const row = screen.getByTestId("model-context-row");
    expect(row.textContent).toContain("model-a-8b");
    expect(screen.getByTestId("kv-usage").textContent).toMatch(/no engine data/i);
    expect(screen.queryByText("0%")).toBeNull();
  });

  it("names the failure and does not claim last good values", async () => {
    setFrame(frameWith([nullFrame("model-a-8b")]));
    renderPage();
    const banner = await screen.findByText(
      /metrics scrape failed for model-a-8b/i,
    );
    expect(banner.textContent).toContain("ECONNREFUSED");
    // The old banner claimed "Showing the last good values" while nothing
    // held one. The copy now says what actually happens.
    expect(banner.textContent).not.toMatch(/last good values/i);
  });

  it("keeps healthy models rendering beside a broken one", async () => {
    setFrame(frameWith([block("model-a-8b"), nullFrame("model-b-27b")]));
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-context-row")).toHaveLength(2),
    );
    // The healthy engine's KV number still renders.
    const kv = screen.getAllByTestId("kv-usage").map((n) => n.textContent ?? "");
    expect(kv.some((t) => t.includes("40%"))).toBe(true);
    expect(kv.some((t) => /no engine data/i.test(t))).toBe(true);
  });

  it("never prints the string 'null'", async () => {
    setFrame(frameWith([nullFrame("model-a-8b")]));
    const { container } = renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("stats-page")).toBeInTheDocument(),
    );
    expect(container.textContent ?? "").not.toMatch(/\bnull\b/);
  });
});
