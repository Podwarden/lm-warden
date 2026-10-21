// A backend with no metrics endpoint (MLX) is "not supported", not a failed
// scrape: no amber banner, a calm neutral note, and busy/idle from the
// proxy's in-flight counts the api puts in engine.num_requests_*. A real
// scrape failure on a metrics-bearing engine keeps its banner.

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
  frameWith,
  installFetchStub,
  overviewFor,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

/** What app/stats/live_engine.py::unsupported_frame emits. */
function mlxFrame(running: number, waiting = 0): LiveEngineFrame {
  return {
    ts: "2026-09-05T19:00:00Z",
    model: "qwen2.5-3b",
    model_id: "id-qwen2.5-3b",
    backend: "mlx",
    max_model_len: 8192,
    engine: {
      num_requests_running: running,
      num_requests_waiting: waiting,
      waiting_by_reason: { capacity: null, deferred: null },
      kv_cache_usage_perc: null,
      kv_tokens_used: null,
      kv_tokens_total: null,
      engine_sleep_state: null,
      preemptions_total: null,
      preemptions_per_s: null,
    },
    throughput: null,
    cache: null,
    latency: null,
    finished: null,
    scrape_error: null,
    metrics_supported: false,
    metrics_unsupported_reason: "mlx backend exposes no metrics endpoint",
  };
}

function failedVllm(): LiveEngineFrame {
  return {
    ...mlxFrame(0),
    model: "vllm-model",
    model_id: "id-vllm-model",
    backend: "vllm",
    engine: null,
    scrape_error: "connect ECONNREFUSED 127.0.0.1:8001",
    metrics_supported: undefined,
    metrics_unsupported_reason: undefined,
  };
}

describe("stats — engine without a metrics endpoint", () => {
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

  it("shows no failure banner, but a neutral note", async () => {
    setFrame(frameWith([mlxFrame(4)]));
    renderPage();
    const note = await screen.findByTestId("metrics-unsupported-note");
    expect(note.textContent).toMatch(/MLX doesn't report live engine metrics/);
    expect(note.textContent).toMatch(/LM Warden's own proxy/);
    expect(screen.queryByText(/metrics scrape failed/i)).toBeNull();
  });

  it("is busy, not Idle, when the proxy counts requests in flight", async () => {
    setFrame(frameWith([mlxFrame(4)]));
    renderPage();
    const chip = await screen.findByTestId("engine-state-chip");
    expect(chip.getAttribute("data-state")).toBe("active");
    expect(chip.textContent).toBe("Active (4 in flight)");
    expect(chip.textContent).not.toMatch(/healthy/i);
    expect(chip.getAttribute("title")).toBe(
      "Serving requests; this engine doesn't report finer health.",
    );
  });

  it("is Idle when the proxy counts nothing in flight", async () => {
    setFrame(frameWith([mlxFrame(0)]));
    renderPage();
    const chip = await screen.findByTestId("engine-state-chip");
    expect(chip.getAttribute("data-state")).toBe("idle");
  });

  it("still shows the amber banner for a real scrape error", async () => {
    setFrame(frameWith([failedVllm()]));
    renderPage();
    const banner = await screen.findByText(/metrics scrape failed for vllm-model/i);
    expect(banner.textContent).toContain("ECONNREFUSED");
    await waitFor(() =>
      expect(screen.getByTestId("stats-page")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("metrics-unsupported-note")).toBeNull();
  });
});

// Power tile: hidden only when the SERVER says this platform cannot report
// power -- not when a Linux GPU simply has no sample yet.
describe("stats — power tile follows power_supported", () => {
  function stubOverview(power_supported: boolean) {
    installFetchStub();
    const inner = globalThis.fetch;
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.startsWith("/api/stats/v2/overview")) {
        const o = overviewFor(url);
        const body = {
          ...o,
          current: { ...o.current, power_w: null, power_supported },
          series: { ...o.series, power: [] },
        };
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return inner(input, init);
    });
  }

  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("null power but supported (a GPU that just started): tile still shown", async () => {
    stubOverview(true);
    renderPage();
    await screen.findByTestId("tile-vram-value");
    expect(screen.getByTestId("tile-power-value")).toBeInTheDocument();
  });

  it("power unsupported by the platform: tile hidden", async () => {
    stubOverview(false);
    renderPage();
    await screen.findByTestId("tile-vram-value");
    expect(screen.queryByTestId("tile-power-value")).toBeNull();
    expect(screen.queryByTestId("chart-power")).toBeNull();
  });
});
