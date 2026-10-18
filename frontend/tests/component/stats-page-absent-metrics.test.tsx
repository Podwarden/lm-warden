// A metric the engine does not report must never render as 0.
//
// Design spec §9.3: a user who reads past a degradation and acts on it has
// been actively misled. "0% KV usage" says *plenty of headroom*; "0 tok/s"
// says *the engine is idle*. For llama.cpp both are false — the truth is
// "this engine is silent about that" — and an operator who trusts either has
// been given a confidently precise wrong answer.
//
// Originally written against /stats/live; retargeted at the merged /stats
// page, which inherits every one of these obligations: the per-model KV row,
// the combined throughput figure, the sleep-state chip and the latency
// panels all say "not reported" (naming the engine) rather than zero.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, cleanup, waitFor } from "@testing-library/react";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

vi.mock("@/lib/live-stats-stream", async (orig) => {
  const actual = await orig<typeof import("@/lib/live-stats-stream")>();
  const { mockStream } = await import("./stats-merged-stream");
  return { ...actual, useLiveStats: () => mockStream.state };
});

import {
  FakeResizeObserver,
  SILENT_LLAMACPP,
  block,
  frameWith,
  installFetchStub,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

const VLLM = block("model-a-8b", {
  engine: {
    num_requests_running: 3,
    num_requests_waiting: 2,
    waiting_by_reason: { capacity: 2, deferred: 0 },
    kv_cache_usage_perc: 0.5,
    kv_tokens_used: 112400,
    kv_tokens_total: 224800,
    engine_sleep_state: 0,
    preemptions_total: 12,
    preemptions_per_s: 1.0,
  },
  throughput: {
    prompt_tokens_per_s: 500,
    generation_tokens_per_s: 50,
    prompt_tokens_total: 1000000,
    generation_tokens_total: 500000,
  },
});

describe("merged stats with a backend that reports fewer metrics", () => {
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

  it("renders absent KV usage as not-reported, never as 0%", async () => {
    setFrame(frameWith([SILENT_LLAMACPP]));
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("kv-usage")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("kv-usage").textContent ?? "").toMatch(
      /not reported/i,
    );
    expect(screen.queryByText("0%")).toBeNull();
  });

  it("names the engine that is silent", async () => {
    setFrame(frameWith([SILENT_LLAMACPP]));
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByText(/llama\.cpp/i).length).toBeGreaterThan(0),
    );
  });

  it("gives a llama.cpp model latency distributions from the proxy's own measurements", async () => {
    // llama.cpp reports NO histograms. The distributions no longer come from
    // the engine: TTFT and duration are measured by the proxy for every
    // backend and persisted, so a GGUF model gets the panels rather than
    // "not reported" — and never a distribution invented from zeros.
    setFrame(frameWith([SILENT_LLAMACPP]));
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("ttft-histogram")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("itl-histogram")).toBeInTheDocument();
    expect(screen.getByTestId("duration-histogram")).toBeInTheDocument();
    expect(screen.getByTestId("ttft-panel").textContent).not.toMatch(/not reported/i);
    expect(screen.getByTestId("latency-section").textContent).toMatch(/proxy-measured/i);
  });

  it("does not render '0 tok/s' for an engine that publishes no rate", async () => {
    setFrame(frameWith([SILENT_LLAMACPP]));
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("timeline-panel")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("timeline-panel").textContent).not.toMatch(
      /\b0 tok\/s now/,
    );
  });

  it("never prints the string 'null'", async () => {
    setFrame(frameWith([SILENT_LLAMACPP]));
    const { container } = renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("kv-usage")).toBeInTheDocument(),
    );
    expect(container.textContent ?? "").not.toMatch(/\bnull\b/);
  });

  it("renders an absent sleep state as a dash, not 'sleep null'", async () => {
    setFrame(frameWith([SILENT_LLAMACPP]));
    const { container } = renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("kv-usage")).toBeInTheDocument(),
    );
    expect(container.textContent ?? "").not.toContain("sleep null");
  });

  it("still renders a vLLM frame exactly as before", async () => {
    setFrame(frameWith([VLLM]));
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("kv-usage")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("kv-usage").textContent ?? "").toContain("50%");
    // The reported rate reaches the combined figure in the timeline heading.
    expect(screen.getByTestId("timeline-panel").textContent).toContain(
      "50 tok/s now",
    );
  });

  it("keeps the sleep-state chip for a vLLM frame", async () => {
    setFrame(frameWith([VLLM]));
    const { container } = renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("kv-usage")).toBeInTheDocument(),
    );
    expect(container.textContent ?? "").toContain("awake");
  });
});
