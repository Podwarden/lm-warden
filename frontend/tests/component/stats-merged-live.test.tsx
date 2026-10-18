// The merged /stats page: what the model selection scopes, and what it must
// never scope.
//
// The settled rule under test: the selection scopes tokens, latency, KV,
// requests and preemptions. It does NOT scope VRAM, GPU utilisation or power
// — two engines share a card, so watts cannot be attributed to one model;
// those panels are badged `host` rather than silently re-scoping.
//
// Also pinned, carried over from the deleted /stats/live page: the one place
// figures ARE combined does not treat an unreported metric as zero.
// llama.cpp publishes no KV gauge and no throughput counters; summing its
// silence as 0 produces a fleet total that is short by an unknown amount and
// looks authoritative.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, cleanup, waitFor, fireEvent } from "@testing-library/react";
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
  boxFor,
  frameWith,
  installFetchStub,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

describe("merged stats — selection scoping", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    installFetchStub();
    setFrame(frameWith([block("model-a-8b"), SILENT_LLAMACPP]));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders one context-and-cache row per loaded model, never a combined one", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-context-row")).toHaveLength(2),
    );
    const names = screen
      .getAllByTestId("live-model-name")
      .map((n) => n.textContent);
    expect(names).toEqual(["model-a-8b", "model-b-27b"]);
  });

  it("drops a model's row when it is deselected", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("id-model-b-27b"));
    await waitFor(() =>
      expect(screen.getAllByTestId("model-context-row")).toHaveLength(1),
    );
    expect(screen.getByTestId("live-model-name").textContent).toBe("model-a-8b");
  });

  it("cannot be emptied — the last model's checkbox is disabled", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("id-model-b-27b"));
    await waitFor(() => expect(boxFor("id-model-a-8b").disabled).toBe(true));
    fireEvent.click(boxFor("id-model-a-8b"));
    expect(screen.getAllByTestId("model-context-row")).toHaveLength(1);
  });

  it("narrows the in-flight request list to the selection", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByText("10.0.0.2").length).toBeGreaterThan(0),
    );
    fireEvent.click(boxFor("id-model-b-27b"));
    await waitFor(() => expect(screen.queryAllByText("10.0.0.2")).toHaveLength(0));
    expect(screen.getAllByText("10.0.0.1").length).toBeGreaterThan(0);
  });

  it("recomputes the by-token rollup from the narrowed rows", async () => {
    // The server aggregates over EVERY in-flight request. Leaving that alone
    // would leave two views of the same instant disagreeing about which
    // requests exist.
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByText("key-b").length).toBeGreaterThan(0),
    );
    fireEvent.click(boxFor("id-model-b-27b"));
    await waitFor(() => expect(screen.queryAllByText("key-b")).toHaveLength(0));
    expect(screen.getAllByText("key-a").length).toBeGreaterThan(0);
  });

  it("does NOT scope the host KPIs by the selection", async () => {
    // VRAM, GPU util and power stay host figures — badged, not re-scoped.
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument(),
    );
    const before = screen.getByTestId("tile-vram-value").textContent;
    fireEvent.click(boxFor("id-model-b-27b"));
    await waitFor(() =>
      expect(screen.getAllByTestId("model-context-row")).toHaveLength(1),
    );
    expect(screen.getByTestId("tile-vram-value").textContent).toBe(before);
    expect(screen.getByTestId("tile-util-value").textContent).toBe("80");
    expect(screen.getByTestId("tile-power-value").textContent).toBe("250");
    // ...and the page SAYS so, next to the badge.
    expect(screen.getByText(/host \/ all models/i)).toBeInTheDocument();
  });

  it("DOES scope prefill and generation throughput", async () => {
    // The throughput stub answers 500/50 tok/s unfiltered and 200/20 narrowed,
    // so the panel's figures prove which response it read. Prefill and
    // generation are asserted separately because keeping them apart is the
    // whole reason this panel replaced the blended tokens-per-second tile.
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    await waitFor(() =>
      expect(screen.getByTestId("tp-prefill-avg").textContent).toBe("500"),
    );
    expect(screen.getByTestId("tp-generation-avg").textContent).toBe("50");

    fireEvent.click(boxFor("id-model-b-27b"));

    await waitFor(() =>
      expect(screen.getByTestId("tp-prefill-avg").textContent).toBe("200"),
    );
    expect(screen.getByTestId("tp-generation-avg").textContent).toBe("20");
  });
});

describe("merged stats — combining a mixed selection", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    installFetchStub();
    setFrame(frameWith([block("model-a-8b"), SILENT_LLAMACPP]));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("sums only the engines that report, never zeroing the silent one", async () => {
    // vLLM reports 20 tok/s; llama.cpp reports nothing. The combined figure
    // in the timeline heading is 20 FROM ONE MODEL — not an average of 10.
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("timeline-panel").textContent).toContain(
        "20 tok/s now",
      ),
    );
  });

  it("shows no rate at all when no selected engine reports one", async () => {
    setFrame(
      frameWith([
        SILENT_LLAMACPP,
        { ...SILENT_LLAMACPP, model: "other", model_id: "id-other" },
      ]),
    );
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("timeline-panel")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("timeline-panel").textContent).not.toMatch(
      /\b0 tok\/s now/,
    );
  });

  it("carries provenance on the preemptions figure for a mixed selection", async () => {
    // vLLM reports a preemption rate (0.00/s is a MEASUREMENT); llama.cpp
    // does not. The figure must say 1 of 2 models report it.
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("preempt-rate")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("preempt-rate").textContent).toContain("0.00");
    expect(screen.getByTestId("preempt-rate-partial").textContent).toMatch(
      /1 of 2 models report this/i,
    );
  });

  it("renders an em dash, never 0, when nothing reports preemptions", async () => {
    setFrame(
      frameWith([
        SILENT_LLAMACPP,
        { ...SILENT_LLAMACPP, model: "other", model_id: "id-other" },
      ]),
    );
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("preempt-rate")).toBeInTheDocument(),
    );
    const tile = screen.getByTestId("preempt-rate");
    expect(tile.textContent).toContain("—");
    expect(tile.textContent).toMatch(/not reported/i);
  });
});

describe("merged stats — throughput survives an idle engine", () => {
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

  it("still shows token throughput when no model is loaded", async () => {
    // The panel sits inside the LIVE section for layout reasons (under Context
    // and cache, beside Preemptions), but it is NOT live data — it reads the
    // persisted history, which outlives the engine. Hiding it with the live
    // panels would throw away a window the operator can still act on.
    setFrame({ ...frameWith([block("model-a-8b")]), models: [] });
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/no model loaded/i)).toBeInTheDocument(),
    );
    expect(screen.getByTestId("throughput-panel")).toBeInTheDocument();
  });
});
