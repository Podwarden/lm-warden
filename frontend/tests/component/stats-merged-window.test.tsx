// The WINDOW governs every history panel, the latency distributions and the
// requests chart. The latter two read the persisted per-request store, which
// is what let them stop saying "not the window". The 5-minute timeline is
// live engine state, cannot honour 7d, and still says so rather than showing
// a narrower span than the button promises. Longer windows bucket up
// (1 min → 2 min → 10 min → 60 min) and the heading says which.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, cleanup, waitFor, fireEvent, act } from "@testing-library/react";
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

// The session forest card (three.js, its own poller) is covered by forest-card.test.tsx.
vi.mock("@/components/forest/ForestCard", () => ({ default: () => null }));

describe("merged stats — the window scopes history, not live", () => {
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

  it("labels the live timeline 'not the window' and nothing else", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("timeline-panel")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("timeline-panel").textContent).toContain(
      "not the window",
    );
    await waitFor(() =>
      expect(screen.getByTestId("ttft-panel").textContent).toContain("last hour"),
    );
    expect(screen.getByTestId("ttft-panel").textContent).not.toContain("not the window");
    expect(screen.getByTestId("itl-panel").textContent).toContain("last hour");
    expect(screen.getByTestId("duration-panel").textContent).toContain("last hour");
    await waitFor(() =>
      expect(screen.getByTestId("requests-panel").textContent).toContain("last hour"),
    );
    expect(screen.getByTestId("requests-panel").textContent).not.toContain(
      "not the window",
    );
  });

  it("re-scopes the latency distributions and the requests chart to the window", async () => {
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("ttft-panel").textContent).toContain("last hour"),
    );
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "7d" }));
    });
    await waitFor(() =>
      expect(screen.getByTestId("ttft-panel").textContent).toContain("last 7 days"),
    );
    await waitFor(() =>
      expect(screen.getByTestId("requests-scope").textContent).toContain("last 7 days"),
    );
    expect(urls.some((u) => u.startsWith("/api/stats/v2/requests?range=7d"))).toBe(true);
    expect(urls.some((u) => u.startsWith("/api/stats/v2/latency?range=7d"))).toBe(true);
  });

  it("says which bucket a long window drew, in the control bar and headings", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("scope-note")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("scope-note").textContent).toContain("1 min buckets");
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "7d" }));
    });
    await waitFor(() =>
      expect(screen.getByTestId("scope-note").textContent).toContain(
        "60 min buckets",
      ),
    );
    expect(screen.getByTestId("chart-util").textContent).toContain(
      "last 7 days",
    );
    expect(screen.getByTestId("chart-util").textContent).toContain(
      "60 min buckets",
    );
  });

  it("leaves the live panels untouched by a window change", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("timeline-panel")).toBeInTheDocument(),
    );
    const before = screen.getByTestId("timeline-panel").textContent;
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "7d" }));
    });
    // The history charts re-fetch; the live panel neither reloads nor
    // relabels — 5 minutes does not become 7 days because a button said so.
    await waitFor(() =>
      expect(screen.getByTestId("scope-note").textContent).toContain(
        "60 min buckets",
      ),
    );
    expect(screen.getByTestId("timeline-panel").textContent).toBe(before);
  });

  it("requests the fixed 24h reference regardless of the selected window", async () => {
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(urls.some((u) => u.includes("range=24h"))).toBe(true),
    );
    // 1h view still fetches the 24h series the reference is computed from.
    expect(urls.some((u) => u.includes("range=1h"))).toBe(true);
  });

  it("fetches the prompt cache unfiltered when no model is loaded", async () => {
    // nothing loaded anywhere: no overview models, no live frame
    const urls = installFetchStub({ models: [] });
    setFrame(null);
    renderPage();
    await waitFor(() =>
      expect(urls).toContain("/api/stats/v2/cache?range=1h"),
    );
    expect(urls.some((u) => u.startsWith("/api/stats/v2/cache") && u.includes("models="))).toBe(false);
  });

  it("persists the window under 'vw.stats.range'", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("range-selector")).toBeInTheDocument(),
    );
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "7d" }));
    });
    expect(window.localStorage.getItem("vw.stats.range")).toBe("7d");
  });
});
