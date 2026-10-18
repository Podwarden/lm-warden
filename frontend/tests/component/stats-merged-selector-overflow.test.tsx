// The model control ADAPTS: inline chips at ≤4 models, a summary button with
// a filterable checklist popover above that, so the control bar stays ONE row
// at any model count — the fleet runs to 8+ models with names like
// `research-lab-model-h-27b-quant-xx-gguf`. Only the layout collapses; the
// behaviour (same option testids, the disabled-last rule) is unchanged.

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
  installFetchStub,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

const NAMES = [
  "model-b-27b-fp8",
  "gpt-oss-20b-gguf",
  "unsloth-model-b-27b-instruct-nvfp4",
  "accelerator-vendor-model-g-35b-a3b-fp4",
  "vendor-e-model-e-8b-instruct",
  "vendor-f-model-f-32b-distill-awq",
  "vendor-d-model-d-24b-instruct-2506",
  "research-lab-model-h-27b-quant-xx-gguf",
];
const EIGHT = NAMES.map((n) => ({
  id: `id-${n}`,
  served_model_name: n,
  gpu_indices: [0],
}));

describe("merged stats — the selector collapses above four models", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    installFetchStub({ models: EIGHT });
    setFrame(null, "connecting");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows a summary button instead of eight chips", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("model-selector-summary")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("model-selector-summary").textContent).toContain(
      "All 8 models",
    );
    expect(screen.queryAllByTestId("model-selector-option")).toHaveLength(0);
  });

  it("opens a filterable checklist with every model", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("model-selector-summary")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByTestId("model-selector-summary"));
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(8),
    );
    fireEvent.change(screen.getByTestId("model-selector-filter"), {
      target: { value: "gguf" },
    });
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    expect(
      screen
        .getAllByTestId("model-selector-option")
        .map((o) => o.getAttribute("data-model-id")),
    ).toEqual(["id-gpt-oss-20b-gguf", "id-research-lab-model-h-27b-quant-xx-gguf"]);
  });

  it("summarises a partial selection and keeps the full names in the title", async () => {
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("model-selector-summary")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByTestId("model-selector-summary"));
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(8),
    );
    const target = screen
      .getAllByTestId("model-selector-option")
      .find(
        (o) =>
          o.getAttribute("data-model-id") ===
          "id-research-lab-model-h-27b-quant-xx-gguf",
      )!;
    fireEvent.click(target.querySelector("input")!);
    await waitFor(() =>
      expect(screen.getByTestId("model-selector-summary").textContent).toContain(
        "7 of 8 models",
      ),
    );
    expect(
      screen.getByTestId("model-selector-summary").getAttribute("title"),
    ).toContain("model-b-27b-fp8");
  });
});
