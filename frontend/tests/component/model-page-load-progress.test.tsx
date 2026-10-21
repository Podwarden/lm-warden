// Integration tests for the model detail page's load-progress + last_error
// gating (Task 8).
//
// Covers the brief's failing-test list for the page:
//   - loading model never shows last_error
//   - failed model shows red last_error
//
// The page fetches ``GET /api/models/{id}`` (the row) and, while loading,
// ``GET /api/models/{id}/load-progress`` (the snapshot).  Both are stubbed.

import { Suspense } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, act, fireEvent } from "@testing-library/react";
import { SWRConfig } from "swr";
import { ModelDetailPage } from "@/components/models/model-detail-page";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

class InertEventSource {
  close() {}
  addEventListener() {}
  removeEventListener() {}
}

function fakeModel(overrides: Record<string, unknown> = {}) {
  return {
    id: "abc",
    served_model_name: "llama3-8b",
    hf_repo: "meta-llama/Llama-3-8B",
    hf_revision: "main",
    gpu_indices: [0],
    tensor_parallel_size: 1,
    backend: "vllm",
    mmproj_filename: null,
    n_gpu_layers: null,
    dtype: null,
    max_model_len: 8192,
    gpu_memory_utilization: 0.9,
    trust_remote_code: false,
    extra_args: [],
    extra_env: {},
    status: "loading",
    pulled_bytes: 1,
    pulled_total: 1,
    last_error: null,
    ...overrides,
  };
}

function fakeProgress(overrides: Record<string, unknown> = {}) {
  return {
    model_id: "abc",
    backend: "vllm",
    layout: null,
    elapsed_s: 30,
    state: "progressing",
    stalled_for_s: 0,
    give_up_at: null,
    reason: null,
    hint: null,
    usual_total_s: null,
    ready_eta: null,
    current_step: "WEIGHTS",
    steps: [
      { key: "START", label: "Start engine", status: "done", duration_s: 5, elapsed_s: null, count: null, total: null, usual_s: null },
      { key: "WEIGHTS", label: "Load weights", status: "running", duration_s: null, elapsed_s: 25, count: 3, total: 8, usual_s: null },
    ],
    vitals: { series: [], last_log_age_s: 5, last_log_line: "Loading shards", log_quiet_normal: false, gpu_mem_total_mib: null, gpu_mem_supported: true },
    ...overrides,
  };
}

async function renderWith(model: Record<string, unknown>, progress: Record<string, unknown> | null = null) {
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    if (url === "/api/models/abc") {
      return new Response(JSON.stringify(model), { status: 200 });
    }
    if (url === "/api/models/abc/load-progress") {
      if (progress === null) return new Response('{"detail":"no active load"}', { status: 404 });
      return new Response(JSON.stringify(progress), { status: 200 });
    }
    if (url === "/api/models/abc/capabilities") {
      return new Response(JSON.stringify({ runs: [] }), { status: 200 });
    }
    return new Response("{}", { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div>loading</div>}>
        <ModelDetailPage id="abc" />
      </Suspense>
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByText(/Engine:/);
}

describe("ModelDetailPage — load-progress and last_error gating", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("EventSource", InertEventSource as unknown as typeof EventSource);
    // jsdom does not implement scrollIntoView; stub it so the "View log"
    // action does not throw.
    Element.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("loading model never shows last_error", async () => {
    // Even with a non-null last_error (from a prior failed load), a loading
    // row must NOT show the red box — the colour rule from the owner.
    await renderWith(
      fakeModel({ status: "loading", last_error: "old error from a prior load" }),
      fakeProgress(),
    );
    expect(screen.queryByTestId("last-error-box")).toBeNull();
    // The checklist is shown instead.
    expect(screen.getByTestId("load-checklist")).toBeInTheDocument();
  });

  it("failed model shows red last_error", async () => {
    await renderWith(
      fakeModel({ status: "failed", last_error: "engine crashed" }),
      null, // no active load-progress (the load is over)
    );
    expect(screen.getByTestId("last-error-box")).toBeInTheDocument();
    expect(screen.getByTestId("last-error-box")).toHaveTextContent("engine crashed");
  });

  it("failed model with GPUs released does not show the Force unload link", async () => {
    await renderWith(
      fakeModel({ status: "failed", last_error: "load timed out — GPUs released" }),
      null,
    );
    expect(screen.getByTestId("last-error-box")).toBeInTheDocument();
    expect(screen.queryByTestId("failed-force-unload-link")).toBeNull();
  });

  it("failed model where the engine still holds GPUs shows the Force unload link", async () => {
    await renderWith(
      fakeModel({ status: "failed", last_error: "engine crashed" }),
      null,
    );
    expect(screen.getByTestId("failed-force-unload-link")).toBeInTheDocument();
  });

  it("renders the checklist while loading with a progressing snapshot", async () => {
    await renderWith(fakeModel({ status: "loading" }), fakeProgress());
    expect(screen.getByTestId("load-checklist")).toBeInTheDocument();
    expect(screen.getByTestId("load-status-pill")).toHaveAttribute("data-state", "progressing");
  });

  it("renders the collapsible Live logs (collapsed by default)", async () => {
    await renderWith(fakeModel({ status: "loading" }), fakeProgress());
    expect(screen.getByTestId("live-logs-collapsible")).toBeInTheDocument();
    expect(screen.getByTestId("live-logs-toggle")).toHaveTextContent("Show log");
    expect(screen.queryByTestId("live-logs-body")).toBeNull();
  });

  it("View log from the stalled pill opens the Live logs card", async () => {
    // The stalled pill's "View log" action must open the Live logs card as
    // well as scroll to it — a manual open that wins until the state changes.
    await renderWith(
      fakeModel({ status: "loading" }),
      fakeProgress({
        state: "stalled",
        stalled_for_s: 600,
        give_up_at: "2026-10-18T12:34:56Z",
        steps: [
          { key: "WEIGHTS", label: "Load weights", status: "stalled", duration_s: null, elapsed_s: 600, count: 3, total: 8, usual_s: null },
        ],
      }),
    );
    // When stalled, the auto-open effect opens the card.  Simulate the user
    // manually closing it first (manual toggle wins).
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
    act(() => {
      fireEvent.click(screen.getByTestId("live-logs-toggle"));
    });
    expect(screen.queryByTestId("live-logs-body")).toBeNull();

    // Now click "View log" in the stalled pill — it should open the card.
    act(() => {
      fireEvent.click(screen.getByTestId("pill-view-log"));
    });
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
  });

  it("shows a Starting placeholder before the first load-progress poll", async () => {
    // When the status is loading but the load-progress poll has not resolved
    // yet (or returns 404), a quiet "Starting…" placeholder shows in the
    // status card instead of an empty gap.
    await renderWith(fakeModel({ status: "loading" }), null);
    expect(screen.getByTestId("load-starting-placeholder")).toBeInTheDocument();
    expect(screen.getByTestId("load-starting-placeholder")).toHaveTextContent(
      "Starting…",
    );
    // The checklist must not render yet.
    expect(screen.queryByTestId("load-checklist")).toBeNull();
  });
});
