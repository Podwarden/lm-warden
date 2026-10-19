// Tests for the load-progress SWR hooks (Task 8 / fix round 1).
//
// Covers F2: ``useActiveLoadProgress(enabled)`` must not poll the all-models
// endpoint when ``enabled`` is false, and must start polling when it flips to
// true.  The per-model ``useLoadProgress`` hook is already covered by the
// page integration tests; here we test the list-page hook in isolation so
// the gating is pinned independently of the page render.

import { describe, it, expect, afterEach, vi } from "vitest";
import { renderHook, cleanup, act } from "@testing-library/react";
import { SWRConfig } from "swr";
import {
  useActiveLoadProgress,
  useLoadProgress,
} from "@/lib/load-progress";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

// SWR cache isolation: each test gets a fresh Map so prior fetches do not
// satisfy new hooks.  ``dedupingInterval: 0`` so a re-render re-evaluates.
function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      {children}
    </SWRConfig>
  );
}

function stubFetch(urls: Record<string, unknown>) {
  const calls: string[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    calls.push(url);
    const body = urls[url] ?? "[]";
    return new Response(JSON.stringify(body), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useActiveLoadProgress — gated on enabled", () => {
  it("does not fetch when enabled is false", () => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    const calls = stubFetch({
      "/api/models/load-progress": [],
    });
    renderHook(() => useActiveLoadProgress(false), { wrapper });
    expect(
      calls.filter((u) => u === "/api/models/load-progress"),
    ).toHaveLength(0);
  });

  it("fetches when enabled is true", async () => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    const calls = stubFetch({
      "/api/models/load-progress": [
        { model_id: "m1", backend: "vllm", elapsed_s: 10, state: "progressing", stalled_for_s: 0, current_step: "WEIGHTS", steps: [], vitals: { series: [], log_quiet_normal: false, gpu_mem_total_mib: null } },
      ],
    });
    const { result } = renderHook(() => useActiveLoadProgress(true), {
      wrapper,
    });
    // Wait for the fetch to land.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(
      calls.filter((u) => u === "/api/models/load-progress").length,
    ).toBeGreaterThan(0);
    expect(result.current).toBeDefined();
    expect(result.current?.length).toBe(1);
  });

  it("starts polling when enabled flips from false to true", async () => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    const calls = stubFetch({
      "/api/models/load-progress": [],
    });
    const { rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) => useActiveLoadProgress(enabled),
      { wrapper, initialProps: { enabled: false } },
    );
    // No fetch while disabled.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(
      calls.filter((u) => u === "/api/models/load-progress"),
    ).toHaveLength(0);

    // Flip to enabled — fetch fires.
    rerender({ enabled: true });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(
      calls.filter((u) => u === "/api/models/load-progress").length,
    ).toBeGreaterThan(0);
  });
});

describe("useLoadProgress — gated on status === loading", () => {
  it("does not fetch when status is not loading", () => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    const calls = stubFetch({
      "/api/models/m1/load-progress": {
        model_id: "m1", backend: "vllm", elapsed_s: 10, state: "progressing", stalled_for_s: 0, current_step: "WEIGHTS", steps: [], vitals: { series: [], log_quiet_normal: false, gpu_mem_total_mib: null },
      },
    });
    renderHook(() => useLoadProgress("m1", "loaded"), { wrapper });
    expect(
      calls.filter((u) => u === "/api/models/m1/load-progress"),
    ).toHaveLength(0);
  });

  it("fetches when status is loading", async () => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    const calls = stubFetch({
      "/api/models/m1/load-progress": {
        model_id: "m1", backend: "vllm", elapsed_s: 10, state: "progressing", stalled_for_s: 0, current_step: "WEIGHTS", steps: [], vitals: { series: [], log_quiet_normal: false, gpu_mem_total_mib: null },
      },
    });
    const { result } = renderHook(() => useLoadProgress("m1", "loading"), {
      wrapper,
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 10));
    });
    expect(
      calls.filter((u) => u === "/api/models/m1/load-progress").length,
    ).toBeGreaterThan(0);
    expect(result.current?.model_id).toBe("m1");
  });
});
