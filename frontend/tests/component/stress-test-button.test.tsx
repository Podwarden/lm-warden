// The Stress test button on the model-detail action row.
//
// Three things are worth pinning down, all of them contract rather than
// styling:
//
//   1. It is gated on `status === "loaded"`. The test probes a live engine
//      over HTTP; on any other status there is no engine to probe.
//   2. It sits BEFORE Delete. The destructive action stays last in the row,
//      and a button inserted after it would put "Delete" mid-row where a
//      mis-click is cheapest to make.
//   3. It opens the modal, which is always mounted with an `open` boolean —
//      the same pattern as DeleteModelModal.
import { Suspense } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, act, fireEvent } from "@testing-library/react";
import { SWRConfig } from "swr";
import ModelDetailPage from "@/app/models/[id]/page";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

// See tests/component/model-settings.test.tsx for why the params promise is
// forged with a resolved shape rather than awaited.
function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = "fulfilled";
  p.value = value;
  return p;
}

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
    status: "loaded",
    pulled_bytes: 1,
    pulled_total: 1,
    last_error: null,
    ...overrides,
  };
}

async function renderWith(model: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    if (url === "/api/models/abc") {
      return new Response(JSON.stringify(model), { status: 200 });
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
        <ModelDetailPage params={syncResolved({ id: "abc" })} />
      </Suspense>
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByText(/Engine:/);
}

describe("ModelDetailPage — Stress test button", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("EventSource", InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("is enabled only when the model is loaded", async () => {
    await renderWith(fakeModel({ status: "loaded" }));
    expect(screen.getByTestId("stress-test-button")).not.toBeDisabled();
    cleanup();

    // `pulled` has weights on disk but no engine process; `failed` may have
    // neither. Both are load-able, neither is probe-able.
    await renderWith(fakeModel({ status: "pulled" }));
    expect(screen.getByTestId("stress-test-button")).toBeDisabled();
    cleanup();

    await renderWith(fakeModel({ status: "failed" }));
    expect(screen.getByTestId("stress-test-button")).toBeDisabled();
  });

  it("sits before Delete so the destructive action stays last", async () => {
    await renderWith(fakeModel());
    const buttons = Array.from(document.querySelectorAll("button"));
    const stress = buttons.findIndex((b) => b.textContent === "Stress test");
    const del = buttons.findIndex((b) => b.textContent === "Delete");
    expect(stress).toBeGreaterThan(-1);
    expect(del).toBeGreaterThan(stress);
  });

  it("opens the modal, which is not rendered until then", async () => {
    await renderWith(fakeModel());
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByTestId("stress-test-button"));
    });
    expect(screen.getByRole("dialog")).toHaveTextContent("Stress test");
    expect(screen.getByTestId("stress-acknowledge")).toBeInTheDocument();
  });
});
