// The Force unload control on the model-detail action row (#244).
//
// The contract under test is the frontend mirror of the backend's
// `_unloadable_statuses(force)` in app/models/routes_api.py:
//
//   plain        → ("loaded", "failed")
//   ?force=true  → + ("loading", "unloading")          (#236)
//
// #244 is the bug where the mirror kept the pre-#236 gate, so a row stranded
// in `loading` had no unload control at all. Four things are pinned down:
//
//   1. `loading` / `unloading` get a Force unload control, and it is enabled.
//      Every other status gets the plain Unload, enabled only for the
//      plain-gate statuses. The two never render together.
//   2. Force unload is confirmed: the row button opens a dialog and sends
//      nothing; only the dialog's confirm posts — with ?force=true.
//   3. Cancel sends nothing.
//   4. The reason it is offered is on the page, not just in the dialog.
import { Suspense } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  cleanup,
  act,
  fireEvent,
  waitFor,
} from "@testing-library/react";
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

interface Call {
  url: string;
  method: string;
}

async function renderWith(model: Record<string, unknown>): Promise<Call[]> {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    calls.push({ url, method: init?.method ?? "GET" });
    if (url === "/api/models/abc") {
      return new Response(JSON.stringify(model), { status: 200 });
    }
    if (url === "/api/models/abc/capabilities") {
      return new Response(JSON.stringify({ runs: [] }), { status: 200 });
    }
    if (url.startsWith("/api/models/abc/unload")) {
      return new Response(JSON.stringify({ status: "unloading" }), {
        status: 202,
      });
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
  return calls;
}

function unloadCalls(calls: Call[]): Call[] {
  return calls.filter(
    (c) => c.method === "POST" && c.url.startsWith("/api/models/abc/unload"),
  );
}

describe("ModelDetailPage — Force unload", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("EventSource", InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("is offered, enabled, for loading and unloading — the statuses only ?force=true accepts", async () => {
    for (const status of ["loading", "unloading"] as const) {
      await renderWith(fakeModel({ status }));
      expect(screen.getByTestId("force-unload-button")).not.toBeDisabled();
      // Not alongside a plain Unload: the slot is one control or the other.
      expect(screen.queryByTestId("unload-button")).not.toBeInTheDocument();
      cleanup();
    }
  });

  it("is not offered for the plain-gate statuses, which keep the ordinary Unload", async () => {
    // `loaded` / `failed` are what the plain route accepts.
    for (const status of ["loaded", "failed"] as const) {
      await renderWith(fakeModel({ status }));
      expect(screen.queryByTestId("force-unload-button")).not.toBeInTheDocument();
      expect(screen.getByTestId("unload-button")).not.toBeDisabled();
      cleanup();
    }
    // Nothing to unload from these: no engine was ever started. The plain
    // Unload stays, disabled — the server would 409 either way, and #236
    // deliberately left `pulling` out of the force set.
    for (const status of ["registered", "pulling", "pulled"] as const) {
      await renderWith(fakeModel({ status }));
      expect(screen.queryByTestId("force-unload-button")).not.toBeInTheDocument();
      expect(screen.getByTestId("unload-button")).toBeDisabled();
      cleanup();
    }
  });

  it("says on the page why it is offered, not only in the dialog", async () => {
    await renderWith(fakeModel({ status: "loading" }));
    const hint = screen.getByTestId("force-unload-hint");
    expect(hint).toHaveTextContent(/stranded/);
    expect(hint).toHaveTextContent(/Force unload/);
    cleanup();

    await renderWith(fakeModel({ status: "loaded" }));
    expect(screen.queryByTestId("force-unload-hint")).not.toBeInTheDocument();
  });

  it("confirms first, then POSTs ?force=true", async () => {
    const calls = await renderWith(fakeModel({ status: "loading" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByTestId("force-unload-button"));
    });
    // The row button alone must not fire the action.
    expect(screen.getByRole("dialog")).toHaveTextContent("Force unload");
    expect(unloadCalls(calls)).toHaveLength(0);

    await act(async () => {
      fireEvent.click(screen.getByTestId("force-unload-confirm"));
    });
    await waitFor(() => expect(unloadCalls(calls)).toHaveLength(1));
    expect(unloadCalls(calls)[0]).toEqual({
      url: "/api/models/abc/unload?force=true",
      method: "POST",
    });
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("Cancel closes the dialog and sends nothing", async () => {
    const calls = await renderWith(fakeModel({ status: "unloading" }));
    await act(async () => {
      fireEvent.click(screen.getByTestId("force-unload-button"));
    });
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(unloadCalls(calls)).toHaveLength(0);
  });

  it("surfaces a refusal inside the dialog instead of closing it", async () => {
    const model = fakeModel({ status: "loading" });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url === "/api/models/abc") {
        return new Response(JSON.stringify(model), { status: 200 });
      }
      if (url === "/api/models/abc/capabilities") {
        return new Response(JSON.stringify({ runs: [] }), { status: 200 });
      }
      if (url.startsWith("/api/models/abc/unload") && init?.method === "POST") {
        return new Response(
          JSON.stringify({ detail: "cannot unload from status 'loading'" }),
          { status: 409 },
        );
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

    await act(async () => {
      fireEvent.click(screen.getByTestId("force-unload-button"));
    });
    await act(async () => {
      fireEvent.click(screen.getByTestId("force-unload-confirm"));
    });
    await waitFor(() =>
      expect(screen.getByTestId("force-unload-error")).toHaveTextContent(
        "cannot unload from status 'loading'",
      ),
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });
});
