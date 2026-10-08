// Component tests for the DeleteModelModal — the per-model confirm dialog
// introduced in S6 (epic/overhaul, #105). The behaviour that warrants
// explicit pin-down is the **chained delete**:
//
//   1. DELETE /api/models/{id}        always runs
//   2. DELETE /api/cache/models/{repo}?force=true  runs only when the
//      operator opted in via the "Also free cache" checkbox AND step 1
//      succeeded.
//
// We mock `fetch` globally (mirroring the cache-table test) so the
// auth-fetch wrapper runs end-to-end. The tests assert call order,
// call count, skip-on-row-failure, and the cache-failed inline notice.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { DeleteModelModal } from "@/components/models/delete-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface Call {
  url: string;
  method: string;
}

function installFetchStub(
  handler: (url: string, init?: RequestInit) => Response | Promise<Response>,
): Call[] {
  const calls: Call[] = [];
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, method: init?.method ?? "GET" });
    if (url === "/api/auth/refresh") {
      return json({ access_token: "test-jwt-refreshed" });
    }
    if (url === "/api/csrf") {
      return json({ csrf: "test-csrf" });
    }
    return handler(url, init);
  });
  vi.stubGlobal("fetch", mock);
  return calls;
}

describe("DeleteModelModal", () => {
  beforeEach(() => {
    // Seed auth so the eager refresh path is skipped.
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("calls only DELETE /api/models/{id} when the checkbox is unchecked", async () => {
    const onDeleted = vi.fn();
    const calls = installFetchStub((url, init) => {
      if (url === "/api/models/m-1" && init?.method === "DELETE") {
        return new Response(null, { status: 204 });
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <DeleteModelModal
        open
        onClose={() => {}}
        modelId="m-1"
        servedModelName="opt-125m"
        hfRepo="facebook/opt-125m"
        onDeleted={onDeleted}
      />,
    );
    fireEvent.click(screen.getByTestId("delete-confirm"));
    await waitFor(() => expect(onDeleted).toHaveBeenCalledTimes(1));
    const deletes = calls.filter((c) => c.method === "DELETE");
    expect(deletes).toHaveLength(1);
    expect(deletes[0].url).toBe("/api/models/m-1");
  });

  it("chains DELETE row → DELETE cache?force=true in order when checkbox is ticked", async () => {
    const onDeleted = vi.fn();
    const calls = installFetchStub((url, init) => {
      if (url === "/api/models/m-1" && init?.method === "DELETE") {
        return new Response(null, { status: 204 });
      }
      if (
        url.startsWith("/api/cache/models/") &&
        init?.method === "DELETE"
      ) {
        return new Response(null, { status: 204 });
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <DeleteModelModal
        open
        onClose={() => {}}
        modelId="m-1"
        servedModelName="opt-125m"
        hfRepo="facebook/opt-125m"
        onDeleted={onDeleted}
      />,
    );
    fireEvent.click(screen.getByTestId("free-cache-checkbox"));
    fireEvent.click(screen.getByTestId("delete-confirm"));
    await waitFor(() => expect(onDeleted).toHaveBeenCalledTimes(1));
    const deletes = calls.filter((c) => c.method === "DELETE");
    expect(deletes).toHaveLength(2);
    // Order matters: row first, cache second. The opposite ordering
    // would orphan the model row if the cache leg failed.
    expect(deletes[0].url).toBe("/api/models/m-1");
    expect(deletes[1].url).toBe(
      `/api/cache/models/${encodeURIComponent("facebook/opt-125m")}?force=true`,
    );
  });

  it("skips the cache DELETE entirely when the row DELETE fails", async () => {
    const onDeleted = vi.fn();
    const calls = installFetchStub((url, init) => {
      if (url === "/api/models/m-1" && init?.method === "DELETE") {
        return json({ detail: "model is loaded" }, 409);
      }
      if (
        url.startsWith("/api/cache/models/") &&
        init?.method === "DELETE"
      ) {
        // If this fires the test should fail — guard with an explicit
        // throw rather than a silent stub so the assertion error is
        // legible.
        throw new Error("cache DELETE must not run when row DELETE failed");
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <DeleteModelModal
        open
        onClose={() => {}}
        modelId="m-1"
        servedModelName="opt-125m"
        hfRepo="facebook/opt-125m"
        onDeleted={onDeleted}
      />,
    );
    // Tick the checkbox — even with opt-in, a row failure must short-circuit.
    fireEvent.click(screen.getByTestId("free-cache-checkbox"));
    fireEvent.click(screen.getByTestId("delete-confirm"));
    await waitFor(() =>
      expect(screen.getByTestId("delete-error")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("delete-error")).toHaveTextContent(
      /model is loaded/i,
    );
    expect(onDeleted).not.toHaveBeenCalled();
    const cacheDeletes = calls.filter(
      (c) => c.method === "DELETE" && c.url.startsWith("/api/cache/"),
    );
    expect(cacheDeletes).toHaveLength(0);
  });

  it("surfaces an inline notice (and still calls onDeleted) when cache DELETE fails after row succeeds", async () => {
    const onDeleted = vi.fn();
    installFetchStub((url, init) => {
      if (url === "/api/models/m-1" && init?.method === "DELETE") {
        return new Response(null, { status: 204 });
      }
      if (
        url.startsWith("/api/cache/models/") &&
        init?.method === "DELETE"
      ) {
        return json({ detail: "filesystem error" }, 500);
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <DeleteModelModal
        open
        onClose={() => {}}
        modelId="m-1"
        servedModelName="opt-125m"
        hfRepo="facebook/opt-125m"
        onDeleted={onDeleted}
      />,
    );
    fireEvent.click(screen.getByTestId("free-cache-checkbox"));
    fireEvent.click(screen.getByTestId("delete-confirm"));
    await waitFor(() =>
      expect(screen.getByTestId("cache-failed-notice")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("cache-failed-notice")).toHaveTextContent(
      /cache delete failed/i,
    );
    // Row is gone, so the parent must still be notified — the modal
    // does NOT pretend the chain failed atomically.
    expect(onDeleted).toHaveBeenCalledTimes(1);
  });

  it("treats a 404 on the cache leg as success (row gone, cache already absent)", async () => {
    const onDeleted = vi.fn();
    installFetchStub((url, init) => {
      if (url === "/api/models/m-1" && init?.method === "DELETE") {
        return new Response(null, { status: 204 });
      }
      if (
        url.startsWith("/api/cache/models/") &&
        init?.method === "DELETE"
      ) {
        return json({ detail: "no such repo in cache" }, 404);
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <DeleteModelModal
        open
        onClose={() => {}}
        modelId="m-1"
        servedModelName="opt-125m"
        hfRepo="facebook/opt-125m"
        onDeleted={onDeleted}
      />,
    );
    fireEvent.click(screen.getByTestId("free-cache-checkbox"));
    fireEvent.click(screen.getByTestId("delete-confirm"));
    await waitFor(() => expect(onDeleted).toHaveBeenCalledTimes(1));
    // No notice should render — 404 means cache was already absent.
    expect(screen.queryByTestId("cache-failed-notice")).not.toBeInTheDocument();
  });
});
