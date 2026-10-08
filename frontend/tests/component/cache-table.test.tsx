// Component tests for the Storage / HF cache table on /stats.
//
// vllm-warden#114 — surfaces ``GET /api/cache/models`` rows with size,
// mtime, owning models, and a per-row Delete button. The two pin-down
// behaviours here:
//
//   1. Delete is disabled when any matching row is active (loaded /
//      loading / unloading / pulling) — matches the backend's hard 409.
//   2. A 409 with "force=true" in the detail flips the row into a
//      Confirm/Cancel prompt, and the Confirm button re-submits
//      ?force=true. A 204 there closes the prompt and calls onMutate.
//
// Network is stubbed at the global ``fetch`` boundary so the auth-fetch
// wrapper is exercised end-to-end (Authorization header, JSON
// content-type, etc.).
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { CacheTable, type CachedRepoView } from "@/components/stats/cache-table";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

function makeRow(
  overrides: Partial<CachedRepoView> & { repo: string },
): CachedRepoView {
  const { repo, ...rest } = overrides;
  return {
    repo,
    path: `/cache/models--${repo.replace("/", "--")}`,
    size_bytes: 1_000_000_000,
    last_modified: Date.now() / 1000 - 3600,
    matched_models: [],
    ...rest,
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface DeleteCall {
  url: string;
  method: string;
}

function installFetchStub(
  handler: (url: string, init?: RequestInit) => Response | Promise<Response>,
): DeleteCall[] {
  const calls: DeleteCall[] = [];
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

describe("CacheTable", () => {
  beforeEach(() => {
    // Seed auth so the eager refresh path is skipped.
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders an empty-state message when there are no repos", () => {
    render(<CacheTable repos={[]} onMutate={() => {}} />);
    expect(screen.getByText(/no cached huggingface repos/i)).toBeInTheDocument();
  });

  it("renders one row per repo, sorted by size descending", () => {
    const repos = [
      makeRow({ repo: "a/small", size_bytes: 1_000 }),
      makeRow({ repo: "b/huge", size_bytes: 1_000_000_000 }),
      makeRow({ repo: "c/medium", size_bytes: 500_000 }),
    ];
    render(<CacheTable repos={repos} onMutate={() => {}} />);
    const rowOrder = screen
      .getAllByTestId(/^cache-row-/)
      .filter((el) => el.tagName === "TR")
      .map((el) => el.getAttribute("data-testid"));
    expect(rowOrder).toEqual([
      "cache-row-b/huge",
      "cache-row-c/medium",
      "cache-row-a/small",
    ]);
  });

  it("disables the Delete button when any matching row is active", () => {
    const repos = [
      makeRow({
        repo: "active/repo",
        matched_models: [
          { id: "m1", served_model_name: "active-served", status: "loaded" },
        ],
      }),
    ];
    render(<CacheTable repos={repos} onMutate={() => {}} />);
    const btn = screen.getByTestId("cache-row-active/repo-delete");
    expect(btn).toBeDisabled();
    expect(btn.getAttribute("title")).toMatch(/unload first/i);
  });

  it("orphan row: Delete is enabled and calls DELETE then onMutate on 204", async () => {
    const onMutate = vi.fn();
    installFetchStub((url, init) => {
      if (url.startsWith("/api/cache/models/") && init?.method === "DELETE") {
        return new Response(null, { status: 204 });
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <CacheTable
        repos={[makeRow({ repo: "orphan/repo" })]}
        onMutate={onMutate}
      />,
    );
    fireEvent.click(screen.getByTestId("cache-row-orphan/repo-delete"));
    await waitFor(() => expect(onMutate).toHaveBeenCalledTimes(1));
  });

  it("409 with force-required message → renders Confirm/Cancel prompt; Confirm re-submits ?force=true", async () => {
    const onMutate = vi.fn();
    let stage = 0;
    const calls = installFetchStub((url, init) => {
      if (url.startsWith("/api/cache/models/") && init?.method === "DELETE") {
        if (stage === 0) {
          stage = 1;
          return json(
            { detail: "repo backs 1 pulled-but-unloaded row(s); pass ?force=true" },
            409,
          );
        }
        // Second call must include ?force=true.
        expect(url).toMatch(/\?force=true$/);
        return new Response(null, { status: 204 });
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <CacheTable
        repos={[
          makeRow({
            repo: "pulled/repo",
            matched_models: [
              { id: "m1", served_model_name: "pulled-served", status: "pulled" },
            ],
          }),
        ]}
        onMutate={onMutate}
      />,
    );
    // First click → 409 → force prompt visible.
    fireEvent.click(screen.getByTestId("cache-row-pulled/repo-delete"));
    await waitFor(() =>
      expect(screen.getByTestId("cache-row-pulled/repo-force-msg")).toBeInTheDocument(),
    );
    // Confirm → second DELETE with ?force=true → 204 → onMutate.
    fireEvent.click(screen.getByTestId("cache-row-pulled/repo-force"));
    await waitFor(() => expect(onMutate).toHaveBeenCalledTimes(1));
    const deletes = calls.filter((c) => c.method === "DELETE");
    expect(deletes).toHaveLength(2);
    expect(deletes[1].url).toMatch(/\?force=true$/);
  });

  it("Cancel button on the force prompt closes it without calling onMutate", async () => {
    const onMutate = vi.fn();
    installFetchStub((url, init) => {
      if (url.startsWith("/api/cache/models/") && init?.method === "DELETE") {
        return json(
          { detail: "repo backs 1 pulled-but-unloaded row(s); pass ?force=true" },
          409,
        );
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <CacheTable
        repos={[
          makeRow({
            repo: "pulled/repo",
            matched_models: [
              { id: "m1", served_model_name: "pulled-served", status: "pulled" },
            ],
          }),
        ]}
        onMutate={onMutate}
      />,
    );
    fireEvent.click(screen.getByTestId("cache-row-pulled/repo-delete"));
    await waitFor(() =>
      expect(screen.getByTestId("cache-row-pulled/repo-force-msg")).toBeInTheDocument(),
    );
    // Cancel → prompt vanishes, Delete reappears, onMutate untouched.
    fireEvent.click(screen.getByText(/cancel/i));
    await waitFor(() =>
      expect(screen.queryByTestId("cache-row-pulled/repo-force-msg")).toBeNull(),
    );
    expect(onMutate).not.toHaveBeenCalled();
    expect(screen.getByTestId("cache-row-pulled/repo-delete")).toBeInTheDocument();
  });

  it("non-409 error renders an inline message and does not call onMutate", async () => {
    const onMutate = vi.fn();
    installFetchStub((url, init) => {
      if (url.startsWith("/api/cache/models/") && init?.method === "DELETE") {
        return json({ detail: "rmtree exploded" }, 500);
      }
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    });
    render(
      <CacheTable
        repos={[makeRow({ repo: "broken/repo" })]}
        onMutate={onMutate}
      />,
    );
    fireEvent.click(screen.getByTestId("cache-row-broken/repo-delete"));
    await waitFor(() =>
      expect(screen.getByTestId("cache-row-broken/repo-err")).toHaveTextContent(
        /rmtree exploded/,
      ),
    );
    expect(onMutate).not.toHaveBeenCalled();
  });
});
