// The "Requests" panel — GET /api/stats/v2/requests — the chart that replaced
// the "Recently finished" table, with the table kept underneath as detail.
//
// The old table was ten identical-looking rows kept 15 minutes; nothing
// aggregated, so there was no question it answered. The rows are persisted
// now and the panel honours the window. `model_id` on a row is the models
// table's ROW id (the fixture's differs from the served name on purpose), and
// the page scopes the endpoint with it.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, cleanup, waitFor, fireEvent, within } from "@testing-library/react";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

vi.mock("@/lib/live-stats-stream", async (orig) => {
  const actual = await orig<typeof import("@/lib/live-stats-stream")>();
  const { mockStream } = await import("./stats-merged-stream");
  return { ...actual, useLiveStats: () => mockStream.state };
});

import {
  FakeResizeObserver,
  HISTORY_ROWS,
  NOW_EPOCH,
  SILENT_LLAMACPP,
  block,
  boxFor,
  frameWith,
  historyFor,
  installFetchStub,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

describe("merged stats — requests chart and its table", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    setFrame(frameWith([block("model-a-8b"), SILENT_LLAMACPP]));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("states the window and the count, not a ring's retention", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("requests-scope")).toBeInTheDocument(),
    );
    const scope = screen.getByTestId("requests-scope").textContent ?? "";
    expect(scope).toContain("last hour");
    expect(scope).toContain("2 requests");
    expect(screen.getByTestId("requests-panel").textContent).not.toContain("not the window");
    expect(screen.getByTestId("requests-panel").textContent).not.toContain("kept 15 min");
  });

  it("keeps the table as a detail view with exact values", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(2),
    );
    const first = screen.getAllByTestId("finished-row")[0];
    expect(first.textContent).toContain("key-a");
    expect(first.textContent).toContain("stop");
    expect(first.textContent).toContain("23.1 s"); // duration
    expect(first.textContent).toContain("400 ms"); // ttft 0.4s
    // A request whose first token never arrived: a dash, not 0.
    const second = screen.getAllByTestId("finished-row")[1];
    expect(second.textContent).toContain("length");
    expect(second.textContent).toContain("—");
    expect(second.textContent).toContain("orphan");
    // Subordinate: inside a collapsed <details>, under the chart.
    const details = screen.getByTestId("requests-table");
    expect(details.tagName).toBe("DETAILS");
    expect(details).not.toHaveAttribute("open");
  });

  it("colours by client when there is more than one, and says so", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("colour-legend")).toBeInTheDocument(),
    );
    const legend = screen.getByTestId("colour-legend");
    expect(legend.textContent).toContain("colour = client");
    const groups = within(legend).getAllByTestId("colour-legend-group");
    expect(groups.map((g) => g.getAttribute("data-group"))).toEqual(["key-a", "key-b"]);
    // Finish reason is carried by shape, and the key says so in words.
    expect(screen.getByTestId("requests-panel").textContent).toContain("shape = finish");
  });

  it("does not spend colour on a single dominant client", async () => {
    // One client, one model: colour would encode nothing, and the legend
    // says that rather than painting a false series.
    installFetchStub({
      history: (url: string) =>
        historyFor(
          url,
          HISTORY_ROWS.map((r) => ({ ...r, token_name: "key-a", client_ip: "10.0.0.1", model: "model-a-8b", model_id: "id-model-a-8b" })),
        ),
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("colour-legend-none")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("colour-legend-none").textContent).toContain("colour carries nothing");
  });

  it("falls back to colouring by model when one client dominates but models differ", async () => {
    installFetchStub({
      history: (url: string) =>
        historyFor(
          url,
          HISTORY_ROWS.map((r) => ({ ...r, token_name: "key-a", client_ip: "10.0.0.1" })),
        ),
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("colour-legend")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("colour-legend").textContent).toContain("colour = model");
  });

  it("isolates a client when its legend chip is clicked", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(2),
    );
    const chips = within(screen.getByTestId("colour-legend")).getAllByTestId(
      "colour-legend-group",
    );
    fireEvent.click(chips[1]); // hide key-b
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(1),
    );
    expect(screen.getByTestId("finished-row").textContent).toContain("key-a");
    expect(chips[1]).toHaveAttribute("aria-pressed", "false");
  });

  it("scopes the endpoint with the ROW id, which is what it filters on", async () => {
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(2),
    );
    fireEvent.click(boxFor("id-model-b-27b"));
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(1),
    );
    expect(screen.getByTestId("finished-row").textContent).toContain("key-a");
    const scoped = urls.filter(
      (u) => u.startsWith("/api/stats/v2/requests") && u.includes("models="),
    );
    expect(scoped.length).toBeGreaterThan(0);
    const raw =
      new URL(scoped[scoped.length - 1], "http://x").searchParams.get("models") ?? "";
    expect(raw.split(",")).toEqual(["id-model-a-8b"]);
  });

  it("says when the window holds more than it draws", async () => {
    installFetchStub({
      history: (url: string) => ({ ...historyFor(url), total: 8431, stride: 5 }),
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("requests-scope").textContent).toContain("8,431 requests"),
    );
    expect(screen.getByTestId("requests-scope").textContent).toContain("drawing every 5th");
  });

  it("says precisely when history is younger than the window", async () => {
    installFetchStub({
      history: (url: string) => {
        const h = historyFor(url);
        return {
          ...h,
          coverage: { ...h.coverage, earliest_epoch: NOW_EPOCH - 20 * 60, covers_window: false },
        };
      },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("requests-coverage")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("requests-coverage").textContent).toMatch(
      /history begins 20 min ago — covers 20 min of the last hour/,
    );
  });

  it("says when retention is shorter than the window", async () => {
    installFetchStub({
      history: (url: string) => {
        const h = historyFor(url);
        return { ...h, coverage: { ...h.coverage, retention_days: 0, covers_window: false } };
      },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("requests-coverage").textContent).toMatch(/retention is 0 d/),
    );
  });

  it("distinguishes an empty window from a store with nothing in it", async () => {
    installFetchStub({
      history: (url: string) => ({
        ...historyFor(url, []),
        coverage: { earliest_epoch: null, retention_days: 30, max_rows: 200000, covers_window: false },
      }),
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("requests-empty").textContent).toMatch(/no requests recorded yet/i),
    );
  });

  it("links a finished row's token name to its details page when token_id is present", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(2),
    );
    // f1 carries token_id "tok-a" -- rendered as a link.
    const first = screen.getAllByTestId("finished-row")[0];
    const link = within(first).getByRole("link", { name: "key-a" });
    expect(link).toHaveAttribute("href", "/tokens/tok-a");
    expect(link.className).toContain("hover:underline");
  });

  it("does not link a finished row recorded before token_id existed", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("finished-row")).toHaveLength(2),
    );
    // f2 has token_id: null (an older row) -- plain text, no link.
    const second = screen.getAllByTestId("finished-row")[1];
    expect(within(second).queryByRole("link", { name: "key-b" })).toBeNull();
    expect(second.textContent).toContain("key-b");
  });
});

describe("merged stats — in-flight requests and the by-token rollup", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    setFrame(frameWith([block("model-a-8b"), SILENT_LLAMACPP]));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("links the token name in the in-flight table when token_id is present", async () => {
    installFetchStub();
    renderPage();
    const panel = await screen.findByTestId("live-requests");
    const link = await within(panel).findByRole("link", { name: "key-a" });
    expect(link).toHaveAttribute("href", "/tokens/tok-a");
  });

  it("renders anonymous in-flight rows as plain text, no link", async () => {
    installFetchStub({
      requests: {
        ts: "2026-09-05T19:00:00Z",
        count: 1,
        requests: [
          {
            id: "r-anon", token_id: null, token_name: null, client_ip: "10.0.0.9",
            model: "model-a-8b", path: "/v1/chat/completions",
            prompt_tokens: 10, completion_tokens: 1, context_tokens: 11,
            max_model_len: 8192, context_pct: 0.001, elapsed_s: 1, phase: "decode",
            orphan: false,
          },
        ],
        by_token: [],
        by_ip: [],
      },
    });
    renderPage();
    const panel = await screen.findByTestId("live-requests");
    expect(within(panel).getByText("anonymous")).toBeInTheDocument();
    expect(within(panel).queryByRole("link")).toBeNull();
  });

  it("links the token name in the By-token rollup and groups two ids sharing a name separately", async () => {
    installFetchStub({
      requests: {
        ts: "2026-09-05T19:00:00Z",
        count: 2,
        requests: [
          {
            id: "r1", token_id: "tok-old", token_name: "shared-name", client_ip: "10.0.0.1",
            model: "model-a-8b", path: "/v1/chat/completions",
            prompt_tokens: 100, completion_tokens: 10, context_tokens: 110,
            max_model_len: 8192, context_pct: 0.013, elapsed_s: 1, phase: "decode",
            orphan: false,
          },
          {
            id: "r2", token_id: "tok-new", token_name: "shared-name", client_ip: "10.0.0.2",
            model: "model-a-8b", path: "/v1/chat/completions",
            prompt_tokens: 200, completion_tokens: 20, context_tokens: 220,
            max_model_len: 8192, context_pct: 0.027, elapsed_s: 2, phase: "decode",
            orphan: false,
          },
        ],
        by_token: [],
        by_ip: [],
      },
    });
    renderPage();
    const panel = await screen.findByTestId("by-token-panel");
    const links = await within(panel).findAllByRole("link", { name: "shared-name" });
    // Two distinct token_ids sharing a display name stay two separate rows.
    expect(links).toHaveLength(2);
    const hrefs = links.map((l) => l.getAttribute("href")).sort();
    expect(hrefs).toEqual(["/tokens/tok-new", "/tokens/tok-old"]);
  });
});

describe("merged stats — latency from the store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    setFrame(frameWith([block("model-a-8b"), SILENT_LLAMACPP]));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the three distributions with exact percentile markers", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("ttft-histogram")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("itl-histogram")).toBeInTheDocument();
    expect(screen.getByTestId("duration-histogram")).toBeInTheDocument();
    const ttft = screen.getByTestId("ttft-panel").textContent ?? "";
    expect(ttft).toContain("p50");
    expect(ttft).toContain("p99");
    expect(ttft).toContain("400 ms");
    expect(ttft).toContain("n=2");
    // The per-request mean is labelled as such — it is not the engine's
    // per-token histogram.
    expect(screen.getByTestId("itl-panel").textContent).toContain("mean per request");
  });

  it("switches to the last-N basis and says which is showing", async () => {
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("latency-basis")).toBeInTheDocument(),
    );
    fireEvent.click(within(screen.getByTestId("latency-basis")).getByRole("button", { name: /last 500/ }));
    await waitFor(() =>
      expect(screen.getByTestId("ttft-panel").textContent).toContain("last 2 of 500 requests"),
    );
    expect(screen.getByTestId("ttft-panel").textContent).toContain("spanning 1 min");
    expect(urls.some((u) => u.includes("/api/stats/v2/latency?") && u.includes("basis=last&n=500"))).toBe(true);
    expect(window.localStorage.getItem("vw.stats.latency-basis")).toBe("last");
  });

  it("is empty, not zero, when nothing was recorded in scope", async () => {
    installFetchStub({
      latency: (url: string) => {
        const l = latencyForEmpty(url);
        return l;
      },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("ttft-empty")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("ttft-histogram")).toBeNull();
    expect(screen.getByTestId("ttft-panel").textContent).toContain("n=0");
  });
});

function latencyForEmpty(url: string) {
  const params = new URL(url, "http://x").searchParams;
  const empty = { count: 0, p50: null, p90: null, p99: null, mean: null,
    buckets: { le: [0.1, null], counts: [0, 0], count: 0, sum: 0 } };
  return {
    basis: params.get("basis") ?? "window",
    range: params.get("range") ?? "1h",
    n: null,
    since_epoch: NOW_EPOCH - 3600,
    selected_model_ids: null,
    count: 0,
    span_s: null,
    oldest_epoch: null,
    newest_epoch: null,
    ttft: empty, itl: empty, duration: empty,
    coverage: { earliest_epoch: NOW_EPOCH - 10 * 86400, retention_days: 30, max_rows: 200000, covers_window: true },
  };
}
