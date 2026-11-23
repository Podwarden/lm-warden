// The redesigned /stats page (owner-approved 2026-11-22).
//
// Pins: one scope per page (every feed above the Requests block asks for the
// scope's model ROW id); the fixed-size rule (every panel above the bottom
// Requests block keeps a fixed height whatever the data volume, state or tab;
// jsdom has no layout, so the test reads the explicit heights each panel
// declares, and that the live table always renders exactly 8 row slots); idle
// reads "idle", never "0 tok/s"; notices go to the one reserved line; a null
// frame from a failed scrape does not crash the page; the removed panels
// (Token throughput, Preemptions, model selector, per-key table) stay gone.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";
import type { LiveEngineFrame } from "@/lib/live-stats";
import { FIXED } from "@/lib/stats-redesign";

const nav = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: nav.push, replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/stats",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/live-stats-stream", async (orig) => {
  const actual = await orig<typeof import("@/lib/live-stats-stream")>();
  const { mockStream } = await import("./stats-merged-stream");
  return { ...actual, useLiveStats: () => mockStream.state };
});

import {
  FakeResizeObserver,
  REQUESTS,
  block,
  frameWith,
  installFetchStub,
  renderPage,
  setFrame,
} from "./stats-merged-harness";

const A = block("model-a-8b", { model_id: "id-model-a-8b" });
const B = block("model-b-27b", { model_id: "id-model-b-27b" });

function fixedHeights(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const el of document.querySelectorAll<HTMLElement>("[data-fixed-height]")) {
    const id = el.getAttribute("data-testid") ?? "?";
    out[`${id}#${Object.keys(out).filter((k) => k.startsWith(id + "#")).length}`] = `${el.getAttribute("data-fixed-height")}|${el.getAttribute("data-fixed-height-phone") ?? "-"}`;
  }
  return out;
}

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  constructor(public url: string) {
    FakeES.all.push(this);
  }
  close() {}
}

beforeEach(() => {
  FakeES.all = [];
  vi.stubGlobal("EventSource", FakeES as unknown as typeof EventSource);
  setAccessToken("t");
  setCsrfToken("c");
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  window.localStorage.clear();
});
afterEach(() => {
  cleanup();
  setFrame(null, "connecting");
});

describe("stats redesign — one scope", () => {
  it("asks every scoped feed for the scope model's row id, one model only", async () => {
    setFrame(frameWith([A, B]));
    const urls = installFetchStub();
    renderPage();
    await waitFor(() => expect(urls.some((u) => u.startsWith("/api/stats/v2/speed"))).toBe(true));
    const speed = urls.find((u) => u.startsWith("/api/stats/v2/speed?"))!;
    expect(speed).toContain("model=id-model-a-8b");
    expect(urls.find((u) => u.startsWith("/api/stats/v2/cache"))).toContain("models=id-model-a-8b");
    expect(urls.find((u) => u.startsWith("/api/stats/v2/requests"))).toContain("models=id-model-a-8b");
    expect(urls.some((u) => u.includes("models=id-model-a-8b,") || u.includes("%2C"))).toBe(false);
  });

  it("switches the whole page when another model tab is chosen", async () => {
    setFrame(frameWith([A, B]));
    const urls = installFetchStub();
    renderPage();
    const tabs = await screen.findAllByTestId("scope-tab");
    expect(tabs).toHaveLength(2);
    fireEvent.click(tabs[1]);
    await waitFor(() => expect(urls.some((u) => u.includes("speed") && u.includes("model=id-model-b-27b"))).toBe(true));
    expect(screen.getByTestId("latency-scope").textContent).toContain("model-b-27b");
  });

  it("has no model selector, throughput panel, preemptions panel or per-key table", async () => {
    setFrame(frameWith([A]));
    const urls = installFetchStub();
    renderPage();
    await screen.findByTestId("now-card");
    expect(screen.queryByTestId("model-selector")).toBeNull();
    expect(screen.queryByText(/preemptions \/ s/i)).toBeNull();
    expect(screen.queryByText(/Token throughput/i)).toBeNull();
    expect(screen.queryByText(/Tokens per API key/i)).toBeNull();
    expect(urls.some((u) => u.startsWith("/api/stats/v2/throughput"))).toBe(false);
    expect(urls.some((u) => u.startsWith("/api/stats/v2/tokens-per-key"))).toBe(false);
  });
});

describe("stats redesign — fixed size", () => {
  const volumes = [0, 3, 12, 60];

  it("keeps every panel above the Requests block at the same declared height for every volume and state", async () => {
    const seen: Record<string, string>[] = [];
    for (const n of volumes) {
      for (const f of [frameWith([A]), frameWith([{ ...A, engine: null, throughput: null, scrape_error: "boom" } as LiveEngineFrame])]) {
        setFrame(f);
        installFetchStub({ speedN: n, requests: n ? REQUESTS : { ...REQUESTS, count: 0, requests: [] } });
        renderPage();
        await screen.findByTestId("now-card");
        await waitFor(() => expect(screen.getByTestId("tile-ttft").textContent).toMatch(/n=/));
        seen.push(fixedHeights());
        cleanup();
      }
    }
    for (const h of seen) expect(h).toEqual(seen[0]);
    const s = seen[0];
    // Desktop | phone heights, both fixed (per breakpoint, never by data).
    expect(s["scope-bar#0"]).toBe(`${FIXED.scopeBar}|-`);
    expect(s["now-card#0"]).toBe(`${FIXED.nowCard}|${FIXED.nowCardPhone}`);
    expect(s["tile-ttft#0"]).toBe(`${FIXED.tile}|${FIXED.tile}`);
    expect(s["cache-card#0"]).toBe(`${FIXED.cacheCard}|${FIXED.cacheCardPhone}`);
    expect(s["engine-details#0"]).toBe(`${FIXED.engineCard}|${FIXED.engineCardPhone}`);
    expect(s["notice-line#0"]).toBe(`${FIXED.notice}|-`);
  });

  it("keeps the same heights across a tab switch", async () => {
    setFrame(frameWith([A, B]));
    installFetchStub();
    renderPage();
    await screen.findByTestId("now-card");
    const before = fixedHeights();
    fireEvent.click(screen.getAllByTestId("scope-tab")[1]);
    await waitFor(() => expect(screen.getByTestId("latency-scope").textContent).toContain("model-b-27b"));
    expect(fixedHeights()).toEqual(before);
  });

  it("always renders exactly 8 row slots in the live table (ghost rows fill the rest)", async () => {
    for (const reqs of [{ ...REQUESTS, count: 0, requests: [] }, REQUESTS]) {
      setFrame(frameWith([A]));
      installFetchStub({ requests: reqs });
      renderPage();
      const body = await screen.findByTestId("live-requests-table-body");
      await waitFor(() =>
        expect(body.querySelectorAll('[data-testid="req-row"]').length).toBe(reqs.requests.length),
      );
      const rows = body.querySelectorAll('[data-testid="req-row"]').length;
      const ghosts = body.querySelectorAll('[data-testid="ghost-row"]').length;
      expect(rows + ghosts).toBe(FIXED.tableRows);
      if (!reqs.requests.length) expect(screen.getByTestId("table-empty").textContent).toMatch(/No requests in flight/);
      cleanup();
    }
  });

  it("gives the model tab and the state chip a fixed width", async () => {
    setFrame(frameWith([A]));
    installFetchStub();
    renderPage();
    const tab = await screen.findByTestId("scope-tab");
    expect(tab.style.getPropertyValue("--tw")).toBe(`${FIXED.tabW}px`);
    expect(tab.style.getPropertyValue("--twp")).toBe(`${FIXED.tabWPhone}px`);
    expect(tab.querySelector<HTMLElement>('[data-testid="state-chip"]')!.style.width).toBe(`${FIXED.chipW}px`);
  });
});

describe("stats redesign — states", () => {
  it("says idle, never 0 tok/s, when nothing generates", async () => {
    const idle = block("model-a-8b", {
      model_id: "id-model-a-8b",
      throughput: { prompt_tokens_per_s: 0, generation_tokens_per_s: 0, prompt_tokens_total: 1, generation_tokens_total: 1 },
      engine: { ...A.engine!, num_requests_running: 0, num_requests_waiting: 0 },
      now: { generation_tps_10s: 0, prefill_computed_tps_10s: 0, window_s: 10, warden_queue: 0, warden_running: 0 },
    });
    setFrame(frameWith([idle]));
    installFetchStub();
    renderPage();
    const gen = await screen.findByTestId("now-generation");
    expect(gen.textContent).toMatch(/idle · last request/);
    expect(gen.textContent).not.toMatch(/0\s*tok\/s/);
    expect(screen.getByTestId("scope-tab").textContent).toContain("Idle");
  });

  it("shows Queueing when the warden's own queue holds requests", async () => {
    const q = block("model-a-8b", {
      model_id: "id-model-a-8b",
      engine: { ...A.engine!, num_requests_waiting: 0 },
      now: { generation_tps_10s: 40, prefill_computed_tps_10s: 0, window_s: 10, warden_queue: 2, warden_running: 1 },
    });
    setFrame(frameWith([q]));
    installFetchStub();
    renderPage();
    expect((await screen.findByTestId("scope-tab")).textContent).toContain("Queueing");
  });

  it("puts a scrape failure on the notice line and still renders on a null frame", async () => {
    const broken = { ...A, engine: null, throughput: null, cache: null, latency: null, finished: null, scrape_error: "connection refused" } as LiveEngineFrame;
    setFrame(frameWith([broken]));
    installFetchStub();
    renderPage();
    const notice = await screen.findByTestId("notice-line");
    expect(notice.textContent).toMatch(/Metrics scrape failed for model-a-8b/);
    expect(document.body.textContent).not.toMatch(/\bnull\b|undefined|NaN/);
  });

  it("shows the low-volume rules: values below 5, p50 and max below 20", async () => {
    setFrame(frameWith([A]));
    installFetchStub({ speedN: 3 });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("tile-ttft").textContent).toMatch(/too few for percentiles/));
    cleanup();
    setFrame(frameWith([A]));
    installFetchStub({ speedN: 12 });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("tile-ttft").textContent).toMatch(/p50 · max/));
    cleanup();
    setFrame(frameWith([A]));
    installFetchStub({ speedN: 0 });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("tile-ttft").textContent).toMatch(/No requests in window/));
  });

  it("asks for the window columns of a Group by and shows them", async () => {
    setFrame(frameWith([A]));
    const urls = installFetchStub();
    renderPage();
    const seg = await screen.findByTestId("group-by");
    await act(async () => {
      fireEvent.click(Array.from(seg.querySelectorAll("button")).find((b) => b.textContent === "Key")!);
    });
    await waitFor(() => expect(urls.some((u) => u.startsWith("/api/stats/v2/speed?") && u.includes("group=key"))).toBe(true));
    await waitFor(() => expect(screen.getByTestId("live-requests-table").textContent).toContain("key-a"));
  });
});

describe("stats redesign — owner review 2026-11-22", () => {
  it("never says Asleep while requests run, nor when the engine exports no sleep gauge", async () => {
    const serving = block("model-a-8b", {
      model_id: "id-model-a-8b",
      engine: { ...A.engine!, num_requests_running: 2, engine_sleep_state: 1 },
    });
    setFrame(frameWith([serving]));
    installFetchStub();
    renderPage();
    expect((await screen.findByTestId("scope-tab")).textContent).not.toContain("Asleep");
    cleanup();
    const asleep = block("model-a-8b", {
      model_id: "id-model-a-8b",
      engine: { ...A.engine!, num_requests_running: 0, num_requests_waiting: 0, engine_sleep_state: 1, engine_sleeping: null },
    });
    setFrame(frameWith([asleep]));
    installFetchStub();
    renderPage();
    expect((await screen.findByTestId("scope-tab")).textContent).not.toContain("Asleep");
  });

  it("shows no install label for a single install, and the Now card repeats no scope", async () => {
    setFrame(frameWith([A]));
    installFetchStub();
    renderPage();
    const now = await screen.findByTestId("now-card");
    expect(screen.getByTestId("scope-bar").textContent).not.toMatch(/Install/);
    expect(now.textContent).not.toMatch(/install|model-a-8b/i);
    expect(screen.queryByTestId("host-vram")).toBeNull();
  });

  it("labels a non-streamed request as running, not waiting", async () => {
    setFrame(frameWith([A]));
    const ns = {
      ...REQUESTS,
      count: 1,
      requests: [{ ...REQUESTS.requests[0], phase: "waiting", is_stream: false, completion_tokens: 0 }],
    };
    installFetchStub({ requests: ns });
    renderPage();
    const table = await screen.findByTestId("live-requests-table");
    await waitFor(() => expect(table.textContent).toContain("running · not streamed"));
    expect(table.textContent).not.toMatch(/\bwaiting\b/);
  });
});

describe("stats redesign — Prompt Lens (Q1): the eye opens the Prompt Lens page", () => {
  const eyes = () => screen.queryAllByRole("button", { name: "Open in Prompt Lens" });

  it("/stats: no eye column when Prompt Lens is off", async () => {
    setFrame(frameWith([A, B]));
    const urls = installFetchStub({ promptLens: false });
    renderPage();
    await screen.findByTestId("now-card");
    await waitFor(() => expect(urls).toContain("/api/admin/prompt-lens/status"));
    await waitFor(() => expect(screen.getAllByTestId("req-row").length).toBe(2));
    expect(eyes()).toHaveLength(0);
  });

  it("/stats carries no dock and opens no Prompt Lens stream", async () => {
    setFrame(frameWith([A, B]));
    installFetchStub({ promptLens: true });
    renderPage();
    await waitFor(() => expect(eyes()).toHaveLength(2));
    expect(screen.queryByTestId("prompt-lens-dock")).toBeNull();
    expect(screen.queryByTestId("prompt-lens-view")).toBeNull();
    expect(FakeES.all.filter((e) => e.url.includes("prompt-lens"))).toHaveLength(0);
  });

  it("/stats: the eye on a Requests row opens /prompt-lens?req=<that request>", async () => {
    nav.push.mockReset();
    setFrame(frameWith([A, B]));
    installFetchStub({ promptLens: true });
    renderPage();
    await waitFor(() => expect(eyes()).toHaveLength(2));
    const row = screen.getAllByTestId("req-row").find((r) => r.textContent?.includes("key-b"))!;
    fireEvent.click(within(row).getByRole("button", { name: "Open in Prompt Lens" }));
    expect(nav.push).toHaveBeenCalledTimes(1);
    expect(nav.push.mock.calls[0][0]).toMatch(/^\/prompt-lens\?req=[^&]+$/);
  });

  it("says Prompt Lens, never god mode", async () => {
    setFrame(frameWith([A, B]));
    installFetchStub({ promptLens: true });
    renderPage();
    await waitFor(() => expect(eyes()).toHaveLength(2));
    expect(document.body.textContent ?? "").not.toMatch(/god[ -]?mode/i);
    for (const el of document.querySelectorAll("[aria-label],[title]")) {
      expect(`${el.getAttribute("aria-label") ?? ""} ${el.getAttribute("title") ?? ""}`).not.toMatch(/god[ -]?mode/i);
    }
  });
});
