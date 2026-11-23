/**
 * The key holder's Stats view (/key/stats, owner decisions D2, D5, D6, D8).
 *
 * It must never render fleet data: it only ever calls the key holder's own
 * /api/me/* routes (never the admin stats feeds, the live engine stream, host
 * or system routes), shows no Key, Client IP or Install column, no host,
 * engine or install section, and no "N ahead" queue count. The eye icon and
 * the Prompt Lens dock appear only when the server says both switches are on;
 * the eye opens that request in the dock (the old capture overlay is gone).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";

const nav = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({
  usePathname: () => "/key/stats",
  useRouter: () => ({ replace: vi.fn(), push: nav.push }),
}));

import KeyStatsPage from "@/app/key/stats/page";
import { FIXED } from "@/lib/stats-redesign";

const NOW = Date.now() / 1000;

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.all.push(this);
  }
  close() {
    this.closed = true;
  }
}

function startEv(reqId: string, requestId: string) {
  return {
    type: "request_start", req_id: reqId, ts: NOW, token_id: "tok1", request_id: requestId, model: "id-qwen",
    served_name: "qwen", stream: true, prompt: "my prompt", session_id: "sess-abcdefgh", turn: 1, seq: 1,
  };
}

function speed(url: string) {
  const model = new URL(url, "http://x").searchParams.get("model");
  return {
    range: "1h",
    model_id: model,
    models: [{ model_id: "id-qwen", model: "qwen", requests: 6 }],
    last_finished_at: NOW - 60,
    count: 6,
    stride: 1,
    summary: model
      ? {
          ttft: { n: 6, mode: "max", p50: 1.25, max: 1.5 },
          decode: { n: 6, mode: "max", p50: 40, max: 44 },
          duration: { n: 6, mode: "max", p50: 6, max: 7 },
          warden_queue: { n: 6, mode: "max", p50: 0.25, max: 0.3 },
          prompt: { n: 6, mode: "max", p50: 1000, max: 1100 },
        }
      : null,
    points: model
      ? Array.from({ length: 6 }, (_, i) => ({
          id: `own-${i}`, finished_at: NOW - 600 + i * 60, model_id: "id-qwen", model: "qwen",
          prompt_tokens: 1000, completion_tokens: 201, finish_reason: "stop", queued_s: 0.25,
          ttft: 1.25, decode: 40, duration: 6, prompt_speed: 1000,
        }))
      : [],
  };
}

const LIVE = {
  count: 2,
  state: { qwen: "waiting" },
  requests: [
    {
      id: "own-live", model: "qwen", model_id: "id-qwen", phase: "answering", is_stream: true,
      prompt_tokens: 1000, completion_tokens: 50, out_tokens: 50, context_tokens: 1050, max_model_len: 4096,
      context_pct: 0.25, elapsed_s: 5, queued_s: 0, ttft_s: 1, ttft_total_s: 1, decode_tps_5s: 42,
      session_id: "sess-abcdefgh", forest_session: "f1", expected_start_s: null, expected_start_basis: null,
    },
    {
      id: "own-wait", model: "qwen", model_id: "id-qwen", phase: "queued", is_stream: true,
      prompt_tokens: 1000, completion_tokens: 0, out_tokens: 0, context_tokens: 1000, max_model_len: 4096,
      context_pct: 0.24, elapsed_s: 1, queued_s: null, ttft_s: null, ttft_total_s: null, decode_tps_5s: null,
      session_id: null, forest_session: null, expected_start_s: 4, expected_start_basis: "own",
    },
  ],
};

function stub(lensOn: boolean) {
  const urls: string[] = [];
  const f = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    urls.push(url);
    const j = (b: unknown, s = 200) => new Response(JSON.stringify(b), { status: s });
    if (url.startsWith("/api/me/key"))
      return j({ id: "tok1", name: "farm-07", prefix: "vw_k2x7a", created_at: "2026-09-12 10:00:00", expires_at: null,
        is_revoked: false, is_expired: false, is_paused: false, prompt_lens_view_available: lensOn });
    if (url.startsWith("/api/me/stats/speed")) return j(speed(url));
    if (url.startsWith("/api/me/stats/requests")) return j(LIVE);
    if (url.startsWith("/api/me/stats/tokens"))
      return j({ range: "1h", model_id: "id-qwen", since_minute: Math.floor(NOW / 60) - 60, now_minute: Math.floor(NOW / 60),
        minutes: [{ minute: Math.floor(NOW / 60) - 5, requests: 2, prompt_tokens: 2000, completion_tokens: 400 }] });
    if (url.startsWith("/api/me/stats/cache"))
      return j({ range: "1h", model_id: "id-qwen", cache: { requests: 6, prompt_tokens: 6000, own_reused_tokens: 2400,
        own_known_prompt_tokens: 6000, own_known_requests: 6, outcomes: { hit: 6, partial: 0, lost: 0, diverged: 0, cold: 0, compacted: 0, harness_rewrite: 0 },
        prefill_saved_s: 1.2, prefill_saved_basis: "configured", prefill_rate_tok_s: 2000 } });
    if (url === "/api/me/prompt-lens/ticket") return j({ ticket: "k1" });
    if (url.startsWith("/api/me/prompt-lens/kept")) return j({ kept: [] });
    if (url.includes("/capture"))
      return j({ id: "own-0", model: "qwen", started_ts: NOW - 70, ended_ts: NOW - 60, live: false, prompt: "my prompt",
        prompt_elided: 0, reply: "my reply", reasoning: "", finish_reason: "stop", prompt_tokens: 3, completion_tokens: 3,
        timing: { queued_s: 0.25, ttft_s: 1.25, duration_s: 6, decode_tps: 40 },
        note: "Captures are kept briefly in server memory, not on disk, and the platform's admins can see them too. You only ever see captures of requests made with your key." });
    return j({}, 404);
  });
  vi.stubGlobal("fetch", f);
  return urls;
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <KeyStatsPage />
    </SWRConfig>,
  );
}

beforeEach(() => {
  sessionStorage.setItem("vw-sessions-token", "jwt-1");
  sessionStorage.setItem("vw-sessions-exp", String(Date.now() + 600_000));
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
  FakeES.all = [];
  vi.stubGlobal("EventSource", FakeES as unknown as typeof EventSource);
  localStorage.clear();
});
afterEach(() => {
  cleanup();
  sessionStorage.clear();
});

describe("key holder stats — nothing fleet-wide", () => {
  it("only ever calls the key holder's own /api/me routes", async () => {
    const urls = stub(false);
    renderPage();
    await screen.findByTestId("key-stats-page");
    await waitFor(() => expect(urls.some((u) => u.startsWith("/api/me/stats/cache"))).toBe(true));
    for (const u of urls) expect(u.startsWith("/api/me/")).toBe(true);
    expect(urls.some((u) => u.includes("token_id"))).toBe(false);
  });

  it("renders no Key, Client IP or Install column, no host or engine section, no queue count", async () => {
    stub(false);
    renderPage();
    const table = await screen.findByTestId("live-requests-table");
    await waitFor(() => expect(table.querySelectorAll('[data-testid="req-row"]').length).toBe(2));
    const heads = Array.from(table.querySelectorAll('[role="columnheader"]')).map((h) => h.textContent);
    expect(heads).not.toContain("Key");
    expect(heads).not.toContain("Client IP");
    expect(heads).not.toContain("Install");
    for (const id of ["host", "engine", "now"]) expect(screen.queryByTestId(id)).toBeNull();
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/ahead|Queued behind|VRAM|GPU util|Power draw|Engine details|Prefill speed/);
    // The waiting estimate comes from the key's own requests: a time, not a count.
    expect(text).toMatch(/waiting ~3 s/);
    expect(screen.getByTestId("tile-prompt").textContent).toMatch(/Prompt speed/);
    expect(screen.getByTestId("key-privacy-note").textContent).toMatch(/nothing fleet-wide/);
  });

  it("keeps the fixed panel heights and 8 table slots", async () => {
    stub(false);
    renderPage();
    await screen.findByTestId("cache-card");
    expect(screen.getByTestId("cache-card").getAttribute("data-fixed-height")).toBe(`${FIXED.cacheCard}`);
    expect(screen.getByTestId("tile-ttft").style.height).toBe(`${FIXED.tile}px`);
    const body = screen.getByTestId("live-requests-table-body");
    await waitFor(() => expect(body.querySelectorAll('[data-testid="req-row"]').length).toBe(2));
    expect(body.querySelectorAll('[data-testid="req-row"], [data-testid="ghost-row"]').length).toBe(FIXED.tableRows);
  });
});

describe("key holder stats — Prompt Lens (D8, Q1): eyes open the Prompt Lens page", () => {
  it("/key/stats: no eye column when Prompt Lens is off", async () => {
    const urls = stub(false);
    renderPage();
    await screen.findByTestId("live-requests-table");
    await waitFor(() => expect(screen.getAllByTestId("req-row").length).toBe(2));
    await waitFor(() => expect(urls.some((u) => u.startsWith("/api/me/stats/speed?range=1h&model="))).toBe(true));
    expect(screen.queryAllByRole("button", { name: "Open in Prompt Lens" })).toHaveLength(0);
    expect(screen.queryAllByTestId("chart-view")).toHaveLength(0);
    const heads = Array.from(screen.getByTestId("live-requests-table").querySelectorAll('[role="columnheader"]'));
    expect(heads).toHaveLength(9);
  });

  it("/key/stats carries no dock and opens no stream", async () => {
    const urls = stub(true);
    renderPage();
    await screen.findAllByRole("button", { name: "Open in Prompt Lens" });
    expect(screen.queryByTestId("prompt-lens-dock")).toBeNull();
    expect(FakeES.all).toHaveLength(0);
    expect(urls.some((u) => u.includes("ticket"))).toBe(false);
  });

  it("/key/stats: the eye opens /key/prompt-lens?req=<that request>; no capture poll is ever fetched", async () => {
    nav.push.mockReset();
    const urls = stub(true);
    renderPage();
    await waitFor(() => expect(screen.getAllByTestId("chart-view").length).toBe(6));
    const table = screen.getByTestId("live-requests-table");
    const eyes = within(table).getAllByRole("button", { name: "Open in Prompt Lens" });
    expect(eyes).toHaveLength(2);
    fireEvent.click(eyes[0]);
    expect(nav.push).toHaveBeenCalledTimes(1);
    expect(nav.push.mock.calls[0][0]).toMatch(/^\/key\/prompt-lens\?req=[^&]+$/);
    expect(urls.some((u) => u.includes("/capture"))).toBe(false);
  });

  it("/key/stats: a chart dot opens the page too", async () => {
    nav.push.mockReset();
    stub(true);
    renderPage();
    await waitFor(() => expect(screen.getAllByTestId("chart-view").length).toBe(6));
    fireEvent.click(screen.getAllByTestId("chart-view")[0]);
    expect(nav.push).toHaveBeenCalledTimes(1);
    expect(nav.push.mock.calls[0][0]).toMatch(/^\/key\/prompt-lens\?req=/);
  });

  it("says Prompt Lens, never god mode", async () => {
    stub(true);
    renderPage();
    await screen.findAllByRole("button", { name: "Open in Prompt Lens" });
    expect(document.body.textContent ?? "").not.toMatch(/god[ -]?mode/i);
    for (const el of document.querySelectorAll("[aria-label],[title]")) {
      expect(`${el.getAttribute("aria-label") ?? ""} ${el.getAttribute("title") ?? ""}`).not.toMatch(/god[ -]?mode/i);
    }
    for (const t of document.querySelectorAll("title")) expect(t.textContent ?? "").not.toMatch(/god[ -]?mode/i);
  });
});

describe("key holder stats — anchored window for an idle key (owner 2026-11-22)", () => {
  const LAST = NOW - 3 * 86400;

  function stubIdle(lastAt: number | null) {
    const urls: string[] = [];
    const f = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      urls.push(url);
      const j = (b: unknown) => new Response(JSON.stringify(b), { status: 200 });
      const anchored = url.includes("until=");
      if (url.startsWith("/api/me/key"))
        return j({ id: "tok1", name: "farm-07", prefix: "vw_k2x7a", created_at: "2026-09-12 10:00:00", expires_at: null,
          is_revoked: false, is_expired: false, is_paused: false, prompt_lens_view_available: false });
      if (url.startsWith("/api/me/stats/speed")) {
        if (!anchored) return j({ ...speed(url), count: 0, points: [], models: [], summary: null, last_request_at: lastAt, last_finished_at: lastAt });
        return j({ ...speed(url), last_request_at: lastAt, until: LAST + 1 });
      }
      if (url.startsWith("/api/me/stats/requests")) return j({ count: 0, requests: [], state: {} });
      if (url.startsWith("/api/me/stats/tokens")) return j({ range: "1h", model_id: "id-qwen", since_minute: 0, now_minute: 0, minutes: [] });
      if (url.startsWith("/api/me/stats/cache")) return j({ range: "1h", model_id: "id-qwen", cache: { requests: 0, prompt_tokens: 0, own_reused_tokens: 0, own_known_prompt_tokens: 0, own_known_requests: 0, outcomes: {}, prefill_saved_s: null, prefill_saved_basis: "configured", prefill_rate_tok_s: null } });
      return j({});
    });
    vi.stubGlobal("fetch", f);
    return urls;
  }

  it("anchors every window feed to the last request and says so, with Back to now", async () => {
    const urls = stubIdle(LAST);
    renderPage();
    const notice = await screen.findByTestId("notice-line");
    await waitFor(() => expect(notice.textContent).toMatch(/No requests in the last hour · showing the hour up to your last request, 3 d ago/));
    await waitFor(() => {
      for (const p of ["speed?range=1h&model=", "tokens", "cache"])
        expect(urls.some((u) => u.includes(`/api/me/stats/${p}`) && u.includes("until="))).toBe(true);
    });
    await waitFor(() => expect(screen.getByTestId("tile-ttft").textContent).toMatch(/n=6/));
    expect(notice.getAttribute("data-fixed-height")).toBe("28");
    const n = urls.length;
    fireEvent.click(screen.getByTestId("notice-action"));
    await waitFor(() => expect(screen.getByTestId("notice-line").textContent).toMatch(/your last request was 3 d ago/));
    expect(screen.getByTestId("notice-action").textContent).toBe("Show last activity");
    await waitFor(() => expect(screen.getByTestId("tile-ttft").textContent).toMatch(/n=0/));
    expect(urls.slice(n).some((u) => u.includes("until="))).toBe(false);
  });

  it("keeps the empty state for a key that was never used", async () => {
    const urls = stubIdle(null);
    renderPage();
    const notice = await screen.findByTestId("notice-line");
    await waitFor(() => expect(urls.some((u) => u.startsWith("/api/me/stats/speed"))).toBe(true));
    expect(notice.textContent).toMatch(/Signed in with your API key/);
    expect(screen.queryByTestId("notice-action")).toBeNull();
    expect(urls.some((u) => u.includes("until="))).toBe(false);
    expect(screen.getByTestId("scope-tab-empty").textContent).toMatch(/no requests from this key yet/);
  });
});
