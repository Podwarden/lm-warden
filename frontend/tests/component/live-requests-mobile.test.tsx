// Phone layout of the In-flight panel: below md (768px) a two-line card per
// request replaces the table, with the same animated rows.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, screen, within } from "@testing-library/react";
import { LiveRequestsPanel, ROW_EXIT_MS, sharedModel } from "@/components/stats/live-panels";
import { ENTER_FLIP_MS } from "@/lib/use-exiting-rows";
import { MOBILE_QUERY } from "@/lib/use-media-query";
import type { LiveRequestRow } from "@/lib/live-stats";

function setViewport(mobile: boolean, reduced = false) {
  vi.stubGlobal(
    "matchMedia",
    (q: string) =>
      ({
        matches: q === MOBILE_QUERY ? mobile : q.includes("reduce") ? reduced : false,
        media: q,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {},
      }) as unknown as MediaQueryList,
  );
}

function req(id: string, over: Partial<LiveRequestRow> = {}): LiveRequestRow {
  return {
    id,
    token_id: "t1",
    token_name: "dev",
    client_ip: "10.0.0.7",
    model: "qwen3.8-27b",
    path: "/v1/chat/completions",
    prompt_tokens: 31_000,
    completion_tokens: 200,
    context_tokens: 31_200,
    max_model_len: 131_072,
    context_pct: 0.238,
    elapsed_s: 12.3,
    phase: "answering",
    orphan: false,
    session_id: "0b1c2d3e-aaaa-bbbb-cccc-ddddeeeeffff",
    cache_est_tokens: null,
    cache_est_pct: null,
    ...over,
  };
}

const panel = (rows: LiveRequestRow[]) => (
  <LiveRequestsPanel rows={rows} error={null} isLoading={false} />
);

beforeEach(() => {
  vi.useFakeTimers();
  setViewport(true);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("In-flight panel on a phone", () => {
  it("renders cards, not the table, below md; the table at md and up", () => {
    const { unmount } = render(panel([req("a")]));
    expect(screen.getByTestId("live-cards")).toBeInTheDocument();
    expect(document.querySelector("table")).toBeNull();
    unmount();
    setViewport(false);
    render(panel([req("a")]));
    expect(screen.queryByTestId("live-cards")).toBeNull();
    expect(document.querySelector("table")).not.toBeNull();
  });

  it("puts session, token, elapsed on line 1 and phase + context on line 2", () => {
    render(panel([req("a", { phase: "thinking" })]));
    const card = screen.getByTestId("live-card");
    expect(within(card).getByTestId("card-session").textContent).toBe("0b1c2d3e");
    const link = within(card).getByText("dev");
    expect(link.closest("a")?.getAttribute("href")).toBe("/tokens/t1");
    expect(link.closest("a")?.className).toContain("py-3"); // 44px tap area
    expect(card.textContent).toContain("12s");
    expect(within(card).getByText("thinking")).toBeInTheDocument();
    expect(within(card).getByRole("meter")).toBeInTheDocument();
    expect(card.textContent).toContain("31.2k/131.1k");
  });

  it("the card's session carries the same source tooltip and inferred dimming", () => {
    const { unmount } = render(panel([req("a", { session_source: "session_id_header" })]));
    expect(screen.getByTestId("card-session").getAttribute("title")).toContain("from the session-id header");
    unmount();
    render(panel([req("a", { session_id: null, session_source: "prompt_hash" })]));
    const s = screen.getByTestId("card-session");
    expect(s.className).toContain("opacity-60");
    expect(s.getAttribute("title")).toBe("conversation inferred from first message");
  });

  it("moves client IP and model into the card's title and aria-label", () => {
    render(panel([req("a")]));
    const card = screen.getByTestId("live-card");
    expect(card.getAttribute("title")).toBe("10.0.0.7 · qwen3.8-27b");
    expect(card.getAttribute("aria-label")).toContain("10.0.0.7");
    expect(card.textContent).not.toContain("10.0.0.7");
  });

  it("shows a shared model once in the panel header, a per-card suffix otherwise", () => {
    const { unmount } = render(panel([req("a"), req("b")]));
    expect(screen.getByTestId("live-requests").textContent).toContain("· qwen3.8-27b");
    expect(screen.queryByTestId("card-model")).toBeNull();
    unmount();
    render(panel([req("a"), req("b", { model: "llama-8b" })]));
    expect(screen.getByTestId("live-requests").textContent).not.toContain("· qwen3.8-27b");
    expect(screen.getAllByTestId("card-model").map((n) => n.textContent)).toEqual([
      "qwen3.8-27b",
      "llama-8b",
    ]);
  });

  it("sharedModel ignores nothing: empty and mixed are null", () => {
    expect(sharedModel([])).toBeNull();
    expect(sharedModel([req("a"), req("b", { model: "x" })])).toBeNull();
    expect(sharedModel([req("a"), req("b")])).toBe("qwen3.8-27b");
  });

  it("keeps the badges, orphan tag and anonymous tokens", () => {
    render(
      panel([
        req("a", { phase: "prefill", cache_est_tokens: 30_000, cache_est_pct: 0.96, slow_cause: "queued", ahead_count: 3, dp_rank: 3 }),
        req("b", { token_id: null, token_name: null, orphan: true }),
      ]),
    );
    expect(screen.getByTestId("badge-cache").textContent).toBe("96%");
    expect(screen.getByTestId("badge-cause").textContent).toBe("3·r3");
    expect(screen.getByText("anonymous")).toBeInTheDocument();
    expect(screen.getByText("orphan")).toBeInTheDocument();
  });

  it.each([
    ["queued", "queued"],
    ["prefill", "prefill"],
    ["waiting", "waiting"],
    ["thinking", "thinking"],
    ["tool_call", "tool call"],
    ["answering", "answering"],
  ])("card shows the %s phase", (phase, text) => {
    render(panel([req("a", { phase, elapsed_s: 4.2 })]));
    const pill = within(screen.getByTestId("live-card")).getByText(new RegExp(`^${text}`));
    expect(pill.closest("span[aria-label]")?.getAttribute("aria-label")).toContain(text);
  });

  it("animates cards out and in like table rows", () => {
    const { rerender } = render(panel([req("a"), req("b")]));
    rerender(panel([req("a"), req("b"), req("n")]));
    const fresh = screen.getAllByTestId("live-card")[2];
    expect(fresh.getAttribute("data-entering")).toBe("true");
    act(() => void vi.advanceTimersByTime(ENTER_FLIP_MS + 1));
    expect(fresh.hasAttribute("data-entering")).toBe(false);

    rerender(panel([req("a"), req("n")]));
    const cards = screen.getAllByTestId("live-card");
    expect(cards).toHaveLength(3);
    const gone = cards.find((c) => c.hasAttribute("data-exiting"))!;
    expect(gone.getAttribute("aria-hidden")).toBe("true");
    expect(screen.getByText("2 active")).toBeInTheDocument();
    act(() => void vi.advanceTimersByTime(ROW_EXIT_MS + 5));
    expect(screen.getAllByTestId("live-card")).toHaveLength(2);
  });

  it("reduced motion removes cards at once", () => {
    setViewport(true, true);
    const { rerender } = render(panel([req("a"), req("b")]));
    rerender(panel([req("a")]));
    expect(screen.getAllByTestId("live-card")).toHaveLength(1);
  });
});
