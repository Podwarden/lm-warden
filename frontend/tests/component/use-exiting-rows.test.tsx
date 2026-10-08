// Row enter/exit animation for the in-flight table: the hook that keeps a
// vanished row around for its exit, flags new rows as entering, and the table
// behaviour built on it.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, renderHook, screen } from "@testing-library/react";
import { ENTER_FLIP_MS, useExitingRows } from "@/lib/use-exiting-rows";
import { LiveRequestsPanel, ROW_EXIT_MS } from "@/components/stats/live-panels";
import type { LiveRequestRow } from "@/lib/live-stats";

type R = { id: string; v?: number };
const keyOf = (r: R) => r.id;
const ids = (items: { key: string; state: string }[]) => items.map((i) => `${i.key}:${i.state}`);

function setReducedMotion(on: boolean) {
  vi.stubGlobal(
    "matchMedia",
    (q: string) =>
      ({
        matches: on && q.includes("reduce"),
        media: q,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {},
      }) as unknown as MediaQueryList,
  );
}

beforeEach(() => {
  vi.useFakeTimers();
  setReducedMotion(false);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useExitingRows", () => {
  it("renders the first paint as live (no enter animation on load)", () => {
    const { result } = renderHook(() => useExitingRows([{ id: "a" }, { id: "b" }], keyOf, 280));
    expect(ids(result.current)).toEqual(["a:live", "b:live"]);
  });

  it("keeps a vanished row as exiting in place, then drops it after the timeout", () => {
    const { result, rerender } = renderHook(({ rows }) => useExitingRows(rows, keyOf, 280), {
      initialProps: { rows: [{ id: "a" }, { id: "b" }, { id: "c" }] as R[] },
    });
    rerender({ rows: [{ id: "a" }, { id: "c" }] });
    expect(ids(result.current)).toEqual(["a:live", "b:exiting", "c:live"]);
    act(() => void vi.advanceTimersByTime(279));
    expect(result.current.map((i) => i.key)).toContain("b");
    act(() => void vi.advanceTimersByTime(2));
    expect(ids(result.current)).toEqual(["a:live", "c:live"]);
  });

  it("does not restart the exit when a new poll array still lacks the row", () => {
    const { result, rerender } = renderHook(({ rows }) => useExitingRows(rows, keyOf, 280), {
      initialProps: { rows: [{ id: "a" }, { id: "b" }] as R[] },
    });
    rerender({ rows: [{ id: "a" }] });
    act(() => void vi.advanceTimersByTime(200));
    rerender({ rows: [{ id: "a" }] }); // a fresh array, same content
    act(() => void vi.advanceTimersByTime(100)); // 300 ms since it left
    expect(ids(result.current)).toEqual(["a:live"]);
  });

  it("cancels the exit when the row reappears", () => {
    const { result, rerender } = renderHook(({ rows }) => useExitingRows(rows, keyOf, 280), {
      initialProps: { rows: [{ id: "a" }, { id: "b" }] as R[] },
    });
    rerender({ rows: [{ id: "a" }] });
    expect(ids(result.current)).toContain("b:exiting");
    rerender({ rows: [{ id: "a" }, { id: "b" }] });
    expect(ids(result.current)).toEqual(["a:live", "b:live"]);
    act(() => void vi.advanceTimersByTime(1000));
    expect(ids(result.current)).toEqual(["a:live", "b:live"]);
  });

  it("flags a new row entering, then flips it to live", () => {
    const { result, rerender } = renderHook(({ rows }) => useExitingRows(rows, keyOf, 280), {
      initialProps: { rows: [{ id: "a" }] as R[] },
    });
    rerender({ rows: [{ id: "a" }, { id: "n" }] });
    expect(ids(result.current)).toEqual(["a:live", "n:entering"]);
    act(() => void vi.advanceTimersByTime(ENTER_FLIP_MS + 1));
    expect(ids(result.current)).toEqual(["a:live", "n:live"]);
  });

  it("passes updated row data through for survivors", () => {
    const { result, rerender } = renderHook(({ rows }) => useExitingRows(rows, keyOf, 280), {
      initialProps: { rows: [{ id: "a", v: 1 }] as R[] },
    });
    rerender({ rows: [{ id: "a", v: 2 }] });
    expect(result.current[0].row.v).toBe(2);
  });

  it("does nothing animated under prefers-reduced-motion", () => {
    setReducedMotion(true);
    const { result, rerender } = renderHook(({ rows }) => useExitingRows(rows, keyOf, 280), {
      initialProps: { rows: [{ id: "a" }, { id: "b" }] as R[] },
    });
    rerender({ rows: [{ id: "a" }, { id: "n" }] });
    expect(ids(result.current)).toEqual(["a:live", "n:live"]);
  });
});

function req(id: string, over: Partial<LiveRequestRow> = {}): LiveRequestRow {
  return {
    id,
    token_id: "t1",
    token_name: "dev",
    client_ip: "10.0.0.1",
    model: "model-a",
    path: "/v1/chat/completions",
    prompt_tokens: 1000,
    completion_tokens: 0,
    context_tokens: 1000,
    max_model_len: 8000,
    context_pct: 0.125,
    elapsed_s: 1,
    phase: "answering",
    orphan: false,
    session_id: null,
    cache_est_tokens: null,
    cache_est_pct: null,
    ...over,
  };
}

const panel = (rows: LiveRequestRow[]) => (
  <LiveRequestsPanel rows={rows} error={null} isLoading={false} />
);

describe("in-flight table: rows animate out and in", () => {
  it("keeps a removed row in the DOM (aria-hidden, data-exiting) for the window, then removes it", () => {
    const { rerender } = render(panel([req("a"), req("b")]));
    expect(screen.getAllByTestId("live-row")).toHaveLength(2);
    rerender(panel([req("a")]));
    const rows = screen.getAllByTestId("live-row");
    expect(rows).toHaveLength(2);
    const gone = rows.find((r) => r.hasAttribute("data-exiting"))!;
    expect(gone.getAttribute("aria-hidden")).toBe("true");
    expect(gone.className).toContain("pointer-events-none");
    // the count is live rows only
    expect(screen.getByText("1 active")).toBeInTheDocument();
    act(() => void vi.advanceTimersByTime(ROW_EXIT_MS + 5));
    expect(screen.getAllByTestId("live-row")).toHaveLength(1);
  });

  it("survivors keep their DOM node across polls (keyed by id, not index)", () => {
    const { rerender } = render(panel([req("a"), req("b")]));
    const b = screen.getAllByTestId("live-row")[1];
    rerender(panel([req("b")]));
    act(() => void vi.advanceTimersByTime(ROW_EXIT_MS + 5));
    expect(screen.getAllByTestId("live-row")[0]).toBe(b);
  });

  it("a new row arrives collapsed, then expands", () => {
    const { rerender } = render(panel([req("a")]));
    rerender(panel([req("a"), req("n")]));
    const fresh = screen.getAllByTestId("live-row")[1];
    expect(fresh.getAttribute("data-entering")).toBe("true");
    act(() => void vi.advanceTimersByTime(ENTER_FLIP_MS + 1));
    expect(fresh.hasAttribute("data-entering")).toBe(false);
  });

  it("the last row leaving still animates before the empty message", () => {
    const { rerender } = render(panel([req("a")]));
    rerender(panel([]));
    expect(screen.getAllByTestId("live-row")).toHaveLength(1);
    expect(screen.getByText("0 active")).toBeInTheDocument();
    act(() => void vi.advanceTimersByTime(ROW_EXIT_MS + 5));
    expect(screen.queryByTestId("live-row")).toBeNull();
    expect(screen.getByText(/Nothing in flight/)).toBeInTheDocument();
  });

  it("under reduced motion rows vanish at once", () => {
    setReducedMotion(true);
    const { rerender } = render(panel([req("a"), req("b")]));
    rerender(panel([req("a")]));
    expect(screen.getAllByTestId("live-row")).toHaveLength(1);
  });
});

describe("in-flight table: compact phase column", () => {
  it("reserves badge slots only while some visible row has the badge", () => {
    const { rerender } = render(panel([req("a")]));
    expect(screen.queryByTestId("badge-cache")).toBeNull();
    expect(document.querySelector(".w-\\[3\\.25rem\\]")).toBeNull();
    expect(document.querySelector(".w-\\[3\\.75rem\\]")).toBeNull();
    rerender(panel([req("a", { cache_est_pct: 0.9 })]));
    expect(document.querySelector(".w-\\[3\\.25rem\\]")).not.toBeNull();
    expect(document.querySelector(".w-\\[3\\.75rem\\]")).toBeNull();
    rerender(
      panel([req("a", { cache_est_pct: 0.9 }), req("b", { phase: "prefill", slow_cause: "slow" })]),
    );
    // both slots now exist on every row, so rows stay aligned
    expect(document.querySelectorAll(".w-\\[3\\.75rem\\]")).toHaveLength(2);
    expect(document.querySelectorAll(".w-\\[3\\.25rem\\]")).toHaveLength(2);
  });

  it("sizes the Phase column to content and gives the free width to the context bar", () => {
    render(panel([req("a")]));
    const phaseTh = screen.getByText("Phase");
    expect(phaseTh.className).toContain("w-px");
    expect(phaseTh.className).toContain("whitespace-nowrap");
    expect(screen.getByText("Context window").className).toContain("w-full");
  });
});
