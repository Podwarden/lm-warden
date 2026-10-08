// LiveRequestsPanel's optional row callbacks (forest full window: hover marks
// the session's limb, click moves the camera to it). Without the props the
// panel must render exactly as before.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { LiveRequestsPanel } from "@/components/stats/live-panels";
import type { LiveRequestRow } from "@/lib/live-stats";

function row(over: Partial<LiveRequestRow> = {}): LiveRequestRow {
  return {
    id: "r1",
    token_id: null,
    token_name: "dev",
    client_ip: "10.0.0.1",
    model: "model-a",
    path: "/v1/chat/completions",
    prompt_tokens: 100,
    completion_tokens: 0,
    context_tokens: 100,
    max_model_len: 2000,
    context_pct: 0.05,
    elapsed_s: 1.2,
    phase: "prefill",
    orphan: false,
    session_id: "sess-abcdef12",
    cache_est_tokens: null,
    cache_est_pct: null,
    ...over,
  };
}

describe("LiveRequestsPanel row callbacks", () => {
  afterEach(cleanup);

  it("without callbacks the rows carry no interactive affordance (rendering unchanged)", () => {
    render(<LiveRequestsPanel rows={[row()]} error={null} isLoading={false} />);
    const tr = screen.getByTestId("live-row");
    expect(tr.className).not.toMatch(/cursor-pointer/);
    expect(tr.getAttribute("tabindex")).toBeNull();
  });

  it("hover reports the row, leave reports null, click reports the row", () => {
    const onRowHover = vi.fn();
    const onRowClick = vi.fn();
    const r = row();
    render(
      <LiveRequestsPanel rows={[r]} error={null} isLoading={false} onRowHover={onRowHover} onRowClick={onRowClick} />,
    );
    const tr = screen.getByTestId("live-row");
    fireEvent.mouseEnter(tr);
    expect(onRowHover).toHaveBeenLastCalledWith(r);
    fireEvent.mouseLeave(tr);
    expect(onRowHover).toHaveBeenLastCalledWith(null);
    fireEvent.click(tr);
    expect(onRowClick).toHaveBeenCalledWith(r);
    expect(tr.className).toMatch(/cursor-pointer/);
  });
});

describe("LiveRequestsPanel compact (forest full window, spec §6.4 columns)", () => {
  afterEach(cleanup);
  const headers = () => screen.getAllByRole("columnheader").map((h) => h.textContent);

  it("default render is unchanged: the Stats page columns and the 56rem table", () => {
    render(<LiveRequestsPanel rows={[row()]} error={null} isLoading={false} />);
    expect(headers()).toEqual(["Token", "Client IP", "Session", "Model", "Phase", "Context window", "Elapsed"]);
    expect(screen.getByRole("table").className).toMatch(/min-w-\[56rem\]/);
  });

  it("compact shows key, session, phase, context, cache, generated, elapsed with no min-width", () => {
    const r = row({ token_name: "farm", context_tokens: 41_200, completion_tokens: 212, cache_est_pct: 0.97, cache_est_tokens: 40_000, elapsed_s: 6.1 });
    render(<LiveRequestsPanel rows={[r]} error={null} isLoading={false} compact />);
    expect(headers()).toEqual(["Key", "Session", "Phase", "Ctx", "Cache", "Gen", "Time"]);
    expect(screen.getByRole("table").className).not.toMatch(/min-w-/);
    const tr = within(screen.getByTestId("live-row"));
    expect(tr.getByText("farm")).toBeInTheDocument();
    expect(tr.getByText("sess-abc")).toBeInTheDocument();
    expect(tr.getByText("41.2k")).toBeInTheDocument();
    expect(tr.getByText("97%")).toBeInTheDocument();
    expect(tr.getByText("212")).toBeInTheDocument();
    expect(tr.queryByTestId("badge-cache")).toBeNull(); // the cache has its own column
  });

  it("compact rows keep the hover/click callbacks", () => {
    const onRowClick = vi.fn();
    const r = row();
    render(<LiveRequestsPanel rows={[r]} error={null} isLoading={false} compact onRowClick={onRowClick} />);
    fireEvent.click(screen.getByTestId("live-row"));
    expect(onRowClick).toHaveBeenCalledWith(r);
  });
});

describe("LiveRequestsPanel row callbacks: exits, unmount, keyboard", () => {
  afterEach(cleanup);

  it("a hovered row that exits clears the hover", () => {
    const onRowHover = vi.fn();
    const r = row();
    const { rerender } = render(<LiveRequestsPanel rows={[r]} error={null} isLoading={false} onRowHover={onRowHover} />);
    fireEvent.mouseEnter(screen.getByTestId("live-row"));
    expect(onRowHover).toHaveBeenLastCalledWith(r);
    rerender(<LiveRequestsPanel rows={[]} error={null} isLoading={false} onRowHover={onRowHover} />);
    expect(onRowHover).toHaveBeenLastCalledWith(null);
  });

  it("unmounting while a row is hovered clears the hover", () => {
    const onRowHover = vi.fn();
    const { unmount } = render(<LiveRequestsPanel rows={[row()]} error={null} isLoading={false} onRowHover={onRowHover} />);
    fireEvent.mouseEnter(screen.getByTestId("live-row"));
    unmount();
    expect(onRowHover).toHaveBeenLastCalledWith(null);
  });

  it.each([false, true])("clickable rows are keyboard targets (compact=%s): tabIndex 0, Enter and Space, a name", (compact) => {
    const onRowClick = vi.fn();
    const r = row();
    render(<LiveRequestsPanel rows={[r]} error={null} isLoading={false} onRowClick={onRowClick} compact={compact} />);
    const tr = screen.getByTestId("live-row");
    expect(tr).toHaveAttribute("tabindex", "0");
    expect(tr.getAttribute("aria-label")).toMatch(/session sess-abc/);
    fireEvent.keyDown(tr, { key: "Enter" });
    fireEvent.keyDown(tr, { key: " " });
    expect(onRowClick).toHaveBeenCalledTimes(2);
  });

  it("without onRowClick a row has no keyboard affordance or name (default unchanged)", () => {
    render(<LiveRequestsPanel rows={[row()]} error={null} isLoading={false} onRowHover={vi.fn()} />);
    const tr = screen.getByTestId("live-row");
    expect(tr).not.toHaveAttribute("tabindex");
    expect(tr).not.toHaveAttribute("aria-label");
  });
});
