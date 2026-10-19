// Component tests for the collapsible Live logs card (Task 8).
//
// Covers the brief's failing-test list:
//   - live logs collapsed by default with last-line preview
//   - live logs auto-open on stalled and on failed
//   - manual toggle persists and wins until state changes
//   - localStorage throwing still renders
//
// LogStream is mocked out so the tests do not need an EventSource and can
// assert on the open/closed state via the presence of the body.

import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";
import { LiveLogsCollapsible } from "@/components/models/live-logs-collapsible";

// Mock LogStream so no EventSource / SSE is needed.
vi.mock("@/components/models/log-stream", () => ({
  LogStream: () => <div data-testid="log-stream-mock">stream</div>,
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("LiveLogsCollapsible — collapsed by default", () => {
  it("is collapsed by default with a last-line preview", () => {
    render(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="progressing"
        lastLogLine="Loading safetensors checkpoint shards: 50%"
        lastLogAgeS={5}
      />,
    );
    // The toggle button says "Show log" (collapsed).
    expect(screen.getByTestId("live-logs-toggle")).toHaveTextContent("Show log");
    // The body (LogStream) is not rendered.
    expect(screen.queryByTestId("live-logs-body")).toBeNull();
    // The preview line is shown.
    expect(screen.getByTestId("live-logs-preview")).toHaveTextContent(
      /Loading safetensors checkpoint shards: 50%/,
    );
    // Age is shown.
    expect(screen.getByText(/last line 5s ago:/)).toBeInTheDocument();
  });
});

describe("LiveLogsCollapsible — auto-open on stalled and on failed", () => {
  it("auto-opens when the load state becomes stalled", () => {
    const { rerender } = render(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="progressing"
        lastLogLine="quiet"
        lastLogAgeS={10}
      />,
    );
    expect(screen.queryByTestId("live-logs-body")).toBeNull();

    rerender(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="stalled"
        lastLogLine="quiet"
        lastLogAgeS={120}
      />,
    );
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
    expect(screen.getByTestId("live-logs-toggle")).toHaveTextContent("Hide log");
  });

  it("auto-opens when the model status becomes failed", () => {
    const { rerender } = render(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="progressing"
        lastLogLine="quiet"
        lastLogAgeS={10}
      />,
    );
    expect(screen.queryByTestId("live-logs-body")).toBeNull();

    rerender(
      <LiveLogsCollapsible
        modelId="m1"
        status="failed"
        loadState={null}
        lastLogLine="quiet"
        lastLogAgeS={10}
      />,
    );
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
  });
});

describe("LiveLogsCollapsible — manual toggle persists and wins until state changes", () => {
  it("a manual open persists in localStorage", () => {
    render(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="progressing"
        lastLogLine="x"
        lastLogAgeS={1}
      />,
    );
    expect(screen.queryByTestId("live-logs-body")).toBeNull();
    expect(localStorage.getItem("lmw.liveLogs.open")).toBeNull();

    act(() => {
      fireEvent.click(screen.getByTestId("live-logs-toggle"));
    });
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
    expect(localStorage.getItem("lmw.liveLogs.open")).toBe("true");
  });

  it("a manual close after auto-open wins until the state changes again", () => {
    // Start stalled → auto-open.
    const { rerender } = render(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="stalled"
        lastLogLine="x"
        lastLogAgeS={120}
      />,
    );
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();

    // Manually close.
    act(() => {
      fireEvent.click(screen.getByTestId("live-logs-toggle"));
    });
    expect(screen.queryByTestId("live-logs-body")).toBeNull();

    // Re-render with the SAME state — should NOT auto-open again.
    rerender(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="stalled"
        lastLogLine="x"
        lastLogAgeS={120}
      />,
    );
    expect(screen.queryByTestId("live-logs-body")).toBeNull();

    // State changes to progressing then back to stalled → auto-opens again.
    rerender(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="progressing"
        lastLogLine="x"
        lastLogAgeS={10}
      />,
    );
    rerender(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="stalled"
        lastLogLine="x"
        lastLogAgeS={130}
      />,
    );
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
  });
});

describe("LiveLogsCollapsible — localStorage throwing still renders", () => {
  beforeEach(() => {
    // Make localStorage.getItem throw.
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
  });

  it("still renders when localStorage throws", () => {
    render(
      <LiveLogsCollapsible
        modelId="m1"
        status="loading"
        loadState="progressing"
        lastLogLine="x"
        lastLogAgeS={1}
      />,
    );
    // Collapsed (default) — no throw, no body.
    expect(screen.queryByTestId("live-logs-body")).toBeNull();
    expect(screen.getByTestId("live-logs-toggle")).toHaveTextContent("Show log");

    // Toggling still works (setItem is swallowed).
    act(() => {
      fireEvent.click(screen.getByTestId("live-logs-toggle"));
    });
    expect(screen.getByTestId("live-logs-body")).toBeInTheDocument();
  });
});
