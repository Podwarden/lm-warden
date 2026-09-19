// The panel that replaced the single "Tokens / sec" tile.
//
// That tile showed (prompt + completion) / 60 over one minute — prefill and
// generation averaged into a number that is neither, and that moves when the
// prompt-to-completion ratio moves even though the hardware did not. This
// panel reports the two separately, each as average / max / mode.

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { ThroughputPanel } from "@/components/stats/throughput-panel";
import type { ThroughputResponse } from "@/lib/request-history";

afterEach(() => {
  cleanup();
});

const COVERAGE = {
  earliest_epoch: 1_757_000_000,
  retention_days: 30,
  max_rows: 200000,
  covers_window: true,
};

const FIXTURE: ThroughputResponse = {
  basis: "request",
  range: "24h",
  since_epoch: 1_757_000_000,
  selected_model_ids: null,
  prefill: { count: 1284, avg: 412.4, max: 980.0, mode: 380.2 },
  generation: { count: 1284, avg: 48.6, max: 71.3, mode: 46.1 },
  coverage: COVERAGE,
};

function renderPanel(over: Partial<React.ComponentProps<typeof ThroughputPanel>> = {}) {
  return render(
    <ThroughputPanel
      data={FIXTURE}
      basis="request"
      onBasisChange={() => {}}
      isLoading={false}
      error={undefined}
      range="24h"
      {...over}
    />,
  );
}

describe("ThroughputPanel", () => {
  it("reports prefill and generation as separate series", () => {
    renderPanel();
    expect(screen.getByTestId("tp-prefill-avg").textContent).toBe("412");
    expect(screen.getByTestId("tp-prefill-max").textContent).toBe("980");
    expect(screen.getByTestId("tp-prefill-mode").textContent).toBe("380");
    expect(screen.getByTestId("tp-generation-avg").textContent).toBe("49");
    expect(screen.getByTestId("tp-generation-max").textContent).toBe("71");
    expect(screen.getByTestId("tp-generation-mode").textContent).toBe("46");
  });

  it("renders an absent mode as a dash, never as zero", () => {
    // A sample too small to have a mode is not a deployment that ran at
    // 0 tok/s, and the panel must not let those two look alike.
    renderPanel({
      data: {
        ...FIXTURE,
        prefill: { count: 3, avg: 412.4, max: 980.0, mode: null },
      },
    });
    expect(screen.getByTestId("tp-prefill-mode").textContent).toBe("—");
  });

  it("says how many samples the figures rest on", () => {
    renderPanel();
    expect(screen.getByTestId("throughput-note").textContent).toContain("1,284");
  });

  it("switches basis when the toggle is used", () => {
    const onBasisChange = vi.fn();
    renderPanel({ onBasisChange });
    fireEvent.click(screen.getByRole("button", { name: /wall clock/i }));
    expect(onBasisChange).toHaveBeenCalledWith("wallclock");
  });

  it("names the wall-clock basis in the sample note when it is showing", () => {
    // The two bases count different things — requests vs minutes — and a
    // count with no unit beside it would read as the same quantity.
    renderPanel({
      basis: "wallclock",
      data: {
        ...FIXTURE,
        basis: "wallclock",
        prefill: { count: 1440, avg: 38.0, max: 1100.0, mode: 0 },
        generation: { count: 1440, avg: 12.0, max: 160.0, mode: 0 },
      },
    });
    expect(screen.getByTestId("throughput-note").textContent).toMatch(/minute/i);
    // Zero IS a reading on this basis — an idle minute — so it prints as 0.
    expect(screen.getByTestId("tp-prefill-mode").textContent).toBe("0");
  });

  it("does not call the per-request prompt figure a prefill rate", () => {
    // prompt_tokens / ttft counts prefix-cached tokens the engine never
    // computed. On a production deployment the median prompt is 46k tokens with a
    // 1.9s TTFT — 26,000 tok/s, roughly 13x more than four A4000s can
    // physically prefill through a 27B. The figure is real, but it measures
    // cache hits, so it must not be labelled as a compute rate.
    renderPanel({ basis: "request" });
    expect(screen.queryByText(/^prefill$/i)).toBeNull();
    expect(screen.getByTestId("throughput-panel").textContent).toMatch(/prefix.cache/i);
  });

  it("says the per-request generation figure is one request's share", () => {
    // (completion-1)/(duration-ttft) is what ONE request experienced under
    // concurrent batching, which is a fraction of the engine's aggregate.
    renderPanel({ basis: "request" });
    expect(screen.getByTestId("throughput-panel").textContent).toMatch(
      /not the engine|one request/i,
    );
  });

  it("does not claim TTFT includes the admission queue wait", () => {
    // It does not. `started_monotonic` in app/proxy/routes.py is read AFTER
    // the scheduler admits the request, so the warden's queue was never
    // inside ttft_s — it is now its own column (migration 0032). What DOES
    // still sit inside ttft_s is the engine's own waiting queue, and that is
    // what the caveat has to name.
    renderPanel();
    const note = screen.getByTestId("throughput-note").textContent ?? "";
    expect(note).not.toMatch(/includes queue wait/i);
    expect(note).toMatch(/engine/i);
  });

  it("shows an empty state rather than zeros when nothing was served", () => {
    renderPanel({
      data: {
        ...FIXTURE,
        prefill: { count: 0, avg: null, max: null, mode: null },
        generation: { count: 0, avg: null, max: null, mode: null },
      },
    });
    expect(screen.getByTestId("throughput-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("tp-prefill-avg")).toBeNull();
  });
});
