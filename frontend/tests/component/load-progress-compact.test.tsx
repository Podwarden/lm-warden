// Component tests for the compact load-progress card bar (Task 8).
//
// Covers:
//   - renders nothing when progress is undefined
//   - shows the current step name and a green dot while progressing
//   - shows the amber pill and "no progress for Xm" while stalled
//   - shows the overall bar only when usual_total_s is known
//   - shows elapsed always

import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { LoadProgressCompact } from "@/components/models/load-progress/load-progress-compact";
import type { LoadProgressOut } from "@/lib/load-progress";

function compactSnapshot(overrides: Partial<LoadProgressOut> = {}): LoadProgressOut {
  return {
    model_id: "m1",
    backend: "vllm",
    layout: null,
    elapsed_s: overrides.elapsed_s ?? 30,
    state: overrides.state ?? "progressing",
    stalled_for_s: overrides.stalled_for_s ?? 0,
    give_up_at: null,
    reason: null,
    hint: null,
    usual_total_s: overrides.usual_total_s ?? null,
    ready_eta: null,
    current_step: overrides.current_step ?? "WEIGHTS",
    steps: overrides.steps ?? [
      {
        key: "START",
        label: "Start engine",
        status: "done",
        duration_s: 5,
        elapsed_s: null,
        count: null,
        total: null,
        usual_s: null,
      },
      {
        key: "WEIGHTS",
        label: "Load weights",
        status: "running",
        duration_s: null,
        elapsed_s: 25,
        count: 3,
        total: 8,
        usual_s: null,
      },
    ],
    vitals: {
      series: [],
      last_log_age_s: null,
      last_log_line: null,
      log_quiet_normal: false,
      gpu_mem_total_mib: null,
      gpu_mem_supported: true,
    },
  };
}

afterEach(() => cleanup());

describe("LoadProgressCompact", () => {
  it("renders nothing when progress is undefined", () => {
    const { container } = render(<LoadProgressCompact progress={undefined} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the current step name and a green dot while progressing", () => {
    render(<LoadProgressCompact progress={compactSnapshot()} />);
    expect(screen.getByTestId("load-progress-compact")).toBeInTheDocument();
    expect(screen.getByText("Load weights")).toBeInTheDocument();
  });

  it("maps a raw key to its label when falling back to current_step", () => {
    // When no step is running/stalled, the compact card falls back to
    // current_step.  If current_step is a raw key (e.g. "WEIGHTS"), it must
    // be mapped to the label from the steps array — the single source of
    // labels — not shown as the raw key.
    const s = compactSnapshot({
      current_step: "WEIGHTS",
      steps: [
        {
          key: "START",
          label: "Start engine",
          status: "done",
          duration_s: 5,
          elapsed_s: null,
          count: null,
          total: null,
          usual_s: null,
        },
        {
          key: "WEIGHTS",
          label: "Load weights",
          status: "done",
          duration_s: 10,
          elapsed_s: null,
          count: null,
          total: null,
          usual_s: null,
        },
      ],
    });
    render(<LoadProgressCompact progress={s} />);
    // The raw key "WEIGHTS" must NOT appear; the label "Load weights" must.
    expect(screen.getByText("Load weights")).toBeInTheDocument();
  });

  it("shows nothing for current_step when it is an unknown key", () => {
    // If current_step is neither a known key nor a known label, show nothing
    // rather than a confusing raw string.
    const s = compactSnapshot({
      current_step: "UNKNOWN_STEP",
      steps: [
        {
          key: "START",
          label: "Start engine",
          status: "done",
          duration_s: 5,
          elapsed_s: null,
          count: null,
          total: null,
          usual_s: null,
        },
      ],
    });
    const { container } = render(<LoadProgressCompact progress={s} />);
    // The raw key must not appear anywhere in the compact card.
    expect(container.textContent).not.toContain("UNKNOWN_STEP");
  });

  it("shows the amber pill and 'no progress for Xm' while stalled", () => {
    render(
      <LoadProgressCompact
        progress={compactSnapshot({ state: "stalled", stalled_for_s: 600 })}
      />,
    );
    expect(screen.getByText(/no progress for 10m/)).toBeInTheDocument();
  });

  it("shows the overall bar only when usual_total_s is known", () => {
    const { rerender } = render(
      <LoadProgressCompact progress={compactSnapshot({ usual_total_s: null })} />,
    );
    expect(screen.queryByTestId("compact-overall-bar")).toBeNull();

    rerender(
      <LoadProgressCompact
        progress={compactSnapshot({ usual_total_s: 360, elapsed_s: 120 })}
      />,
    );
    expect(screen.getByTestId("compact-overall-bar")).toBeInTheDocument();
  });

  it("shows elapsed always, even without history", () => {
    render(
      <LoadProgressCompact
        progress={compactSnapshot({ usual_total_s: null, elapsed_s: 45 })}
      />,
    );
    expect(screen.getByText(/45s elapsed/)).toBeInTheDocument();
  });
});

describe("LoadProgressCompact — give_up state", () => {
  it("shows amber 'Giving up — stopping engine', not green", () => {
    render(
      <LoadProgressCompact
        progress={compactSnapshot({ state: "give_up", stalled_for_s: 1200 })}
      />,
    );
    expect(screen.getByText("Giving up — stopping engine")).toBeInTheDocument();
    // Not the green "Loading" text.
    expect(screen.queryByText("Loading")).toBeNull();
  });

  it("shows amber for give_up, not the 'no progress for Xm' text", () => {
    render(
      <LoadProgressCompact
        progress={compactSnapshot({ state: "give_up", stalled_for_s: 600 })}
      />,
    );
    expect(screen.queryByText(/no progress for/)).toBeNull();
  });
});

describe("LoadProgressCompact — formatDuration (shared helper)", () => {
  it("shows one decimal for sub-10s elapsed", () => {
    render(
      <LoadProgressCompact
        progress={compactSnapshot({ usual_total_s: null, elapsed_s: 5.2 })}
      />,
    );
    // 5.2s — one decimal, matching the checklist's behaviour.
    expect(screen.getByText(/5\.2s elapsed/)).toBeInTheDocument();
  });

  it("rounds to an integer for 10–60s", () => {
    render(
      <LoadProgressCompact
        progress={compactSnapshot({ usual_total_s: null, elapsed_s: 45 })}
      />,
    );
    expect(screen.getByText(/45s elapsed/)).toBeInTheDocument();
  });
});
