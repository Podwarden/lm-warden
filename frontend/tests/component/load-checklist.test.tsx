// Component tests for the load-progress checklist (Task 8).
//
// Covers the brief's failing-test list:
//   - renders steps in order with done durations
//   - exact bar from count/total
//   - usual-time bar never reaches 100% while running
//   - stalled shows amber pill, give-up time and force-unload
//   - no overall bar or eta without history
//
// The checklist is a pure render of a ``LoadProgressOut`` snapshot, so the
// tests build snapshots directly and assert on the DOM — no API mocking,
// no SWR.  The page-level tests (model-page-load-progress.test.tsx) cover
// the integration with the model detail page.

import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { LoadChecklist } from "@/components/models/load-progress/load-checklist";
import type { LoadProgressOut } from "@/lib/load-progress";

function step(overrides: Partial<LoadProgressOut["steps"][number]>): LoadProgressOut["steps"][number] {
  return {
    key: overrides.key ?? "START",
    label: overrides.label ?? "Start engine",
    status: overrides.status ?? "pending",
    duration_s: overrides.duration_s ?? null,
    elapsed_s: overrides.elapsed_s ?? null,
    count: overrides.count ?? null,
    total: overrides.total ?? null,
    usual_s: overrides.usual_s ?? null,
    detail: overrides.detail ?? null,
  };
}

function vitals(overrides: Partial<LoadProgressOut["vitals"]> = {}): LoadProgressOut["vitals"] {
  return {
    series: overrides.series ?? [],
    last_log_age_s: overrides.last_log_age_s ?? null,
    last_log_line: overrides.last_log_line ?? null,
    log_quiet_normal: overrides.log_quiet_normal ?? false,
    gpu_mem_total_mib: overrides.gpu_mem_total_mib ?? null,
    gpu_mem_supported: overrides.gpu_mem_supported ?? true,
  };
}

function snapshot(overrides: Partial<LoadProgressOut> = {}): LoadProgressOut {
  return {
    model_id: "m1",
    backend: "vllm",
    layout: overrides.layout ?? null,
    elapsed_s: overrides.elapsed_s ?? 10,
    state: overrides.state ?? "progressing",
    stalled_for_s: overrides.stalled_for_s ?? 0,
    give_up_at: overrides.give_up_at ?? null,
    reason: overrides.reason ?? null,
    hint: overrides.hint ?? null,
    usual_total_s: overrides.usual_total_s ?? null,
    ready_eta: overrides.ready_eta ?? null,
    current_step: overrides.current_step ?? "WEIGHTS",
    steps: overrides.steps ?? [
      step({ key: "START", label: "Start engine", status: "done", duration_s: 5 }),
      step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 5, count: 3, total: 8 }),
      step({ key: "COMPILE", label: "Compile kernels", status: "pending", usual_s: 120 }),
      step({ key: "KV", label: "Size KV cache", status: "pending", usual_s: 13 }),
      step({ key: "TUNE", label: "Tune kernels", status: "pending", usual_s: 300 }),
      step({ key: "GRAPHS", label: "Capture CUDA graphs", status: "pending", usual_s: 3 }),
      step({ key: "WARMUP", label: "Warm-up request", status: "pending", usual_s: 2 }),
    ],
    vitals: overrides.vitals ?? vitals(),
  };
}

afterEach(() => cleanup());

describe("LoadChecklist — steps in order", () => {
  it("renders steps in order with done durations", () => {
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "done", duration_s: 5.2 }),
        step({ key: "WEIGHTS", label: "Load weights", status: "done", duration_s: 14 }),
        step({ key: "COMPILE", label: "Compile kernels", status: "running", elapsed_s: 30, usual_s: 225 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const rows = screen.getAllByTestId(/^step-row-/);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent("Start engine");
    expect(rows[0]).toHaveTextContent("5.2s");
    expect(rows[1]).toHaveTextContent("Load weights");
    expect(rows[1]).toHaveTextContent("14s");
    expect(rows[2]).toHaveTextContent("Compile kernels");
    expect(rows[2]).toHaveTextContent("30s");
    expect(rows[2]).toHaveTextContent("usually ~3m45s");
  });

  it("renders steps the engine never reports gracefully (llama.cpp has no compile/graph)", () => {
    // llama.cpp only has start, weights, warmup.
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "done", duration_s: 1 }),
        step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 10 }),
        step({ key: "WARMUP", label: "Warm-up request", status: "pending" }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const rows = screen.getAllByTestId(/^step-row-/);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent("Start engine");
    expect(rows[1]).toHaveTextContent("Load weights");
    expect(rows[2]).toHaveTextContent("Warm-up request");
  });
});

describe("LoadChecklist — exact bar from count/total", () => {
  it("renders an exact bar from count/total for the weights step", () => {
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "done", duration_s: 5 }),
        step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 5, count: 3, total: 8 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const bar = screen.getByTestId("step-bar-WEIGHTS");
    // 3/8 = 37.5 → rounded to 38
    expect(bar).toHaveAttribute("aria-valuenow", "38");
  });

  it("renders an exact bar for graphs capture", () => {
    const s = snapshot({
      steps: [
        step({ key: "GRAPHS", label: "Capture CUDA graphs", status: "running", elapsed_s: 1, count: 48, total: 48 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const bar = screen.getByTestId("step-bar-GRAPHS");
    expect(bar).toHaveAttribute("aria-valuenow", "100");
  });
});

describe("LoadChecklist — usual-time bar never reaches 100% while running", () => {
  it("caps the usual-time bar at 95% when elapsed exceeds usual", () => {
    // A log-silent step (compile) with usual 120s but already 200s elapsed.
    const s = snapshot({
      steps: [
        step({ key: "COMPILE", label: "Compile kernels", status: "running", elapsed_s: 200, usual_s: 120 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const bar = screen.getByTestId("step-bar-COMPILE");
    expect(bar).toHaveAttribute("aria-valuenow", "95");
  });

  it("shows the usual-time bar at 100% only when the step is done", () => {
    const s = snapshot({
      steps: [
        step({ key: "COMPILE", label: "Compile kernels", status: "done", duration_s: 130, usual_s: 120 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const bar = screen.getByTestId("step-bar-COMPILE");
    expect(bar).toHaveAttribute("aria-valuenow", "100");
  });
});

describe("LoadChecklist — stalled state", () => {
  it("shows amber pill, give-up time and force-unload", () => {
    const s = snapshot({
      state: "stalled",
      stalled_for_s: 600,
      give_up_at: "2026-10-18T12:34:56Z",
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "stalled", elapsed_s: 600, count: 3, total: 8 }),
      ],
    });
    const onForceUnload = vi.fn();
    render(<LoadChecklist progress={s} onForceUnload={onForceUnload} />);
    const pill = screen.getByTestId("load-status-pill");
    expect(pill).toHaveAttribute("data-state", "stalled");
    expect(pill).toHaveTextContent(/no progress for 10m/);
    // Give-up time is rendered nearby.
    expect(screen.getByText(/give up at/)).toBeInTheDocument();
    // Force-unload action present and clickable.
    const btn = screen.getByTestId("pill-force-unload");
    expect(btn).not.toBeDisabled();
  });

  it("shows a green pulsing pill while progressing", () => {
    const s = snapshot({ state: "progressing" });
    render(<LoadChecklist progress={s} />);
    const pill = screen.getByTestId("load-status-pill");
    expect(pill).toHaveAttribute("data-state", "progressing");
    expect(pill).toHaveTextContent(/Loading · step \d+ of \d+ · alive/);
  });

  it("stalled force-unload button has no red/destructive styling", () => {
    // The no-red-while-loading rule: the stalled pill's "Force unload now"
    // button must use an outline variant with an amber accent, never the
    // destructive (red) variant.
    const s = snapshot({
      state: "stalled",
      stalled_for_s: 600,
      give_up_at: "2026-10-18T12:34:56Z",
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "stalled", elapsed_s: 600 }),
      ],
    });
    render(<LoadChecklist progress={s} onForceUnload={vi.fn()} />);
    const btn = screen.getByTestId("pill-force-unload");
    // No red/destructive class from the button's variant mapping.
    expect(btn.className).not.toContain("bg-red-600");
    expect(btn.className).not.toContain("bg-red-500");
    expect(btn.className).not.toContain("hover:bg-red-500");
    // Should have the outline variant and an amber accent.
    expect(btn.className).toContain("border");
    expect(btn.className).toContain("amber");
  });
});

describe("LoadChecklist — give_up state", () => {
  it("shows an amber 'Giving up — stopping engine' pill, not green", () => {
    const s = snapshot({
      state: "give_up",
      stalled_for_s: 1200,
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "stalled", elapsed_s: 1200, count: 3, total: 8 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const pill = screen.getByTestId("load-status-pill");
    expect(pill).toHaveAttribute("data-state", "give_up");
    expect(pill).toHaveTextContent("Giving up — stopping engine");
    // Amber (warning variant), not green (success).
    expect(pill.className).toContain("amber");
    expect(pill.className).not.toContain("emerald");
  });

  it("does not offer Force unload for give_up — the warden is already stopping", () => {
    const s = snapshot({ state: "give_up", stalled_for_s: 1200 });
    render(<LoadChecklist progress={s} onForceUnload={vi.fn()} />);
    expect(screen.queryByTestId("pill-force-unload")).toBeNull();
    // View log is also not offered (no onViewLog handler passed).
    expect(screen.queryByTestId("pill-view-log")).toBeNull();
  });

  it("greys vitals and shows the hint block for give_up (stalled-like)", () => {
    const s = snapshot({
      state: "give_up",
      stalled_for_s: 1200,
      reason: "Stalled past the give-up deadline",
      hint: "The engine is being stopped.",
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "stalled", elapsed_s: 1200 }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.getByTestId("stalled-hint")).toBeInTheDocument();
    expect(screen.getByTestId("stalled-hint")).toHaveTextContent(
      "Stalled past the give-up deadline",
    );
  });
});

describe("LoadChecklist — no overall bar or ETA without history", () => {
  it("does not render the overall bar or ETA when usual_total_s is null", () => {
    const s = snapshot({ usual_total_s: null, ready_eta: null });
    render(<LoadChecklist progress={s} />);
    expect(screen.queryByTestId("overall-bar")).toBeNull();
    expect(screen.queryByTestId("ready-eta")).toBeNull();
  });

  it("renders the overall bar and ETA when usual_total_s is known", () => {
    const s = snapshot({
      usual_total_s: 360,
      ready_eta: "2026-10-18T12:34:56Z",
      elapsed_s: 120,
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.getByTestId("overall-bar")).toBeInTheDocument();
    expect(screen.getByTestId("ready-eta")).toHaveTextContent(/ready ≈/);
  });

  it("caps the overall bar at 95% while loading", () => {
    const s = snapshot({
      usual_total_s: 100,
      elapsed_s: 500, // way past usual
    });
    render(<LoadChecklist progress={s} />);
    const bar = screen.getByTestId("overall-bar").querySelector('[role="progressbar"]')!
      .parentElement!.querySelector('[role="progressbar"]');
    // The inner div has the width; check the progressbar's aria-valuenow.
    const pb = screen.getByTestId("overall-bar").querySelector('[role="progressbar"]');
    expect(pb).toHaveAttribute("aria-valuenow", "95");
  });
});

describe("LoadChecklist — vitals under the running step", () => {
  it("renders vitals tiles under the running step", () => {
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "done", duration_s: 5 }),
        step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 5, count: 3, total: 8 }),
        step({ key: "COMPILE", label: "Compile kernels", status: "pending" }),
      ],
      vitals: vitals({
        series: [
          { t_s: 0, cpu_cores: null, gpu_mem_mib: 1000 },
          { t_s: 5, cpu_cores: 1.2, gpu_mem_mib: 2000 },
        ],
        last_log_age_s: 5,
        last_log_line: "Loading safetensors",
        log_quiet_normal: false,
      }),
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.getByTestId("vitals-tiles")).toBeInTheDocument();
    expect(screen.getByTestId("vital-cpu-headline")).toHaveTextContent("1.2 cores busy");
    expect(screen.getByTestId("vital-gpu-headline")).toHaveTextContent(/GiB/);
    expect(screen.getByTestId("vital-log-age")).toHaveTextContent("5s ago");
  });

  it("shows 'normal here' when log_quiet_normal is true", () => {
    const s = snapshot({
      vitals: vitals({ log_quiet_normal: true, last_log_age_s: 120 }),
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.getByTestId("vital-log-quiet-normal")).toHaveTextContent("normal here");
  });
});

describe("LoadChecklist — TUNE step", () => {
  it("renders TUNE row with its bar and detail", () => {
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "done", duration_s: 5 }),
        step({ key: "WEIGHTS", label: "Load weights", status: "done", duration_s: 14 }),
        step({ key: "COMPILE", label: "Compile kernels", status: "done", duration_s: 52 }),
        step({ key: "KV", label: "Size KV cache", status: "done", duration_s: 54 }),
        step({
          key: "TUNE",
          label: "Tune kernels",
          status: "running",
          elapsed_s: 200,
          count: 6,
          total: 22,
          detail: "trtllm_fp4_block_scale_moe · op 2",
        }),
        step({ key: "GRAPHS", label: "Capture CUDA graphs", status: "pending" }),
        step({ key: "WARMUP", label: "Warm-up request", status: "pending" }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    const row = screen.getByTestId("step-row-TUNE");
    expect(row).toHaveTextContent("Tune kernels");
    expect(row).toHaveTextContent("trtllm_fp4_block_scale_moe · op 2 · 6/22");
    // The bar renders with the exact (count/total) progress.
    expect(screen.getByTestId("step-bar-TUNE")).toBeInTheDocument();
  });

  it("does not render a skipped step", () => {
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "done", duration_s: 5 }),
        step({ key: "WEIGHTS", label: "Load weights", status: "done", duration_s: 14 }),
        step({ key: "COMPILE", label: "Compile kernels", status: "done", duration_s: 52 }),
        step({ key: "KV", label: "Size KV cache", status: "done", duration_s: 54 }),
        step({ key: "TUNE", label: "Tune kernels", status: "skipped" }),
        step({
          key: "GRAPHS",
          label: "Capture CUDA graphs",
          status: "running",
          elapsed_s: 10,
          count: 5,
          total: 83,
        }),
        step({ key: "WARMUP", label: "Warm-up request", status: "pending" }),
      ],
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.queryByTestId("step-row-TUNE")).toBeNull();
    expect(screen.getByTestId("step-row-GRAPHS")).toBeInTheDocument();
  });
});

describe("LoadChecklist — vitals charts", () => {
  it("renders charts with series data", () => {
    const s = snapshot({
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 5 }),
      ],
      vitals: vitals({
        series: [
          { t_s: 0, cpu_cores: null, gpu_mem_mib: 1024 },
          { t_s: 5, cpu_cores: 2.5, gpu_mem_mib: 4096 },
          { t_s: 10, cpu_cores: 3.0, gpu_mem_mib: 8192 },
        ],
        gpu_mem_total_mib: 16384,
      }),
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.getByTestId("vital-cpu")).toBeInTheDocument();
    expect(screen.getByTestId("vital-gpu")).toBeInTheDocument();
    expect(screen.getByTestId("vital-cpu-headline")).toHaveTextContent("3.0 cores busy");
    expect(screen.getByTestId("vital-gpu-headline")).toHaveTextContent(/GiB/);
  });

  it("shows 'Not available' only when the driver cannot report GPU memory", () => {
    const s = snapshot({
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 5 }),
      ],
      vitals: vitals({
        gpu_mem_supported: false,
        series: [
          { t_s: 0, cpu_cores: null, gpu_mem_mib: null },
          { t_s: 5, cpu_cores: 1.0, gpu_mem_mib: null },
        ],
      }),
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.getByTestId("vital-gpu-unavailable")).toHaveTextContent(
      "Not available with the docker driver",
    );
  });

  it("waits, not 'Not available', before the engine claims GPU memory", () => {
    const s = snapshot({
      steps: [
        step({ key: "START", label: "Start engine", status: "running", elapsed_s: 30 }),
      ],
      vitals: vitals({
        gpu_mem_supported: true,
        series: [
          { t_s: 0, cpu_cores: null, gpu_mem_mib: null },
          { t_s: 5, cpu_cores: 1.0, gpu_mem_mib: null },
        ],
      }),
    });
    render(<LoadChecklist progress={s} />);
    expect(screen.queryByTestId("vital-gpu-unavailable")).toBeNull();
    expect(screen.getByTestId("vital-gpu-waiting")).toHaveTextContent(
      "Waiting for the engine to claim GPU memory",
    );
  });

  it("does not produce NaN with an empty series", () => {
    const s = snapshot({
      steps: [
        step({ key: "WEIGHTS", label: "Load weights", status: "running", elapsed_s: 5 }),
      ],
      vitals: vitals({ series: [] }),
    });
    render(<LoadChecklist progress={s} />);
    // Should not crash or produce NaN in headlines.
    expect(screen.getByTestId("vital-cpu-headline")).toHaveTextContent("—");
    expect(screen.getByTestId("vital-gpu-headline")).toHaveTextContent("—");
  });
});
