// Extensions to the model-card tests for load-progress (Task 8).
//
// Covers the brief's failing-test list for the card:
//   - loading model never shows last_error (shows compact bar instead)
//   - failed model shows red last_error
//
// Also covers that the compact bar is shown for loading and not for other
// statuses, and that the card accepts an optional loadProgress prop.

import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { ModelCard, type ModelRow } from "@/components/models/model-card";
import type { LoadProgressOut } from "@/lib/load-progress";

function row(overrides: Partial<ModelRow> = {}): ModelRow {
  return {
    id: "m1",
    served_model_name: "qwen",
    hf_repo: "org/repo",
    hf_revision: "main",
    gpu_indices: [0],
    tensor_parallel_size: 1,
    status: "loaded",
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
    ...overrides,
  };
}

function progress(overrides: Partial<LoadProgressOut> = {}): LoadProgressOut {
  return {
    model_id: "m1",
    backend: "vllm",
    layout: null,
    elapsed_s: 30,
    state: "progressing",
    stalled_for_s: 0,
    give_up_at: null,
    reason: null,
    hint: null,
    usual_total_s: null,
    ready_eta: null,
    current_step: "WEIGHTS",
    steps: [
      { key: "START", label: "Start engine", status: "done", duration_s: 5, elapsed_s: null, count: null, total: null, usual_s: null },
      { key: "WEIGHTS", label: "Load weights", status: "running", duration_s: null, elapsed_s: 25, count: 3, total: 8, usual_s: null },
    ],
    vitals: { series: [], last_log_age_s: null, last_log_line: null, log_quiet_normal: false, gpu_mem_total_mib: null, gpu_mem_supported: true },
    ...overrides,
  };
}

afterEach(() => cleanup());

describe("model card — load-progress compact bar", () => {
  it("shows the compact bar while loading", () => {
    render(<ModelCard model={row({ status: "loading" })} loadProgress={progress()} />);
    expect(screen.getByTestId("load-progress-compact")).toBeInTheDocument();
  });

  it("does not show the compact bar when not loading", () => {
    render(<ModelCard model={row({ status: "loaded" })} loadProgress={progress()} />);
    expect(screen.queryByTestId("load-progress-compact")).toBeNull();
  });

  it("renders the compact bar with undefined progress while loading", () => {
    // The list page passes undefined before the first poll — the compact bar
    // renders nothing, which is fine (the status badge still shows "loading").
    render(<ModelCard model={row({ status: "loading" })} />);
    expect(screen.queryByTestId("load-progress-compact")).toBeNull();
  });
});

describe("model card — last_error gating", () => {
  it("loading model never shows last_error", () => {
    render(
      <ModelCard
        model={row({ status: "loading", last_error: "old error" })}
        loadProgress={progress()}
      />,
    );
    expect(screen.queryByText(/Error:/)).toBeNull();
  });

  it("failed model shows red last_error", () => {
    render(
      <ModelCard model={row({ status: "failed", last_error: "engine crashed" })} />,
    );
    expect(screen.getByText(/Error:/)).toBeInTheDocument();
    expect(screen.getByText(/engine crashed/)).toBeInTheDocument();
  });

  it("loaded model with a stale last_error does not show it", () => {
    // The old code showed last_error for every status.  Now only failed shows it.
    render(
      <ModelCard model={row({ status: "loaded", last_error: "old error" })} />,
    );
    expect(screen.queryByText(/Error:/)).toBeNull();
  });

  it("pulled model with a stale last_error does not show it", () => {
    render(
      <ModelCard model={row({ status: "pulled", last_error: "old error" })} />,
    );
    expect(screen.queryByText(/Error:/)).toBeNull();
  });
});
