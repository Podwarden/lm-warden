// The backend badge, and the rule that makes it useful rather than noisy.
//
// It appears ONLY for a non-default backend. A badge on every card is
// decoration an operator learns to skip; a badge that appears exactly when
// something is unusual is information they will actually read. D6 says NULL
// means vLLM and existing rows are not backfilled, so a pre-C row -- which is
// every row today -- must look exactly as it did before the column existed.

import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { ModelCard, backendBadgeLabel, type ModelRow } from "@/components/models/model-card";

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

afterEach(() => cleanup());

describe("model card backend badge", () => {
  it("shows llama.cpp for a llama.cpp row", () => {
    render(<ModelCard model={row({ backend: "llamacpp" })} />);
    expect(screen.getByTestId("backend-badge").textContent).toBe("llama.cpp");
  });

  it("shows nothing for a vLLM row", () => {
    render(<ModelCard model={row({ backend: "vllm" })} />);
    expect(screen.queryByTestId("backend-badge")).toBeNull();
  });

  it("shows nothing for a NULL backend, which is every pre-C row", () => {
    render(<ModelCard model={row({ backend: null })} />);
    expect(screen.queryByTestId("backend-badge")).toBeNull();
  });

  it("shows nothing when the field is absent entirely", () => {
    render(<ModelCard model={row()} />);
    expect(screen.queryByTestId("backend-badge")).toBeNull();
  });

  it("keeps the status badge alongside it", () => {
    render(<ModelCard model={row({ backend: "llamacpp" })} />);
    expect(screen.getByText("loaded")).toBeInTheDocument();
  });

  it("passes an unknown backend through rather than hiding it", () => {
    // A UI older than its API is the mid-rollout case. Showing the raw name is
    // strictly better than pretending the row is vLLM.
    expect(backendBadgeLabel("sglang")).toBe("sglang");
  });
});

describe("model card GPU line", () => {
  // "GPUs: 1" on a card for a model pinned to GPU index 1 reads as "one GPU".
  // Both readings are plausible, which is what makes it a bug rather than a
  // nit: the reporting operator's box has two cards with one model on each.
  it("labels a single index as an index, not a count", () => {
    render(<ModelCard model={row({ gpu_indices: [1] })} />);
    expect(screen.getByTestId("model-card-gpu-indices").textContent).toBe("1");
    expect(screen.getByText(/GPU index:/)).toBeInTheDocument();
  });

  it("pluralises for a multi-GPU row", () => {
    render(<ModelCard model={row({ gpu_indices: [0, 1] })} />);
    expect(screen.getByTestId("model-card-gpu-indices").textContent).toBe("0, 1");
    expect(screen.getByText(/GPU indices:/)).toBeInTheDocument();
  });
});
