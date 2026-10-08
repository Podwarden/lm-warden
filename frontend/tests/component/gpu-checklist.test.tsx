import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";

afterEach(cleanup);

const GPUS: GpuInfo[] = [
  { index: 0, name: "NVIDIA RTX A4000", memory_total_mib: 16376, memory_used_mib: 1024, utilization_pct: 5 },
  { index: 1, name: "NVIDIA RTX A4000", memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 },
];

describe("GpuChecklist", () => {
  it("renders one checkbox per present GPU", () => {
    render(<GpuChecklist gpus={GPUS} selected={[0]} onChange={() => {}} />);
    expect(screen.getByLabelText(/#0/)).toBeChecked();
    expect(screen.getByLabelText(/#1/)).not.toBeChecked();
  });

  it("toggling a GPU emits a sorted number[]", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={GPUS} selected={[1]} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText(/#0/));
    expect(onChange).toHaveBeenCalledWith([0, 1]);
  });

  it("renders a removable ghost row + warning banner for a missing configured index", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={GPUS} selected={[0, 5]} onChange={onChange} />);
    const ghost = screen.getByLabelText(/GPU 5 — not present/);
    expect(ghost).toBeChecked();
    expect(screen.getByRole("alert")).toHaveTextContent(/not present/i);
    fireEvent.click(ghost); // unchecking removes it
    expect(onChange).toHaveBeenCalledWith([0]);
  });

  it("shows an empty-state message when no GPUs are present and none selected", () => {
    render(<GpuChecklist gpus={[]} selected={[]} onChange={() => {}} />);
    expect(screen.getByText(/no gpus detected/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Occupancy.
//
// The operator's requirement, verbatim: a GPU already taken by another model
// must STILL be selectable, and the warning must name the model. Both halves
// matter -- co-locating two small models on one card is a legitimate thing to
// do deliberately, so blocking it would be wrong; doing it by accident is the
// thing worth catching.
// ---------------------------------------------------------------------------

const OCCUPIED: GpuInfo[] = [
  {
    index: 0,
    name: "NVIDIA RTX A4000",
    memory_total_mib: 16376,
    memory_used_mib: 12007,
    utilization_pct: 0,
    holders: [
      {
        pid: 1,
        memory_mib: 11998,
        process: "llama-server",
        kind: "model",
        model_id: "m1",
        label: "qwen3.8-27b",
      },
    ],
  },
  {
    index: 1,
    name: "Quadro RTX 5000",
    memory_total_mib: 16384,
    memory_used_mib: 0,
    utilization_pct: 0,
    holders: [],
  },
];

describe("GpuChecklist — occupied GPUs", () => {
  it("names the occupying model on the row", () => {
    render(<GpuChecklist gpus={OCCUPIED} selected={[1]} onChange={() => {}} />);
    expect(screen.getByTestId("gpu-holder-0")).toHaveTextContent("qwen3.8-27b");
  });

  it("leaves an occupied GPU selectable", () => {
    const onChange = vi.fn();
    render(<GpuChecklist gpus={OCCUPIED} selected={[]} onChange={onChange} />);
    const box = screen.getByLabelText(/#0/);
    expect(box).not.toBeDisabled();
    fireEvent.click(box);
    expect(onChange).toHaveBeenCalledWith([0]);
  });

  it("warns, naming the model, once an occupied GPU is selected", () => {
    render(<GpuChecklist gpus={OCCUPIED} selected={[0]} onChange={() => {}} />);
    const warn = screen.getByTestId("gpu-occupied-warning");
    expect(warn).toHaveTextContent("qwen3.8-27b");
    expect(warn).toHaveTextContent(/GPU 0/);
  });

  it("stays quiet while only free GPUs are selected", () => {
    render(<GpuChecklist gpus={OCCUPIED} selected={[1]} onChange={() => {}} />);
    expect(screen.queryByTestId("gpu-occupied-warning")).toBeNull();
  });

  it("does not report a model as competing with itself", () => {
    // The model settings page renders this checklist for a row that may be
    // LOADED — so that row's own engine is holding the card. "GPU 0 is already
    // serving qwen3.8-27b" while editing qwen3.8-27b is a false alarm, and a
    // warning that cries wolf is worse than none: the operator learns to skip
    // it, including on the occasion it is real.
    render(
      <GpuChecklist
        gpus={OCCUPIED}
        selected={[0]}
        onChange={() => {}}
        excludeModelId="m1"
      />,
    );
    expect(screen.queryByTestId("gpu-occupied-warning")).toBeNull();
    expect(screen.queryByTestId("gpu-holder-0")).toBeNull();
  });

  it("still reports a DIFFERENT model on the same card", () => {
    render(
      <GpuChecklist
        gpus={OCCUPIED}
        selected={[0]}
        onChange={() => {}}
        excludeModelId="some-other-model"
      />,
    );
    expect(screen.getByTestId("gpu-occupied-warning")).toHaveTextContent("qwen3.8-27b");
  });

  it("ignores holders that are not ours", () => {
    // A desktop session or another container on the box is not a model this
    // warden can name, and "in use by (unknown)" is noise, not information.
    const external: GpuInfo[] = [
      {
        ...OCCUPIED[0],
        holders: [
          {
            pid: 9,
            memory_mib: 50,
            process: "Xorg",
            kind: "external",
            model_id: null,
            label: null,
          },
        ],
      },
    ];
    render(<GpuChecklist gpus={external} selected={[0]} onChange={() => {}} />);
    expect(screen.queryByTestId("gpu-occupied-warning")).toBeNull();
    expect(screen.queryByTestId("gpu-holder-0")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// allowed_gpu_indices, from GET /api/system/gpus.
// ---------------------------------------------------------------------------

describe("GpuChecklist — the setup allowlist", () => {
  it("disables a GPU outside the allowlist and explains why", () => {
    render(
      <GpuChecklist gpus={GPUS} selected={[1]} onChange={() => {}} allowedIndices={[1]} />,
    );
    expect(screen.getByLabelText(/#0/)).toBeDisabled();
    expect(screen.getByLabelText(/#1/)).not.toBeDisabled();
    expect(screen.getByTestId("gpu-not-allowed-0")).toHaveTextContent(/not allowed/i);
  });

  it("treats a null allowlist as no restriction, not as none allowed", () => {
    // The distinction is load-bearing: a setup draft written before the key
    // existed has no allowlist, and reading that as "zero GPUs" would lock the
    // operator out of their own box.
    render(
      <GpuChecklist gpus={GPUS} selected={[0]} onChange={() => {}} allowedIndices={null} />,
    );
    expect(screen.getByLabelText(/#0/)).not.toBeDisabled();
    expect(screen.getByLabelText(/#1/)).not.toBeDisabled();
  });

  it("keeps an already-selected out-of-allowlist GPU removable", () => {
    // Stored state can predate a narrowed allowlist. Disabling the checkbox
    // would leave a row the server will reject and no way to clear it.
    const onChange = vi.fn();
    render(
      <GpuChecklist gpus={GPUS} selected={[0]} onChange={onChange} allowedIndices={[1]} />,
    );
    const box = screen.getByLabelText(/#0/);
    expect(box).toBeChecked();
    expect(box).not.toBeDisabled();
    fireEvent.click(box);
    expect(onChange).toHaveBeenCalledWith([]);
  });
});
