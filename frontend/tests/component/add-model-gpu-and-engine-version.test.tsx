// The Add-model dialog's GPU and engine-version controls.
//
// Operator's ask, verbatim: "when adding a model we need to let user 1. select
// GPU(s) this model will work on (even if those GPUs currently taken with other
// model — we need to warn user about this) 2. engine (and engine version), name
// of the model, etc."
//
// Engine selection, served-model name and the .gguf pre-selection already
// shipped with sub-project C; what did not is (a) any awareness that a GPU is
// occupied, (b) any awareness of `allowed_gpu_indices`, and (c) an engine
// VERSION control. This file covers those three.
//
// The version control is the delicate one. `supports_version_pin` is the engine
// fact and `version_pin_available` is what THIS deployment can honour; they
// disagree on the deployment this was reported from. Gating on the first would
// light up a selector that silently does nothing — #177 verbatim.
//
// #215 (slice b) sharpens the deployment fact: when the RUNNING DRIVER cannot
// honour a pin (`supports_version_select` false — the backends response's
// `driver: "subprocess"`, which this dialog already reads), the wizard no
// longer offers a pin at all. A typed pin on such a deployment is persisted and
// then refused at load with EnginePinUnsupported, and no form action would
// unlock one, so the selector is hidden and one line says why. When the driver
// CAN honour a pin, the per-backend obstacle (llama.cpp) keeps the old
// disabled-and-explained state with the server's sentence.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { AddModelModal } from "@/components/models/add-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const GIB = 1024 * 1024 * 1024;

const SAFETENSORS_REPO = {
  files: [
    {
      filename: "model.safetensors",
      size: 8 * GIB,
      kind: "safetensors_single",
      quant: "fp16",
      params: 7_000_000_000,
    },
    { filename: "config.json", size: 1024, kind: "config", quant: null, params: null },
  ],
  config: { architectures: ["LlamaForCausalLM"] },
  repo: { id: "meta-llama/Llama-3-8B" },
  errors: [],
};

const FIT = {
  verdict: "green",
  breakdown: {
    total_vram: 24 * GIB,
    weights_budget: 20 * GIB,
    kv_reserve: 1 * GIB,
    file_size: 8 * GIB,
    ratio: 0.4,
    dtype_bytes: 2,
    max_model_len_used: 4096,
  },
  recommended_max_model_len: null,
  warnings: [],
};

// Both cards busy, mirroring the reported deployment: llama-3.1-8b on GPU
// index 1 under vLLM, qwen3.8-27b on index 0 under llama.cpp.
const TWO_BUSY_GPUS = [
  {
    index: 0,
    name: "NVIDIA RTX A4000",
    memory_total_mib: 16376,
    memory_used_mib: 12007,
    utilization_pct: 0,
    holders: [
      {
        pid: 1627396,
        memory_mib: 11998,
        process: "llama-server",
        kind: "model",
        model_id: "m-qwen",
        label: "qwen3.8-27b",
      },
    ],
  },
  {
    index: 1,
    name: "Quadro RTX 5000",
    memory_total_mib: 16384,
    memory_used_mib: 14476,
    utilization_pct: 0,
    holders: [
      {
        pid: 1627846,
        memory_mib: 14472,
        process: "VLLM::EngineCore",
        kind: "model",
        model_id: "m-llama",
        label: "llama-3.1-8b",
      },
    ],
  },
];

const FREE_GPUS = [
  {
    index: 0,
    name: "RTX 4090",
    memory_total_mib: 24576,
    memory_used_mib: 0,
    utilization_pct: 0,
    holders: [],
  },
  {
    index: 1,
    name: "RTX 4090",
    memory_total_mib: 24576,
    memory_used_mib: 0,
    utilization_pct: 0,
    holders: [],
  },
];

// `bonus` as deployed: the subprocess driver cannot swap the engine image, so
// neither backend can honour a pin, and the route says why.
const BACKENDS_NO_PIN = {
  default: "vllm",
  driver: "subprocess",
  engine_version: "0.26.0",
  backends: [
    {
      name: "llamacpp",
      display_name: "llama.cpp",
      version: "b10731",
      supports_version_pin: true,
      version_pin_available: false,
      version_pin_reason:
        "This deployment runs the in-container engine driver: the engine version is fixed by the warden image, so a pin is refused rather than applied. Version selection requires the docker engine driver (VW_ENGINE_DRIVER=docker).",
    },
    {
      name: "vllm",
      display_name: "vLLM",
      version: "0.26.0",
      supports_version_pin: true,
      version_pin_available: false,
      version_pin_reason:
        "This deployment runs the in-container engine driver: the engine version is fixed by the warden image, so a pin is refused rather than applied. Version selection requires the docker engine driver (VW_ENGINE_DRIVER=docker).",
    },
  ],
};

// A docker-driver deployment: vLLM can be pinned, llama.cpp still cannot.
const BACKENDS_VLLM_PIN = {
  ...BACKENDS_NO_PIN,
  driver: "docker",
  backends: [
    {
      ...BACKENDS_NO_PIN.backends[0],
      version_pin_reason:
        "llama.cpp cannot be version-pinned on any driver: its binary is compiled into the warden image and there is no llama.cpp image catalogue to pin against.",
    },
    {
      ...BACKENDS_NO_PIN.backends[1],
      version_pin_available: true,
      version_pin_reason: null,
    },
  ],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

let posted: Record<string, unknown> | null = null;

function installFetchStub(opts: {
  gpus: unknown[];
  allowedIndices?: number[] | null;
  backends?: unknown;
}) {
  posted = null;
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") return json({ access_token: "t" });
    if (url === "/api/csrf") return json({ csrf: "c" });
    if (url.startsWith("/api/models/discover")) return json(SAFETENSORS_REPO);
    if (url === "/api/models/fit-preview") return json(FIT);
    if (url === "/api/models/templates") return json([]);
    if (url === "/api/system/backends") return json(opts.backends ?? BACKENDS_NO_PIN);
    if (url === "/api/system/gpus") {
      return json({
        probed_at: "x",
        probe_error: null,
        gpus: opts.gpus,
        allowed_indices:
          opts.allowedIndices === undefined ? null : opts.allowedIndices,
      });
    }
    if (url === "/api/models" && init?.method === "POST") {
      posted = init?.body ? JSON.parse(init.body as string) : {};
      return json({ id: "m1", served_model_name: "x" }, 201);
    }
    if (/^\/api\/models\/.+\/pull$/.test(url)) return new Response(null, { status: 202 });
    throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
  });
  vi.stubGlobal("fetch", mock);
}

async function openDialog(opts: Parameters<typeof installFetchStub>[0]) {
  installFetchStub(opts);
  // Fresh SWR cache per test: `/api/system/backends` differs between cases and
  // the global cache would otherwise serve the first test's answer to the rest.
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <AddModelModal open onClose={() => {}} />
    </SWRConfig>,
  );
  fireEvent.change(screen.getByLabelText(/hf repo/i), {
    target: { value: "meta-llama/Llama-3-8B" },
  });
  fireEvent.click(screen.getByRole("button", { name: /discover/i }));
  await screen.findByTestId("file-table");
}

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("add-model — GPU occupancy", () => {
  it("names the model already serving from a GPU", async () => {
    await openDialog({ gpus: TWO_BUSY_GPUS });
    await waitFor(() =>
      expect(screen.getByTestId("gpu-holder-0")).toHaveTextContent("qwen3.8-27b"),
    );
    expect(screen.getByTestId("gpu-holder-1")).toHaveTextContent("llama-3.1-8b");
  });

  it("still lets the operator pick an occupied GPU, and warns", async () => {
    await openDialog({ gpus: TWO_BUSY_GPUS });
    // The default selection already lands on an occupied card here — every
    // card on this box is busy — so the warning must be up without any click.
    await waitFor(() =>
      expect(screen.getByTestId("gpu-occupied-warning")).toBeInTheDocument(),
    );
    const box = await screen.findByLabelText(/#1/);
    expect(box).not.toBeDisabled();
    fireEvent.click(box);
    expect(screen.getByTestId("gpu-occupied-warning")).toHaveTextContent(
      "llama-3.1-8b",
    );
  });

  it("says nothing about occupancy when the cards are free", async () => {
    await openDialog({ gpus: FREE_GPUS });
    await waitFor(() => expect(screen.getByTestId("gpu-list")).toBeInTheDocument());
    expect(screen.queryByTestId("gpu-occupied-warning")).toBeNull();
  });
});

describe("add-model — allowed_gpu_indices", () => {
  it("does not offer a GPU the server would reject", async () => {
    await openDialog({ gpus: FREE_GPUS, allowedIndices: [1] });
    await waitFor(() => expect(screen.getByLabelText(/#0/)).toBeDisabled());
    expect(screen.getByLabelText(/#1/)).not.toBeDisabled();
  });

  it("defaults the selection to an allowed GPU, not merely the first one", async () => {
    // The old default was `gpus[0].index` unconditionally, which on a
    // deployment whose allowlist excludes GPU 0 pre-fills a form that cannot
    // be submitted.
    await openDialog({ gpus: FREE_GPUS, allowedIndices: [1] });
    await waitFor(() => expect(screen.getByLabelText(/#1/)).toBeChecked());
    expect(screen.getByLabelText(/#0/)).not.toBeChecked();
  });
});

describe("add-model — engine version", () => {
  it("hides the version selector when the driver cannot honour a pin", async () => {
    // supports_version_select is false on this box (the backends response
    // answers `driver: "subprocess"` for the same fact). The wizard offers no
    // pin the deployment would refuse at load: the selector is not rendered,
    // and the one-line help says the engine version is fixed by this build.
    await openDialog({ gpus: FREE_GPUS });
    const help = await screen.findByTestId("engine-version-fixed");
    expect(help).toHaveTextContent(
      "The engine version is fixed by this LM Warden build.",
    );
    expect(screen.queryByTestId("engine-version")).toBeNull();
  });

  it("shows the engine version this deployment actually runs", async () => {
    // Control: with the docker driver the selector renders as before, and the
    // placeholder still names the version this warden image runs.
    await openDialog({ gpus: FREE_GPUS, backends: BACKENDS_VLLM_PIN });
    const field = (await screen.findByTestId("engine-version")) as HTMLInputElement;
    await waitFor(() => expect(field.placeholder).toBe("0.26.0"));
  });

  it("keeps the note a full sentence, with its full stop", async () => {
    // R10: save/8 re-indented this line and dropped the trailing ".", so the
    // operator-facing note no longer ends as a sentence.
    await openDialog({ gpus: FREE_GPUS, backends: BACKENDS_VLLM_PIN });
    const note = await screen.findByTestId("engine-version-note");
    await waitFor(() =>
      expect(note.textContent).toBe(
        "Leave blank to use the version this warden runs (0.26.0).",
      ),
    );
  });

  it("follows the engine as the operator switches backends", async () => {
    // The two backends carry different versions and different reasons; the
    // field is about the SELECTED engine, not about vLLM.
    await openDialog({ gpus: FREE_GPUS, backends: BACKENDS_VLLM_PIN });
    const field = (await screen.findByTestId("engine-version")) as HTMLInputElement;
    await waitFor(() => expect(field).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "llamacpp" },
    });
    await waitFor(() => expect(screen.getByTestId("engine-version")).toBeDisabled());
    expect(screen.getByTestId("engine-version-note").textContent).toMatch(
      /any driver/,
    );
  });

  it("sends the pin when the deployment can honour it", async () => {
    await openDialog({ gpus: FREE_GPUS, backends: BACKENDS_VLLM_PIN });
    fireEvent.click(screen.getByLabelText("select model.safetensors"));
    const field = await screen.findByTestId("engine-version");
    await waitFor(() => expect(field).not.toBeDisabled());
    fireEvent.change(field, { target: { value: "0.25.1" } });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted!.engine_vllm_version).toBe("0.25.1");
    expect(posted!.engine_channel).toBe("cuda-stable");
  });

  it("sends no pin at all when the deployment cannot honour one", async () => {
    // Sending a version the driver will discard is the silent-no-op this whole
    // capability split exists to prevent. #215b: the field is hidden under the
    // subprocess driver, and the submit path still drops the pin.
    await openDialog({ gpus: FREE_GPUS });
    fireEvent.click(screen.getByLabelText("select model.safetensors"));
    await screen.findByTestId("engine-version-fixed");
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted!.engine_vllm_version).toBeUndefined();
    expect(posted!.engine_channel).toBeUndefined();
  });
});
