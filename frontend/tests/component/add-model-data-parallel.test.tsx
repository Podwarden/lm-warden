// #286 task 5: the Add-model wizard's "Data-parallel replicas" control.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import { AddModelModal } from "@/components/models/add-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const GIB = 1024 * 1024 * 1024;

const REPO = {
  files: [
    { filename: "model.safetensors", size: 8 * GIB, kind: "safetensors_single", quant: "fp16", params: 7_000_000_000 },
    { filename: "config.json", size: 1024, kind: "config", quant: null, params: null },
  ],
  config: { architectures: ["LlamaForCausalLM"] },
  repo: { id: "org/qwen" },
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

const SEVEN_GPUS = [0, 1, 2, 3, 4, 5, 6].map((index) => ({
  index,
  name: "RTX 4090",
  memory_total_mib: 24576,
  memory_used_mib: 0,
  utilization_pct: 0,
  holders: [],
}));

const cap = (name: string, extra: Record<string, unknown>) => ({
  name,
  display_name: name,
  version: "1",
  supports_version_pin: false,
  version_pin_available: false,
  version_pin_reason: "fixed",
  vram_cap_fraction: null,
  ...extra,
});

const BACKENDS = {
  default: "vllm",
  driver: "subprocess",
  engine_version: "0.26.0",
  backends: [
    cap("llamacpp", { supports_data_parallel: false }),
    cap("vllm", { supports_data_parallel: true }),
  ],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

let posted: Record<string, unknown> | null = null;
let fitBodies: Record<string, unknown>[] = [];

async function openDialog(backends: unknown = BACKENDS, templates: unknown[] = [], pickTemplate = "") {
  posted = null;
  fitBodies = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/auth/refresh") return json({ access_token: "t" });
      if (url === "/api/csrf") return json({ csrf: "c" });
      if (url.startsWith("/api/models/discover")) return json(REPO);
      if (url === "/api/models/fit-preview") {
        fitBodies.push(JSON.parse(init!.body as string));
        return json(FIT);
      }
      if (url === "/api/models/templates") return json(templates);
      if (url === "/api/system/backends") return json(backends);
      if (url === "/api/system/gpus")
        return json({ probed_at: "x", probe_error: null, gpus: SEVEN_GPUS, allowed_indices: null });
      if (url === "/api/models" && init?.method === "POST") {
        posted = JSON.parse(init.body as string);
        return json({ id: "m1", served_model_name: "x" }, 201);
      }
      if (/^\/api\/models\/.+\/pull$/.test(url)) return new Response(null, { status: 202 });
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    }),
  );
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <AddModelModal open onClose={() => {}} />
    </SWRConfig>,
  );
  if (pickTemplate) {
    const select = await screen.findByTestId("template-select");
    await waitFor(() => expect(within(select).getByText(pickTemplate)).toBeInTheDocument());
    fireEvent.change(select, { target: { value: pickTemplate } });
  }
  fireEvent.change(screen.getByLabelText(/hf repo/i), { target: { value: "org/qwen" } });
  fireEvent.click(screen.getByRole("button", { name: /discover/i }));
  await screen.findByTestId("file-table");
  fireEvent.click(screen.getByLabelText("select model.safetensors"));
  await waitFor(() => expect(screen.getByLabelText(/#6/)).toBeInTheDocument());
}

async function selectAllSevenGpus() {
  for (let i = 0; i < 7; i++) {
    const box = screen.getByLabelText(new RegExp(`#${i}`)) as HTMLInputElement;
    if (!box.checked) fireEvent.click(box);
  }
  await waitFor(() => expect(screen.getByLabelText(/#6/)).toBeChecked());
}

const dp = () => screen.getByTestId("data-parallel-size") as HTMLInputElement;
const setName = () => {
  const name = screen.getByLabelText(/served model name/i);
  if ((name as HTMLInputElement).value === "") fireEvent.change(name, { target: { value: "qwen" } });
};

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const tpl = (id: string, n?: number) => ({
  id,
  label: id,
  source: "user",
  hf_repo: "org/qwen",
  hf_revision: "main",
  max_model_len: 4096,
  engine: null,
  ...(n === undefined ? {} : { data_parallel_size: n }),
});
const TPL_DP7 = tpl("dp7", 7);
const TPL_LEGACY = tpl("legacy");

describe("add-model — data-parallel replicas (#286)", () => {
  it("offers the control for vLLM and hides it for llama.cpp", async () => {
    await openDialog();
    expect(dp().value).toBe("1");
    fireEvent.change(screen.getByTestId("backend-select"), { target: { value: "llamacpp" } });
    await waitFor(() => expect(screen.queryByTestId("data-parallel-size")).toBeNull());
  });

  it("hides the control when the backend reports supports_data_parallel=false", async () => {
    await openDialog({
      ...BACKENDS,
      backends: [cap("llamacpp", { supports_data_parallel: false }), cap("vllm", { supports_data_parallel: false })],
    });
    await waitFor(() => expect(screen.queryByTestId("data-parallel-size")).toBeNull());
  });

  it("falls back to vllm=true when the server omits the flag", async () => {
    await openDialog({ ...BACKENDS, backends: [cap("llamacpp", {}), cap("vllm", {})] });
    expect(dp()).toBeInTheDocument();
  });

  it("7 GPUs + 7 replicas posts data_parallel_size=7 and no tensor_parallel_size", async () => {
    await openDialog();
    await selectAllSevenGpus();
    fireEvent.change(dp(), { target: { value: "7" } });
    setName();
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted!.data_parallel_size).toBe(7);
    expect(posted).not.toHaveProperty("tensor_parallel_size");
    expect(posted!.gpu_indices).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it("shows the derived per-replica layout inline", async () => {
    await openDialog();
    await selectAllSevenGpus();
    fireEvent.change(dp(), { target: { value: "7" } });
    expect(screen.getByTestId("dp-derived")).toHaveTextContent(/7 replicas/i);
    expect(screen.getByTestId("dp-derived")).toHaveTextContent(/1 GPU/i);
    expect(screen.queryByTestId("dp-invalid-error")).toBeNull();
  });

  it("7 GPUs + 3 replicas shows the error inline and on submit, and does not POST", async () => {
    await openDialog();
    await selectAllSevenGpus();
    fireEvent.change(dp(), { target: { value: "3" } });
    expect(screen.getByTestId("dp-invalid-error")).toHaveTextContent(/must divide the number of selected GPUs \(7\)/i);
    setName();
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await screen.findAllByText(/Data-parallel replicas must divide the number of selected GPUs \(7\)/);
    expect(posted).toBeNull();
  });

  it("replicas=1 sends no data_parallel_size key", async () => {
    await openDialog();
    setName();
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).not.toHaveProperty("data_parallel_size");
  });

  it("carries data_parallel_size in the fit-preview body and refetches when it changes", async () => {
    await openDialog();
    await selectAllSevenGpus();
    await waitFor(() => expect(fitBodies.length).toBeGreaterThan(0));
    expect(fitBodies.at(-1)).not.toHaveProperty("data_parallel_size");
    fireEvent.change(dp(), { target: { value: "7" } });
    await waitFor(() => expect(fitBodies.at(-1)!.data_parallel_size).toBe(7), { timeout: 2000 });
  });

  it("applying a data-parallel template sets the replicas control and posts it (#286 re-review)", async () => {
    await openDialog(BACKENDS, [TPL_DP7, TPL_LEGACY], "dp7");
    await selectAllSevenGpus();
    expect(dp().value).toBe("7");
    setName();
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted!.template_id).toBe("dp7");
    expect(posted!.data_parallel_size).toBe(7);
  });

  it("a template without data_parallel_size leaves one replica", async () => {
    await openDialog(BACKENDS, [TPL_DP7, TPL_LEGACY], "legacy");
    expect(dp().value).toBe("1");
  });
});
