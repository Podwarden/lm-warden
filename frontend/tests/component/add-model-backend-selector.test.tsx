// The operator's actual ask, at the point of decision: "let the users choose
// when loading model -- vllm or llama.cpp".
//
// D6, in three rules that are easy to over-implement:
//   * the persisted default is `vllm`, always, for a client that never chooses;
//   * picking a `.gguf` PRE-SELECTS llama.cpp -- in this wizard only, never
//     persisted logic and never a server-side inference;
//   * the pre-selection is a SUGGESTION. Once the operator has answered,
//     changing the weights file must not silently undo "serve this GGUF on
//     vLLM for the throughput". A GGUF vLLM can also serve stays available to
//     vLLM.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { AddModelModal } from "@/components/models/add-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const GIB = 1024 * 1024 * 1024;

const GGUF_REPO = {
  files: [
    {
      filename: "model-IQ3_XXS.gguf",
      size: 10 * GIB,
      kind: "gguf",
      quant: "IQ3_XXS",
      params: 27_000_000_000,
    },
    {
      filename: "model-IQ2_S.gguf",
      size: 9 * GIB,
      kind: "gguf",
      quant: "IQ2_S",
      params: 27_000_000_000,
    },
    {
      filename: "mmproj-BF16.gguf",
      size: 1 * GIB,
      kind: "mmproj",
      quant: null,
      params: null,
    },
    { filename: "README.md", size: 1024, kind: "other", quant: null, params: null },
  ],
  config: null,
  repo: { id: "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF" },
  errors: [],
};

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

const GPUS = [
  {
    index: 0,
    name: "RTX 4090",
    memory_total_mib: 24576,
    memory_used_mib: 0,
    utilization_pct: 0,
  },
];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const FIT = {
  verdict: "green",
  breakdown: {
    total_vram: 24 * GIB,
    weights_budget: 20 * GIB,
    kv_reserve: 1 * GIB,
    file_size: 10 * GIB,
    ratio: 0.45,
    dtype_bytes: 2,
    max_model_len_used: 4096,
  },
  recommended_max_model_len: null,
  warnings: [],
};

let posted: Record<string, unknown> | null = null;
let fitPosts: Record<string, unknown>[] = [];

function installFetchStub(discovery: unknown) {
  posted = null;
  fitPosts = [];
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") return json({ access_token: "t" });
    if (url === "/api/csrf") return json({ csrf: "c" });
    if (url.startsWith("/api/models/discover")) return json(discovery);
    if (url === "/api/models/fit-preview") {
      fitPosts.push(init?.body ? JSON.parse(init.body as string) : {});
      return json(FIT);
    }
    if (url === "/api/models/templates") return json([]);
    if (url === "/api/system/gpus")
      return json({ probed_at: "x", probe_error: null, gpus: GPUS });
    if (url === "/api/models" && init?.method === "POST") {
      posted = init?.body ? JSON.parse(init.body as string) : {};
      return json({ id: "m1", served_model_name: "x" }, 201);
    }
    if (/^\/api\/models\/.+\/pull$/.test(url)) return new Response(null, { status: 202 });
    throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
  });
  vi.stubGlobal("fetch", mock);
}

async function openAt(repo: string, discovery: unknown) {
  installFetchStub(discovery);
  render(<AddModelModal open onClose={() => {}} />);
  fireEvent.change(screen.getByLabelText(/hf repo/i), { target: { value: repo } });
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

describe("add-model backend selector", () => {
  it("defaults to vLLM for a safetensors file", async () => {
    await openAt("meta-llama/Llama-3-8B", SAFETENSORS_REPO);
    fireEvent.click(screen.getByLabelText("select model.safetensors"));
    expect((screen.getByTestId("backend-select") as HTMLSelectElement).value).toBe(
      "vllm",
    );
  });

  it("pre-selects llama.cpp when a .gguf weights file is picked", async () => {
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    expect((screen.getByTestId("backend-select") as HTMLSelectElement).value).toBe(
      "llamacpp",
    );
  });

  it("keeps an explicit choice when the file selection changes afterwards", async () => {
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });
    fireEvent.click(screen.getByLabelText("select model-IQ2_S.gguf"));
    expect((screen.getByTestId("backend-select") as HTMLSelectElement).value).toBe(
      "vllm",
    );
  });

  it("offers only mmproj-kind files as the projector", async () => {
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    const select = screen.getByTestId("mmproj-select") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.textContent)).toEqual([
      "(none)",
      "mmproj-BF16.gguf",
    ]);
  });

  it("hides the projector picker for the vLLM backend", async () => {
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });
    expect(screen.queryByTestId("mmproj-select")).toBeNull();
  });

  it("sends backend and mmproj_filename in the create payload", async () => {
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    fireEvent.change(screen.getByTestId("mmproj-select"), {
      target: { value: "mmproj-BF16.gguf" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).toMatchObject({
      backend: "llamacpp",
      filename: "model-IQ3_XXS.gguf",
      mmproj_filename: "mmproj-BF16.gguf",
    });
  });

  it("omits mmproj_filename when none is chosen", async () => {
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).not.toHaveProperty("mmproj_filename");
    expect(posted).toMatchObject({ backend: "llamacpp" });
  });

  it("shows the vLLM GGUF architecture warning only for the vLLM backend", async () => {
    // add-model-modal.tsx's two GGUF banners say "vLLM-known GGUF allowlist"
    // and "verify vLLM supports this GGUF". Both are statements about vLLM's
    // loader, and both are false and confusing next to a llama.cpp selection.
    // Sub-project E replaces them with a real per-backend verdict strip; C's
    // job is only to stop showing the wrong one.
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    expect(screen.queryByTestId("gguf-warn")).toBeNull();
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });
    expect(screen.getByTestId("gguf-warn")).toBeInTheDocument();
  });

  it("hides the per-file GGUF arch warning for llama.cpp too", async () => {
    // The sibling test above covers `gguf-warn`. This covers the OTHER banner
    // that test's comment already named -- `gguf-arch-warning`, whose text is
    // "not in the vLLM-known GGUF allowlist". It went ungated because the
    // GGUF_REPO fixture carries no `warnings`, so the banner never rendered
    // in a llama.cpp test and nothing could catch it.
    //
    // This repo is the live case: ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF serves
    // on llama.cpp today, and its arch is outside KNOWN_GGUF_ARCHES -- so the
    // operator picking it for llama.cpp was told the format llama.cpp is
    // built around was unsupported.
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", {
      ...GGUF_REPO,
      warnings: [
        {
          type: "gguf_arch_unsupported",
          filename: "model-IQ3_XXS.gguf",
          arch: "qwen3_gsq",
        },
      ],
    });
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    expect(screen.queryByTestId("gguf-arch-warning")).toBeNull();

    // It is a true statement about vLLM's loader, so it must still show there.
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });
    const warn = screen.getByTestId("gguf-arch-warning");
    expect(warn).toBeInTheDocument();

    // It used to link to /docs/operating.md#supported-gguf-architectures.
    // Nothing serves /docs: the Caddyfile has no `handle /docs*` and the
    // Dockerfile never copies docs/ into the image, so that href 404'd in
    // every deployment there has ever been. The banner now carries the
    // remedy itself instead of pointing at a page that does not exist.
    expect(warn.querySelector('a[href^="/docs/"]')).toBeNull();
    expect(
      screen.getByTestId("gguf-arch-warning-remedy"),
    ).toBeInTheDocument();
  });

  it("tells the fit preview which engine the verdict is about", async () => {
    // Without this the server defaults to vLLM and applies
    // gpu_memory_utilization to a backend that has no such flag -- which is
    // what reported "won't fit" for an 11.28 GiB GGUF on a 16 GiB card.
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    await waitFor(() => expect(fitPosts.length).toBeGreaterThan(0));
    expect(fitPosts[fitPosts.length - 1]).toMatchObject({ backend: "llamacpp" });
  });

  it("re-asks for a verdict when the engine changes", async () => {
    // `fitByFilename` is keyed by filename alone, so a cached verdict would
    // otherwise survive the switch and describe the previous engine -- a
    // wrong number rendered with full confidence.
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-IQ3_XXS.gguf"));
    await waitFor(() => expect(fitPosts.length).toBeGreaterThan(0));
    const before = fitPosts.length;

    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });
    await waitFor(() => expect(fitPosts.length).toBeGreaterThan(before));
    expect(fitPosts[fitPosts.length - 1]).toMatchObject({ backend: "vllm" });
  });

  it("does not offer the projector as a weights file", async () => {
    // A projector is a GGUF, but llama.cpp loading it as -m finds no language
    // model and fails in a way that reads like a corrupt download.
    await openAt("ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF", GGUF_REPO);
    const radio = screen.getByLabelText(
      "select mmproj-BF16.gguf",
    ) as HTMLInputElement;
    expect(radio.disabled).toBe(true);
  });
});
