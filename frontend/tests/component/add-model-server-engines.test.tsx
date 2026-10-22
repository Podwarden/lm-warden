// The Add model dialog offers the engines the SERVER reports
// (GET /api/system/backends), not a hard-coded pair. On the Mac the server
// has llamacpp + mlx and default mlx; on Linux vllm + llamacpp, default vllm.
// Behaviour is driven by capabilities, never by the OS.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  cleanup,
  waitFor,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import { AddModelModal } from "@/components/models/add-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const GIB = 1024 * 1024 * 1024;

const cap = (
  name: string,
  display_name: string,
  extra: Record<string, unknown> = {},
) => ({
  name,
  display_name,
  version: null,
  supports_version_pin: false,
  version_pin_available: false,
  version_pin_reason: "fixed",
  vram_cap_fraction: null,
  supports_tensor_parallel: false,
  supports_data_parallel: false,
  needs_local_model_path: false,
  ...extra,
});

const MAC = {
  default: "mlx",
  driver: "subprocess",
  engine_version: null,
  backends: [
    cap("llamacpp", "llama.cpp", { needs_local_model_path: true }),
    cap("mlx", "MLX", { needs_local_model_path: true }),
  ],
};
const LINUX = {
  default: "vllm",
  driver: "subprocess",
  engine_version: null,
  backends: [
    cap("vllm", "vLLM", {
      supports_tensor_parallel: true,
      supports_data_parallel: true,
    }),
    cap("llamacpp", "llama.cpp", { needs_local_model_path: true }),
  ],
};

const SAFETENSORS_REPO = {
  files: [
    {
      filename: "model.safetensors",
      size: 2 * GIB,
      kind: "safetensors_single",
      quant: "4bit",
      params: 3_000_000_000,
    },
    {
      filename: "config.json",
      size: 1024,
      kind: "config",
      quant: null,
      params: null,
    },
  ],
  config: { architectures: ["Qwen2ForCausalLM"] },
  repo: { id: "mlx-community/Qwen2.5-3B-4bit" },
  errors: [],
};
const GGUF_REPO = {
  files: [
    {
      filename: "model-Q4.gguf",
      size: 2 * GIB,
      kind: "gguf",
      quant: "Q4",
      params: 3_000_000_000,
    },
    {
      filename: "model-Q8.gguf",
      size: 3 * GIB,
      kind: "gguf",
      quant: "Q8",
      params: 3_000_000_000,
    },
  ],
  config: null,
  repo: { id: "org/gguf" },
  errors: [],
};
const FIT = {
  verdict: "green",
  breakdown: {
    total_vram: 24 * GIB,
    weights_budget: 20 * GIB,
    kv_reserve: GIB,
    file_size: 2 * GIB,
    ratio: 0.1,
    dtype_bytes: 2,
    max_model_len_used: 4096,
  },
  recommended_max_model_len: null,
  warnings: [],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

let posted: Record<string, unknown> | null = null;

type BackendsMode = "pending" | "reject";

async function openAt(
  backends: unknown,
  discovery: unknown,
  mode?: BackendsMode,
) {
  posted = null;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/auth/refresh") return json({ access_token: "t" });
      if (url === "/api/csrf") return json({ csrf: "c" });
      if (url === "/api/system/backends") {
        if (mode === "pending") return new Promise<Response>(() => {});
        if (mode === "reject") return json({ detail: "boom" }, 500);
        return json(backends);
      }
      if (url.startsWith("/api/models/discover")) return json(discovery);
      if (url === "/api/models/fit-preview") return json(FIT);
      if (url === "/api/models/templates") return json([]);
      if (url === "/api/system/gpus")
        return json({
          probed_at: "x",
          probe_error: null,
          gpus: [
            {
              index: 0,
              name: "GPU",
              memory_total_mib: 24576,
              memory_used_mib: 0,
              utilization_pct: 0,
            },
          ],
        });
      if (url === "/api/models" && init?.method === "POST") {
        posted = JSON.parse(init.body as string);
        return json({ id: "m1", served_model_name: "x" }, 201);
      }
      if (/^\/api\/models\/.+\/pull$/.test(url))
        return new Response(null, { status: 202 });
      throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
    }),
  );
  render(
    <SWRConfig
      value={{
        provider: () => new Map(),
        dedupingInterval: 0,
        shouldRetryOnError: false,
      }}
    >
      <AddModelModal open onClose={() => {}} />
    </SWRConfig>,
  );
  fireEvent.change(screen.getByLabelText(/hf repo/i), {
    target: { value: "org/repo" },
  });
  fireEvent.click(screen.getByRole("button", { name: /discover/i }));
  await screen.findByTestId("file-table");
  // The GPU list loads independently; Add before it lands is refused
  // ("Select at least one GPU"), which on a slow runner raced the click.
  await waitFor(() =>
    expect(
      screen
        .getAllByRole("checkbox")
        .some((c) => (c as HTMLInputElement).checked),
    ).toBe(true),
  );
  if (mode) return;
  // the capability list lands independently of discovery
  await waitFor(() => {
    const sel = screen.getByTestId("backend-select") as HTMLSelectElement;
    expect(sel.options.length).toBe(2);
    expect(Array.from(sel.options).map((o) => o.value)).toEqual(
      (backends as typeof MAC).backends.map((b) => b.name),
    );
  });
}

const select = () => screen.getByTestId("backend-select") as HTMLSelectElement;

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("add model — engines come from the server", () => {
  it("Mac response: MLX is offered and preselected, vLLM is not offered", async () => {
    await openAt(MAC, SAFETENSORS_REPO);
    const values = Array.from(select().options).map((o) => o.value);
    expect(values).toContain("mlx");
    expect(values).not.toContain("vllm");
    expect(select().value).toBe("mlx");
    expect(screen.getByTestId("backend-description").textContent).not.toMatch(
      /tensor-parallel multi-GPU/,
    );
  });

  it("Mac response: picking a .gguf switches to llama.cpp", async () => {
    await openAt(MAC, GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-Q4.gguf"));
    expect(select().value).toBe("llamacpp");
  });

  it("Mac response: submitting sends backend mlx and no filename", async () => {
    await openAt(MAC, SAFETENSORS_REPO);
    fireEvent.click(screen.getByLabelText("select model.safetensors"));
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).toMatchObject({ backend: "mlx", hf_repo: "org/repo" });
    expect(posted).not.toHaveProperty("filename");
  });

  it("Linux response: unchanged — vLLM preselected, sends its filename, TP copy shown", async () => {
    await openAt(LINUX, SAFETENSORS_REPO);
    expect(select().value).toBe("vllm");
    expect(screen.getByTestId("backend-description").textContent).toMatch(
      /tensor-parallel multi-GPU/,
    );
    fireEvent.click(screen.getByLabelText("select model.safetensors"));
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).toMatchObject({
      backend: "vllm",
      filename: "model.safetensors",
    });
  });

  it("Linux response: a .gguf still pre-selects llama.cpp and sends its filename", async () => {
    await openAt(LINUX, GGUF_REPO);
    fireEvent.click(screen.getByLabelText("select model-Q4.gguf"));
    expect(select().value).toBe("llamacpp");
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).toMatchObject({
      backend: "llamacpp",
      filename: "model-Q4.gguf",
    });
  });

  it.each<BackendsMode>(["pending", "reject"])(
    "backends list %s and no explicit choice: backend is omitted from the create body",
    async (mode) => {
      await openAt(LINUX, SAFETENSORS_REPO, mode);
      fireEvent.click(screen.getByLabelText("select model.safetensors"));
      fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
      await waitFor(() => expect(posted).not.toBeNull());
      expect(posted).toMatchObject({ hf_repo: "org/repo" });
      expect(posted).not.toHaveProperty("backend");
    },
  );

  it("llama.cpp + .safetensors still sends the filename", async () => {
    await openAt(MAC, SAFETENSORS_REPO);
    fireEvent.change(select(), { target: { value: "llamacpp" } });
    fireEvent.click(screen.getByLabelText("select model.safetensors"));
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted).toMatchObject({
      backend: "llamacpp",
      filename: "model.safetensors",
    });
  });

  it("engine copy: llama.cpp says single .gguf, MLX says whole repository", async () => {
    await openAt(MAC, SAFETENSORS_REPO);
    expect(screen.getByTestId("backend-description").textContent).toMatch(
      /whole repository/,
    );
    fireEvent.change(select(), { target: { value: "llamacpp" } });
    expect(screen.getByTestId("backend-description").textContent).toMatch(
      /single \.gguf/,
    );
    expect(screen.getByTestId("backend-description").textContent).not.toMatch(
      /whole repository/,
    );
  });
});
