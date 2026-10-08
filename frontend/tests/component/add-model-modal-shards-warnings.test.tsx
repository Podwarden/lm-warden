// Component tests for S4 shard grouping (#112) + per-file GGUF arch
// warnings (#101) in the Add Model modal.
//
// These complement add-model-modal.test.tsx — that file exercises the
// 4-stage state machine end-to-end; this file pins the two S4-specific UX
// additions: collapsing sharded weights under a single disclosure row and
// surfacing arch warnings inline with the offending file row.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
  within,
} from "@testing-library/react";
import { AddModelModal } from "@/components/models/add-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const GIB = 1024 * 1024 * 1024;

const GPUS_2x24 = [
  {
    index: 0,
    name: "RTX 4090",
    memory_total_mib: 24576,
    memory_used_mib: 0,
    utilization_pct: 0,
  },
  {
    index: 1,
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

interface ShardSetup {
  files: Array<{
    filename: string;
    size: number;
    kind:
      | "safetensors_single"
      | "safetensors_sharded"
      | "gguf"
      | "pytorch_bin"
      | "config"
      | "tokenizer"
      | "other";
    quant: string | null;
    params: number | null;
  }>;
  warnings?: Array<{ type: string; filename: string; arch: string | null }>;
}

function installFetchStub(setup: ShardSetup) {
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") return json({ access_token: "x" });
    if (url === "/api/csrf") return json({ csrf: "test-csrf" });
    if (url.startsWith("/api/models/discover")) {
      return json({
        files: setup.files,
        config: { architectures: ["LlamaForCausalLM"] },
        repo: { id: "meta-llama/Llama-3-8B" },
        errors: [],
        warnings: setup.warnings ?? [],
      });
    }
    if (url === "/api/models/fit-preview") {
      // Mid-band green for every preview body — we don't test fit math here.
      return json({
        verdict: "green",
        breakdown: {
          total_vram: 48 * GIB,
          weights_budget: 20 * GIB,
          kv_reserve: 1 * GIB,
          file_size: 8 * GIB,
          ratio: 0.4,
          dtype_bytes: 2,
          max_model_len_used: 4096,
        },
        recommended_max_model_len: null,
        warnings: [],
      });
    }
    if (url === "/api/system/gpus") {
      return json({
        probed_at: "2026-05-19T12:00:00Z",
        probe_error: null,
        gpus: GPUS_2x24,
      });
    }
    throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("AddModelModal — shard grouping (#112)", () => {
  it("collapses 4 sharded safetensors into one family row with shard count", async () => {
    installFetchStub({
      files: [
        {
          filename: "model-00001-of-00004.safetensors",
          size: 5 * GIB,
          kind: "safetensors_sharded",
          quant: null,
          params: 8_000_000_000,
        },
        {
          filename: "model-00002-of-00004.safetensors",
          size: 5 * GIB,
          kind: "safetensors_sharded",
          quant: null,
          params: null,
        },
        {
          filename: "model-00003-of-00004.safetensors",
          size: 5 * GIB,
          kind: "safetensors_sharded",
          quant: null,
          params: null,
        },
        {
          filename: "model-00004-of-00004.safetensors",
          size: 5 * GIB,
          kind: "safetensors_sharded",
          quant: null,
          params: null,
        },
        {
          filename: "config.json",
          size: 1024,
          kind: "config",
          quant: null,
          params: null,
        },
      ],
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Family toggle is the disclosure-triangle row — exactly one for the
    // 4-shard set. The members are hidden by default.
    const toggle = await screen.findByTestId("shard-family-toggle");
    expect(toggle).toBeInTheDocument();
    // The `data-shard-total` attribute lives on the containing <tr> — find it
    // by climbing to the closest row.
    const familyRow = toggle.closest("tr");
    expect(familyRow?.getAttribute("data-shard-total")).toBe("4");
    // No member rows visible yet (collapsed).
    expect(screen.queryAllByTestId("shard-member-row")).toHaveLength(0);

    // Expanding shows all 4 members.
    fireEvent.click(toggle);
    await waitFor(() => {
      expect(screen.queryAllByTestId("shard-member-row")).toHaveLength(4);
    });

    // Collapsing hides them again — disclosure is bidirectional.
    fireEvent.click(toggle);
    await waitFor(() => {
      expect(screen.queryAllByTestId("shard-member-row")).toHaveLength(0);
    });
  });

  it("does not group a single-shard file (degenerate family)", async () => {
    installFetchStub({
      files: [
        {
          filename: "model-00001-of-00001.safetensors",
          size: 5 * GIB,
          kind: "safetensors_sharded",
          quant: null,
          params: 7_000_000_000,
        },
        {
          filename: "config.json",
          size: 1024,
          kind: "config",
          quant: null,
          params: null,
        },
      ],
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "fake/repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    await screen.findByTestId("file-table");
    // No disclosure-triangle row for a single-shard family (would be pure UI noise).
    expect(screen.queryByTestId("shard-family-toggle")).not.toBeInTheDocument();
  });

  it("does not group loose single-file weights", async () => {
    installFetchStub({
      files: [
        {
          filename: "model.safetensors",
          size: 5 * GIB,
          kind: "safetensors_single",
          quant: null,
          params: 7_000_000_000,
        },
        {
          filename: "config.json",
          size: 1024,
          kind: "config",
          quant: null,
          params: null,
        },
      ],
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "fake/repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    await screen.findByTestId("file-table");
    expect(screen.queryByTestId("shard-family-toggle")).not.toBeInTheDocument();
  });
});

describe("AddModelModal — GGUF arch warnings (#101)", () => {
  it("renders an inline warning row when discovery emits gguf_arch_unsupported", async () => {
    installFetchStub({
      files: [
        {
          filename: "exotic-Q4_K_M.gguf",
          size: 4 * GIB,
          kind: "gguf",
          quant: "Q4_K_M",
          params: 7_000_000_000,
        },
        {
          filename: "config.json",
          size: 1024,
          kind: "config",
          quant: null,
          params: null,
        },
      ],
      warnings: [
        {
          type: "gguf_arch_unsupported",
          filename: "exotic-Q4_K_M.gguf",
          arch: "exoticnewmodel",
        },
      ],
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "fake/repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Discovery pre-selects the backend from the filename, and every weights
    // file here is a .gguf -- so the modal lands on llama.cpp, where this
    // banner is a statement about a loader that is not in use. The warning is
    // about vLLM's GGUF allowlist, so it belongs to the case where the
    // operator deliberately overrides that pre-selection back to vLLM.
    await screen.findByTestId("backend-select");
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });

    const warn = await screen.findByTestId("gguf-arch-warning");
    expect(warn.getAttribute("data-warning-type")).toBe(
      "gguf_arch_unsupported",
    );
    expect(warn.getAttribute("data-arch")).toBe("exoticnewmodel");
  });

  it("renders an unknown-arch warning when discovery emits gguf_arch_unknown", async () => {
    installFetchStub({
      files: [
        {
          filename: "RandomCustom-Q4_K_M.gguf",
          size: 4 * GIB,
          kind: "gguf",
          quant: "Q4_K_M",
          params: 7_000_000_000,
        },
        {
          filename: "config.json",
          size: 1024,
          kind: "config",
          quant: null,
          params: null,
        },
      ],
      warnings: [
        {
          type: "gguf_arch_unknown",
          filename: "RandomCustom-Q4_K_M.gguf",
          arch: null,
        },
      ],
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "fake/repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Discovery pre-selects the backend from the filename, and every weights
    // file here is a .gguf -- so the modal lands on llama.cpp, where this
    // banner is a statement about a loader that is not in use. The warning is
    // about vLLM's GGUF allowlist, so it belongs to the case where the
    // operator deliberately overrides that pre-selection back to vLLM.
    await screen.findByTestId("backend-select");
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });

    const warn = await screen.findByTestId("gguf-arch-warning");
    expect(warn.getAttribute("data-warning-type")).toBe("gguf_arch_unknown");
    // arch attribute is either absent or the empty string when null — DOM
    // string round-trip ambiguity. Either form is fine for the visual layer.
    const archAttr = warn.getAttribute("data-arch");
    expect([null, ""]).toContain(archAttr);
  });

  it("does not render a warning row when discovery emits no warnings", async () => {
    installFetchStub({
      files: [
        {
          filename: "Llama-3-8B-Q4_K_M.gguf",
          size: 4 * GIB,
          kind: "gguf",
          quant: "Q4_K_M",
          params: 8_000_000_000,
        },
        {
          filename: "config.json",
          size: 1024,
          kind: "config",
          quant: null,
          params: null,
        },
      ],
      warnings: [],
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "fake/repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    await screen.findByTestId("file-table");
    expect(screen.queryByTestId("gguf-arch-warning")).not.toBeInTheDocument();
  });
});
