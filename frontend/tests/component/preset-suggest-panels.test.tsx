// Settings redesign T3 (plan §4.4): the preset card and the suggest row,
// extracted from the settings page, plus the shared diff list.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  cleanup,
  act,
  fireEvent,
  within,
  waitFor,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import { PresetStrip } from "@/components/models/settings/preset-strip";
import { SuggestPanel } from "@/components/models/settings/suggest-panel";
import { DiffList, computeDiff } from "@/components/models/settings/diff-list";
import { formatSettingValue } from "@/lib/model-settings";
import { snapshotToDraft, labelFor, type ModelSettings } from "@/lib/model-settings";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const SETTINGS: ModelSettings = {
  id: "abc",
  served_model_name: "llama3-8b",
  hf_repo: "meta-llama/Llama-3-8B",
  hf_revision: "main",
  gpu_indices: [0],
  tensor_parallel_size: 1,
  dtype: null,
  max_model_len: null,
  gpu_memory_utilization: 0.9,
  trust_remote_code: false,
  extra_args: [],
  extra_env: {},
  backend: null,
  n_gpu_layers: null,
  mmproj_filename: null,
  supports_vision: null,
  supports_tools: null,
  supports_reasoning: null,
  status: "pulled",
  pulled_bytes: 0,
  pulled_total: null,
  last_error: null,
};
const DRAFT = snapshotToDraft(SETTINGS);

const PRESETS = {
  presets: [
    {
      id: "a4000-tight-awq",
      name: "A4000 tight (AWQ)",
      description: "Conservative VRAM budget for a single RTX A4000.",
      target_archetype: "1x A4000 16GB",
      settings: { gpu_memory_utilization: 0.78, max_model_len: 8192 },
    },
    {
      id: "h100-single-shot",
      name: "H100 single-shot",
      description: "Single H100.",
      target_archetype: "1x H100 80GB",
      settings: { gpu_memory_utilization: 0.95 },
    },
    {
      id: "dev-tiny",
      name: "Dev (tiny)",
      description: "Smoke profile.",
      target_archetype: "any GPU ≥ 6GB",
      settings: { gpu_memory_utilization: 0.5 },
    },
    {
      id: "moe-balanced",
      name: "MoE balanced (2x A4000)",
      description: "MoE.",
      target_archetype: "2x A4000 16GB",
      settings: { tensor_parallel_size: 2 },
    },
  ],
};

function stubFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url === "/api/presets") return new Response(JSON.stringify(PRESETS));
      if (url === "/api/models/abc/suggest-config") {
        return new Response(
          JSON.stringify({
            gpu_memory_utilization: 0.85,
            max_model_len: 16384,
            kv_cache_dtype: null,
            disclaimer: "Heuristic suggestion.",
          }),
        );
      }
      throw new Error(`unexpected fetch ${url}`);
    }),
  );
}

async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

function wrap(node: React.ReactNode) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      {node}
    </SWRConfig>,
  );
}

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
  stubFetch();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("PresetStrip", () => {
  it("is a 'Start from a preset' card with the explanatory sentence", async () => {
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={() => {}} />);
    await flush();
    expect(
      screen.getByRole("heading", { name: "Start from a preset" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("presets-card").textContent).toMatch(
      /Fills in the form only\. You see the changes before anything is saved\./,
    );
  });

  it("chips show the name and the target hardware on two lines", async () => {
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={() => {}} />);
    await flush();
    const chip = screen.getByTestId("preset-chip-a4000-tight-awq");
    expect(within(chip).getByTestId("preset-chip-name").textContent).toBe(
      "A4000 tight (AWQ)",
    );
    expect(within(chip).getByTestId("preset-chip-archetype").textContent).toBe(
      "1x A4000 16GB",
    );
    expect(chip.getAttribute("title")).toMatch(/Conservative VRAM budget/);
  });

  it("presets-strip holds exactly the four chips, even with the confirm open", async () => {
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={() => {}} />);
    await flush();
    const strip = screen.getByTestId("presets-strip");
    expect(within(strip).getAllByRole("button")).toHaveLength(4);
    fireEvent.click(screen.getByTestId("preset-chip-a4000-tight-awq"));
    expect(screen.getByTestId("preset-confirm")).toBeInTheDocument();
    expect(within(strip).getAllByRole("button")).toHaveLength(4);
  });

  it("moves focus into the confirm when it opens, so Escape works at once", async () => {
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={() => {}} />);
    await flush();
    const chip = screen.getByTestId("preset-chip-dev-tiny");
    fireEvent.click(chip);
    const dialog = screen.getByTestId("preset-confirm");
    await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement));
    fireEvent.keyDown(document.activeElement!, { key: "Escape" });
    expect(screen.queryByTestId("preset-confirm")).toBeNull();
    expect(document.activeElement).toBe(chip);
  });

  it("confirm diff rows show labels and keep data-diff-key", async () => {
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={() => {}} />);
    await flush();
    fireEvent.click(screen.getByTestId("preset-chip-a4000-tight-awq"));
    const rows = within(screen.getByTestId("preset-confirm")).getAllByTestId(
      "preset-diff-row",
    );
    const util = rows.find(
      (r) => r.getAttribute("data-diff-key") === "gpu_memory_utilization",
    )!;
    expect(util.textContent).toMatch(/GPU memory utilization/);
    expect(util.textContent).not.toMatch(/gpu_memory_utilization/);
    expect(util.textContent).toMatch(/0\.9.*→.*0\.78/);
  });

  it("Apply hands over the sparse settings and returns focus to the chip", async () => {
    const onApply = vi.fn();
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={onApply} />);
    await flush();
    const chip = screen.getByTestId("preset-chip-a4000-tight-awq");
    fireEvent.click(chip);
    fireEvent.click(screen.getByTestId("preset-apply"));
    expect(onApply).toHaveBeenCalledWith({
      gpu_memory_utilization: 0.78,
      max_model_len: 8192,
    });
    expect(screen.queryByTestId("preset-confirm")).toBeNull();
    expect(document.activeElement).toBe(chip);
  });

  it("Cancel closes without applying and returns focus to the chip", async () => {
    const onApply = vi.fn();
    wrap(<PresetStrip disabled={false} draft={DRAFT} onApply={onApply} />);
    await flush();
    const chip = screen.getByTestId("preset-chip-dev-tiny");
    fireEvent.click(chip);
    fireEvent.click(screen.getByTestId("preset-cancel"));
    expect(onApply).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(chip);
  });

  it("disables the chips when disabled", async () => {
    wrap(<PresetStrip disabled draft={DRAFT} onApply={() => {}} />);
    await flush();
    for (const b of within(screen.getByTestId("presets-strip")).getAllByRole("button")) {
      expect(b).toBeDisabled();
    }
  });
});

describe("SuggestPanel", () => {
  it("is one row: the sentence and a Suggest values button", async () => {
    wrap(<SuggestPanel modelId="abc" disabled={false} draft={DRAFT} onApply={() => {}} />);
    expect(screen.getByTestId("suggest-panel").textContent).toMatch(
      /Not sure\? Get values sized from the model config and the VRAM on these GPUs\./,
    );
    expect(screen.getByRole("button", { name: "Suggest values" })).toBeInTheDocument();
  });

  it("shows labelled diff rows, applies, and returns focus to the button", async () => {
    const onApply = vi.fn();
    wrap(<SuggestPanel modelId="abc" disabled={false} draft={DRAFT} onApply={onApply} />);
    const btn = screen.getByTestId("suggest-fetch");
    fireEvent.click(btn);
    const result = await screen.findByTestId("suggest-result");
    expect(within(result).getByTestId("suggest-rationale").textContent).toBe(
      "Heuristic suggestion.",
    );
    const keys = within(result)
      .getAllByTestId("suggest-diff-row")
      .map((r) => r.getAttribute("data-diff-key"));
    expect(keys).toEqual(["max_model_len", "gpu_memory_utilization"]);
    expect(result.textContent).toMatch(/Max model length/);
    await waitFor(() => expect(result).toContainElement(document.activeElement as HTMLElement));
    fireEvent.click(within(result).getByTestId("suggest-apply"));
    expect(onApply).toHaveBeenCalledWith({
      gpu_memory_utilization: 0.85,
      max_model_len: 16384,
    });
    expect(screen.queryByTestId("suggest-result")).toBeNull();
    expect(document.activeElement).toBe(btn);
  });
});

describe("DiffList helpers", () => {
  it("computeDiff skips equal and unknown keys, in PATCHABLE_KEYS order", () => {
    expect(
      computeDiff(DRAFT, {
        gpu_memory_utilization: 0.9,
        max_model_len: 4096,
        nonsense: 1,
        hf_revision: "v2",
      }),
    ).toEqual([
      { key: "hf_revision", before: "main", after: "v2" },
      { key: "max_model_len", before: null, after: 4096 },
    ]);
  });

  it("formatSettingValue: one operator wording for the save bar and the preset/suggest diffs", () => {
    expect(formatSettingValue("max_model_len", null)).toBe("auto");
    expect(formatSettingValue("dtype", null)).toBe("default");
    expect(formatSettingValue("supports_vision", true)).toBe("Yes");
    expect(formatSettingValue("supports_tools", null)).toBe("Auto");
    expect(formatSettingValue("trust_remote_code", false)).toBe("off");
    expect(formatSettingValue("gpu_indices", [0, 1])).toBe("0, 1");
    expect(formatSettingValue("extra_args", ["--a", "1"])).toBe("--a 1");
    expect(formatSettingValue("extra_env", { A: "1" })).toBe("A=1");
    expect(formatSettingValue("hf_revision", "")).toBe("");
  });

  it("DiffList words null as the save bar does, not as JSON", () => {
    render(
      <DiffList
        rows={[{ key: "max_model_len", before: null, after: 8192 }]}
        testIdPrefix="x"
      />,
    );
    expect(screen.getByTestId("x-row").textContent).toMatch(/auto.*→.*8192/);
    expect(screen.getByTestId("x-row").textContent).not.toMatch(/null/);
  });

  it("labelFor uses MODEL_HINTS and falls back to the key", () => {
    expect(labelFor("max_model_len")).toBe("Max model length");
    expect(labelFor("not_a_key")).toBe("not_a_key");
  });

  it("renders the empty state", () => {
    render(<DiffList rows={[]} testIdPrefix="x" />);
    expect(screen.getByTestId("x-empty")).toBeInTheDocument();
  });
});
