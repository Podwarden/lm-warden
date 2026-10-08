// Component tests for the S4 settings-page additions: presets strip,
// Suggest panel, and Effective argv panel. The existing model-settings.test
// already pins the dirty-diff PATCH + 409 banner contract; this file
// complements it with the new UX surfaces.
//
// We reuse the same syncResolved + SWRConfig provider-reset pattern as
// model-settings.test so the React 19 `use(params)` call hydrates the page
// synchronously and SWR doesn't bleed cached fetches across tests.

import { Suspense } from "react";
import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
  act,
  within,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import ModelSettingsPage from "@/app/models/[id]/settings/page";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

// React 19 `use()` reads a `status`/`value` shape that the runtime attaches
// to a Promise once observed. Forging that shape up-front lets the page
// hydrate on the first render without driving a microtask queue (which
// jsdom doesn't reliably flush for React's suspense replay). Test-only
// hack — production gets a real async-resolving params promise from Next.
function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & {
    status?: string;
    value?: T;
  };
  p.status = "fulfilled";
  p.value = value;
  return p;
}

function renderPage(id: string) {
  const paramsPromise = syncResolved({ id });
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div data-testid="suspense-fallback">loading</div>}>
        <ModelSettingsPage params={paramsPromise} />
      </Suspense>
    </SWRConfig>,
  );
}

interface SettingsResponse {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number;
  dtype: string | null;
  max_model_len: number | null;
  gpu_memory_utilization: number;
  trust_remote_code: boolean;
  extra_args: string[];
  extra_env: Record<string, string>;
  status: string;
  pulled_bytes: number;
  pulled_total: number | null;
  last_error: string | null;
}

function fakeSettings(
  overrides: Partial<SettingsResponse> = {},
): SettingsResponse {
  return {
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
    status: "pulled",
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
    ...overrides,
  };
}

// Shared fake — four IDs matching the backend builtin.json (locked by
// tests/unit/presets/test_routes.py:test_presets_response_shape).
const PRESETS_FIXTURE = {
  presets: [
    {
      id: "a4000-tight-awq",
      name: "A4000 — tight AWQ",
      description: "Conservative 2× A4000 AWQ deployment.",
      target_archetype: "a4000-pair-awq",
      settings: {
        gpu_memory_utilization: 0.78,
        max_model_len: 8192,
      },
    },
    {
      id: "h100-single-shot",
      name: "H100 — single shot",
      description: "Single H100, no shard.",
      target_archetype: "h100-single",
      settings: {
        gpu_memory_utilization: 0.95,
        max_model_len: 32768,
      },
    },
    {
      id: "dev-tiny",
      name: "Dev tiny",
      description: "Low-VRAM smoke profile.",
      target_archetype: "dev",
      settings: {
        gpu_memory_utilization: 0.5,
        max_model_len: 2048,
      },
    },
    {
      id: "moe-balanced",
      name: "MoE balanced",
      description: "MoE-friendly balanced config.",
      target_archetype: "moe",
      settings: {
        gpu_memory_utilization: 0.85,
        max_model_len: 16384,
      },
    },
  ],
};

const EFFECTIVE_ARGV_FIXTURE = {
  argv: [
    "vllm",
    "serve",
    "meta-llama/Llama-3-8B",
    "--port",
    "10000",
    "--gpu-memory-utilization",
    "0.9",
  ],
};

function installFetchStub(opts: {
  settings: SettingsResponse;
  presets?: typeof PRESETS_FIXTURE | null;
  argv?: typeof EFFECTIVE_ARGV_FIXTURE | null;
  suggest?: Record<string, unknown> | null;
}) {
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    if (url === "/api/models/abc/settings" && method === "GET") {
      return new Response(JSON.stringify(opts.settings), { status: 200 });
    }
    if (url === "/api/models/abc/settings" && method === "PATCH") {
      return new Response('{"ok":true}', { status: 200 });
    }
    if (url === "/api/presets") {
      return new Response(JSON.stringify(opts.presets ?? PRESETS_FIXTURE), {
        status: 200,
      });
    }
    if (url === "/api/models/abc/effective-argv") {
      return new Response(JSON.stringify(opts.argv ?? EFFECTIVE_ARGV_FIXTURE), {
        status: 200,
      });
    }
    if (url === "/api/models/abc/suggest-config") {
      return new Response(
        JSON.stringify(
          opts.suggest ?? {
            gpu_memory_utilization: 0.85,
            max_model_len: 16384,
            kv_cache_dtype: null,
            disclaimer:
              "Heuristic suggestion — verify against your workload before saving.",
          },
        ),
        { status: 200 },
      );
    }
    throw new Error(`unexpected fetch: ${method} ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

// Flush the SWR fetch + first commit pass. `syncResolved` makes `use(params)`
// return synchronously, so we only need to drain the fetch microtask + give
// React one render pass to write the form.
async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Settings page — PresetStrip (#S4)", () => {
  it("renders all four preset chips from /api/presets", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    // Wait for the strip to mount — it only renders after the SWR fetch
    // settles, which on syncResolved is the same tick as flush().
    const strip = await screen.findByTestId("presets-strip");
    const chips = within(strip).getAllByRole("button");
    // 4 chips, in JSON order.
    expect(chips.length).toBe(4);
    expect(screen.getByTestId("preset-chip-a4000-tight-awq")).toBeInTheDocument();
    expect(screen.getByTestId("preset-chip-h100-single-shot")).toBeInTheDocument();
    expect(screen.getByTestId("preset-chip-dev-tiny")).toBeInTheDocument();
    expect(screen.getByTestId("preset-chip-moe-balanced")).toBeInTheDocument();
  });

  it("opens the confirm popover with a diff list when a chip is clicked", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("preset-chip-a4000-tight-awq"));

    // Popover renders with a diff list — there should be at least one
    // diff row since the preset changes gpu_memory_utilization (0.9 → 0.78).
    const confirm = await screen.findByTestId("preset-confirm");
    expect(confirm).toBeInTheDocument();
    const diffRows = within(confirm).queryAllByTestId("preset-diff-row");
    expect(diffRows.length).toBeGreaterThan(0);
    // Apply + Cancel both surfaced.
    expect(within(confirm).getByTestId("preset-apply")).toBeInTheDocument();
    expect(within(confirm).getByTestId("preset-cancel")).toBeInTheDocument();
  });

  it("applies the preset and increments the dirty count when Apply is clicked", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("preset-chip-a4000-tight-awq"));
    const confirm = await screen.findByTestId("preset-confirm");

    fireEvent.click(within(confirm).getByTestId("preset-apply"));

    // Popover dismisses after apply.
    await waitFor(() => {
      expect(screen.queryByTestId("preset-confirm")).not.toBeInTheDocument();
    });

    // Save button becomes enabled — proves the draft is dirty.
    const saveBtn = screen.getByTestId("settings-save");
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
  });

  it("Cancel dismisses the popover without dirtying the draft", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("preset-chip-a4000-tight-awq"));
    const confirm = await screen.findByTestId("preset-confirm");
    fireEvent.click(within(confirm).getByTestId("preset-cancel"));

    await waitFor(() => {
      expect(screen.queryByTestId("preset-confirm")).not.toBeInTheDocument();
    });

    // Save stays disabled — no dirty keys.
    expect(screen.getByTestId("settings-save")).toBeDisabled();
  });
});

describe("Settings page — SuggestPanel (#S4)", () => {
  it("fetches suggested values and renders the disclaimer rationale", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("suggest-fetch"));

    const result = await screen.findByTestId("suggest-result");
    expect(result).toBeInTheDocument();
    // Rationale renders the disclaimer string verbatim.
    const rationale = within(result).getByTestId("suggest-rationale");
    expect(rationale.textContent).toMatch(/Heuristic suggestion/);
    // Diff list non-empty (suggestion changes max_model_len 0 → 16384, gmu 0.9 → 0.85).
    const diffRows = within(result).queryAllByTestId("suggest-diff-row");
    expect(diffRows.length).toBeGreaterThan(0);
  });

  it("applies suggested values and dismisses the popover", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("suggest-fetch"));
    const result = await screen.findByTestId("suggest-result");

    fireEvent.click(within(result).getByTestId("suggest-apply"));

    await waitFor(() => {
      expect(screen.queryByTestId("suggest-result")).not.toBeInTheDocument();
    });
    // Save becomes enabled (draft is dirty after applying sparse settings).
    const saveBtn = screen.getByTestId("settings-save");
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
  });

  it("Dismiss closes the popover without touching the draft", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("suggest-fetch"));
    const result = await screen.findByTestId("suggest-result");
    fireEvent.click(within(result).getByTestId("suggest-dismiss"));

    await waitFor(() => {
      expect(screen.queryByTestId("suggest-result")).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("settings-save")).toBeDisabled();
  });

  it("does not show a 'disclaimer' diff row even though the response includes it", async () => {
    installFetchStub({
      settings: fakeSettings(),
      suggest: {
        gpu_memory_utilization: 0.85,
        max_model_len: 16384,
        kv_cache_dtype: null,
        disclaimer: "Verify before saving.",
      },
    });
    renderPage("abc");
    await flush();

    fireEvent.click(await screen.findByTestId("suggest-fetch"));
    const result = await screen.findByTestId("suggest-result");
    const diffRows = within(result).queryAllByTestId("suggest-diff-row");
    // None of the diff rows should mention "disclaimer" — the panel strips
    // that key + null entries before computing the diff.
    for (const row of diffRows) {
      expect(row.textContent).not.toMatch(/disclaimer/i);
      // kv_cache_dtype is null in the response → should also be filtered.
      expect(row.textContent).not.toMatch(/kv_cache_dtype/i);
    }
  });
});

describe("Settings page — EffectiveArgvPanel (#S4)", () => {
  it("renders the argv tokens in a faux-terminal pre block", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    const pre = await screen.findByTestId("effective-argv-pre");
    expect(pre).toBeInTheDocument();
    // The fixture argv joins to a recognizable command.
    expect(pre.textContent ?? "").toMatch(/vllm/);
    expect(pre.textContent ?? "").toMatch(/serve/);
    expect(pre.textContent ?? "").toMatch(/--gpu-memory-utilization/);
    expect(pre.textContent ?? "").toMatch(/0\.9/);
  });

  it("Copy button writes argv to the clipboard and shows a 'Copied' badge", async () => {
    // Typed signature here so writeText.mock.calls[0][0] survives tsc.
    const writeText = vi.fn(
      async (_text: string): Promise<void> => undefined,
    );
    // jsdom doesn't ship navigator.clipboard.
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    // #149 — copyToClipboard (lib/utils.ts) only takes the
    // navigator.clipboard branch when window.isSecureContext is true.
    // jsdom leaves it undefined, so we pin it here to drive the
    // intended path. The legacy execCommand fallback is exercised
    // separately in copy-to-clipboard.test.ts.
    Object.defineProperty(window, "isSecureContext", {
      configurable: true,
      writable: true,
      value: true,
    });

    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    const copyBtn = await screen.findByTestId("effective-argv-copy");
    await waitFor(() => expect(copyBtn).not.toBeDisabled());
    fireEvent.click(copyBtn);

    expect(writeText).toHaveBeenCalledTimes(1);
    const arg = writeText.mock.calls[0][0];
    expect(arg).toContain("vllm");
    expect(arg).toContain("--gpu-memory-utilization");
    // Button text flips to "Copied" briefly.
    await waitFor(() => expect(copyBtn.textContent).toMatch(/Copied/i));
  });

  it("renders an error message if /effective-argv fails", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === "string" ? input : (input as Request).url;
      const method = (init?.method ?? "GET").toUpperCase();
      if (url === "/api/models/abc/settings" && method === "GET") {
        return new Response(JSON.stringify(fakeSettings()), { status: 200 });
      }
      if (url === "/api/presets") {
        return new Response(JSON.stringify(PRESETS_FIXTURE), { status: 200 });
      }
      if (url === "/api/models/abc/effective-argv") {
        return new Response('{"detail":"engine offline"}', { status: 500 });
      }
      throw new Error(`unexpected: ${method} ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    renderPage("abc");
    await flush();

    const err = await screen.findByTestId("effective-argv-error");
    expect(err).toBeInTheDocument();
  });
});

describe("Settings page — SettingsSection collapsing (#S4)", () => {
  it("toggles section visibility via the disclosure button", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();

    // The Memory section contains gpu_memory_utilization — find its button
    // by the section testId attached to <section data-testid="section-memory">.
    const memorySection = await screen.findByTestId("section-memory");
    const toggle = within(memorySection).getByRole("button", {
      name: /memory/i,
    });
    // Body starts visible. The body's aria-controls/aria-expanded pair tells
    // us the section's open state — read aria-expanded.
    expect(toggle.getAttribute("aria-expanded")).toBe("true");

    fireEvent.click(toggle);
    await waitFor(() => {
      expect(toggle.getAttribute("aria-expanded")).toBe("false");
    });

    // Expand again — bidirectional.
    fireEvent.click(toggle);
    await waitFor(() => {
      expect(toggle.getAttribute("aria-expanded")).toBe("true");
    });
  });
});

// Settings redesign (plan §2.1, §4.4–4.5, Task 4).
describe("Settings page — redesign: presets when loaded, stale command", () => {
  it("does not render the preset card at all on a loaded model", async () => {
    // Replaces the old behaviour (chips rendered but disabled): a preset can
    // only produce locked keys, so on a loaded model it would be a dead control.
    installFetchStub({ settings: fakeSettings({ status: "loaded" }) });
    renderPage("abc");
    await flush();
    await screen.findByTestId("effective-argv-pre");
    expect(screen.queryByTestId("presets-card")).toBeNull();
    expect(screen.queryByTestId("presets-strip")).toBeNull();
  });

  it("notes that unsaved changes are not in the effective command yet", async () => {
    installFetchStub({ settings: fakeSettings() });
    renderPage("abc");
    await flush();
    await screen.findByTestId("effective-argv-pre");
    expect(screen.queryByTestId("effective-argv-stale")).toBeNull();
    fireEvent.change(screen.getByLabelText(/HF revision/i), { target: { value: "v2" } });
    await waitFor(() => expect(screen.getByTestId("effective-argv-stale")).toBeInTheDocument());
  });

  it("tints the saved extra args in the command", async () => {
    installFetchStub({
      settings: fakeSettings({ extra_args: ["--enforce-eager"] }),
      argv: { argv: [...EFFECTIVE_ARGV_FIXTURE.argv, "--enforce-eager"] },
    });
    renderPage("abc");
    await flush();
    await screen.findByTestId("effective-argv-pre");
    expect(screen.getByTestId("effective-argv-legend")).toBeInTheDocument();
  });
});
