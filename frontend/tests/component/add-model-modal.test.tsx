// Component tests for the #86 4-state-machine Add Model modal.
//
// Stages: enter-repo → discovering → select-file → submitting
//
// Test policy:
//   - We drive the modal entirely through user-visible interactions; the
//     network is stubbed at the global `fetch` boundary so the auth-fetch
//     wrapper is exercised end-to-end (CSRF, JSON headers, etc.).
//   - The classifier (`@/lib/fit`) is imported directly and trusted; the
//     ladder-tier assertions only check the *rendered* `data-verdict` on
//     the badge for the selected row.
//   - The portal-mounted Modal does not auto-unmount between tests in our
//     globals:false config, so we `cleanup()` in afterEach to avoid
//     "multiple elements with..." collisions.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
  act,
} from "@testing-library/react";
import { AddModelModal } from "@/components/models/add-model-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

// ---- fetch stub helpers ---------------------------------------------------

interface DiscoveredFileFixture {
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
}

interface DiscoveryFixture {
  files: DiscoveredFileFixture[];
  config: Record<string, unknown> | null;
  repo: Record<string, unknown>;
  errors: string[];
}

interface FitFixture {
  verdict: "green" | "yellow" | "orange" | "red";
  breakdown: {
    total_vram: number;
    weights_budget: number;
    kv_reserve: number;
    file_size: number;
    ratio: number;
    dtype_bytes: number;
    max_model_len_used: number;
  };
  recommended_max_model_len: number | null;
  warnings: string[];
}

const GIB = 1024 * 1024 * 1024;

function makeDiscovery(
  overrides: Partial<DiscoveryFixture> = {},
): DiscoveryFixture {
  return {
    files: [
      {
        filename: "model.safetensors",
        size: 8 * GIB,
        kind: "safetensors_single",
        quant: "fp16",
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
    config: { architectures: ["LlamaForCausalLM"] },
    repo: { id: "meta-llama/Llama-3-8B" },
    errors: [],
    ...overrides,
  };
}

/**
 * Build a FitPreviewResponse fixture whose `kv_reserve` term steers the
 * modal's CLIENT-SIDE reclassifier into the requested band.
 *
 * Critically, the modal's FileRow always recomputes the verdict locally
 * via `classifyFit(file.size, liveBudget)` whenever a GPU is selected,
 * which it is by default (the modal preselects GPU 0). The server's
 * `verdict` field is only consulted when *no* GPU is selected. So the
 * fixture must encode the target band via the math, not the field.
 *
 * `liveBudget = floor(totalVram * GMU) - kv_reserve`, with totalVram
 * derived from the selected GPU set. The default test setup uses one
 * 24 GiB GPU + GMU=0.9 → cap ≈ 21.6 GiB. For an 8 GiB file the
 * achievable ratios span ~0.37 (kv_reserve=0) up through ∞ (kv_reserve
 * ≥ cap → budget<=0 → red short-circuit). We pick mid-band kv_reserve
 * values per verdict so a small float rounding can't tip the band.
 */
function makeFit(
  verdict: FitFixture["verdict"],
  fileSize = 8 * GIB,
  opts: { capBytes?: number; serverVerdict?: FitFixture["verdict"] } = {},
): FitFixture {
  // Default cap mirrors the test fixture: GPU 0 alone, 24 GiB, GMU=0.9
  // (Math.floor matches `vramBudget()` in the modal exactly).
  const cap =
    opts.capBytes ?? Math.floor(24 * 1024 * 1024 * 1024 * 0.9);

  // Target ratios per band — choose mid-band values for stability.
  // green band: ratio < 0.55 ; pick 0.45
  // yellow band: 0.55 ≤ ratio < 0.80 ; pick 0.65
  // orange band: 0.80 ≤ ratio < 1.0 ; pick 0.90
  // red band: ratio ≥ 1.0 ; pick 1.20 (still positive budget, not the
  // negative-budget short-circuit)
  const ratioByVerdict: Record<FitFixture["verdict"], number> = {
    green: 0.45,
    yellow: 0.65,
    orange: 0.90,
    red: 1.20,
  };
  const targetRatio = ratioByVerdict[verdict];
  const liveBudget = Math.round(fileSize / targetRatio);
  // kv_reserve = cap - liveBudget. If targetRatio is too small (file
  // size doesn't fit in the cap at that ratio), clamp kv_reserve to 0
  // and accept the resulting ratio (still in band as long as it's < 0.55
  // for green). For the default test cap, an 8 GiB file at ratio 0.45
  // gives kv_reserve ≈ 3.81 GiB — positive and well-formed.
  const kv_reserve = Math.max(0, cap - liveBudget);

  return {
    verdict: opts.serverVerdict ?? verdict,
    breakdown: {
      total_vram: cap * 2,
      weights_budget: liveBudget,
      kv_reserve,
      file_size: fileSize,
      ratio: targetRatio,
      dtype_bytes: 2,
      max_model_len_used: 4096,
    },
    recommended_max_model_len: null,
    warnings: [],
  };
}

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

interface RouteHandlers {
  discover?: () => Response | Promise<Response>;
  fitPreview?: (body: Record<string, unknown>) => Response | Promise<Response>;
  gpus?: () => Response | Promise<Response>;
  create?: (body: Record<string, unknown>) => Response | Promise<Response>;
  pull?: () => Response | Promise<Response>;
  refresh?: () => Response | Promise<Response>;
  csrf?: () => Response | Promise<Response>;
}

interface FetchSpy {
  calls: Array<{ url: string; init?: RequestInit }>;
  countByUrl: (predicate: (url: string) => boolean) => number;
}

function installFetchStub(routes: RouteHandlers): FetchSpy {
  const calls: FetchSpy["calls"] = [];
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({ url, init });
    // Auth/CSRF endpoints — auth-fetch.ts may call these eagerly (when
    // accessToken is null) or as a 401 recovery step. For test isolation we
    // seed both tokens in beforeEach so the eager-refresh path is skipped,
    // but the 401-replay path will still fire if a test forces a 401 from
    // /api/models/discover (the gated-repo case). Returning a fresh token
    // lets the replay land back on the same mocked 401 — which is what
    // the modal then surfaces as the auth_required CTA.
    if (url === "/api/auth/refresh") {
      return (routes.refresh ?? (() => json({ access_token: "test-jwt-refreshed" })))();
    }
    if (url === "/api/csrf") {
      return (routes.csrf ?? (() => json({ csrf: "test-csrf" })))();
    }
    if (url.startsWith("/api/models/discover")) {
      return (routes.discover ?? (() => json(makeDiscovery())))();
    }
    if (url === "/api/models/fit-preview") {
      const body = init?.body ? JSON.parse(init.body as string) : {};
      return (routes.fitPreview ?? (() => json(makeFit("green"))))(body);
    }
    if (url === "/api/system/gpus") {
      return (routes.gpus ??
        (() =>
          json({
            probed_at: "2026-05-19T12:00:00Z",
            probe_error: null,
            gpus: GPUS_2x24,
          })))();
    }
    if (url === "/api/models" && init?.method === "POST") {
      const body = init?.body ? JSON.parse(init.body as string) : {};
      return (routes.create ??
        (() =>
          json({ id: "new-model", served_model_name: body.served_model_name }, 201)))(
        body,
      );
    }
    if (/^\/api\/models\/.+\/pull$/.test(url) && init?.method === "POST") {
      return (routes.pull ?? (() => new Response(null, { status: 202 })))();
    }
    throw new Error(`Unmocked fetch: ${init?.method ?? "GET"} ${url}`);
  });
  vi.stubGlobal("fetch", mock);
  return {
    calls,
    countByUrl: (predicate) => calls.filter((c) => predicate(c.url)).length,
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// ---- shared lifecycle -----------------------------------------------------

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// ---- tests ----------------------------------------------------------------

describe("AddModelModal — state machine", () => {
  it("starts in enter-repo and rejects an invalid HF repo client-side", async () => {
    installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);

    // enter-repo stage has the Discover button, not Add.
    expect(screen.getByRole("button", { name: /discover/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^add$/i })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "not-a-valid-repo" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    expect(await screen.findByText(/owner\/name/i)).toBeInTheDocument();
    // Still on enter-repo — no spinner, no file table.
    expect(screen.queryByTestId("file-table")).not.toBeInTheDocument();
  });

  it("transitions enter-repo → discovering → select-file on a successful discover", async () => {
    installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // file-table is the canonical select-file marker.
    expect(await screen.findByTestId("file-table")).toBeInTheDocument();
    // Add button replaces Discover.
    expect(screen.getByRole("button", { name: /^add$/i })).toBeInTheDocument();
    // The default weights file is preselected; served name is auto-derived.
    const nameInput = screen.getByLabelText(/served model name/i) as HTMLInputElement;
    expect(nameInput.value.length).toBeGreaterThan(0);
  });

  it("shows the gated-repo CTA and hides the file table on 401 auth_required", async () => {
    installFetchStub({
      discover: () =>
        json(
          {
            detail: {
              error_code: "auth_required",
              message: "HF Hub requires a token for this repo.",
              repo_id: "meta-llama/Llama-3-8B",
              revision: "main",
            },
          },
          401,
        ),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // CTA link to /setup/hf-token is the contract for gated-repo state.
    const cta = await screen.findByRole("link", { name: /open token settings/i });
    expect(cta).toHaveAttribute("href", "/setup/hf-token");
    // Table must be hidden — operator cannot pick a file without a token.
    expect(screen.queryByTestId("file-table")).not.toBeInTheDocument();
  });

  it("surfaces a non-401 discovery error and routes back to enter-repo", async () => {
    installFetchStub({
      discover: () =>
        json(
          {
            detail: {
              error_code: "repo_not_found",
              message: "Repository not found",
              repo_id: "x/y",
              revision: "main",
            },
          },
          404,
        ),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "x/y" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByText(/repository not found/i)).toBeInTheDocument();
    // Bounced back to enter-repo — the Discover button is back.
    expect(screen.getByRole("button", { name: /discover/i })).toBeInTheDocument();
    expect(screen.queryByTestId("file-table")).not.toBeInTheDocument();
  });
});

describe("AddModelModal — fit badge tiers", () => {
  // The FileRow uses the CLIENT-SIDE classifier (`@/lib/fit`) whenever a
  // GPU is selected (which is the default state — modal preselects GPU 0).
  // makeFit() encodes the band via kv_reserve so the recomputed liveBudget
  // lands in the requested tier; this is the same path a real GPU-checkbox
  // tick would exercise.
  it.each([
    ["green"],
    ["yellow"],
    ["orange"],
    ["red"],
  ])("renders %s verdict on the row badge", async (verdictTier) => {
    installFetchStub({
      fitPreview: () =>
        json(makeFit(verdictTier as FitFixture["verdict"])),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(() => {
      expect(badge).toHaveAttribute("data-verdict", verdictTier);
    });
  });
});

describe("AddModelModal — GPU checkbox recomputes verdict client-side", () => {
  it("ticking a second GPU widens the budget and never fires a second fit-preview", async () => {
    // Default GPU selection (just GPU 0) + an 8 GiB file with a contrived
    // kv_reserve that lands the single-GPU ratio in the "yellow" band.
    // Ticking GPU 1 doubles totalVram, which leaves the same kv_reserve
    // but widens liveBudget enough to flip the verdict to "green" — a
    // band move the client-side classifier owns end-to-end.
    //
    // Single GPU (24 GiB, GMU=0.9) → cap ≈ 21.6 GiB. makeFit("yellow")
    // picks kv_reserve so single-GPU ratio = 0.65 → yellow.
    // Two GPUs (48 GiB, GMU=0.9) → cap ≈ 43.2 GiB. liveBudget grows by
    // +21.6 GiB → ratio drops to ~0.226 → green.
    //
    // The critical assertion is that the badge updates WITHOUT a second
    // POST /api/models/fit-preview: the client-side reclassifier owns
    // the recompute path on every GPU-checkbox tick.
    const spy = installFetchStub({
      fitPreview: () => json(makeFit("yellow")),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Wait for first paint with the yellow verdict (single GPU baseline).
    await waitFor(() => {
      const badge = screen.getByTestId("fit-badge-model.safetensors");
      expect(badge).toHaveAttribute("data-verdict", "yellow");
    });
    const fitPreviewCalls0 = spy.countByUrl((u) => u === "/api/models/fit-preview");
    expect(fitPreviewCalls0).toBe(1);

    // Tick GPU 1 → totalVram doubles → liveBudget widens → green.
    const gpu1 = screen.getByLabelText(/#1 RTX 4090/i);
    await act(async () => {
      fireEvent.click(gpu1);
    });
    await waitFor(() => {
      const badge = screen.getByTestId("fit-badge-model.safetensors");
      expect(badge).toHaveAttribute("data-verdict", "green");
    });

    // No additional fit-preview call fired. The client owns the recompute.
    const fitPreviewCalls1 = spy.countByUrl((u) => u === "/api/models/fit-preview");
    expect(fitPreviewCalls1).toBe(1);
  });
});

describe("AddModelModal — GGUF soft-warn", () => {
  it("shows the GGUF warning banner when a .gguf file is selected and still allows submit", async () => {
    installFetchStub({
      discover: () =>
        json(
          makeDiscovery({
            files: [
              {
                filename: "model.q4_K_M.gguf",
                size: 6 * GIB,
                kind: "gguf",
                quant: "q4_K_M",
                params: 7_000_000_000,
              },
            ],
          }),
        ),
      fitPreview: () => json(makeFit("green", 6 * GIB)),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "TheBloke/Llama-7B-GGUF" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    await screen.findByTestId("file-table");

    // Sub-project C: this banner is a statement about VLLM's loader ("verify
    // the inferred arch matches a vLLM-known family"), so it is now gated on
    // the vLLM backend. A .gguf pre-selects llama.cpp (D6), where the banner
    // would be false and confusing, so it starts hidden...
    expect(screen.queryByTestId("gguf-warn")).toBeNull();

    // ...and reappears the moment the operator puts the model back on vLLM,
    // which is exactly when it becomes true again. What this test has always
    // protected -- that the banner is INFORMATIONAL and never blocks submit --
    // is unchanged.
    fireEvent.change(screen.getByTestId("backend-select"), {
      target: { value: "vllm" },
    });
    expect(await screen.findByTestId("gguf-warn")).toBeInTheDocument();
    const submitBtn = screen.getByRole("button", { name: /^add$/i });
    expect(submitBtn).not.toBeDisabled();
  });
});

describe("AddModelModal — submit happy path", () => {
  it("POSTs to /api/models with filename + gpu_indices and closes on 201", async () => {
    const onClose = vi.fn();
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={onClose} />);

    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // Default GPU is #0; tick #1 as well so gpu_indices=[0,1] in the body.
    fireEvent.click(screen.getByLabelText(/#1 RTX 4090/i));

    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const postCall = spy.calls.find(
      (c) => c.url === "/api/models" && c.init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const body = JSON.parse(postCall!.init!.body as string);
    expect(body).toMatchObject({
      hf_repo: "meta-llama/Llama-3-8B",
      gpu_indices: [0, 1],
      filename: "model.safetensors",
    });
    expect(typeof body.served_model_name).toBe("string");
    expect((body.served_model_name as string).length).toBeGreaterThan(0);
  });

  it("surfaces 409 duplicate name error and stays on select-file", async () => {
    installFetchStub({
      create: () =>
        json({ detail: "served_model_name 'llama3-8b' already exists" }, 409),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    expect(await screen.findByText(/already exists/i)).toBeInTheDocument();
    // Still on select-file (Add button visible).
    expect(screen.getByRole("button", { name: /^add$/i })).toBeInTheDocument();
  });
});

// ---- CR fix-up regressions (!86) -----------------------------------------
//
// Tests below pin the 4 Important findings from CR-86 (FE pass). Each one is
// labeled with its CR id (I1..I4) so future readers can trace a failure back
// to the original review.

describe("AddModelModal — I1 Cancel-during-discovering race", () => {
  it("ignores a late-arriving discover response after Cancel is clicked", async () => {
    // Pin discover behind a manual gate so we can interleave Cancel between
    // the click and the resolution. Pre-fix, the late response would
    // unconditionally setStage("select-file") and populate form state —
    // yanking the operator off enter-repo. Post-fix, the stage re-check
    // bails before any setter runs.
    let resolveDiscover: (r: Response) => void = () => {};
    const discoverPromise = new Promise<Response>((resolve) => {
      resolveDiscover = resolve;
    });
    installFetchStub({
      discover: () => discoverPromise,
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Spinner visible — discover in flight.
    expect(await screen.findByText(/discovering files/i)).toBeInTheDocument();

    // Operator hits Cancel on the spinner — stage flips back to enter-repo.
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.getByRole("button", { name: /discover/i })).toBeInTheDocument();

    // Now the late response lands. Pre-fix: stage flips to select-file.
    // Post-fix: stillDiscovering() returns false, every setter short-circuits.
    await act(async () => {
      resolveDiscover(json(makeDiscovery()));
      // Drain the post-await microtask chain. Without this the assertions
      // below run before startDiscovery's `if (!stillDiscovering()) return`
      // path executes — yielding a false pass.
      await Promise.resolve();
      await Promise.resolve();
    });

    // Still on enter-repo: file table must NOT appear, Discover button still
    // present. authRequired CTA must also stay hidden.
    expect(screen.queryByTestId("file-table")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /discover/i })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /open token settings/i })).not.toBeInTheDocument();
  });

  it("ignores a late 401 auth_required after Cancel is clicked", async () => {
    // Same race, different branch. The 401-auth-required path also calls
    // setAuthRequired + setStage("select-file") and was equally unguarded
    // pre-fix.
    let resolveDiscover: (r: Response) => void = () => {};
    const discoverPromise = new Promise<Response>((resolve) => {
      resolveDiscover = resolve;
    });
    installFetchStub({
      discover: () => discoverPromise,
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    expect(await screen.findByText(/discovering files/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

    await act(async () => {
      resolveDiscover(
        json(
          {
            detail: {
              error_code: "auth_required",
              message: "HF token required",
              repo_id: "meta-llama/Llama-3-8B",
              revision: "main",
            },
          },
          401,
        ),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    // The gated-repo CTA must NOT appear — operator backed out.
    expect(screen.queryByRole("link", { name: /open token settings/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /discover/i })).toBeInTheDocument();
  });
});

describe("AddModelModal — I2 tooltip tracks live GPU-dependent budget", () => {
  it("updates weights_budget and ratio in the badge tooltip when a 2nd GPU is ticked", async () => {
    // The badge already recomputes verdict client-side on GPU toggle (covered
    // by the existing GPU-checkbox test). The tooltip surfacing
    // `weights_budget` and `ratio` was reading the SERVER's snapshot
    // unconditionally, so it would contradict the verdict after a GPU
    // change. Post-fix, the tooltip recomputes both using `liveBudget`.
    installFetchStub({
      fitPreview: () => json(makeFit("yellow")),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Wait for first paint with cached fit data.
    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(() => {
      expect(badge).toHaveAttribute("data-verdict", "yellow");
      // Tooltip is in the `title` attribute. Wait until it's populated by
      // the post-fetch render.
      expect(badge.getAttribute("title")).toMatch(/weights_budget:/);
    });

    function parseTooltip(t: string | null): { budget: string; ratio: string } {
      const lines = (t ?? "").split("\n");
      const budgetLine = lines.find((l) => l.startsWith("weights_budget:")) ?? "";
      const ratioLine = lines.find((l) => l.startsWith("ratio:")) ?? "";
      return { budget: budgetLine, ratio: ratioLine };
    }

    const singleGpu = parseTooltip(badge.getAttribute("title"));

    // Tick GPU 1 → totalVram doubles → liveBudget widens → both badge and
    // tooltip must reflect that.
    await act(async () => {
      fireEvent.click(screen.getByLabelText(/#1 RTX 4090/i));
    });
    await waitFor(() => {
      expect(badge).toHaveAttribute("data-verdict", "green");
    });
    const dualGpu = parseTooltip(badge.getAttribute("title"));

    // Both lines must have changed. Pre-fix, they'd be identical to
    // `singleGpu` because the tooltip rendered `fit.breakdown.weights_budget`
    // (a server-time snapshot) directly.
    expect(dualGpu.budget).not.toBe(singleGpu.budget);
    expect(dualGpu.ratio).not.toBe(singleGpu.ratio);
  });
});

describe("AddModelModal — I3 stale fit-preview after close", () => {
  it("drops a fit-preview response that resolves after the modal closes", async () => {
    // Gate the fit-preview behind a manual resolver. Close the modal while
    // the request is in flight, then resolve it. Pre-fix, the response
    // would land into the unmounted component's setFitByFilename, surfacing
    // a stale entry the next time the modal opens. Post-fix, the openGen
    // capture short-circuits the setter.
    //
    // We're testing the guard behaviour, not the (unobservable) absence of
    // a console warning. The assertion: after reopening with the same repo,
    // the fit-preview must be REFETCHED rather than served from the stale
    // cache. Two fit-preview calls observed → guard is working.
    let resolveFit: (r: Response) => void = () => {};
    let fitPromise = new Promise<Response>((resolve) => {
      resolveFit = resolve;
    });
    let fitCallCount = 0;
    const spy = installFetchStub({
      fitPreview: () => {
        fitCallCount += 1;
        // First call hangs; subsequent calls return immediately so the
        // reopen path can complete its first paint.
        if (fitCallCount === 1) return fitPromise;
        return json(makeFit("green"));
      },
    });

    const onClose = vi.fn();
    let view = render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // First fit-preview is in flight. Close the modal (Cancel button on
    // select-file). resetForm bumps openGen.
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalled();

    // Now resolve the stale fit response. Post-fix this MUST be discarded.
    await act(async () => {
      resolveFit(json(makeFit("red")));
      await Promise.resolve();
      await Promise.resolve();
    });

    // Reopen the modal — fresh state, new gen, repo entered fresh. The
    // setup mounts with `open` controlled by the parent in real life;
    // here we re-render with the same component instance and a fresh
    // sequence of interactions.
    view.unmount();
    view = render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // Wait for the second fit-preview to land + render its verdict.
    await waitFor(() => {
      const badge = screen.getByTestId("fit-badge-model.safetensors");
      // The second call returns "green" — proves the cache was NOT
      // populated by the stale resolution (which returned "red").
      expect(badge).toHaveAttribute("data-verdict", "green");
    });

    // Two fit-preview calls total: one stale + one fresh. The fact that
    // a SECOND call fired at all proves the stale cache write was suppressed.
    const fitCalls = spy.countByUrl((u) => u === "/api/models/fit-preview");
    expect(fitCalls).toBe(2);
  });
});

describe("AddModelModal — I4 focus moves to first weights radio on select-file", () => {
  it("moves focus into the file table when the stage transitions", async () => {
    installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // The first selectable weights radio (model.safetensors) should now
    // have focus. Pre-fix, focus fell back to <body> because the autoFocus
    // target on enter-repo unmounted with the form.
    const firstRadio = screen.getByLabelText(/select model\.safetensors/i);
    await waitFor(() => {
      expect(document.activeElement).toBe(firstRadio);
    });
  });
});

// ---- #87: Advanced section + tooltip math + recommended hint -----------
//
// Tests below pin the contract for issue #87:
//   - Collapsible advanced section (collapsed by default)
//   - Three new inputs (parallelism_strategy, max_batch_size, max_model_len)
//   - Single-host PP is NOT blocked client-side
//   - Tooltip math surfaces all four primitives
//   - "Recommended max_model_len" hint on orange rows, not on green/yellow
//   - Create payload carries the three new fields

describe("AddModelModal — #87 advanced section collapse + inputs", () => {
  async function openModalToSelectFile() {
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");
  }

  it("renders the advanced section collapsed by default", async () => {
    installFetchStub({});
    await openModalToSelectFile();

    const advanced = screen.getByTestId("advanced-section");
    // <details> exposes its open state directly. Collapsed-by-default is the
    // contract — operators shouldn't see the knobs unless they ask.
    expect(advanced).not.toHaveAttribute("open");
    // The body element exists in the DOM (Testing Library doesn't hide
    // closed <details> children), but the disclosure widget is closed.
    expect(advanced.tagName.toLowerCase()).toBe("details");
  });

  it("expands when the summary is clicked, revealing all three inputs", async () => {
    installFetchStub({});
    await openModalToSelectFile();

    const advanced = screen.getByTestId("advanced-section") as HTMLDetailsElement;
    expect(advanced.open).toBe(false);

    // jsdom doesn't fire the native toggle on summary click; set the prop
    // directly the way a real click would, then dispatch a toggle event so
    // any React listeners (none today, but future-proofed) run.
    advanced.open = true;
    await act(async () => {
      advanced.dispatchEvent(new Event("toggle"));
    });

    // All three inputs render. They're always in the DOM (because <details>
    // doesn't unmount its children), but the contract is that the operator
    // can see and interact with them once expanded.
    expect(screen.getByTestId("parallelism-strategy")).toBeInTheDocument();
    expect(screen.getByTestId("max-batch-size")).toBeInTheDocument();
    expect(screen.getByTestId("max-model-len")).toBeInTheDocument();
  });

  it("accepts valid values in all three advanced inputs", async () => {
    installFetchStub({});
    await openModalToSelectFile();

    const strategy = screen.getByTestId("parallelism-strategy") as HTMLSelectElement;
    const batch = screen.getByTestId("max-batch-size") as HTMLInputElement;
    const len = screen.getByTestId("max-model-len") as HTMLInputElement;

    fireEvent.change(strategy, { target: { value: "tp" } });
    expect(strategy.value).toBe("tp");

    fireEvent.change(batch, { target: { value: "16" } });
    expect(batch.value).toBe("16");

    fireEvent.change(len, { target: { value: "8192" } });
    expect(len.value).toBe("8192");
  });
});

describe("AddModelModal — #87 single-host PP is allowed", () => {
  it("does not disable Add when parallelism_strategy=pp on a single-host config", async () => {
    // The fetch stub serves a single-host GPU snapshot (GPUS_2x24, but only
    // one of them is selected by default — which is moot for the PP test
    // because the constraint is "single HOST", not "single GPU"). The point:
    // selecting pp must NOT add any client-side block on submission. The
    // backend's cmd_builder (#88) emits --pipeline-parallel-size and vLLM
    // accepts it; the CTO decision on #87 was to mirror that here.
    installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    const strategy = screen.getByTestId("parallelism-strategy") as HTMLSelectElement;
    fireEvent.change(strategy, { target: { value: "pp" } });
    expect(strategy.value).toBe("pp");

    // Add button must remain enabled.
    const submitBtn = screen.getByRole("button", { name: /^add$/i });
    expect(submitBtn).not.toBeDisabled();
  });
});

describe("AddModelModal — #87 tooltip math breakdown", () => {
  it("renders all four primitives (bytes_per_token, kv_reserve, weights_budget, ratio)", async () => {
    installFetchStub({
      fitPreview: () => json(makeFit("yellow")),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(() => {
      const t = badge.getAttribute("title") ?? "";
      // All four primitives must appear in the tooltip. We assert on the
      // *labels* rather than specific numeric values because the live
      // budget recompute already covers value-correctness elsewhere.
      expect(t).toMatch(/bytes_per_token:/);
      expect(t).toMatch(/kv_reserve:/);
      expect(t).toMatch(/weights_budget:/);
      expect(t).toMatch(/ratio:/);
    });
  });
});

describe("AddModelModal — #87 recommended_max_model_len hint", () => {
  it("renders the hint on an orange row using the backend's recommended value", async () => {
    // Backend always returns a recommended_max_model_len for orange/red.
    // Override the fixture to include it (default makeFit() sets null).
    installFetchStub({
      fitPreview: () => {
        const f = makeFit("orange");
        f.recommended_max_model_len = 2048;
        return json(f);
      },
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    const hint = await screen.findByTestId(
      "recommended-max-model-len-model.safetensors",
    );
    expect(hint).toBeInTheDocument();
    expect(hint).toHaveAttribute("data-recommended-max-model-len", "2048");
    // Localised number rendering ("2,048") — assert on the digits, agnostic
    // to thousands separator differences.
    expect(hint.textContent ?? "").toMatch(/2[,. ]?048/);
  });

  it("does NOT render the hint on a green row", async () => {
    installFetchStub({
      fitPreview: () => json(makeFit("green")),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(() => {
      expect(badge).toHaveAttribute("data-verdict", "green");
    });
    expect(
      screen.queryByTestId("recommended-max-model-len-model.safetensors"),
    ).not.toBeInTheDocument();
  });

  it("does NOT render the hint on a yellow row", async () => {
    installFetchStub({
      fitPreview: () => json(makeFit("yellow")),
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(() => {
      expect(badge).toHaveAttribute("data-verdict", "yellow");
    });
    expect(
      screen.queryByTestId("recommended-max-model-len-model.safetensors"),
    ).not.toBeInTheDocument();
  });
});

describe("AddModelModal — #87 create payload includes advanced fields", () => {
  it("POSTs parallelism_strategy + max_batch_size + max_model_len with the user's values", async () => {
    const onClose = vi.fn();
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // Set all three advanced fields. The section is collapsed by default
    // but the inputs are still in the DOM (<details> children stay mounted).
    fireEvent.change(screen.getByTestId("parallelism-strategy"), {
      target: { value: "pp" },
    });
    fireEvent.change(screen.getByTestId("max-batch-size"), {
      target: { value: "8" },
    });
    fireEvent.change(screen.getByTestId("max-model-len"), {
      target: { value: "4096" },
    });

    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const postCall = spy.calls.find(
      (c) => c.url === "/api/models" && c.init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const body = JSON.parse(postCall!.init!.body as string);
    expect(body).toMatchObject({
      parallelism_strategy: "pp",
      max_batch_size: 8,
      max_model_len: 4096,
    });
  });

  it("omits max_model_len when left blank (backend uses config default)", async () => {
    const onClose = vi.fn();
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // Don't touch the advanced inputs — defaults are auto / 1 / "".
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const postCall = spy.calls.find(
      (c) => c.url === "/api/models" && c.init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const body = JSON.parse(postCall!.init!.body as string);
    // Defaults flow through:
    expect(body.parallelism_strategy).toBe("auto");
    expect(body.max_batch_size).toBe(1);
    // Blank max_model_len → omit so backend takes config default. The
    // backend's pydantic field is `max_model_len: int | None = Field(None,
    // gt=0)` — sending null would also work but omitting is cleaner.
    expect("max_model_len" in body).toBe(false);
  });
});

describe("AddModelModal — #106 base-repo + tokenizer-repo payload", () => {
  // The Advanced section ships a single "Base repo" input that fans out to
  // BOTH hf_config_repo and tokenizer_repo by default — that's the common
  // GGUF-republish case (unsloth pushes a quantized repo without config.json
  // and reuses the upstream tokenizer). A nested <details> exposes a
  // tokenizer-only override for the rarer split case. These tests pin both
  // shapes so the FE can't regress past the contract the BE schema enforces.
  it("fans base-repo out to both hf_config_repo and tokenizer_repo", async () => {
    const onClose = vi.fn();
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "unsloth/Qwen3-30B-A3B-GGUF" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    fireEvent.change(screen.getByTestId("base-repo"), {
      target: { value: "Qwen/Qwen3-30B-A3B" },
    });

    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const postCall = spy.calls.find(
      (c) => c.url === "/api/models" && c.init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const body = JSON.parse(postCall!.init!.body as string);
    expect(body).toMatchObject({
      hf_config_repo: "Qwen/Qwen3-30B-A3B",
      tokenizer_repo: "Qwen/Qwen3-30B-A3B",
    });
  });

  it("lets tokenizer-repo override win for tokenizer_repo while base-repo keeps hf_config_repo", async () => {
    const onClose = vi.fn();
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "unsloth/Qwen3-30B-A3B-GGUF" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    fireEvent.change(screen.getByTestId("base-repo"), {
      target: { value: "Qwen/Qwen3-30B-A3B" },
    });
    fireEvent.change(screen.getByTestId("tokenizer-repo"), {
      target: { value: "Qwen/Qwen3-30B-A3B-Instruct" },
    });

    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const postCall = spy.calls.find(
      (c) => c.url === "/api/models" && c.init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const body = JSON.parse(postCall!.init!.body as string);
    expect(body).toMatchObject({
      hf_config_repo: "Qwen/Qwen3-30B-A3B",
      tokenizer_repo: "Qwen/Qwen3-30B-A3B-Instruct",
    });
  });

  it("omits both fields when neither base-repo nor tokenizer-repo is set", async () => {
    const onClose = vi.fn();
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const postCall = spy.calls.find(
      (c) => c.url === "/api/models" && c.init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const body = JSON.parse(postCall!.init!.body as string);
    expect("hf_config_repo" in body).toBe(false);
    expect("tokenizer_repo" in body).toBe(false);
  });
});

// ---- #87 CR fix-up: debounced refetch + stale-guard + FE-fallback hint ----
//
// Five tests pin the fix-up landed in the same commit as the original #87
// work (see changelog entry under #87). They use REAL timers and wait out
// the 300 ms debounce window with `waitFor` rather than fake timers, because
// Testing Library's polling-based queries don't compose with `vi.useFakeTimers`
// — the polling itself relies on the macrotask queue advancing, which fake
// timers freeze. The existing 25 tests above all use real timers; these
// follow the same pattern so the file stays internally consistent.

// 300 ms debounce + a small jitter budget. Bumping past 700 ms would mask
// real regressions (a debounce that fires twice within 700 ms would still
// pass). Pinning the timeout to ~5× the debounce keeps the assertion
// tight while leaving CI slack.
const DEBOUNCE_WAIT_MS = 1500;

describe("AddModelModal — #87 CR fix-up: FE-fallback recommendation hint", () => {
  it("renders the FE-solved hint on an orange row when the backend returns recommended_max_model_len=null", async () => {
    // The backend's degraded-config branch returns null for the
    // recommendation field even on orange verdicts. The FE must fall back
    // to `recommendMaxModelLen()` using `bytes_per_token` derived from
    // `kv_reserve / max_model_len_used` and the live VRAM cap.
    //
    // Default cap (GPU 0, 24 GiB, GMU=0.9) ≈ 21.6 GiB → liveCapBytes used
    // by the solver. makeFit("orange") sets `max_model_len_used=4096`, so
    // bytesPerToken = round(kv_reserve / 4096). The solver then computes
    // L = floor((cap - fileSize/0.70) / bytesPerToken), which lands at a
    // positive integer for an 8 GiB file. We don't pin the exact L (float
    // precision noise across the cap math); we pin that the hint renders
    // AND that its value is a positive integer above the threshold.
    installFetchStub({
      fitPreview: () => {
        const f = makeFit("orange");
        // Explicitly null — the backend's degraded-config branch.
        f.recommended_max_model_len = null;
        return json(f);
      },
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    const hint = await screen.findByTestId(
      "recommended-max-model-len-model.safetensors",
    );
    const recValue = Number(
      hint.getAttribute("data-recommended-max-model-len"),
    );
    expect(Number.isInteger(recValue)).toBe(true);
    expect(recValue).toBeGreaterThan(0);
    // The rendered text must include the localised number — the digits
    // must appear in the textContent in some thousands-separator form.
    expect(hint.textContent ?? "").toMatch(new RegExp(String(recValue).slice(0, 1)));
  });
});

describe("AddModelModal — #87 CR fix-up: debounced refetch carries overrides", () => {
  it("refetches fit-preview with max_batch_size in the body after a debounced edit", async () => {
    // After first-paint, edit max_batch_size in the Advanced section. The
    // 300 ms debounce should fire ONE additional POST /api/models/fit-preview
    // whose body includes `max_batch_size: 16`. Pre-fix, the body wouldn't
    // include it (and no refetch fired at all), so the tooltip math
    // described a different submission than Add would POST.
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // Wait for the first-paint fit-preview to land so we have a clean
    // baseline call count.
    await waitFor(() => {
      expect(
        spy.countByUrl((u) => u === "/api/models/fit-preview"),
      ).toBe(1);
    });

    // Edit max_batch_size. The Advanced section's <details> doesn't unmount
    // its children, so the input is in the DOM even when collapsed.
    fireEvent.change(screen.getByTestId("max-batch-size"), {
      target: { value: "16" },
    });

    // Wait out the debounce window — the second POST must land within it.
    await waitFor(
      () => {
        expect(
          spy.countByUrl((u) => u === "/api/models/fit-preview"),
        ).toBe(2);
      },
      { timeout: DEBOUNCE_WAIT_MS },
    );

    const secondCall = spy.calls.filter(
      (c) => c.url === "/api/models/fit-preview",
    )[1];
    const body = JSON.parse(secondCall.init!.body as string);
    expect(body.max_batch_size).toBe(16);
  });

  it("refetches fit-preview with max_model_len in the body after a debounced edit", async () => {
    // Mirror of the batch test, for max_model_len. Both fields feed
    // kv_reserve, so both must trigger the debounced refetch.
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    await waitFor(() => {
      expect(
        spy.countByUrl((u) => u === "/api/models/fit-preview"),
      ).toBe(1);
    });

    fireEvent.change(screen.getByTestId("max-model-len"), {
      target: { value: "8192" },
    });

    await waitFor(
      () => {
        expect(
          spy.countByUrl((u) => u === "/api/models/fit-preview"),
        ).toBe(2);
      },
      { timeout: DEBOUNCE_WAIT_MS },
    );

    const secondCall = spy.calls.filter(
      (c) => c.url === "/api/models/fit-preview",
    )[1];
    const body = JSON.parse(secondCall.init!.body as string);
    expect(body.max_model_len).toBe(8192);
  });
});

describe("AddModelModal — #87 CR fix-up: tooltip reflects new kv_reserve after debounced refetch", () => {
  it("tooltip kv_reserve scales ~16× when max_batch_size flips from 1 to 16", async () => {
    // Backend KV-reserve math: kv_reserve = bytes_per_token * max_model_len
    // * max_batch_size. Holding bytes_per_token and max_model_len constant,
    // bumping max_batch_size 1→16 must scale kv_reserve by 16×.
    //
    // We simulate the BE by returning a fit-preview whose kv_reserve
    // depends on the posted body.max_batch_size, then assert the tooltip's
    // kv_reserve line moves with it after the debounced refetch.
    const KV_BASELINE_BYTES = 1 * GIB; // 1 GiB at max_batch_size=1
    installFetchStub({
      fitPreview: (body) => {
        const mbs =
          typeof body.max_batch_size === "number" ? body.max_batch_size : 1;
        // Build an orange-band fixture by hand because makeFit() doesn't
        // parametrise kv_reserve directly — we need control over it here.
        const cap = Math.floor(24 * 1024 * 1024 * 1024 * 0.9);
        const kvReserve = KV_BASELINE_BYTES * mbs;
        const liveBudget = Math.max(1, cap - kvReserve);
        const fileSize = 8 * GIB;
        return json({
          verdict: "orange" as const,
          breakdown: {
            total_vram: cap * 2,
            weights_budget: liveBudget,
            kv_reserve: kvReserve,
            file_size: fileSize,
            ratio: fileSize / liveBudget,
            dtype_bytes: 2,
            max_model_len_used: 4096,
          },
          recommended_max_model_len: null,
          warnings: [],
        });
      },
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Baseline tooltip — first-paint posts max_batch_size omitted → BE
    // defaults to 1 → kv_reserve = 1 GiB.
    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(() => {
      expect(badge.getAttribute("title") ?? "").toMatch(/kv_reserve:/);
    });

    function parseKvBytes(title: string | null): number {
      // Tooltip line format: "kv_reserve: 1.00 GiB" (or "MiB", etc.). Pull
      // out the numeric magnitude and the unit, then compare in bytes.
      const line =
        (title ?? "").split("\n").find((l) => l.startsWith("kv_reserve:")) ??
        "";
      const m = line.match(/kv_reserve:\s+([\d.]+)\s+(B|KiB|MiB|GiB|TiB)/);
      if (!m) return NaN;
      const n = Number(m[1]);
      const unit = m[2];
      const mult: Record<string, number> = {
        B: 1,
        KiB: 1024,
        MiB: 1024 ** 2,
        GiB: 1024 ** 3,
        TiB: 1024 ** 4,
      };
      return n * mult[unit];
    }

    const baselineKv = parseKvBytes(badge.getAttribute("title"));
    expect(baselineKv).toBeGreaterThan(0);

    // Bump max_batch_size → 16. The debounced refetch should land within
    // the 300 ms window + a render tick. The new kv_reserve must be
    // ~16× the baseline (allow ±2% for `formatBytes` rounding to 2 dp).
    fireEvent.change(screen.getByTestId("max-batch-size"), {
      target: { value: "16" },
    });

    await waitFor(
      () => {
        const kv = parseKvBytes(badge.getAttribute("title"));
        // ~16× ± 2% tolerance — formatBytes rounds to 2 decimals at the
        // chosen unit, so a perfect 16× ratio in bytes might show as
        // 15.97× or 16.03× after the format pass.
        expect(kv / baselineKv).toBeGreaterThan(15.5);
        expect(kv / baselineKv).toBeLessThan(16.5);
      },
      { timeout: DEBOUNCE_WAIT_MS },
    );
  });
});

describe("AddModelModal — #87 CR fix-up: same-session stale-guard (fitSeqRef)", () => {
  it("drops a slow first-paint response that lands after a fresh debounced refetch", async () => {
    // Same-session stale-guard. Setup:
    //   - First-paint fit-preview is gated behind a manual resolver
    //     (returns "red" — easy to distinguish).
    //   - Operator immediately edits max_batch_size, kicking off the
    //     debounced refetch which returns "green" promptly.
    //   - Then we resolve the slow first-paint response with "red".
    // Post-fix, fitSeqRef must short-circuit the slow write, leaving
    // the badge on "green". Pre-fix, the slow response would land last
    // and overwrite the fresh "green" answer with stale "red".
    let resolveSlow: (r: Response) => void = () => {};
    const slowPromise = new Promise<Response>((resolve) => {
      resolveSlow = resolve;
    });
    let fitCallCount = 0;
    installFetchStub({
      fitPreview: (body) => {
        fitCallCount += 1;
        if (fitCallCount === 1) {
          // First-paint — hold it.
          return slowPromise;
        }
        // Debounced refetch — return "green" immediately.
        const f = makeFit("green");
        // Carry the override through so we can assert this is the
        // fresh call, not a re-issue of the first-paint.
        if (typeof body.max_batch_size === "number") {
          f.breakdown.max_model_len_used = 4096;
        }
        return json(f);
      },
    });
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "meta-llama/Llama-3-8B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByTestId("file-table");

    // First-paint is in flight (hanging). Edit max_batch_size to kick the
    // debounced refetch. The fresh response lands, sets verdict=green.
    fireEvent.change(screen.getByTestId("max-batch-size"), {
      target: { value: "16" },
    });

    const badge = await screen.findByTestId("fit-badge-model.safetensors");
    await waitFor(
      () => {
        expect(badge).toHaveAttribute("data-verdict", "green");
      },
      { timeout: DEBOUNCE_WAIT_MS },
    );

    // NOW resolve the slow first-paint with a stale "red". fitSeqRef
    // must prevent it from overwriting the fresh "green".
    await act(async () => {
      resolveSlow(json(makeFit("red")));
      await Promise.resolve();
      await Promise.resolve();
    });

    // Badge must still be "green" — the stale write was discarded.
    expect(badge).toHaveAttribute("data-verdict", "green");
  });
});

// ---- #283 — pasted-value whitespace ----------------------------------------
//
// A repo id (or revision) pasted from the HuggingFace Hub UI carries
// incidental leading/trailing whitespace. The modal must trim it at the
// discover gate and forward the clean value to /api/models/discover;
// internal whitespace must still be rejected by the owner/name format
// check, and a clean value must pass through byte-identical.

describe("AddModelModal — #283 pasted-value whitespace", () => {
  function discoverParams(spy: FetchSpy): URLSearchParams {
    const call = spy.calls.find((c) => c.url.startsWith("/api/models/discover"));
    expect(call, "GET /api/models/discover was not called").toBeDefined();
    return new URLSearchParams(call!.url.split("?")[1]);
  }

  it("trims a leading space and discovers with the clean repo id", async () => {
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: " google/gemma-4-E4B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Discovery proceeds: file table appears, no format error.
    expect(await screen.findByTestId("file-table")).toBeInTheDocument();
    expect(screen.queryByText(/owner\/name/i)).not.toBeInTheDocument();
    expect(discoverParams(spy).get("repo_id")).toBe("google/gemma-4-E4B");
  });

  it("trims a trailing space and discovers with the clean repo id", async () => {
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "google/gemma-4-E4B " },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByTestId("file-table")).toBeInTheDocument();
    expect(screen.queryByText(/owner\/name/i)).not.toBeInTheDocument();
    expect(discoverParams(spy).get("repo_id")).toBe("google/gemma-4-E4B");
  });

  it("still rejects internal whitespace (not silently accepted)", async () => {
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "google /gemma-4-E4B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Trimming must not paper over an interior space — the format check
    // still refuses and discovery is never issued.
    expect(await screen.findByText(/owner\/name/i)).toBeInTheDocument();
    expect(
      spy.countByUrl((u) => u.startsWith("/api/models/discover")),
    ).toBe(0);
    expect(screen.queryByTestId("file-table")).not.toBeInTheDocument();
  });

  it("is unaffected by a clean value", async () => {
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "google/gemma-4-E4B" },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByTestId("file-table")).toBeInTheDocument();
    expect(discoverParams(spy).get("repo_id")).toBe("google/gemma-4-E4B");
  });

  it("trims a pasted revision before it reaches discover", async () => {
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "google/gemma-4-E4B" },
    });
    fireEvent.change(screen.getByLabelText(/hf revision/i), {
      target: { value: " v0.1 " },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByTestId("file-table")).toBeInTheDocument();
    expect(discoverParams(spy).get("revision")).toBe("v0.1");
  });

  it("falls back to main when the revision is whitespace only", async () => {
    const spy = installFetchStub({});
    render(<AddModelModal open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/hf repo/i), {
      target: { value: "google/gemma-4-E4B" },
    });
    fireEvent.change(screen.getByLabelText(/hf revision/i), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByRole("button", { name: /discover/i }));

    // Whitespace-only is empty after trim, so the existing `|| "main"`
    // fallback must apply (a bare `"   "` is truthy and would be sent as-is).
    expect(await screen.findByTestId("file-table")).toBeInTheDocument();
    expect(discoverParams(spy).get("revision")).toBe("main");
  });
});
