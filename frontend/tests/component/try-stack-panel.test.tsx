// TryStackHistory (#162) — presentational attempt list on the model detail
// try-stack panel.
//
// Pins the contract from the backend GET /api/models/{id}/try-stack shape:
//   - the engine combo (channel + vLLM version) renders per attempt
//   - a failed attempt surfaces its error and the classifier's suggested next
//     combo (suggested_next.suggestion)
//   - an ok attempt renders without a suggestion (nothing to suggest)
//
// The container (TryStackPanel) wraps the SWR fetch + POST round-trips; this
// suite drives the pure presentational export directly, no fetch stub needed.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import {
  TryStackHistory,
  TryStackPanel,
  type StackAttempt,
} from "@/components/models/try-stack-panel";
import type { ModelStatus } from "@/components/models/model-card";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const FAILED: StackAttempt = {
  id: "a1",
  channel: "cuda-stable",
  vllm_version: "0.20.0",
  image: "vllm/vllm-openai:v0.20.0",
  result: "failed",
  error: "oom",
  category: "oom",
  suggested_next: { suggestion: "lower gpu_memory_utilization" },
  created_at: "now",
};

const OK: StackAttempt = {
  id: "a2",
  channel: "cuda-nightly",
  vllm_version: "0.21.0",
  image: "vllm/vllm-openai:nightly",
  result: "ok",
  error: null,
  category: null,
  suggested_next: null,
  created_at: "later",
};

describe("TryStackHistory", () => {
  afterEach(cleanup);

  it("renders attempts with result + suggestion", () => {
    render(<TryStackHistory attempts={[FAILED]} />);
    expect(screen.getByText(/cuda-stable/)).toBeInTheDocument();
    expect(screen.getByText(/0\.20\.0/)).toBeInTheDocument();
    expect(screen.getByText(/oom/)).toBeInTheDocument();
    expect(screen.getByText(/lower gpu_memory_utilization/)).toBeInTheDocument();
  });

  it("renders an ok attempt without a suggestion", () => {
    render(<TryStackHistory attempts={[OK]} />);
    expect(screen.getByText(/cuda-nightly/)).toBeInTheDocument();
    expect(screen.queryByText(/Suggested next/i)).toBeNull();
  });

  it("renders an empty-state hint when there are no attempts", () => {
    render(<TryStackHistory attempts={[]} />);
    expect(screen.getByText(/no attempts/i)).toBeInTheDocument();
  });
});

// TryStackPanel container — the try / report / save round-trips.
//
// The defect this pins: a "pending" attempt must be reportable as ok/failed
// THROUGH THE UI (POST /api/models/{id}/try-stack/{attempt_id}), otherwise the
// "Save working combo as template" button (gated on result === "ok") is
// unreachable. Network is stubbed at the global `fetch` boundary so the
// auth-fetch wrapper is exercised end-to-end; SWR cache is isolated per test
// via a fresh provider Map.

const PENDING: StackAttempt = {
  id: "att-1",
  channel: "cuda-stable",
  vllm_version: "0.20.0",
  image: "vllm/vllm-openai:v0.20.0",
  result: "pending",
  error: null,
  category: null,
  suggested_next: null,
  created_at: "now",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface Captured {
  url: string;
  method: string;
  body: Record<string, unknown> | null;
}

// Stub GET try-stack (returns `attempts`) + POST report. The GET result flips
// to `ok` after a successful report so the panel re-renders the save button.
// Accepts an optional `initial` list so dup-warning tests can seed a failed
// history without first driving the round-trip.
function installTryStackStub(
  modelId: string,
  initial: StackAttempt[] = [PENDING],
) {
  const calls: Captured[] = [];
  let attempts: StackAttempt[] = [...initial];
  const listUrl = `/api/models/${modelId}/try-stack`;
  const loadUrl = `/api/models/${modelId}/load`;
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(init.body as string) : null;
    calls.push({ url, method, body });
    if (url === "/api/auth/refresh") return json({ access_token: "test-jwt-refreshed" });
    if (url === "/api/csrf") return json({ csrf: "test-csrf" });
    // #177: driver capability — docker driver honors the pin, so the version
    // selector is enabled (the default for these existing round-trip tests).
    if (url === "/api/system/backends") {
      return json(capableBackends("0.21.0"));
    }
    // #177: the version field is now a combobox backed by this endpoint.
    if (url.startsWith("/api/templates/engine-versions")) {
      return json({
        channel: "cuda-stable",
        family: "vllm/vllm-openai",
        versions: ["0.21.0", "0.20.0"],
        error: null,
      });
    }
    if (url === listUrl && method === "GET") return json({ attempts });
    if (url === listUrl && method === "POST") {
      const attemptId = `att-${attempts.length + 1}`;
      attempts = [
        ...attempts,
        {
          id: attemptId,
          channel: body?.channel ?? "cuda-stable",
          vllm_version: body?.vllm_version ?? "0.20.0",
          image: `vllm/vllm-openai:${body?.vllm_version ?? "0.20.0"}`,
          result: "pending",
          error: null,
          category: null,
          suggested_next: null,
          created_at: "now",
        },
      ];
      return json({ attempt_id: attemptId, image: `vllm/vllm-openai:${body?.vllm_version ?? "0.20.0"}` }, 201);
    }
    if (url === loadUrl && method === "POST") return json({ ok: true }, 202);
    if (url === `${listUrl}/att-1` && method === "POST") {
      attempts = [{ ...PENDING, result: body?.result === "ok" ? "ok" : "failed", error: body?.error ?? null }];
      return json({ ok: true });
    }
    if (url === "/api/models/templates" && method === "POST") return json({ id: "saved" }, 201);
    throw new Error(`Unmocked fetch: ${method} ${url}`);
  });
  vi.stubGlobal("fetch", mock);
  return { calls };
}

/** A GET /api/system/backends body where the pin IS available for vLLM. */
function capableBackends(version: string | null) {
  return {
    default: "vllm",
    driver: "docker",
    engine_version: version,
    backends: [
      {
        name: "vllm",
        display_name: "vLLM",
        version,
        supports_version_pin: true,
        version_pin_available: true,
        version_pin_reason: null,
        version_pin_reason_code: null,
      },
    ],
  };
}

function renderPanel(modelId: string, modelStatus: ModelStatus = "pulled") {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <TryStackPanel
        modelId={modelId}
        hfRepo="openai/gpt-oss-20b"
        maxModelLen={4096}
        tensorParallelSize={2}
        modelStatus={modelStatus}
        backend="vllm"
      />
    </SWRConfig>,
  );
}

describe("TryStackPanel report round-trip", () => {
  beforeEach(() => {
    setAccessToken("test-jwt");
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("offers a report control while the latest attempt is pending", async () => {
    installTryStackStub("m1");
    renderPanel("m1");
    await waitFor(() => expect(screen.getByTestId("try-stack-report")).toBeInTheDocument());
    expect(screen.getByTestId("try-stack-report-ok")).toBeInTheDocument();
    expect(screen.getByTestId("try-stack-report-failed")).toBeInTheDocument();
    // Save-as-template is NOT offered yet (attempt is still pending).
    expect(screen.queryByTestId("try-stack-save-template")).toBeNull();
  });

  it("POSTs result=ok to the attempt endpoint and then reveals save-as-template", async () => {
    const stub = installTryStackStub("m2");
    renderPanel("m2");
    await waitFor(() => expect(screen.getByTestId("try-stack-report-ok")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("try-stack-report-ok"));

    await waitFor(() => {
      const post = stub.calls.find(
        (c) => c.url === "/api/models/m2/try-stack/att-1" && c.method === "POST",
      );
      expect(post).toBeTruthy();
      expect(post?.body).toEqual({ result: "ok" });
    });
    // Once the attempt is ok, the save-as-template control appears and the
    // pending report control is gone.
    await waitFor(() => expect(screen.getByTestId("try-stack-save-template")).toBeInTheDocument());
    expect(screen.queryByTestId("try-stack-report")).toBeNull();
  });

  it("includes the failure detail when reporting result=failed", async () => {
    const stub = installTryStackStub("m3");
    renderPanel("m3");
    await waitFor(() => expect(screen.getByTestId("try-stack-report-error")).toBeInTheDocument());
    fireEvent.change(screen.getByTestId("try-stack-report-error"), {
      target: { value: "CUDA OOM at load" },
    });
    fireEvent.click(screen.getByTestId("try-stack-report-failed"));

    await waitFor(() => {
      const post = stub.calls.find(
        (c) => c.url === "/api/models/m3/try-stack/att-1" && c.method === "POST",
      );
      expect(post?.body).toEqual({ result: "failed", error: "CUDA OOM at load" });
    });
  });
});

// #177: the vLLM-version field is a typeable dropdown populated from the
// published image-resolving versions (GET /api/templates/engine-versions),
// but it MUST remain free-text — an operator may need an unpublished version
// or a pinned digest the catalog won't list. This pins both halves: the
// fetched options surface AND a typed value still drives the Try button.
describe("TryStackPanel vLLM-version combobox (#177)", () => {
  beforeEach(() => {
    setAccessToken("test-jwt");
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("offers fetched versions as suggestions yet accepts a typed value", async () => {
    installTryStackStub("mv");
    renderPanel("mv");

    const field = await screen.findByTestId("try-stack-version");
    // Try is disabled until a non-empty version is present.
    expect(screen.getByTestId("try-stack-submit")).toBeDisabled();

    // Focusing surfaces the fetched options (0.21.0 / 0.20.0).
    fireEvent.focus(field);
    await waitFor(() => expect(screen.getByText("0.21.0")).toBeInTheDocument());
    expect(screen.getByText("0.20.0")).toBeInTheDocument();

    // Picking a suggestion commits it and enables Try.
    fireEvent.mouseDown(screen.getByText("0.20.0"));
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-submit")).not.toBeDisabled(),
    );
    expect((field as HTMLInputElement).value).toBe("0.20.0");

    // The field is still free-text: a version NOT in the fetched list is
    // accepted verbatim and keeps Try enabled (escape hatch).
    fireEvent.change(field, { target: { value: "0.99.0-unpublished" } });
    expect((field as HTMLInputElement).value).toBe("0.99.0-unpublished");
    expect(screen.getByTestId("try-stack-submit")).not.toBeDisabled();
  });

  it("works as plain free-text when no versions are available", async () => {
    // Override the engine-versions stub to an empty list (non-resolvable
    // channel / Docker Hub hiccup): the field still drives the Try button.
    const modelId = "mv2";
    const listUrl = `/api/models/${modelId}/try-stack`;
    const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      if (url === "/api/auth/refresh") return json({ access_token: "test-jwt-refreshed" });
      if (url === "/api/csrf") return json({ csrf: "test-csrf" });
      if (url === "/api/system/backends") {
        return json(capableBackends(null));
      }
      if (url.startsWith("/api/templates/engine-versions")) {
        return json({ channel: "cuda-stable", family: null, versions: [], error: null });
      }
      if (url === listUrl && method === "GET") return json({ attempts: [] });
      throw new Error(`Unmocked fetch: ${method} ${url}`);
    });
    vi.stubGlobal("fetch", mock);

    renderPanel(modelId);
    const field = await screen.findByTestId("try-stack-version");
    expect(screen.getByTestId("try-stack-submit")).toBeDisabled();
    fireEvent.change(field, { target: { value: "0.20.0" } });
    expect((field as HTMLInputElement).value).toBe("0.20.0");
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-submit")).not.toBeDisabled(),
    );
  });
});

// #177 driver-capability guard: when the deployment runs the in-container
// subprocess engine (GET /api/system/engine -> supports_version_select:false),
// the version selector cannot take effect — the supervisor refuses the pin.
// The panel must DISABLE the controls and show an explanatory note instead of
// letting the operator pin a version that won't take effect.
describe("TryStackPanel driver-capability guard (#177)", () => {
  beforeEach(() => {
    setAccessToken("test-jwt");
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  // The panel reads /api/system/backends, not /api/system/engine. The engine
  // route answers only for vLLM and only about the DRIVER; the backends route
  // answers per backend resolved against the active driver, and carries the
  // sentence that explains a refusal -- which the panel now renders verbatim
  // instead of writing its own copy about a driver it cannot identify.
  function installEngineStub(opts: {
    supports: boolean;
    vllmVersion: string | null;
    withEngine?: boolean; // false => never answer /api/system/backends (loading)
  }) {
    const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      if (url === "/api/auth/refresh") return json({ access_token: "test-jwt-refreshed" });
      if (url === "/api/csrf") return json({ csrf: "test-csrf" });
      if (url === "/api/system/backends") {
        if (opts.withEngine === false) return new Promise<Response>(() => {}); // never resolves
        return json({
          default: "vllm",
          driver: opts.supports ? "docker" : "subprocess",
          engine_version: opts.vllmVersion,
          backends: [
            {
              name: "vllm",
              display_name: "vLLM",
              version: opts.vllmVersion,
              supports_version_pin: true,
              version_pin_available: opts.supports,
              version_pin_reason: opts.supports
                ? null
                : "This deployment runs the in-container engine driver: the " +
                  "engine version is fixed by the warden image, so a pin is " +
                  "refused rather than applied. Version selection requires " +
                  "the docker engine driver (VW_ENGINE_DRIVER=docker).",
              version_pin_reason_code: opts.supports ? null : "driver",
            },
          ],
        });
      }
      if (url.startsWith("/api/templates/engine-versions")) {
        return json({ channel: "cuda-stable", family: "vllm/vllm-openai", versions: ["0.20.0"], error: null });
      }
      if (url.startsWith("/api/models/") && method === "GET") return json({ attempts: [] });
      throw new Error(`Unmocked fetch: ${method} ${url}`);
    });
    vi.stubGlobal("fetch", mock);
  }

  it("disables the version selector and shows a note when version-select is unsupported", async () => {
    installEngineStub({ supports: false, vllmVersion: "0.20.0" });
    renderPanel("ms");

    await waitFor(() =>
      expect(screen.getByTestId("try-stack-driver-note")).toBeInTheDocument(),
    );
    // The note names the baked vLLM version.
    expect(screen.getByTestId("try-stack-driver-note").textContent).toMatch(/0\.20\.0/);
    // Controls are disabled.
    expect(screen.getByTestId("try-stack-version")).toBeDisabled();
    expect(screen.getByTestId("try-stack-channel")).toBeDisabled();
    expect(screen.getByTestId("try-stack-submit")).toBeDisabled();
  });

  it("keeps controls enabled and hides the note when version-select is supported", async () => {
    installEngineStub({ supports: true, vllmVersion: "0.21.0" });
    renderPanel("md");

    // Wait until we know the driver is capable, then assert the note is absent.
    await waitFor(() => expect(screen.getByTestId("try-stack-channel")).not.toBeDisabled());
    expect(screen.queryByTestId("try-stack-driver-note")).toBeNull();
    expect(screen.getByTestId("try-stack-channel")).not.toBeDisabled();
    // (Try stays disabled only because no version is typed yet — not the guard.)
    expect(screen.getByTestId("try-stack-version")).not.toBeDisabled();
  });

  it("never shows a version the deployment is not running", async () => {
    // Operator report: the panel's note read "(vLLM 0.26.0)" while the field
    // one line below showed 0.20.0 as its placeholder. That placeholder was a
    // hardcoded literal carried over from the template registry's default, so
    // the panel contradicted itself on every deployment whose engine was not
    // that exact version -- and on this driver the field is disabled, so the
    // operator can never type over it.
    installEngineStub({ supports: false, vllmVersion: "0.26.0" });
    renderPanel("mp");

    await waitFor(() =>
      expect(screen.getByTestId("try-stack-driver-note")).toBeInTheDocument(),
    );
    const field = screen.getByTestId("try-stack-version") as HTMLInputElement;
    expect(screen.getByTestId("try-stack-driver-note").textContent).toMatch(/0\.26\.0/);
    expect(field.placeholder).toBe("0.26.0");
  });

  it("falls back to no placeholder rather than inventing a version", async () => {
    // vllm_version is null when importlib cannot find the package. An empty
    // placeholder says nothing; a stale literal says something false.
    installEngineStub({ supports: true, vllmVersion: null });
    renderPanel("mq");

    const field = (await screen.findByTestId("try-stack-version")) as HTMLInputElement;
    await waitFor(() => expect(field.placeholder).toBe(""));
  });

  it("does not disable while engine info is still loading (no flash)", async () => {
    installEngineStub({ supports: false, vllmVersion: null, withEngine: false });
    renderPanel("ml");

    // The channel select renders immediately; while engine info is undefined we
    // must NOT disable it (avoids a disabled flash for the common capable case).
    const channel = await screen.findByTestId("try-stack-channel");
    expect(channel).not.toBeDisabled();
    expect(screen.queryByTestId("try-stack-driver-note")).toBeNull();
  });
});

// Unified Load — the panel button now both pins the combo AND starts the
// engine in one click. This block pins the sequencing + the engine-busy
// guard + the duplicate-failed-attempt warning + the Retry link.
describe("TryStackPanel unified Load", () => {
  beforeEach(() => {
    setAccessToken("test-jwt");
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("fires POST try-stack then POST /load in order on a Load click", async () => {
    const stub = installTryStackStub("u1", []);
    renderPanel("u1");

    const field = await screen.findByTestId("try-stack-version");
    fireEvent.change(field, { target: { value: "0.20.0" } });
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-submit")).not.toBeDisabled(),
    );
    fireEvent.click(screen.getByTestId("try-stack-submit"));

    await waitFor(() => {
      const posts = stub.calls.filter((c) => c.method === "POST");
      expect(posts).toHaveLength(2);
      expect(posts[0].url).toBe("/api/models/u1/try-stack");
      expect(posts[0].body).toEqual({ channel: "cuda-stable", vllm_version: "0.20.0" });
      expect(posts[1].url).toBe("/api/models/u1/load");
    });
  });

  it("disables Load and shows a busy note while the model is loaded", async () => {
    installTryStackStub("u2", []);
    renderPanel("u2", "loaded");

    const field = await screen.findByTestId("try-stack-version");
    fireEvent.change(field, { target: { value: "0.20.0" } });
    // Even with a valid version, the engine-busy lock keeps Load disabled.
    expect(screen.getByTestId("try-stack-submit")).toBeDisabled();
    const note = screen.getByTestId("try-stack-busy-note");
    expect(note.textContent).toMatch(/loaded/);
    expect(note.textContent).toMatch(/unload/i);
  });

  it("disables Load while the model is loading or unloading", async () => {
    installTryStackStub("u2b", []);
    const { rerender } = renderPanel("u2b", "loading");
    const field = await screen.findByTestId("try-stack-version");
    fireEvent.change(field, { target: { value: "0.20.0" } });
    expect(screen.getByTestId("try-stack-submit")).toBeDisabled();
    expect(screen.getByTestId("try-stack-busy-note").textContent).toMatch(/loading/);

    rerender(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <TryStackPanel
          modelId="u2b"
          hfRepo="r"
          maxModelLen={null}
          tensorParallelSize={null}
          modelStatus="unloading"
          backend="vllm"
        />
      </SWRConfig>,
    );
    await screen.findByTestId("try-stack-version");
    expect(screen.getByTestId("try-stack-submit")).toBeDisabled();
    expect(screen.getByTestId("try-stack-busy-note").textContent).toMatch(/unloading/);
  });

  it("warns before re-submitting a previously-failed combo and gates the first click", async () => {
    const failed: StackAttempt = {
      id: "att-1",
      channel: "cuda-stable",
      vllm_version: "0.20.0",
      image: "vllm/vllm-openai:v0.20.0",
      result: "failed",
      error: "CUDA OOM",
      category: "oom",
      suggested_next: null,
      created_at: "2026-05-27T03:00:00Z",
    };
    const stub = installTryStackStub("u3", [failed]);
    renderPanel("u3");

    const field = await screen.findByTestId("try-stack-version");
    fireEvent.change(field, { target: { value: "0.20.0" } });

    // The warning surfaces once the selected combo matches the failed row.
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-dup-warning")).toBeInTheDocument(),
    );
    const banner = screen.getByTestId("try-stack-dup-warning");
    expect(banner.textContent).toMatch(/0\.20\.0/);
    expect(banner.textContent).toMatch(/CUDA OOM/);
    expect(banner.textContent).toMatch(/2026-05-27/);

    // First click — arms the confirm gate, does NOT submit.
    expect(screen.getByTestId("try-stack-submit")).not.toBeDisabled();
    fireEvent.click(screen.getByTestId("try-stack-submit"));
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-dup-confirm")).toBeInTheDocument(),
    );
    expect(stub.calls.filter((c) => c.method === "POST")).toHaveLength(0);

    // Second click — submits the pair.
    fireEvent.click(screen.getByTestId("try-stack-submit"));
    await waitFor(() => {
      const posts = stub.calls.filter((c) => c.method === "POST");
      expect(posts.map((p) => p.url)).toEqual([
        "/api/models/u3/try-stack",
        "/api/models/u3/load",
      ]);
    });
  });

  it("clears the dup-warning + confirm state when the operator changes channel or version", async () => {
    const failed: StackAttempt = {
      id: "att-1",
      channel: "cuda-stable",
      vllm_version: "0.20.0",
      image: null,
      result: "failed",
      error: "boom",
      category: null,
      suggested_next: null,
      created_at: "now",
    };
    installTryStackStub("u4", [failed]);
    renderPanel("u4");

    const field = await screen.findByTestId("try-stack-version");
    fireEvent.change(field, { target: { value: "0.20.0" } });
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-dup-warning")).toBeInTheDocument(),
    );
    // Arm the confirm gate.
    fireEvent.click(screen.getByTestId("try-stack-submit"));
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-dup-confirm")).toBeInTheDocument(),
    );

    // Changing the typed version to one with no matching failure clears
    // both the banner and the confirm prompt.
    fireEvent.change(field, { target: { value: "0.21.0" } });
    await waitFor(() =>
      expect(screen.queryByTestId("try-stack-dup-warning")).toBeNull(),
    );
    expect(screen.queryByTestId("try-stack-dup-confirm")).toBeNull();
  });

  it("offers Retry on history rows that prefills the picker", async () => {
    const failed: StackAttempt = {
      id: "att-9",
      channel: "cuda-edge",
      vllm_version: "0.21.0",
      image: null,
      result: "failed",
      error: "kernel mismatch",
      category: null,
      suggested_next: null,
      created_at: "now",
    };
    installTryStackStub("u5", [failed]);
    renderPanel("u5");

    const retry = await screen.findByTestId("try-stack-retry-att-9");
    fireEvent.click(retry);

    // The picker now holds the row's combo: the version field shows 0.21.0
    // and the channel select is cuda-edge.
    await waitFor(() => {
      const field = screen.getByTestId("try-stack-version") as HTMLInputElement;
      expect(field.value).toBe("0.21.0");
    });
    const ch = screen.getByTestId("try-stack-channel") as HTMLSelectElement;
    expect(ch.value).toBe("cuda-edge");
    // Because the prefilled combo matches a failed row, the dup-warning is
    // immediately visible (a natural consequence of the Retry, not a
    // separate hook).
    expect(screen.getByTestId("try-stack-dup-warning")).toBeInTheDocument();
  });

  it("hides Retry on the latest attempt while it is still pending", async () => {
    installTryStackStub("u6", [PENDING]);
    renderPanel("u6");

    await screen.findByTestId("try-stack-history");
    // PENDING is the latest pending attempt — no Retry link on it.
    expect(screen.queryByTestId("try-stack-retry-att-1")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// #262/#265 — pinning an engine goes through the model write path now, so this
// panel can receive FastAPI's LIST-shaped 422 detail as well as the string and
// {message} shapes it already saw. `String(d.detail)` on a list renders
// `[object Object]`, which is what the operator was shown.
// ---------------------------------------------------------------------------

describe("TryStackPanel error details", () => {
  beforeEach(() => {
    setAccessToken("test-jwt");
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  /** Like installTryStackStub, but the try-stack POST fails with `failure`. */
  function installFailingPin(modelId: string, failure: Response) {
    const listUrl = `/api/models/${modelId}/try-stack`;
    const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      if (url === "/api/auth/refresh") return json({ access_token: "test-jwt-refreshed" });
      if (url === "/api/csrf") return json({ csrf: "test-csrf" });
      if (url === "/api/system/backends") return json(capableBackends("0.21.0"));
      if (url.startsWith("/api/templates/engine-versions")) {
        return json({
          channel: "cuda-stable",
          family: "vllm/vllm-openai",
          versions: ["0.21.0", "0.20.0"],
          error: null,
        });
      }
      if (url === listUrl && method === "GET") return json({ attempts: [] });
      if (url === listUrl && method === "POST") return failure.clone();
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal("fetch", mock);
  }

  async function pin(modelId: string) {
    renderPanel(modelId);
    const field = await screen.findByTestId("try-stack-version");
    fireEvent.change(field, { target: { value: "0.20.0" } });
    await waitFor(() =>
      expect(screen.getByTestId("try-stack-submit")).not.toBeDisabled(),
    );
    fireEvent.click(screen.getByTestId("try-stack-submit"));
    return screen.findByTestId("try-stack-error");
  }

  it("renders a list-shaped 422 detail rather than [object Object]", async () => {
    installFailingPin(
      "e1",
      new Response(
        JSON.stringify({
          detail: [
            { loc: [], msg: "Value error, extra_env key 'FOO_BAR' is not in the allowlist", type: "value_error" },
          ],
        }),
        { status: 422 },
      ),
    );
    const err = await pin("e1");
    expect(err.textContent).toMatch(/FOO_BAR/);
    expect(err.textContent).not.toMatch(/object Object/);
  });

  it("still renders a string detail", async () => {
    installFailingPin(
      "e2",
      new Response(
        JSON.stringify({ detail: "model must be unloaded before editing settings" }),
        { status: 409 },
      ),
    );
    const err = await pin("e2");
    expect(err.textContent).toMatch(/must be unloaded/);
  });

  it("renders a {message} detail", async () => {
    installFailingPin(
      "e3",
      new Response(
        JSON.stringify({ detail: { error_code: "session_only", message: "needs a signed-in session" } }),
        { status: 403 },
      ),
    );
    const err = await pin("e3");
    expect(err.textContent).toMatch(/needs a signed-in session/);
  });
});
