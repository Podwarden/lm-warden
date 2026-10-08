// /stats with a model selection.
//
// Operator report: the page listed the loaded models and then showed one
// combined set of numbers, with no way to ask about one of them. This file
// pins the control end of the fix — the selector's behaviour on the page, and
// that the request it produces actually narrows.
//
// The rule under test everywhere below: AT LEAST ONE MODEL STAYS SELECTED, and
// the last one is enforced by a DISABLED input rather than by silently
// re-ticking. A checkbox that unticks and springs back leaves the operator
// unable to tell whether their click registered at all, which is the same
// defect as an affordance gated on a different condition than its effect.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, fireEvent } from "@testing-library/react";
import { SWRConfig } from "swr";
import StatsPage from "@/app/stats/page";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";
import { __resetLiveStatsStreamForTests } from "@/lib/live-stats-stream";
import { MODEL_SELECTION_KEY } from "@/lib/model-selection";

// The session forest card (three.js, its own poller) is covered by forest-card.test.tsx.
vi.mock("@/components/forest/ForestCard", () => ({ default: () => null }));

class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const MODELS = [
  { id: "m-llama", served_model_name: "llama-3.1-8b", gpu_indices: [1] },
  { id: "m-qwen", served_model_name: "qwen3.8-27b", gpu_indices: [0] },
];

function overviewFor(url: string) {
  const params = new URL(url, "http://x").searchParams;
  const models = params.get("models");
  const selected = models ? models.split(",") : null;
  // Numbers that differ per selection, so an assertion can tell whether the
  // page actually re-fetched rather than re-rendering the same payload.
  const tps = selected === null ? 11 : selected.length === 1 ? 5 : 11;
  return {
    range: "1h",
    now_minute: 31538880,
    since_minute: 31538820,
    selected_model_ids: selected,
    selected_gpu_indices: selected === null ? null : [0],
    current: {
      vram_used_mib: 12000,
      vram_total_mib: 32000,
      vram_pct: 38,
      gpu_util_pct: 80,
      power_w: 250,
      tps,
    },
    active_models: MODELS,
    series: { vram: [], util: [], power: [], tokens: [] },
  };
}

function installFetchStub() {
  const urls: string[] = [];
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    urls.push(url);
    if (url === "/api/auth/refresh") return json({ access_token: "t" });
    if (url === "/api/csrf") return json({ csrf: "c" });
    if (url.startsWith("/api/stats/v2/overview")) return json(overviewFor(url));
    if (url.startsWith("/api/stats/v2/tokens-per-key")) {
      return json({ range: "1h", since_minute: 0, rows: [] });
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return urls;
}

function renderPage() {
  return render(
    <SWRConfig
      value={{
        provider: () => new Map(),
        dedupingInterval: 0,
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
      }}
    >
      <StatsPage />
    </SWRConfig>,
  );
}

function optionFor(id: string): HTMLElement {
  const el = screen
    .getAllByTestId("model-selector-option")
    .find((o) => o.getAttribute("data-model-id") === id);
  if (!el) throw new Error(`no option for ${id}`);
  return el;
}

function boxFor(id: string): HTMLInputElement {
  return optionFor(id).querySelector("input") as HTMLInputElement;
}

describe("StatsPage — model selection", () => {
  beforeEach(() => {
    // Re-stub after vitest's per-test global clear (unstubGlobals: true),
    // which runs before this hook: the page's recharts panels need a
    // ResizeObserver or they render nothing.
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    setAccessToken("test-jwt");
    setCsrfToken("test-csrf");
    window.localStorage.clear();
  });
  afterEach(() => {
    cleanup();
    // R19: this file renders the page with the REAL useLiveStats singleton
    // and a real access token. Without teardown, the module's stream and the
    // token leak into the next test — the async residue that made
    // `vitest:frontend` flaky. Reset both so no test inherits the previous
    // test's stream or token.
    __resetLiveStatsStreamForTests();
    setAccessToken(null);
  });

  it("offers every loaded model, all selected by default", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    expect(boxFor("m-llama").checked).toBe(true);
    expect(boxFor("m-qwen").checked).toBe(true);
  });

  it("narrows the request to one model when the other is unticked", async () => {
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() =>
      expect(urls.some((u) => u.includes("models=m-llama"))).toBe(true),
    );
    // ...and specifically NOT a request naming both.
    const last = urls.filter((u) => u.startsWith("/api/stats/v2/overview")).pop()!;
    expect(last).toContain("models=m-llama");
    expect(last).not.toContain("m-qwen");
  });

  it("supports any combination, not just one or all", async () => {
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() => expect(boxFor("m-qwen").checked).toBe(false));
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() => expect(boxFor("m-qwen").checked).toBe(true));
    expect(
      urls.some((u) => u.includes("models=m-llama%2Cm-qwen")),
    ).toBe(true);
  });

  it("DISABLES the last selected model rather than letting it be unticked", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() => expect(boxFor("m-llama").disabled).toBe(true));
    expect(optionFor("m-llama").getAttribute("data-locked")).toBe("true");
    // The disabled state carries its own explanation -- the control says the
    // rule instead of the operator having to infer it from a dead click.
    expect(optionFor("m-llama").getAttribute("title")).toMatch(
      /at least one model/i,
    );
  });

  it("cannot be driven to an empty selection even by clicking anyway", async () => {
    // jsdom will happily dispatch a click at a disabled input if a test asks
    // it to. The state must refuse regardless -- `disabled` is the honest
    // signal, not the enforcement.
    const urls = installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() => expect(boxFor("m-llama").disabled).toBe(true));
    fireEvent.click(boxFor("m-llama"));
    fireEvent.click(boxFor("m-llama"));
    expect(boxFor("m-llama").checked).toBe(true);
    // And no request was ever made with an empty selection, which the API
    // rejects with a 400.
    expect(urls.every((u) => !u.includes("models=&") && !u.endsWith("models="))).toBe(
      true,
    );
  });

  it("says what the numbers cover, from the response not the checkboxes", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() => {
      const caption = screen.getByTestId("stats-covering");
      expect(caption.textContent).toContain("llama-3.1-8b");
      expect(caption.textContent).not.toContain("qwen3.8-27b");
    });
  });

  it("says the host figures do NOT narrow with the selection", async () => {
    // The settled rule on the merged page: the selection scopes tokens,
    // latency, KV and requests. VRAM, GPU utilisation and power stay HOST
    // figures — two engines share a card, so watts cannot be attributed to
    // one model — and the caption says so instead of silently re-scoping.
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() =>
      expect(screen.getByTestId("stats-covering").textContent).toMatch(
        /stay host figures/i,
      ),
    );
  });

  it("says the per-key table did NOT narrow, because it cannot", async () => {
    // token_usage_minute is keyed by API token and has no model column, so
    // per-model per-key usage is unanswerable. Leaving the table looking as
    // though it narrowed with everything else above it would be the lie.
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() =>
      expect(screen.getByTestId("tokens-per-key-scope").textContent).toMatch(
        /all models/i,
      ),
    );
  });

  it("keeps a host key unfiltered beside the scoped one, and the selector offers the fleet", async () => {
    // The merged page holds FOUR overview keys, by design: the selected
    // window unfiltered (host charts + the selector's own option list), the
    // selected window scoped (tokens/tps), and the same pair at a fixed 24h
    // for the reference lines. active_models is deliberately NOT narrowed by
    // ?models= -- the selector is built from it, so narrowing would make a
    // deselected model impossible to re-select.
    const urls = installFetchStub();
    renderPage();
    // Wait until the selection has resolved and produced the scoped keys.
    await waitFor(() =>
      expect(urls.some((u) => u.includes("models="))).toBe(true),
    );
    const overviewCalls = urls.filter((u) =>
      u.startsWith("/api/stats/v2/overview"),
    );
    // One call per distinct key, nothing redundant.
    expect(overviewCalls.length).toBe(new Set(overviewCalls).size);
    // The host keys never carry a selection; the scoped keys always do.
    expect(overviewCalls.some((u) => !u.includes("models="))).toBe(true);
    expect(overviewCalls[0]).not.toContain("models=");
    // ...and the selector still offers the whole fleet.
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
  });

  it("keeps offering a model that the selection excludes", async () => {
    // The dead end this avoids: a selector whose options shrink to its own
    // selection can never be widened again.
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() => expect(boxFor("m-qwen").checked).toBe(false));
    expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2);
    expect(boxFor("m-qwen").disabled).toBe(false);
  });

  it("remembers the selection for the next page load", async () => {
    installFetchStub();
    renderPage();
    await waitFor(() =>
      expect(screen.getAllByTestId("model-selector-option")).toHaveLength(2),
    );
    fireEvent.click(boxFor("m-qwen"));
    await waitFor(() =>
      expect(window.localStorage.getItem(MODEL_SELECTION_KEY)).toBe(
        JSON.stringify(["m-llama"]),
      ),
    );
  });
});
