// Component tests for the rebuilt /stats page (S7, #124).
//
// Drives the page through the SWR-backed fetch boundary the same way the
// rest of the suite does (see cache-table.test.tsx). We stub `fetch`,
// answer the two /api/stats/v2/* endpoints with documented fixtures, and
// assert the current-row tiles + active-model strip + per-key table
// populate correctly. Recharts itself is rendered into jsdom under a
// ResizeObserver stub — we deliberately do NOT assert on chart contents
// (those are exercised by the underlying recharts test suite); we only
// pin that the page mounts and the chart panels are present.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  cleanup,
  waitFor,
  fireEvent,
  act,
  within,
} from "@testing-library/react";
import { SWRConfig } from "swr";
import StatsPage from "@/app/stats/page";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

// SWR maintains a module-level cache that persists across renders within
// the same vitest worker — so a payload supplied by test A bleeds into
// test B. Wrap each render in a fresh SWRConfig provider so the cache is
// per-test. We also disable revalidate-on-mount so the wait-for-fetch
// path is deterministic.
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

// Recharts uses ResizeObserver inside ResponsiveContainer; jsdom has no
// implementation, so chart components silently render nothing. Stub a
// no-op so the page mounts cleanly under the test runner.
class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", FakeResizeObserver);

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface FixtureSet {
  overview?: unknown;
  tokensPerKey?: unknown;
  throughput?: unknown;
}

function installFetchStub(fixtures: FixtureSet) {
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") {
      return json({ access_token: "test-jwt-refreshed" });
    }
    if (url === "/api/csrf") {
      return json({ csrf: "test-csrf" });
    }
    if (url.startsWith("/api/stats/v2/overview")) {
      return json(fixtures.overview ?? {});
    }
    if (url.startsWith("/api/stats/v2/throughput")) {
      return json(fixtures.throughput ?? FIXTURE_THROUGHPUT);
    }
    if (url.startsWith("/api/stats/v2/tokens-per-key")) {
      return json(fixtures.tokensPerKey ?? { range: "1h", since_minute: 0, rows: [] });
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

const FIXTURE_OVERVIEW = {
  range: "1h",
  now_minute: 31538880,
  since_minute: 31538820,
  current: {
    vram_used_mib: 12000,
    vram_total_mib: 32000,
    vram_pct: 38,
    gpu_util_pct: 80,
    power_w: 250.0,
    tps: 14.0,
  },
  active_models: [
    { id: "m-active", served_model_name: "served-name-1" },
  ],
  series: {
    vram: [{ minute: 31538879, used_mib: 3000, total_mib: 32000 }],
    util: [{ minute: 31538879, max_pct: 50 }],
    power: [{ minute: 31538879, watts: 230.0 }],
    tokens: [{ minute: 31538879, prompt: 1000, completion: 500 }],
  },
};

const FIXTURE_THROUGHPUT = {
  basis: "request",
  range: "1h",
  since_epoch: 1_757_000_000,
  selected_model_ids: null,
  prefill: { count: 1284, avg: 412.4, max: 980.0, mode: 380.2 },
  generation: { count: 1284, avg: 48.6, max: 71.3, mode: 46.1 },
  coverage: {
    earliest_epoch: 1_756_000_000,
    retention_days: 30,
    max_rows: 200000,
    covers_window: true,
  },
};

const FIXTURE_TPK = {
  range: "1h",
  since_minute: 31538820,
  rows: [
    {
      token_id: "t-heavy",
      name: "Heavy Key",
      prefix: "pwm_aaa",
      requests: 15,
      prompt_tokens: 1500,
      completion_tokens: 750,
      total_tokens: 2250,
    },
    {
      token_id: "t-orphan",
      name: "(unknown)",
      prefix: null,
      requests: 1,
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    },
  ],
};

describe("StatsPage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("has no God Mode link (the token page's dock replaced /godmode)", () => {
    installFetchStub({ overview: FIXTURE_OVERVIEW, tokensPerKey: FIXTURE_TPK });
    renderPage();

    expect(screen.queryByTestId("godmode-link")).toBeNull();
    expect(screen.queryByRole("link", { name: /god mode/i })).toBeNull();
  });

  it("renders the three host tiles populated from the overview payload", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument();
    });

    // VRAM tile: 12000/32000 MiB → 11.7 / 31.3 GiB
    expect(screen.getByTestId("tile-vram-value").textContent).toContain("11.7");
    expect(screen.getByTestId("tile-vram-value").textContent).toContain("31.3");

    // GPU util tile
    expect(screen.getByTestId("tile-util-value").textContent).toBe("80");

    // Power tile — 250W → "250" (≥100 → integer)
    expect(screen.getByTestId("tile-power-value").textContent).toBe("250");

    // No blended tokens-per-second tile: prefill and generation move
    // independently, so their average described neither and has been replaced
    // by the throughput panel below.
    expect(screen.queryByTestId("tile-tps-value")).toBeNull();
  });

  it("renders prefill and generation separately in the throughput panel", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
      throughput: FIXTURE_THROUGHPUT,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByTestId("throughput-panel")).toBeInTheDocument();
    });
    expect(screen.getByTestId("tp-prefill-avg").textContent).toBe("412");
    expect(screen.getByTestId("tp-prefill-mode").textContent).toBe("380");
    expect(screen.getByTestId("tp-generation-avg").textContent).toBe("49");
    expect(screen.getByTestId("tp-generation-mode").textContent).toBe("46");
  });

  it("defaults the throughput panel to the wall-clock basis", async () => {
    // The per-request basis cannot answer "how fast is the rig" on a
    // prefix-cached, concurrent workload: prompt/ttft counts cached tokens as
    // computed, and per-request decode is a share of the engine's aggregate.
    // Wall clock is distorted by neither, so it is what the panel opens on.
    const mock = installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
      throughput: FIXTURE_THROUGHPUT,
    });
    renderPage();
    await waitFor(() => {
      const urls = mock.mock.calls.map((c) => String(c[0]));
      expect(
        urls.some((u) => u.startsWith("/api/stats/v2/throughput") && u.includes("basis=wallclock")),
      ).toBe(true);
    });
  });

  it("re-reads the throughput endpoint on the selected window", async () => {
    // The panel follows the page's range picker like every other history
    // panel — a 24h reading must not sit under a 1h heading.
    const mock = installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
      throughput: FIXTURE_THROUGHPUT,
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("throughput-panel")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "24h" }));

    await waitFor(() => {
      const urls = mock.mock.calls.map((c) => String(c[0]));
      expect(
        urls.some((u) => u.startsWith("/api/stats/v2/throughput") && u.includes("range=24h")),
      ).toBe(true);
    });
  });

  it("names the loaded models in the selector", async () => {
    // Was a read-only "Loaded:" strip. The models are now the page's model
    // SELECTION, so the same names appear as checkboxes -- one control that
    // narrows the stats, the live view and god mode alike.
    installFetchStub({ overview: FIXTURE_OVERVIEW });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("model-selector")).toBeInTheDocument();
    });
    expect(screen.getByTestId("model-selector").textContent).toContain(
      "served-name-1",
    );
  });

  it("renders the per-key tokens table sorted total_tokens DESC by default", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByTestId("tokens-per-key-table")).toBeInTheDocument();
    });

    const rows = screen.getAllByTestId("tokens-per-key-row");
    expect(rows).toHaveLength(2);
    // Heavy Key first (2250 total_tokens) — backend already sorts but the
    // page also re-sorts client-side; assert the visible order.
    expect(rows[0].textContent).toContain("Heavy Key");
    expect(rows[0].textContent).toContain("2,250");
    expect(rows[1].textContent).toContain("(unknown)");
    expect(rows[1].textContent).toContain("orphan");
  });

  it("links a per-key token name to its details page, but leaves a deleted token unlinked", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: FIXTURE_TPK,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByTestId("tokens-per-key-table")).toBeInTheDocument();
    });
    const rows = screen.getAllByTestId("tokens-per-key-row");
    const heavy = rows.find((r) => r.textContent?.includes("Heavy Key"))!;
    const link = within(heavy).getByRole("link", { name: "Heavy Key" });
    expect(link).toHaveAttribute("href", "/tokens/t-heavy");
    expect(link.className).toContain("hover:underline");

    // The deleted-token row (name "(unknown)") stays plain text.
    const orphan = rows.find((r) => r.textContent?.includes("(unknown)"))!;
    expect(within(orphan).queryByRole("link")).toBeNull();
  });

  it("renders the three host chart panels and no VRAM-over-time chart", async () => {
    installFetchStub({ overview: FIXTURE_OVERVIEW });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("chart-util")).toBeInTheDocument();
    });
    expect(screen.getByTestId("chart-power")).toBeInTheDocument();
    expect(screen.getByTestId("chart-tokens")).toBeInTheDocument();
    // VRAM is a step function (pre-allocated at load) — its chart was a flat
    // block and is gone; current VRAM lives on the per-GPU cards instead.
    expect(screen.queryByTestId("chart-vram")).toBeNull();
  });

  it("persists the range selection to localStorage under 'vw.stats.range'", async () => {
    installFetchStub({ overview: FIXTURE_OVERVIEW });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("range-selector")).toBeInTheDocument();
    });
    // Click the '7d' button.
    const sevenDay = screen.getByRole("button", { name: "7d" });
    act(() => {
      fireEvent.click(sevenDay);
    });
    expect(window.localStorage.getItem("vw.stats.range")).toBe("7d");
  });

  it("renders the power tile as '—' when telemetry is unavailable", async () => {
    installFetchStub({
      overview: {
        ...FIXTURE_OVERVIEW,
        current: { ...FIXTURE_OVERVIEW.current, power_w: null },
        series: { ...FIXTURE_OVERVIEW.series, power: [] },
      },
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("tile-power-value")).toBeInTheDocument();
    });
    expect(screen.getByTestId("tile-power-value").textContent).toBe("—");
  });

  it("hides the selector entirely when no model is loaded", async () => {
    // There is no selection to make. An empty fieldset would be a control with
    // nothing behind it, which is the defect class this whole branch is about.
    installFetchStub({
      overview: { ...FIXTURE_OVERVIEW, active_models: [] },
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("model-selector")).toBeNull();
  });

  it("shows an empty-state when the per-key table has no rows", async () => {
    installFetchStub({
      overview: FIXTURE_OVERVIEW,
      tokensPerKey: { range: "1h", since_minute: 0, rows: [] },
    });
    renderPage();
    await waitFor(() => {
      expect(screen.getByTestId("tile-vram-value")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("tokens-per-key-table")).toBeNull();
    expect(screen.getByText(/no token usage in this window/i)).toBeInTheDocument();
  });
});
