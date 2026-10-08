// Shared harness for the merged /stats page tests.
//
// NOT a test file — the stats-merged-*.test.tsx files import from here. Each
// of them still registers its own `vi.mock('@/lib/live-stats-stream', ...)`
// (vi.mock hoisting is per test file); the factory reads `mockStream.state`
// from ./stats-merged-stream, NOT from this module — this module imports the
// page, which imports the mocked stream module, so importing it from inside
// the factory deadlocks collection (see the note in stats-merged-stream.ts).
// `mockStream` / `setFrame` are re-exported here for the tests' convenience.

import { render } from "@testing-library/react";
import { screen } from "@testing-library/react";
import { vi } from "vitest";
import { SWRConfig } from "swr";
import StatsPage from "@/app/stats/page";
import type { LiveEngineFrame } from "@/lib/live-stats";

export { mockStream, setFrame } from "./stats-merged-stream";

// ---------------------------------------------------------------------------
// Frame builders
// ---------------------------------------------------------------------------

export function block(
  name: string,
  overrides: Partial<LiveEngineFrame> = {},
): LiveEngineFrame {
  return {
    ts: "2026-09-05T19:00:00Z",
    model: name,
    model_id: `id-${name}`,
    backend: "vllm",
    max_model_len: 8192,
    engine: {
      num_requests_running: 2,
      num_requests_waiting: 1,
      waiting_by_reason: {},
      kv_cache_usage_perc: 0.4,
      kv_tokens_used: 100,
      kv_tokens_total: 250,
      engine_sleep_state: 0,
      preemptions_total: 0,
      preemptions_per_s: 0,
    },
    throughput: {
      prompt_tokens_per_s: 10,
      generation_tokens_per_s: 20,
      prompt_tokens_total: 1000,
      generation_tokens_total: 2000,
    },
    cache: {
      prefix_hit_rate: null,
      prefix_hit_rate_cumulative: null,
      mm_hit_rate_cumulative: null,
      external_prefix_hit_rate_cumulative: null,
    },
    latency: {
      ttft_p50: null, ttft_p90: null, ttft_p99: null, ttft_mean: null,
      itl_p50: null, itl_p99: null, tpot_p50: null,
      e2e_p50: null, e2e_p90: null, e2e_p99: null,
      buckets: null,
    },
    finished: {},
    scrape_error: null,
    ...overrides,
  };
}

/** llama.cpp: no KV gauge, no throughput counters, no latency histograms. */
export const SILENT_LLAMACPP = block("model-b-27b", {
  backend: "llamacpp",
  engine: {
    num_requests_running: 1,
    num_requests_waiting: 0,
    waiting_by_reason: {},
    kv_cache_usage_perc: null,
    kv_tokens_used: null,
    kv_tokens_total: null,
    engine_sleep_state: null,
    preemptions_total: null,
    preemptions_per_s: null,
  },
  throughput: {
    prompt_tokens_per_s: null,
    generation_tokens_per_s: null,
    prompt_tokens_total: null,
    generation_tokens_total: null,
  },
});

export function frameWith(blocks: LiveEngineFrame[]): LiveEngineFrame {
  return { ...blocks[0], models: blocks };
}

// ---------------------------------------------------------------------------
// Fetch stub
// ---------------------------------------------------------------------------

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export const REQUESTS = {
  ts: "2026-09-05T19:00:00Z",
  count: 2,
  requests: [
    {
      id: "r1", token_id: "tok-a", token_name: "key-a", client_ip: "10.0.0.1",
      model: "model-a-8b", path: "/v1/chat/completions",
      prompt_tokens: 100, completion_tokens: 10, context_tokens: 110,
      max_model_len: 8192, context_pct: 0.013, elapsed_s: 1, phase: "decode",
      orphan: false,
    },
    {
      id: "r2", token_id: "tok-b", token_name: "key-b", client_ip: "10.0.0.2",
      model: "model-b-27b", path: "/v1/chat/completions",
      prompt_tokens: 200, completion_tokens: 20, context_tokens: 220,
      max_model_len: 8192, context_pct: 0.027, elapsed_s: 2, phase: "decode",
      orphan: false,
    },
  ],
  by_token: [],
  by_ip: [],
};

// Per-request history rows (GET /api/stats/v2/requests). `model_id` is the
// models table's ROW id and `model` the served name, and they DIFFER here on
// purpose (MODELS below: "id-model-a-8b" vs "model-a-8b"). That divergence is
// the whole reason ?models= has to filter on the row id — a fixture where the
// two matched would pass either way.
export const NOW_EPOCH = 1_788_000_000;

export const HISTORY_ROWS = [
  {
    id: "f1", finished_at: NOW_EPOCH - 60, token_id: "tok-a", token_name: "key-a", client_ip: "10.0.0.1",
    model: "model-a-8b", model_id: "id-model-a-8b",
    prompt_tokens: 41200, completion_tokens: 812,
    duration_s: 23.1, ttft_s: 0.4, finish_reason: "stop",
    orphan: false, started_iso: "2026-09-05T18:59:00Z",
  },
  {
    // No token_id: an older row recorded before the column existed.
    id: "f2", finished_at: NOW_EPOCH - 120, token_id: null, token_name: "key-b", client_ip: "10.0.0.2",
    model: "model-b-27b", model_id: "id-model-b-27b",
    prompt_tokens: 3100, completion_tokens: 96,
    duration_s: 4.2, ttft_s: null, finish_reason: "length",
    orphan: true, started_iso: "2026-09-05T18:58:00Z",
  },
];

const RANGE_S: Record<string, number> = { "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800 };

export function historyFor(url: string, rows = HISTORY_ROWS) {
  const params = new URL(url, "http://x").searchParams;
  const range = params.get("range") ?? "1h";
  const raw = params.get("models");
  const selected = raw ? raw.split(",") : null;
  const want = selected === null ? null : new Set(selected);
  const kept = want === null ? rows : rows.filter((r) => want.has(r.model_id));
  return {
    range,
    since_epoch: NOW_EPOCH - RANGE_S[range],
    now_epoch: NOW_EPOCH,
    selected_model_ids: selected,
    total: kept.length,
    stride: 1,
    requests: kept,
    coverage: {
      earliest_epoch: NOW_EPOCH - 10 * 86400,
      retention_days: 30,
      max_rows: 200000,
      covers_window: true,
    },
  };
}

const DIST = (count: number) => ({
  count,
  p50: 0.4, p90: 0.9, p99: 1.2, mean: 0.5,
  buckets: {
    le: [0.1, 0.5, 1, 2, null],
    counts: [0, count, count, count, count],
    count,
    sum: 0.5 * count,
  },
});

export function latencyFor(url: string) {
  const params = new URL(url, "http://x").searchParams;
  const range = params.get("range") ?? "1h";
  const basis = params.get("basis") ?? "window";
  const n = basis === "last" ? Number(params.get("n") ?? 500) : null;
  const raw = params.get("models");
  const selected = raw ? raw.split(",") : null;
  const count = selected === null ? 2 : selected.length;
  return {
    basis,
    range,
    n,
    since_epoch: basis === "window" ? NOW_EPOCH - RANGE_S[range] : null,
    selected_model_ids: selected,
    count,
    span_s: 60,
    oldest_epoch: NOW_EPOCH - 120,
    newest_epoch: NOW_EPOCH - 60,
    ttft: DIST(count),
    itl: DIST(count),
    duration: DIST(count),
    coverage: {
      earliest_epoch: NOW_EPOCH - 10 * 86400,
      retention_days: 30,
      max_rows: 200000,
      covers_window: true,
    },
  };
}

export const MODELS = [
  { id: "id-model-a-8b", served_model_name: "model-a-8b", gpu_indices: [0] },
  { id: "id-model-b-27b", served_model_name: "model-b-27b", gpu_indices: [1] },
];

export function overviewFor(url: string, models = MODELS) {
  const params = new URL(url, "http://x").searchParams;
  const range = params.get("range") ?? "24h";
  const raw = params.get("models");
  const selected = raw ? raw.split(",") : null;
  return {
    range,
    now_minute: 31538880,
    since_minute: 31538820,
    selected_model_ids: selected,
    selected_gpu_indices: selected === null ? null : [0],
    current: {
      vram_used_mib: 12000,
      vram_total_mib: 32000,
      vram_pct: 38,
      gpu_util_pct: 80,
      power_w: 250.0,
      // Distinguishable per scope, so a test can tell WHICH response a tile
      // read: unfiltered 11, narrowed 5.
      tps: selected !== null && selected.length < models.length ? 5 : 11,
    },
    active_models: models,
    series: {
      vram: [{ minute: 31538879, used_mib: 3000, total_mib: 32000 }],
      util: [{ minute: 31538879, max_pct: 50 }],
      power: [{ minute: 31538879, watts: 230.0 }],
      tokens: [{ minute: 31538879, prompt: 1000, completion: 500 }],
    },
  };
}

export function throughputFor(url: string, models = MODELS) {
  const params = new URL(url, "http://x").searchParams;
  const raw = params.get("models");
  const selected = raw ? raw.split(",") : null;
  const narrowed = selected !== null && selected.length < models.length;
  // Distinguishable per scope, exactly as overviewFor's tps was: a test can
  // tell WHICH response the panel read.
  return {
    basis: params.get("basis") ?? "request",
    range: params.get("range") ?? "24h",
    since_epoch: 1_757_000_000,
    selected_model_ids: selected,
    prefill: narrowed
      ? { count: 40, avg: 200, max: 300, mode: 180 }
      : { count: 100, avg: 500, max: 900, mode: 480 },
    generation: narrowed
      ? { count: 40, avg: 20, max: 30, mode: 19 }
      : { count: 100, avg: 50, max: 70, mode: 48 },
    coverage: {
      earliest_epoch: 1_756_000_000,
      retention_days: 30,
      max_rows: 200000,
      covers_window: true,
    },
  };
}

export interface StubOptions {
  models?: typeof MODELS;
  /** A fixed body for /api/stats/v2/requests, or a function of the URL. */
  history?: unknown | ((url: string) => unknown);
  /** A fixed body for /api/stats/v2/latency, or a function of the URL. */
  latency?: unknown | ((url: string) => unknown);
  /** A fixed body for /api/stats/v2/throughput, or a function of the URL. */
  throughput?: unknown | ((url: string) => unknown);
  requests?: unknown;
}

export function installFetchStub(opts: StubOptions = {}): string[] {
  const urls: string[] = [];
  const mock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    urls.push(url);
    if (url === "/api/auth/refresh") return json({ access_token: "t" });
    if (url === "/api/csrf") return json({ csrf: "c" });
    if (url.startsWith("/api/stats/v2/overview")) {
      return json(overviewFor(url, opts.models ?? MODELS));
    }
    if (url.startsWith("/api/stats/v2/tokens-per-key")) {
      return json({ range: "1h", since_minute: 0, rows: [] });
    }
    if (url.startsWith("/api/stats/requests")) {
      return json(opts.requests ?? REQUESTS);
    }
    if (url.startsWith("/api/stats/v2/requests")) {
      const h = opts.history;
      if (typeof h === "function") return json((h as (u: string) => unknown)(url));
      return json(h ?? historyFor(url));
    }
    if (url.startsWith("/api/stats/v2/throughput")) {
      const t = opts.throughput;
      if (typeof t === "function") return json((t as (u: string) => unknown)(url));
      return json(t ?? throughputFor(url, opts.models ?? MODELS));
    }
    if (url.startsWith("/api/stats/v2/latency")) {
      const l = opts.latency;
      if (typeof l === "function") return json((l as (u: string) => unknown)(url));
      return json(l ?? latencyFor(url));
    }
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return urls;
}

export class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// ---------------------------------------------------------------------------
// Render + queries
// ---------------------------------------------------------------------------

export function renderPage() {
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

export function boxFor(id: string): HTMLInputElement {
  const opt = screen
    .getAllByTestId("model-selector-option")
    .find((o) => o.getAttribute("data-model-id") === id);
  if (!opt) throw new Error(`no option for ${id}`);
  return opt.querySelector("input") as HTMLInputElement;
}
