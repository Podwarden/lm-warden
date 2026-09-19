"use client";

// /stats — THE stats page. History and live, merged.
//
// /ui/stats/live is gone (404, no redirect — one stats page, and a silent
// redirect would leave people believing the old route still exists).
// Everything it showed moves here, next to the minute-level history this page
// already kept — which is the existing answer to "the charts reset". Live's
// ~80 seconds of client-side React state was the anomaly.
//
// Layout, top to bottom (approved mockup):
//   1. Header — title, live indicator + state chip, God mode.
//   2. ONE control bar — MODELS │ WINDOW, one compact scope note.
//   3. Host — VRAM / GPU util / power / tokens-per-sec KPIs + three history
//      charts. Badged `host`: the model selection does NOT scope VRAM, GPU
//      utilisation or power — two engines share a card, so watts cannot be
//      attributed to one model. It DOES scope tokens, latency, KV, requests
//      and preemptions.
//   4. Live — 5-minute rolling timeline, per-model context & cache (never
//      combined: each engine has its own pool and its own max_model_len),
//      preemptions.
//   5. Latency — TTFT / mean ITL / duration distributions from the persisted
//      per-request history, on the window or on the last N requests (the
//      toggle says which).
//   6. Requests — in flight, then the requests chart over the window, with
//      the table as its detail view.
//   7. The page's existing per-key table + system config, unchanged.
//
// The WINDOW governs every history panel, the latency distributions and the
// requests chart — those two read the persisted per-request store, which is
// what let them stop saying "not the window". The 5-min timeline still
// cannot honour 7d (it is live engine state) and says so on the panel rather
// than showing a narrower span than the button promises. Where the store
// cannot serve a window in full — history younger than the window, or a
// retention shorter than it — the panel says precisely which. Longer windows
// bucket up (1 min → 5 min → 30 min) and the heading says which.
//
// Charts carry a FIXED 24h reference — dashed busy-median, dotted peak — see
// @/lib/live-history for the busy_median/peak mirror semantics.

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import {
  STATS_RANGES,
  formatTps,
  formatWatts,
  mibToGib,
  type StatsRange,
  type StatsV2Overview,
  type StatsV2PowerPoint,
  type StatsV2TokensPoint,
  type StatsV2TokensPerKey,
  type StatsV2TokensPerKeyRow,
  type StatsV2UtilPoint,
} from "@/lib/stats-v2";
import { usePersistedRange } from "@/lib/use-persisted-range";
import { useModelSelection } from "@/lib/model-selection";
import { useLiveStats } from "@/lib/live-stats-stream";
import {
  aggregateRequests,
  combineThroughput,
  formatCompact,
  formatInt,
  liveFramesOf,
  sumReported,
  type LiveEngineFrame,
  type LiveRequestsSnapshot,
} from "@/lib/live-stats";
import {
  WINDOW_META,
  bucketPoints,
  combineTimeline,
  hostBusyMinutes,
  maxOf,
  meanOf,
  pushSample,
  referenceFor,
  type Reference,
  type TimelineSample,
} from "@/lib/live-history";
import {
  LATENCY_LAST_N,
  type LatencyBasis,
  type LatencyResponse,
  type RequestHistoryResponse,
  type ThroughputBasis,
  type ThroughputResponse,
} from "@/lib/request-history";
import { ModelSelector, type SelectableModel } from "@/components/stats/model-selector";
import {
  Banner,
  CombinedFigure,
  ConnectionStatus,
  EmptyState,
  LastUpdated,
  LiveRequestsPanel,
  ModelContextRows,
  Panel,
  ScopeNote,
  SinceEngineStart,
  StateChip,
  Timeline,
  engineStateOf,
} from "@/components/stats/live-panels";
import { BasisToggle, LatencyPanels } from "@/components/stats/latency-panels";
import { ThroughputPanel } from "@/components/stats/throughput-panel";
import { RequestsPanel } from "@/components/stats/requests-panel";
import { StatCard } from "@/components/stat-card";
import {
  PowerChart,
  TokensChart,
  UtilChart,
} from "@/components/stats/v2-charts";
import { SystemConfigSection } from "@/components/stats/system-config-section";
import { Skeleton } from "@/components/ui/skeleton";
import { TIMELINE_SPAN_MS } from "@/lib/live-history";

// localStorage key for the window selector — unchanged from the old page so a
// remembered range survives the merge.
const RANGE_KEY = "vw.stats.range";

// 30s poll on the minute-bucketed overview — faster would re-render identical
// numbers. Paused when the tab is hidden.
const REFRESH_MS = 30_000;
const refreshInterval = () =>
  typeof document !== "undefined" && document.hidden ? 0 : REFRESH_MS;

// Per-request registry poll (Plane B) — spec cadence, hidden-tab paused.
const REQ_REFRESH_MS = 1_500;
const reqRefreshInterval = () =>
  typeof document !== "undefined" && document.hidden ? 0 : REQ_REFRESH_MS;

// Per-request history. 15s keeps "just ended" fresh on the chart; the
// distributions move slowly and poll with the overview.
const HISTORY_REFRESH_MS = 15_000;
const historyRefreshInterval = () =>
  typeof document !== "undefined" && document.hidden ? 0 : HISTORY_REFRESH_MS;

// localStorage key for the latency basis (window vs last N).
const BASIS_KEY = "vw.stats.latency-basis";

// localStorage key for the throughput basis (per-request speed vs wall-clock
// throughput). Separate from BASIS_KEY: they answer different questions and
// an operator who wants engine speed does not thereby want the last N
// requests' latency.
const THROUGHPUT_BASIS_KEY = "vw.stats.throughput-basis";

type Sort = { col: keyof StatsV2TokensPerKeyRow | "total_tokens"; dir: "asc" | "desc" };

/** The caption under a host chart heading: the same two reference numbers as
 *  the lines, in text — a line is read at a glance, a figure is quoted. */
function refCaption(ref: Reference | undefined, fmt: (v: number) => string) {
  if (!ref || ref.peak === null) return null;
  const text =
    ref.median === null
      ? `no busy minutes in 24h · peak ${fmt(ref.peak)}`
      : `24h busy median ${fmt(ref.median)} · peak ${fmt(ref.peak)}`;
  return (
    <span title={`median over the ${ref.busyMinutes} minutes in the last 24h when the box was working`}>
      {text}
    </span>
  );
}

export default function StatsPage() {
  const [range, setRange] = usePersistedRange(RANGE_KEY, "1h");
  const [sort, setSort] = useState<Sort>({ col: "total_tokens", dir: "desc" });
  const meta = WINDOW_META[range];

  // ---- Plane A: the live engine stream -----------------------------------
  const engine = useLiveStats();
  const frame = engine.frame;
  const allFrames = useMemo(() => liveFramesOf(frame), [frame]);

  // ---- history: host overview, deliberately UNFILTERED -------------------
  //
  // VRAM, GPU utilisation and power are host facts here. The v2 API can
  // narrow them to the cards a selection occupies, but this page does not ask
  // it to: the panels are badged `host` instead (see the header comment).
  const hostKey = `/api/stats/v2/overview?range=${range}`;
  const host = useSWR<StatsV2Overview>(hostKey, authFetchJSON, {
    refreshInterval,
    keepPreviousData: true,
  });

  // The selector's option list: the overview's active models, plus any model
  // the live stream names that the overview has not caught up with — either
  // source alone can lag the other during a load/unload.
  const loadedModels = useMemo<SelectableModel[]>(() => {
    const out = new Map<string, SelectableModel>();
    for (const m of host.data?.active_models ?? []) out.set(m.id, m);
    for (const f of allFrames) {
      if (f.model_id !== null && !out.has(f.model_id)) {
        out.set(f.model_id, {
          id: f.model_id,
          served_model_name: f.model ?? f.model_id,
        });
      }
    }
    return [...out.values()];
  }, [host.data, allFrames]);
  const modelIds = useMemo(() => loadedModels.map((m) => m.id), [loadedModels]);
  const selection = useModelSelection(modelIds);
  const isNarrowed =
    modelIds.length > 0 && selection.selected.length < modelIds.length;

  const shownFrames = useMemo(
    () =>
      allFrames.filter(
        (f) => f.model_id !== null && selection.selected.includes(f.model_id),
      ),
    [allFrames, selection.selected],
  );
  const shownNames = useMemo(
    () => new Set(shownFrames.map((f) => f.model).filter((m): m is string => !!m)),
    [shownFrames],
  );

  // ---- history: the SCOPED series (tokens / tps follow the selection) ----
  //
  // Settled-selection pattern from the old page: the key narrows only after
  // the selection has resolved against the loaded models, so the page never
  // sends `?models=` empty (a 400 by contract) and never narrows to a guess.
  const [pendingSelection, setPendingSelection] = useState<string | null>(null);
  useEffect(() => {
    setPendingSelection(selection.queryParam);
  }, [selection.queryParam]);
  const scopedKey = pendingSelection
    ? `/api/stats/v2/overview?range=${range}&models=${encodeURIComponent(pendingSelection)}`
    : null;
  const scoped = useSWR<StatsV2Overview>(scopedKey, authFetchJSON, {
    refreshInterval,
    keepPreviousData: true,
  });
  // Tokens and tps read the scoped response; before the selection settles the
  // unfiltered one is the correct thing to show.
  const tokensData = scoped.data ?? host.data;

  // ---- the fixed 24h reference -------------------------------------------
  //
  // Always 24h, never the selected window. When range === "24h" these keys
  // equal the ones above and SWR collapses them to one request.
  const refHost = useSWR<StatsV2Overview>(
    "/api/stats/v2/overview?range=24h",
    authFetchJSON,
    { refreshInterval, keepPreviousData: true },
  );
  const refScopedKey = pendingSelection
    ? `/api/stats/v2/overview?range=24h&models=${encodeURIComponent(pendingSelection)}`
    : null;
  const refScoped = useSWR<StatsV2Overview>(refScopedKey, authFetchJSON, {
    refreshInterval,
    keepPreviousData: true,
  });

  const references = useMemo(() => {
    const s = refHost.data?.series;
    if (!s) return undefined;
    // "Busy" is decided ONCE at host level and applied to every series — a
    // per-series threshold would be meaningless for a saturation-shaped
    // series that does not fall when the box goes quiet.
    const busy = hostBusyMinutes(s.util, s.tokens);
    const tokenSeries = (refScoped.data ?? refHost.data)?.series.tokens ?? [];
    return {
      util: referenceFor(s.util, (p) => p.max_pct, busy),
      power: referenceFor(s.power, (p) => p.watts, busy),
      tokens: referenceFor(tokenSeries, (p) => p.prompt, busy),
    };
  }, [refHost.data, refScoped.data]);

  // ---- window bucketing ---------------------------------------------------
  //
  // 7d of minute samples is 10,080 rows; longer windows bucket up and the
  // heading says which. Saturation-shaped series keep their bucket MAX (a
  // flattened peak would hide the event worth seeing); power and tokens keep
  // the bucket MEAN so their per-minute unit stays honest.
  const series = host.data?.series;
  const utilSeries = useMemo<StatsV2UtilPoint[]>(
    () =>
      bucketPoints(series?.util ?? [], meta.bucketMinutes, (b) => ({
        minute: b[0].minute,
        max_pct: Math.round(maxOf(b.map((p) => p.max_pct))),
      })),
    [series, meta.bucketMinutes],
  );
  const powerSeries = useMemo<StatsV2PowerPoint[]>(
    () =>
      bucketPoints(series?.power ?? [], meta.bucketMinutes, (b) => ({
        minute: b[0].minute,
        watts: meanOf(b.map((p) => p.watts)),
      })),
    [series, meta.bucketMinutes],
  );
  const tokensSeries = useMemo<StatsV2TokensPoint[]>(
    () =>
      bucketPoints(tokensData?.series.tokens ?? [], meta.bucketMinutes, (b) => ({
        minute: b[0].minute,
        prompt: Math.round(meanOf(b.map((p) => p.prompt))),
        completion: Math.round(meanOf(b.map((p) => p.completion))),
      })),
    [tokensData, meta.bucketMinutes],
  );

  // ---- Plane B: in-flight requests ---------------------------------------
  const requests = useSWR<LiveRequestsSnapshot>("/api/stats/requests", authFetchJSON, {
    refreshInterval: reqRefreshInterval,
    keepPreviousData: true,
  });
  const shownRequests = useMemo(() => {
    const rows = requests.data?.requests ?? [];
    return shownNames.size === 0 ? rows : rows.filter((r) => shownNames.has(r.model));
  }, [requests.data, shownNames]);
  const rollups = useMemo(() => aggregateRequests(shownRequests), [shownRequests]);

  // ---- per-request history: the requests chart + latency distributions ----
  //
  // Scoped SERVER-side, on the same settled-selection rule as the history
  // keys above. It has to be the server: the endpoint samples AFTER
  // filtering, so narrowing here instead would let a busy unselected model
  // push every selected row out of the response. Both follow the WINDOW —
  // the store is persisted, so 7d is a real 7d (and the response says when
  // it is not).
  const modelsQs = pendingSelection ? `&models=${encodeURIComponent(pendingSelection)}` : "";
  const history = useSWR<RequestHistoryResponse>(
    `/api/stats/v2/requests?range=${range}${modelsQs}`,
    authFetchJSON,
    { refreshInterval: historyRefreshInterval, keepPreviousData: true },
  );
  const [basis, setBasis] = useState<LatencyBasis>(() => {
    try {
      return window.localStorage.getItem(BASIS_KEY) === "last" ? "last" : "window";
    } catch {
      return "window";
    }
  });
  const chooseBasis = (b: LatencyBasis) => {
    setBasis(b);
    try {
      window.localStorage.setItem(BASIS_KEY, b);
    } catch {
      /* private mode */
    }
  };
  const latencyKey =
    basis === "last"
      ? `/api/stats/v2/latency?range=${range}&basis=last&n=${LATENCY_LAST_N}${modelsQs}`
      : `/api/stats/v2/latency?range=${range}&basis=window${modelsQs}`;
  const latency = useSWR<LatencyResponse>(latencyKey, authFetchJSON, {
    refreshInterval,
    keepPreviousData: true,
  });

  // ---- prefill / generation throughput ------------------------------------
  //
  // Replaces the single blended tokens-per-second tile. Scoped by the model
  // selection and bound to the window like every other history panel: a 24h
  // reading must not sit under a 1h heading.
  // Wall clock by default. The per-request basis cannot answer "how fast is
  // the rig" on this workload and is wrong in BOTH directions: prompt/ttft
  // counts prefix-cached tokens the engine never computed (46k-token prompts
  // at 1.9s TTFT read as ~26k tok/s, ~13x what four A4000s can physically
  // prefill through a 27B), while per-request decode is one request's share of
  // a batched engine. See the header comment on throughput-panel.tsx.
  const [throughputBasis, setThroughputBasis] = useState<ThroughputBasis>(() => {
    try {
      return window.localStorage.getItem(THROUGHPUT_BASIS_KEY) === "request"
        ? "request"
        : "wallclock";
    } catch {
      return "wallclock";
    }
  });
  const chooseThroughputBasis = (b: ThroughputBasis) => {
    setThroughputBasis(b);
    try {
      window.localStorage.setItem(THROUGHPUT_BASIS_KEY, b);
    } catch {
      /* private mode */
    }
  };
  const throughput = useSWR<ThroughputResponse>(
    `/api/stats/v2/throughput?range=${range}&basis=${throughputBasis}${modelsQs}`,
    authFetchJSON,
    { refreshInterval, keepPreviousData: true },
  );

  // ---- 5-minute live ring -------------------------------------------------
  //
  // Keyed on wall time, never on the request set, and kept PER MODEL so the
  // selection can change at render time without a rebuilt ring. An absent
  // metric is stored as null, never 0 — a flat line at zero is a CLAIM about
  // throughput the engine did not make.
  const [ring, setRing] = useState<TimelineSample[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const lastTs = useRef<string | null>(null);
  useEffect(() => {
    if (!frame) return;
    if (frame.ts === lastTs.current) return;
    lastTs.current = frame.ts;
    const at = Number.isNaN(Date.parse(frame.ts)) ? Date.now() : Date.parse(frame.ts);
    const perModel: TimelineSample["perModel"] = {};
    for (const b of liveFramesOf(frame)) {
      if (b.model_id === null) continue;
      perModel[b.model_id] = {
        gen: b.throughput?.generation_tokens_per_s ?? null,
        running: b.engine?.num_requests_running ?? null,
        waiting: b.engine?.num_requests_waiting ?? null,
      };
    }
    setRing((r) => pushSample(r, { at, perModel }));
    setNow(Date.now());
  }, [frame]);
  const timelinePoints = useMemo(
    () => combineTimeline(ring, selection.selected),
    [ring, selection.selected],
  );

  // ---- combined live figures ---------------------------------------------
  const combined = useMemo(() => combineThroughput(shownFrames), [shownFrames]);
  const preemptRate = useMemo(
    () => sumReported(shownFrames.map((f) => f.engine?.preemptions_per_s)),
    [shownFrames],
  );
  const preemptTotal = useMemo(
    () => sumReported(shownFrames.map((f) => f.engine?.preemptions_total)),
    [shownFrames],
  );
  const state = engineStateOf(shownFrames);
  const scrapeErrors = shownFrames.filter((f) => f.scrape_error !== null);
  const noModel = frame !== null && allFrames.length === 0;

  // Defined once and placed twice, because the two branches below are mutually
  // exclusive. It LIVES in the live section for layout (under Context and
  // cache, level with Preemptions) but it is NOT live data -- it reads the
  // persisted request history, which outlives the engine. Hiding it with the
  // live panels when the engine goes idle would drop a window the operator can
  // still act on.
  const throughputPanel = (
    <ThroughputPanel
      className="lg:col-span-2 lg:row-start-2"
      data={throughput.data}
      basis={throughputBasis}
      onBasisChange={chooseThroughputBasis}
      isLoading={throughput.isLoading && !throughput.data}
      error={throughput.error}
      range={range}
      human={meta.human}
    />
  );

  // ---- what the numbers cover, read off the RESPONSE ---------------------
  const covering = scoped.data?.selected_model_ids ?? null;
  const coveringNames = useMemo(() => {
    if (!covering) return null;
    const byId = new Map(loadedModels.map((m) => [m.id, m.served_model_name]));
    return covering.map((id) => byId.get(id) ?? id);
  }, [covering, loadedModels]);
  const coveringNarrowed =
    covering !== null && covering.length < Math.max(loadedModels.length, 1);

  // Detect whether the host's GPUs report power at all.
  const powerSupported = useMemo(() => {
    if (!host.data) return true;
    if (host.data.current.power_w !== null) return true;
    return host.data.series.power.length > 0;
  }, [host.data]);

  // ---- per-key table (unchanged from the old page) ------------------------
  const tpk = useSWR<StatsV2TokensPerKey>(
    `/api/stats/v2/tokens-per-key?range=${range}`,
    authFetchJSON,
    { refreshInterval },
  );
  const sortedRows = useMemo(() => {
    const rows = tpk.data?.rows ?? [];
    if (rows.length === 0) return rows;
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = a[sort.col];
      const bv = b[sort.col];
      if (typeof av === "string" && typeof bv === "string") {
        return av.localeCompare(bv) * dir;
      }
      const an = typeof av === "number" ? av : 0;
      const bn = typeof bv === "number" ? bv : 0;
      return (an - bn) * dir;
    });
  }, [tpk.data, sort]);

  function toggleSort(col: Sort["col"]) {
    setSort((s) =>
      s.col === col
        ? { col, dir: s.dir === "asc" ? "desc" : "asc" }
        : { col, dir: col === "name" ? "asc" : "desc" },
    );
  }

  const data = host.data;
  const one = selection.selected.length === 1;
  const scopePhrase = one
    ? loadedModels.find((m) => m.id === selection.selected[0])?.served_model_name ??
      "1 model"
    : `${selection.selected.length} models, combined`;

  return (
    <div className="space-y-6" data-testid="stats-page">
      {/* 1 ── Header ---------------------------------------------------- */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold">Stats</h1>
          <ConnectionStatus state={engine} />
          <StateChip state={noModel ? "none" : state} />
          {frame && <LastUpdated ts={frame.ts} />}
        </div>
      </div>

      {/* 2 ── ONE control bar: MODELS │ WINDOW │ scope note -------------- */}
      <div
        data-testid="stats-control-bar"
        className="flex flex-wrap items-center gap-3 rounded-md border border-chat-rule bg-chat-surface/40 px-3 py-2"
      >
        <ModelSelector models={loadedModels} selection={selection} />
        {loadedModels.length > 0 && (
          <span aria-hidden="true" className="hidden h-6 w-px bg-chat-rule sm:block" />
        )}
        <span className="text-xs uppercase tracking-wider text-chat-dim">Window</span>
        <div
          role="group"
          aria-label="Window"
          data-testid="range-selector"
          className="inline-flex h-8 rounded-md border border-chat-rule bg-chat-surface/50 p-0.5 text-xs"
        >
          {STATS_RANGES.map((r: StatsRange) => {
            const active = r === range;
            return (
              <button
                key={r}
                type="button"
                onClick={() => setRange(r)}
                data-active={active}
                aria-pressed={active}
                className={
                  "rounded px-3 transition-colors " +
                  (active
                    ? "bg-chat-accent/20 text-chat-fg shadow-inner"
                    : "text-chat-muted hover:bg-chat-surface-2")
                }
              >
                {r}
              </button>
            );
          })}
        </div>
        <span
          data-testid="scope-note"
          className="ml-auto font-mono text-[11px] text-chat-dim"
        >
          {loadedModels.length === 0
            ? meta.bucketLabel
            : `${selection.selected.length} of ${loadedModels.length} · ${
                one ? scopePhrase : "combined"
              } · ${meta.bucketLabel}`}
        </span>
      </div>

      {/* What the history numbers cover, read off the RESPONSE. Lags a click
          by one round trip on purpose, so it is never briefly wrong. */}
      {coveringNames && (
        <p data-testid="stats-covering" className="text-xs text-chat-dim">
          {coveringNarrowed ? "Tokens showing" : "Tokens showing all"}{" "}
          <span className="font-mono text-chat-muted">{coveringNames.join(", ")}</span>
          {coveringNarrowed && (
            <span>
              {" "}
              — tokens, latency, KV and requests narrow to this selection; VRAM,
              GPU utilisation and power stay host figures
            </span>
          )}
        </p>
      )}

      {host.error && !host.data && (
        <p className="text-sm text-chat-negative">
          Failed to load stats
          {host.error instanceof Error ? `: ${host.error.message}` : "."}
        </p>
      )}

      {/* Stream / scrape problems ---------------------------------------- */}
      {engine.status === "terminal-error" && (
        <Banner tone="error">
          Live engine stream disconnected
          {engine.errorCode ? ` (HTTP ${engine.errorCode})` : ""}. Reload the
          page to reconnect. History panels keep updating.
        </Banner>
      )}
      {scrapeErrors.map((f) => (
        <Banner key={f.model_id ?? "?"} tone="warn">
          {/* No "showing the last good values": nothing holds a last good
              frame — the stream replaces state every message — so the honest
              copy is what actually happens. */}
          Metrics scrape failed for {f.model ?? "this model"}: {f.scrape_error}.
          Its live panels are paused until the next successful scrape; host
          charts and requests are unaffected.
        </Banner>
      ))}

      {/* 3 ── HOST — not scoped by the model selection ------------------- */}
      <section aria-label="Host" className="space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-chat-dim">
          Host
          <span className="rounded border border-chat-rule px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-wide">
            host / all models
          </span>
        </div>

        <div data-testid="current-row" className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {!data ? (
            <>
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </>
          ) : (
            <>
              <StatCard
                label="VRAM"
                value={
                  <span data-testid="tile-vram-value">
                    {mibToGib(data.current.vram_used_mib)} /{" "}
                    {mibToGib(data.current.vram_total_mib)}
                  </span>
                }
                unit="GiB"
                hint={`${data.current.vram_pct}% used · host`}
                title={`${data.current.vram_used_mib} / ${data.current.vram_total_mib} MiB`}
              />
              <StatCard
                label="GPU util"
                value={
                  <span data-testid="tile-util-value">{data.current.gpu_util_pct}</span>
                }
                unit="%"
                hint="max across GPUs · host"
              />
              <StatCard
                label="Power"
                value={
                  <span data-testid="tile-power-value">
                    {formatWatts(data.current.power_w)}
                  </span>
                }
                unit={data.current.power_w === null ? undefined : "W"}
                hint={
                  data.current.power_w === null
                    ? "telemetry unavailable"
                    : "sum across GPUs · host"
                }
                title={
                  data.current.power_w === null
                    ? "No GPU on this host reports power.draw."
                    : undefined
                }
              />
            </>
          )}
        </div>

        <div data-testid="charts-grid" className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* No VRAM-over-time chart, on purpose: both engines pre-allocate
              weights + KV cache at load, so VRAM only moves on load/unload
              and its chart is a flat block. Current VRAM is on the per-GPU
              cards in SystemConfigSection below. */}
          <ChartPanel
            title="GPU utilisation"
            testid="chart-util"
            badge="host"
            note={
              <>
                {refCaption(references?.util, (v) => `${Math.round(v)}%`)}
                {references?.util && " · "}
                {meta.human} · {meta.bucketLabel}
              </>
            }
          >
            {!data ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <UtilChart points={utilSeries} range={range} reference={references?.util} />
            )}
          </ChartPanel>
          <ChartPanel
            title="Power draw"
            testid="chart-power"
            badge="host"
            note={
              <>
                {refCaption(references?.power, (v) => `${Math.round(v)} W`)}
                {references?.power && " · "}
                {meta.human} · {meta.bucketLabel}
              </>
            }
          >
            {!data ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <PowerChart
                points={powerSeries}
                range={range}
                supported={powerSupported}
                reference={references?.power}
              />
            )}
          </ChartPanel>
          <ChartPanel
            title="Tokens / minute"
            testid="chart-tokens"
            // Three host charts in a two-column grid: the tokens chart takes
            // the full second row rather than sitting orphaned at half width.
            className="lg:col-span-2"
            note={
              <>
                {refCaption(references?.tokens, (v) =>
                  v >= 1000 ? `${Math.round(v / 1000)}k` : `${Math.round(v)}`,
                )}
                {references?.tokens && " · "}
                {one ? scopePhrase : "selected, combined"} · {meta.human} ·{" "}
                {meta.bucketLabel}
              </>
            }
          >
            {!data ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <TokensChart
                points={tokensSeries}
                range={range}
                reference={references?.tokens}
              />
            )}
          </ChartPanel>
        </div>
      </section>

      {/* 4 ── LIVE — scoped to the selection ----------------------------- */}
      <section aria-label="Live" data-testid="live-section" className="space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-chat-dim">
          Live
          <span
            data-testid="live-scope-badge"
            className="max-w-[40ch] truncate rounded border border-chat-rule px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-wide"
            title={scopePhrase}
          >
            {loadedModels.length === 0 ? "no models" : scopePhrase}
          </span>
        </div>

        {noModel ? (
          <div className="space-y-4">
            <EmptyState
              title="No model loaded"
              body="The engine is idle. Load a model to see the live timeline, latency distributions and per-session telemetry. Host charts above keep their history."
            />
            {throughputPanel}
          </div>
        ) : (
          <>
            <Panel
              title="Last 5 minutes"
              testid="timeline-panel"
              right={
                <ScopeNote>
                  live · not the window · survives idle
                  {(() => {
                    const gen = combined.generation_tokens_per_s;
                    return gen.value !== null
                      ? ` · ${formatTps(gen.value)} tok/s now`
                      : "";
                  })()}
                </ScopeNote>
              }
            >
              {timelinePoints.length < 2 ? (
                <p className="py-6 text-center text-sm text-chat-dim">
                  Collecting — the timeline fills as frames arrive.
                </p>
              ) : (
                <Timeline points={timelinePoints} spanMs={TIMELINE_SPAN_MS} now={now} />
              )}
            </Panel>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <Panel
                title="Context and cache"
                testid="context-cache-panel"
                className="lg:col-span-2"
                right={<ScopeNote>per model · never combined</ScopeNote>}
              >
                {/* Each engine has its own KV pool and its own max_model_len;
                    one averaged percentage would describe nothing. */}
                <ModelContextRows frames={shownFrames} />
              </Panel>
              {throughputPanel}
              <div className="space-y-4">
                <Panel
                  title="Preemptions"
                  testid="preemptions-panel"
                  right={<ScopeNote>the real pressure signal</ScopeNote>}
                >
                  <CombinedFigure
                    label="preemptions / s"
                    metric={preemptRate}
                    unit="/s"
                    format={(n) => (n === null ? "—" : n.toFixed(2))}
                    testid="preempt-rate"
                  />
                  <p className="mt-2 text-[11px] leading-relaxed text-chat-dim">
                    KV percentage is only the leading indicator; this is what
                    callers actually feel.{" "}
                    <span className="text-chat-dim">
                      {preemptTotal.value === null
                        ? ""
                        : `${formatInt(preemptTotal.value)} preempted since engine start.`}
                    </span>
                  </p>
                </Panel>
                <SinceEngineStart frames={shownFrames} />
              </div>
            </div>
          </>
        )}
      </section>

      {/* 5 ── LATENCY — from the persisted per-request history ----------- */}
      <section aria-label="Latency" data-testid="latency-section" className="space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-xs font-semibold uppercase tracking-wider text-chat-dim">
          Latency
          <span className="rounded border border-chat-rule px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-wide">
            proxy-measured · every backend
          </span>
          <span className="ml-auto normal-case tracking-normal">
            <BasisToggle basis={basis} onChange={chooseBasis} />
          </span>
        </div>
        <LatencyPanels
          data={latency.data}
          human={meta.human}
          isLoading={latency.isLoading && !latency.data}
          error={latency.error}
        />
      </section>

      {/* 6 ── REQUESTS ---------------------------------------------------- */}
      <section aria-label="Requests" className="space-y-4">
        <div className="text-xs font-semibold uppercase tracking-wider text-chat-dim">
          Requests
        </div>
        <LiveRequestsPanel
          rows={shownRequests}
          error={requests.error}
          isLoading={requests.isLoading && !requests.data}
        />
        <RequestsPanel
          data={history.data}
          range={range}
          human={meta.human}
          isLoading={history.isLoading && !history.data}
          error={history.error}
        />
        {/* Rollups over the SAME narrowed rows as the in-flight table, so the
            two views cannot disagree about which requests exist. */}
        {shownRequests.length > 0 && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Panel title="By token" bodyClassName="p-0" testid="by-token-panel">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-chat-rule text-left text-[11px] uppercase tracking-wider text-chat-dim">
                    <tr>
                      <th className="px-4 py-2 font-medium">Token</th>
                      <th className="px-3 py-2 text-right font-medium">Reqs</th>
                      <th className="px-3 py-2 text-right font-medium">Context</th>
                      <th className="px-3 py-2 text-right font-medium">Prompt</th>
                      <th className="px-3 py-2 text-right font-medium">Compl.</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-chat-rule/70 font-mono tabular-nums text-chat-muted">
                    {rollups.by_token.map((r, i) => (
                      <tr key={`${r.token_id ?? "anon"}-${i}`}>
                        <td className="px-4 py-2 text-xs">
                          {r.token_id ? (
                            <Link
                              href={`/tokens/${encodeURIComponent(r.token_id)}`}
                              className="hover:underline underline-offset-2"
                            >
                              {r.token_name ?? r.token_id}
                            </Link>
                          ) : (
                            r.token_name ?? <span className="text-chat-dim">anonymous</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">{r.requests}</td>
                        <td className="px-3 py-2 text-right text-chat-fg">
                          {formatCompact(r.context_tokens)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {formatCompact(r.prompt_tokens)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {formatCompact(r.completion_tokens)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
            <Panel title="By client IP" bodyClassName="p-0" testid="by-ip-panel">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-chat-rule text-left text-[11px] uppercase tracking-wider text-chat-dim">
                    <tr>
                      <th className="px-4 py-2 font-medium">Client IP</th>
                      <th className="px-3 py-2 text-right font-medium">Reqs</th>
                      <th className="px-3 py-2 text-right font-medium">Context</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-chat-rule/70 font-mono tabular-nums text-chat-muted">
                    {rollups.by_ip.map((r, i) => (
                      <tr key={`${r.client_ip ?? "unknown"}-${i}`}>
                        <td className="px-4 py-2 text-xs">{r.client_ip ?? "—"}</td>
                        <td className="px-3 py-2 text-right">{r.requests}</td>
                        <td className="px-3 py-2 text-right text-chat-fg">
                          {formatCompact(r.context_tokens)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
        )}
      </section>

      {/* 7 ── the page's original bottom half ---------------------------- */}
      <SystemConfigSection />

      <section aria-label="Tokens per API key" className="space-y-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-chat-muted">
            Tokens per API key
          </h2>
          <span className="text-xs text-chat-dim">
            last {range}
            {isNarrowed && (
              // Says so rather than letting the table look as though it
              // narrowed with everything else above it. token_usage_minute is
              // keyed by API token and carries no model column, so per-model
              // per-key usage is a question the rollup cannot answer at all.
              <span data-testid="tokens-per-key-scope">
                {" "}
                · all models (per-key usage is not recorded per model)
              </span>
            )}
          </span>
        </div>
        {tpk.isLoading && !tpk.data ? (
          <Skeleton className="h-24 w-full" />
        ) : tpk.error && !tpk.data ? (
          <p className="text-sm text-chat-negative">
            Failed to load per-key tokens
            {tpk.error instanceof Error ? `: ${tpk.error.message}` : "."}
          </p>
        ) : sortedRows.length === 0 ? (
          <div className="rounded-md border border-dashed border-chat-rule bg-chat-surface/30 p-6 text-center text-sm text-chat-muted">
            No token usage in this window.
          </div>
        ) : (
          <div
            data-testid="tokens-per-key-table"
            className="overflow-x-auto rounded-md border border-chat-rule"
          >
            <table className="w-full text-sm">
              <thead className="border-b border-chat-rule bg-chat-surface/50 text-left text-xs uppercase text-chat-muted">
                <tr>
                  <SortableHeader label="Name" col="name" sort={sort} onSort={toggleSort} />
                  <th className="px-3 py-2">Prefix</th>
                  <SortableHeader
                    label="Requests"
                    col="requests"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                  <SortableHeader
                    label="Prompt"
                    col="prompt_tokens"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                  <SortableHeader
                    label="Completion"
                    col="completion_tokens"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                  <SortableHeader
                    label="Total"
                    col="total_tokens"
                    sort={sort}
                    onSort={toggleSort}
                    align="right"
                  />
                </tr>
              </thead>
              <tbody className="divide-y divide-chat-rule">
                {sortedRows.map((row) => (
                  <tr key={row.token_id} data-testid="tokens-per-key-row" className="text-chat-fg">
                    <td className="px-3 py-2">
                      {row.name === "(unknown)" ? (
                        row.name
                      ) : (
                        <Link
                          href={`/tokens/${encodeURIComponent(row.token_id)}`}
                          className="hover:underline underline-offset-2"
                        >
                          {row.name}
                        </Link>
                      )}
                      {row.name === "(unknown)" && (
                        <span
                          className="ml-2 rounded bg-chat-warn/20 px-1.5 py-0.5 text-[10px] uppercase text-chat-warn"
                          title="The api_tokens row was deleted; usage remains in the rollup."
                        >
                          orphan
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-chat-muted">
                      {row.prefix ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.requests.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.prompt_tokens.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.completion_tokens.toLocaleString()}
                    </td>
                    <td className="px-3 py-2 text-right font-semibold tabular-nums text-chat-fg">
                      {row.total_tokens.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

// Chart panel wrapper — heading, host badge and the reference/window note.
interface ChartPanelProps {
  title: string;
  testid: string;
  badge?: string;
  note?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}

function ChartPanel({ title, testid, badge, note, className, children }: ChartPanelProps) {
  return (
    <div data-testid={testid} className={className ? `space-y-2 ${className}` : "space-y-2"}>
      <h2 className="flex flex-wrap items-baseline gap-2 text-sm font-semibold uppercase tracking-wider text-chat-muted">
        {title}
        {badge && (
          <span className="rounded border border-chat-rule px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-wide text-chat-dim">
            {badge}
          </span>
        )}
        {note && (
          <span className="ml-auto text-[10px] font-normal normal-case tracking-normal text-chat-dim">
            {note}
          </span>
        )}
      </h2>
      {children}
    </div>
  );
}

interface SortableHeaderProps {
  label: string;
  col: Sort["col"];
  sort: Sort;
  onSort: (col: Sort["col"]) => void;
  align?: "left" | "right";
}

function SortableHeader({ label, col, sort, onSort, align = "left" }: SortableHeaderProps) {
  const active = sort.col === col;
  const arrow = !active ? "" : sort.dir === "asc" ? " ↑" : " ↓";
  return (
    <th className={"px-3 py-2 " + (align === "right" ? "text-right" : "text-left")}>
      <button
        type="button"
        onClick={() => onSort(col)}
        data-testid={`sort-${col}`}
        aria-pressed={active}
        className={
          "uppercase tracking-wider transition-colors " +
          (active ? "text-chat-accent" : "text-chat-muted hover:text-chat-fg")
        }
      >
        {label}
        {arrow}
      </button>
    </th>
  );
}
