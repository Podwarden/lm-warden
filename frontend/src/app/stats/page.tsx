"use client";

// /stats — the operator's Stats page, redesigned (owner-approved 2026-11-22).
//
// One scope per page: ONE model on this install, chosen in the scope bar. Every
// panel above the bottom Requests block shows exactly that scope; latency, speed
// and cache are never averaged across models. Breakdowns (by key, client IP,
// model) happen only in the bottom Requests table's Group by, which shows every
// request by default (owner decision D9).
//
// Fixed size: every panel above the Requests block keeps its width and height in
// every state (idle, no data, low volume, models changing). Notices use one
// reserved line under the scope bar instead of pushing the page down.
//
// Top to bottom: header and window · scope bar · notice line · Now · Latency and
// speed · Tokens / second · Cache · Host (D4: below the request-facing panels) ·
// System configuration · Engine details · Requests chart · live Requests table.
//
// Prompt Lens (spec 2026-11-22, Q1): with the platform switch on, an eye on
// each live Requests row opens that request on the Prompt Lens page
// (/prompt-lens?req=<id>; it was a dock along the bottom of this page until
// 2026-11-23).
//
// Multisite (several installs) is not built yet: see INSTALL_SELECTOR_ENABLED in
// components/stats/redesign/scope-bar.tsx (TODO B12).

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";

import { promptLensHref } from "@/components/prompt-lens/prompt-lens-page";
import { usePromptLensEnabled } from "@/components/prompt-lens/prompt-lens-view";
import { AdminCacheCard } from "@/components/stats/redesign/cache-card";
import { EngineDetails } from "@/components/stats/redesign/engine-details";
import { NowCard, type SparkPoint } from "@/components/stats/redesign/now-card";
import { FixedCard, Section } from "@/components/stats/redesign/primitives";
import { AdminRequestsTable } from "@/components/stats/redesign/requests-table";
import { NoticeLine, ScopeBar, type ScopeTab } from "@/components/stats/redesign/scope-bar";
import { LatencyScatter, SpeedTiles } from "@/components/stats/redesign/speed-tiles";
import { ConnectionStatus, LastUpdated, metricsUnsupportedNote } from "@/components/stats/live-panels";
import { RequestsPanel } from "@/components/stats/requests-panel";
import { SystemConfigSection } from "@/components/stats/system-config-section";
import { PowerChart, TokensChart, UtilChart } from "@/components/stats/v2-charts";
import { Skeleton } from "@/components/ui/skeleton";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { CacheResponse } from "@/lib/cache-obs";
import {
  WINDOW_META,
  bucketPoints,
  hostBusyMinutes,
  maxOf,
  meanOf,
  referenceFor,
  tokenRateBuckets,
} from "@/lib/live-history";
import {
  type LiveEngineFrame,
  type LiveRequestsSnapshot,
  type PrefillModelSnapshot,
  liveFramesOf,
} from "@/lib/live-stats";
import { useLiveStats } from "@/lib/live-stats-stream";
import type { RequestHistoryResponse } from "@/lib/request-history";
import {
  STATS_RANGES,
  type StatsRange,
  type StatsV2Bookkeeping,
  type StatsV2Overview,
} from "@/lib/stats-v2";
import {
  FIXED,
  type GroupBy,
  type SpeedResponse,
  type SpeedSummary,
  adminState,
  isAsleep,
  thermalNotice,
} from "@/lib/stats-redesign";
import type { GpuLiveResponse } from "@/lib/system-info";
import { usePersistedRange } from "@/lib/use-persisted-range";

const RANGE_KEY = "vw.stats.range";
const MODEL_KEY = "vw.stats.scope-model";
const REFRESH_MS = 30_000;
const REQ_REFRESH_MS = 1_500;
const HISTORY_REFRESH_MS = 15_000;
const SPARK_LEN = 150; // 5 min at the 2 s SSE cadence

const swr = { refreshInterval: REFRESH_MS, keepPreviousData: true };

function useScopeModel(ids: readonly string[]): [string | null, (id: string) => void] {
  const [chosen, setChosen] = useState<string | null>(() => {
    try {
      return window.localStorage.getItem(MODEL_KEY);
    } catch {
      return null;
    }
  });
  const sel = chosen && ids.includes(chosen) ? chosen : (ids[0] ?? null);
  const set = (id: string) => {
    setChosen(id);
    try {
      window.localStorage.setItem(MODEL_KEY, id);
    } catch {
      /* private mode */
    }
  };
  return [sel, set];
}

/** The 5-minute sparkline per model: starts from the server ring (history_5m on
 *  the connection's first frame), then one point per frame. */
function useSpark(frame: LiveEngineFrame | null): Record<string, SparkPoint[]> {
  const [rings, setRings] = useState<Record<string, SparkPoint[]>>({});
  const lastTs = useRef<string | null>(null);
  useEffect(() => {
    if (!frame || frame.ts === lastTs.current) return;
    lastTs.current = frame.ts;
    setRings((prev) => {
      const next = { ...prev };
      for (const b of liveFramesOf(frame)) {
        if (!b.model_id) continue;
        const seed = b.history_5m && !prev[b.model_id] ? b.history_5m.map((h) => ({ gen: h.gen, prefill: h.prefill })) : prev[b.model_id] ?? [];
        const pt = { gen: b.now?.generation_tps_10s ?? b.throughput?.generation_tokens_per_s ?? null, prefill: b.now?.prefill_computed_tps_10s ?? null };
        next[b.model_id] = [...seed, pt].slice(-SPARK_LEN);
      }
      return next;
    });
  }, [frame]);
  return rings;
}

export default function StatsPage() {
  const [range, setRange] = usePersistedRange(RANGE_KEY, "1h");
  const meta = WINDOW_META[range];

  // ---- live engine stream: the scope's model block ---------------------------
  const engine = useLiveStats();
  const frame = engine.frame;
  const blocks = useMemo(() => liveFramesOf(frame).filter((b) => b.model_id !== null), [frame]);
  const ids = useMemo(() => blocks.map((b) => b.model_id as string), [blocks]);
  const [model, setModel] = useScopeModel(ids);
  const block = blocks.find((b) => b.model_id === model);
  const spark = useSpark(frame);

  // ---- feeds, all for the scope's model ----------------------------------------
  const mq = model ? `&models=${encodeURIComponent(model)}` : "";
  const host = useSWR<StatsV2Overview>(`/api/stats/v2/overview?range=${range}`, authFetchJSON, swr);
  const refHost = useSWR<StatsV2Overview>("/api/stats/v2/overview?range=24h", authFetchJSON, swr);
  const scoped = useSWR<StatsV2Overview>(model ? `/api/stats/v2/overview?range=${range}${mq}` : null, authFetchJSON, swr);
  const speed = useSWR<SpeedResponse>(
    model ? `/api/stats/v2/speed?range=${range}&model=${encodeURIComponent(model)}` : null,
    authFetchJSON,
    swr,
  );
  const [group, setGroup] = useState<GroupBy>("none");
  const groups = useSWR<SpeedResponse>(group !== "none" ? `/api/stats/v2/speed?range=${range}&group=${group}` : null, authFetchJSON, swr);
  const cache = useSWR<CacheResponse>(model ? `/api/stats/v2/cache?range=${range}${mq}` : null, authFetchJSON, swr);
  const requests = useSWR<LiveRequestsSnapshot>("/api/stats/requests", authFetchJSON, {
    refreshInterval: REQ_REFRESH_MS,
    keepPreviousData: true,
  });
  const prefill = useSWR<PrefillModelSnapshot>("/api/stats/v2/prefill-model", authFetchJSON, {
    refreshInterval: 60_000,
    keepPreviousData: true,
  });
  const history = useSWR<RequestHistoryResponse>(
    model ? `/api/stats/v2/requests?range=${range}${mq}` : null,
    authFetchJSON,
    { refreshInterval: HISTORY_REFRESH_MS, keepPreviousData: true },
  );
  const bookkeeping = useSWR<StatsV2Bookkeeping>("/api/stats/v2/bookkeeping", authFetchJSON, {
    refreshInterval: 60_000,
    keepPreviousData: true,
  });

  // One install (the 80 % case): no install label anywhere; the scope bar's
  // model tab is the whole scope. Multisite (B12) will name installs.
  const liveRows = useMemo(() => requests.data?.requests ?? [], [requests.data]);

  // ---- tabs: one per loaded model with its own state ---------------------------
  const tabs: ScopeTab[] = blocks.map((b) => ({
    id: b.model_id as string,
    name: b.model ?? (b.model_id as string),
    state: adminState({
      running: b.engine?.num_requests_running ?? null,
      engineWaiting: b.engine?.num_requests_waiting ?? null,
      wardenQueue: b.now?.warden_queue ?? null,
      asleep: isAsleep(b),
      noData: b.engine === null && b.metrics_supported !== false,
    }),
  }));
  const tab = tabs.find((t) => t.id === model);

  // ---- GPU throttling: the one prominent place for it (notice line) ---------------
  const gpus = useSWR<GpuLiveResponse>("/api/system/gpus", authFetchJSON, { refreshInterval: 10_000, keepPreviousData: true });
  const thermal = thermalNotice(gpus.data);

  // ---- notice line: the most important notice, one reserved line ----------------
  const gpuProbe = host.data?.current.gpu_probe;
  const gpuLost = gpuProbe?.state === "failing";
  const dropped = bookkeeping.data?.total?.count ?? 0;
  const notices: { strong: string; rest?: string }[] = [];
  if (engine.status === "terminal-error")
    notices.push({ strong: "Live engine stream disconnected", rest: "reload the page to reconnect; window panels keep updating" });
  if (thermal) notices.push(thermal);
  if (block?.scrape_error) notices.push({ strong: `Metrics scrape failed for ${block.model ?? "this model"}`, rest: block.scrape_error });
  if (gpuLost) notices.push({ strong: "GPU telemetry unavailable", rest: gpuProbe?.error ?? "the GPU probe is failing" });
  if (block?.metrics_supported === false) notices.push({ strong: "Engine reports no metrics", rest: metricsUnsupportedNote(block.backend) });
  if (dropped > 0) notices.push({ strong: `${dropped} stats writes dropped`, rest: "since the warden started; some request figures are missing" });
  if (host.error && !host.data) notices.push({ strong: "Failed to load stats", rest: host.error instanceof Error ? host.error.message : undefined });

  // ---- host ---------------------------------------------------------------------
  const series = host.data?.series;
  const utilSeries = useMemo(
    () => bucketPoints(series?.util ?? [], meta.bucketMinutes, (b) => ({ minute: b[0].minute, max_pct: Math.round(maxOf(b.map((p) => p.max_pct))) })),
    [series, meta.bucketMinutes],
  );
  const powerSeries = useMemo(
    () => bucketPoints(series?.power ?? [], meta.bucketMinutes, (b) => ({ minute: b[0].minute, watts: meanOf(b.map((p) => p.watts)) })),
    [series, meta.bucketMinutes],
  );
  const references = useMemo(() => {
    const s = refHost.data?.series;
    if (!s) return undefined;
    const busy = hostBusyMinutes(s.util, s.tokens);
    return { util: referenceFor(s.util, (p) => p.max_pct, busy), power: referenceFor(s.power, (p) => p.watts, busy) };
  }, [refHost.data]);
  const tokensData = scoped.data;
  const tokenBuckets = useMemo(
    () =>
      tokenRateBuckets(tokensData?.series.tokens ?? [], {
        bucketMinutes: meta.bucketMinutes,
        sinceMinute: tokensData?.since_minute ?? 0,
        nowMinute: tokensData?.now_minute ?? 0,
        measuredFromMinute: tokensData?.tokens_measured_from_minute ?? null,
      }),
    [tokensData, meta.bucketMinutes],
  );
  const powerSupported = host.data ? host.data.current.power_supported !== false : true;
  const data = host.data;
  // Rows of the scope's model that are still prefilling: the engine counts a
  // prompt only once its prefill completes, so "no prefill running" would be
  // false while a big prompt is being read.
  const prefilling = liveRows.filter((r) => r.model === block?.model && r.phase === "prefill").length;

  // ---- derived scope figures ------------------------------------------------------
  const sum = speed.data?.summary;
  const p50 = (s: SpeedSummary | undefined) => (s && (s.mode === "max" || s.mode === "p95") ? s.p50 : null);
  const typical = model ? (prefill.data?.models?.[model]?.rate_tok_s ?? null) : null;
  const cacheRow = cache.data?.models.find((m) => m.model_id === model);
  const scopeName = block?.model ?? model ?? "no model";
  const scopeLabel = `window · ${scopeName}`;
  const nowEpoch = Date.now() / 1000;
  const sinceEpoch = nowEpoch - ({ "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800 } as const)[range];
  const reqChartRef = useRef<HTMLDivElement>(null);
  const lensOn = usePromptLensEnabled();
  const router = useRouter();
  const openInLens = useCallback((id: string) => router.push(promptLensHref("admin", { req: id })), [router]);

  return (
    <div data-testid="stats-page" className="mx-auto max-w-[1180px] pb-16">
      {/* Header: title, Live, updated, window ------------------------------------ */}
      <div data-testid="stats-header" className="flex h-8 items-center gap-2.5 whitespace-nowrap">
        <h1 className="text-[22px] font-semibold">Stats</h1>
        <ConnectionStatus state={engine} />
        {/* "updated N s ago" yields its room to the window selector on phones. */}
        <span className="hidden min-w-0 overflow-hidden sm:inline">{frame && <LastUpdated ts={frame.ts} />}</span>
        <div role="group" aria-label="Window" data-testid="range-selector" className="ml-auto inline-flex flex-none overflow-hidden rounded-md border border-chat-rule">
          {STATS_RANGES.map((r: StatsRange) => (
            <button
              key={r}
              type="button"
              aria-pressed={r === range}
              data-active={r === range}
              onClick={() => setRange(r)}
              className={r === range ? "bg-chat-surface-2 px-2.5 py-0.5 text-xs text-chat-accent" : "px-2.5 py-0.5 text-xs text-chat-muted hover:text-chat-fg"}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
      <ScopeBar tabs={tabs} selected={model} onSelect={setModel} />
      <NoticeLine notices={notices} />

      <Section title="Now" scope={`live · ${scopeName}`} note="SSE every 2 s · speeds smoothed over 10 s" testid="now">
        <NowCard
          modelId={model}
          modelName={block?.model ?? null}
          state={tab?.state ?? "No data"}
          prefilling={prefilling}
          running={block?.engine?.num_requests_running ?? null}
          engineWaiting={block?.engine?.num_requests_waiting ?? null}
          wardenQueue={block?.now?.warden_queue ?? null}
          genNow={block?.now?.generation_tps_10s ?? block?.throughput?.generation_tokens_per_s ?? null}
          prefillNow={block?.now?.prefill_computed_tps_10s ?? null}
          kvPct={block?.engine?.kv_cache_usage_perc ?? null}
          maxModelLen={block?.max_model_len ?? null}
          windowP50Ttft={p50(sum?.ttft)}
          windowP50Decode={p50(sum?.decode)}
          windowN={sum?.ttft.n ?? 0}
          typicalPrefill={typical}
          lastFinishedAt={speed.data?.last_finished_at ?? null}
          spark={model ? (spark[model] ?? []) : []}
        />
      </Section>

      <Section
        title="Latency and speed"
        scope={scopeLabel}
        note="per request · p50 large · p95 small · n · percentiles hidden when n is too small"
        testid="latency"
      >
        <SpeedTiles data={speed.data} view="admin" typicalPrefill={typical} onJumpToRequests={() => reqChartRef.current?.scrollIntoView({ behavior: "smooth" })} />
        <LatencyScatter data={speed.data} view="admin" sinceEpoch={sinceEpoch} nowEpoch={nowEpoch} />
      </Section>

      <Section title="Tokens / second" scope={scopeLabel} note={meta.bucketLabel} testid="tokens">
        <FixedCard height={FIXED.tokensChart} testid="tokens-card">
          {!tokensData ? (
            <Skeleton className="h-96 w-full" />
          ) : (
            <TokensChart buckets={tokenBuckets} bucketMinutes={meta.bucketMinutes} range={range} />
          )}
        </FixedCard>
      </Section>

      <Section title="Cache" scope={scopeLabel} note="one defined figure for the scope's model · per key: Group by key in the table below" testid="cache">
        <AdminCacheCard row={cacheRow} learnedRate={typical} hintRate={prefill.data?.hint_tok_s ?? null} />
      </Section>

      {/* Host (D4: below the request-facing panels), host-wide ------------------- */}
      {/* Host (D4: below the request-facing panels), host-wide. The VRAM / GPU
          util / power numbers live in the header badges and the per-GPU cards
          (System configuration): the charts here are their history. */}
      <Section title="Host" scope="host-wide" note="current figures: header badges and the per-GPU cards below" testid="host">
        <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-2">
          <div data-testid="chart-util" className="space-y-2">
            <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-chat-muted">GPU utilisation</h3>
            {!data ? <Skeleton className="h-64 w-full" /> : <UtilChart points={utilSeries} range={range} reference={references?.util} />}
          </div>
          {(
            <div data-testid="chart-power" className="space-y-2">
              <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-chat-muted">Power draw</h3>
              {!data ? (
                <Skeleton className="h-64 w-full" />
              ) : (
                <PowerChart points={powerSeries} range={range} supported={powerSupported} reference={references?.power} />
              )}
            </div>
          )}
        </div>
      </Section>

      <SystemConfigSection />

      <Section title="Engine details" scope={`since engine start · ${scopeName}`} testid="engine">
        <EngineDetails frame={block} />
      </Section>

      {/* Bottom: the only lists of requests and the only place for breakdowns ---- */}
      <Section title="Requests" scope={`${scopeLabel}`} note="breakdowns by key, model and client IP live only here" testid="requests">
        <div ref={reqChartRef}>
          <RequestsPanel
            data={history.data}
            range={range}
            human={meta.human}
            isLoading={history.isLoading && !history.data}
            error={history.error}
          />
        </div>
        <div className="mt-4">
          <AdminRequestsTable
            rows={liveRows}
            scopeModel={block?.model ?? null}
            multiModel={blocks.length > 1}
            windowGroups={groups.data?.groups}
            onGroupChange={setGroup}
            lastFinishedAt={speed.data?.last_finished_at ?? null}
            range={range}
            onEye={lensOn ? openInLens : undefined}
          />
        </div>
      </Section>
    </div>
  );
}
