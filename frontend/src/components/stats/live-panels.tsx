"use client";

// Live panels for the merged /stats page — everything that used to live on
// /stats/live, rebuilt around the design decisions in the approved mockup:
//
//   * every number states its time scope: the live timeline says "last 5 min
//     · not the window", cumulative figures are greyed and labelled "since
//     engine start";
//   * absent metrics render as absent, never as 0 (design spec §9.3) — a
//     llama.cpp engine that publishes no KV gauge gets "not reported", and a
//     combined figure carries its provenance ("N of M models report this");
//   * colour is never the only signal — in retro-dark `--chat-positive` and
//     `--chat-warn` are the same amber, so states also differ by shape
//     (pill vs rounded rect) or wording.
//
// Everything here paints through the app's `--chat-*` theme tokens (the
// `chat-*` Tailwind utilities from @podwarden/chat-ui's preset) — no literal
// colours — so retro / retro-dark / any future theme just work. SVG marks use
// `currentColor` inherited from a token-classed wrapper for the same reason.

import { useEffect, useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import type { LiveStatsState } from "@/lib/live-stats-stream";
import {
  formatCompact,
  formatElapsed,
  formatInt,
  formatLatency,
  formatPct,
  isPartial,
  pressureOf,
  sumReported,
  type CombinedMetric,
  type HistogramBuckets,
  type LiveEngineFrame,
  type LiveRequestRow,
  type Pressure,
} from "@/lib/live-stats";
import {
  bucketIncrements,
  quantileFromBuckets,
  type TimelinePoint,
} from "@/lib/live-history";

// Display names for the frame's `backend` field, so an empty state can say
// WHICH engine is silent rather than "this engine build". Falls back to a
// neutral phrase for a backend this build of the UI has not heard of.
export const ENGINE_LABELS: Record<string, string> = {
  vllm: "vLLM",
  llamacpp: "llama.cpp",
};

// ===========================================================================
// Shared primitives
// ===========================================================================

export function Panel({
  title,
  right,
  children,
  className,
  bodyClassName,
  testid,
}: {
  title: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  testid?: string;
}) {
  return (
    <section
      data-testid={testid}
      className={cn("rounded-lg border border-chat-rule bg-chat-surface/50", className)}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-chat-rule px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-chat-muted">
          {title}
        </h2>
        {right}
      </div>
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

/** The right-hand scope note every panel heading carries. */
export function ScopeNote({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[11px] normal-case tracking-normal text-chat-dim">
      {children}
    </span>
  );
}

// Fill classes for the pressure meter. The numeric label beside every meter
// carries the exact figure, and the threshold ticks stay visible, so the
// encoding never rests on colour alone (retro-dark maps positive and warn to
// the same amber on purpose).
const PRESSURE_FILL: Record<Pressure, string> = {
  healthy: "bg-chat-positive",
  warm: "bg-chat-warn",
  hot: "bg-chat-negative",
};

export function Meter({
  fraction,
  className,
  showTicks = true,
}: {
  fraction: number;
  className?: string;
  showTicks?: boolean;
}) {
  const clamped = Math.max(0, Math.min(1, fraction));
  return (
    <div
      className={cn(
        "relative w-full overflow-hidden rounded-full bg-chat-surface-2",
        className,
      )}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-500 ease-out motion-reduce:transition-none",
          PRESSURE_FILL[pressureOf(clamped)],
        )}
        style={{ width: `${clamped * 100}%` }}
      />
      {showTicks && (
        <>
          <span aria-hidden="true" className="absolute inset-y-0 left-[60%] w-px bg-chat-page/40" />
          <span aria-hidden="true" className="absolute inset-y-0 left-[85%] w-px bg-chat-page/40" />
        </>
      )}
    </div>
  );
}

// ===========================================================================
// Header: connection dot + engine state chip
// ===========================================================================

const STATUS_META: Record<
  LiveStatsState["status"],
  { dotClass: string; label: string; pulse: boolean }
> = {
  connecting: { dotClass: "bg-chat-dim", label: "Connecting", pulse: true },
  connected: { dotClass: "bg-chat-positive", label: "Live", pulse: true },
  reconnecting: { dotClass: "bg-chat-warn", label: "Reconnecting", pulse: true },
  "terminal-error": { dotClass: "bg-chat-negative", label: "Disconnected", pulse: false },
};

export function ConnectionStatus({ state }: { state: LiveStatsState }) {
  const meta = STATUS_META[state.status];
  return (
    <span
      className="inline-flex items-center gap-1.5 text-xs text-chat-muted"
      role="status"
      aria-label={`Engine stream ${meta.label}`}
    >
      <span
        aria-hidden="true"
        className={cn(
          "h-2 w-2 rounded-full",
          meta.dotClass,
          meta.pulse && "motion-safe:animate-pulse",
        )}
      />
      {meta.label}
    </span>
  );
}

export type EngineState = "healthy" | "queueing" | "idle" | "none";

/**
 * The one part of the old verdict strip that was not a restatement of
 * something else on the page. running/waiting live in the timeline and the
 * in-flight table, tok/s is a KPI — the WORD survives, the strip does not.
 */
export function engineStateOf(
  selectedFrames: readonly LiveEngineFrame[],
): EngineState {
  if (selectedFrames.length === 0) return "none";
  let running = 0;
  let waiting = 0;
  for (const f of selectedFrames) {
    running += f.engine?.num_requests_running ?? 0;
    waiting += f.engine?.num_requests_waiting ?? 0;
  }
  if (running === 0 && waiting === 0) return "idle";
  return waiting > 0 ? "queueing" : "healthy";
}

const STATE_META: Record<EngineState, { label: string; className: string; title: string }> = {
  healthy: {
    label: "Healthy",
    // Rounded RECT — queueing is a pill. In retro-dark positive and warn are
    // the same amber, so the shape has to carry the difference.
    className:
      "rounded-md border border-chat-positive/50 bg-chat-positive/10 text-chat-positive",
    title: "Decode-bound and keeping up. No queue.",
  },
  queueing: {
    label: "Queueing",
    className:
      "rounded-full border border-chat-warn/50 bg-chat-warn/10 text-chat-warn",
    title:
      "More requests than decode slots — the queue, not the GPU, is what callers feel.",
  },
  idle: {
    label: "Idle",
    className: "rounded-md border border-chat-rule text-chat-dim",
    title: "No requests in flight for this selection.",
  },
  none: {
    label: "No models",
    className: "rounded-md border border-chat-rule text-chat-dim",
    title: "Nothing is loaded; host figures continue regardless.",
  },
};

export function StateChip({ state }: { state: EngineState }) {
  const meta = STATE_META[state];
  return (
    <span
      data-testid="engine-state-chip"
      data-state={state}
      title={meta.title}
      className={cn("px-2.5 py-0.5 text-xs font-semibold", meta.className)}
    >
      {meta.label}
    </span>
  );
}

// Self-contained 1s ticker so "updated Ns ago" stays live without
// re-rendering the whole page each second.
export function LastUpdated({ ts }: { ts: string }) {
  const [, setNow] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setNow((n) => n + 1), 1_000);
    return () => clearInterval(id);
  }, []);
  const ms = Date.parse(ts);
  if (Number.isNaN(ms)) return null;
  const ago = Math.max(0, Math.round((Date.now() - ms) / 1000));
  return (
    <span className="text-xs tabular-nums text-chat-dim" title={ts}>
      updated {ago}s ago
    </span>
  );
}

// ===========================================================================
// 5-minute rolling timeline
// ===========================================================================

/** Split a series into contiguous non-null runs so an absent sample renders
 *  as a GAP in the line, never as a dip to zero. */
function segments(
  points: readonly TimelinePoint[],
  value: (p: TimelinePoint) => number | null,
): TimelinePoint[][] {
  const out: TimelinePoint[][] = [];
  let cur: TimelinePoint[] = [];
  for (const p of points) {
    if (value(p) === null) {
      if (cur.length > 1) out.push(cur);
      cur = [];
    } else {
      cur.push(p);
    }
  }
  if (cur.length > 1) out.push(cur);
  return out;
}

export function Timeline({
  points,
  spanMs,
  now,
}: {
  points: readonly TimelinePoint[];
  spanMs: number;
  now: number;
}) {
  const W = 600;
  const H = 120;
  const x = (at: number) => ((at - (now - spanMs)) / spanMs) * W;
  const gens = points.map((p) => p.gen).filter((v): v is number => v !== null);
  const runs = points
    .map((p) => p.running)
    .filter((v): v is number => v !== null);
  const waits = points
    .map((p) => p.waiting)
    .filter((v): v is number => v !== null);
  const maxGen = Math.max(60, ...gens) * 1.15;
  const maxRun = Math.max(4, ...runs, ...waits) * 1.25;
  const yGen = (v: number) => H - (v / maxGen) * H;
  const yRun = (v: number) => H - (v / maxRun) * H;

  // Idle bands: an explicit "nothing happened here", which is why the chart
  // does not look broken at rest.
  const idleRects: { x0: number; x1: number }[] = [];
  let start: number | null = null;
  for (let i = 0; i <= points.length; i++) {
    const idle = i < points.length && points[i].idle;
    if (idle && start === null) start = points[i].at;
    if (!idle && start !== null) {
      idleRects.push({ x0: x(start), x1: x(points[i - 1].at) });
      start = null;
    }
  }

  const poly = (pts: TimelinePoint[], y: (v: number) => number, v: (p: TimelinePoint) => number | null) =>
    pts.map((p) => `${x(p.at).toFixed(1)},${y(v(p) as number).toFixed(1)}`).join(" ");

  return (
    <div data-testid="live-timeline">
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-32 w-full"
          aria-hidden="true"
        >
          <g className="text-chat-dim">
            {idleRects.map((r, i) => (
              <rect
                key={i}
                x={r.x0}
                y={0}
                width={Math.max(0.5, r.x1 - r.x0)}
                height={H}
                fill="currentColor"
                opacity={0.1}
              />
            ))}
          </g>
          <g className="text-chat-rule">
            {[0.25, 0.5, 0.75].map((f) => (
              <line
                key={f}
                x1={0}
                x2={W}
                y1={H * f}
                y2={H * f}
                stroke="currentColor"
                strokeDasharray="2 4"
                strokeWidth={1}
              />
            ))}
          </g>
          <g className="text-chat-accent">
            {segments(points, (p) => p.gen).map((seg, i) => (
              <polyline
                key={`g${i}`}
                points={poly(seg, yGen, (p) => p.gen)}
                fill="none"
                stroke="currentColor"
                strokeWidth={1.8}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
          <g className="text-chat-positive">
            {segments(points, (p) => p.running).map((seg, i) => (
              <polyline
                key={`r${i}`}
                points={poly(seg, yRun, (p) => p.running)}
                fill="none"
                stroke="currentColor"
                strokeWidth={1.3}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
          <g className="text-chat-negative">
            {segments(points, (p) => p.waiting).map((seg, i) => (
              <polyline
                key={`w${i}`}
                points={poly(seg, yRun, (p) => p.waiting)}
                fill="none"
                stroke="currentColor"
                strokeWidth={1.3}
                strokeDasharray="4 3"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </g>
        </svg>
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-chat-dim">
        <span>5m ago</span>
        <span>{Math.round(maxGen)} tok/s full scale</span>
        <span>now</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-chat-muted">
        <LegendSwatch className="bg-chat-accent" label="generation tok/s" />
        <LegendSwatch className="bg-chat-positive" label="running" />
        <LegendSwatch className="bg-chat-negative" label="waiting (dashed)" />
        <LegendSwatch className="bg-chat-dim/40" label="idle (no traffic)" />
      </div>
    </div>
  );
}

function LegendSwatch({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden="true" className={cn("h-2 w-2 rounded-sm", className)} />
      {label}
    </span>
  );
}

// ===========================================================================
// Latency distribution (cumulative buckets in the engine-histogram shape;
// served from the per-request store, see components/stats/latency-panels.tsx)
// ===========================================================================

/** Fractional bar-space position of a value inside the bucket edges. */
function markerPosition(hist: HistogramBuckets, v: number): number | null {
  let prevLe = 0;
  for (let i = 0; i < hist.le.length; i++) {
    const le = hist.le[i];
    if (le === null) return (i + 0.5) / hist.le.length;
    if (v <= le) {
      const frac = le > prevLe ? (v - prevLe) / (le - prevLe) : 0.5;
      return (i + Math.max(0, Math.min(1, frac))) / hist.le.length;
    }
    prevLe = le;
  }
  return null;
}

function tickLabels(hist: HistogramBuckets): { pct: number; label: string }[] {
  const n = hist.le.length;
  if (n === 0) return [];
  const picks = [0.2, 0.45, 0.7, 0.9]
    .map((f) => Math.min(n - 1, Math.round(f * n)))
    .filter((v, i, a) => a.indexOf(v) === i);
  return picks
    .filter((i) => hist.le[i] !== null)
    .map((i) => ({
      pct: ((i + 1) / n) * 100,
      label: formatLatency(hist.le[i] as number).replace(" ", ""),
    }));
}

export function LatencyHistogram({
  hist,
  markers,
  testid,
}: {
  /** Buckets with counts cumulative along the boundaries. */
  hist: HistogramBuckets;
  /** Percentile markers drawn ON the measured distribution. */
  markers: { label: string; value: number | null }[];
  testid?: string;
}) {
  const bars = bucketIncrements(hist);
  const max = Math.max(1, ...bars);
  const W = 600;
  const H = 110;
  const bw = W / bars.length;
  return (
    <div data-testid={testid}>
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="h-28 w-full"
          aria-hidden="true"
        >
          <g className="text-chat-accent">
            {bars.map((v, i) =>
              v > 0 ? (
                <rect
                  key={i}
                  x={i * bw + 0.6}
                  y={H - (v / max) * H}
                  width={Math.max(1, bw - 1.2)}
                  height={(v / max) * H}
                  fill="currentColor"
                  opacity={0.85}
                />
              ) : null,
            )}
          </g>
          <g className="text-chat-fg">
            {markers.map((m) => {
              if (m.value === null) return null;
              const pos = markerPosition(hist, m.value);
              if (pos === null) return null;
              return (
                <line
                  key={m.label}
                  x1={pos * W}
                  x2={pos * W}
                  y1={0}
                  y2={H}
                  stroke="currentColor"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
          </g>
          <line
            x1={0}
            x2={W}
            y1={H - 0.5}
            y2={H - 0.5}
            stroke="currentColor"
            className="text-chat-rule"
          />
        </svg>
        <div className="pointer-events-none absolute inset-0" aria-hidden="true">
          {tickLabels(hist).map((t) => (
            <span
              key={t.pct}
              style={{ left: `${t.pct}%` }}
              className="absolute bottom-0 -translate-x-full font-mono text-[9px] text-chat-dim"
            >
              {t.label}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-3 font-mono text-[11px] text-chat-muted">
        {markers.map((m) => (
          <span key={m.label}>
            {m.label}{" "}
            <span className="text-chat-fg">
              {m.value === null ? "—" : formatLatency(m.value)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ===========================================================================
// In-flight requests table
// ===========================================================================

function PhasePill({ phase }: { phase: string }) {
  const decode = phase === "decode";
  return (
    <span
      className={cn(
        // Decode is a rounded rect, prefill a pill — shape as well as tone,
        // because retro-dark's positive and warn are the same amber.
        "inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-medium",
        decode
          ? "rounded-md bg-chat-positive/15 text-chat-positive"
          : "rounded-full bg-chat-warn/15 text-chat-warn",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          decode ? "bg-chat-positive" : "bg-chat-warn motion-safe:animate-pulse",
        )}
      />
      {phase}
    </span>
  );
}

function ContextBar({
  used,
  total,
  pct,
}: {
  used: number;
  total: number;
  pct: number;
}) {
  const clamped = Math.max(0, Math.min(1, pct));
  return (
    <div className="flex min-w-[14rem] items-center gap-2.5">
      <Meter fraction={clamped} className="h-2 flex-1" showTicks={false} />
      <span className="shrink-0 font-mono text-xs tabular-nums text-chat-muted">
        {formatCompact(used)}/{formatCompact(total)}
      </span>
      <span className="w-9 shrink-0 text-right font-mono text-xs font-medium tabular-nums text-chat-fg">
        {Math.round(clamped * 100)}%
      </span>
    </div>
  );
}

export function LiveRequestsPanel({
  rows,
  error,
  isLoading,
}: {
  // Already narrowed to the selected models by the page. Taking rows rather
  // than the whole snapshot is what makes it impossible for this panel and
  // the finished table below it to disagree about which requests exist.
  rows: LiveRequestRow[];
  error: unknown;
  isLoading: boolean;
}) {
  return (
    <Panel
      title="In flight"
      testid="live-requests"
      right={
        <span className="text-xs tabular-nums text-chat-dim">
          {`${rows.length} active`}
        </span>
      }
      bodyClassName="p-0"
    >
      {isLoading ? (
        <div className="p-4">
          <div className="h-10 w-full animate-pulse rounded bg-chat-surface-2/60" />
        </div>
      ) : error && rows.length === 0 ? (
        <p className="p-4 text-sm text-chat-negative">
          Failed to load live requests
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      ) : rows.length === 0 ? (
        <p className="p-6 text-center text-sm text-chat-dim">
          Nothing in flight for this selection. Live sessions appear here the
          moment they hit the engine.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[56rem] text-sm">
            <thead className="border-b border-chat-rule text-left text-[11px] uppercase tracking-wider text-chat-dim">
              <tr>
                <th className="px-4 py-2 font-medium">Token</th>
                <th className="px-3 py-2 font-medium">Client IP</th>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Phase</th>
                <th className="px-3 py-2 font-medium">Context window</th>
                <th className="px-3 py-2 text-right font-medium">Elapsed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-chat-rule/70">
              {rows.map((r) => (
                <tr key={r.id} className="text-chat-fg">
                  <td className="px-4 py-2.5">
                    <span className="font-mono text-xs">
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
                    </span>
                    {r.orphan && (
                      <span
                        title="Client disconnected but the upstream is still draining."
                        className="ml-2 rounded bg-chat-negative/15 px-1.5 py-0.5 text-[10px] font-medium uppercase text-chat-negative"
                      >
                        orphan
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-xs text-chat-muted">
                    {r.client_ip ?? "—"}
                  </td>
                  <td
                    className="max-w-[12rem] truncate px-3 py-2.5 font-mono text-xs text-chat-muted"
                    title={r.model}
                  >
                    {r.model}
                  </td>
                  <td className="px-3 py-2.5">
                    <PhasePill phase={r.phase} />
                  </td>
                  <td className="px-3 py-2.5">
                    <ContextBar
                      used={r.context_tokens}
                      total={r.max_model_len}
                      pct={r.context_pct}
                    />
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono tabular-nums text-chat-muted">
                    {formatElapsed(r.elapsed_s)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

// ===========================================================================
// Per-model context & cache rows — never combined across models
// ===========================================================================

/** How many per-model rows render before the tail collapses to a line. */
export const KV_ROWS_SHOWN = 4;

export function ModelContextRows({ frames }: { frames: readonly LiveEngineFrame[] }) {
  const shown = frames.slice(0, KV_ROWS_SHOWN);
  const extra = frames.length - shown.length;
  if (frames.length === 0) {
    return <p className="text-sm text-chat-dim">No models selected.</p>;
  }
  return (
    <div className="space-y-4">
      {shown.map((f) => (
        <ModelContextRow key={f.model_id ?? f.model ?? "?"} frame={f} />
      ))}
      {extra > 0 && (
        <p data-testid="kv-rows-overflow" className="text-xs text-chat-dim">
          +{extra} more selected — select fewer to compare.
        </p>
      )}
    </div>
  );
}

function ModelContextRow({ frame }: { frame: LiveEngineFrame }) {
  const e = frame.engine;
  const engineLabel = ENGINE_LABELS[frame.backend ?? ""] ?? "this engine";
  // A null FRAME (failed scrape: engine/cache/latency all null but model_id
  // kept) is not an engine that reports nothing — it is a tick with no data.
  // Say that, rather than borrowing the "does not expose this metric" copy
  // that belongs to llama.cpp.
  if (e === null && frame.cache === null && frame.scrape_error !== null) {
    return (
      <div
        data-testid="model-context-row"
        data-model-id={frame.model_id ?? ""}
        className="border-t border-chat-rule pt-3 first:border-t-0 first:pt-0"
      >
        <span
          data-testid="live-model-name"
          className="max-w-[24ch] truncate font-mono text-xs text-chat-fg"
          title={frame.model ?? undefined}
        >
          {frame.model}
        </span>
        <p className="mt-1 text-xs text-chat-dim" data-testid="kv-usage">
          No engine data this tick — see the scrape warning above.
        </p>
      </div>
    );
  }
  // `null` means the engine does not report KV usage at all (llama.cpp has no
  // such gauge). Clamping to 0 would render "0%" against an empty meter --
  // "plenty of headroom" -- a confident, precise, wrong answer.
  const kv =
    e === null || e.kv_cache_usage_perc === null
      ? null
      : Math.max(0, Math.min(1, e.kv_cache_usage_perc));
  const interval = frame.cache?.prefix_hit_rate ?? null;
  const cumulative = frame.cache?.prefix_hit_rate_cumulative ?? null;
  return (
    <div
      data-testid="model-context-row"
      data-model-id={frame.model_id ?? ""}
      className="border-t border-chat-rule pt-3 first:border-t-0 first:pt-0"
    >
      <div className="flex flex-wrap items-baseline gap-2">
        <span
          data-testid="live-model-name"
          className="max-w-[24ch] truncate font-mono text-xs text-chat-fg"
          title={frame.model ?? undefined}
        >
          {frame.model}
        </span>
        {frame.backend && (
          <span className="rounded-full border border-chat-rule px-2 py-0.5 text-[10px] uppercase tracking-wide text-chat-dim">
            {engineLabel}
          </span>
        )}
        {frame.max_model_len !== null && (
          <span className="font-mono text-[11px] text-chat-dim">
            window {formatCompact(frame.max_model_len)}
          </span>
        )}
        <span className="ml-auto text-[11px] text-chat-dim">
          {/* `sleep null` shipped once. An engine with no sleep state is not
              asleep and not awake — it has no answer, and a dash says so. */}
          {e === null || e.engine_sleep_state === null
            ? "—"
            : e.engine_sleep_state === 0
              ? "awake"
              : `sleep ${e.engine_sleep_state}`}
        </span>
      </div>
      {kv === null ? (
        <p className="mt-2 text-xs text-chat-dim" data-testid="kv-usage">
          KV usage not reported — {engineLabel} does not expose a KV-cache
          usage metric.
        </p>
      ) : (
        <div className="mt-2">
          <div className="flex items-baseline justify-between gap-3">
            <span className="font-mono text-sm text-chat-fg" data-testid="kv-usage">
              {formatPct(kv)}
            </span>
            <span className="font-mono text-xs text-chat-muted">
              {formatCompact(e?.kv_tokens_used)} /{" "}
              {formatCompact(e?.kv_tokens_total)} tokens
            </span>
          </div>
          <Meter fraction={kv} className="mt-1.5 h-2.5" />
          <p className="mt-1 text-[11px] leading-relaxed text-chat-dim">
            KV pool for this engine. Includes blocks the prefix cache still
            holds from finished requests, which is why it can exceed any single
            prompt.
          </p>
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-xs">
        <span className="text-chat-muted">
          prefix cache{" "}
          <span className="font-mono text-chat-fg">
            {interval === null ? "—" : formatPct(interval)}
          </span>{" "}
          <span className="text-chat-dim">this interval</span>
        </span>
        {/* Cumulative is a LIFETIME figure: greyed and labelled, so it cannot
            be mistaken for the live number beside it. */}
        <span className="text-chat-dim">
          {cumulative === null ? "—" : formatPct(cumulative)} since engine start
        </span>
      </div>
    </div>
  );
}

// ===========================================================================
// Preemptions + combined tiles
// ===========================================================================

export function CombinedFigure({
  label,
  metric,
  format,
  unit,
  testid,
}: {
  label: string;
  metric: CombinedMetric;
  format: (n: number | null) => string;
  unit?: string;
  testid?: string;
}) {
  const partial = isPartial(metric);
  return (
    <div data-testid={testid}>
      <div className="text-xs uppercase tracking-wider text-chat-dim">{label}</div>
      <div className="mt-1 font-mono text-2xl tabular-nums text-chat-fg">
        {/* `format(null)` is an em dash. A metric no selected engine reports
            must NEVER render as 0 — 0 says "measured, and idle". */}
        {format(metric.value)}
        {unit && metric.value !== null && (
          <span className="ml-1 text-xs text-chat-dim">{unit}</span>
        )}
      </div>
      {partial && (
        <div
          data-testid={testid ? `${testid}-partial` : undefined}
          className="mt-0.5 text-[10px] text-chat-warn"
          title="The remaining selected engines do not publish this metric, so they contribute nothing to the total rather than zero."
        >
          {metric.reporting} of {metric.total} models report this
        </div>
      )}
      {metric.value === null && metric.total > 0 && (
        <div className="mt-0.5 text-[10px] text-chat-dim">
          not reported by {metric.total === 1 ? "this engine" : "these engines"}
        </div>
      )}
    </div>
  );
}

// ===========================================================================
// "Since engine start" strip — every cumulative figure in one greyed place
// ===========================================================================

export function SinceEngineStart({
  frames,
}: {
  frames: readonly LiveEngineFrame[];
}) {
  // Finished reasons and token totals are lifetime counters. They used to sit
  // beside 2-second figures with nothing to tell them apart; here they are
  // greyed and labelled as one block.
  const reasons = new Map<string, number>();
  let anyReason = false;
  for (const f of frames) {
    for (const [reason, count] of Object.entries(f.finished ?? {})) {
      if (count === null || count === undefined || count <= 0) continue;
      anyReason = true;
      reasons.set(reason, (reasons.get(reason) ?? 0) + count);
    }
  }
  // Sums that skip an unreported total, with provenance — never `?? 0`,
  // which would leave a fleet total silently short by an unknown amount.
  const promptTotal = sumReported(
    frames.map((f) => f.throughput?.prompt_tokens_total),
  );
  const genTotal = sumReported(
    frames.map((f) => f.throughput?.generation_tokens_total),
  );
  const totalNote = (m: CombinedMetric) =>
    isPartial(m) ? ` (${m.reporting} of ${m.total} models)` : "";
  return (
    <Panel
      title="Since engine start"
      testid="since-engine-start"
      right={<ScopeNote>cumulative · not the window</ScopeNote>}
      className="opacity-80"
    >
      <div className="flex flex-wrap gap-x-8 gap-y-2 text-xs text-chat-dim">
        <span>
          prompt{" "}
          <span className="font-mono text-chat-muted">
            {promptTotal.value === null ? "—" : formatCompact(promptTotal.value)}
          </span>{" "}
          tokens{totalNote(promptTotal)}
        </span>
        <span>
          generated{" "}
          <span className="font-mono text-chat-muted">
            {genTotal.value === null ? "—" : formatCompact(genTotal.value)}
          </span>{" "}
          tokens{totalNote(genTotal)}
        </span>
        {anyReason ? (
          [...reasons.entries()].map(([reason, count]) => (
            <span key={reason}>
              {reason}{" "}
              <span className="font-mono text-chat-muted">{formatInt(count)}</span>
            </span>
          ))
        ) : (
          <span>no completions counted</span>
        )}
      </div>
    </Panel>
  );
}

// ===========================================================================
// Misc states
// ===========================================================================

export function Banner({
  tone,
  children,
}: {
  tone: "warn" | "error";
  children: React.ReactNode;
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn(
        "rounded-md border px-4 py-2.5 text-sm",
        tone === "error"
          ? "border-chat-negative/40 bg-chat-negative/10 text-chat-negative"
          : "border-chat-warn/40 bg-chat-warn/10 text-chat-warn",
      )}
    >
      {children}
    </div>
  );
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-dashed border-chat-rule bg-chat-surface/30 p-10 text-center">
      <p className="text-base font-medium text-chat-fg">{title}</p>
      <p className="mx-auto mt-1.5 max-w-md text-sm text-chat-dim">{body}</p>
    </div>
  );
}

// Re-export so the page has one import site for the marker math it also
// needs when labelling (`p50`, `p99` on the windowed distribution).
export { quantileFromBuckets };
