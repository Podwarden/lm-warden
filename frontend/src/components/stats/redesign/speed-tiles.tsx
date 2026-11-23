"use client";

// LATENCY AND SPEED (design 3.3): four same-size tiles (p50 large, p95 or max
// small, n, a dot strip of the last 50 requests) and one per-request latency
// chart (log y, metric switch, x switch, dots coloured by key, rolling median of
// the last 20). The low-volume rules come from the server (app/stats/speed.py).
// The key holder's view swaps Prefill speed for Prompt speed (cache effects
// included): computed tokens need cached_tokens, which can reveal another key.

import { useState } from "react";

import {
  FIXED,
  type SpeedPoint,
  type SpeedResponse,
  type SpeedSummary,
  fmtRate,
  fmtSeconds,
  fmtTokens,
  rollingMedian,
  tileText,
} from "@/lib/stats-redesign";

import { FixedCard, Seg, Swatch, hueOf, useWidth } from "./primitives";

type MetricId = "ttft" | "decode" | "prefill" | "prompt" | "duration";

interface Def {
  id: MetricId;
  title: string;
  unit: "s" | "tok/s";
  color: string;
  get: (p: SpeedPoint) => number | null | undefined;
  sub: string;
  def: string;
  /** When the window has requests but none counts for this figure. */
  none?: { list: string; sub: (n: number) => string };
}

function defs(view: "admin" | "key", wardenQueue: SpeedSummary | undefined, typicalPrefill: number | null): Def[] {
  const q = wardenQueue && wardenQueue.mode !== "none" && wardenQueue.mode !== "list" ? wardenQueue.p50 : null;
  const out: Def[] = [
    {
      id: "ttft",
      title: "Time to first token",
      unit: "s",
      color: "rgb(var(--vw-ttft))",
      get: (p) => p.ttft,
      sub: q != null ? `warden queue p50 ${fmtSeconds(q)} · included` : "warden queue included",
      def: "Warden queue + slot held to first token. Streaming requests only.",
      none: { list: "No streamed requests", sub: (n) => `${n} in window, none streamed a first token` },
    },
    {
      id: "decode",
      title: "Decode speed",
      unit: "tok/s",
      color: "rgb(var(--vw-completion))",
      get: (p) => p.decode,
      sub: "per request, after the first token",
      def: "(out tokens − 1) / (duration − TTFT). One request's share of the batch.",
      none: { list: "None long enough", sub: (n) => `${n} in window, none streamed 16+ tokens` },
    },
    view === "key"
      ? {
          id: "prompt",
          title: "Prompt speed",
          unit: "tok/s",
          color: "rgb(var(--vw-completion))",
          get: (p) => p.prompt_speed,
          sub: "from your own requests only",
          def: "Prompt tokens / TTFT. Includes cache effects.",
          none: { list: "No streamed requests", sub: (n) => `${n} in window, none streamed a first token` },
        }
      : {
          id: "prefill",
          title: "Prefill speed",
          unit: "tok/s",
          color: "rgb(var(--vw-completion))",
          get: (p) => (p.prefill_alone ? p.prefill : null),
          sub: typicalPrefill != null ? `learned typical ${fmtRate(typicalPrefill)} tok/s` : "learned typical: not yet",
          def: "(prompt − cached) / TTFT. Computed tokens only; ≥ 512 fresh, prefilled alone.",
          none: {
            list: "None prefilled alone",
            sub: (n) => `${n} in window: shared prefill or < 512 new tokens`,
          },
        },
    {
      id: "duration",
      title: "Duration",
      unit: "s",
      color: "rgb(var(--vw-dur))",
      get: (p) => p.duration,
      sub: "slot held to last byte",
      def: "Plotted per request in the Requests chart at the bottom. Click to jump.",
    },
  ];
  return out;
}

function DotStrip({ values, color }: { values: number[]; color: string }) {
  const last = values.slice(-50);
  if (!last.length) {
    return (
      <svg className="h-[22px] w-full" viewBox="0 0 200 26" preserveAspectRatio="none">
        <line x1="0" y1="13" x2="200" y2="13" stroke="currentColor" className="text-chat-rule" strokeDasharray="2 3" />
      </svg>
    );
  }
  const lo = Math.log(Math.max(1e-6, Math.min(...last)));
  const hi = Math.log(Math.max(1e-6, Math.max(...last)));
  const w = 200 / 50;
  return (
    <svg className="h-[22px] w-full" viewBox="0 0 200 26" preserveAspectRatio="none" aria-hidden="true">
      {last.map((v, i) => {
        const f = hi > lo ? (Math.log(Math.max(1e-6, v)) - lo) / (hi - lo) : 0.5;
        return <rect key={i} x={i * w + 0.6} y={22 - f * 18} width={w - 1.2} height={3} rx={1} fill={color} />;
      })}
    </svg>
  );
}

function Tile({
  d,
  summary,
  points,
  lastFinishedAt,
  windowCount,
  onJump,
}: {
  d: Def;
  summary: SpeedSummary | undefined;
  points: SpeedPoint[];
  lastFinishedAt: number | null;
  windowCount: number;
  onJump?: () => void;
}) {
  const fmt = d.unit === "s" ? fmtSeconds : fmtRate;
  const t = tileText(summary, fmt, lastFinishedAt, windowCount, d.none);
  const vals = points.map(d.get).filter((v): v is number => v != null && Number.isFinite(v) && v > 0);
  const big = t.big ?? "";
  const m = big.match(/^(.*?) ?(ms|s)$/);
  return (
    <FixedCard
      height={FIXED.tile}
      testid={`tile-${d.id}`}
      title={d.def}
      onClick={d.id === "duration" ? onJump : undefined}
      className="flex flex-col gap-1"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-chat-muted">{d.title}</h3>
        <span data-testid="tile-n" className="rounded border border-chat-rule px-1.5 text-[11px] tabular-nums text-chat-dim">
          n={t.n}
        </span>
      </div>
      <div className="h-[34px] overflow-hidden">
        {t.big !== null ? (
          <div className="text-[28px] font-semibold leading-[1.1] tabular-nums">
            {m ? m[1] : big}
            <small className="ml-0.5 text-[13px] font-medium text-chat-muted">{m ? m[2] : d.unit}</small>
          </div>
        ) : (
          <div
            className={`overflow-hidden text-ellipsis whitespace-nowrap pt-2 text-[15px] tabular-nums ${
              t.n === 0 ? "text-chat-dim" : "text-chat-fg"
            }`}
          >
            {t.list}
          </div>
        )}
      </div>
      <div className="h-[18px] overflow-hidden text-ellipsis whitespace-nowrap text-xs tabular-nums text-chat-muted">{t.sub}</div>
      <div className="h-[18px] overflow-hidden text-ellipsis whitespace-nowrap text-xs text-chat-muted">{t.n === 0 ? " " : d.sub}</div>
      <DotStrip values={vals} color={d.color} />
      <div className="mt-auto h-[28px] overflow-hidden text-[11px] leading-[14px] text-chat-dim">{d.def}</div>
    </FixedCard>
  );
}

export function SpeedTiles({
  data,
  view,
  typicalPrefill,
  onJumpToRequests,
}: {
  data: SpeedResponse | undefined;
  view: "admin" | "key";
  typicalPrefill: number | null;
  onJumpToRequests?: () => void;
}) {
  const ds = defs(view, data?.summary?.warden_queue, typicalPrefill);
  const points = data?.points ?? [];
  return (
    <div data-testid="speed-tiles" className="grid gap-2.5 [grid-template-columns:repeat(auto-fit,minmax(230px,1fr))]">
      {ds.map((d) => (
        <Tile
          key={d.id}
          d={d}
          summary={data?.summary?.[d.id]}
          points={points}
          lastFinishedAt={data?.last_finished_at ?? null}
          windowCount={data?.count ?? 0}
          onJump={onJumpToRequests}
        />
      ))}
    </div>
  );
}

// ---- the per-request latency chart -------------------------------------------

const PAD = { l: 52, r: 12, t: 10, b: 26 };

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(10 ** e);
  return out;
}

export function LatencyScatter({
  data,
  view,
  sinceEpoch,
  nowEpoch,
}: {
  data: SpeedResponse | undefined;
  view: "admin" | "key";
  sinceEpoch: number;
  nowEpoch: number;
}) {
  const [metric, setMetric] = useState<MetricId>("ttft");
  const [xMode, setXMode] = useState<"time" | "prompt">("time");
  // Drawn in real pixels: a fixed plot height in a fixed card, whatever the
  // card's width, so the x axis and the low points are never clipped.
  const [boxRef, W] = useWidth<HTMLDivElement>();
  const H = FIXED.latencyPlot;
  const metricOpts =
    view === "key"
      ? ([
          { v: "ttft", label: "TTFT" },
          { v: "decode", label: "Decode tok/s" },
          { v: "prompt", label: "Prompt tok/s" },
        ] as const)
      : ([
          { v: "ttft", label: "TTFT" },
          { v: "decode", label: "Decode tok/s" },
          { v: "prefill", label: "Prefill tok/s" },
        ] as const);
  const get = (p: SpeedPoint): number | null | undefined =>
    metric === "ttft" ? p.ttft : metric === "decode" ? p.decode : metric === "prefill" ? p.prefill : metric === "prompt" ? p.prompt_speed : p.duration;
  const fmt = metric === "ttft" ? fmtSeconds : fmtRate;
  const pts = (data?.points ?? [])
    .map((p) => ({ p, v: get(p) }))
    .filter((x): x is { p: SpeedPoint; v: number } => x.v != null && Number.isFinite(x.v) && x.v > 0);
  const keys = [...new Map(pts.map((x) => [x.p.token_id ?? "", x.p.token_name ?? "anonymous"])).entries()].slice(0, 6);

  let body;
  if (!pts.length) {
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block">
        <rect x={PAD.l} y={10} width={W - 64} height={H - 36} fill="none" stroke="currentColor" className="text-chat-rule" strokeDasharray="3 4" />
        <text x={W / 2} y={H / 2} textAnchor="middle" className="fill-chat-dim text-[13px]">
          No requests in window
        </text>
      </svg>
    );
  } else {
    const vs = pts.map((x) => x.v);
    const lo = 10 ** Math.floor(Math.log10(Math.max(1e-3, Math.min(...vs))));
    const hi = 10 ** Math.ceil(Math.log10(Math.max(...vs) * 1.01));
    const y = (v: number) => PAD.t + (H - PAD.t - PAD.b) * (1 - (Math.log10(v) - Math.log10(lo)) / Math.max(1e-9, Math.log10(hi) - Math.log10(lo)));
    let x: (p: SpeedPoint) => number;
    let xt: [number, string][];
    if (xMode === "prompt") {
      const xs = pts.map((q) => Math.max(1, q.p.prompt_tokens));
      const xl = 10 ** Math.floor(Math.log10(Math.min(...xs)));
      const xh = 10 ** Math.ceil(Math.log10(Math.max(...xs)) + 1e-9);
      const span = Math.max(1e-9, Math.log10(xh) - Math.log10(xl));
      const xv = (n: number) => PAD.l + ((W - PAD.l - PAD.r) * (Math.log10(Math.max(1, n)) - Math.log10(xl))) / span;
      x = (p) => xv(p.prompt_tokens);
      xt = logTicks(xl, xh).map((t) => [xv(t), `${fmtTokens(t)} tok`]);
    } else {
      const span = Math.max(1, nowEpoch - sinceEpoch);
      const xv = (t: number) => PAD.l + ((W - PAD.l - PAD.r) * (t - sinceEpoch)) / span;
      x = (p) => xv(p.finished_at);
      xt = [0, 0.25, 0.5, 0.75, 1].map((f) => [xv(sinceEpoch + f * span), f === 1 ? "now" : `-${Math.round(((1 - f) * span) / 60)}m`]);
    }
    const med = xMode === "time" ? rollingMedian(vs) : [];
    const medPath = med
      .map((m, i) => (m == null ? null : `${x(pts[i].p).toFixed(1)},${y(m).toFixed(1)}`))
      .filter(Boolean)
      .join(" L");
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block" role="img" aria-label="per-request latency">
        {logTicks(lo, hi).map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="currentColor" className="text-chat-rule/60" />
            <text x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="fill-chat-dim text-[10px]">
              {fmt(t)}
            </text>
          </g>
        ))}
        {xt.map(([px, l]) => (
          <text key={l + px} x={px} y={H - 8} textAnchor="middle" className="fill-chat-dim text-[10px]">
            {l}
          </text>
        ))}
        {medPath && <path d={`M${medPath}`} fill="none" stroke="currentColor" className="text-chat-fg" strokeOpacity={0.7} strokeWidth={2} />}
        {pts.map(({ p, v }) => {
          const c = view === "key" ? hueOf(p.model_id) : hueOf(p.token_id ?? "");
          const hollow = metric === "prefill" && !p.prefill_alone;
          return (
            <circle key={p.id} cx={x(p)} cy={y(v)} r={5} fill={hollow ? "none" : c} stroke={hollow ? c : "rgb(var(--vw-surface, 42 31 14))"} strokeWidth={hollow ? 1.5 : 2}>
              <title>
                {view === "admin" ? `${p.token_name ?? "anonymous"} · ` : ""}
                {fmt(v)} · prompt {fmtTokens(p.prompt_tokens)}
              </title>
            </circle>
          );
        })}
      </svg>
    );
  }
  return (
    <FixedCard height={FIXED.latencyChart} phoneHeight={FIXED.latencyChart + 30} testid="latency-chart" className="mt-2.5">
      <div className="mb-1 flex flex-wrap items-center gap-x-3.5 gap-y-2">
        <span className="text-[10px] uppercase tracking-[0.08em] text-chat-dim">Show</span>
        <Seg label="Metric" value={metric} options={metricOpts} onChange={setMetric} />
        {view === "admin" && (
          <>
            <span className="text-[10px] uppercase tracking-[0.08em] text-chat-dim">X axis</span>
            <Seg
              label="X axis"
              value={xMode}
              options={[
                { v: "time", label: "Time" },
                { v: "prompt", label: "Prompt size" },
              ]}
              onChange={setXMode}
            />
          </>
        )}
      </div>
      <div ref={boxRef} data-testid="latency-plot" style={{ height: H }} className="w-full overflow-hidden">
        {body}
      </div>
      <div className="mt-1.5 flex h-[18px] gap-3 overflow-hidden whitespace-nowrap text-[11px] text-chat-muted">
        {view === "admin" &&
          keys.map(([id, name]) => (
            <span key={id} className="inline-flex items-center gap-1">
              <Swatch color={hueOf(id)} />
              {name}
            </span>
          ))}
        <span>
          log scale · line = rolling median (last 20){view === "admin" ? " · hollow = shared prefill, not in p50" : " · your requests only"}
        </span>
      </div>
    </FixedCard>
  );
}
