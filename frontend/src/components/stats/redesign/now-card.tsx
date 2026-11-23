"use client";

// NOW (design 3.2): one fixed card for the scope's model. Generation now and
// prefill now (computed tokens only) are the server's ~10 s rates (B5); the
// 5-minute sparkline starts full from the server ring sent on connect.
// Idle shows "idle · last request N min ago", never "0 tok/s".

import { FIXED, type ScopeState, fmtAgo, fmtRate, fmtSeconds, fmtTokens } from "@/lib/stats-redesign";

import { FixedCard, Label, Line, StateChip, Swatch } from "./primitives";
import { LiveNumber } from "@/components/ui/live-number";

export interface SparkPoint {
  gen: number | null;
  prefill: number | null;
}

export interface NowCardProps {
  modelId: string | null;
  modelName: string | null;
  state: ScopeState;
  /** Requests of this model still prefilling (the engine counts a prompt only
   *  once its prefill completes). */
  prefilling?: number;
  running: number | null;
  engineWaiting: number | null;
  wardenQueue: number | null;
  genNow: number | null;
  prefillNow: number | null;
  kvPct: number | null;
  maxModelLen: number | null;
  windowP50Ttft: number | null;
  windowP50Decode: number | null;
  windowN: number;
  typicalPrefill: number | null;
  lastFinishedAt: number | null;
  spark: readonly SparkPoint[];
}

/** 5 minutes at the 2 s cadence: the x scale is fixed, not the ring's length. */
const SPARK_SLOTS = 150;

function Spark({ points }: { points: readonly SparkPoint[] }) {
  const H = 70;
  const W = 300;
  if (points.length < 2) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: H }} className="w-full">
        <line x1="0" x2={W} y1={H - 2} y2={H - 2} stroke="rgb(var(--vw-rule-soft, 74 56 32))" />
        <text x={W / 2} y={H / 2} textAnchor="middle" className="fill-chat-dim text-[11px]">
          collecting
        </text>
      </svg>
    );
  }
  // null (not watched, no data) and 0 (idle) break the line: an idle or
  // unwatched stretch is a gap, never a line drawn at zero.
  const vals = (k: "gen" | "prefill") => points.map((p) => (p[k] != null && (p[k] as number) > 0 ? (p[k] as number) : null));
  const gens = vals("gen");
  const pres = vals("prefill");
  const gmax = Math.max(80, ...gens.map((v) => v ?? 0)) * 1.1;
  const pmax = Math.max(2000, ...pres.map((v) => v ?? 0)) * 1.1;
  const bw = W / Math.max(1, SPARK_SLOTS - 1);
  const x0 = (SPARK_SLOTS - points.length) * bw; // newest at the right edge
  const path = (v: (number | null)[], max: number) => {
    let d = "";
    let pen = false;
    v.forEach((y, i) => {
      if (y == null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${(x0 + i * bw).toFixed(1)},${(H - 2 - ((H - 4) * y) / max).toFixed(1)} `;
      pen = true;
    });
    return d.trim();
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: H }} className="w-full" aria-label="last 5 minutes">
      <line x1="0" x2={W} y1={H - 2} y2={H - 2} stroke="currentColor" className="text-chat-rule" />
      <path d={path(pres, pmax) || "M0,0"} fill="none" stroke="rgb(var(--vw-prompt))" strokeWidth="1.5" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
      <path d={path(gens, gmax) || "M0,0"} fill="none" stroke="rgb(var(--vw-completion))" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Big({ value, idle }: { value: number | null; idle: string }) {
  return (
    <div className="h-9 overflow-hidden">
      {value == null || value < 1 ? (
        <div className="overflow-hidden text-ellipsis whitespace-nowrap pt-2.5 text-[13px] text-chat-dim">{idle}</div>
      ) : (
        <div className="whitespace-nowrap text-[28px] font-semibold leading-[1.1] tabular-nums">
          <LiveNumber reserve={5}>{fmtRate(value)}</LiveNumber>
          <small className="ml-1 text-xs font-medium text-chat-muted">tok/s</small>
        </div>
      )}
    </div>
  );
}

export function NowCard(p: NowCardProps) {
  const asleep = p.state === "Asleep";
  const noData = p.state === "No data";
  const quiet = p.state === "Idle" || asleep || noData;
  const idleText = noData
    ? "no data"
    : asleep
      ? "asleep"
      : p.lastFinishedAt == null
        ? "idle · no requests yet"
        : `idle · last request ${fmtAgo(p.lastFinishedAt)}`;
  // An Idle chip never sits next to a rate; a busy engine whose rate is not
  // known yet (just connected, or after an unwatched gap) says so.
  const gen = quiet ? null : p.genNow;
  const pre = quiet ? null : p.prefillNow;
  const genIdle = quiet ? idleText : p.genNow == null ? "measuring" : idleText;
  const kv = p.kvPct != null ? Math.max(0, Math.min(100, p.kvPct * 100)) : null;
  return (
    <FixedCard
      height={FIXED.nowCard}
      phoneHeight={FIXED.nowCardPhone}
      testid="now-card"
      className="grid grid-cols-2 grid-rows-[auto_auto_1fr] gap-x-[18px] gap-y-1.5 md:grid-cols-[240px_1fr_1fr] md:grid-rows-[auto_1fr]"
    >
      <div className="col-span-2 flex min-w-0 flex-col gap-1 md:col-span-1 md:row-span-2">
        {/* No model or install name here: the scope bar says what this is. */}
        <Line>
          <StateChip state={p.state} />
        </Line>
        <Line
          className="text-xs tabular-nums text-chat-muted"
          title="running in the engine · waiting in the engine's own queue · waiting in the warden's admission queue"
        >
          running <LiveNumber reserve={2} className="font-bold text-chat-fg">{String(p.running ?? "-")}</LiveNumber> · waiting{" "}
          <LiveNumber reserve={2} className="font-bold text-chat-fg">{String(p.engineWaiting ?? "-")}</LiveNumber> · queue{" "}
          <LiveNumber reserve={2} className="font-bold text-chat-fg">{String(p.wardenQueue ?? "-")}</LiveNumber>
        </Line>
        <Line className="text-xs tabular-nums text-chat-dim">
          {p.windowN >= 5
            ? `window p50 TTFT ${fmtSeconds(p.windowP50Ttft)} · decode ${fmtRate(p.windowP50Decode)} tok/s`
            : `window n=${p.windowN} · too few for p50`}
        </Line>
        <Line className="text-xs tabular-nums text-chat-muted">
          KV in use <LiveNumber reserve={3}>{kv != null ? String(Math.round(kv)) : "-"}</LiveNumber>
          {kv != null ? " %" : ""} · window {fmtTokens(p.maxModelLen)}
        </Line>
        <div className="relative h-1.5 rounded bg-chat-surface-2" title="red mark = 90 %">
          <span className="absolute inset-y-0 left-0 rounded bg-amber-500" style={{ width: `${kv ?? 0}%` }} />
          <i className="absolute -inset-y-[3px] left-[90%] w-0.5 bg-red-400" />
        </div>
      </div>
      <div data-testid="now-generation" className="flex min-w-0 flex-col gap-1">
        <Label>Generation now</Label>
        <Big value={gen} idle={genIdle} />
      </div>
      <div data-testid="now-prefill" className="flex min-w-0 flex-col gap-1">
        <Label>Prefill now (computed)</Label>
        <Big
          value={pre}
          idle={noData ? "no data" : !quiet && (p.prefilling ?? 0) > 0 ? `${p.prefilling} prefilling · counted when it finishes` : "no prefill finished in 10 s"}
        />
      </div>
      <div className="col-span-2 min-w-0 md:col-start-2">
        <Spark points={p.spark} />
        <div className="mt-1 flex h-[18px] gap-3 overflow-hidden whitespace-nowrap text-[11px] text-chat-muted">
          <span>last 5 min</span>
          <span className="inline-flex items-center gap-1">
            <Swatch color="rgb(var(--vw-completion))" />
            generation
          </span>
          <span className="inline-flex items-center gap-1">
            <Swatch color="rgb(var(--vw-prompt))" />
            prefill (dashed)
          </span>
          <span className="text-chat-dim">
            typical prefill {p.typicalPrefill != null ? `${fmtRate(p.typicalPrefill)} tok/s` : "not learned yet"}
          </span>
        </div>
      </div>
    </FixedCard>
  );
}
