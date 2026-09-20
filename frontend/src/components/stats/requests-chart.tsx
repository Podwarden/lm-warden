"use client";

// The requests chart: one mark per completed request over the window.
//
//   x       time the request finished
//   y       duration, log scale — a 400 ms cache hit and a 4-minute
//           generation both need to be readable on one axis
//   colour  ONE categorical field, chosen from the data (client when there
//           is more than one and none dominates; else model; else nothing) —
//           see pickColourField
//   size    generated tokens (area ∝ count)
//   shape   finish reason: filled = stop, hollow = length, cross = other/none.
//           Never colour — colour is already carrying identity.
//   tooltip prompt tokens, TTFT, exact values
//
// Colour is painted through `text-chat-*` classes and `currentColor`, so the
// theme owns it. The legend doubles as a filter: clicking a group isolates it,
// which is the non-colour affordance that makes a six-hue palette usable.

import { useMemo } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "@/lib/utils";
import { formatCompact, formatLatency } from "@/lib/live-stats";
import {
  DURATION_FLOOR_S,
  classFor,
  finishClassOf,
  groupKeyOf,
  logTicks,
  markRadius,
  type ColourField,
  type ColourGroup,
  type FinishClass,
  type RequestHistoryRow,
} from "@/lib/request-history";
import type { StatsRange } from "@/lib/stats-v2";

export interface ChartPoint {
  id: string;
  /** Epoch ms. */
  at: number;
  /** Clamped to DURATION_FLOOR_S for the log axis; `row` keeps the truth. */
  y: number;
  r: number;
  finish: FinishClass;
  className: string;
  group: string;
  row: RequestHistoryRow;
}

export function toPoints(
  rows: readonly RequestHistoryRow[],
  field: ColourField,
  groups: readonly ColourGroup[],
): ChartPoint[] {
  const maxGen = rows.reduce((m, r) => Math.max(m, r.completion_tokens), 0);
  return rows.map((r) => {
    const group = groupKeyOf(r, field);
    return {
      id: r.id,
      at: r.finished_at * 1000,
      y: Math.max(DURATION_FLOOR_S, r.duration_s),
      r: markRadius(r.completion_tokens, maxGen),
      finish: finishClassOf(r.finish_reason),
      className: classFor(groups, group),
      group,
      row: r,
    };
  });
}

// ---- marks ---------------------------------------------------------------

interface MarkProps {
  cx?: number;
  cy?: number;
  payload?: ChartPoint;
}

/** Filled circle / hollow ring / cross, all in `currentColor`. */
export function Mark({ cx, cy, payload }: MarkProps) {
  if (cx === undefined || cy === undefined || !payload) return null;
  const r = payload.r;
  const common = {
    className: payload.className,
    "data-testid": "request-mark",
    "data-finish": payload.finish,
    "data-group": payload.group,
  };
  if (payload.finish === "stop") {
    return <circle {...common} cx={cx} cy={cy} r={r} fill="currentColor" opacity={0.75} />;
  }
  if (payload.finish === "length") {
    return (
      <circle
        {...common}
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        opacity={0.9}
      />
    );
  }
  const d = r;
  return (
    <path
      {...common}
      d={`M${cx - d},${cy - d}L${cx + d},${cy + d}M${cx - d},${cy + d}L${cx + d},${cy - d}`}
      stroke="currentColor"
      strokeWidth={1.5}
      fill="none"
      opacity={0.9}
    />
  );
}

// ---- tooltip -------------------------------------------------------------

interface TooltipShape {
  active?: boolean;
  payload?: readonly { payload?: ChartPoint }[];
}

function RequestTooltip({ active, payload }: TooltipShape) {
  const p = payload?.[0]?.payload;
  if (!active || !p) return null;
  const r = p.row;
  return (
    <div className="rounded border border-chat-rule bg-chat-surface px-3 py-2 font-mono text-[11px] text-chat-fg shadow">
      <div className="text-chat-muted">{new Date(p.at).toLocaleString()}</div>
      <div>
        {r.token_name ?? <span className="text-chat-dim">anonymous</span>}
        {r.client_ip ? <span className="text-chat-dim"> · {r.client_ip}</span> : null}
      </div>
      <div className="text-chat-muted">{r.model}</div>
      <div className="mt-1 grid grid-cols-2 gap-x-3">
        <span className="text-chat-dim">duration</span>
        <span>{formatLatency(r.duration_s)}</span>
        <span className="text-chat-dim">TTFT</span>
        <span>{r.ttft_s === null ? "—" : formatLatency(r.ttft_s)}</span>
        <span className="text-chat-dim">prompt</span>
        <span>{formatCompact(r.prompt_tokens)}</span>
        <span className="text-chat-dim">generated</span>
        <span>{formatCompact(r.completion_tokens)}</span>
        <span className="text-chat-dim">finish</span>
        <span>
          {r.finish_reason ?? "—"}
          {r.orphan ? " · orphan" : ""}
        </span>
      </div>
    </div>
  );
}

// ---- the chart -----------------------------------------------------------

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}
function fmtDate(ts: number): string {
  return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

export function RequestsChart({
  points,
  range,
  domain,
}: {
  points: readonly ChartPoint[];
  range: StatsRange;
  /** Epoch ms bounds of the window, from the response — so the x-axis is
   *  the window the button promised, not the span the data happens to fill. */
  domain: [number, number];
}) {
  const tickFmt = range === "7d" ? fmtDate : fmtTime;
  const [lo, hi] = useMemo(() => {
    let mn = Infinity;
    let mx = 0;
    for (const p of points) {
      if (p.y < mn) mn = p.y;
      if (p.y > mx) mx = p.y;
    }
    if (!Number.isFinite(mn)) return [DURATION_FLOOR_S, 10];
    return [mn, Math.max(mx, mn * 10)];
  }, [points]);
  const ticks = useMemo(() => logTicks(lo, hi), [lo, hi]);
  return (
    // text-chat-muted feeds `currentColor` for the axes and grid; each mark
    // sets its own text-chat-* class.
    <div className="h-72 w-full text-chat-muted" data-testid="requests-chart">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="currentColor" strokeOpacity={0.15} strokeDasharray="3 3" />
          <XAxis
            dataKey="at"
            type="number"
            domain={domain}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="currentColor"
            fontSize={11}
          />
          <YAxis
            dataKey="y"
            type="number"
            scale="log"
            domain={[ticks[0], ticks[ticks.length - 1]]}
            ticks={ticks}
            tickFormatter={(v) => formatLatency(Number(v)).replace(" ", "")}
            stroke="currentColor"
            fontSize={11}
            width={56}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3", stroke: "currentColor", strokeOpacity: 0.4 }}
            content={(p: unknown) => <RequestTooltip {...(p as TooltipShape)} />}
            isAnimationActive={false}
          />
          <Scatter
            data={points as ChartPoint[]}
            shape={(p: unknown) => <Mark {...(p as MarkProps)} />}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---- legend --------------------------------------------------------------

export function ColourLegend({
  field,
  groups,
  hidden,
  onToggle,
}: {
  field: ColourField;
  groups: readonly ColourGroup[];
  hidden: ReadonlySet<string>;
  onToggle: (key: string) => void;
}) {
  if (field === "none") {
    return (
      <span className="text-[11px] text-chat-dim" data-testid="colour-legend-none">
        one {groups.length ? "client and one model" : "series"} — colour carries nothing here
      </span>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]" data-testid="colour-legend">
      <span className="text-chat-dim">colour = {field}</span>
      {groups.map((g) => {
        const off = hidden.has(g.key);
        return (
          <button
            key={g.key}
            type="button"
            onClick={() => onToggle(g.key)}
            aria-pressed={!off}
            data-testid="colour-legend-group"
            data-group={g.key}
            title={off ? "click to show" : "click to hide"}
            className={cn(
              "inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 font-mono transition-opacity",
              g.className,
              off ? "opacity-30 line-through" : "hover:bg-chat-surface-2",
            )}
          >
            <span aria-hidden="true" className="h-2 w-2 rounded-full bg-current" />
            <span className="max-w-[16ch] truncate">{g.key}</span>
            <span className="text-chat-dim">{g.count}</span>
          </button>
        );
      })}
    </div>
  );
}
