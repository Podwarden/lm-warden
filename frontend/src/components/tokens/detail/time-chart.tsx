"use client";

// One chart body for the token details page, reproducing the approved
// mockup's `lineChart()` (docs/superpowers/specs/2026-09-18-token-details-mockup.html)
// on recharts. The mockup wins over v2-charts.tsx here (spec §4.6), so nothing
// is imported from it. Hover is computed from the pointer, like the mockup,
// rather than from recharts' Tooltip, so the crosshair and the tooltip land
// exactly where the mockup puts them.

import { useId, useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { Line, LineChart, ReferenceArea, ReferenceLine, XAxis, YAxis } from "recharts";
import type { SeriesBin } from "@/lib/token-series";
import { day, fmtWhen } from "@/lib/token-format";
import { useElementWidth } from "@/lib/use-element-width";
import { SERIES } from "./styles";

export const PLOT_H = 210;
export const MARGIN = { l: 52, r: 8, t: 8, b: 24 } as const;

export type NumKey = {
  [K in keyof SeriesBin]: SeriesBin[K] extends number | null ? K : never;
}[keyof SeriesBin];

/** A line reads a SeriesBin field by `key`, or -- for a series the bins do
 *  not carry, such as one model's rate -- computes it with `get`, in which
 *  case `key` is any id unique among the chart's lines. */
export type ChartLine = {
  color: string;
  dashed?: boolean;
  /** An explicit stroke-dasharray; wins over `dashed`. */
  dash?: string;
} & (
  | { key: NumKey; get?: undefined }
  | { key: string; get: (b: SeriesBin) => number | null }
);

export function lineValue(L: ChartLine, b: SeriesBin): number | null {
  return L.get ? L.get(b) : b[L.key as NumKey];
}

export interface TipRow {
  name: string;
  color: string;
  value: string;
}

export function niceMax(v: number): number {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const s of [1, 2, 2.5, 5, 10]) if (s * e >= v) return s * e;
  return 10 * e;
}

export const yTicks = (max: number): number[] => [0, 1, 2, 3, 4].map((i) => (max / 4) * i);

export const xTicks = (fromMin: number, toMin: number): number[] =>
  [0, 1, 2, 3, 4, 5].map((i) => fromMin + ((toMin - fromMin) / 5) * i);

export function nearestBin(bins: SeriesBin[], binMinutes: number, tMin: number): SeriesBin | null {
  if (bins.length === 0) return null;
  const half = binMinutes / 2;
  return bins.reduce((best, c) =>
    Math.abs(c.minute + half - tMin) < Math.abs(best.minute + half - tMin) ? c : best, bins[0]);
}

/** Spec §4.4: the part of the period before `latency_since` is hatched. A null
 *  `latency_since` means nothing was ever recorded: hatch it all, dated now. */
export function hatchSpec(
  fromMin: number, toMin: number, latencySinceSec: number | null, nowSec: number,
): { toMin: number; label: string } | null {
  const since = latencySinceSec ?? nowSec;
  const sinceMin = since / 60;
  if (sinceMin <= fromMin) return null;
  return { toMin: Math.min(sinceMin, toMin), label: `Not recorded before ${day(since)}` };
}

export interface TimeChartProps {
  name: string;
  bins: SeriesBin[];
  binMinutes: number;
  fromMin: number;
  toMin: number;
  lines: ChartLine[];
  yFmt: (v: number) => string;
  tipRows: (b: SeriesBin) => TipRow[];
  hatch?: { toMin: number; label: string } | null;
}

type TickProps = { x?: number | string; y?: number | string; index?: number; payload?: { value: number } };

export function TimeChart({ name, bins, binMinutes, fromMin, toMin, lines, yFmt, tipRows, hatch }: TimeChartProps) {
  const plotRef = useRef<HTMLDivElement>(null);
  const W = useElementWidth(plotRef);
  const iw = Math.max(1, W - MARGIN.l - MARGIN.r);
  const span = Math.max(1, toMin - fromMin);
  const half = binMinutes / 2;
  const hatchId = `vw-hatch-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;

  // Computed lines are materialised into the rows under their own key, so
  // recharts reads every line the same way.
  const data = useMemo(() => bins.map((b) => {
    const row: Record<string, unknown> = { ...b, t: b.minute + half };
    for (const L of lines) if (L.get) row[L.key] = L.get(b);
    return row;
  }), [bins, half, lines]);
  const { ymax, hasValue } = useMemo(() => {
    let m = 0;
    let any = false;
    for (const b of bins) for (const L of lines) {
      const v = lineValue(L, b);
      if (v != null) any = true;
      if (v != null && v > m) m = v;
    }
    return { ymax: niceMax(m * 1.05), hasValue: any };
  }, [bins, lines]);
  const yt = yTicks(ymax);

  const [active, setActive] = useState<SeriesBin | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const [tipW, setTipW] = useState(0);
  useLayoutEffect(() => {
    if (tipRef.current) setTipW(tipRef.current.offsetWidth);
  }, [active]);

  function onMove(e: PointerEvent<HTMLDivElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    if (r.width <= 0) return;
    const t = fromMin + ((e.clientX - r.left) / r.width) * span;
    setActive(nearestBin(bins, binMinutes, t));
  }

  const cx = active ? MARGIN.l + ((active.minute + half - fromMin) / span) * iw : 0;
  const tipLeft = Math.min(Math.max(cx + 12, 0), Math.max(0, W - tipW));

  return (
    <div ref={plotRef} className="relative h-[210px]" data-testid={`chart-${name}`}>
      {W > 0 && (
        <LineChart
          width={W}
          height={PLOT_H}
          data={data}
          margin={{ top: MARGIN.t, right: MARGIN.r, bottom: 0, left: 0 }}
          accessibilityLayer={false}
        >
          <defs>
            <pattern id={hatchId} width={7} height={7} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={7} stroke={SERIES.grid} strokeWidth={2} />
            </pattern>
          </defs>
          <XAxis
            dataKey="t"
            type="number"
            domain={[fromMin, toMin]}
            allowDataOverflow
            ticks={xTicks(fromMin, toMin)}
            interval={0}
            height={MARGIN.b}
            axisLine={false}
            tickLine={false}
            tick={({ x, index, payload }: TickProps) => (
              <text
                x={Number(x)}
                y={PLOT_H - 6}
                textAnchor={index === 0 ? "start" : index === 5 ? "end" : "middle"}
                fill={SERIES.axis}
                fontSize={11}
              >
                {fmtWhen((payload?.value ?? fromMin) * 60, span)}
              </text>
            )}
          />
          <YAxis
            type="number"
            domain={[0, ymax]}
            // With no value at all (a period wholly before latency_since)
            // recharts finds no data extent and drops the axis, the grid and
            // the hatch. There is no line to clip then, so let the explicit
            // domain stand; with data it stays off (see the baseline test).
            allowDataOverflow={!hasValue}
            ticks={yt}
            interval={0}
            width={MARGIN.l}
            axisLine={false}
            tickLine={false}
            tick={({ y, payload }: TickProps) => (
              <text x={MARGIN.l - 8} y={Number(y) + 4} textAnchor="end" fill={SERIES.axis} fontSize={11}>
                {yFmt(payload?.value ?? 0)}
              </text>
            )}
          />
          {hatch && (
            <ReferenceArea
              x1={fromMin}
              x2={hatch.toMin}
              y1={0}
              y2={ymax}
              fill={`url(#${hatchId})`}
              fillOpacity={0.8}
              stroke="none"
              ifOverflow="hidden"
              zIndex={-200}
              label={(lp: { viewBox?: { x: number; y: number; width: number; height: number } }) => {
                const vb = lp.viewBox;
                if (!vb || vb.width <= 120) return <g />;
                return (
                  <text x={vb.x + vb.width / 2} y={vb.y + vb.height / 2} textAnchor="middle" fill={SERIES.muted} fontSize={12}>
                    {hatch.label}
                  </text>
                );
              }}
            />
          )}
          {yt.map((v, i) => (
            <ReferenceLine
              key={`grid-${i}`}
              y={v}
              stroke={SERIES.grid}
              strokeWidth={1}
              strokeDasharray={i === 0 ? undefined : "3 4"}
              ifOverflow="visible"
              zIndex={-100}
            />
          ))}
          {lines.map((L) => (
            <Line
              key={L.key}
              dataKey={L.key}
              type="linear"
              stroke={L.color}
              strokeWidth={L.dashed || L.dash ? 1.5 : 1.8}
              strokeDasharray={L.dash ?? (L.dashed ? "5 4" : undefined)}
              strokeLinejoin="round"
              strokeLinecap="round"
              dot={false}
              activeDot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
          {active && (
            <ReferenceLine
              x={active.minute + half}
              stroke={SERIES.muted}
              strokeOpacity={0.35}
              strokeWidth={1}
              ifOverflow="hidden"
              zIndex={1100}
            />
          )}
        </LineChart>
      )}
      <div
        data-testid="chart-hit"
        className="absolute"
        style={{ left: MARGIN.l, top: MARGIN.t, right: MARGIN.r, bottom: MARGIN.b }}
        onPointerMove={onMove}
        onPointerLeave={() => setActive(null)}
      />
      {active && (
        <div
          ref={tipRef}
          role="tooltip"
          className="pointer-events-none absolute z-[5] whitespace-nowrap rounded-md border border-chat-rule bg-chat-page px-2.5 py-[7px] text-[12px] tabular-nums shadow-[0_6px_18px_rgba(0,0,0,.35)]"
          style={{ left: tipLeft, top: 6 }}
        >
          <div className="mb-[3px] text-chat-muted">
            {fmtWhen(active.minute * 60, 4000)}
            {binMinutes > 1 ? ` – ${fmtWhen((active.minute + binMinutes) * 60, 4000)}` : ""}
          </div>
          {tipRows(active).map((r) => (
            <div key={r.name} className="flex justify-between gap-4">
              <span className="text-chat-muted">
                <span aria-hidden className="mr-1.5 inline-block h-2 w-2 rounded-[2px]" style={{ background: r.color }} />
                {r.name}
              </span>
              <span>{r.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
