"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { aggregateThroughput, type ModelSample, type StatsRange } from "@/lib/stats";

interface Props {
  samples: ModelSample[];
  /** [startMs, endMs] for the X-axis. Computed in the page from the
   *  range selector so the axis always covers the requested window
   *  even when the data is sparse. */
  bounds: [number, number];
  /** Range key — only used to pick the tick formatter (HH:MM for short
   *  windows, "MMM dd" for the 7d window where minute-granular ticks
   *  would overlap). */
  range: StatsRange;
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtDate(ts: number): string {
  // Compact "MMM dd" for the 7d view — minute-granular ticks across a
  // week would just be a black smear. Locale-dependent on purpose so
  // the operator sees their own date format.
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
  });
}

function pickTickFormatter(range: StatsRange): (ts: number) => string {
  return range === "7d" ? fmtDate : fmtTime;
}

function fmtTokens(n: number): string {
  return n.toLocaleString();
}

/**
 * Line chart of total tokens/minute (prompt + completion, summed across
 * all models) over the selected range. A secondary line shows request
 * counts on the same axis — request counts are typically a small fraction
 * of token counts so they sit near the X axis without needing a second
 * scale (operators care about the shape more than the absolute number).
 */
export function ThroughputChart({ samples, bounds, range }: Props) {
  const points = aggregateThroughput(samples);
  const tickFmt = pickTickFormatter(range);

  if (points.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
        No throughput samples in this window.
      </div>
    );
  }

  return (
    <div className="h-72 w-full rounded-lg border border-slate-700 bg-slate-900/30 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            // Pin the domain to the range selector so the axis always
            // spans the requested window — recharts' default
            // ["dataMin","dataMax"] collapses to a point when only one
            // minute is present, and silently truncates the right edge
            // when the latest minute hasn't yet been rolled up.
            domain={bounds}
            // Without allowDataOverflow recharts re-expands the domain
            // to fit any point that lands outside our bounds (which can
            // happen during a refresh race). Clamp instead.
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={fmtTokens}
            width={64}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              border: "1px solid #334155",
              fontSize: "12px",
            }}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? fmtTokens(value) : String(value),
              String(name),
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Line
            type="monotone"
            dataKey="tokens"
            name="Tokens"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="requests"
            name="Requests"
            stroke="#60a5fa"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
