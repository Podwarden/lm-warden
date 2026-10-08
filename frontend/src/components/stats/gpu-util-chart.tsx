"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { aggregateGpuUtil, type GpuSample, type StatsRange } from "@/lib/stats";

interface Props {
  samples: GpuSample[];
  /** [startMs, endMs] for the X-axis. Computed in the page from the
   *  range selector so the axis covers the requested window even when
   *  the data is sparse. */
  bounds: [number, number];
  /** Range key — picks the tick formatter (HH:MM vs "MMM dd"). */
  range: StatsRange;
}

// Distinct, accessible-on-dark palette. Cycled if there are more GPUs than
// colours — eight is plenty for any realistic single-host deployment.
const GPU_COLORS = [
  "#34d399", // emerald
  "#60a5fa", // blue
  "#fbbf24", // amber
  "#f472b6", // pink
  "#a78bfa", // violet
  "#fb923c", // orange
  "#22d3ee", // cyan
  "#f87171", // red
];

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtDate(ts: number): string {
  // 7d view — minute-granular HH:MM ticks across a week overlap into a
  // smear. "MMM dd" is the operator-readable compromise.
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
  });
}

function pickTickFormatter(range: StatsRange): (ts: number) => string {
  return range === "7d" ? fmtDate : fmtTime;
}

/**
 * Stacked-or-overlaid area chart of GPU utilisation per minute. One area
 * per `gpu_index`. We render overlaid (not stacked) because utilisation is
 * already a percentage — stacking would imply summation, which is wrong.
 */
export function GpuUtilChart({ samples, bounds, range }: Props) {
  const { points, gpuIndexes } = aggregateGpuUtil(samples);
  const tickFmt = pickTickFormatter(range);

  if (points.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
        No GPU samples in this window.
      </div>
    );
  }

  return (
    <div className="h-72 w-full rounded-lg border border-slate-700 bg-slate-900/30 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <defs>
            {gpuIndexes.map((idx, i) => {
              const color = GPU_COLORS[i % GPU_COLORS.length];
              return (
                <linearGradient
                  key={idx}
                  id={`gpu-fill-${idx}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop offset="5%" stopColor={color} stopOpacity={0.6} />
                  <stop offset="95%" stopColor={color} stopOpacity={0.05} />
                </linearGradient>
              );
            })}
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            // Pin to the range selector so sparse data doesn't shrink
            // the axis; see throughput-chart.tsx for the longer
            // rationale on why we don't use ["dataMin","dataMax"].
            domain={bounds}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            stroke="#94a3b8"
            fontSize={11}
            width={48}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              border: "1px solid #334155",
              fontSize: "12px",
            }}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? `${value.toFixed(0)}%` : String(value),
              String(name),
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          {gpuIndexes.map((idx, i) => {
            const color = GPU_COLORS[i % GPU_COLORS.length];
            return (
              <Area
                key={idx}
                type="monotone"
                dataKey={`gpu${idx}`}
                name={`GPU ${idx}`}
                stroke={color}
                fill={`url(#gpu-fill-${idx})`}
                strokeWidth={2}
                connectNulls={false}
                isAnimationActive={false}
              />
            );
          })}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
