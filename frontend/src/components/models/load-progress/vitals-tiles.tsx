"use client";

// Three vital-signs tiles under the running step: CPU cores (area chart),
// GPU memory (area chart), and last log line age (text).
//
// The data comes from ``VitalsOut`` (Task 6): ``series`` is an array of up
// to 120 points with ``t_s`` (seconds since load start), ``cpu_cores`` and
// ``gpu_mem_mib`` (``null`` where a sample is missing).  ``gpu_mem_total_mib``
// is the summed capacity of the model's GPUs, or ``null`` when not known
// (e.g. the docker driver).  ``last_log_age_s`` / ``last_log_line`` describe
// the most recent log line, and ``log_quiet_normal`` is the "slow but alive"
// verdict.
//
// No red anywhere — the colour rule from the owner.  Stalled/give_up states
// grey the charts (strokes go muted); the page passes a ``stalled`` flag.
//
// Charts use recharts (already a dependency).  The house style follows
// ``gpu-util-chart.tsx``: dark theme, no animation, slate axis styling.
// Charts are ~120 px tall, in the same three-column row.  On narrow widths
// they stack.  X axis in mm:ss since load start.  Null points are gaps,
// not zeros.

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { VitalsOut, VitalsPointOut } from "@/lib/load-progress";
import { SERIES } from "@/components/tokens/detail/styles";

export interface VitalsTilesProps {
  vitals: VitalsOut;
  /** Grey out the tiles when the load is stalled. */
  stalled?: boolean;
}

// Normalise the typed series points for the chart: optional fields become
// explicit nulls, so a missing sample is a gap for recharts.
interface SeriesPoint {
  t_s: number;
  cpu_cores: number | null;
  gpu_mem_mib: number | null;
}

function toPoints(vitals: VitalsOut): SeriesPoint[] {
  return (vitals.series ?? []).map((p: VitalsPointOut) => ({
    t_s: p.t_s,
    cpu_cores: p.cpu_cores ?? null,
    gpu_mem_mib: p.gpu_mem_mib ?? null,
  }));
}

/** Format seconds since load start as mm:ss for the X axis. */
function fmtMmSs(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function formatAge(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return r === 0 ? `${m}m` : `${m}m${r}s`;
}

/** Compact y-axis label for a MiB value: "860G", "1.7T". Short enough for a 40 px axis. */
export function formatAxisMiB(mib: number): string {
  const gib = mib / 1024;
  if (gib >= 1000) return `${(gib / 1024).toFixed(1)}T`;
  return `${Math.round(gib)}G`;
}

function formatGiB(mib: number): string {
  const gib = mib / 1024;
  if (gib >= 100) return `${Math.round(gib)} GiB`;
  if (gib >= 10) return `${gib.toFixed(1)} GiB`;
  return `${gib.toFixed(2)} GiB`;
}

export function VitalsTiles({ vitals, stalled = false }: VitalsTilesProps) {
  const points = toPoints(vitals);
  const logAge = vitals.last_log_age_s ?? null;
  const logLine = vitals.last_log_line ?? null;
  const quietNormal = vitals.log_quiet_normal === true;
  const gpuTotal = vitals.gpu_mem_total_mib ?? null;

  // CPU headline: the last non-null cpu_cores value.
  const lastCpu = [...points].reverse().find((p) => p.cpu_cores != null)?.cpu_cores ?? null;

  // GPU headline: the last non-null gpu_mem_mib value.
  const lastGpu = [...points].reverse().find((p) => p.gpu_mem_mib != null)?.gpu_mem_mib ?? null;

  // Whether ALL gpu_mem_mib values are null (docker driver).  Only true when
  // there ARE samples but every gpu_mem_mib is null — an empty series is
  // "no samples yet", not "not available".
  const gpuAllNull = points.length > 0 && points.every((p) => p.gpu_mem_mib == null);
  // The docker driver never reports GPU memory. On the local driver an
  // all-null series only means the engine has not claimed its GPUs yet.
  const gpuUnsupported = vitals.gpu_mem_supported === false;

  const tileClass = stalled
    ? "rounded-md border border-slate-700 bg-slate-900/30 p-3 text-slate-500"
    : "rounded-md border border-slate-700 bg-slate-900/50 p-3 text-slate-300";

  // Chart stroke colours: muted when stalled, blue/emerald when active.
  // Theme tokens, not hex: the light theme redefines these variables.
  const cpuStroke = stalled ? SERIES.dim : SERIES.prompt;
  const gpuStroke = stalled ? SERIES.dim : SERIES.completion;
  const axisStroke = SERIES.axis;
  const gridStroke = SERIES.grid;

  return (
    <div
      className="grid grid-cols-1 gap-2 sm:grid-cols-3"
      data-testid="vitals-tiles"
    >
      {/* CPU chart */}
      <div className={tileClass} data-testid="vital-cpu">
        <div className="text-xs font-medium text-slate-400">CPU</div>
        <div className="mt-0.5 text-sm text-slate-300" data-testid="vital-cpu-headline">
          {lastCpu != null ? `${lastCpu.toFixed(1)} cores busy` : "—"}
        </div>
        <div className="mt-1 h-[120px]">
          {points.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="cpu-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={cpuStroke} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={cpuStroke} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="t_s"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={fmtMmSs}
                  stroke={axisStroke}
                  tick={{ fontSize: 10, fill: axisStroke }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke={axisStroke}
                  tick={{ fontSize: 10, fill: axisStroke }}
                  tickLine={false}
                  axisLine={false}
                  width={28}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgb(var(--chat-surface))",
                    color: "rgb(var(--chat-fg))",
                    border: `1px solid ${SERIES.grid}`,
                    borderRadius: "4px",
                    fontSize: "11px",
                  }}
                  labelFormatter={(label: unknown) => fmtMmSs(Number(label))}
                  formatter={(v: unknown) => [
                    v == null ? "—" : `${Number(v).toFixed(2)} cores`,
                    "CPU",
                  ]}
                />
                <Area
                  type="monotone"
                  dataKey="cpu_cores"
                  stroke={cpuStroke}
                  strokeWidth={1.5}
                  fill="url(#cpu-fill)"
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center text-xs text-slate-500">
              No samples yet
            </div>
          )}
        </div>
      </div>

      {/* GPU memory chart */}
      <div className={tileClass} data-testid="vital-gpu">
        <div className="text-xs font-medium text-slate-400">GPU memory</div>
        <div className="mt-0.5 text-sm text-slate-300" data-testid="vital-gpu-headline">
          {gpuUnsupported ? (
            <span className="text-slate-500" data-testid="vital-gpu-unavailable">
              Not available with the docker driver
            </span>
          ) : gpuAllNull ? (
            <span className="text-slate-500" data-testid="vital-gpu-waiting">
              Waiting for the engine to claim GPU memory
            </span>
          ) : lastGpu != null ? (
            <span>
              {formatGiB(lastGpu)}
              {gpuTotal != null && ` / ${formatGiB(gpuTotal)}`}
            </span>
          ) : (
            "—"
          )}
        </div>
        <div className="mt-1 h-[120px]">
          {gpuAllNull ? (
            <div className="flex h-full items-center text-xs text-slate-500" />
          ) : points.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="gpu-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={gpuStroke} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={gpuStroke} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="t_s"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={fmtMmSs}
                  stroke={axisStroke}
                  tick={{ fontSize: 10, fill: axisStroke }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke={axisStroke}
                  tick={{ fontSize: 10, fill: axisStroke }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                  domain={gpuTotal != null ? [0, gpuTotal] : undefined}
                  tickFormatter={formatAxisMiB}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgb(var(--chat-surface))",
                    color: "rgb(var(--chat-fg))",
                    border: `1px solid ${SERIES.grid}`,
                    borderRadius: "4px",
                    fontSize: "11px",
                  }}
                  labelFormatter={(label: unknown) => fmtMmSs(Number(label))}
                  formatter={(v: unknown) => [
                    v == null ? "—" : formatGiB(Number(v)),
                    "GPU",
                  ]}
                />
                <Area
                  type="monotone"
                  dataKey="gpu_mem_mib"
                  stroke={gpuStroke}
                  strokeWidth={1.5}
                  fill="url(#gpu-fill)"
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center text-xs text-slate-500">
              No samples yet
            </div>
          )}
        </div>
      </div>

      {/* Last log line — text tile (unchanged from the old design) */}
      <div className={tileClass} data-testid="vital-log">
        <div className="text-xs font-medium text-slate-400">Last log line</div>
        <div className="mt-1 text-sm">
          {logAge !== null ? (
            <span data-testid="vital-log-age">{formatAge(logAge)} ago</span>
          ) : (
            <span className="text-slate-500">—</span>
          )}
          {quietNormal && (
            <span
              className="ml-2 text-xs text-emerald-400"
              data-testid="vital-log-quiet-normal"
            >
              normal here
            </span>
          )}
        </div>
        {logLine && (
          <div
            className="mt-0.5 truncate text-xs text-slate-500"
            title={logLine}
            data-testid="vital-log-line"
          >
            {logLine}
          </div>
        )}
      </div>
    </div>
  );
}
