"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Bar, BarChart, Tooltip, XAxis, YAxis } from "recharts";
import { tokenBars } from "@/lib/forest/panels";
import type { ForestState } from "@/lib/forest/types";
import { formatCompact } from "@/lib/live-stats";
import { GlassDetails } from "./glass";

const MINUTES = 30;

/**
 * An axis tick: short enough for its column (≤ 5 characters), so no digit is clipped: "150k", "2.5k", "1.2M". The
 * shared compact format ("150.0k") was clipped to its last characters in the 34 px axis and read as repeats.
 */
export function formatAxis(n: number): string {
  if (!Number.isFinite(n)) return "";
  const a = Math.abs(n);
  const short = (v: number, unit: string) => `${v >= 10 ? Math.round(v) : Number(v.toFixed(1))}${unit}`;
  if (a >= 1e6) return short(n / 1e6, "M");
  if (a >= 1e3) return short(n / 1e3, "k");
  return String(Math.round(n));
}
/** The token colours: cached (teal) → computed (orange); generated in gold, on its own axis. */
const COLORS = { cached: "#5cc6b3", computed: "#ec8a52", gen: "#e0b84d" } as const;
const LABELS = { cached: "prompt from cache", computed: "prompt computed", gen: "generated" } as const;
/** Chart width before the panel is measured (and in environments without ResizeObserver). */
const W = 320;
const H = 130;

/**
 * The axis' ticks (re-review N4): 0, then even steps of 1, 2, 2.5 or 5 × 10^k up to the first at or above `max`, at most
 * 4 steps. Explicit, so the chart never thins them (it dropped 0 and 450k and the rest read as a log scale).
 */
export function niceTicks(max: number): number[] {
  if (!(max > 0) || !Number.isFinite(max)) return [0, 1];
  const p = 10 ** Math.floor(Math.log10(max / 4));
  // token counts are whole: never a step under 1, and never 2.5 below 10
  const step = Math.max(1, [1, 2, 2.5, 5, 10].map((m) => m * p).filter((st) => st >= 1 && (st >= 10 || Number.isInteger(st))).find((st) => Math.ceil(max / st) <= 4) ?? 10 * p);
  return Array.from({ length: Math.ceil(max / step - 1e-9) + 1 }, (_, i) => i * step);
}

/** One axis label, formatted by `formatAxis` (data-axis-tick: the component test reads them). */
function axisTick(fill: string, anchor: "start" | "end") {
  function AxisTick(p: { x?: number | string; y?: number | string; payload?: { value: number } }) {
    const v = p.payload?.value ?? 0;
    return (
      <text x={Number(p.x) + (anchor === "end" ? -2 : 2)} y={p.y} dy={3} fill={fill} fontSize={9} textAnchor={anchor} data-axis-tick={anchor === "end" ? "left" : "right"} data-value={v}>
        {formatAxis(v)}
      </text>
    );
  }
  return AxisTick;
}

/**
 * Right, top (spec §6.4): one bar per minute for the last 30 completed minutes before the shown forest time `atRel`
 * (relative to `state.t0`), on wall-clock minutes. Prompt from cache and computed are stacked; generated has its own scale.
 */
export function TokenBars({ state, atRel }: { state: ForestState | null; atRel: number | null }) {
  const bars = useMemo(
    () => (state && atRel !== null ? tokenBars(state, atRel, MINUTES) : []),
    [state, atRel],
  );
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(W);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const measure = () => el.clientWidth > 0 && setWidth(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const total = bars.reduce((a, b) => a + b.cached + b.computed + b.gen, 0);
  const pt = niceTicks(Math.max(0, ...bars.map((b) => b.cached + b.computed)));
  const gt = niceTicks(Math.max(0, ...bars.map((b) => b.gen)));
  const clock = (minute: number) =>
    new Date(minute * 60 * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); // minute: wall-clock
  return (
    <GlassDetails
      title="Tokens per minute"
      note="last 30 min"
      storageKey="forest.tokens.open"
      defaultOpen={() => true}
      testid="forest-tokens"
    >
      <div ref={box} data-testid="token-bars" data-total={total}>
        {bars.length === 0 ? (
          <p className="py-6 text-center text-white/60">Waiting for data…</p>
        ) : (
          <BarChart width={width} height={H} data={bars} margin={{ top: 6, right: 0, bottom: 6, left: 0 }} barCategoryGap={1}>
            <XAxis dataKey="minute" hide />
            <YAxis yAxisId="prompt" width={34} ticks={pt} domain={[0, pt.at(-1)!]} interval={0} tickFormatter={formatAxis} tick={axisTick("rgba(238,242,236,.6)", "end")} axisLine={false} tickLine={false} />
            <YAxis yAxisId="gen" orientation="right" width={30} ticks={gt} domain={[0, gt.at(-1)!]} interval={0} tickFormatter={formatAxis} tick={axisTick(COLORS.gen, "start")} axisLine={false} tickLine={false} />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,.08)" }}
              contentStyle={{ background: "rgba(14,18,16,.9)", border: "1px solid rgba(255,255,255,.12)", borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: "#eef2ec" }}
              labelFormatter={(m) => clock(Number(m))}
              formatter={(v, name) => [formatCompact(Number(v)), LABELS[name as keyof typeof LABELS] ?? name]}
            />
            <Bar yAxisId="prompt" dataKey="cached" stackId="p" fill={COLORS.cached} isAnimationActive={false} />
            <Bar yAxisId="prompt" dataKey="computed" stackId="p" fill={COLORS.computed} isAnimationActive={false} />
            <Bar yAxisId="gen" dataKey="gen" fill={COLORS.gen} isAnimationActive={false} />
          </BarChart>
        )}
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-white/70">
          {(Object.keys(COLORS) as (keyof typeof COLORS)[]).map((k) => (
            <span key={k} className="inline-flex items-center gap-1">
              <span aria-hidden="true" className="h-2 w-2 rounded-sm" style={{ background: COLORS[k] }} />
              {LABELS[k]}
              {k === "gen" && " (right scale)"}
            </span>
          ))}
        </div>
      </div>
    </GlassDetails>
  );
}
