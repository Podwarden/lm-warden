"use client";

// Historical charts for the redesigned /stats page (S7, #124).
//
// Three charts, all minute-bucketed, all consuming `/api/stats/v2/overview`
// `series` arrays directly — no client-side pivoting (v1's whole pain
// point: see the comment block at the top of @/lib/stats). Each chart
// receives its own slice + the [start, end] bounds derived from the
// range selector.
//
// Visual language matches the v1 charts (gpu-util-chart, throughput-chart)
// so the page reads as one piece: slate panel, emerald primary, sibling
// hues for the secondary series. Animations are off everywhere — a 30s
// poll on a flickering chart is exhausting.

import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  ErrorBar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Rectangle,
  ResponsiveContainer,
  type BarShapeProps,
  type TooltipContentProps,
} from "recharts";
import type { CacheHitSummary, Reference, TokenRateBucket } from "@/lib/live-history";
import {
  withTs,
  formatTpsK,
  type StatsV2UtilPoint,
  type StatsV2PowerPoint,
  type StatsRange,
} from "@/lib/stats-v2";
import { rangeBounds } from "@/lib/stats";

// ---- shared tick formatters ---------------------------------------------

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtDate(ts: number): string {
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
  });
}

function pickTickFormatter(range: StatsRange): (ts: number) => string {
  // 7d resolution would render every-minute HH:MM ticks as a smear.
  return range === "7d" ? fmtDate : fmtTime;
}

// Empty-state / children switcher shared by all four chart components — keeps
// the dashed-border placeholder rendering in one place.
interface ChartShellProps {
  emptyLabel: string;
  hasData: boolean;
  /** Two stacked charts (tokens) need more room than one chart's h-64. */
  tall?: boolean;
  children: React.ReactNode;
}

function ChartShell({ emptyLabel, hasData, tall = false, children }: ChartShellProps) {
  if (!hasData) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
        {emptyLabel}
      </div>
    );
  }
  return (
    // text-chat-muted feeds `currentColor` inside the SVG: the 24h reference
    // lines below draw with it, so they resolve through the theme tokens
    // rather than a literal colour.
    <div className={`${tall ? "h-96" : "h-64"} w-full rounded-lg border border-slate-700 bg-slate-900/30 p-2 text-chat-muted`}>
      {children}
    </div>
  );
}

// ---- fixed 24h reference lines -------------------------------------------
//
// Dashed = the 24h median of BUSY minutes, dotted = the 24h peak. Fixed at
// 24h regardless of the selected window — a reference that moved with the
// view could never answer "is this normal for this box?". In retro-dark
// several tokens share a hue, so the two references differ by dash PATTERN
// and label, never by colour alone. `null` median (no busy minutes in 24h)
// draws no line rather than a baseline invented from idle samples.

// A reference label is placed from where its line actually sits in the
// domain, not from a fixed corner. Two failures drove this:
//
//   * a line at the TOP of the scale (GPU utilisation pinned at 100%, VRAM
//     at the card total) had its label drawn ABOVE the line, outside the
//     plot, where it was clipped by the panel edge;
//   * median and peak both hugged the RIGHT edge, so when the two values are
//     close (511 W against a 555 W peak) the labels sat on top of each other.
//
// So: a line in the upper part of the domain gets its label below, otherwise
// above; and peak keeps the right edge while median takes the left. They can
// never collide horizontally, whatever the two values are.
const LABEL_FLIP_FRACTION = 0.85;

export function refLabelPosition(value: number, yMax: number, side: "Left" | "Right") {
  const high = yMax > 0 && value / yMax > LABEL_FLIP_FRACTION;
  return `inside${high ? "Bottom" : "Top"}${side}` as const;
}

function refLines(
  reference: Reference | undefined,
  fmt: (v: number) => string,
  yMax: number,
  names: { peak: string; median: string } = { peak: "24h peak", median: "24h busy median" },
): React.ReactNode[] {
  if (!reference) return [];
  const out: React.ReactNode[] = [];
  if (reference.peak !== null) {
    out.push(
      <ReferenceLine
        key="ref-peak"
        y={reference.peak}
        stroke="currentColor"
        strokeOpacity={0.6}
        strokeDasharray="1 3"
        label={{
          value: `${names.peak} ${fmt(reference.peak)}`,
          position: refLabelPosition(reference.peak, yMax, "Right"),
          fill: "currentColor",
          fontSize: 10,
        }}
      />,
    );
  }
  if (reference.median !== null) {
    out.push(
      <ReferenceLine
        key="ref-median"
        y={reference.median}
        stroke="currentColor"
        strokeOpacity={0.6}
        strokeDasharray="6 4"
        label={{
          value: `${names.median} ${fmt(reference.median)}`,
          position: refLabelPosition(reference.median, yMax, "Left"),
          fill: "currentColor",
          fontSize: 10,
        }}
      />,
    );
  }
  return out;
}

// The peak is included in the y-scale ON PURPOSE: a chart auto-scaled to a
// quiet hour makes that hour look busy, which is the failure the reference
// exists to prevent.
function ceilingWith(dataMax: number, reference: Reference | undefined): number {
  const peak = reference?.peak ?? 0;
  return Math.max(dataMax, peak) * 1.08 || 1;
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0f172a",
  border: "1px solid #334155",
  fontSize: "12px",
};

// There is deliberately no VRAM-over-time chart. Both engines pre-allocate
// weights and KV cache at load (vLLM from gpu_memory_utilization, llama.cpp
// from n_ctx), so VRAM is a step function that moves only on load/unload
// and its chart is a flat block. Current VRAM is on the per-GPU cards in
// SystemConfigSection; the `vram` series stays in the API for other readers.

// ---- GPU util over time --------------------------------------------------

interface UtilChartProps {
  points: readonly StatsV2UtilPoint[];
  range: StatsRange;
  reference?: Reference;
}

export function UtilChart({ points, range, reference }: UtilChartProps) {
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  return (
    <ChartShell hasData={data.length > 0} emptyLabel="No GPU util samples in this window.">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="util-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#60a5fa" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
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
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? `${value.toFixed(0)}%` : String(value),
              String(name),
            ]}
          />
          <Area
            type="monotone"
            dataKey="max_pct"
            name="GPU util (max)"
            stroke="#60a5fa"
            fill="url(#util-fill)"
            strokeWidth={2}
            isAnimationActive={false}
          />
          {refLines(reference, (v) => `${Math.round(v)}%`, 100)}
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

// ---- Power over time -----------------------------------------------------

interface PowerChartProps {
  points: readonly StatsV2PowerPoint[];
  range: StatsRange;
  /** True iff the host has at least one card that reports power.draw.
   *  Lets the empty state distinguish "no samples yet" from "card can't
   *  do it" — a virtualised host will never produce power data and the
   *  operator should know to stop waiting. */
  supported?: boolean;
  reference?: Reference;
}

export function PowerChart({
  points,
  range,
  supported = true,
  reference,
}: PowerChartProps) {
  const data = withTs(points);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  const yMax = ceilingWith(
    data.reduce((m, p) => Math.max(m, p.watts), 0),
    reference,
  );
  const emptyLabel = supported
    ? "No power samples in this window."
    : "Power telemetry not reported by this host's GPUs.";
  return (
    <ChartShell hasData={data.length > 0} emptyLabel={emptyLabel}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <defs>
            <linearGradient id="power-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#fbbf24" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
          <XAxis
            dataKey="ts"
            type="number"
            domain={bounds}
            allowDataOverflow
            scale="time"
            tickFormatter={tickFmt}
            stroke="#94a3b8"
            fontSize={11}
          />
          <YAxis
            domain={[0, yMax]}
            stroke="#94a3b8"
            fontSize={11}
            width={56}
            tickFormatter={(v) => `${Math.round(Number(v))}`}
            label={{
              value: "W",
              angle: -90,
              position: "insideLeft",
              fill: "#64748b",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(v) => tickFmt(Number(v))}
            formatter={(value, name) => [
              typeof value === "number" ? `${value.toFixed(1)} W` : String(value),
              String(name),
            ]}
          />
          <Area
            type="monotone"
            dataKey="watts"
            name="Power draw"
            stroke="#fbbf24"
            fill="url(#power-fill)"
            strokeWidth={2}
            isAnimationActive={false}
          />
          {refLines(reference, (v) => `${Math.round(v)} W`, yMax)}
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

// ---- Tokens over time (tokens / second) -----------------------------------

interface TokensChartProps {
  /** Dense tok/s buckets over the window (tokenRateBuckets): one row per
   *  bucket, `*_tps` null before the measured range. */
  buckets: readonly TokenRateBucket[];
  /** The width the buckets were built with. > 1 draws the busiest-minute
   *  whisker and a range tooltip; 1 draws neither (the whisker would equal
   *  the bar). */
  bucketMinutes: number;
  range: StatsRange;
  /** Computed over the PROMPT series (the dominant one), like the rest a
   *  fixed 24h reference — in per-minute tokens; the chart converts to
   *  tok/s. */
  reference?: Reference;
  /** Cache hit over the last hour, for the Prompt header. */
  cacheSummary?: CacheHitSummary;
}

/** The cached part of the prompt: the sky token the in-flight table's cached
 *  segment uses, theme-aware (a CSS variable, not a hex). */
const CACHED_COLOR = "rgb(var(--vw-ttft))";
const PROMPT_COLOR = "#a78bfa";

// Prompt and completion differ by orders of magnitude (a prompt-heavy box
// ran 350,352 prompt against 36 completion in the same minute), so a shared
// axis draws completion as a flat line at 0. Each series gets its own axis:
// prompt keeps the existing ceilingWith rule (window max AND the fixed 24h
// reference peak, +8% headroom); completion scales from its own max alone.
// The bars are tok/s and the 24h reference is a per-MINUTE statistic, so the
// ceiling compares against peak / 60 — the same conversion the lines use.
export function tokensYMax(
  buckets: readonly TokenRateBucket[],
  reference: Reference | undefined,
): { prompt: number; completion: number } {
  let promptMax = 0;
  let completionMax = 0;
  for (const b of buckets) {
    if (b.prompt_tps !== null && b.prompt_tps > promptMax) promptMax = b.prompt_tps;
    if (b.completion_tps !== null && b.completion_tps > completionMax) completionMax = b.completion_tps;
  }
  const refTps = reference
    ? { ...reference, peak: reference.peak !== null ? reference.peak / 60 : null }
    : undefined;
  return {
    prompt: ceilingWith(promptMax, refTps),
    completion: completionMax * 1.08 || 1,
  };
}

// ---- the per-bar tooltip ---------------------------------------------------
//
// Bars use the mark as the hit target (no crosshair). The token SUM stays in
// the tooltip so the old "how many tokens" reading is not lost; wide buckets
// add the busiest minute, 1-min buckets read as a plain rate.

const WHISKER_EPS = 1e-9;

function tpsText(tps: number): string {
  return tps >= 10 ? Math.round(tps).toLocaleString() : tps.toFixed(1);
}

function tokenCountText(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(2)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}k`;
  return tokens.toLocaleString();
}

/** Prompt, cached, computed and cache-hit % of a bucket. `cached` is null when
 *  no request in the bucket was measured: then nothing is claimed about caching. */
export function promptSplit(b: TokenRateBucket): {
  prompt: number;
  cached: number | null;
  computed: number;
  pct: number | null;
} {
  if (b.cached_measured_requests <= 0) {
    return { prompt: b.prompt_tokens, cached: null, computed: b.prompt_tokens, pct: null };
  }
  const cached = Math.min(b.cached_tokens, b.prompt_tokens);
  return {
    prompt: b.prompt_tokens,
    cached,
    computed: b.prompt_tokens - cached,
    pct: b.prompt_tokens > 0 ? cached / b.prompt_tokens : null,
  };
}

interface TokensRateTipExtras {
  /** Prompt panel only: break the bucket into cached and computed. */
  split?: boolean;
  bucketMinutes: number;
  range: StatsRange;
  tps: (b: TokenRateBucket) => number | null;
  peak: (b: TokenRateBucket) => number | null;
  tokens: (b: TokenRateBucket) => number;
}

function TokensRateTip(props: TooltipContentProps<number, string> & TokensRateTipExtras & {
  series: string;
  color: string;
}) {
  const { active, payload, label, contentStyle, series, color, bucketMinutes, range, tps, peak, tokens, split } =
    props;
  if (!active || payload == null || payload.length === 0) return null;
  const row = payload[0].payload as TokenRateBucket & { ts?: number } | undefined;
  const startTs = row?.ts ?? Number(label);
  const tickFmt = pickTickFormatter(range);
  const rangeLabel =
    bucketMinutes > 1
      ? `${tickFmt(startTs)}-${tickFmt(startTs + bucketMinutes * 60_000)} (${bucketMinutes} min)`
      : tickFmt(startTs);
  let valueLabel: string;
  const rate = row ? tps(row) : null;
  if (row === undefined || rate === null) {
    valueLabel = "no data";
  } else if (bucketMinutes > 1) {
    const busiest = row ? peak(row) : null;
    valueLabel = `mean ${tpsText(rate)} tok/s${
      busiest !== null && busiest !== undefined ? `, busiest minute ${tpsText(busiest)} tok/s` : ""
    }, ${tokenCountText(tokens(row))} tokens`;
  } else {
    valueLabel = `${tpsText(rate)} tok/s (${tokenCountText(tokens(row))} tokens)`;
  }
  return (
    // Custom content is not wrapped in recharts' default-tooltip div, so the
    // shared TOOLTIP_STYLE is applied here (plus the padding that div gave).
    <div role="tooltip" style={{ ...TOOLTIP_STYLE, padding: 8, ...contentStyle }}>
      <div className="font-semibold">{rangeLabel}</div>
      <div>
        <span style={{ color }}>{series}:</span> {valueLabel}
      </div>
      {split && row && rate !== null && <PromptSplitLines row={row} />}
    </div>
  );
}

function PromptSplitLines({ row }: { row: TokenRateBucket }) {
  const sp = promptSplit(row);
  if (sp.cached === null) {
    return (
      <div data-testid="tokens-split-tip">
        No cached-token data for this period (not reported by the engine, or recorded before
        it was tracked).
      </div>
    );
  }
  const partial = row.cached_measured_requests < row.requests;
  return (
    <div data-testid="tokens-split-tip">
      <div>
        prompt {tokenCountText(sp.prompt)}: <span style={{ color: CACHED_COLOR }}>cached</span>{" "}
        {tokenCountText(sp.cached)} + <span style={{ color: PROMPT_COLOR }}>computed</span>{" "}
        {tokenCountText(sp.computed)}
      </div>
      <div>cache hit {sp.pct === null ? "n/a" : `${Math.round(sp.pct * 100)}%`}</div>
      {partial && (
        <div>
          {row.cached_measured_requests} of {row.requests} requests measured; the rest count as
          computed
        </div>
      )}
    </div>
  );
}

const CREDIT_NOTE =
  "Tokens are credited to the minute a request FINISHES, so a long request lands all at once and per-minute bars spike around the true rate.";

/** Legend with a swatch AND a label per series: colour is never the only cue. */
function CacheLegend() {
  const item = (color: string, label: string) => (
    <span className="inline-flex items-center gap-1 text-chat-muted">
      <span
        aria-hidden="true"
        className="inline-block h-2 w-2 rounded-sm"
        style={{ background: color }}
      />
      {label}
    </span>
  );
  return (
    <span data-testid="tokens-legend" className="inline-flex items-center gap-3">
      {item(CACHED_COLOR, "cached (measured)")}
      {item(PROMPT_COLOR, "computed")}
    </span>
  );
}

function CacheHitText({ summary }: { summary?: CacheHitSummary }) {
  if (!summary) return null;
  const partial = summary.measuredRequests < summary.requests;
  return (
    <span
      data-testid="cache-hit-summary"
      className="text-chat-muted"
      title={
        summary.pct === null
          ? "No request in the last hour reported cached tokens."
          : partial
            ? `${summary.measuredRequests} of ${summary.requests} requests measured; the rest count as computed, so this is a floor.`
            : "All requests in the last hour reported cached tokens."
      }
    >
      cache hit{" "}
      <span className="font-medium text-chat-fg">
        {summary.pct === null ? "not measured" : `${Math.round(summary.pct * 100)}%`}
      </span>{" "}
      (last hour)
    </span>
  );
}

export function TokensChart({
  buckets,
  bucketMinutes,
  range,
  reference,
  cacheSummary,
}: TokensChartProps) {
  // Two stacked small multiples with independent Y axes instead of two bars
  // on one axis: stacking would let the operator read "total throughput" at a
  // glance but obscure the mix, and one shared axis made completion invisible
  // (see tokensYMax). Same X domain and tick formatter in both; syncId keeps
  // the hover on the same bucket in both charts.
  const wide = bucketMinutes > 1;
  const data = withTs(buckets).map((b) => {
    // The whisker is the gap between the bar (the bucket MEAN) and the
    // busiest minute's rate: [0, peak - mean] relative to the bar's value.
    // null keeps recharts' own skip rule (no bar, no whisker).
    const whisker = (mean: number | null, peakTps: number | null): [number, number] | null =>
      wide && mean !== null && peakTps !== null && peakTps - mean > WHISKER_EPS
        ? [0, peakTps - mean]
        : null;
    const cachedTps = b.prompt_tps !== null ? b.prompt_cached_tps : null;
    return {
      ...b,
      // 0 rather than null inside a stack, so recharts still draws the
      // computed bar above it; a zero-height segment claims nothing.
      prompt_cached_tps: b.prompt_tps === null ? null : (cachedTps ?? 0),
      // The rest of the prompt: everything not MEASURED as cached, so a request
      // whose engine did not report it is computed by definition.
      prompt_computed_tps:
        b.prompt_tps === null ? null : Math.max(0, b.prompt_tps - (cachedTps ?? 0)),
      prompt_whisker: whisker(b.prompt_tps, b.prompt_peak_tps),
      completion_whisker: whisker(b.completion_tps, b.completion_peak_tps),
    };
  });
  // Stack the cached segment only when some bucket has a measurement: a
  // window of pre-migration rows stays plain purple, with no cached claim.
  const stackCached = buckets.some((b) => b.prompt_tps !== null && b.prompt_cached_tps !== null);
  // Unmeasured rows are "no data"; a window of only those reads as empty.
  const hasData = data.some((b) => b.measured);
  const bounds = rangeBounds(range);
  const tickFmt = pickTickFormatter(range);
  const yMax = tokensYMax(buckets, reference);
  const axisProps = {
    stroke: "#94a3b8",
    fontSize: 11,
    tickFormatter: (v: number) => formatTpsK(Number(v)),
    width: 72,
  };
  const xAxis = (
    <XAxis
      dataKey="ts"
      type="number"
      domain={bounds}
      allowDataOverflow
      scale="time"
      tickFormatter={tickFmt}
      stroke="#94a3b8"
      fontSize={11}
    />
  );
  const tooltip = (series: string, color: string, extra: TokensRateTipExtras) => (
    // filterNull off: a null bucket must still show "no data", not nothing.
    <Tooltip
      contentStyle={TOOLTIP_STYLE}
      cursor={false}
      filterNull={false}
      content={(p: TooltipContentProps<number, string>) => (
        <TokensRateTip {...p} series={series} color={color} {...extra} />
      )}
    />
  );
  const panel = (cfg: {
    testid: string;
    title: string;
    color: string;
    dataKey: "prompt_computed_tps" | "completion_tps";
    whiskerKey: "prompt_whisker" | "completion_whisker";
    /** Stack a cached segment under the main bar (the Prompt panel). */
    cachedKey?: "prompt_cached_tps";
    /** Header extras: the cache-hit summary and the info note. */
    header?: React.ReactNode;
    yMax: number;
    refs: React.ReactNode[];
    tip: TokensRateTipExtras;
  }) => (
    <div data-testid={cfg.testid} className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-1 pb-0.5 text-[11px]">
        <span className="font-semibold" style={{ color: cfg.color }}>
          {cfg.title}
        </span>
        {cfg.header}
      </div>
      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} syncId="tokens" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            {xAxis}
            <YAxis domain={[0, cfg.yMax]} {...axisProps} />
            {tooltip(cfg.title, cfg.color, cfg.tip)}
            {cfg.cachedKey && (
              <Bar
                dataKey={cfg.cachedKey}
                name="cached (measured)"
                stackId="prompt"
                fill={CACHED_COLOR}
                fillOpacity={0.9}
                maxBarSize={24}
                isAnimationActive={false}
                // The cached segment is the bar's TOP when nothing was computed
                // in the bucket, so it takes the rounded corners the computed
                // segment would otherwise carry.
                shape={(p: BarShapeProps) => (
                  <Rectangle
                    {...p}
                    radius={
                      (p.payload as { prompt_computed_tps?: number | null })
                        ?.prompt_computed_tps
                        ? 0
                        : [4, 4, 0, 0]
                    }
                  />
                )}
              />
            )}
            <Bar
              dataKey={cfg.dataKey}
              name={cfg.cachedKey ? "computed" : cfg.title}
              stackId={cfg.cachedKey ? "prompt" : undefined}
              fill={cfg.color}
              fillOpacity={0.85}
              radius={[4, 4, 0, 0]}
              maxBarSize={24}
              isAnimationActive={false}
            >
              {wide && (
                <ErrorBar
                  dataKey={cfg.whiskerKey}
                  stroke="currentColor"
                  strokeOpacity={0.7}
                  strokeWidth={1}
                  width={4}
                  isAnimationActive={false}
                />
              )}
            </Bar>
            {cfg.refs}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
  // The 24h references are per-minute values on a tok/s axis: divide both by
  // 60 (an unconverted line would sit outside the domain and recharts would
  // drop it), and relabel "…minute" so nobody reads them as a peak second.
  const refTps = reference
    ? {
        ...reference,
        peak: reference.peak !== null ? reference.peak / 60 : null,
        median: reference.median !== null ? reference.median / 60 : null,
      }
    : undefined;
  const refFmt = (v: number) => `${formatTpsK(v)} tok/s`;
  return (
    <ChartShell hasData={hasData} emptyLabel="No token usage in this window." tall>
      <div className="flex h-full min-h-0 flex-col gap-2">
        {panel({
          testid: "tokens-prompt-chart",
          title: "Prompt",
          color: PROMPT_COLOR,
          dataKey: "prompt_computed_tps",
          whiskerKey: "prompt_whisker",
          cachedKey: stackCached ? "prompt_cached_tps" : undefined,
          header: (
            <>
              <CacheLegend />
              <CacheHitText summary={cacheSummary} />
              <span
                data-testid="tokens-credit-note"
                tabIndex={0}
                role="note"
                aria-label={CREDIT_NOTE}
                title={CREDIT_NOTE}
                className="cursor-help text-chat-dim"
              >
                ⓘ
              </span>
            </>
          ),
          yMax: yMax.prompt,
          refs: refLines(refTps, refFmt, yMax.prompt, {
            peak: "24h peak minute",
            median: "24h busy-minute median",
          }),
          tip: {
            split: true,
            bucketMinutes,
            range,
            tps: (b) => b.prompt_tps,
            peak: (b) => b.prompt_peak_tps,
            tokens: (b) => b.prompt_tokens,
          },
        })}
        {panel({
          testid: "tokens-completion-chart",
          title: "Completion",
          color: "#34d399",
          dataKey: "completion_tps",
          whiskerKey: "completion_whisker",
          yMax: yMax.completion,
          refs: [],
          tip: {
            bucketMinutes,
            range,
            tps: (b) => b.completion_tps,
            peak: (b) => b.completion_peak_tps,
            tokens: (b) => b.completion_tokens,
          },
        })}
      </div>
    </ChartShell>
  );
}
