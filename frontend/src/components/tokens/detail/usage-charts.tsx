"use client";

// The four usage panels (spec §4.4; mockup `.charts`).

import { useMemo, useState } from "react";
import {
  binIsSplit, fillBins, modelRatesByBin, startsBeforeByModel, timingSampleLine,
  type SeriesBin, type TokenSeries,
} from "@/lib/token-series";
import { compact, day, fmtDateTime, secs } from "@/lib/token-format";
import { cn } from "@/lib/utils";
import { modelStyle, SERIES, type ModelMark } from "./styles";
import { hatchSpec, TimeChart, type ChartLine, type NumKey, type TipRow } from "./time-chart";

function Panel({ title, control, foot, children }: {
  title: string; control?: React.ReactNode; foot?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-vw-rule-soft/50 bg-chat-surface px-3 pb-2 pt-3">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-[14px] font-semibold">{title}</h3>
        {control}
      </div>
      {children}
      <div className="mt-0.5 min-h-[18px] text-[12px] text-chat-dim">{foot}</div>
    </div>
  );
}

function MiniSeg<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: [T, string][]; onChange: (v: T) => void;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex overflow-hidden rounded-[5px] border border-chat-rule">
      {options.map(([v, text]) => (
        <button
          key={v}
          type="button"
          aria-pressed={value === v}
          onClick={() => onChange(v)}
          className={cn("cursor-pointer px-[9px] py-0.5 text-[12px]", value === v ? "bg-chat-surface-2 text-chat-fg" : "bg-transparent text-chat-muted")}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

type LegendItem = { label: string; color: string; dashed?: boolean; mark?: ModelMark };

function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="flex flex-wrap gap-3 text-[12px] text-chat-muted">
      {items.map((i) => (
        <span key={i.label}>
          <i
            aria-hidden
            className="mr-[5px] inline-block h-0 w-3.5 border-t-2 align-middle"
            style={{ borderTopColor: i.color, borderTopStyle: i.mark ?? (i.dashed ? "dashed" : "solid") }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}

const TIMING_KEYS: NumKey[] = ["queue_p50", "queue_p95", "ttft_p50", "ttft_p95", "duration_p50", "duration_p95"];

export function UsageCharts({ series, nowSec }: { series: TokenSeries; nowSec: number }) {
  const [tok, setTok] = useState<"prompt" | "completion">("prompt");
  const [lat, setLat] = useState<"ttft" | "dur">("ttft");
  const w = series.bin_minutes;
  const fromMin = series.from_minute;
  const toMin = series.to_minute;
  const bins = useMemo(() => fillBins(series), [series]);

  const latSince = series.latency_since ?? nowSec;
  // Mockup `latKnown`: a bin that ends before latency_since has no timings.
  const latBins = useMemo(
    () => bins.map((b) => ((b.minute + w) * 60 > latSince ? b
      : ({ ...b, ...Object.fromEntries(TIMING_KEYS.map((k) => [k, null])) } as SeriesBin))),
    [bins, w, latSince],
  );
  const hatch = hatchSpec(fromMin, toMin, series.latency_since, nowSec);
  const latNote = fromMin * 60 < latSince
    ? `Per-request timings are kept from ${day(latSince)} onwards, for 30 days.`
    : "Median solid, 95th percentile dashed. Bins with no requests are left empty.";
  const sample = timingSampleLine(series.timing_sample);
  const timingFoot = (
    <>
      {latNote}
      {sample && <div>{sample}</div>}
    </>
  );

  const T = tok === "prompt"
    ? { key: "prompt_per_min" as const, peak: "peak_prompt" as const, color: SERIES.prompt, name: "Prefill" }
    : { key: "completion_per_min" as const, peak: "peak_completion" as const, color: SERIES.completion, name: "Generation" };
  const tokens = useMemo(() => tokenLines(series, T.key, T.color, nowSec), [series, T.key, T.color, nowSec]);
  const L = lat === "ttft"
    ? { a: "ttft_p50" as const, b: "ttft_p95" as const, color: SERIES.ttft }
    : { a: "duration_p50" as const, b: "duration_p95" as const, color: SERIES.dur };
  const common = { bins, binMinutes: w, fromMin, toMin };

  const tokenTip = (b: SeriesBin): TipRow[] => [
    { name: `${T.name} / min`, color: tokens.split ? SERIES.muted : T.color, value: compact(b[T.key]) },
    ...(tokens.split ? tokens.modelRows(b) : []),
    ...(w > 1 ? [{ name: "Busiest minute", color: SERIES.dim, value: compact(b[T.peak]) }] : []),
  ];

  return (
    <div className="grid grid-cols-2 gap-3.5 [@media(max-width:720px)]:grid-cols-1">
      <Panel
        title="Tokens per minute"
        control={<MiniSeg label="Token series" value={tok} onChange={setTok} options={[["prompt", "Prefill"], ["completion", "Generation"]]} />}
        foot={
          <>
            Average per minute within each bin, idle minutes counted as zero. Hover for the busiest minute.
            {tokens.note && <div>{tokens.note}</div>}
          </>
        }
      >
        {tokens.split && (
          <div className="mb-1" data-testid="model-legend">
            <Legend items={tokens.legend} />
          </div>
        )}
        <TimeChart
          name="tokens"
          {...common}
          lines={tokens.lines}
          yFmt={compact}
          tipRows={tokenTip}
        />
      </Panel>

      <Panel title="Requests per minute" control={<Legend items={[{ label: "Requests", color: SERIES.requests }]} />}>
        <TimeChart
          name="requests"
          {...common}
          lines={[{ key: "requests_per_min", color: SERIES.requests }]}
          yFmt={compact}
          tipRows={(b) => [
            { name: "Requests / min", color: SERIES.requests, value: compact(b.requests_per_min) },
            { name: "In this bin", color: SERIES.dim, value: String(Math.round(b.requests)) },
          ]}
        />
      </Panel>

      <Panel
        title="Queue wait"
        control={<Legend items={[{ label: "Median", color: SERIES.queue }, { label: "95th percentile", color: SERIES.queue, dashed: true }]} />}
        foot={timingFoot}
      >
        <TimeChart
          name="queue"
          {...common}
          bins={latBins}
          hatch={hatch}
          lines={[{ key: "queue_p95", color: SERIES.queue, dashed: true }, { key: "queue_p50", color: SERIES.queue }]}
          yFmt={secs}
          tipRows={(b) => [
            { name: "Median", color: SERIES.queue, value: secs(b.queue_p50) },
            { name: "95th pct", color: SERIES.queue, value: secs(b.queue_p95) },
            { name: "Requests", color: SERIES.dim, value: String(b.n) },
          ]}
        />
      </Panel>

      <Panel
        title="Latency"
        control={<MiniSeg label="Latency series" value={lat} onChange={setLat} options={[["ttft", "First token"], ["dur", "Full response"]]} />}
        foot={timingFoot}
      >
        <TimeChart
          name="latency"
          {...common}
          bins={latBins}
          hatch={hatch}
          lines={[{ key: L.b, color: L.color, dashed: true }, { key: L.a, color: L.color }]}
          yFmt={secs}
          tipRows={(b) => [
            { name: "Median", color: L.color, value: secs(b[L.a]) },
            { name: "95th pct", color: L.color, value: secs(b[L.b]) },
            { name: "Requests", color: SERIES.dim, value: String(b.n) },
          ]}
        />
      </Panel>
    </div>
  );
}


/** The tokens chart's lines. One line per model (`by_model` order and
 *  colours) when the window saw more than one model; bins before per-model
 *  recording began keep the total, as a dashed "All models" line. With one
 *  model (or none) it is the single total line, exactly as before 0036. */
export function tokenLines(
  series: TokenSeries,
  key: "prompt_per_min" | "completion_per_min",
  color: string,
  nowSec: number,
): {
  split: boolean;
  lines: ChartLine[];
  legend: LegendItem[];
  modelRows: (b: SeriesBin) => TipRow[];
  note: string | null;
} {
  const models = series.by_model;
  if (models.length <= 1) {
    return { split: false, lines: [{ key, color }], legend: [], modelRows: () => [], note: null };
  }
  const since = series.by_model_since;
  const rates = modelRatesByBin(series);
  const rate = (b: SeriesBin, id: string) => rates.get(b.minute)?.[id]?.[key] ?? 0;
  const before = startsBeforeByModel(series);
  const lines: ChartLine[] = [];
  const legend: LegendItem[] = [];
  if (before) {
    lines.push({ key: "all-models", color: SERIES.muted, dashed: true, get: (b) => (binIsSplit(b.minute, since) ? null : b[key]) });
    legend.push({ label: "All models", color: SERIES.muted, dashed: true });
  }
  models.forEach((m, i) => {
    // `m<i>`, not the model id: a recharts dataKey is a property path, and an
    // id is free text.
    const { color: c, mark, dash } = modelStyle(i);
    lines.push({ key: `m${i}`, color: c, dash, get: (b) => (binIsSplit(b.minute, since) ? rate(b, m.model_id) : null) });
    legend.push({ label: m.model, color: c, mark });
  });
  return {
    split: true,
    lines,
    legend,
    modelRows: (b) => (binIsSplit(b.minute, since)
      ? models
        .map((m, i) => ({ name: m.model, color: modelStyle(i).color, v: rate(b, m.model_id) }))
        .filter((r) => r.v > 0)
        .map(({ name, color, v }) => ({ name, color, value: compact(v) }))
      : []),
    note: before && since != null ? `Not split by model before ${fmtDateTime(since, nowSec)}.` : null,
  };
}
