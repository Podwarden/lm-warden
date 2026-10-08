"use client";

// Prefill and generation tokens per second, each as average / max / mode.
//
// This REPLACES the "Tokens / sec" tile, which showed
// (prompt + completion) / 60 over the last full minute. That number blended
// two quantities that move independently — prefill is compute-bound and runs
// in the hundreds or thousands of tok/s, generation is memory-bandwidth-bound
// and runs in the tens — so it described neither, and it moved when the
// prompt-to-completion ratio moved even though the hardware had not.
//
// Three statistics rather than one, because neither series is normal.
//
// WHY WALL CLOCK IS THE DEFAULT, learned the hard way on a production deployment:
// the per-request basis cannot answer "how fast is this rig", in BOTH
// directions at once.
//
//   prompt/ttft   The median request there carries a 46,000-token prompt and
//                 sees its first token in 1.9s -- about 26,000 tok/s. Prefilling
//                 46k tokens through a 27B is ~2.5e15 FLOPs; four A4000s at a
//                 generous 100 TFLOPS would need ~25 SECONDS. Those tokens were
//                 never computed: vLLM's prefix cache served them. The figure is
//                 a cache-hit measure, not a compute rate, so it is NOT called
//                 prefill any more.
//   decode rate   (completion-1)/(duration-ttft) is what ONE request experienced
//                 while the engine batched it against others -- a share of the
//                 aggregate, not the engine's speed. Measured 1.5 tok/s per
//                 request where the box as a whole does far more.
//
// Wall clock reads the minute rollup and is distorted by neither, so it opens
// there. The per-request basis stays available and now says what it measures.
//
// One caveat this panel used to state and no longer does: prefill does NOT
// include the warden's admission queue. `started_monotonic` in
// app/proxy/routes.py is read once the scheduler slot is held, so that wait
// was never inside ttft_s; it is its own `queued_s` column as of migration
// 0032. What remains inside ttft_s is the ENGINE's waiting queue (vLLM's
// continuous batching), which the proxy has no vantage point on.

import { formatInt } from "@/lib/live-stats";
import {
  coverageNote,
  type RateSummary,
  type ThroughputBasis,
  type ThroughputResponse,
} from "@/lib/request-history";
import { formatTps, type StatsRange } from "@/lib/stats-v2";
import { Panel } from "./live-panels";

const DASH = "—";

/** A rate, or a dash for absent. Zero is a READING (an idle minute on the
 *  wall-clock basis) and prints as 0; absent is not a rate at all. */
function formatRate(v: number | null): string {
  return v === null ? DASH : formatTps(v);
}

export function BasisToggle({
  basis,
  onChange,
}: {
  basis: ThroughputBasis;
  onChange: (b: ThroughputBasis) => void;
}) {
  const opts: { key: ThroughputBasis; label: string; title: string }[] = [
    {
      key: "request",
      label: "per request",
      title:
        "Engine speed measured on each request: prompt tokens over time-to-first-token, and decode tokens over the decode phase. Idle time is invisible to it.",
    },
    {
      key: "wallclock",
      label: "wall clock",
      title:
        "Tokens counted per minute across the window, with idle minutes included as zero. How much work the box did, not how fast it was when busy.",
    },
  ];
  return (
    <div
      role="group"
      aria-label="Throughput basis"
      data-testid="throughput-basis"
      className="inline-flex h-7 rounded-md border border-chat-rule bg-chat-surface/50 p-0.5 text-[11px]"
    >
      {opts.map((o) => {
        const active = o.key === basis;
        return (
          <button
            key={o.key}
            type="button"
            title={o.title}
            onClick={() => onChange(o.key)}
            aria-pressed={active}
            data-active={active}
            className={
              "rounded px-2.5 transition-colors " +
              (active
                ? "bg-chat-accent/20 text-chat-fg shadow-inner"
                : "text-chat-muted hover:bg-chat-surface-2")
            }
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function RateColumn({
  name,
  testid,
  summary,
  caption,
  maxTitle,
}: {
  name: string;
  testid: string;
  summary: RateSummary;
  caption: string;
  maxTitle?: string;
}) {
  const rows: { key: string; label: string; value: number | null; title?: string }[] = [
    { key: "avg", label: "avg", value: summary.avg },
    { key: "max", label: "max", value: summary.max, title: maxTitle },
    {
      key: "mode",
      label: "mode",
      value: summary.mode,
      title:
        "The most common reading, over bins about 5% wide. Absent when the sample is too small or nothing repeats.",
    },
  ];
  // Horizontal stat row rather than a stacked list: this panel sits beside the
  // Preemptions tile and has to match its height, not tower over it.
  return (
    <div className="space-y-1.5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-chat-muted">
          {name}
        </p>
        <p className="text-[11px] leading-tight text-chat-dim">{caption}</p>
      </div>
      <dl className="flex items-baseline gap-4">
        {rows.map((r) => (
          <div key={r.key} className="min-w-0" title={r.title}>
            <dt className="text-[10px] uppercase tracking-wider text-chat-dim">
              {r.label}
            </dt>
            <dd className="flex items-baseline gap-1">
              <span
                data-testid={`tp-${testid}-${r.key}`}
                className="text-lg font-semibold tabular-nums text-chat-fg"
              >
                {formatRate(r.value)}
              </span>
              <span className="text-[10px] text-chat-dim">tok/s</span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function ThroughputPanel({
  data,
  basis,
  onBasisChange,
  isLoading,
  error,
  range,
  human,
  className,
}: {
  data: ThroughputResponse | undefined;
  basis: ThroughputBasis;
  onBasisChange: (b: ThroughputBasis) => void;
  isLoading: boolean;
  error: unknown;
  range: StatsRange;
  /** Human label for the window ("past 24 hours"). Defaults to the range. */
  human?: string;
  /** Grid placement from the page — this panel is laid out by its parent. */
  className?: string;
}) {
  const windowLabel = human ?? `past ${range}`;
  const perRequest = basis === "request";
  const empty =
    !!data && data.prefill.count === 0 && data.generation.count === 0;
  const coverage = data
    ? coverageNote(data.coverage, data.since_epoch, Date.now() / 1000, windowLabel)
    : null;

  return (
    <Panel
      title="Token throughput"
      testid="throughput-panel"
      className={className}
      right={<BasisToggle basis={basis} onChange={onBasisChange} />}
    >
      <div className="space-y-2 p-4">
        {error ? (
          <p className="py-4 text-center text-sm text-chat-negative">
            Throughput unavailable.
          </p>
        ) : isLoading && !data ? (
          <p className="py-4 text-center text-sm text-chat-dim">Loading…</p>
        ) : empty ? (
          <p data-testid="throughput-empty" className="py-4 text-center text-sm text-chat-dim">
            {perRequest
              ? `No requests completed in the ${windowLabel}.`
              : `No traffic recorded in the ${windowLabel}.`}
          </p>
        ) : data ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <RateColumn
              name={perRequest ? "Prompt intake" : "Prompt"}
              testid="prefill"
              summary={data.prefill}
              caption={
                perRequest
                  ? "prompt tokens ÷ TTFT · counts prefix-cache hits, so not a prefill rate"
                  : "prompt tokens ÷ minute"
              }
              maxTitle={
                perRequest
                  ? "A prefix-cache hit counts prompt tokens the engine never computed, so this reads far above what the hardware can prefill. It measures cache hits, not compute."
                  : undefined
              }
            />
            <RateColumn
              name="Generation"
              testid="generation"
              summary={data.generation}
              caption={
                perRequest
                  ? "decode tokens ÷ decode seconds · one request's share, not the engine total"
                  : "completion tokens ÷ minute"
              }
            />
          </div>
        ) : null}

        <p data-testid="throughput-note" className="text-[10px] leading-tight text-chat-dim">
          {data
            ? perRequest
              ? `${formatInt(data.prefill.count)} requests · ${windowLabel}`
              : `${formatInt(data.prefill.count)} minutes · ${windowLabel} · idle minutes counted as zero`
            : windowLabel}
          {perRequest && data && !empty
            ? " · excludes the admission queue (its own column since 0032); the engine's waiting queue is still inside TTFT"
            : ""}
          {coverage ? ` · ${coverage}` : ""}
        </p>
      </div>
    </Panel>
  );
}
