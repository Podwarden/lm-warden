"use client";

// Latency distributions from the persisted per-request history:
// time to first token, mean inter-token latency per request, and duration.
//
// These REPLACE the engine-histogram panels. The engine publishes only
// cumulative histograms, so the old "last 5 minutes" was really "since you
// opened the tab, at most 5 minutes", and llama.cpp publishes none at all.
// The proxy measures TTFT (first streamed frame) and duration for every
// backend identically, and those are what the store keeps — so every model
// gets these panels, and they honour the window.
//
// What is given up, plainly: the engine's ITL histogram is per TOKEN with
// real bucket resolution; the store's ITL is the MEAN gap of each request.
// Contention inside a request averages out here. The panel says "mean per
// request" so the two are never confused.
//
// Two bases, and the heading says which is showing: the page's window, or
// the last N requests. A fixed count does not go silent when traffic is
// thin, which makes it the better statistic for latency on a quiet box.

import { formatInt } from "@/lib/live-stats";
import {
  LATENCY_LAST_N,
  coverageNote,
  formatSpan,
  type LatencyBasis,
  type LatencyDistribution,
  type LatencyResponse,
} from "@/lib/request-history";
import { LatencyHistogram, Panel, ScopeNote } from "./live-panels";

export function BasisToggle({
  basis,
  onChange,
}: {
  basis: LatencyBasis;
  onChange: (b: LatencyBasis) => void;
}) {
  const opts: { key: LatencyBasis; label: string; title: string }[] = [
    { key: "window", label: "window", title: "Every request that finished inside the selected window." },
    {
      key: "last",
      label: `last ${LATENCY_LAST_N}`,
      title: `The newest ${LATENCY_LAST_N} requests regardless of time — does not go silent when traffic is thin.`,
    },
  ];
  return (
    <div
      role="group"
      aria-label="Latency basis"
      data-testid="latency-basis"
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
              (active ? "bg-chat-accent/20 text-chat-fg shadow-inner" : "text-chat-muted hover:bg-chat-surface-2")
            }
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** The scope note every latency panel carries: basis, count, span. */
export function latencyScope(data: LatencyResponse | undefined, human: string): string {
  if (!data) return human;
  if (data.basis === "last") {
    const n = data.n ?? LATENCY_LAST_N;
    if (data.count === 0) return `last ${n} requests · none recorded`;
    const span = data.span_s === null ? "" : ` · spanning ${formatSpan(data.span_s)}`;
    return `last ${formatInt(data.count)}${data.count < n ? ` of ${n}` : ""} requests${span}`;
  }
  return `${human} · n=${formatInt(data.count)}`;
}

function DistributionBody({
  dist,
  data,
  human,
  testid,
  empty,
}: {
  dist: LatencyDistribution;
  data: LatencyResponse;
  human: string;
  testid: string;
  empty: string;
}) {
  if (dist.count === 0) {
    const note =
      data.basis === "window"
        ? coverageNote(data.coverage, data.since_epoch, Date.now() / 1000, human)
        : null;
    return (
      <p className="py-4 text-sm text-chat-dim" data-testid={`${testid}-empty`}>
        {note ?? empty}
      </p>
    );
  }
  return (
    <LatencyHistogram
      testid={`${testid}-histogram`}
      hist={dist.buckets}
      markers={[
        { label: "p50", value: dist.p50 },
        { label: "p90", value: dist.p90 },
        { label: "p99", value: dist.p99 },
      ]}
    />
  );
}

export function LatencyPanels({
  data,
  human,
  isLoading,
  error,
}: {
  data: LatencyResponse | undefined;
  human: string;
  isLoading: boolean;
  error: unknown;
}) {
  const scope = latencyScope(data, human);
  const note =
    data && data.basis === "window" && data.count > 0
      ? coverageNote(data.coverage, data.since_epoch, Date.now() / 1000, human)
      : null;
  const body = (
    key: "ttft" | "itl" | "duration",
    testid: string,
    empty: string,
  ) => {
    if (isLoading && !data) {
      return <div className="h-28 w-full animate-pulse rounded bg-chat-surface-2/60" />;
    }
    if (error && !data) {
      return (
        <p className="py-4 text-sm text-chat-negative">
          Failed to load latency history
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      );
    }
    if (!data) return null;
    return (
      <DistributionBody dist={data[key]} data={data} human={human} testid={testid} empty={empty} />
    );
  };
  return (
    <div className="space-y-2">
      {note && (
        <p className="text-[11px] text-chat-warn" data-testid="latency-coverage">
          {note}
        </p>
      )}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel
          title="Time to first token"
          testid="ttft-panel"
          right={<ScopeNote>{scope}</ScopeNote>}
        >
          {body("ttft", "ttft", "No first tokens recorded in this scope.")}
          <p className="mt-3 text-[11px] leading-relaxed text-chat-dim">
            Measured by the proxy at the first streamed frame — every backend,
            the same way. Non-streaming requests have no first frame and are
            not counted here. TTFT is bimodal (prefix-cache hits against cold
            prefills), so read the shape, not the p50.
          </p>
        </Panel>
        <Panel
          title="Inter-token latency"
          testid="itl-panel"
          right={<ScopeNote>{scope} · mean per request</ScopeNote>}
        >
          {body("itl", "itl", "No multi-token responses recorded in this scope.")}
          <p className="mt-3 text-[11px] leading-relaxed text-chat-dim">
            The MEAN gap between tokens of each request: decode time over
            generated tokens, from the proxy. Not the engine&apos;s per-token
            histogram — contention inside a request averages out here.
          </p>
        </Panel>
        <Panel title="Duration" testid="duration-panel" right={<ScopeNote>{scope}</ScopeNote>}>
          {body("duration", "duration", "No requests recorded in this scope.")}
          <p className="mt-3 text-[11px] leading-relaxed text-chat-dim">
            Wall time from the proxy accepting the request to the end of its
            response. Long tails are long generations, not slow ones — see the
            requests chart below.
          </p>
        </Panel>
      </div>
    </div>
  );
}
