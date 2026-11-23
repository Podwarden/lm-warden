"use client";

// CACHE (design 3.5): one fixed card for the scope's model. The headline is the
// one defined figure: engine-measured cached / the prompts of those same
// requests (B4), never mixed with TTFT estimates. The key holder's card is the
// own lens (reuse of their OWN earlier prompts, no "misrouted").

import { type CacheSummary, engineBlockNote } from "@/lib/cache-obs";
import { FIXED, type MeCache, fmtRate, fmtTokens } from "@/lib/stats-redesign";

import { FixedCard, Label, Line } from "./primitives";

function Big({ children }: { children: React.ReactNode }) {
  return <div className="h-9 overflow-hidden whitespace-nowrap text-[28px] font-semibold leading-[1.1] tabular-nums">{children}</div>;
}

function Idle({ children }: { children: React.ReactNode }) {
  return <div className="h-9 overflow-hidden text-ellipsis whitespace-nowrap pt-2.5 text-[13px] text-chat-dim">{children}</div>;
}

function saved(s: number | null | undefined) {
  if (s == null) return <Idle>-</Idle>;
  return (
    <Big>
      {s >= 60 ? Math.round(s / 60) : Math.round(s)}
      <small className="ml-1 text-xs font-medium text-chat-muted">{s >= 60 ? "min" : "s"}</small>
    </Big>
  );
}

const OUTCOME_COLORS: Record<string, string> = {
  hit: "#60a5fa",
  partial: "#93c5fd88",
  lost: "rgb(248 113 113)",
  misrouted: "#fb923c",
  diverged: "#f472b6",
  cold: "rgb(154 140 116)",
  compacted: "#a78bfa",
  harness_rewrite: "#2dd4bf",
};

function OutcomeBar({ outcomes, own }: { outcomes: Record<string, number>; own: boolean }) {
  const entries = Object.entries(outcomes).filter(([k, n]) => n > 0 && !(own && k === "misrouted"));
  return (
    <>
      <div className="mt-1.5 flex h-1.5 gap-0.5 overflow-hidden rounded-sm bg-chat-surface-2">
        {entries.map(([k, n]) => (
          <span key={k} style={{ flex: n, background: OUTCOME_COLORS[k] ?? "gray" }} title={`${k} ${n}`} />
        ))}
      </div>
      <Line className="text-[11px] tabular-nums text-chat-dim">
        {entries.length ? entries.map(([k, n]) => `${k.replace("_", " ")} ${n}`).join(" · ") : "-"}
        {own ? " · own lens only" : ""}
      </Line>
    </>
  );
}

const GRID = "grid grid-cols-2 items-start gap-x-[18px] gap-y-1.5 md:grid-cols-[200px_1fr_150px_150px]";

export function AdminCacheCard({
  row,
  learnedRate,
  hintRate = null,
}: {
  row: CacheSummary | undefined;
  learnedRate: number | null;
  /** The configured rate, used (marked "est.") until a rate is learned. */
  hintRate?: number | null;
}) {
  const rate = learnedRate ?? hintRate;
  const est = learnedRate == null;
  const measuredCached = row?.measured_cached_tokens ?? 0;
  const measuredPrompt = row?.measured_prompt_tokens ?? 0;
  const n = row?.requests ?? 0;
  const m = row?.measured_requests ?? 0;
  const pct = measuredPrompt > 0 ? (100 * measuredCached) / measuredPrompt : null;
  const coverageLow = n > 0 && m / n < 0.8;
  const eff = row && row.reusable_tokens > 0 ? (100 * row.reused_tokens) / row.reusable_tokens : null;
  const block = engineBlockNote(row);
  return (
    <FixedCard height={FIXED.cacheCard} phoneHeight={FIXED.cacheCardPhone} testid="cache-card" className={GRID}>
      <div className="min-w-0">
        <Label>Cache hit</Label>
        {pct == null ? (
          <Idle>No measured requests in window</Idle>
        ) : (
          <Big>
            {Math.round(pct)}
            <small className="ml-1 text-xs font-medium text-chat-muted">%</small>
          </Big>
        )}
        <Line className={`text-[11px] ${coverageLow ? "text-chat-dim/60" : "text-chat-dim"}`}>
          measured on {m} of {n} requests
        </Line>
      </div>
      <div className="col-span-2 min-w-0 md:col-span-1">
        <Label>Cached vs computed prompt tokens · outcomes</Label>
        <div className="mt-1.5 flex h-2.5 gap-0.5 overflow-hidden rounded-sm bg-chat-surface-2">
          {pct != null && (
            <>
              <span style={{ width: `${pct}%`, background: "rgb(var(--vw-ttft))" }} />
              <span style={{ flex: 1, background: "rgb(var(--vw-prompt))" }} />
            </>
          )}
        </div>
        <Line className="text-xs tabular-nums text-chat-muted">
          {pct == null ? "-" : `${fmtTokens(measuredCached)} of ${fmtTokens(measuredPrompt)} prompt tokens from cache`}
        </Line>
        <OutcomeBar outcomes={row?.outcomes ?? {}} own={false} />
      </div>
      <div className="min-w-0">
        <Label>Prefill saved</Label>
        {saved(measuredCached > 0 && rate ? measuredCached / rate : null)}
        <Line className="text-[11px] text-chat-dim">
          {rate ? `at ${fmtRate(rate)} tok/s${est ? " (est., not learned yet)" : ""}` : " "}
        </Line>
      </div>
      <div className="min-w-0">
        <Label>Repeats served</Label>
        {eff == null ? (
          <Idle>-</Idle>
        ) : (
          <Big>
            {Math.round(eff)}
            <small className="ml-1 text-xs font-medium text-chat-muted">%</small>
          </Big>
        )}
        <Line
          className="text-[11px] text-chat-dim"
          title="Of the prompt tokens that matched an earlier prompt (any key), the share the engine actually served from its cache."
        >
          of text seen before, served from cache
        </Line>
        {block && (
          // Desktop only: on a phone the stacked columns have no room for a
          // fifth line inside the fixed height.
          <Line className="hidden text-[11px] text-chat-dim md:block" title={block.long}>
            {block.short}
          </Line>
        )}
      </div>
    </FixedCard>
  );
}

export function OwnCacheCard({ data }: { data: MeCache | undefined }) {
  const c = data?.cache;
  const pct = c && c.own_known_prompt_tokens > 0 ? (100 * c.own_reused_tokens) / c.own_known_prompt_tokens : null;
  return (
    <FixedCard height={FIXED.cacheCard} phoneHeight={FIXED.cacheCardPhone} testid="cache-card" className={GRID}>
      <div className="min-w-0">
        <Label>From your own earlier prompts</Label>
        {pct == null ? (
          <Idle>No measured requests in window</Idle>
        ) : (
          <Big>
            {Math.round(pct)}
            <small className="ml-1 text-xs font-medium text-chat-muted">%</small>
          </Big>
        )}
        <Line className="text-[11px] text-chat-dim">
          measured on {c?.own_known_requests ?? 0} of {c?.requests ?? 0} requests
        </Line>
      </div>
      <div className="col-span-2 min-w-0 md:col-span-1">
        <Label>Reused from your prompts vs computed · outcomes</Label>
        <div className="mt-1.5 flex h-2.5 gap-0.5 overflow-hidden rounded-sm bg-chat-surface-2">
          {pct != null && (
            <>
              <span style={{ width: `${pct}%`, background: "rgb(var(--vw-ttft))" }} />
              <span style={{ flex: 1, background: "rgb(var(--vw-prompt))" }} />
            </>
          )}
        </div>
        <Line className="text-xs tabular-nums text-chat-muted">
          {pct == null ? "-" : `${fmtTokens(c?.own_reused_tokens)} of ${fmtTokens(c?.own_known_prompt_tokens)} prompt tokens reused from your own prompts`}
        </Line>
        <OutcomeBar outcomes={c?.outcomes ?? {}} own />
      </div>
      <div className="min-w-0">
        <Label>Prefill saved</Label>
        {saved(c && c.own_reused_tokens > 0 ? c.prefill_saved_s : null)}
        <Line className="text-[11px] text-chat-dim">
          {c?.prefill_rate_tok_s ? `est. at the configured ${fmtRate(c.prefill_rate_tok_s)} tok/s` : " "}
        </Line>
      </div>
      <div className="min-w-0" />
    </FixedCard>
  );
}
