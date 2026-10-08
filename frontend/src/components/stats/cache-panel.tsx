"use client";

// Prompt-cache observation per model (spec 2026-10-07 §5.1/§5.2). One component
// for both lenses: the operator's Stats panel and the token page's card.

import {
  coveredRequests, efficiency, fmtPct, fromCache, isEstimated, OUTCOMES, outcomeLabel, prefillSaved,
  topProblem, type CacheResponse, type CacheSummary,
} from "@/lib/cache-obs";

const HINT_RATE_TITLE =
  "assumed prefill rate (VW_PREFILL_TOK_S_HINT); the warden has not learned this box's rate yet";

const OUTCOME_CLASS: Record<string, string> = {
  hit: "bg-sky-500", partial: "bg-sky-300", lost: "bg-rose-500",
  misrouted: "bg-amber-500", diverged: "bg-violet-500", cold: "bg-slate-400",
};

function fmtDuration(s: number | null, estimated = false): string {
  if (s === null) return "—";
  const mark = estimated ? "≈" : "";
  return mark + fmtDur(s);
}

function fmtDur(s: number): string {
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function Row({ s, lens }: { s: CacheSummary; lens: "operator" | "own" }) {
  const est = isEstimated(s);
  const title = est
    ? `${s.backend ?? "This engine"} does not report cached tokens; estimated from TTFT`
    : undefined;
  const cls = est ? "underline decoration-dotted" : "";
  // Own lens never shows "other replica": that outcome is derived from
  // prefixes cached by other keys' traffic.
  const shown = OUTCOMES.filter((o) => lens === "operator" || o !== "misrouted");
  const total = shown.reduce((a, o) => a + s.outcomes[o], 0);
  const hint = topProblem(s, { lens });
  // Prefill saved = C / prefill rate; an unlearned (hint) rate is an assumption.
  const hintRate = s.rate_source === "hint";
  const savedTitle = [title, hintRate ? HINT_RATE_TITLE : undefined].filter(Boolean).join("; ") || undefined;
  const savedCls = est || hintRate ? "underline decoration-dotted" : "";
  const covered = coveredRequests(s);
  return (
    <div className="border-t border-chat-rule pt-3 first:border-t-0 first:pt-0" data-testid="cache-row">
      <div className="font-mono text-xs text-chat-fg">{s.model}</div>
      <div className="mt-2 grid grid-cols-3 gap-3 text-sm">
        <div><div className="text-[11px] text-chat-dim">from cache</div>
          <span data-testid="cache-from" title={title} className={`font-mono ${cls}`}>{fmtPct(fromCache(s), est)}</span></div>
        <div><div className="text-[11px] text-chat-dim">efficiency</div>
          <span data-testid="cache-efficiency" title={title ?? "Of the prompt tokens that could have been reused, the share that was."} className={`font-mono ${cls}`}>{fmtPct(efficiency(s), est)}</span></div>
        <div><div className="text-[11px] text-chat-dim">prefill saved</div>
          <span data-testid="cache-saved" title={savedTitle} className={`font-mono ${savedCls}`}>{fmtDuration(prefillSaved(s), est || hintRate)}</span></div>
      </div>
      {covered < s.requests && (
        <p data-testid="cache-coverage" className="mt-1 text-[11px] text-chat-dim">
          cache figures from {covered} of {s.requests} requests
        </p>
      )}
      {total > 0 && (
        <div className="mt-2 flex h-2 w-full overflow-hidden rounded" role="img"
             aria-label={shown.map((o) => `${outcomeLabel(o)} ${s.outcomes[o]}`).join(", ")}>
          {shown.map((o) => s.outcomes[o] > 0 && (
            <div key={o} className={OUTCOME_CLASS[o]} style={{ width: `${(s.outcomes[o] / total) * 100}%` }}
                 title={`${outcomeLabel(o)}: ${s.outcomes[o]}`} />
          ))}
        </div>
      )}
      {total > 0 && (
        <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-chat-dim" data-testid="cache-legend">
          {shown.map((o) => s.outcomes[o] > 0 && (
            <li key={o} className="flex items-center gap-1">
              <span className={`inline-block h-2 w-2 rounded-full ${OUTCOME_CLASS[o]}`} aria-hidden="true" />
              {outcomeLabel(o)} {s.outcomes[o]}
            </li>
          ))}
        </ul>
      )}
      {(s.deep_diverged ?? 0) > 0 && (
        <p data-testid="cache-deep-diverged" className="mt-1 text-[11px] text-chat-dim"
           title="The client rewrote a message after a long unchanged start: the start was reused, everything after the break was recomputed.">
          {s.deep_diverged} hit or partial request{s.deep_diverged === 1 ? "" : "s"} broke their prefix mid-prompt
        </p>
      )}
      {hint && <p data-testid="cache-hint" className="mt-2 text-xs text-chat-muted">{hint.text}</p>}
    </div>
  );
}

export function CachePanel({ data, lens = "operator" }: { data: CacheResponse | null; lens?: "operator" | "own" }) {
  if (!data) return null;
  if (data.models.length === 0)
    return <p className="text-xs text-chat-dim">No requests in this window.</p>;
  return <div className="space-y-3">{data.models.map((s) => <Row key={s.model_id} s={s} lens={lens} />)}</div>;
}
