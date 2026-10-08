"use client";

// Rail "Traffic" card (plan §4.7): the four-way split since the process
// started, as a stacked bar (role="img" with the full numbers) and a legend
// holding the same numbers as text; then "Why it fell back or was refused" when there are
// reasons. No traffic is one empty state, never four zeros.

import { cn } from "@/lib/utils";
import {
  fmtClock,
  fmtInt,
  fmtPct,
  trafficSplit,
  type RouterStatsOut,
  type TrafficKey,
} from "@/lib/router";
import { CARD, TONE_BAR, type Tone } from "./styles";

const KEY_TONE: Record<TrafficKey, Tone> = {
  local: "ok",
  passthrough: "info",
  fallback: "amber",
  refused: "danger",
};

const ARIA_WORD: Record<TrafficKey, string> = {
  local: "Local",
  passthrough: "Anthropic",
  fallback: "fell back",
  refused: "refused or error",
};

interface Props {
  stats: RouterStatsOut | undefined;
  /** An incident is on: a non-zero "Fell back" row is drawn hot. */
  alarm: boolean;
  className?: string;
}

export function TrafficPanel({ stats, alarm, className }: Props) {
  const split = stats ? trafficSplit(stats.totals) : null;
  const reasons = stats
    ? Object.entries(stats.by_reason)
        .filter(([, n]) => n > 0)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
    : [];

  let content: React.ReactNode;
  if (!stats || !split) {
    content = (
      <div aria-hidden="true" className="mt-3 h-3 animate-pulse rounded-full bg-chat-surface-2 motion-reduce:animate-none" />
    );
  } else if (stats.since === null) {
    content = (
      <p className="mb-0 mt-1 text-[13px] text-chat-muted">
        <b className="font-semibold text-chat-fg">No requests yet.</b> Counters start with the first
        request after routing is on.
      </p>
    );
  } else if (split.total === 0) {
    content = (
      <p className="mb-0 mt-1 text-[13px] text-chat-muted">
        <b className="font-semibold text-chat-fg">No requests since reset</b> ({fmtClock(stats.since)}).
      </p>
    );
  } else {
    const label =
      split.segments.map((s) => `${ARIA_WORD[s.key]} ${fmtInt(s.n)}`).join(", ") + ` (of ${fmtInt(split.total)})`;
    content = (
      <>
        <div role="img" aria-label={label} className="flex h-3 gap-0.5 overflow-hidden rounded-full bg-chat-surface-2">
          {split.segments
            .filter((s) => s.n > 0)
            .map((s) => (
              <i
                key={s.key}
                className={cn("block h-full min-w-[3px]", TONE_BAR[KEY_TONE[s.key]])}
                style={{ width: `${s.pct}%` }}
              />
            ))}
        </div>
        <ul className="m-0 mt-3 grid list-none gap-1.5 p-0">
          {split.segments.map((s) => {
            const hot = alarm && s.key === "fallback" && s.n > 0;
            return (
              <li
                key={s.key}
                className="grid grid-cols-[10px_minmax(0,1fr)_auto_44px] items-center gap-2 text-[13px]"
              >
                <span aria-hidden="true" className={cn("h-2.5 w-2.5 rounded-[3px]", TONE_BAR[KEY_TONE[s.key]])} />
                <span className={cn(hot && "text-vw-danger-fg")}>{s.label}</span>
                <span className={cn("text-right font-semibold tabular-nums", hot && "text-vw-danger-fg")}>
                  {fmtInt(s.n)}
                </span>
                <span className="text-right text-[12px] tabular-nums text-chat-muted">{fmtPct(s.pct)}</span>
              </li>
            );
          })}
        </ul>
      </>
    );
  }

  return (
    <section aria-labelledby="router-traffic-h" data-testid="router-traffic" className={cn(CARD, className)}>
      <h2 id="router-traffic-h" className="m-0 text-[15px] font-semibold">
        Traffic
      </h2>
      {stats?.since && (
        <p className="mb-3 mt-0.5 text-[12px] text-chat-muted">
          Since {fmtClock(stats.since)} · this process (a restart zeroes it)
        </p>
      )}
      {content}
      {split && split.total > 0 && reasons.length > 0 && (
        <div className="mt-3 border-t border-vw-rule-soft/45 pt-2.5 text-[12.5px]">
          <h3 className="m-0 text-[12.5px] font-semibold">Why it fell back or was refused</h3>
          <dl className="m-0 mt-1.5 grid gap-1">
            {reasons.map(([reason, n]) => (
              <div key={reason} className="flex justify-between gap-2">
                <dt className="min-w-0 font-mono [overflow-wrap:anywhere]">{reason}</dt>
                <dd className="m-0 tabular-nums">{fmtInt(n)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </section>
  );
}
