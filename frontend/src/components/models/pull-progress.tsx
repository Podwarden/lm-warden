"use client";

import { cn } from "@/lib/utils";

interface PullProgressProps {
  status: string;
  pulledBytes: number | null;
  pulledTotal: number | null;
  className?: string;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

/**
 * Presentational pull-progress bar for the model-detail page.
 *
 * Intentionally NOT an SSE consumer: the parent page already polls
 * `/api/models/{id}` every 2s via SWR, and the row contains `status`,
 * `pulled_bytes`, and `pulled_total`. Duplicating that as a second SSE
 * stream (the plan's first draft did) would mint a separate ticket,
 * race against the SWR poll, and double-up reconnect logic — for no UX
 * gain at 2s cadence.
 *
 * Visible for statuses where progress is meaningful (`pulling`,
 * `registered` — the pre-pull placeholder). Returns null otherwise so
 * the detail page can render it unconditionally and let this component
 * decide whether to occupy space.
 */
export function PullProgress({
  status,
  pulledBytes,
  pulledTotal,
  className,
}: PullProgressProps) {
  if (status !== "pulling" && status !== "registered") return null;

  const bytes = pulledBytes ?? 0;
  const hasTotal = pulledTotal !== null && pulledTotal > 0;
  // Clamp to 100 so a backend over-report (e.g., a hub size estimate
  // smaller than the actual download) renders as "done" rather than a
  // bar overflowing its track.
  const pct = hasTotal
    ? Math.min(100, Math.round((bytes / (pulledTotal as number)) * 100))
    : null;

  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex justify-between text-xs text-slate-400">
        <span>
          {formatBytes(bytes)}
          {hasTotal ? ` / ${formatBytes(pulledTotal as number)}` : ""}
        </span>
        {pct !== null && <span>{pct}%</span>}
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded bg-slate-700"
        role="progressbar"
        // Indeterminate ARIA pattern: omit aria-valuenow entirely so
        // screen readers announce "busy" rather than a misleading 0%.
        // Once a total is known, pin valuenow/min/max so the percent is
        // discoverable without sighted access to the bar.
        {...(pct !== null
          ? { "aria-valuenow": pct, "aria-valuemin": 0, "aria-valuemax": 100 }
          : {})}
        aria-label="Pull progress"
      >
        <div
          className={cn(
            "h-full bg-emerald-500 transition-[width]",
            pct === null && "animate-pulse",
          )}
          style={pct !== null ? { width: `${pct}%` } : { width: "100%" }}
        />
      </div>
    </div>
  );
}
