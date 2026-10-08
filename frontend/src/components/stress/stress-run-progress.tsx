"use client";

/**
 * Running phase of the stress-test modal: a progress bar plus the per-probe
 * attempt list.
 *
 * Progress comes from POLLING the run row, not SSE — design §9. bench-v2's
 * SSE replay with a binary offset index is the single largest piece of
 * machinery that design refuses to bring back, and the run row already carries
 * everything a progress display needs at a 2s cadence.
 *
 * The bar copies the indeterminate-ARIA handling from
 * `@/components/models/pull-progress` (`:65-84`): when no total is known,
 * `aria-valuenow` is OMITTED entirely so a screen reader announces "busy"
 * rather than a misleading 0%. It is copied rather than imported because
 * PullProgress is a pull-specific component — it returns null unless
 * `status` is "pulling"/"registered" and formats its numbers as bytes, so
 * feeding it probe counts would render "12 B / 40 B".
 *
 * The attempt list follows `try-stack-panel.tsx`: one row per attempt, the
 * outcome as a `<Badge variant={...}>` on the right.
 */

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  badgeVariantForOutcome,
  normaliseObservations,
  type StressRun,
} from "./types";

/**
 * Seconds as a coarse duration. Deliberately coarse: the estimate is a phase's
 * own pace extrapolated over its remainder, which is worth a minute's
 * precision at most. "~7 min" is honest; "7 m 12 s" implies a confidence the
 * number does not have.
 */
function formatEta(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const mins = Math.round(seconds / 60);
  if (mins < 90) return `${mins} min`;
  return `${(mins / 60).toFixed(1)} h`;
}

function ProbeBar({
  completed,
  total,
}: {
  completed: number | null;
  total: number | null;
}) {
  const hasTotal = total !== null && total > 0;
  const pct = hasTotal
    ? Math.min(100, Math.round(((completed ?? 0) / total) * 100))
    : null;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-400">
        <span data-testid="stress-progress-count">
          {completed ?? 0}
          {hasTotal ? ` / ${total}` : ""} probe(s)
        </span>
        {pct !== null && <span>{pct}%</span>}
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded bg-slate-700"
        role="progressbar"
        {...(pct !== null
          ? { "aria-valuenow": pct, "aria-valuemin": 0, "aria-valuemax": 100 }
          : {})}
        aria-label="Stress test progress"
      >
        {/* Indeterminate is a NARROW pulsing segment, never a full bar. A
            full green bar beside "0 probe(s)" reads as "finished", which is
            the worst of the three possible wrong answers — reported from a
            live run on 2026-09-04. */}
        <div
          className={cn(
            "h-full bg-emerald-500 transition-[width]",
            pct === null && "animate-pulse",
          )}
          style={pct !== null ? { width: `${pct}%` } : { width: "20%" }}
        />
      </div>
    </div>
  );
}

export function StressRunProgress({
  run,
  pollError,
}: {
  /** null until the first poll lands — the POST returns only a run_id. */
  run: StressRun | null;
  /** Last polling error, if any. Shown, but not fatal: see below. */
  pollError?: string | null;
}) {
  const observations = normaliseObservations(run?.observations);
  const progress = run?.progress ?? null;
  const completed = progress?.completed ?? (observations.length || null);

  return (
    <div className="space-y-4 text-sm" data-testid="stress-running">
      <p className="text-slate-300">
        Running. The engine may crash and be restarted while this runs; requests
        served by this model will fail until it recovers.
      </p>

      <ProbeBar completed={completed} total={progress?.total ?? null} />

      {progress?.phase && (
        <p className="text-xs text-slate-400" data-testid="stress-phase">
          Phase: <span className="font-mono text-slate-300">{progress.phase}</span>
          {progress.note ? ` — ${progress.note}` : ""}
          {typeof progress.eta_s === "number" && progress.eta_s > 0 ? (
            <span className="ml-1 text-slate-500" data-testid="stress-eta">
              (~{formatEta(progress.eta_s)} left in this phase)
            </span>
          ) : null}
        </p>
      )}

      {/* A poll failing is expected mid-run: the warden restarts the engine
          after a crash and the API can blip. Surface it, keep polling, and do
          NOT tear the modal down — the run continues server-side regardless of
          what this browser can see. */}
      {pollError && (
        <p className="text-xs text-amber-300" data-testid="stress-poll-error">
          Progress update failed ({pollError}). Still polling — the run
          continues on the server.
        </p>
      )}

      {observations.length === 0 ? (
        <p className="text-xs text-slate-500" data-testid="stress-no-probes">
          No probes recorded yet.
        </p>
      ) : (
        <div className="max-h-64 space-y-1 overflow-y-auto rounded-md border border-slate-700 p-2 text-xs">
          {observations.map((o, i) => (
            <div
              key={`${o.axis ?? "probe"}-${o.value ?? i}-${i}`}
              className="flex items-center justify-between gap-2 border-b border-slate-800 pb-1 last:border-b-0"
              data-testid="stress-probe-row"
            >
              <span className="min-w-0 truncate text-slate-300">
                <span className="font-mono text-slate-100">
                  {o.value ?? "—"}
                </span>
                {o.axis ? (
                  <span className="ml-1 text-slate-500">
                    {o.axis === "concurrency" ? "concurrent" : "tokens"}
                  </span>
                ) : null}
                {o.gate_tripped ? (
                  <span className="ml-2 text-amber-300">
                    gate: {o.gate_tripped}
                  </span>
                ) : null}
              </span>
              <Badge variant={badgeVariantForOutcome(o.cls)}>{o.cls}</Badge>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
