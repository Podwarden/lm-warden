"use client";

// Compact load-progress for the model card / models list (Task 8).
//
// Renders a one-line status pill, an overall bar, and the current step name —
// the brief's "compact form" for the cards.  Takes a ``LoadProgressOut |
// undefined`` so the card can render nothing when no load is active (the
// ``useActiveLoadProgress`` hook returns ``undefined`` before the first poll,
// or the model is not loading at all).
//
// No red anywhere: this component only renders while a load is in flight
// (the card only mounts it for ``status === "loading"``), and the failed
// state is the card's existing red ``Error:`` line, gated on
// ``status === "failed"``.

import type { LoadProgressOut } from "@/lib/load-progress";
import { formatDuration } from "@/lib/load-progress";

export interface LoadProgressCompactProps {
  progress: LoadProgressOut | undefined;
}

export function LoadProgressCompact({ progress }: LoadProgressCompactProps) {
  if (!progress) return null;

  // ``stalled`` and ``give_up`` both render amber; ``give_up`` says
  // "Giving up — stopping engine" instead of "no progress for Xm".
  // Red stays reserved for ``failed``, which the card owns separately.
  const stalled = progress.state === "stalled" || progress.state === "give_up";
  const giveUp = progress.state === "give_up";
  const steps = progress.steps ?? [];
  const activeStep = steps.find(
    (s) => s.status === "running" || s.status === "stalled",
  );
  // When no step is running/stalled, fall back to ``current_step``.  The
  // backend normally returns a label (e.g. "Start engine"), but if it ever
  // returns a raw key (e.g. "WEIGHTS"), map it through the steps' label set
  // — the single source of labels on the frontend — so the user never sees
  // a raw key.  If the value is neither a known key nor a known label, show
  // nothing rather than a confusing string.
  const labelByKey = new Map(steps.map((s) => [s.key, s.label]));
  const labelSet = new Set(steps.map((s) => s.label));
  const fallback = progress.current_step;
  let currentStepName: string | null;
  if (activeStep) {
    currentStepName = activeStep.label;
  } else if (fallback && labelByKey.has(fallback)) {
    currentStepName = labelByKey.get(fallback)!;
  } else if (fallback && labelSet.has(fallback)) {
    currentStepName = fallback;
  } else {
    currentStepName = null;
  }

  // Overall bar only when usual-total history exists.
  const showOverall =
    progress.usual_total_s != null && progress.usual_total_s > 0;
  const overallPct = showOverall
    ? Math.min(
        95,
        Math.max(0, Math.round((progress.elapsed_s / progress.usual_total_s!) * 100)),
      )
    : null;

  return (
    <div className="space-y-1" data-testid="load-progress-compact">
      <div className="flex items-center gap-2 text-xs">
        <span
          className={
            stalled
              ? "inline-flex h-2 w-2 animate-pulse rounded-full bg-amber-400"
              : "inline-flex h-2 w-2 animate-pulse rounded-full bg-emerald-400"
          }
          aria-hidden
        />
        <span className={stalled ? "text-amber-300" : "text-emerald-300"}>
          {giveUp
            ? "Giving up — stopping engine"
            : stalled
              ? `Loading · no progress for ${Math.max(1, Math.round(progress.stalled_for_s / 60))}m`
              : "Loading"}
        </span>
        {currentStepName && (
          <span className="truncate text-slate-400">{currentStepName}</span>
        )}
      </div>
      {overallPct !== null && (
        <div
          className="h-1 w-full overflow-hidden rounded bg-slate-700"
          role="progressbar"
          aria-valuenow={overallPct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Overall load progress"
          data-testid="compact-overall-bar"
        >
          <div
            className="h-full bg-emerald-500 opacity-70 transition-[width]"
            style={{ width: `${overallPct}%` }}
          />
        </div>
      )}
      {/* Elapsed always shows so the card conveys "it is doing something"
          even without history. */}
      <div className="text-xs text-slate-500">
        {formatDuration(progress.elapsed_s)} elapsed
      </div>
    </div>
  );
}
