"use client";

// The load-progress checklist with step bars and vital signs (Task 8).
//
// Consumes a ``LoadProgressOut`` snapshot (Task 6) and renders the CI-job-style
// checklist the owner approved: a status pill, an overall bar (only when
// usual-total history exists), the steps with per-step progress bars, and
// the vital-signs tiles under the running step.
//
// Visual rules (from the spec and the brief — verbatim values):
//   - Green pulsing pill ``Loading · step k of N · alive`` while progressing.
//   - Amber pulsing ``Loading · no progress for Xm`` with give-up time and
//     ``Force unload now`` / ``View log`` when stalled.
//   - Overall bar only when ``usual_total_s`` is known; ``elapsed of usual
//     ~T`` and ``ready ≈ HH:MM`` with it.
//   - Exact step bars when ``count`` and ``total`` are present (weights,
//     graphs).  Usual-time bars are dimmed and capped at 95 % until ``done``.
//   - Steps the engine never reports (llama.cpp has no compile/graph step)
//     render gracefully — the snapshot simply omits them.
//   - No red anywhere unless the state is failed.  This component does not
//     render the failed state at all; the page's red ``last_error`` box owns
//     that, gated on ``status === "failed"``.
//
// ``onForceUnload`` and ``onViewLog`` are wired by the page so the pill's
// actions reach the ForceUnloadModal and the Live logs card respectively.

import type { LoadProgressOut, StepOut } from "@/lib/load-progress";
import { formatDuration } from "@/lib/load-progress";
import { LoadStatusPill } from "./load-status-pill";
import { VitalsTiles } from "./vitals-tiles";

export interface LoadChecklistProps {
  progress: LoadProgressOut;
  /** Open the force-unload confirmation.  Optional — the pill degrades. */
  onForceUnload?: () => void;
  /** Scroll to / open the Live logs card.  Optional. */
  onViewLog?: () => void;
}

function formatTime(iso: string | null): string | null {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    return d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return null;
  }
}

/** Step status icon: checkmark for done, spinner glyph for running, etc. */
function stepIcon(status: StepOut["status"]): string {
  switch (status) {
    case "done":
      return "✓";
    case "running":
      return "▸";
    case "stalled":
      return "⏸";
    default:
      return "○";
  }
}

/** Compute the width % for a step's exact (count/total) bar. */
function exactBarPct(step: StepOut): number | null {
  if (step.count == null || step.total == null || step.total <= 0) return null;
  return Math.min(100, Math.round((step.count / step.total) * 100));
}

/** Compute the width % for a usual-time bar, capped at 95 % until done. */
function usualBarPct(step: StepOut): number | null {
  if (step.usual_s == null || step.usual_s <= 0) return null;
  if (step.status === "done") return 100;
  const elapsed = step.elapsed_s ?? 0;
  const pct = Math.round((elapsed / step.usual_s) * 100);
  // Capped at 95 % until the engine says the step ended — no fake precision.
  return Math.min(95, Math.max(0, pct));
}

/** The overall bar: elapsed of usual total, capped at 95 % while loading. */
function overallBarPct(progress: LoadProgressOut): number | null {
  if (progress.usual_total_s == null || progress.usual_total_s <= 0) return null;
  const pct = Math.round((progress.elapsed_s / progress.usual_total_s) * 100);
  return Math.min(95, Math.max(0, pct));
}

function StepRow({ step }: { step: StepOut }) {
  const exact = exactBarPct(step);
  const usual = usualBarPct(step);
  const hasExact = exact !== null;
  const hasUsual = usual !== null;
  // No bar at all when neither exact nor usual is available: show elapsed only.
  const showBar = hasExact || hasUsual;

  const iconColour =
    step.status === "done"
      ? "text-emerald-400"
      : step.status === "running"
        ? "text-blue-400"
        : step.status === "stalled"
          ? "text-amber-400"
          : "text-slate-500";

  const labelExtra: string[] = [];
  if (step.status === "done" && step.duration_s != null) {
    labelExtra.push(formatDuration(step.duration_s));
  }
  if (step.status === "running" && step.elapsed_s != null) {
    labelExtra.push(formatDuration(step.elapsed_s));
    if (step.usual_s != null) {
      labelExtra.push(`usually ~${formatDuration(step.usual_s)}`);
    }
  }
  if (step.status === "stalled" && step.elapsed_s != null) {
    labelExtra.push(formatDuration(step.elapsed_s));
  }
  // TUNE detail: "<op name> · op N" from the backend, then the op's counter
  // (e.g. "trtllm_fp4_block_scale_moe · op 2 · 6/22").
  if (step.detail) {
    labelExtra.push(step.detail);
  }
  if (step.count != null && step.total != null) {
    labelExtra.push(`${step.count}/${step.total}`);
  }
  if (step.status === "pending" && step.usual_s != null) {
    labelExtra.push(`usually ~${formatDuration(step.usual_s)}`);
  }

  // Bar colour: exact bars are blue (live), usual-time bars are dimmed.
  const barColour = hasExact
    ? "bg-blue-500"
    : step.status === "done"
      ? "bg-emerald-500"
      : "bg-slate-500";
  const barOpacity = hasExact || step.status === "done" ? "" : "opacity-60";

  return (
    <div
      className="space-y-1"
      data-testid={`step-row-${step.key}`}
      data-step-status={step.status}
    >
      <div className="flex items-center gap-2 text-sm">
        <span className={`font-mono ${iconColour}`} aria-hidden>
          {stepIcon(step.status)}
        </span>
        <span className="flex-1 truncate">
          {step.label}
          {labelExtra.length > 0 && (
            <span className="ml-2 text-xs text-slate-500">
              {labelExtra.join(" · ")}
            </span>
          )}
        </span>
      </div>
      {showBar && (
        <div
          className="h-1.5 w-full overflow-hidden rounded bg-slate-700"
          role="progressbar"
          aria-valuenow={hasExact ? exact! : usual!}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={step.label}
          data-testid={`step-bar-${step.key}`}
        >
          <div
            className={`h-full transition-[width] ${barColour} ${barOpacity}`}
            style={{ width: `${hasExact ? exact! : usual!}%` }}
          />
        </div>
      )}
    </div>
  );
}

export function LoadChecklist({
  progress,
  onForceUnload,
  onViewLog,
}: LoadChecklistProps) {
  // Skipped steps (e.g. TUNE on a model with no AutoTuner) are not rendered.
  const allSteps = progress.steps ?? [];
  const steps = allSteps.filter((s) => s.status !== "skipped");
  const totalSteps = steps.length;
  // 1-based index of the running/stalled step for "step k of N".
  const activeIdx = steps.findIndex(
    (s) => s.status === "running" || s.status === "stalled",
  );
  const currentStepIndex = activeIdx >= 0 ? activeIdx + 1 : totalSteps;

  // ``stalled`` and ``give_up`` both render vitals greyed and the hint
  // block; the pill itself distinguishes them (``give_up`` says "Giving
  // up — stopping engine" rather than "no progress for Xm").  Red stays
  // reserved for ``failed``, which this component does not render.
  const stalled = progress.state === "stalled" || progress.state === "give_up";
  const overall = overallBarPct(progress);
  const readyTime = formatTime(progress.ready_eta ?? null);

  return (
    <div
      className="space-y-4 rounded-md border border-slate-700 bg-slate-900/30 p-4"
      data-testid="load-checklist"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <LoadStatusPill
          state={progress.state}
          currentStepIndex={currentStepIndex}
          totalSteps={totalSteps}
          stalledForS={progress.stalled_for_s}
          giveUpAt={progress.give_up_at ?? null}
          onForceUnload={onForceUnload}
          onViewLog={onViewLog}
        />
        {progress.layout && (
          <span className="text-xs text-slate-500">{progress.layout}</span>
        )}
      </div>

      {/* Overall bar — only when usual-total history exists. */}
      {overall !== null && (
        <div className="space-y-1" data-testid="overall-bar">
          <div className="flex justify-between text-xs text-slate-400">
            <span>
              {formatDuration(progress.elapsed_s)} of usual ~
              {formatDuration(progress.usual_total_s!)}
            </span>
            {readyTime && (
              <span data-testid="ready-eta">ready ≈ {readyTime}</span>
            )}
          </div>
          <div
            className="h-2 w-full overflow-hidden rounded bg-slate-700"
            role="progressbar"
            aria-valuenow={overall}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Overall load progress"
          >
            <div
              className="h-full bg-emerald-500 opacity-70 transition-[width]"
              style={{ width: `${overall}%` }}
            />
          </div>
        </div>
      )}

      {/* Step rows in order, with vitals under the running step. */}
      <div className="space-y-3">
        {steps.map((step, i) => (
          <div key={step.key} className="space-y-2">
            <StepRow step={step} />
            {/* Vitals render under the running/stalled step only. */}
            {(step.status === "running" || step.status === "stalled") && (
              <VitalsTiles vitals={progress.vitals} stalled={stalled} />
            )}
            {/* Separator between steps (not after the last). */}
            {i < steps.length - 1 && (
              <div className="h-px bg-slate-800" />
            )}
          </div>
        ))}
      </div>

      {/* Stalled hint/reason from the snapshot, if any. */}
      {stalled && (progress.reason || progress.hint) && (
        <div
          className="rounded-md border border-amber-800/40 bg-amber-950/40 p-3 text-xs text-amber-300"
          data-testid="stalled-hint"
        >
          {progress.reason && (
            <p className="font-semibold">{progress.reason}</p>
          )}
          {progress.hint && <p className="mt-1">{progress.hint}</p>}
        </div>
      )}
    </div>
  );
}
