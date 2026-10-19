"use client";

// The status pill at the top of the load-progress checklist.
//
// Three variants while loading (no red — the anti-goal):
//   - progressing: green pulsing "Loading · step k of N · alive"
//   - stalled:     amber pulsing "Loading · no progress for Xm" with the
//                  give-up time and Force-unload / View-log actions.
//   - give_up:     amber pulsing "Giving up — stopping engine" — the cap
//                  policy is killing the engine.  Amber, never green or
//                  red; red stays reserved for ``failed``.
//
// The brief's colour rule: no red unless the state is failed.  Stalled and
// give_up are warning colours (amber), not red.  The pill itself does not
// render the failed state — that is the page's red ``last_error`` box,
// gated on ``status === "failed"`` — so this component only knows the
// three loading variants and is intentionally narrow.
//
// "View log" scrolls to the Live logs card.  The page passes a ref/id so
// the pill does not need to know the DOM structure.

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export interface LoadStatusPillProps {
  /** Load state from the snapshot: ``progressing``, ``stalled``, or
   *  ``give_up``.  ``give_up`` is amber, never green or red. */
  state: "progressing" | "stalled" | "give_up";
  /** 1-based index of the current step, for "step k of N". */
  currentStepIndex: number;
  /** Total steps in the checklist (e.g. 6 for vLLM, 3 for llama.cpp). */
  totalSteps: number;
  /** Seconds the load has been stalled, for "no progress for Xm". */
  stalledForS: number;
  /** Wall-clock give-up time, for the amber variant's deadline. */
  giveUpAt: string | null;
  /** Click handler for the Force unload action. */
  onForceUnload?: () => void;
  /** Click handler for the View log action (scrolls to Live logs). */
  onViewLog?: () => void;
}

function formatMinutes(s: number): string {
  const m = Math.max(1, Math.round(s / 60));
  return `${m}m`;
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

export function LoadStatusPill({
  state,
  currentStepIndex,
  totalSteps,
  stalledForS,
  giveUpAt,
  onForceUnload,
  onViewLog,
}: LoadStatusPillProps) {
  if (state === "give_up") {
    // The cap policy is killing the engine.  Amber, never green or red;
    // red stays reserved for ``failed``.  The engine is being stopped, so
    // there is no give-up deadline or Force-unload action to offer — the
    // warden is already doing it.
    return (
      <div className="flex flex-wrap items-center gap-2">
        <Badge
          variant="warning"
          className="animate-pulse"
          data-testid="load-status-pill"
          data-state="give_up"
        >
          Giving up — stopping engine
        </Badge>
        {onViewLog && (
          <Button
            size="sm"
            variant="outline"
            onClick={onViewLog}
            data-testid="pill-view-log"
          >
            View log
          </Button>
        )}
      </div>
    );
  }

  if (state === "stalled") {
    const giveUp = formatTime(giveUpAt);
    return (
      <div className="flex flex-wrap items-center gap-2">
        <Badge
          variant="warning"
          className="animate-pulse"
          data-testid="load-status-pill"
          data-state="stalled"
        >
          Loading · no progress for {formatMinutes(stalledForS)}
        </Badge>
        {giveUp && (
          <span className="text-xs text-amber-300">
            give up at {giveUp}
          </span>
        )}
        {onForceUnload && (
          <Button
            size="sm"
            variant="outline"
            onClick={onForceUnload}
            data-testid="pill-force-unload"
            className="border-amber-500/30 text-amber-300 hover:bg-amber-950/40"
          >
            Force unload now
          </Button>
        )}
        {onViewLog && (
          <Button
            size="sm"
            variant="outline"
            onClick={onViewLog}
            data-testid="pill-view-log"
          >
            View log
          </Button>
        )}
      </div>
    );
  }

  return (
    <Badge
      variant="success"
      className="animate-pulse"
      data-testid="load-status-pill"
      data-state="progressing"
    >
      Loading · step {currentStepIndex} of {totalSteps} · alive
    </Badge>
  );
}
