"use client";

// Collapsible Live logs card (Task 8).
//
// Wraps the existing ``LogStream`` so the live logs are collapsed by default
// in every state (loading, loaded, failed).  Collapsed, it is one slim bar:
// ``Live logs · last line <age> ago: "<last log line, truncated>"`` and a
// ``Show log`` button that opens the full streaming log.
//
// It opens by itself when a load turns amber (stalled) or fails.  The
// open/closed choice is remembered per browser in ``localStorage`` under
// ``lmw.liveLogs.open`` (try/catch guarded — a private window or blocked
// site data must not break the page).  A manual toggle wins over the
// automatic opening until the next state change.
//
// "State change" means the load-progress state (``progressing`` →
// ``stalled`` → ``give_up``) OR the model row status (``loading`` →
// ``failed``).  We track the last state we auto-opened for and only
// auto-open again when it differs — so once you close it, it stays closed
// through subsequent polls of the same state.

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { LogStream } from "@/components/models/log-stream";

export interface LiveLogsCollapsibleHandle {
  /** Open the card.  Counts as a manual open: records the current state so the
   *  auto-open effect does not immediately re-close it.  Safe to call when the
   *  card is already open (no-op). */
  open: () => void;
}

export interface LiveLogsCollapsibleProps {
  modelId: string;
  /** Model row status — used to detect failed → auto-open. */
  status: string;
  /**
   * Load-progress state while loading (``progressing`` / ``stalled`` /
   * ``give_up``), or ``null`` when the model is not loading.  Used to
   * auto-open on stalled and on give-up.
   */
  loadState?: string | null;
  /** Last log line for the collapsed preview. */
  lastLogLine?: string | null;
  /** Age of the last log line in seconds, for the collapsed preview. */
  lastLogAgeS?: number | null;
}

const STORAGE_KEY = "lmw.liveLogs.open";

function readStoredOpen(): boolean {
  // localStorage is wrapped in try/catch: a private window, blocked site
  // data, or a quota error must not break the page.  Default closed.
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "true") return true;
    if (v === "false") return false;
  } catch {
    /* ignore — localStorage unavailable */
  }
  return false;
}

function writeStoredOpen(open: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, open ? "true" : "false");
  } catch {
    /* ignore — localStorage unavailable */
  }
}

function formatAge(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return r === 0 ? `${m}m` : `${m}m${r}s`;
}

function truncate(line: string, max = 120): string {
  if (line.length <= max) return line;
  return line.slice(0, max - 1) + "…";
}

export const LiveLogsCollapsible = forwardRef<LiveLogsCollapsibleHandle, LiveLogsCollapsibleProps>(
  function LiveLogsCollapsible({
    modelId,
    status,
    loadState,
    lastLogLine,
    lastLogAgeS,
  }, ref) {
  const [open, setOpen] = useState<boolean>(readStoredOpen);
  // Track the last load-state we auto-opened for, so a manual close wins
  // until the state changes again.  Initialised to a sentinel that differs
  // from any real state, so the FIRST render with a stalled/failed state
  // auto-opens (the user hasn't made a manual choice yet).  After a manual
  // toggle, the ref is set to the current state, so re-rendering the same
  // state does not re-open.
  const lastAutoState = useRef<string | null>("__init__");
  const lastAutoStatus = useRef<string>("__init__");

  const shouldAutoOpen =
    loadState === "stalled" ||
    loadState === "give_up" ||
    status === "failed";

  // Auto-open on state change, but a manual toggle wins until the state
  // changes again.  We only auto-open when the state is NEW (different from
  // the one we last acted on).
  useEffect(() => {
    const stateKey = `${loadState ?? ""}:${status}`;
    const lastKey = `${lastAutoState.current ?? ""}:${lastAutoStatus.current}`;
    if (stateKey !== lastKey && shouldAutoOpen) {
      setOpen(true);
      writeStoredOpen(true);
      lastAutoState.current = loadState ?? null;
      lastAutoStatus.current = status;
    } else if (stateKey !== lastKey) {
      // State changed but not to an auto-open state — record it so we
      // don't re-evaluate and re-open on every render.
      lastAutoState.current = loadState ?? null;
      lastAutoStatus.current = status;
    }
  }, [loadState, status, shouldAutoOpen]);

  const toggle = useCallback(() => {
    setOpen((prev) => {
      const next = !prev;
      writeStoredOpen(next);
      // A manual toggle is the user's choice — record the current state so
      // the auto-open effect does not immediately re-open.
      lastAutoState.current = loadState ?? null;
      lastAutoStatus.current = status;
      return next;
    });
  }, [loadState, status]);

  // Imperative open: called by the page's "View log" action so the pill can
  // both scroll to the card AND open it in one click.  This counts as a manual
  // open — it records the current state so the auto-open effect does not
  // immediately re-close it, matching the manual-wins rule.
  useImperativeHandle(ref, () => ({
    open: () => {
      setOpen(true);
      writeStoredOpen(true);
      lastAutoState.current = loadState ?? null;
      lastAutoStatus.current = status;
    },
  }), [loadState, status]);

  const preview = lastLogLine ? truncate(lastLogLine) : null;
  const age = lastLogAgeS != null ? formatAge(lastLogAgeS) : null;

  return (
    <div data-testid="live-logs-collapsible">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={toggle}
          data-testid="live-logs-toggle"
          aria-expanded={open}
        >
          {open ? "Hide log" : "Show log"}
        </Button>
        <span className="truncate text-xs text-slate-400">
          {preview ? (
            <>
              <span className="text-slate-500">
                {age ? `last line ${age} ago: ` : "last line: "}
              </span>
              <span className="font-mono" data-testid="live-logs-preview">
                &ldquo;{preview}&rdquo;
              </span>
            </>
          ) : (
            <span className="text-slate-500">Live logs</span>
          )}
        </span>
      </div>
      {open && (
        <div className="mt-3" data-testid="live-logs-body">
          {/* Key on modelId only — issue #52/#53 (see page.tsx comment).
              Keeping the same keying strategy as the old inline mount so
              LogStream keeps a single EventSource across toggles. */}
          <LogStream key={modelId} modelId={modelId} status={status} />
        </div>
      )}
    </div>
  );
});
