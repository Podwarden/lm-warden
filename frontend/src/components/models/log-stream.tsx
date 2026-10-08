"use client";

// Live vLLM stdout/stderr tail with dual-mode "stick / free" scrolling.
//
// History:
//   - v17.11 #75 moved the tail from a plain <pre> fed by
//     dangerouslySetInnerHTML(lines.join('\n')) to <Virtuoso>, because
//     re-parsing the whole 5000-line buffer as HTML on every new line cost
//     ~80ms per line during a 200-line/s vLLM startup burst. It also added
//     `elided_count` so the operator knows when the FIFO has evicted lines.
//   - fix/live-log-drag-select moved it back to a plain scroll container,
//     but one keyed, memoized row per line. Virtuoso mounts only the rows
//     in view, so a drag-selection that auto-scrolled past the bottom edge
//     lost every line that scrolled away: a DOM selection cannot survive
//     its nodes being unmounted. With every buffered line (<= MAX_LINES) in
//     the DOM, native selection and drag-autoscroll just work. The #75 cost
//     does not come back: each line's ANSI->HTML is parsed once, when it
//     arrives, and React only diffs the keyed rows.
//
// Sticky-bottom: the "stick / free" latch comes from
// `shared/use-sticky-bottom.ts` and only drives the "Jump to latest"
// button. Following is done here: after new lines render, the view is
// pinned to the bottom if it was at the bottom, unless the operator is
// mid-selection (mouse button held in the log, or a live selection inside
// it). A drag-select near the bottom edge therefore never gets yanked by
// incoming lines.

import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { AnsiLog } from "@/components/ansi-log";
import { Button } from "@/components/ui/button";
import { useStickyBottom } from "@/components/shared/use-sticky-bottom";
import { useEventSource, MAX_RECONNECT, type SseState } from "@/lib/sse";
import { cn } from "@/lib/utils";

/**
 * Statuses where no new log content is being produced and opening an
 * EventSource would only churn single-use SSE tickets through the proxy:
 *   - "registered": log file may not exist yet
 *   - "unloading":  subprocess has been signaled, supervisor is tearing down
 * "failed" is deliberately NOT here — its log file contains the failure
 * traceback, and the 200-line backfill is operator-useful.
 *
 * Pull progress is rendered by the dedicated PullProgress card during
 * `pulling`, and load progress is implicit during `loading` — both
 * states already have richer UX than what live logs would add, but
 * vLLM does produce stdout during loading so we keep the stream open
 * for those. Operators have explicitly asked to see vLLM's stdout
 * during the 20GB-weight-load window so we MUST stream `loading`.
 *
 * Issue #53 — confining the enabled-states list keeps the EventSource
 * lifecycle minimal during the pre-load workflow.
 */
const NON_LOG_PRODUCING_STATUSES = new Set<string>([
  "registered",
  "unloading",
]);

interface LogStreamProps {
  modelId: string;
  /** Current model lifecycle status. When in a non-log-producing
   *  state (see NON_LOG_PRODUCING_STATUSES) the EventSource is NOT
   *  opened — we render an explanatory placeholder instead. Optional
   *  for backwards compatibility with callers that haven't been
   *  updated yet (treated as "stream-eligible"). */
  status?: string;
  className?: string;
  /** Fixed list height in px. Default matches the pre-#75 max-h-[480px]
   *  so the layout doesn't reflow on existing pages. */
  heightPx?: number;
}

// Cap the in-memory line buffer so a long-running tab can't bloat heap.
// Matches the plan's prev.slice(-4999) figure: 5000 lines is roughly
// the window an operator can reasonably scroll through, and at ~200B
// per line it tops out at ~1MB of strings — well below tab budgets.
export const MAX_LINES = 5000;

interface LogEvent {
  line: string;
}

/**
 * Map the SSE hook's state to a status-bar message + tone. Pulled out
 * of the component so the test can target the rendered string for each
 * status without driving the full SSE plumbing.
 */
function renderStatusMessage(s: SseState): { text: string; tone: "info" | "warn" | "error" } | null {
  switch (s.status) {
    case "connected":
      // Once the stream is live the bar disappears so the log lines can
      // own the vertical space. The caller distinguishes
      // connected-but-empty from connected-with-data and renders a
      // dedicated "(no log lines yet)" placeholder for the former (see
      // showEmptyPlaceholder below) so an operator never stares at a
      // blank role="log" div wondering whether the stream is wedged.
      return null;
    case "connecting":
      return { text: "Connecting to log stream…", tone: "info" };
    case "reconnecting":
      return {
        text: `Connection lost — retrying (${s.attempts}/${MAX_RECONNECT})…`,
        tone: "warn",
      };
    case "terminal-error": {
      // Distinguish auth failures from "stream gone" so the operator
      // has a real next step. errorCode === null means EventSource
      // gave up after MAX_RECONNECT without us getting an HTTP status.
      const code = s.errorCode;
      let text: string;
      if (code === 401 || code === 403) {
        text = "Stream unavailable — your session expired. Please re-login.";
      } else if (code === 404) {
        text = "Log stream not found for this model.";
      } else if (code !== null) {
        text = `Stream unavailable (HTTP ${code}). Please refresh.`;
      } else {
        text = `Stream unavailable after ${MAX_RECONNECT} retries. Please refresh.`;
      }
      return { text, tone: "error" };
    }
  }
}

// ---------------------------------------------------------------------------
// Row state
// ---------------------------------------------------------------------------

/** Internal log line — every row carries a stable id so React doesn't
 *  re-mount existing rows when the FIFO evicts the head. The id is a
 *  monotonically-increasing counter; the row's index in the array shifts
 *  on eviction, but its `id` does not. */
interface LogLine {
  id: number;
  text: string;
}

interface LogState {
  lines: LogLine[];
  /** Total lines elided by the FIFO so far. Surfaced as a "… N older
   *  lines elided" banner row at the head of the list. */
  elided: number;
  /** Monotonic counter for the next line's id. Survives eviction. */
  nextId: number;
}

const INITIAL_STATE: LogState = { lines: [], elided: 0, nextId: 0 };

function appendLine(prev: LogState, text: string): LogState {
  const next: LogLine = { id: prev.nextId, text };
  if (prev.lines.length >= MAX_LINES) {
    // FIFO eviction. Slice off the head and bump elided so the banner
    // tells the operator "+1 older line dropped".
    return {
      lines: [...prev.lines.slice(prev.lines.length - MAX_LINES + 1), next],
      elided: prev.elided + 1,
      nextId: prev.nextId + 1,
    };
  }
  return {
    lines: [...prev.lines, next],
    elided: prev.elided,
    nextId: prev.nextId + 1,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LogStream({ modelId, status, className, heightPx = 480 }: LogStreamProps) {
  const [state, setState] = useState<LogState>(INITIAL_STATE);

  // Memoize the message handler so useEventSource's effect-deps array
  // (which deliberately omits onMessage — see sse.ts) doesn't reopen
  // the connection on every parent rerender.
  const onMessage = useCallback((m: LogEvent) => {
    // Defensive: backend pin guarantees {line: string}, but a malformed
    // payload (e.g., bug in a future log filter) shouldn't crash the
    // render. Treat anything non-string as an empty line.
    const next = typeof m?.line === "string" ? m.line : "";
    setState((prev) => appendLine(prev, next));
  }, []);

  // Issue #53 — gate the SSE handshake on the model's lifecycle status.
  const streamEligible = !NON_LOG_PRODUCING_STATUSES.has(status ?? "");

  const sse = useEventSource<LogEvent>(`/api/models/${modelId}/logs/stream`, {
    onMessage,
    enabled: streamEligible,
  });

  // Sticky-bottom latch (drives "Jump to latest") + the follow machinery.
  const sticky = useStickyBottom("stick");
  const { onAtBottomStateChange, jumpToLatest: restick } = sticky;
  const scrollRef = useRef<HTMLDivElement>(null);
  // Last known "is the view at the bottom" — the gate for following.
  const atBottomRef = useRef(true);
  // True from a primary-button mousedown inside the log until the next
  // mouseup anywhere: a drag-select (or scrollbar drag) is in progress.
  const pointerHeldRef = useRef(false);

  const setAtBottom = useCallback(
    (atBottom: boolean) => {
      if (atBottomRef.current === atBottom) return;
      atBottomRef.current = atBottom;
      onAtBottomStateChange(atBottom);
    },
    [onAtBottomStateChange],
  );

  const onScroll = useCallback(() => {
    const el = scrollRef.current;
    if (el) setAtBottom(distanceFromBottom(el) <= AT_BOTTOM_THRESHOLD_PX);
  }, [setAtBottom]);

  useEffect(() => {
    const release = () => {
      pointerHeldRef.current = false;
    };
    window.addEventListener("mouseup", release);
    window.addEventListener("blur", release);
    return () => {
      window.removeEventListener("mouseup", release);
      window.removeEventListener("blur", release);
    };
  }, []);

  // Follow the tail after each append, before paint. Pin to the bottom if
  // we were there — unless the operator is selecting, in which case the
  // view holds still and, once the new lines push the bottom out of view,
  // the latch flips to "free" so "Jump to latest" appears.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || !atBottomRef.current) return;
    if (!pointerHeldRef.current && !selectionInside(el)) {
      el.scrollTop = el.scrollHeight;
    } else if (distanceFromBottom(el) > AT_BOTTOM_THRESHOLD_PX) {
      setAtBottom(false);
    }
  }, [state.lines, setAtBottom]);

  const jumpToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      // "Jump to latest" means "resume tailing": drop a selection left
      // in the log, otherwise it would keep following paused.
      if (selectionInside(el)) document.getSelection()?.removeAllRanges();
      el.scrollTop = el.scrollHeight;
    }
    atBottomRef.current = true;
    restick();
  }, [restick]);

  if (!streamEligible) {
    // Non-log-producing placeholder (registered / unloading).
    return (
      <div
        role="status"
        className={cn(
          "rounded border p-3 text-xs",
          "border-slate-700 bg-slate-900/50 text-slate-400",
          className,
        )}
      >
        Log stream paused (no active subprocess).
      </div>
    );
  }

  const statusBar = renderStatusMessage(sse);

  // Show the status bar when there's nothing in the buffer yet OR when
  // the stream entered a non-connected state after the fact.
  const showStatusBar =
    statusBar !== null && (state.lines.length === 0 || sse.status !== "connected");

  // Connected-but-empty placeholder (subprocess hasn't produced stdout
  // yet — e.g. vLLM loading 20GB of weights).
  const showEmptyPlaceholder = sse.status === "connected" && state.lines.length === 0;

  if (showEmptyPlaceholder) {
    return (
      <div
        role="status"
        className={cn(
          "rounded border p-3 text-xs",
          "border-slate-700 bg-slate-900/50 text-slate-400",
          className,
        )}
      >
        (no log lines yet)
      </div>
    );
  }

  if (state.lines.length === 0 && statusBar !== null) {
    // Pre-first-line placeholder.
    const role = statusBar.tone === "error" ? "alert" : "status";
    const toneClass =
      statusBar.tone === "error"
        ? "border-red-700/60 bg-red-950/40 text-red-200"
        : statusBar.tone === "warn"
          ? "border-amber-700/60 bg-amber-950/30 text-amber-200"
          : "border-slate-700 bg-slate-900/50 text-slate-400";
    return (
      <div
        role={role}
        className={cn("rounded border p-3 text-xs", toneClass, className)}
      >
        {statusBar.text}
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {showStatusBar && statusBar !== null && (
        // Inline status bar above the log when we already have lines.
        <div
          role={statusBar.tone === "error" ? "alert" : "status"}
          className={cn(
            "rounded border px-3 py-1.5 text-xs",
            statusBar.tone === "error"
              ? "border-red-700/60 bg-red-950/40 text-red-200"
              : statusBar.tone === "warn"
                ? "border-amber-700/60 bg-amber-950/30 text-amber-200"
                : "border-slate-700 bg-slate-900/50 text-slate-400",
          )}
        >
          {statusBar.text}
        </div>
      )}

      {state.elided > 0 && (
        // Eviction marker. Lives outside the log scroller (as a fixed
        // header row above the scroll region) so it's always visible
        // regardless of scroll position — the operator should never be
        // surprised that older lines were dropped silently.
        <div
          role="status"
          className="rounded-t border border-b-0 border-slate-700 bg-slate-900/70 px-3 py-1 font-mono text-[11px] text-slate-400"
        >
          … {state.elided} older line{state.elided === 1 ? "" : "s"} elided
        </div>
      )}

      <div
        className={cn(
          "relative rounded border border-slate-700 bg-slate-950",
          state.elided > 0 ? "rounded-t-none border-t-0" : undefined,
        )}
      >
        {/* One native scroll container holding every buffered line, so a
            drag-selection keeps lines that scroll out of view (see the
            header comment). role="log" is an implicit polite live region
            (aria-live=polite, aria-atomic=false). overflow-anchor keeps the
            view steady when the FIFO evicts the head while scrolled up. */}
        <div
          ref={scrollRef}
          role="log"
          aria-label="Model log stream"
          className="overflow-y-auto"
          style={{ height: heightPx, overflowAnchor: "auto" }}
          onScroll={onScroll}
          onMouseDown={(e) => {
            if (e.button === 0) pointerHeldRef.current = true;
          }}
        >
          {state.lines.map((line) => (
            <LogRow key={line.id} text={line.text} />
          ))}
        </div>

        {sticky.mode === "free" && state.lines.length > 0 && (
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="absolute bottom-2 right-2 shadow-lg"
            onClick={jumpToLatest}
          >
            Jump to latest
          </Button>
        )}
      </div>
    </div>
  );
}

// Bottom tolerance for "at the bottom" (a log row is ~20px). ~3 rows of
// slack so sub-row rounding and a just-appended row don't count as the
// operator having scrolled up.
const AT_BOTTOM_THRESHOLD_PX = 64;

function distanceFromBottom(el: HTMLElement): number {
  return el.scrollHeight - el.scrollTop - el.clientHeight;
}

/** True when the document has a non-empty selection touching `el`. */
function selectionInside(el: HTMLElement): boolean {
  const sel = typeof document !== "undefined" ? document.getSelection() : null;
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return false;
  return el.contains(sel.anchorNode) || el.contains(sel.focusNode);
}

// Memoized so an append re-renders only the new row; existing rows keep
// their already-parsed ANSI HTML.
const LogRow = memo(function LogRow({ text }: { text: string }) {
  return (
    <div data-log-line className="px-3 py-0 font-mono text-xs leading-5">
      <AnsiLog text={text} />
    </div>
  );
});
