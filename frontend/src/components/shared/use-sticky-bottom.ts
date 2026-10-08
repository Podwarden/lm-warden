"use client";

// Dual-mode "stick to bottom / free scroll" helper for streaming lists.
//
// Used by:
//   - models/log-stream.tsx       (vLLM stdout/stderr tail — a plain scroll
//                                  container since fix/live-log-drag-select;
//                                  it uses only `mode`, `onAtBottomStateChange`
//                                  and `jumpToLatest`, and does its own follow)
//   - godmode/godmode-viewer.tsx  (Virtuoso; also uses `followOutput`)
//
// Originally also driven the bench/events-tab tail; that consumer was
// removed in epic/overhaul S1. The hook stays in `shared/` because the
// same semantics will be needed by future streaming views (e.g. system
// messages, gpu_samples replay).
//
// The hook is deliberately tiny and unaware of the underlying list — it
// just exposes the mode + the two callbacks the consumer wires into
// react-virtuoso's `followOutput` and `atBottomStateChange`. Doing it
// this way means the hook works equally well with a plain scroll
// container (LogStream needs that for the elided-marker overlay) and
// with Virtuoso's tracker.

import { useCallback, useState } from "react";

export type StickyMode = "stick" | "free";

export interface UseStickyBottomReturn {
  /** Current mode. UI uses this to show/hide the "Jump to latest" button. */
  mode: StickyMode;
  /**
   * Pass to react-virtuoso's `atBottomStateChange` prop. Argument is `true`
   * when the user is at the bottom and `false` once they scroll up — we
   * flip into `free` on `false` and stay there until `jumpToLatest()`.
   */
  onAtBottomStateChange: (atBottom: boolean) => void;
  /**
   * Programmatic re-stick. Call from the "Jump to latest" button.
   *
   * The hook itself doesn't know how to scroll; the consumer is expected
   * to chain a scroll call (e.g. `virtuosoRef.current?.scrollToIndex`) and
   * then this. The scroll lands, the bottom-detection re-fires, and the
   * mode flips back via the natural `atBottomStateChange` path — but if
   * the list is empty there's no event to flip on, so we flip eagerly
   * here too.
   */
  jumpToLatest: () => void;
  /**
   * Value for `<Virtuoso followOutput=...>`. ALWAYS `"auto"` — decoupled
   * from the stick/free latch on purpose.
   *
   * react-virtuoso only auto-scrolls when the list is ALREADY at the
   * bottom, so an always-on follow keeps the tail pinned through a fast
   * append burst yet never yanks an operator who has scrolled up to read
   * (Virtuoso self-gates). We use `"auto"` (instant) rather than
   * `"smooth"`: a ~200 line/s vLLM startup burst outruns the smooth
   * animation, leaving the viewport perpetually shy of the bottom, which
   * makes Virtuoso emit a steady stream of transient `atBottom=false` —
   * the exact signal that used to permanently unstick the tail back when
   * `followOutput` was gated on `mode` (`false` while "free"). The latch
   * now drives nothing but the "Jump to latest" button's visibility.
   */
  followOutput: "auto";
}

export function useStickyBottom(initial: StickyMode = "stick"): UseStickyBottomReturn {
  const [mode, setMode] = useState<StickyMode>(initial);

  const onAtBottomStateChange = useCallback((atBottom: boolean) => {
    // Asymmetric: bottom → "stick" only when the user is genuinely at the
    // bottom. Top → "free" the instant they pull up even one pixel. The
    // asymmetry is intentional — re-sticking on accidental bottom-touch
    // would yank them back into tail mode mid-read.
    setMode((prev) => {
      if (!atBottom) return "free";
      return prev === "free" ? "free" : "stick";
      // ^ Note: we DO NOT auto-flip free → stick here. The "Jump to
      // latest" button is the only path back. Auto-resticking on
      // bottom would re-introduce the "scroll-up to read, scroll
      // slightly past bottom, content jumps" jank we built this hook
      // to eliminate.
    });
  }, []);

  const jumpToLatest = useCallback(() => {
    setMode("stick");
  }, []);

  return {
    mode,
    onAtBottomStateChange,
    jumpToLatest,
    // Always-on follow; Virtuoso self-gates to "only when at bottom".
    // Decoupled from `mode` so a transient `atBottom=false` during a
    // streaming burst can no longer disable tailing. See the followOutput
    // doc on UseStickyBottomReturn for the full rationale.
    followOutput: "auto",
  };
}
