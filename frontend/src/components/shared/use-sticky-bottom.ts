"use client";

// Dual-mode "stick to bottom / free scroll" helper for streaming lists.
//
// Used by:
//   - models/log-stream.tsx       (vLLM stdout/stderr tail — a plain scroll
//                                  container since fix/live-log-drag-select;
//                                  it uses only `mode`, `onAtBottomStateChange`
//                                  and `jumpToLatest`, and does its own follow)
// (prompt-lens/prompt-lens-viewer.tsx used it too until the Prompt Lens dock replaced
// that viewer; the dock follows with dock/use-follow-tail.ts.)
//
// Originally also driven the bench/events-tab tail; that consumer was
// removed in epic/overhaul S1. The hook stays in `shared/` because the
// same semantics will be needed by future streaming views (e.g. system
// messages, gpu_samples replay).
//
// The hook is deliberately tiny and unaware of the underlying list — it
// just exposes the mode and the callbacks a scroll container calls when the
// reader reaches or leaves the bottom, so it works with any plain scroll
// container (LogStream needs that for the elided-marker overlay).

import { useCallback, useState } from "react";

export type StickyMode = "stick" | "free";

export interface UseStickyBottomReturn {
  /** Current mode. UI uses this to show/hide the "Jump to latest" button. */
  mode: StickyMode;
  /**
   * Call with `true` when the reader is at the bottom and `false` once they
   * scroll up — we flip into `free` on `false` and stay there until
   * `jumpToLatest()`.
   */
  onAtBottomStateChange: (atBottom: boolean) => void;
  /**
   * Programmatic re-stick. Call from the "Jump to latest" button.
   *
   * The hook itself doesn't know how to scroll; the consumer scrolls its
   * container to the bottom and then calls this. The scroll lands, the
   * bottom detection re-fires and the mode flips back through
   * `onAtBottomStateChange` — but if the list is empty there's no event to
   * flip on, so we flip eagerly here too.
   */
  jumpToLatest: () => void;
  /**
   * The follow policy. ALWAYS `"auto"` (follow instantly while at the
   * bottom), decoupled from the stick/free latch on purpose: a follow gated
   * on `mode` once unstuck the tail for good on the transient "not at the
   * bottom" a fast append burst causes (tests/component/use-sticky-bottom).
   * The latch drives nothing but the "Jump to latest" button's visibility.
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
    // Always-on follow; the consumer follows only while at the bottom.
    // Decoupled from `mode` so a transient `atBottom=false` during a
    // streaming burst can no longer disable tailing. See the followOutput
    // doc on UseStickyBottomReturn for the full rationale.
    followOutput: "auto",
  };
}
