"use client";

// Drives the usage window (spec §4.3): a drag/arrow-key edit is reflected
// immediately (via `pending`) but the URL — and so the series request — only
// updates after `debounceMs` of quiet. Once the URL (`sel`) catches up with
// what was last dragged, `pending` clears so the window tracks `sel` again
// instead of a frozen snapshot from mid-drag.
//
// Pulled out of the page component (rather than kept inline) so it can be
// exercised directly with `renderHook`, independent of the DOM/pointer-event
// machinery `<HistoryStrip>` needs to drive a real drag.

import { useCallback, useEffect, useRef, useState } from "react";
import { rangeQuery, type Preset, type RangeSel } from "@/lib/token-series";
import type { StripWindow } from "./history-strip";

export interface WindowSelection {
  /** Set the instant a drag/arrow-key edit starts; cleared once the URL
   *  (`sel`) reflects it. Exposed mainly for tests — the page itself only
   *  needs `shownWindow`/`shownSel`. */
  pending: StripWindow | null;
  shownWindow: StripWindow;
  shownSel: RangeSel;
  onWindowChange: (w: StripWindow) => void;
  /** The custom row's Apply: writes the URL at once (no debounce, dropping
   *  any drag still waiting on it) and returns whether the window differs
   *  from the one already in the URL. When it does not, nothing about the
   *  URL -- and so nothing about the series request -- changes, and the
   *  caller must refetch itself or Apply looks like it did nothing. */
  onApply: (w: StripWindow) => boolean;
  onPreset: (p: Preset) => void;
  onCustom: () => void;
}

export function useWindowSelection(
  sel: RangeSel,
  committed: StripWindow,
  replaceQuery: (query: string) => void,
  debounceMs = 250,
): WindowSelection {
  const [pending, setPending] = useState<StripWindow | null>(null);
  const debounce = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(debounce.current), []);

  // The URL caught up with the last drag — stop overriding `committed`.
  useEffect(() => {
    if (pending && sel.kind === "custom" && sel.from === pending.from && sel.to === pending.to) setPending(null);
  }, [sel, pending]);

  const onWindowChange = useCallback((w: StripWindow) => {
    const next = { from: Math.round(w.from), to: Math.round(w.to) };
    setPending(next);
    clearTimeout(debounce.current);
    // Spec §4.3: requests are debounced (~250 ms) while dragging — several
    // calls in quick succession collapse into one URL write.
    debounce.current = setTimeout(() => replaceQuery(rangeQuery({ kind: "custom", ...next })), debounceMs);
  }, [replaceQuery, debounceMs]);

  const onApply = useCallback((w: StripWindow): boolean => {
    const next = { from: Math.round(w.from), to: Math.round(w.to) };
    clearTimeout(debounce.current);
    if (sel.kind === "custom" && sel.from === next.from && sel.to === next.to) {
      setPending(null);
      return false;
    }
    setPending(next);
    replaceQuery(rangeQuery({ kind: "custom", ...next }));
    return true;
  }, [sel, replaceQuery]);

  const onPreset = useCallback((p: Preset) => {
    clearTimeout(debounce.current);
    setPending(null);
    replaceQuery(rangeQuery({ kind: "preset", preset: p }));
  }, [replaceQuery]);

  const onCustom = useCallback(() => {
    if (sel.kind === "custom") return;
    replaceQuery(rangeQuery({ kind: "custom", from: Math.round(committed.from), to: Math.round(committed.to) }));
  }, [sel, committed, replaceQuery]);

  return {
    pending,
    shownWindow: pending ?? committed,
    shownSel: pending ? ({ kind: "custom", ...pending } as const) : sel,
    onWindowChange,
    onApply,
    onPreset,
    onCustom,
  };
}
