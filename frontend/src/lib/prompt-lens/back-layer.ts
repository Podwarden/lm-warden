"use client";

// The phone back gesture for Prompt Lens layers (spec 2026-11-22 section 6):
// the dock's full-screen sheet, the request screen over the list, the
// filter sheet and the section pop-up. Each open layer owns one history
// entry, pushed when it opens; the system back gesture pops that entry and
// closes the top layer only. Closing a layer from the page (✕, "‹ Requests",
// Collapse) takes its entry back off, so the history never fills with stale
// entries.
//
// Layers stack by depth: the entry's state carries `__lensLayer` (copied over
// the router's own state, so Next.js restores the same page when it pops).
// A popstate whose depth is below a layer's depth closes that layer.
//
// Entries removed in one commit (two layers closing together) are taken off
// with one history.go(-n) in a microtask. A layer that mounts before that
// runs (React's StrictMode remount) adopts the entry instead of pushing.

import { useEffect, useRef } from "react";

const KEY = "__lensLayer";

function depthOf(state: unknown): number {
  if (state && typeof state === "object" && KEY in state) {
    const d = (state as Record<string, unknown>)[KEY];
    return typeof d === "number" && d > 0 ? d : 0;
  }
  return 0;
}

let pendingBack = 0;

function scheduleBack() {
  pendingBack += 1;
  if (pendingBack > 1) return;
  queueMicrotask(() => {
    const n = pendingBack;
    pendingBack = 0;
    if (n > 0) window.history.go(-n);
  });
}

/** While `active`, the browser's back gesture calls `onBack` (once) instead
 *  of leaving the page. */
export function useBackLayer(active: boolean, onBack: () => void): void {
  const cb = useRef(onBack);
  cb.current = onBack;
  useEffect(() => {
    if (!active || typeof window === "undefined") return;
    const h = window.history;
    let mine: number;
    if (pendingBack > 0) {
      // A layer closed in this commit left its entry for us: reuse it.
      pendingBack -= 1;
      mine = depthOf(h.state);
      if (mine === 0) {
        mine = 1;
        h.pushState({ ...(h.state ?? {}), [KEY]: mine }, "");
      }
    } else {
      mine = depthOf(h.state) + 1;
      h.pushState({ ...(h.state ?? {}), [KEY]: mine }, "");
    }
    let popped = false;
    const onPop = (e: PopStateEvent) => {
      if (popped || depthOf(e.state) >= mine) return;
      popped = true;
      cb.current();
    };
    window.addEventListener("popstate", onPop);
    return () => {
      window.removeEventListener("popstate", onPop);
      if (!popped && depthOf(h.state) >= mine) scheduleBack();
    };
  }, [active]);
}
