"use client";

// /forest — the session forest, full window (spec §6.4, live view only §10.6).
// three.js is loaded only here: ForestView comes in through next/dynamic with ssr: false, so no forest code enters
// the bundle of any other page (the /stats chunks are checked for it after `next build`).
//
// Two ways in (Task 8):
//   - a key holder with a forest token (from /forest/login; `?key=1` marks that view) → mode "forest-token", scoped
//     to their key. No admin chrome (nav-bar.tsx hides on /forest) and no admin request.
//   - an admin with a session → mode "session".
// session-gate.tsx has already decided which (an expired token, or neither, went to /forest/login).
//
// The Stats card's hand-off (sessionStorage["forest.handoff"], handoff.ts) is read and deleted on arrival. Session
// mode imports its camera view as soon as the scene exists; a stale or malformed one changes nothing. Forest-token
// mode never uses it (a key holder did not come from the admin card): it is only deleted.
//
// Range (follow-up C): 1h / 6h / 24h / 7d, switched in the controls bar (both modes). It opens with the hand-off's
// range (the card's), else `?range=`, else the last choice (localStorage["forest.range"], read in try/catch), else 24 h.
// A switch is remembered there.

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSWRConfig } from "swr";
import { type ForestView as CameraView, takeHandoff } from "@/components/forest/engine/handoff";
import type { ForestMode } from "@/components/forest/panels/InFlight";
import { clearForestToken, forestKeyRequested, isForestTokenCacheKey, readForestToken } from "@/lib/forest/token";
import { type ForestRange, isForestRange } from "@/lib/forest/types";

/** The full window's remembered range. */
const RANGE_KEY = "forest.range";
const DEFAULT_RANGE: ForestRange = "24h";

/** `?range=`, then the remembered choice, then 24 h (the hand-off's range wins over all of them; see the page). */
function initialRange(): ForestRange {
  try {
    const q = new URLSearchParams(window.location.search).get("range");
    if (isForestRange(q)) return q;
  } catch {
    /* no query */
  }
  try {
    const v = window.localStorage.getItem(RANGE_KEY);
    if (isForestRange(v)) return v;
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_RANGE;
}

const ForestView = dynamic(() => import("@/components/forest/ForestView"), {
  ssr: false,
  loading: () => (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#0e1210] text-sm text-[#eef2ec]/70">
      Loading the forest…
    </div>
  ),
});

export default function ForestPage() {
  const router = useRouter();
  // decided on the client (sessionStorage and the query are not known on the server)
  const [mode, setMode] = useState<ForestMode | null>(null);
  const [handoff, setHandoff] = useState<CameraView | null>(null);
  const [range, setRange] = useState<ForestRange>(DEFAULT_RANGE);
  /** Taken once per mount: StrictMode's second effect run must not find the entry deleted and drop the view. */
  const taken = useRef<{ view: CameraView | null } | null>(null);
  useEffect(() => {
    const m: ForestMode = readForestToken() !== null || forestKeyRequested() ? "forest-token" : "session";
    taken.current ??= { view: takeHandoff(Date.now()) }; // deleted in either mode: used at most once
    const view = m === "session" ? taken.current.view : null;
    setHandoff(view);
    setRange(view?.range ?? initialRange());
    setMode(m);
  }, []);

  /**
   * Latched, once: clear the token and the forest-token SWR entries, then go to the key holder's login. Both an auth
   * failure (ForestView's onAuthError) and the explicit Sign out end here.
   */
  const { mutate } = useSWRConfig();
  const leaving = useRef(false);
  const toLogin = useCallback(() => {
    if (leaving.current) return;
    leaving.current = true;
    clearForestToken();
    void mutate(isForestTokenCacheKey, undefined, { revalidate: false });
    router.replace("/forest/login");
  }, [mutate, router]);

  // Session mode: Esc, ✕ or leaving fullscreen go back to /stats. Token mode: a key holder has no admin page to go
  // back to, so there is no close at all (Esc only leaves fullscreen), and an explicit Sign out replaces ✕.
  const close = useCallback(() => router.push("/stats"), [router]);

  const changeRange = useCallback((r: ForestRange) => {
    setRange(r);
    try {
      window.localStorage.setItem(RANGE_KEY, r);
    } catch {
      /* not remembered */
    }
    // a ?range= in the address (the card's) follows the switch, so a reload keeps the choice
    try {
      const u = new URL(window.location.href);
      if (u.searchParams.has("range")) {
        u.searchParams.set("range", r);
        window.history.replaceState(window.history.state, "", u);
      }
    } catch {
      /* no history */
    }
  }, []);

  if (mode === null) return null;
  return (
    <ForestView
      mode={mode}
      onClose={mode === "session" ? close : undefined}
      onSignOut={mode === "forest-token" ? toLogin : undefined}
      onAuthError={mode === "forest-token" ? toLogin : undefined}
      fullscreen
      initialView={handoff}
      range={range}
      onRange={changeRange}
    />
  );
}
