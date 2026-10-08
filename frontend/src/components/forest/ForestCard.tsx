"use client";

// The session forest card on /stats (spec §6.3): full width, 320 px high, live, over the Stats page's range (its
// 1h / 6h / 24h / 7d switch, follow-up C): the card fetches that range and its follow camera frames exactly that span.
// - Lazy: a placeholder until the card first intersects the viewport; only then is ForestView (and three.js with it)
//   fetched, through next/dynamic with ssr: false. Once loaded it stays mounted while the page scrolls.
// - Offscreen (IntersectionObserver, threshold 0) or a hidden tab: the view is `paused` (no render, no polling).
// - Phones ((pointer: coarse) or narrower than 640 px): the view renders one still frame and releases its WebGL
//   context. Tapping it opens /forest the same way as the full-window button, with no hand-off.
// - "Open full window": the camera view and the card's range go to sessionStorage["forest.handoff"] (handoff.ts), the
//   card's scene is disposed, and only then does the router go to /forest?range=<the card's range>. That order matters: two live WebGL contexts (card and full
//   window) must never exist at once.
// - A paused card keeps its (compact) WebGL context and buffers while off-screen; nothing grows meanwhile (review M3).
// - Memoised: its only prop is the range, so the Stats page's 1.5 s re-renders never reach it (review M2).

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { memo, useCallback, useEffect, useRef, useState } from "react";
// pure (no three.js): safe in the /stats entry chunk
import { type ForestView as CameraView, HANDOFF_KEY, encodeHandoff } from "@/components/forest/engine/handoff";
import type { ForestRange } from "@/lib/forest/types";

/** Below this window width the card shows a still frame. */
const PHONE_PX = 640;

function Placeholder() {
  return <div data-testid="forest-card-placeholder" aria-hidden="true" className="absolute inset-0 bg-[#0e1210]" />;
}

const ForestView = dynamic(() => import("./ForestView"), { ssr: false, loading: () => <Placeholder /> });

const isPhone = (): boolean => {
  try {
    if (typeof window.matchMedia === "function" && window.matchMedia("(pointer: coarse)").matches) return true;
  } catch {
    /* no matchMedia: decide on width alone */
  }
  return window.innerWidth < PHONE_PX;
};

/** `?debug=1` on /stats: the full window keeps the e2e debug hooks (`window.__forest`) after the hand-off. */
const debugQuery = (): boolean => {
  try {
    return new URLSearchParams(window.location.search).get("debug") === "1";
  } catch {
    return false;
  }
};

function ForestCard({ range }: { range: ForestRange }) {
  const router = useRouter();
  const ref = useRef<HTMLDivElement>(null);
  /** The card has intersected once: the view is loaded and stays mounted. */
  const [seen, setSeen] = useState(false);
  const [visible, setVisible] = useState(false);
  const [hidden, setHidden] = useState(false);
  /** Decided once, when the card first comes into view. */
  const [still, setStill] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let first = true;
    const show = (v: boolean) => {
      setVisible(v);
      if (v && first) {
        first = false;
        setStill(isPhone());
        setSeen(true);
      }
    };
    if (typeof IntersectionObserver === "undefined") {
      show(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        const last = entries[entries.length - 1];
        if (last) show(last.isIntersecting);
      },
      { threshold: 0 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const onChange = () => setHidden(document.hidden);
    onChange();
    document.addEventListener("visibilitychange", onChange);
    return () => document.removeEventListener("visibilitychange", onChange);
  }, []);

  const openFull = useCallback(
    (view: CameraView | null, release: () => void) => {
      try {
        if (view) sessionStorage.setItem(HANDOFF_KEY, encodeHandoff({ ...view, range }, Date.now()));
        else sessionStorage.removeItem(HANDOFF_KEY); // no view (a still): never a left-over pose
      } catch {
        /* no storage: the full window opens with its own camera */
      }
      release(); // the card's WebGL context goes before the full window's arrives
      // the range is in the query too (review M4): the window keeps the card's range when storage fails or the hand-off
      // is stale (the hand-off's range wins when it is there)
      router.push(`/forest?range=${range}${debugQuery() ? "&debug=1" : ""}`);
    },
    [router, range],
  );

  return (
    <div
      ref={ref}
      data-testid="forest-card"
      className="relative h-[320px] w-full overflow-hidden rounded-lg border border-slate-700 bg-slate-900/50 shadow-sm"
    >
      {seen ? (
        <ForestView
          mode="session"
          variant="card"
          range={range}
          paused={!visible || hidden}
          still={still}
          onOpenFull={openFull}
        />
      ) : (
        <Placeholder />
      )}
    </div>
  );
}

export default memo(ForestCard);
