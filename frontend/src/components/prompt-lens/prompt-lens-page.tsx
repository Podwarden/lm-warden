"use client";

// The pieces the two Prompt Lens pages share (admin /prompt-lens, key holder
// /key/prompt-lens; owner request 2026-11-23: "a standalone page, with menu,
// breadcrumbs, etc. ... take full screen from start").
//
//  * FullScreen: the page breaks out of the app's centred container and fills
//    the viewport below whatever sits above it (nav, breadcrumb, a banner):
//    its height is measured, never hard-coded, and the document does not
//    scroll while it is shown (the list and the request screen scroll).
//  * The deep link: ?req=<request id> opens that request (the Requests eye,
//    a shared link); admin also ?key=<id>[,<id>...] presets the Key filter
//    (the key details page). Picking a request in the list writes ?req= back
//    with history.replaceState, so the address can be shared and a reload
//    opens the same request; it adds no history entry (the back gesture
//    still steps through the layers, lib/prompt-lens/back-layer).

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

import type { RequestView } from "@/lib/prompt-lens/store";

/** The smallest height the page takes on a very short window. */
const MIN_PX = 320;

export function FullScreen({ children, testid }: { children: ReactNode; testid?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const fit = () => {
      const top = el.getBoundingClientRect().top + window.scrollY;
      setHeight(Math.max(MIN_PX, Math.floor(window.innerHeight - top)));
    };
    fit();
    window.addEventListener("resize", fit);
    // A banner above (update notice) can appear or go after the first fit.
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(fit) : null;
    if (ro && el.parentElement) ro.observe(document.body);
    return () => {
      window.removeEventListener("resize", fit);
      ro?.disconnect();
    };
  }, []);
  useEffect(() => {
    const html = document.documentElement;
    const prev = html.style.overflow;
    html.style.overflow = "hidden";
    return () => {
      html.style.overflow = prev;
    };
  }, []);
  return (
    <div
      ref={ref}
      data-testid={testid ?? "prompt-lens-page"}
      // Full width out of the centred container (no transform: it would
      // capture position: fixed children such as the phone filter sheet),
      // and up against the bar above (the container's top padding).
      className="relative -mb-6 -mt-6 ml-[calc(50%-50vw)] w-screen overflow-hidden"
      style={{ height: height ?? "calc(100dvh - 3.5rem - 30px)" }}
    >
      {children}
    </div>
  );
}

/** ?req= and ?key= of the address, read once when the page opens: a pick
 *  later rewrites ?req= and must not select again. */
export function useLensDeepLink(params: URLSearchParams | null): {
  select: { requestId: string; nonce: number } | null;
  keys: string[] | null;
} {
  const [first] = useState(() => {
    const req = params?.get("req") ?? "";
    const key = params?.get("key") ?? "";
    const keys = key
      .split(",")
      .map((k) => k.trim())
      .filter(Boolean);
    return { select: req ? { requestId: req, nonce: 1 } : null, keys: keys.length ? keys : null };
  });
  return first;
}

/** Write the picked request into the address (?req=), keeping the rest. */
export function useWritePick(): (r: RequestView) => void {
  return useCallback((r: RequestView) => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("req", r.requestId ?? r.reqId);
    window.history.replaceState(window.history.state, "", url.pathname + url.search + url.hash);
  }, []);
}

/** Where the Requests eye and the key details page send the reader. */
export function promptLensHref(scope: "admin" | "key", o: { req?: string | null; keys?: string[] | null } = {}): string {
  const q = new URLSearchParams();
  if (o.req) q.set("req", o.req);
  if (scope === "admin" && o.keys && o.keys.length) q.set("key", o.keys.join(","));
  const qs = q.toString();
  return `${scope === "admin" ? "/prompt-lens" : "/key/prompt-lens"}${qs ? `?${qs}` : ""}`;
}
