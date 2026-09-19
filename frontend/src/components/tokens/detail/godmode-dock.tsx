"use client";

// God-mode dock (spec 2026-09-18 §4.5, D6/D7). The ONLY god-mode UI.
//
// Closed = no GodModeViewer mounted = no ticket minted, no EventSource. Opening
// mounts the viewer (it opens the stream and replays this key's recent
// requests from the ring); closing unmounts it, which closes the EventSource.
// There is no hidden or background stream. The dock always starts closed:
// only its height is remembered (localStorage, wrapped in try/catch).

import { useCallback, useEffect, useId, useRef, useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { SseStatus } from "@/lib/sse";
import { cn } from "@/lib/utils";
import {
  GodModeViewer,
  type GodModeCounts,
  type GodModeViewerHandle,
} from "@/components/godmode/godmode-viewer";
import { FONT_SANS, GUTTER_X } from "./styles";

export const DOCK_HEIGHT_KEY = "vw.tokens.dockHeight";
export const DOCK_MIN_PX = 160;

export function clampDockHeight(px: number, viewportH: number): number {
  return Math.round(Math.min(Math.max(px, DOCK_MIN_PX), viewportH * 0.85));
}

const viewportH = () => (typeof window === "undefined" ? 1000 : window.innerHeight);

export function useDockHeight(): [number, (px: number, persist?: boolean) => void] {
  const [height, setHeight] = useState(() => Math.round(viewportH() * 0.45));
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(DOCK_HEIGHT_KEY);
      const n = raw == null ? NaN : Number(raw);
      if (Number.isFinite(n)) setHeight(clampDockHeight(n, viewportH()));
    } catch {
      /* storage unavailable — keep the default */
    }
  }, []);
  // A window shrink (e.g. rotating a laptop lid down, or DevTools docking)
  // can leave a previously-fine height taller than 85% of the new viewport,
  // pushing the bar/grip off-screen with no way to close or resize the dock.
  // Re-clamp on every resize — never persisted, so the saved preference
  // still applies at the original viewport size.
  useEffect(() => {
    const onResize = () => setHeight((h) => clampDockHeight(h, viewportH()));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const set = useCallback((px: number, persist = false) => {
    const v = clampDockHeight(px, viewportH());
    setHeight(v);
    if (persist) {
      try {
        window.localStorage.setItem(DOCK_HEIGHT_KEY, String(v));
      } catch {
        /* persist failed — the height still applies for this page */
      }
    }
  }, []);
  return [height, set];
}

/** Spec §4.5/§5: the dock renders only when the status endpoint says enabled;
 *  loading or any error hides it. SWR keeps the last good `data` next to the
 *  error of a failed refetch, so the error must be checked too (#251). */
export function useGodModeEnabled(): boolean {
  const { data, error } = useSWR<{ enabled: boolean }>("/api/admin/godmode/status", authFetchJSON, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  return !error && data?.enabled === true;
}

/** Operator's hard rule (spec §4.5, D6/D7): opening the dock must always be a
 *  deliberate click. `<GodModeDock>` only renders while god mode is enabled,
 *  so it unmounts the moment `enabled` drops — but the `open` boolean has to
 *  live in the parent (it's a prop here), and would otherwise survive that
 *  unmount. Left alone, a later flip back to enabled would remount the dock
 *  already open and start a stream with no click. Resetting `open` to false
 *  the instant `enabled` goes false closes that gap.
 *
 *  Lives here (not in the page) because a Next.js App Router `page.tsx` may
 *  only export `default` plus a small fixed set of config fields — any other
 *  named export fails `next build`'s page-type check even though `tsc
 *  --noEmit` doesn't catch it. */
export function useDockOpen(enabled: boolean): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!enabled) setOpen(false);
  }, [enabled]);
  return [open, setOpen];
}

export interface GodModeDockProps {
  tokenName: string;
  tokenIds: string[];
  includesEarlier: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  height: number;
  onHeightChange: (px: number, persist?: boolean) => void;
}

const TOOL =
  "cursor-pointer rounded-[5px] border border-chat-rule bg-transparent px-[9px] py-[3px] text-[12px] text-chat-muted hover:bg-chat-surface hover:text-chat-fg";

export function GodModeDock({
  tokenName, tokenIds, includesEarlier, open, onOpenChange, height, onHeightChange,
}: GodModeDockProps) {
  const viewer = useRef<GodModeViewerHandle>(null);
  const [counts, setCounts] = useState<GodModeCounts>({ requests: 0, running: 0 });
  const [following, setFollowing] = useState(true);
  const [status, setStatus] = useState<SseStatus>("connecting");
  const resize = useRef<{ y: number; h: number } | null>(null);
  const bodyId = useId();

  useEffect(() => {
    if (!open) {
      setCounts({ requests: 0, running: 0 });
      setFollowing(true);
      setStatus("connecting");
    }
  }, [open]);

  const toggle = () => onOpenChange(!open);

  return (
    <section
      aria-label="God mode for this token"
      className={cn(FONT_SANS, "fixed bottom-0 left-0 right-0 z-20 flex flex-col border-t border-chat-rule bg-vw-dock shadow-[0_-10px_30px_rgba(0,0,0,.45)]")}
    >
      {open && (
        <div
          title="Drag to resize"
          className="h-1.5 cursor-ns-resize touch-none"
          onPointerDown={(e) => {
            resize.current = { y: e.clientY, h: height };
            e.currentTarget.setPointerCapture?.(e.pointerId);
          }}
          onPointerMove={(e) => {
            const r = resize.current;
            if (r) onHeightChange(r.h + (r.y - e.clientY));
          }}
          onPointerUp={(e) => {
            const r = resize.current;
            resize.current = null;
            if (r) onHeightChange(r.h + (r.y - e.clientY), true);
          }}
          onPointerCancel={() => {
            resize.current = null;
          }}
          onLostPointerCapture={() => {
            resize.current = null;
          }}
        >
          <div className="mx-auto mt-0.5 h-[3px] w-11 rounded-sm bg-chat-rule" />
        </div>
      )}

      <div
        role="button"
        tabIndex={0}
        aria-label="God mode"
        aria-expanded={open}
        aria-controls={bodyId}
        className={cn(GUTTER_X, "flex h-10 cursor-pointer select-none items-center gap-3 hover:bg-chat-surface-2/35")}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("[data-dock-tool]")) return;
          toggle();
        }}
        onKeyDown={(e) => {
          if (e.target !== e.currentTarget) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
          }
        }}
      >
        <span className="flex items-center gap-2 text-[13px] font-semibold">
          <svg className="text-chat-accent-strong" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
            <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
          God mode
        </span>
        <span className="text-[12px] text-chat-dim">
          {open
            ? `${tokenName}${includesEarlier ? " and its earlier keys" : ""} · replayed the last requests, now streaming`
            : "Open to watch this key's prompts and responses as they happen"}
        </span>
        <span className="ml-auto flex items-center gap-2.5 text-[12px] text-chat-muted">
          {open && status === "connected" && (
            <span className="inline-flex items-center gap-1.5 font-medium text-vw-ok-fg">
              <span aria-hidden className="vw-live-dot" />
              Live
            </span>
          )}
          {/* Always rendered, empty while closed — the mockup's #dock-count is,
              and as a flex item it takes a 10px gap that shapes the bar. */}
          <span>{open ? `${counts.requests} requests · ${counts.running} running` : ""}</span>
          {open && (
            <>
              <button type="button" data-dock-tool aria-pressed={following} className={TOOL}
                onClick={() => viewer.current?.setFollowing(!following)}>
                {following ? "Following" : "Paused scroll"}
              </button>
              <button type="button" data-dock-tool className={TOOL} onClick={() => viewer.current?.clear()}>
                Clear
              </button>
            </>
          )}
          <svg
            className={cn("text-chat-muted transition-transform duration-150 ease-[ease] motion-reduce:transition-none", open && "rotate-180")}
            width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden
          >
            <path d="m6 15 6-6 6 6" />
          </svg>
        </span>
      </div>

      <div id={bodyId} className="relative overflow-hidden" style={{ height: open ? height : 0 }}>
        {open && (
          <GodModeViewer
            // A new token set (Include earlier keys toggled) restarts the stream
            // with a clean ring (spec §4.5).
            key={tokenIds.join(",")}
            ref={viewer}
            tokenIds={tokenIds}
            heightPx={height}
            className="h-full"
            onCounts={setCounts}
            onFollowingChange={setFollowing}
            onStatusChange={setStatus}
          />
        )}
      </div>
    </section>
  );
}
