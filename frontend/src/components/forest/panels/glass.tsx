"use client";

// Shared pieces of the forest full window's panels (spec §6.4): the glass
// surface and a collapsible <details> whose open state the browser remembers.

import { forwardRef, useCallback, useState } from "react";
import { cn } from "@/lib/utils";

/** The glass surface of every panel: rgba(14,18,16,.55) with a 10 px blur. */
export const GLASS =
  "rounded-[10px] border border-white/10 bg-[rgba(14,18,16,.55)] text-[#eef2ec] backdrop-blur-[10px]";

/** Below this window width the description and legend start closed. */
export const NARROW_PX = 1100;

function readBool(key: string): boolean | null {
  try {
    const v = localStorage.getItem(key);
    return v === "true" ? true : v === "false" ? false : null;
  } catch {
    return null;
  }
}

function writeBool(key: string, v: boolean): void {
  try {
    localStorage.setItem(key, String(v));
  } catch {
    /* private window, blocked storage: the state just isn't remembered */
  }
}

/** Open state kept in localStorage under `key`; `fallback` when nothing is stored (or storage is unavailable). */
export function usePersistedOpen(key: string, fallback: () => boolean): [boolean, (open: boolean) => void] {
  const [open, setOpenState] = useState<boolean>(() => readBool(key) ?? fallback());
  const setOpen = useCallback(
    (v: boolean) => {
      setOpenState(v);
      writeBool(key, v);
    },
    [key],
  );
  return [open, setOpen];
}

/** Default for the left-hand panels: open on a wide window, closed below 1100 px. */
export const openWhenWide = () => (typeof window === "undefined" ? true : window.innerWidth >= NARROW_PX);

/**
 * A glass <details>. The open state is React's (the summary click is handled here, not by the browser), so it is
 * remembered and stays in step with what is drawn.
 */
export const GlassDetails = forwardRef<
  HTMLDetailsElement,
  {
    title: string;
    storageKey: string;
    defaultOpen: () => boolean;
    note?: React.ReactNode;
    className?: string;
    bodyClassName?: string;
    testid?: string;
    children: React.ReactNode;
  }
>(function GlassDetails({ title, storageKey, defaultOpen, note, className, bodyClassName, testid, children }, ref) {
  const [open, setOpen] = usePersistedOpen(storageKey, defaultOpen);
  return (
    <details ref={ref} open={open} data-testid={testid} className={cn(GLASS, "group px-3 py-2 text-xs", className)}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          setOpen(!open);
        }}
        className="flex cursor-pointer list-none items-center justify-between gap-2 text-[11px] font-semibold uppercase tracking-[0.08em] [&::-webkit-details-marker]:hidden"
      >
        <span>{title}</span>
        <span className="flex items-center gap-2 font-normal normal-case tracking-normal text-white/60">
          {note}
          <span aria-hidden="true">{open ? "▾" : "▸"}</span>
        </span>
      </summary>
      <div className={cn("mt-2", bodyClassName)}>{children}</div>
    </details>
  );
});
