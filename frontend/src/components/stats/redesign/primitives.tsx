"use client";

// Building blocks of the redesigned Stats page (design section 3.0, fixed-size
// rules): every panel above the bottom Requests block has a fixed height; every
// varying text sits in a one-line slot; state chips are a fixed width.

import { type ReactNode, useEffect, useRef, useState } from "react";

import { FIXED, type ScopeState } from "@/lib/stats-redesign";
import { cn } from "@/lib/utils";

/** A section: uppercase label, the scope it shows, a note on the right. */
export function Section({
  title,
  scope,
  note,
  testid,
  children,
}: {
  title: string;
  scope?: string;
  note?: ReactNode;
  testid?: string;
  children: ReactNode;
}) {
  return (
    <section aria-label={title} data-testid={testid} className="mt-6">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="flex min-w-0 items-baseline gap-2 text-xs font-bold uppercase tracking-[0.1em] text-chat-fg">
          {title}
          {scope && (
            <span
              data-testid={testid ? `${testid}-scope` : undefined}
              className="max-w-[60ch] truncate rounded border border-chat-rule px-1.5 text-[10px] font-medium tracking-[0.06em] text-chat-muted"
              title={scope}
            >
              {scope}
            </span>
          )}
        </h2>
        {note && <span className="text-[11px] text-chat-dim">{note}</span>}
      </div>
      {children}
    </section>
  );
}

/** A bordered card of FIXED height (px). Content never grows it: overflow clips. */
export function FixedCard({
  height,
  phoneHeight,
  className,
  testid,
  title,
  onClick,
  children,
}: {
  height: number;
  /** Below 768 px, where the card's columns stack: still fixed, never data-driven. */
  phoneHeight?: number;
  className?: string;
  testid?: string;
  title?: string;
  onClick?: () => void;
  children: ReactNode;
}) {
  const style = phoneHeight
    ? ({ "--h": `${height}px`, "--hp": `${phoneHeight}px` } as React.CSSProperties)
    : { height };
  return (
    <div
      data-testid={testid}
      data-fixed-height={height}
      data-fixed-height-phone={phoneHeight ?? height}
      title={title}
      onClick={onClick}
      style={style}
      className={cn(
        "overflow-hidden rounded-lg border border-chat-rule bg-chat-surface p-3",
        phoneHeight && "h-[var(--hp)] md:h-[var(--h)]",
        onClick && "cursor-pointer",
        className,
      )}
    >
      {children}
    </div>
  );
}

/** One text line: never wraps, ellipsis, keeps its height when empty. */
export function Line({ className, children, title }: { className?: string; children?: ReactNode; title?: string }) {
  return (
    <div title={title} className={cn("h-[18px] overflow-hidden text-ellipsis whitespace-nowrap leading-[18px]", className)}>
      {children ?? " "}
    </div>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-hidden text-ellipsis whitespace-nowrap text-[10px] uppercase tracking-[0.07em] text-chat-dim">
      {children}
    </div>
  );
}

const CHIP_TONE: Record<ScopeState, string> = {
  Queueing: "border-amber-800 bg-amber-950/70 text-amber-300",
  Waiting: "border-amber-800 bg-amber-950/70 text-amber-300",
  Busy: "border-emerald-800 bg-emerald-950/70 text-emerald-300",
  Running: "border-emerald-800 bg-emerald-950/70 text-emerald-300",
  Idle: "border-chat-rule bg-chat-surface-2 text-chat-muted",
  Asleep: "border-chat-rule bg-chat-surface-2 text-chat-muted",
  "No data": "border-chat-rule bg-chat-surface-2 text-chat-dim",
};

/** Fixed-width state chip: "Idle" and "Queueing" take the same space. */
export function StateChip({ state }: { state: ScopeState }) {
  return (
    <span
      data-testid="state-chip"
      style={{ width: FIXED.chipW }}
      className={cn(
        "inline-block flex-none whitespace-nowrap rounded-full border py-px text-center text-[11px]",
        CHIP_TONE[state],
      )}
    >
      {state}
    </span>
  );
}

/** A segmented control (one choice). */
export function Seg<T extends string>({
  value,
  options,
  onChange,
  label,
  testid,
}: {
  value: T;
  options: readonly { v: T; label: string; title?: string }[];
  onChange: (v: T) => void;
  label: string;
  testid?: string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      data-testid={testid}
      className="inline-flex flex-none overflow-hidden rounded-md border border-chat-rule"
    >
      {options.map((o) => (
        <button
          key={o.v}
          type="button"
          title={o.title}
          aria-pressed={o.v === value}
          onClick={() => onChange(o.v)}
          className={cn(
            "whitespace-nowrap px-2.5 py-0.5 text-xs",
            o.v === value ? "bg-chat-surface-2 text-chat-accent" : "text-chat-muted hover:text-chat-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** A stable categorical colour for a key, model or session (the six --vw-model hues). */
export function hueOf(id: string | null | undefined): string {
  let h = 0;
  for (const c of id ?? "") h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return `rgb(var(--vw-model-${(h % 6) + 1}))`;
}

export function Swatch({ color }: { color: string }) {
  return <i aria-hidden="true" className="inline-block h-[9px] w-[9px] flex-none rounded-[2px]" style={{ background: color }} />;
}

/** The rendered width of an element (for SVGs drawn in real pixels, so a fixed
 *  chart height never scales its text or clips its axis). */
export function useWidth<T extends HTMLElement>(fallback = 800): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => {
      const next = Math.round(el.getBoundingClientRect().width);
      if (next > 0) setW(next);
    };
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}
