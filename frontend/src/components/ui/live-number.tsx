"use client";

// A live counter (chars, tokens, rates, durations, counts) whose width never
// jitters, so the text after it never moves (owner request 2026-11-23; shared
// convention ui.live-numbers in podwarden-shared-conventions).
//
//  * Tabular figures (the `num-live` class in globals.css): every digit is
//    as wide as every other, so 1,111 and 8,888 take the same room.
//  * A reserved slot, right-aligned: `reserve` is the widest value expected,
//    in characters (`ch`, the width of "0"). A value wider than that widens
//    the slot once and the slot never narrows again (99 → 100 → 99 keeps the
//    three-digit width), until `resetKey` changes (another request). The
//    reserve counts toward that too: a counter that stops being live and
//    drops its reserve keeps its slot.
//
// Text after the counter goes OUTSIDE it ("<LiveNumber>1,234</LiveNumber>
// chars"), never inside the same string. Units that change with the value
// ("59 s" → "1 min 00 s") should be reserved for their widest form.

import { useRef, type HTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/utils";

const grouped = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** The slot, in characters, that holds `sample` (the widest expected value). */
export function liveWidth(sample: string): number {
  return sample.length;
}

export interface LiveNumberProps extends Omit<HTMLAttributes<HTMLSpanElement>, "children"> {
  /** The value already formatted; or pass `value` to format a number. */
  children?: ReactNode;
  /** A number, shown with thousands separators (no decimals). */
  value?: number;
  /** The widest value expected, in characters. */
  reserve?: number;
  /** A change starts the slot over (e.g. another request is shown). */
  resetKey?: unknown;
  /** "right" (default): digits grow leftwards against what follows (a unit).
   *  "left": the value sits right after its label and the slot's spare room
   *  is after it (label: value pairs in a row of facts). Either way the
   *  slot's width is fixed, so what follows the slot never moves. */
  align?: "right" | "left";
}

export function LiveNumber({ children, value, reserve = 0, resetKey, align = "right", className, style, ...rest }: LiveNumberProps) {
  const text = value !== undefined ? grouped.format(value) : children;
  const len = typeof text === "string" || typeof text === "number" ? String(text).length : 0;
  const widest = useRef<{ key: unknown; n: number }>({ key: resetKey, n: 0 });
  if (widest.current.key !== resetKey) widest.current = { key: resetKey, n: 0 };
  // The reserve counts toward the slot too: a counter that stops being live
  // (and drops its reserve) keeps its width while the same thing is shown.
  widest.current.n = Math.max(widest.current.n, len, reserve);
  const ch = widest.current.n;
  return (
    <span
      data-live-number=""
      {...rest}
      className={cn("num-live", className)}
      style={{ ...style, ...(ch > 0 ? { minWidth: `${ch}ch` } : null), ...(align === "left" ? { textAlign: "left" } : null) }}
    >
      {text}
    </span>
  );
}
