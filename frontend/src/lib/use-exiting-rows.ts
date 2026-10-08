// Keep rows that vanished from a poll on screen just long enough to animate
// out, and flag rows that just appeared so they can animate in.
//
// The in-flight table's data is replaced wholesale every 1.5 s. Without this a
// finished request blinks out and every row below jumps; with it the operator
// can keep their eyes on a surviving row while a neighbour collapses.
//
// The merge is derived during render from the last COMMITTED output (a ref
// written in an effect), so a row that disappears is never absent from even a
// single frame; timers only decide when an exiting row is finally dropped.

import { useEffect, useMemo, useRef, useState } from "react";

export type RowState = "entering" | "live" | "exiting";

export interface AnimatedRow<T> {
  key: string;
  row: T;
  state: RowState;
}

/** How long an `entering` row stays collapsed before expanding (about a frame). */
export const ENTER_FLIP_MS = 30;

export function prefersReducedMotion(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  } catch {
    return false;
  }
}

export function useExitingRows<T>(
  rows: readonly T[],
  keyOf: (row: T) => string,
  exitMs: number,
): AnimatedRow<T>[] {
  const reduced = prefersReducedMotion();
  const committed = useRef<AnimatedRow<T>[]>([]);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const mounted = useRef(false);
  const [tick, setTick] = useState(0);

  const items = useMemo<AnimatedRow<T>[]>(() => {
    const live = rows.map((row) => ({ row, key: keyOf(row) }));
    if (reduced || exitMs <= 0) {
      return live.map((l) => ({ ...l, state: "live" as const }));
    }
    const before = committed.current;
    const beforeState = new Map(before.map((b) => [b.key, b.state]));
    const liveKeys = new Set(live.map((l) => l.key));
    const out: AnimatedRow<T>[] = live.map((l) => {
      const prev = beforeState.get(l.key);
      // A key we have never rendered is new -- except on the very first paint,
      // when everything is simply "already there".
      const state: RowState = prev === undefined ? (mounted.current ? "entering" : "live") : prev === "exiting" ? "live" : prev;
      return { ...l, state };
    });
    // Re-insert rows that left, right after the row that used to precede them.
    let anchor = -1;
    for (const b of before) {
      const at = out.findIndex((o) => o.key === b.key);
      if (at >= 0 && liveKeys.has(b.key)) {
        anchor = at;
        continue;
      }
      out.splice(anchor + 1, 0, { ...b, state: "exiting" });
      anchor += 1;
    }
    return out;
    // `tick` re-runs the merge after a timer fired; keyOf is stable by contract.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, exitMs, reduced, tick]);

  useEffect(() => {
    mounted.current = true;
    committed.current = items;
    const t = timers.current;
    const present = new Set(items.map((i) => i.key));
    // A row that came back, or one that was dropped, no longer owns a timer.
    for (const [key, handle] of t) {
      const stillExiting = items.some((i) => i.key === key && i.state === "exiting");
      if (!stillExiting || !present.has(key)) {
        clearTimeout(handle);
        t.delete(key);
      }
    }
    for (const it of items) {
      if (it.state === "exiting" && !t.has(it.key)) {
        t.set(
          it.key,
          setTimeout(() => {
            t.delete(it.key);
            committed.current = committed.current.filter((c) => c.key !== it.key);
            setTick((n) => n + 1);
          }, exitMs),
        );
      }
    }
    if (items.some((i) => i.state === "entering")) {
      const handle = setTimeout(() => {
        committed.current = committed.current.map((c) =>
          c.state === "entering" ? { ...c, state: "live" as const } : c,
        );
        setTick((n) => n + 1);
      }, ENTER_FLIP_MS);
      return () => clearTimeout(handle);
    }
  }, [items, exitMs]);

  useEffect(
    () => () => {
      for (const h of timers.current.values()) clearTimeout(h);
      timers.current.clear();
    },
    [],
  );

  return items;
}
