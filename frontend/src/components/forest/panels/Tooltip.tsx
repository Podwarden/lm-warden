"use client";

// Hover detail on the forest canvas (spec §1 "hover detail", README §4; final review I4). ForestView picks what is
// under the pointer (`scene.pick`, at most 20 times a second) and shows it here: for a turn of a tree its key, session,
// local time, tokens and tool families; for a ghost when it was planted and moved; for a flower or mushroom its time and
// tokens. Styled like the app's chart tooltip (components/tokens/detail/time-chart.tsx); kept inside the viewport and
// off the open panels.

import { forwardRef } from "react";
import type { PickResult } from "@/components/forest/engine/scene";
import { quietLabel } from "@/lib/forest/timeline";
import type { ForestState, Session, Turn } from "@/lib/forest/types";

export interface TipContent {
  title: string;
  sub?: string;
  rows: { name: string; value: string }[];
}

export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** The pointer's offset from the tooltip's corner (px). */
const GAP = 14;

/**
 * Where to put a `size` tooltip for a pointer at `p`: below right of it, else below left, above right, above left,
 * whichever first stays clear of every panel; always clamped inside the viewport.
 */
export function tooltipPosition(
  p: { x: number; y: number },
  size: { w: number; h: number },
  viewport: { width: number; height: number },
  panels: readonly Rect[],
): { left: number; top: number } {
  const clamp = (v: number, max: number) => Math.max(0, Math.min(v, Math.max(0, max)));
  const at = (dx: -1 | 1, dy: -1 | 1) => ({
    left: clamp(dx > 0 ? p.x + GAP : p.x - GAP - size.w, viewport.width - size.w),
    top: clamp(dy > 0 ? p.y + GAP : p.y - GAP - size.h, viewport.height - size.h),
  });
  const clear = (c: { left: number; top: number }) =>
    panels.every((r) => c.left >= r.right || c.left + size.w <= r.left || c.top >= r.bottom || c.top + size.h <= r.top);
  const tries = [at(1, 1), at(-1, 1), at(1, -1), at(-1, -1)];
  return tries.find(clear) ?? tries[0];
}

const num = (n: number) => n.toLocaleString("en-US");
const clock = (abs: number) => new Date(abs * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function findSession(sessions: readonly Session[], id: string): Session | null {
  for (const s of sessions) {
    if (s.id === id) return s;
    const c = findSession(s.children, id);
    if (c) return c;
  }
  return null;
}

/** What the tooltip says about a pick (times absolute). `state`: the view's data, for the turn's tokens and tools. */
export function describePick(r: PickResult, state: ForestState | null): TipContent | null {
  switch (r.kind) {
    case "turn": {
      const tree = state?.trees.get(r.treeId);
      const t: Turn | undefined = tree ? findSession(tree.sessions, r.sessionId)?.turns[r.turn] : undefined;
      const rows = [{ name: "time", value: clock(r.time) }];
      if (t) {
        rows.push({ name: "context", value: num(t[1]) }, { name: "generated", value: num(t[2]) });
        rows.push({ name: "cached", value: t[3] === null ? "—" : num(t[3]) });
        const tools = [...new Set([...t[4].map((x) => x[0]), ...t[5]])].sort();
        if (tools.length) rows.push({ name: "tools", value: tools.join(", ") });
        if (t[7] > 1) rows.push({ name: "requests", value: num(t[7]) });
      }
      return { title: tree?.key ?? "Session", sub: `session ${r.sessionId.slice(0, 8)}`, rows };
    }
    case "ghost":
      return { title: "Former spot", rows: [{ name: "planted", value: clock(r.plantedAt) }, { name: "moved", value: clock(r.movedAt) }] };
    case "flower":
      return { title: "Flower", sub: "a one-shot question and answer", rows: [{ name: "time", value: clock(r.time) }, { name: "prompt", value: num(r.ctx) }, { name: "answer", value: num(r.gen) }] };
    case "quiet":
      return { title: quietLabel(r.to - r.time), sub: "no traffic: shortened on the spit", rows: [{ name: "from", value: clock(r.time) }, { name: "to", value: clock(r.to) }] };
    case "mushroom":
      return { title: "Mushroom", sub: "an embeddings call", rows: [{ name: "time", value: clock(r.time) }, { name: "prompt", value: num(r.ctx) }] };
    default:
      return null;
  }
}

export const ForestTooltip = forwardRef<HTMLDivElement, { tip: TipContent; left: number; top: number }>(function ForestTooltip(
  { tip, left, top },
  ref,
) {
  return (
    <div
      ref={ref}
      role="tooltip"
      data-testid="forest-tooltip"
      className="pointer-events-none fixed z-[60] whitespace-nowrap rounded-md border border-chat-rule bg-chat-page px-2.5 py-[7px] text-[12px] tabular-nums shadow-[0_6px_18px_rgba(0,0,0,.35)]"
      style={{ left, top }}
    >
      <div className="font-medium">{tip.title}</div>
      {tip.sub && <div className="mb-[3px] text-chat-muted">{tip.sub}</div>}
      {tip.rows.map((r) => (
        <div key={r.name} className="flex justify-between gap-4">
          <span className="text-chat-muted">{r.name}</span>
          <span>{r.value}</span>
        </div>
      ))}
    </div>
  );
});
