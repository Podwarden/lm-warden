"use client";

// Sessions started per hour, with the light top for the share over 10% dead weight. Dragging selects a time span
// (snapped to whole hours); a double click or "Clear selection" drops it.

import { useRef, useState } from "react";

import { barLayout, brushToRange, rangeToBrush } from "@/lib/sessions/histogram";
import { S } from "@/lib/sessions/strings";
import type { Histogram } from "@/lib/sessions/types";

import { fmtStart } from "./format";

const W = 1000;
const H = 70;

export interface Brush {
  since: number;
  until: number;
}

export function ActivityChart({
  h,
  brush,
  onBrush,
  rangeLabel,
}: {
  h: Histogram;
  brush: Brush | null;
  onBrush: (b: Brush | null) => void;
  rangeLabel: string;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const start = useRef<number | null>(null);
  const [drag, setDrag] = useState<{ x0: number; x1: number } | null>(null);
  const bars = barLayout(h, W, H);
  const total = h.bins.reduce((s, b) => s + b.sessions, 0);
  const gap = bars.length > 0 ? Math.min(2, bars[0].w * 0.2) : 0;

  function xOf(clientX: number): number | null {
    const r = ref.current?.getBoundingClientRect();
    if (!r || r.width <= 0) return null;
    return ((clientX - r.left) / r.width) * W;
  }

  function commit(clientX: number) {
    const x0 = start.current;
    start.current = null;
    setDrag(null);
    if (x0 === null) return;
    onBrush(brushToRange(x0, xOf(clientX) ?? x0, h, W));
  }

  const box = drag
    ? { x0: Math.min(drag.x0, drag.x1), x1: Math.max(drag.x0, drag.x1) }
    : brush
      ? rangeToBrush(brush, h, W)
      : null;
  const end = h.start + h.bins.length * h.bin_s;

  return (
    <div className="px-4 pb-2 pt-3">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <b className="text-[15px] font-semibold text-chat-fg">{S.list.chartHead(total, rangeLabel)}</b>
        <span className="text-xs text-chat-muted">{S.list.chartHint}</span>
      </div>
      <svg
        ref={ref}
        role="img"
        aria-label={S.list.chartAria}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="mt-2.5 block h-[70px] w-full cursor-crosshair select-none"
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          const x = xOf(e.clientX);
          if (x === null) return;
          start.current = x;
          setDrag({ x0: x, x1: x });
        }}
        onMouseMove={(e) => {
          if (start.current === null) return;
          const x = xOf(e.clientX);
          if (x !== null) setDrag({ x0: start.current, x1: x });
        }}
        onMouseUp={(e) => commit(e.clientX)}
        onMouseLeave={(e) => {
          if (start.current !== null) commit(e.clientX);
        }}
        onDoubleClick={() => onBrush(null)}
      >
        {bars.map((b) => (
          <g key={b.t}>
            <rect x={b.x + gap / 2} y={H - b.h} width={Math.max(0, b.w - gap)} height={b.h} fill="#5a4c3c" rx={1} />
            {b.hTop > 0 && (
              <rect x={b.x + gap / 2} y={H - b.h} width={Math.max(0, b.w - gap)} height={b.hTop} fill="#b7864f" rx={1} />
            )}
          </g>
        ))}
        {box && box.x1 > box.x0 && (
          <rect
            data-testid="chart-brush"
            x={box.x0}
            y={0}
            width={box.x1 - box.x0}
            height={H}
            fill="rgba(224,169,109,0.10)"
            stroke="#e0a96d"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
      <div className="mt-1 flex items-center justify-between gap-3 text-[11px] text-chat-dim">
        <span>{fmtStart(h.start)}</span>
        {brush && (
          <button
            type="button"
            onClick={() => onBrush(null)}
            className="rounded px-1.5 text-xs text-chat-accent hover:underline focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
          >
            {S.list.clearSelection}
          </button>
        )}
        <span>{fmtStart(end)}</span>
      </div>
    </div>
  );
}
