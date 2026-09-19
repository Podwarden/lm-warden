"use client";

// The history strip (spec §4.3; mockup "history strip + brush"): bars for the
// whole life of the key, the chain's earlier keys in a darker shade, a dashed
// marker per rotation, and an amber brush over the selected period. The brush
// math is the mockup's, in minutes.

import { useRef, type KeyboardEvent, type PointerEvent } from "react";
import type { SeriesBin, TokenSeries } from "@/lib/token-series";
import { day, fmtWhen, hhmm } from "@/lib/token-format";
import { useElementWidth } from "@/lib/use-element-width";
import { SERIES } from "./styles";

export interface StripWindow {
  from: number;
  to: number;
}

export const MIN_WINDOW_MIN = 30;
const H = 54;

export interface HistoryStripProps {
  startSec: number;
  endSec: number;
  chain: TokenSeries | null;
  own: TokenSeries | null;
  showChain: boolean;
  rotations: number[];
  window: StripWindow;
  onWindowChange: (w: StripWindow) => void;
}

const tokensPerMin = (b: SeriesBin) => b.prompt_per_min + b.completion_per_min;

/** The mockup's strip has `nb` equal buckets over `[sFrom, sTo)` (minutes),
 *  whatever the server's bin width. Each server bin's tokens are spread over
 *  the buckets it overlaps, pro rata; returns tokens per bucket. */
export function stripBuckets(bins: SeriesBin[], binMin: number, sFrom: number, sTo: number, nb: number): number[] {
  const out: number[] = new Array(Math.max(0, nb)).fill(0);
  const span = sTo - sFrom;
  if (span <= 0 || nb <= 0 || binMin <= 0) return out;
  const bw = span / nb;
  for (const b of bins) {
    const perMin = tokensPerMin(b);
    const a = Math.max(b.minute, sFrom);
    const e = Math.min(b.minute + binMin, sTo);
    if (!perMin || e <= a) continue;
    for (let k = Math.floor((a - sFrom) / bw); k < nb; k++) {
      const ka = sFrom + k * bw;
      if (ka >= e) break;
      const ov = Math.min(e, ka + bw) - Math.max(a, ka);
      if (ov > 0) out[k] += perMin * ov;
    }
  }
  return out;
}

/** Axis label for the strip. A key younger than a couple of days would read
 *  "Sep 18 … Sep 18" with dates alone, so up to 2 days the strip uses the
 *  charts' span-aware format (HH:MM within a day, "Mon D HH:MM" beyond);
 *  longer strips keep the mockup's plain dates. */
export function stripLabel(sec: number, spanMinutes: number): string {
  return spanMinutes <= 2 * 1440 ? fmtWhen(sec, spanMinutes) : day(sec);
}

export function HistoryStrip({
  startSec, endSec, chain, own, showChain, rotations, window: win, onWindowChange,
}: HistoryStripProps) {
  const ref = useRef<HTMLDivElement>(null);
  const W = useElementWidth(ref);

  const S_FROM = startSec / 60;
  const S_TO = endSec / 60;
  const span = Math.max(1, S_TO - S_FROM);
  const x = (m: number) => ((m - S_FROM) / span) * W;

  // Mockup drawStrip(): max(20, floor(W / 4)) bars, 1 px apart when wider than 3 px.
  const nb = Math.max(20, Math.floor(W / 4));
  const ownVals = own ? stripBuckets(own.bins, own.bin_minutes, S_FROM, S_TO, nb) : [];
  const chainVals = chain ? stripBuckets(chain.bins, chain.bin_minutes, S_FROM, S_TO, nb) : [];
  // Scale against the chain's peak so switching the toggle never rescales.
  const mx = Math.max(1e-9, ...chainVals, ...ownVals);
  const bw = W / nb;
  const gap = bw > 3 ? 1 : 0;
  const barW = Math.max(bw - gap, 0.5);

  // Both edges are clamped to the strip. A preset window's `to` is the page's
  // unfloored "now", while the strip ends at the server's `to_minute`, so on
  // a young key (a few px per second) the brush would otherwise hang past the
  // right edge by up to a minute (#251). Otherwise the mockup's math: left at
  // `from`, at least 6 px wide — shifted left only if that would overhang.
  const l0 = x(Math.max(win.from / 60, S_FROM));
  const width = Math.max(x(Math.min(win.to / 60, S_TO)) - l0, 6);
  const left = Math.max(0, Math.min(l0, W - width));

  const drag = useRef<{ mode: "move" | "l" | "r"; x0: number; from: number; to: number } | null>(null);

  function onPointerDown(e: PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const edge = (e.target as HTMLElement).dataset.edge;
    drag.current = { mode: edge === "l" || edge === "r" ? edge : "move", x0: e.clientX, from: win.from / 60, to: win.to / 60 };
  }

  function onPointerMove(e: PointerEvent<HTMLDivElement>) {
    const d = drag.current;
    if (!d || W <= 0) return;
    const dm = Math.round((e.clientX - d.x0) * (span / W));
    let { from, to } = d;
    if (d.mode === "move") {
      const len = to - from;
      from = Math.min(Math.max(from + dm, S_FROM), S_TO - len);
      to = from + len;
    } else if (d.mode === "l") {
      from = Math.min(Math.max(from + dm, S_FROM), to - MIN_WINDOW_MIN);
    } else {
      to = Math.max(Math.min(to + dm, S_TO), from + MIN_WINDOW_MIN);
    }
    onWindowChange({ from: from * 60, to: to * 60 });
  }

  function endDrag() {
    drag.current = null;
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    let from = win.from / 60;
    let to = win.to / 60;
    const step = Math.max(10, Math.round((to - from) / 10));
    const dir = e.key === "ArrowLeft" ? -1 : 1;
    if (e.shiftKey) {
      from = Math.min(Math.max(from - dir * step, S_FROM), to - MIN_WINDOW_MIN);
    } else {
      const len = to - from;
      from = Math.min(Math.max(from + dir * step, S_FROM), S_TO - len);
      to = from + len;
    }
    onWindowChange({ from: from * 60, to: to * 60 });
  }

  const winEnd = Math.ceil(win.to / 60) * 60; // as usage-section's windowText
  const rangeText = `${day(win.from)} ${hhmm(win.from)} to ${day(winEnd)} ${hhmm(winEnd)}`;

  return (
    <>
      <div ref={ref} className="relative h-[54px] touch-none">
        <svg className="block h-full w-full" viewBox={`0 0 ${W || 1} ${H}`} preserveAspectRatio="none" aria-hidden>
          {chainVals.map((v, k) => {
            if (!v) return null;
            const h = (v / mx) * (H - 4);
            return (
              <rect key={`c${k}`} x={k * bw} y={H - h} width={barW} height={h} rx={1}
                fill={showChain ? SERIES.stripChain : "transparent"} />
            );
          })}
          {ownVals.map((v, k) => {
            if (!v) return null;
            const h = (v / mx) * (H - 4);
            return <rect key={`o${k}`} x={k * bw} y={H - h} width={barW} height={h} rx={1} fill={SERIES.stripOwn} />;
          })}
          {rotations.map((t) => {
            const rx = x(t / 60);
            return (
              <g key={t}>
                <line x1={rx} x2={rx} y1={0} y2={H} stroke={SERIES.muted} strokeDasharray="2 3" opacity={0.7} />
                <text x={rx + 5} y={11} fill={SERIES.muted} fontSize={10.5}>rotated</text>
              </g>
            );
          })}
        </svg>
        <div
          role="slider"
          tabIndex={0}
          aria-label="Selected period. Arrow keys move it, Shift+arrows resize it."
          aria-valuemin={Math.round(S_FROM * 60)}
          aria-valuemax={Math.round(S_TO * 60)}
          aria-valuenow={Math.round(win.from)}
          aria-valuetext={rangeText}
          className="absolute bottom-0 top-0 cursor-grab rounded border-[1.5px] border-chat-accent bg-chat-accent/10 active:cursor-grabbing"
          style={{ left, width }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onKeyDown={onKeyDown}
        >
          <div data-edge="l" className="absolute -left-1.5 top-1/2 -mt-[13px] h-[26px] w-2.5 cursor-ew-resize rounded-[3px] bg-chat-accent" />
          <div data-edge="r" className="absolute -right-1.5 top-1/2 -mt-[13px] h-[26px] w-2.5 cursor-ew-resize rounded-[3px] bg-chat-accent" />
        </div>
      </div>
      <div className="mt-1 flex justify-between text-[11px] tabular-nums text-chat-dim">
        {[0, 1, 2, 3, 4].map((i) => (
          <span key={i}>{stripLabel((S_FROM + (span / 4) * i) * 60, span)}</span>
        ))}
      </div>
    </>
  );
}
