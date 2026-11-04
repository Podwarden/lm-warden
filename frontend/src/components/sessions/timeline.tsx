"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

import { LINK_STYLE, svgFill } from "@/lib/sessions/colours";
import { type Block, LABEL_W, layoutTimeline, type Selection } from "@/lib/sessions/layout";
import { S } from "@/lib/sessions/strings";
import type { SessionResponse } from "@/lib/sessions/types";

const LANE_H = 22;
const MAX_LANES = 10;

const Rect = memo(function Rect({ b, selected }: { b: Block; selected: boolean }) {
  return (
    <g>
      <rect
        data-req={b.requestId}
        data-selected={selected ? "true" : undefined}
        x={b.x}
        y={b.y}
        width={b.w}
        height={b.h}
        // A relayed block takes the hatch, never a share colour: a fill class
        // would override the SVG fill attribute (CSS beats presentation attrs).
        className={`${b.relayed ? "" : svgFill(b.fill)} cursor-pointer ${selected ? "stroke-white" : ""}`}
        strokeWidth={selected ? 1.5 : 0}
        fill={b.relayed ? "url(#relayed-hatch)" : undefined}
      />
      {b.evicted && (
        <path
          data-notch
          data-req={b.requestId}
          d={`M ${b.x} ${b.y} h ${Math.min(5, b.w)} L ${b.x} ${b.y + 5} Z`}
          className="fill-chat-page cursor-pointer"
        />
      )}
      {b.stripe && (
        <rect
          data-req={b.requestId}
          x={b.x}
          y={b.y + b.h - 3}
          width={b.w}
          height={3}
          className={`${svgFill(b.stripe)} cursor-pointer`}
        />
      )}
    </g>
  );
});

export function Timeline({
  s,
  selection,
  onBlock,
  onLane,
}: {
  s: SessionResponse;
  selection: Selection;
  onBlock: (requestId: string, conversationId: string | null) => void;
  onLane: (conversationId: string | null) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1000);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setWidth(Math.max(400, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const L = useMemo(
    () => layoutTimeline(s, { width, laneH: LANE_H, maxLanes: MAX_LANES, expanded }),
    [s, width, expanded],
  );
  const convOf = useMemo(() => new Map(L.blocks.map((b) => [b.requestId, b.conversationId])), [L]);
  const height = L.lanes.length * LANE_H + 18;

  useEffect(() => {
    if (!selection.requestId || !wrap.current) return;
    const el = wrap.current.querySelector(`rect[data-req="${CSS.escape(selection.requestId)}"]`);
    if (el && "scrollIntoView" in el) (el as Element).scrollIntoView?.({ block: "nearest", inline: "center" });
  }, [selection.requestId]);

  function click(e: React.MouseEvent<SVGSVGElement>) {
    const t = e.target as Element;
    const req = t.getAttribute("data-req");
    if (req) {
      onBlock(req, convOf.get(req) ?? null);
      return;
    }
    const lane = t.getAttribute("data-lane");
    if (lane !== null) onLane(lane === "" ? null : lane);
  }

  return (
    <section
      data-testid="session-timeline"
      className="rounded-lg border border-chat-rule bg-chat-surface/50 p-3"
      aria-label={S.timeline.title}
    >
      <div ref={wrap} className="overflow-x-auto">
        <svg width={width} height={height} onClick={click} role="img" aria-label={S.timeline.title}>
          <defs>
            <pattern id="relayed-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" className="fill-slate-500" />
              <line x1="0" y1="0" x2="0" y2="6" className="stroke-slate-300" strokeWidth="2" />
            </pattern>
          </defs>
          {L.lanes.map((l) => {
            const sel = selection.conversationId === l.conversationId && selection.conversationId !== undefined;
            return (
              <g key={String(l.conversationId)}>
                <rect
                  data-lane={l.conversationId ?? ""}
                  x={0}
                  y={l.y}
                  width={width}
                  height={LANE_H}
                  className={sel ? "fill-slate-700/40 cursor-pointer" : "fill-transparent cursor-pointer"}
                />
                <text data-lane={l.conversationId ?? ""} x={4} y={l.y + LANE_H * 0.7} className="fill-current text-xs cursor-pointer">
                  {l.label.length > 20 ? l.label.slice(0, 19) + "…" : l.label}
                </text>
              </g>
            );
          })}
          {L.links.map((k) => (
            <g key={`${k.from}-${k.to}`}>
              <path
                d={k.d}
                fill="none"
                className={LINK_STYLE[k.kind].stroke}
                strokeDasharray={LINK_STYLE[k.kind].dash}
                strokeWidth={1}
              />
              {k.kind === "inferred" && (
                <text x={k.labelX} y={k.labelY} className="fill-amber-500 text-[9px]">
                  {S.timeline.inferred}
                </text>
              )}
            </g>
          ))}
          {L.blocks.map((b) => (
            <Rect key={b.requestId} b={b} selected={selection.requestId === b.requestId} />
          ))}
          {L.ticks.map((t) => (
            <text key={t.x} x={Math.max(LABEL_W, t.x - 20)} y={height - 4} className="fill-current text-[10px] opacity-60">
              {new Date(t.t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </text>
          ))}
        </svg>
      </div>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-xs text-chat-muted">
        <span>{S.timeline.legend}</span>
        {L.hidden > 0 && (
          <button type="button" className="underline" onClick={() => setExpanded(true)}>
            {S.timeline.moreLanes(L.hidden)}
          </button>
        )}
        {expanded && (
          <button type="button" className="underline" onClick={() => setExpanded(false)}>
            {S.timeline.fewerLanes}
          </button>
        )}
      </div>
    </section>
  );
}
