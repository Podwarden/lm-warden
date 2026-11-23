"use client";

// Prompt Lens prompt map (spec 2026-11-22 section 3.1, mockup states 3-7): one
// bar segment per part (flex width = its tokens), brackets above (head = sys +
// tools, nodes = the body) and below (the cached span; for the admin also the
// span another key cached, dashed), and a legend. The map has a fixed height
// in every state (empty, pending, finished), so the request screen does not
// move when the parts or the end of the request arrive.
//
// What a scope may see is decided in `mapParts` (lib/prompt-lens/kvsig.ts); this
// file draws the view it is given. A key holder's view never has an `other`
// part, and the legend draws the other-key swatch for the admin only.

import type { CSSProperties } from "react";

import type { KvsigState } from "@/lib/prompt-lens/events";
import type { KvsigView, MapPart } from "@/lib/prompt-lens/kvsig";
import { cn } from "@/lib/utils";

/** Part colours by state. Complete literals for Tailwind's scanner. */
export const STATE_CLASS: Record<KvsigState, string> = {
  exact: "bg-vw-live text-chat-page",
  loose: "text-chat-page",
  new: "bg-vw-strip-chain text-chat-fg",
  other: "bg-vw-info-fg text-chat-page",
};

const LOOSE_BG = "repeating-linear-gradient(135deg, rgb(var(--vw-queue)) 0 5px, rgb(var(--vw-queue) / .6) 5px 10px)";

/** The inline style a state needs beyond its class (the loose stripes). */
export function stateStyle(state: KvsigState): CSSProperties | undefined {
  return state === "loose" ? { background: LOOSE_BG } : undefined;
}

export function Swatch({ state }: { state: KvsigState }) {
  return (
    <i
      aria-hidden
      data-testid="kvsig-swatch"
      className={cn("inline-block h-2.5 w-2.5 flex-none rounded-[2px]", STATE_CLASS[state])}
      style={stateStyle(state)}
    />
  );
}

const LEGEND: [KvsigState, string][] = [
  ["exact", "exact match: same bytes as an earlier request"],
  ["loose", "loose match only: same after ignoring noise"],
];
/** The "new" legend entry. The admin knows the engine computed it; a key
 *  holder cannot tell computed from another key's cache, so its legend only
 *  says what it knows (G-R11). */
const NEW_TEXT = { admin: "new: computed for this request", key: "new: no earlier request matches" } as const;

const fmt = (n: number) => n.toLocaleString("en-US");
const pct = (n: number, total: number) => `${total > 0 ? Math.min(100, (n / total) * 100) : 0}%`;

/** Map, legend and empty state: the fixed-size block. Its height (header
 *  aside) is the same with or without parts. On a phone the segment bar is
 *  44 px (a touch target) instead of 26, so the block is 18 px taller. */
const MAP_H = "h-[96px]";
const MAP_H_PHONE = "h-[114px]";

export interface KvsigMapProps {
  view: KvsigView;
  onSegment: (p: MapPart) => void;
  /** The highlighted part's index. */
  active?: number | null;
  /** Phone: 44 px tall segments. */
  phone?: boolean;
}

export function KvsigMap({ view, onSegment, active = null, phone = false }: KvsigMapProps) {
  const mapH = phone ? MAP_H_PHONE : MAP_H;
  const isKey = view.scope === "key";
  const total = view.totalTok;
  const headTok = view.parts.filter((p) => p.p !== "nodes").reduce((a, p) => a + p.tok, 0);
  const nodesTok = total - headTok;
  return (
    <section data-testid="kvsig-map" aria-label="kvsig · prompt map" className="min-w-0">
      <div className="flex min-w-0 items-baseline gap-2">
        <h4 className="flex-none text-[13px] font-semibold text-chat-fg">kvsig · prompt map</h4>
        <span className="min-w-0 truncate text-[11px] text-chat-dim">
          {isKey
            ? "which parts of this prompt match your earlier requests"
            : "which parts match this session's earlier requests, and what the engine served from cache"}
        </span>
        {view.empty === null && (
          <span className="ml-auto flex-none text-[11px] tabular-nums text-chat-muted">{fmt(total)} tok</span>
        )}
      </div>

      {view.empty !== null ? (
        <div data-testid="kvsig-empty" className={cn(mapH, "flex items-center text-[12px] text-chat-dim")}>
          {view.empty}
        </div>
      ) : (
        <>
          <div className={cn(mapH, "relative min-w-0 pt-[22px]")}>
            {headTok > 0 && (
              <Bracket left="0%" width={pct(headTok, total)} label="head = sys + tools" />
            )}
            {nodesTok > 0 && (
              <Bracket
                left={pct(headTok, total)}
                width={pct(nodesTok, total)}
                label="nodes: the body, chained message by message"
              />
            )}
            <div data-testid="kvsig-bar" className={cn("flex min-w-0 gap-px overflow-hidden rounded bg-chat-page", phone ? "h-11" : "h-[26px]")}>
              {view.parts.map((p, i) => (
                <button
                  key={`${p.p}:${p.from}`}
                  type="button"
                  data-testid="kvsig-seg"
                  data-state={p.state}
                  aria-label={`${p.label}: ${p.covers}`}
                  title={`${p.label}: ${p.covers}`}
                  onClick={() => onSegment(p)}
                  className={cn(
                    `seg-${p.state}`,
                    "flex h-full min-w-[3px] cursor-pointer items-center justify-center overflow-hidden whitespace-nowrap border-0 p-0 text-[10px] font-semibold",
                    STATE_CLASS[p.state],
                    "focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-chat-accent",
                    active === i && "outline outline-2 -outline-offset-2 outline-chat-fg",
                  )}
                  style={{ flexGrow: p.tok, flexShrink: 0, flexBasis: 0, ...stateStyle(p.state) }}
                >
                  {total > 0 && p.tok / total > 0.06 ? p.label : ""}
                </button>
              ))}
            </div>
            <div data-testid="kvsig-under" className="relative mt-1 h-[44px] min-w-0">
              {view.cachedSpanTok !== null && (
                <UnderBracket row={0} width={pct(view.cachedSpanTok, total)} label={view.cachedSpanLabel} />
              )}
              {view.otherSpanTok !== null && (
                <UnderBracket row={1} dashed width={pct(view.otherSpanTok, total)} label={view.otherSpanLabel} />
              )}
            </div>
          </div>
          <div data-testid="kvsig-legend" className="mb-2 mt-1.5 flex flex-wrap gap-x-3.5 gap-y-1 text-[11px] text-chat-muted">
            {LEGEND.map(([s, text]) => (
              <span key={s} className="inline-flex items-center gap-1.5">
                <Swatch state={s} />
                {text}
              </span>
            ))}
            <span className="inline-flex items-center gap-1.5">
              <Swatch state="new" />
              {NEW_TEXT[view.scope]}
            </span>
            {!isKey && (
              <span className="inline-flex items-center gap-1.5">
                <Swatch state="other" />
                cached by another key
              </span>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function Bracket({ left, width, label }: { left: string; width: string; label: string }) {
  return (
    <div
      className="absolute top-[4px] h-[14px] overflow-hidden text-ellipsis whitespace-nowrap rounded-t-[3px] border border-b-0 border-chat-muted px-1 text-[10px] leading-[12px] text-chat-muted"
      style={{ left, width: `calc(${width} - 2px)` }}
    >
      {label}
    </div>
  );
}

function UnderBracket({ row, width, label, dashed = false }: { row: 0 | 1; width: string; label: string; dashed?: boolean }) {
  return (
    <div className={cn("absolute inset-x-0 h-[22px]", row === 0 ? "top-0" : "top-[22px]")}>
      <div
        className={cn(
          "h-[8px] rounded-b-[3px] border border-t-0",
          dashed ? "border-dashed border-vw-ttft" : "border-vw-info-fg",
        )}
        style={{ width }}
      />
      <div
        className={cn(
          "max-w-full truncate text-[10px] leading-[12px]",
          dashed ? "text-vw-ttft" : "text-vw-info-fg",
        )}
      >
        {label}
      </div>
    </div>
  );
}

/** The map's footer: edge, conv, profile and (admin) the engine outcome. */
export function KvsigFooter({ view }: { view: KvsigView }) {
  if (view.empty !== null) return null;
  return (
    <div data-testid="kvsig-footer" className="mt-2 flex min-w-0 flex-wrap gap-x-4 gap-y-1.5 text-[11px] text-chat-muted">
      {view.footer.map(([k, v]) => (
        <span key={k} className="min-w-0 break-words">
          {k} <b className="font-medium text-chat-fg">{v}</b>
        </span>
      ))}
      <span className="text-chat-dim">Ids are short forms of keyed hashes; they reveal no text.</span>
    </div>
  );
}
