"use client";

// A Prompt Lens section (spec 2026-11-22 §3.3, mockup states 8 and 9): a
// header and a fixed-height box that follows the tail while it streams.
//
// Header: name · whole size (`chars · ≈ tok`, amber over 2,000 tok) · marks
// (trimmed / cut on the server / text dropped) · status · "● following" or
// "● not following" while live · "≈ N tok/s" · ▾ collapse · ⤢ expand.
// Box: 200 px (160 px on a phone, Q11), shrink-to-fit under 500 chars,
// contained scrolling, a top fade and "↓ Jump to latest · N new chars" while
// not following. Nothing in the header changes height as values change, so
// streaming never shifts the layout (the Stats fixed-size rule).
//
// ⤢ only calls `onExpand`; the pop-up itself is Task 12's.

import { useCallback, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { SectionKind } from "@/lib/prompt-lens/section-marks";
import { CLIENT_SECTION_CHARS } from "@/lib/prompt-lens/store";
import { LiveNumber } from "@/components/ui/live-number";
import { cn } from "@/lib/utils";
import { FollowTailContext, useFollowTail } from "./use-follow-tail";

export type { SectionKind };

/** Box height, px (Q11: fixed, no setting). */
export const SECTION_BOX_PX = 200;
export const SECTION_BOX_PHONE_PX = 160;
/** Under this many chars the box shrinks to fit its content. */
export const SECTION_SHORT_CHARS = 500;
/** A section's size is shown in amber over this many tokens. */
export const SECTION_BIG_TOK = 2000;
/** Tool results start collapsed over this many tokens. */
export const TOOL_RESULTS_COLLAPSE_TOK = 4000;

export interface SectionMarks {
  /** Chars of this section the dock no longer holds (its head). */
  trimmed?: number;
  /** The server cut the section (its capture budget). */
  elided?: boolean;
  /** The dock dropped this request's text to stay under its memory bound. */
  stripped?: boolean;
}

export interface SectionBoxProps {
  kind: SectionKind;
  title: string;
  /** Size of the whole section, not only what the box holds. */
  chars: number;
  /** ≈ tokens of the whole section; defaults to chars / 4. */
  tok?: number;
  /** The section is still streaming. */
  live: boolean;
  /** Live tok/s, shown as "≈ N tok/s". */
  rate?: number | null;
  status?: ReactNode;
  collapsed: boolean;
  onToggle: () => void;
  /** ⤢: open the section in the large view (Task 12). */
  onExpand: () => void;
  phone: boolean;
  /** Text shown when the section is empty. */
  empty?: string;
  /** Root test id; the box is always `section-box-<kind>`. */
  testid?: string;
  /** The request's reqId: a new one starts following at the bottom again. */
  resetKey: string;
  /** What changes when the content does; defaults to `chars`. */
  contentKey?: unknown;
  marks?: SectionMarks;
  children?: ReactNode;
}

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

const KIND_EDGE: Record<SectionKind, string> = {
  thinking: "border-l-vw-prompt",
  reply: "border-l-vw-completion",
  tool_calls: "border-l-vw-queue",
  tool_results: "border-l-vw-ttft",
  prompt: "border-l-chat-rule",
  parts: "border-l-chat-rule",
};

/** Header items that give way (truncate) before the actions do, at 375 px. */
const SHRINK = "min-w-0 overflow-hidden text-ellipsis";
/** Live counter slots, in characters (ui.live-numbers): a section's chars
 *  up to "999,999", its tokens up to "99,999", a rate up to "9,999". */
export const LIVE_CHARS_CH = 7;
export const LIVE_TOK_CH = 6;
export const LIVE_RATE_CH = 5;
const PILL = "flex-none rounded border px-[5px] text-[10px] leading-4 tabular-nums";
/** A phone's touch target (44 px), the glyph centred. */
const ICON_BTN_PHONE = "inline-flex min-h-11 min-w-11 items-center justify-center text-[14px]";
const ICON_BTN =
  "min-w-5 cursor-pointer rounded border border-chat-rule bg-transparent px-[5px] text-[11px] leading-4 text-chat-muted hover:border-chat-muted hover:text-chat-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-chat-accent";

function markTexts(m: SectionMarks | undefined): string[] {
  if (!m) return [];
  const out: string[] = [];
  if (m.trimmed && m.trimmed > 0) out.push(`first ${fmt(m.trimmed)} chars trimmed`);
  if (m.elided) out.push("cut on the server");
  if (m.stripped) out.push("text dropped to save memory");
  return out;
}

const MARKS_TITLE = `Prompt Lens keeps the last ${fmt(CLIENT_SECTION_CHARS)} chars of a section and drops the oldest requests' text first. ⤢ shows the whole section from server memory.`;

export function SectionBox({
  kind,
  title,
  chars,
  tok,
  live,
  rate = null,
  status,
  collapsed,
  onToggle,
  onExpand,
  phone,
  empty,
  testid,
  resetKey,
  contentKey,
  marks,
  children,
}: SectionBoxProps) {
  const boxRef = useRef<HTMLDivElement>(null);
  const t = tok ?? Math.round(chars / 4);
  const big = t > SECTION_BIG_TOK;
  // Collapsing and expanding again starts at the bottom, following.
  const tail = useFollowTail(boxRef, contentKey ?? chars, live, `${resetKey}:${collapsed ? "c" : "o"}`);
  const { following, newChars, jump, noteChars } = tail;

  const prevChars = useRef(chars);
  const prevReset = useRef(resetKey);
  useLayoutEffect(() => {
    if (prevReset.current !== resetKey) {
      prevReset.current = resetKey;
      prevChars.current = chars;
      return;
    }
    const added = chars - prevChars.current;
    prevChars.current = chars;
    if (added > 0) noteChars(added);
  }, [chars, resetKey, noteChars]);

  const ctx = useMemo(() => ({ nudge: tail.nudge }), [tail.nudge]);
  const marksShown = markTexts(marks);
  const trimmed = marks?.trimmed ?? 0;
  const isEmpty = chars === 0 && !children;
  const short = chars < SECTION_SHORT_CHARS;
  const px = phone ? SECTION_BOX_PHONE_PX : SECTION_BOX_PX;
  const away = live && !following;

  return (
    <section
      data-testid={testid ?? `section-${kind}`}
      data-kind={kind}
      aria-label={title}
      className={cn(
        "mb-2.5 min-w-0 rounded-lg border border-l-[3px] border-chat-rule bg-chat-surface px-3 pt-2.5",
        KIND_EDGE[kind],
        collapsed ? "pb-1.5" : "pb-2.5",
      )}
    >
      <div
        className={cn(
          "flex min-w-0 gap-2 overflow-hidden whitespace-nowrap",
          // A phone's header holds 44 px buttons: centre instead of baseline.
          phone ? "items-center" : "items-baseline",
          !collapsed && "mb-2",
        )}
      >
        <h4
          title={phone ? title : undefined}
          className={cn(
            "m-0 text-[11px] font-bold uppercase tracking-[.08em] text-chat-muted",
            // A phone's header is narrow: the title gives way before the buttons do.
            phone ? "min-w-0 flex-shrink overflow-hidden text-ellipsis" : "flex-none",
          )}
        >
          {title}
        </h4>
        <span
          data-testid={`section-size-${kind}`}
          data-big={String(big)}
          title="The whole section, not only what the box shows"
          className={cn(PILL, big ? "border-vw-amber-fg/60 text-vw-amber-fg" : "border-chat-rule text-chat-fg")}
        >
          {chars === 0 ? (
            "empty"
          ) : (
            <>
              <LiveNumber value={chars} reserve={live ? LIVE_CHARS_CH : 0} resetKey={resetKey} /> chars ·{" "}
              <LiveNumber reserve={live ? LIVE_TOK_CH + 2 : 0} resetKey={resetKey}>{`≈ ${fmt(t)}`}</LiveNumber> tok
            </>
          )}
        </span>
        {marksShown.length > 0 && (
          <span
            data-testid={`section-marks-${kind}`}
            title={MARKS_TITLE}
            className={cn(PILL, SHRINK, "flex-initial border-vw-amber-fg/40 text-vw-amber-fg")}
          >
            {marksShown.join(" · ")}
          </span>
        )}
        {status != null && status !== "" && (
          <span className={cn(SHRINK, "text-[11px] text-chat-dim")}>{status}</span>
        )}
        {collapsed && <span className={cn(SHRINK, "text-[10px] text-chat-dim")}>collapsed · remembered while this page is open</span>}
        {live && !collapsed && (
          <span className={cn(SHRINK, "text-[10px]", following ? "text-vw-ok-fg" : "text-vw-amber-fg")}>
            {following ? "● following" : "● not following"}
          </span>
        )}
        {rate != null && (
          <span className={cn(SHRINK, "ml-auto text-[11px] text-chat-muted")}>
            ≈ <LiveNumber value={rate} reserve={live ? LIVE_RATE_CH : 0} resetKey={resetKey} /> tok/s
          </span>
        )}
        <span className={cn("flex flex-none items-baseline gap-2", rate == null && "ml-auto")}>
          <span className="flex gap-1">
            <button
              type="button"
              aria-expanded={!collapsed}
              aria-label={collapsed ? `Show ${title}` : `Collapse ${title}`}
              title={collapsed ? "Expand section" : "Collapse section"}
              onClick={onToggle}
              className={cn(ICON_BTN, phone && ICON_BTN_PHONE)}
            >
              {collapsed ? "▸" : "▾"}
            </button>
            <button
              type="button"
              aria-label={`Open ${title} in a large view`}
              title="Open in a large pop-up"
              onClick={onExpand}
              className={cn(ICON_BTN, phone && ICON_BTN_PHONE)}
            >
              ⤢
            </button>
          </span>
        </span>
      </div>
      {!collapsed && (
        <div className="relative">
          <FollowTailContext.Provider value={ctx}>
            <div
              ref={boxRef}
              data-testid={`section-box-${kind}`}
              data-scroll-box=""
              data-following={String(following)}
              tabIndex={0}
              style={short ? { maxHeight: px } : { height: px }}
              className={cn(
                "overflow-y-auto overscroll-contain whitespace-pre-wrap text-[13px] leading-[1.55] text-chat-fg [overflow-anchor:none] [overflow-wrap:anywhere] [scrollbar-width:thin]",
                "focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent",
                kind === "thinking" && "italic text-chat-fg/90",
              )}
            >
              {trimmed > 0 && (
                <div data-testid={`section-trim-note-${kind}`} className="mb-1 text-[11px] not-italic text-chat-dim">
                  … {fmt(trimmed)} earlier chars are not kept on this page (⤢ shows the whole section)
                </div>
              )}
              {isEmpty ? <span className="text-[12px] italic text-chat-dim">{empty ?? "Empty"}</span> : children}
            </div>
          </FollowTailContext.Provider>
          {away && (
            <>
              <div
                data-testid={`section-fade-${kind}`}
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 top-0 h-[18px] bg-gradient-to-b from-chat-surface to-transparent"
              />
              <button
                type="button"
                onClick={jump}
                className="absolute bottom-2 left-1/2 -translate-x-1/2 cursor-pointer whitespace-nowrap rounded-full border-0 bg-chat-accent px-3 py-[3px] text-[12px] font-semibold text-chat-on-accent shadow-[0_4px_14px_rgba(0,0,0,.5)]"
              >
                ↓ Jump to latest{newChars > 0 ? ` · ${fmt(newChars)} new chars` : ""}
              </button>
            </>
          )}
        </div>
      )}
    </section>
  );
}

/** Collapsed state per section kind, kept in the dock's React state only
 *  (Q12: not in localStorage), so it survives switching requests and a new
 *  dock starts from the defaults: Prompt collapsed, Tool results collapsed
 *  over 4,000 tok, the rest open. `tok` is the section's size, needed only for
 *  the Tool results default. */
export function useCollapseMemory(): [(k: SectionKind, tok?: number) => boolean, (k: SectionKind, v: boolean) => void] {
  const [chosen, setChosen] = useState<Partial<Record<SectionKind, boolean>>>({});
  const isCollapsed = useCallback(
    (k: SectionKind, tok = 0) => {
      const v = chosen[k];
      if (v !== undefined) return v;
      if (k === "prompt") return true;
      if (k === "tool_results") return tok > TOOL_RESULTS_COLLAPSE_TOK;
      return false;
    },
    [chosen],
  );
  const setCollapsed = useCallback((k: SectionKind, v: boolean) => {
    setChosen((c) => (c[k] === v ? c : { ...c, [k]: v }));
  }, []);
  return [isCollapsed, setCollapsed];
}
