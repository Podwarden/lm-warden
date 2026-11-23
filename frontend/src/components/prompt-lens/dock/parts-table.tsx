"use client";

// Prompt Lens parts table (spec 2026-11-22 section 3.1, mockup states 3-7):
// Part · Covers · ≈ tok · exact id · loose id · vs earlier · Cache. Ids are the
// short forms (2×4 hex, Q4), shown only here and never on a phone. Over 8 rows
// the table scrolls inside a fixed-height box (the fixed-size rule); with
// `bare` a SectionBox (Task 11) hosts it and does the scrolling instead. On a phone it sits behind "Show the N
// parts ▸" and stacks into cards.

import { useState } from "react";

import { cacheText, shortId, type MapPart } from "@/lib/prompt-lens/kvsig";
import { cn } from "@/lib/utils";
import { Swatch } from "./kvsig-map";

/** Rows shown before the table scrolls in its box. */
export const PARTS_BOX_ROWS = 8;
/** The box's fixed height: 8 rows and the header. */
const BOX = "h-[264px] overflow-y-auto";

const fmt = (n: number) => n.toLocaleString("en-US");

const TH = "sticky top-0 whitespace-nowrap border-b border-chat-rule bg-chat-surface px-1.5 py-1 text-left text-[10px] font-semibold uppercase tracking-[.07em] text-chat-dim";
/** A phone card's field label (the cards have no header row). */
const CARD_LABEL = "text-[10px] uppercase tracking-[.07em] text-chat-dim";
const TD = "border-b border-vw-rule-soft/50 px-1.5 py-1 align-top";

export interface PartsTableProps {
  parts: MapPart[];
  phone: boolean;
  /** The highlighted part's index (a clicked map segment). */
  active?: number | null;
  /** Hosted in a SectionBox (Task 11), which scrolls it: no box of its own. */
  bare?: boolean;
}

function PartName({ p }: { p: MapPart }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <Swatch state={p.state} />
      <span className="font-mono font-medium">{p.label}</span>
    </span>
  );
}

export function PartsTable({ parts, phone, active = null, bare = false }: PartsTableProps) {
  const [open, setOpen] = useState(false);
  if (parts.length === 0) return null;
  const boxed = !bare && parts.length > PARTS_BOX_ROWS;

  if (phone) {
    const cards = (
      <ul className="m-0 list-none p-0">
        {parts.map((p, i) => (
          <li
            key={`${p.p}:${p.from}`}
            data-testid="parts-card"
            aria-current={active === i ? "true" : undefined}
            className={cn(
              "grid min-w-0 grid-cols-[auto_minmax(0,1fr)] gap-x-2.5 border-b border-vw-rule-soft/50 py-1.5 text-[12px]",
              active === i && "bg-chat-surface-2",
            )}
          >
            <PartName p={p} />
            <span className="min-w-0 break-words">{p.covers}</span>
            <span className={CARD_LABEL}>≈ tok</span>
            <span className="tabular-nums text-chat-dim">{fmt(p.tok)}</span>
            <span className={CARD_LABEL}>vs earlier</span>
            <span className="min-w-0 break-words">{p.vsEarlier}</span>
            <span className={CARD_LABEL}>Cache</span>
            <span data-testid="parts-cache" className="text-chat-dim">
              {cacheText(p.cache)}
            </span>
          </li>
        ))}
      </ul>
    );
    return (
      <div className="min-w-0">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
          className="mt-1 inline-flex min-h-11 cursor-pointer items-center rounded-md border border-chat-rule bg-transparent px-3 py-1 text-[12px] text-chat-fg hover:bg-chat-surface"
        >
          {open ? `Hide the ${parts.length} parts ▾` : `Show the ${parts.length} parts ▸`}
        </button>
        {open && (boxed ? <div data-testid="parts-box" className={cn(BOX, "mt-1")}>{cards}</div> : <div className="mt-1">{cards}</div>)}
      </div>
    );
  }

  const table = (
    <table className="w-full border-collapse text-[12px] tabular-nums">
      <thead>
        <tr>
          <th scope="col" className={TH}>Part</th>
          <th scope="col" className={TH}>Covers</th>
          <th scope="col" className={TH}>≈ tok</th>
          <th scope="col" className={TH}>exact id</th>
          <th scope="col" className={TH}>loose id</th>
          <th scope="col" className={TH}>vs earlier</th>
          <th scope="col" className={TH}>Cache</th>
        </tr>
      </thead>
      <tbody>
        {parts.map((p, i) => (
          <tr
            key={`${p.p}:${p.from}`}
            aria-current={active === i ? "true" : undefined}
            className={cn(active === i && "bg-chat-surface-2")}
          >
            <td className={TD}>
              <PartName p={p} />
            </td>
            <td className={TD}>{p.covers}</td>
            <td className={cn(TD, "whitespace-nowrap")}>{fmt(p.tok)}</td>
            <td className={cn(TD, "whitespace-nowrap font-mono text-[11px] text-chat-muted")}>{shortId(p.exact8)}</td>
            <td className={cn(TD, "whitespace-nowrap font-mono text-[11px] text-chat-muted")}>{shortId(p.loose8)}</td>
            <td className={TD}>{p.vsEarlier}</td>
            <td data-testid="parts-cache" className={cn(TD, "whitespace-nowrap")}>
              {cacheText(p.cache)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  return boxed ? (
    <div data-testid="parts-box" className={cn(BOX, "min-w-0 overflow-x-auto")}>
      {table}
    </div>
  ) : (
    <div className="min-w-0 overflow-x-auto">{table}</div>
  );
}
