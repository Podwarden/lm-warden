"use client";

// The grouped sessions table (mockup sessions-list-v3): a row per key / harness / day with its totals, opening onto
// its newest sessions and "Show N more"; or, grouped by None, a flat list of sessions.

import Link from "next/link";
import { useEffect, useState } from "react";

import { sessionHref } from "@/lib/routes";
import { S } from "@/lib/sessions/strings";
import type { ListGroup, ListGroupBy, SessionListItem, Where } from "@/lib/sessions/types";

import { AnatomyBar } from "./anatomy-bar";
import { fmtApprox, fmtDuration, fmtStart, fmtTokens, pct } from "./format";

export const GROUP_MORE = 50;

export interface MorePage {
  items: SessionListItem[];
  more: number;
}

const GRID =
  "grid grid-cols-[22px_minmax(0,1.4fr)_90px_110px_120px_150px_110px] items-center gap-3 px-4 tabular-nums";

/** "48 M · 96%": the share served from cache, never the anatomy's (dead weight is taken out there); no % when
 * the cache is unknown. */
function tokensCell(prompt: number, share: number | null): string {
  const p = pct(share);
  return p === null ? fmtTokens(prompt) : `${fmtTokens(prompt)} · ${p}%`;
}

function Saved({ s, relayedOnly }: { s: number | null; relayedOnly: boolean }) {
  if (relayedOnly) return <span className="text-chat-muted">{S.list.savedRelayed}</span>;
  if (s === null)
    return (
      <span title={S.list.savedUnknown} className="text-chat-muted">
        —
      </span>
    );
  return <span title={S.list.savedNote}>{fmtApprox(s)}</span>;
}

function WhereBadge({ where }: { where: Where }) {
  if (where === "local") return null;
  return (
    <span
      className={`ml-1.5 rounded border px-1.5 text-[11px] ${
        where === "relayed" ? "border-[#3e4659] text-[#7d8fb3]" : "border-chat-rule text-chat-muted"
      }`}
    >
      {where === "relayed" ? S.list.relayed : S.list.hybrid}
    </span>
  );
}

function SessionRow({ it }: { it: SessionListItem }) {
  const small = [
    it.active ? S.list.running : null,
    fmtDuration(it.duration_s),
    it.conversations > 1 ? S.list.agents(it.conversations) : null,
    it.compactions > 0 ? S.list.compactions(it.compactions) : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div role="row" data-testid="session-row" data-sid={it.session_id} className={`${GRID} border-t border-chat-rule/60 bg-chat-page py-2 text-[13px]`}>
      <span role="cell" />
      <span role="cell" className="min-w-0 truncate">
        {it.active && (
          <span
            className="mr-1.5 inline-block h-[7px] w-[7px] rounded-full bg-emerald-500 align-middle"
            title={S.list.active}
            aria-label={S.list.active}
          />
        )}
        <Link
          href={sessionHref(it.session_id)}
          className="text-chat-fg hover:text-chat-accent hover:underline focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
        >
          {fmtStart(it.started_at)}
        </Link>
        <small className="ml-2 text-chat-muted">{small}</small>
        <WhereBadge where={it.where} />
      </span>
      <span role="cell" />
      <span role="cell">{fmtDuration(it.duration_s)}</span>
      <span role="cell">{tokensCell(it.prompt_tokens, it.cache_known ? it.cache_share : null)}</span>
      <span role="cell">
        <AnatomyBar a={it.anatomy} />
      </span>
      <span role="cell">
        <Saved s={it.saved_vs_vanilla_s} relayedOnly={it.where === "relayed"} />
      </span>
    </div>
  );
}

function MoreButton({ label, busy, onClick }: { label: string; busy: boolean; onClick: () => void }) {
  return (
    <div className="border-t border-chat-rule/60 bg-chat-page py-2 pl-[50px] pr-4">
      <button
        type="button"
        disabled={busy}
        onClick={onClick}
        className="text-[13px] text-chat-accent hover:underline disabled:opacity-60 focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
      >
        {label}
      </button>
    </div>
  );
}

/** The first page then the appended ones, each session once: the first page refreshes every 15 s, so an
 * appended page fetched by offset can overlap it (verified finding #59). */
function unique(items: SessionListItem[]): SessionListItem[] {
  const seen = new Set<string>();
  return items.filter((it) => (seen.has(it.session_id) ? false : (seen.add(it.session_id), true)));
}

function groupSmall(g: ListGroup, column: ListGroupBy): string {
  const parts: string[] = [];
  if (column !== "harness") parts.push(g.harnesses.map((h) => S.harness[h] ?? h).join(", "));
  const w = g.where;
  if (w.relayed > 0 || w.hybrid > 0) parts.push(S.list.whereCounts(w));
  if (g.evicted_resumes > 0) parts.push(S.list.groupEvicted(g.evicted_resumes, g.evicted_lost_s));
  return parts.filter(Boolean).join(" · ");
}

function GroupBlock({
  g,
  column,
  open,
  onToggle,
  onMore,
}: {
  g: ListGroup;
  column: ListGroupBy;
  open: boolean;
  onToggle: () => void;
  onMore: (groupKey: string, offset: number) => Promise<MorePage>;
}) {
  const [extra, setExtra] = useState<SessionListItem[]>([]);
  const [more, setMore] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const items = unique([...g.items, ...extra]);
  const left = more ?? g.more;
  const label = column === "harness" ? S.harness[g.label] ?? g.label : g.label;
  const relayedOnly = g.where.relayed > 0 && g.where.local === 0 && g.where.hybrid === 0;

  async function loadMore() {
    setBusy(true);
    try {
      const page = await onMore(g.key, items.length);
      setExtra((xs) => [...xs, ...page.items]);
      setMore(page.more);
    } catch {
      /* the button stays; a click retries */
    } finally {
      setBusy(false);
    }
  }

  return (
    <div role="rowgroup">
      <div
        role="row"
        data-testid="group-row"
        className={`${GRID} border-t border-chat-rule py-[11px] ${open ? "bg-chat-surface-2" : ""}`}
      >
        <span role="cell" aria-hidden className="text-chat-muted">
          {open ? "▾" : "▸"}
        </span>
        <span role="cell" className="min-w-0">
          <button
            type="button"
            aria-expanded={open}
            onClick={onToggle}
            className="max-w-full truncate text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
          >
            <span className="font-semibold text-chat-fg">{label}</span>
            <small className="ml-1.5 text-chat-muted">{groupSmall(g, column)}</small>
          </button>
        </span>
        <span role="cell">{g.sessions.toLocaleString()}</span>
        <span role="cell">{S.list.durationTotal(fmtDuration(g.duration_s))}</span>
        <span role="cell">{tokensCell(g.prompt_tokens, g.cached_share)}</span>
        <span role="cell">
          <AnatomyBar a={g.anatomy} />
        </span>
        <span role="cell">
          <Saved s={g.saved_vs_vanilla_s} relayedOnly={relayedOnly} />
        </span>
      </div>
      {open && items.map((it) => <SessionRow key={it.session_id} it={it} />)}
      {open && left > 0 && <MoreButton label={S.list.showMore(left, label)} busy={busy} onClick={() => void loadMore()} />}
    </div>
  );
}

export function SessionGroups({
  groups,
  sessions,
  total,
  column,
  onMore,
  resetKey,
}: {
  groups: ListGroup[] | null;
  sessions: SessionListItem[];
  total: number;
  column: ListGroupBy;
  onMore: (groupKey: string | null, offset: number) => Promise<MorePage>;
  /** Changes when the query changes: appended pages and opened groups start over. */
  resetKey: string;
}) {
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const [extra, setExtra] = useState<SessionListItem[]>([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setToggled({});
    setExtra([]);
  }, [resetKey]);

  const flat = column === "none" || groups === null;
  const items = unique([...sessions, ...extra]);
  const flatLeft = Math.max(0, total - items.length);

  async function loadFlat() {
    setBusy(true);
    try {
      const page = await onMore(null, items.length);
      setExtra((xs) => [...xs, ...page.items]);
    } catch {
      /* a click retries */
    } finally {
      setBusy(false);
    }
  }

  const c = S.list.cols;
  return (
    <div className="overflow-x-auto">
      <div role="table" className="min-w-[780px]">
        <div role="row" className={`${GRID} border-t border-chat-rule py-2 text-xs text-chat-dim`}>
          <span role="columnheader" aria-hidden />
          <span role="columnheader">{S.list.firstCol[flat ? "none" : column]}</span>
          <span role="columnheader">{flat ? "" : c.sessions}</span>
          <span role="columnheader">{c.duration}</span>
          <span role="columnheader">{c.prompt}</span>
          <span role="columnheader">{c.anatomy}</span>
          <span role="columnheader" title={S.list.savedNote}>
            {c.saved}
          </span>
        </div>
        {flat ? (
          <div role="rowgroup">
            {items.map((it) => (
              <SessionRow key={it.session_id} it={it} />
            ))}
            {flatLeft > 0 && <MoreButton label={S.list.showMoreFlat(flatLeft)} busy={busy} onClick={() => void loadFlat()} />}
          </div>
        ) : (
          groups.map((g) => (
            <GroupBlock
              key={`${resetKey}|${g.key}`}
              g={g}
              column={column}
              open={toggled[g.key] ?? groups.length === 1}
              onToggle={() => setToggled((t) => ({ ...t, [g.key]: !(t[g.key] ?? groups.length === 1) }))}
              onMore={(k, offset) => onMore(k, offset)}
            />
          ))
        )}
      </div>
    </div>
  );
}
