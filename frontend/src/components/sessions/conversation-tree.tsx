"use client";

import { Panel } from "@/components/stats/live-panels";
import { laneOrder } from "@/lib/sessions/layout";
import { S } from "@/lib/sessions/strings";
import type { Conversation, SessionResponse } from "@/lib/sessions/types";

import { AnatomyBar } from "./anatomy-bar";
import { pct } from "./format";

function linkBadge(c: Conversation) {
  if (!c.parent) return null;
  if (c.parent.link === "inferred") return <span className="text-[10px] text-amber-500">{S.tree.inferred}</span>;
  if (c.parent.link === "known") return <span className="text-[10px] text-emerald-400">{S.tree.known}</span>;
  return null;
}

export function ConversationTree({
  s,
  selected,
  onSelect,
}: {
  s: SessionResponse;
  /** The selected conversation; undefined = the whole session. */
  selected: string | null | undefined;
  onSelect: (conversationId: string | null | undefined) => void;
}) {
  const convs = laneOrder(s.conversations);
  // A conversation's "% from cache" only where some turn's cache is known (never 0% for an MLX lane).
  const known = new Set(
    s.turns.filter((t) => t.cache_known ?? t.cached_tokens !== null).map((t) => t.conversation_id),
  );
  const allSelected = selected === undefined;
  return (
    <Panel title={S.tree.title} testid="conversation-tree">
      <ul className="space-y-1 text-sm" role="listbox" aria-label={S.tree.title}>
        <li
          role="option"
          aria-selected={allSelected}
          className="cursor-pointer rounded px-2 py-0.5 text-chat-muted hover:bg-slate-800"
          onClick={() => onSelect(undefined)}
        >
          {S.tree.all}
        </li>
        {convs.map((c) => {
          const sel = selected === c.id;
          const share = pct(c.prefill_tokens > 0 && known.has(c.id) ? c.cached_tokens / c.prefill_tokens : null);
          const indent = c.kind === "main" || c.kind === "conversation" || c.kind === "unlinked" ? "" : "pl-4";
          const glyph =
            c.kind === "main" ? "●" : c.kind === "compaction" ? "↳" : c.parent?.link === "inferred" ? "┆" : c.parent ? "│" : "○";
          return (
            <li
              key={String(c.id)}
              role="option"
              aria-selected={sel}
              onClick={() => onSelect(c.id)}
              className={`cursor-pointer rounded px-2 py-0.5 ${indent} ${sel ? "bg-slate-700/60" : "hover:bg-slate-800"} ${
                c.kind === "compaction" ? "text-yellow-400" : ""
              }`}
            >
              <span className="mr-1">{glyph}</span>
              <span>{c.label}</span> {linkBadge(c)}{" "}
              {c.relayed && (
                <span className="rounded border border-slate-600 px-1 text-[10px] text-slate-300">
                  {S.timeline.relayed}
                </span>
              )}{" "}
              <span className="text-xs text-chat-muted">
                {S.tree.turns(c.turns)}
                {share !== null ? ` · ${share}%` : ""}
              </span>
              {c.anatomy && (
                <span className="ml-2 inline-block w-16 align-middle">
                  <AnatomyBar a={c.anatomy} />
                </span>
              )}
            </li>
          );
        })}
        {s.side_calls.length > 0 && (
          <li className="px-2 py-0.5 text-xs text-chat-muted">{S.tree.sideCalls(s.side_calls.length)}</li>
        )}
      </ul>
    </Panel>
  );
}
