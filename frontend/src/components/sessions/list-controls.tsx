"use client";

// The list's ready views (with counts), Group by, Where, the search box (debounced) and Sort.

import { useEffect, useState } from "react";

import { S } from "@/lib/sessions/strings";
import type { ListGroupBy, ListSort, ListView, ListWhere } from "@/lib/sessions/types";

const VIEW_ORDER: ListView[] = ["all", "running", "biggest", "dead_weight", "cache_trouble", "long_running", "evicted"];
const GROUPS: ListGroupBy[] = ["key", "harness", "day", "none"];
const WHERES: ListWhere[] = ["all", "local", "relayed", "hybrid"];
const SORTS: ListSort[] = ["newest", "prompt_tokens", "dead_weight", "duration", "evicted_lost"];
export const SEARCH_DEBOUNCE_MS = 300;

function Segmented<T extends string>({
  label,
  value,
  options,
  names,
  onChange,
}: {
  label: string;
  value: T;
  options: T[];
  names: Record<T, string>;
  onChange: (v: T) => void;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <span>{label}</span>
      <span role="radiogroup" aria-label={label} className="inline-flex overflow-hidden rounded-lg border border-chat-rule">
        {options.map((o) => (
          <button
            key={o}
            type="button"
            role="radio"
            aria-checked={value === o}
            onClick={() => onChange(o)}
            className={`px-2.5 py-1 focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent ${
              value === o ? "bg-chat-surface-2 text-chat-fg" : "hover:text-chat-fg"
            }`}
          >
            {names[o]}
          </button>
        ))}
      </span>
    </span>
  );
}

export function ListControls({
  views,
  view,
  onView,
  group,
  onGroup,
  where,
  onWhere,
  q,
  onQ,
  sort,
  onSort,
}: {
  views: Record<ListView, number>;
  view: ListView;
  onView: (v: ListView) => void;
  group: ListGroupBy;
  onGroup: (g: ListGroupBy) => void;
  where: ListWhere;
  onWhere: (w: ListWhere) => void;
  q: string;
  onQ: (q: string) => void;
  sort: ListSort;
  onSort: (s: ListSort) => void;
}) {
  const [text, setText] = useState(q);
  // Sync from outside (Clear selection) without trimming what is being typed.
  useEffect(() => setText((t) => (t.trim() === q ? t : q)), [q]);
  useEffect(() => {
    if (text.trim() === q) return;
    const id = setTimeout(() => onQ(text.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- debounce on the typed text only
  }, [text]);

  return (
    <>
      <div role="group" aria-label={S.list.viewsAria} className="flex flex-wrap gap-1.5 border-t border-chat-rule px-4 py-3">
        {VIEW_ORDER.map((v) => {
          const on = v === view;
          return (
            <button
              key={v}
              type="button"
              data-testid={`view-${v}`}
              aria-pressed={on}
              onClick={() => onView(v)}
              className={`rounded-lg border px-2.5 py-1.5 text-[13px] focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent ${
                on ? "border-chat-accent text-chat-accent" : "border-chat-rule bg-chat-surface-2 text-chat-fg hover:border-chat-dim"
              }`}
            >
              {S.list.views[v].label}
              <small className="ml-1.5 text-chat-muted">{S.list.views[v].small(views[v] ?? 0)}</small>
            </button>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-chat-rule px-4 py-2.5 text-[13px] text-chat-muted">
        <Segmented label={S.list.groupByLabel} value={group} options={GROUPS} names={S.list.groupBy} onChange={onGroup} />
        <Segmented label={S.list.whereLabel} value={where} options={WHERES} names={S.list.where} onChange={onWhere} />
        <input
          type="search"
          aria-label={S.list.searchLabel}
          placeholder={S.list.searchPlaceholder}
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="min-w-0 flex-[1_1_220px] rounded-lg border border-chat-rule bg-chat-page px-2.5 py-1.5 text-chat-fg placeholder:text-chat-dim focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
        />
        <label className="inline-flex items-center gap-2">
          {S.list.sortLabel}
          <select
            aria-label={S.list.sortLabel}
            value={sort}
            onChange={(e) => onSort(e.target.value as ListSort)}
            className="rounded-lg border border-chat-rule bg-chat-page px-2 py-1 text-chat-fg"
          >
            {SORTS.map((s) => (
              <option key={s} value={s}>
                {S.list.sort[s]}
              </option>
            ))}
          </select>
        </label>
      </div>
    </>
  );
}
