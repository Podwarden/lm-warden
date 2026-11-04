"use client";

// /sessions — the sessions overview: the activity chart (drag to narrow the time), ready views, grouping and filters,
// and the grouped table. An admin sees every key; a key holder (key mode, sessions/key-token.ts) sees only their own
// key's group, opened, chromeless under SessionsShell.

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import useSWR from "swr";

import { ActivityChart, type Brush } from "@/components/sessions/activity-chart";
import { ListControls } from "@/components/sessions/list-controls";
import { PauseNotes, PauseTable } from "@/components/sessions/pause-table";
import { GROUP_MORE, type MorePage, SessionGroups } from "@/components/sessions/session-groups";
import { SessionsShell } from "@/components/sessions/sessions-shell";
import {
  type ListArgs,
  KeyAuthError,
  sessionsFetcher,
  sessionsKey,
  sessionsListUrl,
  useSessionsMode,
} from "@/lib/sessions/client";
import { SESSIONS_LOGIN } from "@/lib/sessions/key-token";
import { S } from "@/lib/sessions/strings";
import type { ListGroupBy, ListSort, ListView, ListWhere, SessionsListResponse } from "@/lib/sessions/types";

const RANGES = ["1h", "6h", "24h", "7d", "30d"] as const;
type Range = (typeof RANGES)[number];
const RANGE_KEY = "vw-sessions-range";

function readRange(): Range {
  try {
    const v = window.localStorage.getItem(RANGE_KEY);
    return (RANGES as readonly string[]).includes(v ?? "") ? (v as Range) : "24h";
  } catch {
    return "24h";
  }
}

function SessionsList() {
  const mode = useSessionsMode();
  const router = useRouter();
  const [range, setRange] = useState<Range>("24h");
  const [brush, setBrush] = useState<Brush | null>(null);
  const [view, setView] = useState<ListView>("all");
  const [group, setGroup] = useState<ListGroupBy>("key");
  const [where, setWhere] = useState<ListWhere>("all");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<ListSort>("newest");
  const fetcher = sessionsFetcher<SessionsListResponse>(mode);

  useEffect(() => setRange(readRange()), []);

  const args: ListArgs = {
    range,
    group,
    view,
    where,
    q: q || null,
    sort,
    since: brush?.since ?? null,
    until: brush?.until ?? null,
    tzOffsetMin: -new Date().getTimezoneOffset(),
  };
  const url = sessionsListUrl(args);
  const { data, error } = useSWR(sessionsKey(mode, url), ([, , u]) => fetcher(u), {
    refreshInterval: 15_000,
    keepPreviousData: true,
  });

  useEffect(() => {
    if (error instanceof KeyAuthError) router.replace(SESSIONS_LOGIN);
  }, [error, router]);

  function pick(r: Range) {
    setRange(r);
    setBrush(null);
    try {
      window.localStorage.setItem(RANGE_KEY, r);
    } catch {
      /* storage blocked: the choice lasts this page */
    }
  }

  function clearSelection() {
    setBrush(null);
    setView("all");
    setWhere("all");
    setQ("");
  }

  async function more(groupKey: string | null, offset: number): Promise<MorePage> {
    const page = await fetcher(sessionsListUrl({ ...args, groupKey, offset, limit: GROUP_MORE }));
    if (groupKey !== null) {
      const g = page.groups?.[0];
      return { items: g?.items ?? [], more: g?.more ?? 0 };
    }
    return { items: page.sessions, more: Math.max(0, page.total - offset - page.sessions.length) };
  }

  const filtered = brush !== null || view !== "all" || where !== "all" || q !== "";
  const oldest = data?.oldest_data_at != null ? new Date(data.oldest_data_at * 1000).toLocaleString() : null;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          {mode === "admin" && <h1 className="text-xl font-semibold text-chat-fg">{S.list.title}</h1>}
          <p className="max-w-[72ch] text-sm text-chat-muted">{S.list.intro}</p>
        </div>
        <label className="flex items-center gap-2 text-sm text-chat-muted">
          {S.list.rangeLabel}
          <select
            aria-label={S.list.rangeLabel}
            value={range}
            onChange={(e) => pick(e.target.value as Range)}
            className="rounded-lg border border-chat-rule bg-chat-page px-2 py-1 text-chat-fg"
          >
            {RANGES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error && !(error instanceof KeyAuthError) && (
        <p role="alert" className="text-sm text-chat-warn">
          {S.list.unavailable}
        </p>
      )}
      {!data && !error && <p className="text-sm text-chat-muted">{S.list.loading}</p>}
      {data?.truncated_before != null && <p className="text-sm text-chat-warn">{S.list.truncated}</p>}
      {data && (
        <div className="overflow-hidden rounded-[10px] border border-chat-rule bg-chat-surface">
          <ActivityChart h={data.histogram} brush={brush} onBrush={setBrush} rangeLabel={S.list.rangeText[range]} />
          <details className="group border-t border-chat-rule px-4 py-2.5">
            <summary className="cursor-pointer text-[13px] text-chat-muted hover:text-chat-fg">
              {S.pauses.surviveTitle}
            </summary>
            <div className="space-y-2 pt-3">
              <PauseTable p={data.pause_summary} />
              <PauseNotes p={data.pause_summary} />
            </div>
          </details>
          <ListControls
            views={data.views}
            view={view}
            onView={setView}
            group={group}
            onGroup={setGroup}
            where={where}
            onWhere={setWhere}
            q={q}
            onQ={setQ}
            sort={sort}
            onSort={setSort}
          />
          {data.total === 0 ? (
            <div className="border-t border-chat-rule px-4 py-8 text-center text-sm text-chat-muted">
              <p>{filtered ? S.list.emptyFiltered : S.list.empty}</p>
              {filtered && (
                <button
                  type="button"
                  onClick={clearSelection}
                  className="mt-2 text-chat-accent hover:underline focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
                >
                  {S.list.clearSelection}
                </button>
              )}
            </div>
          ) : (
            <SessionGroups
              groups={data.groups ?? null}
              sessions={data.sessions}
              total={data.total}
              column={group}
              onMore={more}
              resetKey={url}
            />
          )}
        </div>
      )}
      {oldest && <p className="text-xs text-chat-dim">{S.list.oldest(oldest)}</p>}
    </section>
  );
}

export default function SessionsPage() {
  const mode = useSessionsMode();
  return mode === "key" ? (
    <SessionsShell>
      <SessionsList />
    </SessionsShell>
  ) : (
    <SessionsList />
  );
}
