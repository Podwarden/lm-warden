"use client";

// Prompt Lens (spec 2026-11-22; a full page since 2026-11-23): one view for
// the key holder's own requests (scope "key", /key/prompt-lens) and the
// admin's (scope "admin", /prompt-lens, every key or a preset Key filter).
// It was a dock along the bottom of the Stats and key pages; the owner asked
// for a page with the normal menu and breadcrumbs that takes the whole screen
// from the start (2026-11-23).
//
// Layout: the header (title, Live, counts, Follow newest, Pause), the filter
// line, then the request list (360 px) beside the request screen, both
// filling the rest of the screen. Mounting the view opens the stream
// (usePromptLensStream); leaving the page closes it.
//
// Scope (G-R11): the key view names no key, shows no client IP, no other-key
// share and no ring minutes (Q8). User-visible naming is "Prompt Lens" (G-R1).
//
// Phone (below 768 px, spec section 6): the header keeps the title and Live;
// the filter line keeps the search and one "Filters (n)" button; the list
// fills the screen and a tapped row replaces it with the request screen
// ("‹ Requests" goes back). The system back gesture closes the top layer
// (pop-up, filter sheet, then the request screen: lib/prompt-lens/back-layer).

import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import useSWR from "swr";

import { LiveNumber } from "@/components/ui/live-number";
import { authFetchJSON } from "@/lib/auth-fetch";
import { applyFilters, useDockFilters, type DockFilters } from "@/lib/prompt-lens/filters";
import { serverNow } from "@/lib/prompt-lens/server-clock";
import { findByRequestId, type DockState, type RequestView } from "@/lib/prompt-lens/store";
import {
  fetchKeptCapture,
  fetchKeptList,
  fetchSection,
  usePromptLensStream,
  type FetchKind,
  type LensScope,
} from "@/lib/prompt-lens/stream";
import { useMe, type MeKey } from "@/lib/key-area/api";
import { useBackLayer } from "@/lib/prompt-lens/back-layer";
import { MAX_RECONNECT, type SseState } from "@/lib/sse";
import { PHONE_QUERY, useMediaQuery } from "@/lib/use-media-query";
import { cn } from "@/lib/utils";
import { FilterLine, type FilterOption, type FilterOptions } from "./dock/filter-line";
import { RequestList, LIST_WIDTH_PX, listOrder } from "./dock/request-list";
import { RequestScreen } from "./dock/request-screen";
import { useCollapseMemory } from "./dock/section-box";

/** Re-renders only when its request (or the scope, collapse or note) changes. */
const MemoScreen = memo(RequestScreen);

// ---------------------------------------------------------------------------
// The platform switch

/** The admin's Prompt Lens switch: `enabled` only when the status endpoint
 *  says so; loading or any error counts as off. SWR keeps the last good
 *  `data` next to the error of a failed refetch, so the error must be checked
 *  too (#251). `key` null skips the request (a page that is not the admin's). */
export function usePromptLensStatus(key: string | null = PROMPT_LENS_STATUS): { enabled: boolean; loading: boolean } {
  const { data, error, isLoading } = useSWR<{ enabled: boolean }>(key, authFetchJSON, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  return { enabled: !error && data?.enabled === true, loading: key !== null && isLoading && data === undefined };
}

export const PROMPT_LENS_STATUS = "/api/admin/prompt-lens/status";

/** Whether the admin's Prompt Lens is on (the menu entry, the Requests eye). */
export function usePromptLensEnabled(): boolean {
  return usePromptLensStatus().enabled;
}

// ---------------------------------------------------------------------------

export interface PromptLensViewProps {
  scope: LensScope;
  /** Filters persist per page under this key (lib/prompt-lens/filters.ts). */
  pageKey: string;
  /** The Model filter's preset. */
  presetModel?: string | null;
  /** Admin only: the Key filter's preset (from the key details page: the key
   *  and its earlier keys, Q9). */
  presetKeys?: string[] | null;
  /** Admin only: key id -> name, for the Key filter and the group headers. */
  keyNames?: Record<string, string>;
  /** Open on this request (a deep link, or a Requests-table eye): the
   *  request-history id or the stream's req_id; a new `nonce` selects again. */
  select?: { requestId: string; nonce: number } | null;
  /** The reader picked a request in the list (not Follow newest): the page
   *  puts it in its URL. */
  onPick?: (r: RequestView) => void;
}

/** How long after connecting a selected request may take to arrive before
 *  the dock says it left server memory. */
export const SELECT_WAIT_MS = 2_000;

/** How long the first connect may take before the dock says it can't connect. */
export const CONNECT_TIMEOUT_MS = 15_000;

export const FILTERS_CLEARED = "Other filters hid this request, so they were cleared.";

export const LEFT_MEMORY = "This request left server memory. Its metrics stay in the Requests table on the Stats page.";

/** A phone's touch target: at least 44 px each way. */
const TOUCH = "min-h-11 min-w-11";

const TOOL =
  "cursor-pointer rounded-[5px] border border-chat-rule bg-transparent px-[9px] py-[3px] text-[12px] text-chat-muted hover:bg-chat-surface hover:text-chat-fg aria-pressed:border-chat-accent/60 aria-pressed:text-chat-fg";

function titleOf(_scope: LensScope): string {
  return "Prompt Lens";
}

function EyeIcon() {
  return (
    <svg className="text-chat-accent-strong" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

interface HeaderProps {
  scope: LensScope;
  phone: boolean;
  scopeText: string;
  status: SseState["status"];
  rows: number;
  running: number;
  lagged: boolean;
  tools: ReactNode;
}

/** The view's header: title, scope, Live, the counts and the tools. The
 *  counts sit in reserved slots (ui.live-numbers). */
function Header({ scope, phone, scopeText, status, rows, running, lagged, tools }: HeaderProps) {
  const title = titleOf(scope);
  const word =
    status === "connected" ? "Live" : status === "connecting" || status === "reconnecting" ? "Connecting" : "Not connected";
  return (
    <div
      data-testid="prompt-lens-header"
      role="group"
      aria-label={[title, word, lagged ? "fell behind" : null, `${rows} requests, ${running} running`].filter(Boolean).join(", ")}
      className={cn(
        "flex flex-none items-center gap-3 overflow-hidden whitespace-nowrap border-b border-chat-rule px-3 sm:px-4",
        phone ? "h-11" : "h-10",
      )}
    >
      <h1 className="m-0 flex flex-none items-center gap-2 text-[14px] font-semibold text-chat-fg">
        <EyeIcon />
        {title}
      </h1>
      {!phone && <span className="min-w-0 overflow-hidden text-ellipsis text-[12px] text-chat-dim">{scopeText}</span>}
      <span className="ml-auto flex flex-none items-center gap-2.5 text-[12px] text-chat-muted">
        {status === "connected" && (
          <span className="inline-flex items-center gap-1.5 font-medium text-vw-ok-fg">
            <span aria-hidden className="vw-live-dot" />
            Live
          </span>
        )}
        {(status === "connecting" || status === "reconnecting") && <span>Connecting…</span>}
        {lagged && <span className="text-vw-amber-fg">{phone ? "fell behind" : "fell behind · some events were dropped"}</span>}
        {!phone && (
          <span data-testid="prompt-lens-counts">
            <LiveNumber value={rows} reserve={3} /> requests · <LiveNumber value={running} reserve={2} /> running
          </span>
        )}
      </span>
      {tools}
    </div>
  );
}

/** What the view says about the stream when it is not simply live. */
function streamMessage(scope: LensScope, sse: SseState): string | null {
  const key = scope === "key";
  const code = sse.errorCode;
  if (sse.status === "reconnecting") {
    if (code === 429) {
      return key
        ? "Too many live streams or tickets for this key right now (4 streams, 30 tickets a minute). Retrying shortly…"
        : "Too many stream tickets right now. Retrying shortly…";
    }
    return `Connection lost, reconnecting (${Math.min(sse.attempts, MAX_RECONNECT)}/${MAX_RECONNECT})…`;
  }
  if (sse.status !== "terminal-error") return null;
  if (code === 401 || code === 403) {
    return key
      ? "The live stream ended: your sign-in ended or expired, or the key was revoked, rotated or paused. Sign in again to watch."
      : "Your session expired. Sign in again to watch.";
  }
  if (code === 404 || code === 409) return "Prompt Lens is off on this server.";
  if (code === 429) {
    return key
      ? "Too many live streams or tickets for this key. Close Prompt Lens in another tab or device, wait a minute, then reopen it."
      : "Too many stream tickets. Wait a minute, then reopen Prompt Lens.";
  }
  if (code !== null) return `The live stream is unavailable (HTTP ${code}). Close and reopen Prompt Lens to try again.`;
  return key
    ? "The live stream ended and could not reconnect. The key may have been revoked, rotated or paused, or it already has 4 live streams open (other tabs or devices). Close and reopen Prompt Lens to try again."
    : "The live stream ended and could not reconnect. Close and reopen Prompt Lens to try again.";
}

function uniq(values: (string | null | undefined)[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const v of values) {
    if (!v || seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function optionsOf(
  scope: LensScope,
  state: DockState,
  rows: RequestView[],
  f: DockFilters,
  presetModel: string | null | undefined,
  keyNames: Record<string, string>,
): FilterOptions {
  const newest = [...rows].reverse();
  const opt = (values: string[], label: (v: string) => string = (v) => v): FilterOption[] =>
    values.map((value) => ({ value, label: label(value) }));
  const sessions = uniq([...newest.map((r) => r.sessionId), ...(f.sessions ?? [])]);
  // A model filter value is a served name or a model row id (the /stats
  // preset). Its label is the served name; a served name already standing
  // for a listed row id is not listed twice.
  const served: Record<string, string> = {};
  for (const id of state.order) {
    const r = state.byId[id];
    if (r.model && r.servedName) served[r.model] = r.servedName;
  }
  const picked = uniq([presetModel, ...(f.models ?? [])]);
  const shown = new Set(picked.map((v) => served[v] ?? v));
  const models = uniq([
    ...picked,
    ...newest.map((r) => r.servedName || r.model).filter((v) => !shown.has(v) || picked.includes(v)),
  ]);
  const modelLabel = (v: string) => served[v] ?? v;
  if (scope === "key") {
    return { keys: [], sessions: opt(sessions, (s) => s.slice(0, 8)), models: opt(models, modelLabel), ips: [] };
  }
  const labels: Record<string, string> = { ...keyNames };
  for (const id of state.order) {
    const r = state.byId[id];
    if (r.tokenId && !labels[r.tokenId]) labels[r.tokenId] = r.tokenLabel || r.tokenId.slice(0, 8);
  }
  const keys = uniq([...Object.keys(keyNames), ...(f.keys ?? []), ...newest.map((r) => r.tokenId)]);
  const ips = uniq([...(f.ips ?? []), ...newest.map((r) => r.clientIp)]);
  return {
    keys: opt(keys, (k) => labels[k] ?? k.slice(0, 8)),
    sessions: opt(sessions, (s) => s.slice(0, 8)),
    models: opt(models, modelLabel),
    ips: opt(ips),
  };
}

export function PromptLensView({
  scope,
  pageKey,
  presetModel,
  presetKeys,
  keyNames = {},
  select,
  onPick,
}: PromptLensViewProps) {
  const phone = useMediaQuery(PHONE_QUERY);
  const [filters, setFilters] = useDockFilters(pageKey, scope, {
    model: presetModel ?? null,
    keys: scope === "admin" ? presetKeys ?? null : null,
  });
  const tokenIds = scope === "admin" && filters.keys && filters.keys.length ? filters.keys : null;
  const idsKey = tokenIds ? tokenIds.join(",") : "";
  const { state, sse, dispatch, restart } = usePromptLensStream(scope, { enabled: true, tokenIds });
  // A generation per key set: a kept fetch started for an older set is
  // discarded (the stream reset its state).
  const gen = useRef(0);

  // keep_captures (G-R14): the privacy line says captures are on disk.
  const me = useMe<MeKey>(scope === "key" ? "/api/me/key" : null);
  const keepsCaptures = scope === "key" && me.data?.keep_captures === true;

  // --- kept captures: the list, for this key set ---------------------------
  useEffect(() => {
    let live = true;
    const ids = idsKey ? idsKey.split(",") : null;
    fetchKeptList(scope, ids).then(
      (kept) => {
        if (live && kept.length) dispatch({ type: "kept_list", kept });
      },
      () => {
        /* off (404/409) or rate-limited: the live stream still shows */
      },
    );
    return () => {
      live = false;
    };
  }, [scope, idsKey, dispatch]);

  const rows = useMemo(() => state.order.map((id) => state.byId[id]), [state.order, state.byId]);
  const filtered = useMemo(() => applyFilters(rows, filters, scope), [rows, filters, scope]);

  // --- selection, Follow newest, Pause --------------------------------------
  const [selected, setSelected] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [frozen, setFrozen] = useState<string[] | null>(null);
  const paused = frozen !== null;
  const order = useMemo(() => listOrder(filtered, scope), [filtered, scope]);

  const known = useRef<Set<string>>(new Set());
  useEffect(() => {
    // Grown in place (no new Set per update); rebuilt only when it gets far
    // larger than the store (evicted ids pile up).
    const fresh: string[] = [];
    for (const id of state.order) {
      if (!known.current.has(id)) {
        known.current.add(id);
        fresh.push(id);
      }
    }
    if (known.current.size > 4 * state.order.length + 1000) known.current = new Set(state.order);
    if (!follow || paused || fresh.length === 0) return;
    const ok = new Set(filtered.map((r) => r.reqId));
    let pick: RequestView | null = null;
    for (const id of fresh) {
      const r = state.byId[id];
      if (r && r.endedTs === null && ok.has(id) && (pick === null || r.startedTs > pick.startedTs)) pick = r;
    }
    if (pick) setSelected(pick.reqId);
  }, [state.order, state.byId, filtered, follow, paused]);

  // A new key set starts over (the stream resets its state): so do the
  // selection, the paused order and the kept fetches.
  const fetched = useRef<Set<string>>(new Set());
  const firstIds = useRef(true);
  useEffect(() => {
    if (firstIds.current) {
      firstIds.current = false;
      return;
    }
    gen.current += 1;
    setSelected(null);
    setFrozen(null);
    fetched.current = new Set();
  }, [idsKey]);

  // Phone: the list, or the request screen in its place (its own history
  // entry, so the back gesture returns to the list).
  const [view, setView] = useState<"list" | "request">("list");
  const backToList = useCallback(() => setView("list"), []);
  useBackLayer(phone && view === "request", backToList);

  const onPickRef = useRef(onPick);
  onPickRef.current = onPick;
  const stateRef = useRef(state);
  stateRef.current = state;
  const pick = useCallback((reqId: string) => {
    setSelected(reqId);
    setFollow(false);
    setView("request");
    const r = stateRef.current.byId[reqId];
    if (r) onPickRef.current?.(r);
  }, []);

  const newCount = useMemo(() => {
    if (!frozen) return 0;
    const f = new Set(frozen);
    return order.filter((id) => !f.has(id)).length;
  }, [frozen, order]);

  // --- select (the eye icon) -------------------------------------------------
  const [pending, setPending] = useState<string | null>(null);
  const [leftMemory, setLeftMemory] = useState(false);
  const nonce = select?.nonce;
  const selectId = select?.requestId;
  useEffect(() => {
    if (nonce === undefined || !selectId) return;
    setFilterNote(false);
    setPending(selectId);
    setLeftMemory(false);
    setFollow(false);
    setView("request");
  }, [nonce, selectId]);

  const filtersRef = useRef(filters);
  filtersRef.current = filters;
  const [filterNote, setFilterNote] = useState(false);
  useEffect(() => {
    if (pending === null) return;
    const r = findByRequestId(state, pending) ?? state.byId[pending] ?? null;
    if (!r) return;
    setPending(null);
    setLeftMemory(false);
    setSelected(r.reqId);
    // Session: the request's own. If any other filter would still hide it,
    // those are cleared (the Key filter is the server's: it streamed, so it
    // matches) and a note says so.
    let next: DockFilters = { ...filtersRef.current, sessions: r.sessionId ? [r.sessionId] : null };
    const hidden = applyFilters([r], next, scope).length === 0;
    if (hidden) next = { ...next, models: null, ips: null, phase: "all", finish: null, has: [], q: "" };
    setFilterNote(hidden);
    setFilters(next);
  }, [pending, state, setFilters, scope]);

  const connected = sse.status === "connected";
  const [timedOut, setTimedOut] = useState(false);
  // Connecting, or reconnecting with a mint that hangs: no new attempt for
  // 15 s means the stream is not coming (an attempt restarts the wait).
  const waiting = sse.status === "connecting" || sse.status === "reconnecting";
  useEffect(() => {
    if (!waiting) {
      if (sse.status === "connected") setTimedOut(false);
      return;
    }
    const t = setTimeout(() => setTimedOut(true), CONNECT_TIMEOUT_MS);
    return () => clearTimeout(t);
  }, [waiting, sse.status, sse.attempts]);
  const retry = useCallback(() => {
    setTimedOut(false);
    restart();
  }, [restart]);
  const down = sse.status === "terminal-error" || timedOut;
  useEffect(() => {
    if (pending === null || !connected) return;
    const t = setTimeout(() => setLeftMemory(true), SELECT_WAIT_MS);
    return () => clearTimeout(t);
  }, [pending, connected]);

  // --- a kept capture opened: fetch it and its sections ----------------------
  const current = selected ? state.byId[selected] ?? null : null;
  const needsKept = current !== null && current.kept && !current.hasStart ? current.reqId : null;
  useEffect(() => {
    if (needsKept === null || fetched.current.has(needsKept)) return;
    fetched.current.add(needsKept);
    const id = needsKept;
    const mine = gen.current;
    const stillMine = () => gen.current === mine;
    (async () => {
      let cap;
      try {
        cap = await fetchKeptCapture(scope, id);
      } catch {
        if (stillMine()) fetched.current.delete(id);
        return;
      }
      if (!stillMine()) return;
      dispatch({ type: "kept_capture", capture: cap });
      // Kept starts carry no prompt_parts: the prompt comes from section/prompt.
      const sizes = (cap.sections ?? {}) as Record<string, { chars?: number; count?: number } | undefined>;
      const kinds: FetchKind[] = ["prompt"];
      if ((sizes.thinking?.chars ?? 0) > 0) kinds.push("thinking");
      if ((sizes.reply?.chars ?? 0) > 0) kinds.push("reply");
      if ((sizes.tool_calls?.count ?? 0) > 0) kinds.push("tool_calls");
      await Promise.all(
        kinds.map((kind) =>
          fetchSection(scope, id, kind).then(
            (section) => {
              if (stillMine()) dispatch({ type: "section_fill", req_id: id, section });
            },
            () => {
              /* a section that failed stays empty; the pop-up refetches */
            },
          ),
        ),
      );
    })();
  }, [needsKept, scope, dispatch]);

  const [isCollapsed, setCollapsed] = useCollapseMemory();
  const collapse = useMemo(
    () => [isCollapsed, setCollapsed] as ReturnType<typeof useCollapseMemory>,
    [isCollapsed, setCollapsed],
  );

  // --- header text ------------------------------------------------------------
  const running = rows.filter((r) => r.endedTs === null).length;
  let scopeText: string;
  if (scope === "key") scopeText = "all sessions of your key · live and recent";
  else if (tokenIds) scopeText = `${tokenIds.map((k) => keyNames[k] ?? k.slice(0, 8)).join(", ")} · live and recent`;
  else scopeText = "all keys · all sessions · live and recent";

  const ringMinutes =
    scope === "admin" && state.ring && typeof state.ring.oldest_ts === "number"
      ? Math.max(0, Math.round((serverNow() - state.ring.oldest_ts) / 60))
      : null;

  const options = useMemo(
    () => optionsOf(scope, state, rows, filters, presetModel, keyNames),
    [scope, state, rows, filters, presetModel, keyNames],
  );
  const message = timedOut && waiting
    ? "Prompt Lens can't connect: the live stream has not opened. Check the connection, then retry."
    : streamMessage(scope, sse);
  const offerRetry = down;

  // A phone's header has no tools (mockup state 7): the list follows newest.
  const tools = phone ? null : (
    <span className="flex flex-none items-center gap-2">
      <button type="button" aria-pressed={follow} className={TOOL} onClick={() => setFollow((v) => !v)}>
        Follow newest
      </button>
      <button type="button" aria-pressed={paused} className={TOOL} onClick={() => setFrozen(paused ? null : order)}>
        Pause
      </button>
    </span>
  );

  let screen: ReactNode;
  if (current) {
    screen = (
      <>
        {filterNote && (
          <p data-testid="prompt-lens-filter-note" role="status" className="mb-2 text-[12px] text-vw-amber-fg">
            {FILTERS_CLEARED}
          </p>
        )}
        <MemoScreen
          request={current}
          scope={scope}
          phone={phone}
          collapse={collapse}
          keepsCaptures={keepsCaptures}
          onBack={phone ? backToList : undefined}
        />
      </>
    );
  } else if (pending !== null && down) {
    screen = <p className="text-[13px] text-chat-muted">Can&apos;t open the request: Prompt Lens is not connected.</p>;
  } else if (leftMemory || (selected !== null && pending === null)) {
    screen = <p className="text-[13px] text-chat-muted">{LEFT_MEMORY}</p>;
  } else if (pending !== null) {
    screen = <p className="text-[13px] text-chat-dim">Looking for the request…</p>;
  } else {
    screen = <p className="text-[13px] text-chat-dim">Pick a request in the list to see its prompt and reply.</p>;
  }

  if (phone && !current) {
    screen = (
      <>
        <button type="button" onClick={backToList} className={cn(TOOL, TOUCH, "mb-2 text-chat-fg")}>
          ‹ Requests
        </button>
        {screen}
      </>
    );
  }
  const showList = !phone || view === "list";
  const showScreen = !phone || view === "request";

  return (
    <section
      data-testid="prompt-lens-view"
      aria-label={titleOf(scope)}
      className="flex h-full min-h-0 flex-col overflow-hidden bg-vw-dock font-sans"
    >
      <Header
        scope={scope}
        phone={phone}
        scopeText={scopeText}
        status={sse.status}
        rows={rows.length}
        running={running}
        lagged={state.lagged}
        tools={tools}
      />
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <FilterLine
          scope={scope}
          phone={phone}
          filters={filters}
          onChange={setFilters}
          options={options}
          shown={filtered.length}
          total={rows.length}
          ringMinutes={ringMinutes}
        />
        <div className="relative flex min-h-0 flex-1">
          {message && (
            <div
              data-testid="prompt-lens-status"
              role="status"
              className="absolute inset-x-0 top-0 z-[3] border-b border-vw-amber-fg/40 bg-vw-dock px-3 py-1.5 text-[12px] text-vw-amber-fg"
            >
              <span>{message}</span>
              {offerRetry && (
                <button
                  type="button"
                  onClick={retry}
                  className="ml-2 rounded border border-vw-amber-fg/50 px-2 py-0.5 text-[12px] text-vw-amber-fg hover:bg-vw-amber-fg/10"
                >
                  Retry
                </button>
              )}
            </div>
          )}
          {showList && (
            <div
              data-testid="prompt-lens-list"
              className={cn(
                "overflow-y-auto",
                phone ? "min-w-0 flex-1 overscroll-contain" : "flex-none border-r border-chat-rule",
              )}
              style={phone ? undefined : { width: LIST_WIDTH_PX }}
            >
              <RequestList
                scope={scope}
                rows={filtered}
                frozen={frozen}
                newCount={newCount}
                onShowNew={() => setFrozen(order)}
                selected={selected}
                onSelect={pick}
                keyNames={scope === "admin" ? keyNames : undefined}
              />
            </div>
          )}
          {showScreen && (
            <div
              data-testid="prompt-lens-screen"
              className={cn("min-w-0 flex-1 overflow-y-auto", phone ? "overscroll-contain px-3 pb-6 pt-2" : "p-3")}
            >
              {screen}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
