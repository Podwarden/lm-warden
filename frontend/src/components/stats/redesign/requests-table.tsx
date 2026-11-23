"use client";

// The live Requests table (design 3.8 B), at the bottom of the page. It replaces
// In flight, By token, By client IP and Tokens per API key. Fixed height: 8 rows
// plus header (6 on phones), ghost rows when fewer, scroll inside when more, the
// empty text inside the box. Group by is the page's only breakdown tool; a key
// group never mixes models. The key holder's variant has no Key, Client IP or
// Install column. With Prompt Lens on, each request row gets an eye icon that
// opens that request on the Prompt Lens page (spec 2026-11-22 Q1: the table
// stays the metrics list, Prompt Lens is the content list).

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Eye } from "lucide-react";

import type { LiveRequestRow } from "@/lib/live-stats";
import { sessionHref } from "@/lib/routes";
import {
  FIXED,
  type GroupBy,
  type MeLiveRow,
  type SpeedGroup,
  fmtAgo,
  fmtElapsed,
  fmtRate,
  fmtSeconds,
  fmtTokens,
  groupLive,
} from "@/lib/stats-redesign";
import { PHONE_QUERY } from "@/lib/use-media-query";
import { cn } from "@/lib/utils";

import { Seg, Swatch, hueOf } from "./primitives";

const FIN_SHOW_MS = 5_000;
const RECENT_MS = 60_000;

type Fin<T> = T & { _fin?: number };

/** Rows that just left the server's snapshot stay as "finished" for a while. */
export function useWithFinished<T extends { id: string }>(rows: readonly T[], keepMs: number): Fin<T>[] {
  const gone = useRef(new Map<string, Fin<T>>());
  const last = useRef(new Map<string, T>());
  const [, tick] = useState(0);
  const now = Date.now();
  const ids = new Set(rows.map((r) => r.id));
  for (const [id, r] of last.current) {
    if (!ids.has(id) && !gone.current.has(id)) gone.current.set(id, { ...r, _fin: now });
  }
  for (const id of ids) gone.current.delete(id);
  for (const [id, r] of gone.current) if (now - (r._fin ?? now) > keepMs) gone.current.delete(id);
  last.current = new Map(rows.map((r) => [r.id, r]));
  useEffect(() => {
    if (gone.current.size === 0) return;
    const t = setTimeout(() => tick((n) => n + 1), 1000);
    return () => clearTimeout(t);
  });
  return [...rows, ...gone.current.values()];
}

interface Col<R> {
  head: string;
  width: string;
  cell: (r: R) => React.ReactNode;
  cls?: string;
  /** Hidden below 768 px. */
  opt?: boolean;
  /** The grid width below 768 px, when it differs from `width`. */
  narrow?: string;
}

/** A request whose client did not ask for a stream: the proxy sees no token
 *  until the reply is complete. Once the engine admitted it, it is running,
 *  never "waiting" (owner report 2026-11-22). */
export function noStream(r: { phase: string; is_stream?: boolean }): boolean {
  return r.phase === "running_no_stream" || (r.is_stream === false && r.phase === "waiting");
}

const NO_STREAM_TIP =
  "The client did not ask for streaming, so tokens appear only at the end. The request is with the engine.";

function PhasePill({ phase, fin, text, title }: { phase: string; fin?: boolean; text?: string; title?: string }) {
  const tone =
    fin
      ? "border-chat-rule text-chat-muted"
      : phase === "answering" || phase === "decode"
        ? "border-emerald-800 text-emerald-300"
        : phase === "running_no_stream"
          ? "border-emerald-900 text-emerald-200"
          : phase === "thinking"
          ? "border-blue-800 text-blue-300"
          : phase === "prefill"
            ? "border-amber-800 text-amber-300"
            : phase === "queued"
              ? "border-red-900 text-red-400"
              : phase === "tool_call"
                ? "border-purple-800 text-purple-200"
                : "border-chat-rule text-chat-muted";
  const label =
    phase === "running_no_stream" ? "running · not streamed" : phase === "tool_call" ? "tool call" : phase === "decode" ? "answering" : phase;
  return (
    <span title={title} className={cn("inline-block rounded border px-1.5 py-px text-[11px]", tone)}>
      {text ?? (fin ? "✓ done" : label)}
    </span>
  );
}

function Ctx({ pct, cachedPct }: { pct: number | null | undefined; cachedPct?: number | null }) {
  return (
    <div className="relative h-1.5 overflow-hidden rounded bg-chat-surface-2">
      <span className="absolute inset-y-0 left-0 bg-emerald-400" style={{ width: `${Math.min(100, (pct ?? 0) * 100)}%` }} />
      {cachedPct != null && <em className="absolute inset-y-0 left-0 bg-sky-400" style={{ width: `${Math.min(100, cachedPct * 100)}%` }} />}
    </div>
  );
}

function Table<R>({
  cols,
  rows,
  rowKey,
  rowClass,
  empty,
  testid,
  footLeft,
}: {
  cols: Col<R>[];
  rows: readonly R[];
  rowKey: (r: R) => string;
  rowClass?: (r: R) => string;
  empty: string;
  testid: string;
  footLeft: string;
}) {
  // Phones keep Status, Key, Phase, tok/s and Elapsed (with Prompt Lens on:
  // the eye in place of Elapsed, spec 2026-11-22 section 6); a tap opens
  // every column of that row in an overlay (design 3.8).
  const [detail, setDetail] = useState<string | null>(null);
  const openRow = (r: R) => {
    if (typeof window !== "undefined" && window.matchMedia?.(PHONE_QUERY).matches) setDetail(rowKey(r));
  };
  const shown = detail ? rows.find((r) => rowKey(r) === detail) : undefined;
  const tpl = cols.map((c) => c.width).join(" ");
  const tplNarrow = cols.filter((c) => !c.opt).map((c) => c.narrow ?? c.width).join(" ");
  const ghosts = Math.max(0, FIXED.tableRows - rows.length);
  const style = { "--tpl": tpl, "--tpl-n": tplNarrow } as React.CSSProperties;
  const grid = "grid items-center gap-x-2 px-2.5 [grid-template-columns:var(--tpl-n)] md:[grid-template-columns:var(--tpl)]";
  const cell = (c: Col<R>) => cn("min-w-0 overflow-hidden text-ellipsis whitespace-nowrap tabular-nums", c.cls, c.opt && "hidden md:block");
  return (
    <div data-testid={testid} className="relative overflow-hidden rounded-lg border border-chat-rule bg-chat-surface" style={style}>
      <div role="row" className={cn(grid, "h-8 border-b border-chat-rule text-[10px] uppercase tracking-[0.07em] text-chat-dim")}>
        {cols.map((c, i) => (
          <div key={i} role="columnheader" className={cell(c)}>
            {c.head}
          </div>
        ))}
      </div>
      <div
        data-testid={`${testid}-body`}
        data-rows={FIXED.tableRows}
        className="relative h-[264px] overflow-y-auto md:h-[304px]"
      >
        {rows.map((r) => (
          <div
            key={rowKey(r)}
            role="row"
            data-testid="req-row"
            onClick={() => openRow(r)}
            className={cn(grid, "h-11 border-b border-chat-rule/30 text-xs transition-opacity max-md:cursor-pointer md:h-[38px]", rowClass?.(r))}
          >
            {cols.map((c, i) => (
              <div key={i} className={cell(c)}>
                {c.cell(r)}
              </div>
            ))}
          </div>
        ))}
        {Array.from({ length: ghosts }, (_, i) => (
          <div key={`g${i}`} data-testid="ghost-row" aria-hidden="true" className={cn(grid, "h-11 border-b border-chat-rule/30 opacity-25 md:h-[38px]")}>
            {cols.map((c, j) => (
              <div key={j} className={cn("h-1.5 w-3/5 rounded bg-chat-surface-2", c.opt && "hidden md:block")} />
            ))}
          </div>
        ))}
        {rows.length === 0 && (
          <div data-testid="table-empty" className="pointer-events-none absolute inset-x-0 top-[76px] text-center text-[13px] text-chat-dim">
            {empty}
          </div>
        )}
        {rows.length > FIXED.tableRows && (
          <div className="pointer-events-none sticky bottom-2 float-right mr-2.5 rounded-full border border-chat-rule bg-chat-surface-2 px-2 text-[11px] text-chat-muted">
            +{rows.length - FIXED.tableRows} more · scroll
          </div>
        )}
      </div>
      <div className="flex flex-wrap justify-between gap-2 border-t border-chat-rule px-2.5 py-1.5 text-[11px] text-chat-dim">
        <span>{footLeft}</span>
        <span>fixed height · {FIXED.tableRows} rows (6 on phones) · tap a row on a phone for details</span>
      </div>
      {shown && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Request details"
          data-testid="row-details"
          className="fixed inset-0 z-[80] flex items-end justify-center bg-black/60 p-3"
          onClick={(e) => e.target === e.currentTarget && setDetail(null)}
        >
          <div className="w-full max-w-[480px] rounded-xl border border-chat-rule bg-chat-surface p-4">
            <button type="button" onClick={() => setDetail(null)} className="float-right rounded border border-chat-rule px-2 text-chat-muted">
              Close
            </button>
            <dl className="mt-6 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-xs">
              {cols
                .filter((c) => c.head)
                .map((c, i) => (
                  <div key={i} className="contents">
                    <dt className="text-[10px] uppercase tracking-[0.06em] text-chat-dim">{c.head}</dt>
                    <dd className="min-w-0 tabular-nums">{c.cell(shown)}</dd>
                  </div>
                ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}

function ttftCell(ttft: number | null | undefined, total: number | null | undefined, queued: number | null | undefined) {
  if (total == null) return <span className="text-chat-dim">-</span>;
  return <span title={`warden queue ${fmtSeconds(queued ?? 0)} + engine ${fmtSeconds(ttft ?? 0)}`}>{fmtSeconds(total)}</span>;
}

function p50(v: number | null, n: number, fmt: (x: number) => string) {
  return v != null ? fmt(v) : <span className="text-chat-dim" title="fewer than 5 requests">{n ? "n<5" : "-"}</span>;
}

/** The eye column (Q1): opens the row's request on the Prompt Lens page. Kept
 *  on phones (not optional, a 44 px target there), and the click never opens
 *  the phone details. */
function eyeCol<R extends { id: string }>(onEye: (requestId: string) => void): Col<R> {
  return {
    head: "",
    width: "30px",
    narrow: "44px",
    cell: (r) => (
      <button
        type="button"
        aria-label="Open in Prompt Lens"
        title="Open in Prompt Lens"
        onClick={(e) => {
          e.stopPropagation();
          onEye(r.id);
        }}
        className="inline-flex h-6 w-6 items-center justify-center rounded text-chat-muted hover:bg-chat-surface-2 hover:text-chat-accent max-md:h-11 max-md:w-11"
      >
        <Eye aria-hidden className="h-3.5 w-3.5" />
      </button>
    ),
  };
}

// ---- admin -----------------------------------------------------------------

export function AdminRequestsTable({
  rows,
  scopeModel,
  multiModel,
  windowGroups,
  onGroupChange,
  lastFinishedAt,
  range,
  onEye,
}: {
  rows: readonly LiveRequestRow[];
  scopeModel: string | null;
  multiModel: boolean;
  /** /api/stats/v2/speed?group= for the chosen Group by (window columns). */
  windowGroups: SpeedGroup[] | null | undefined;
  onGroupChange: (g: GroupBy) => void;
  lastFinishedAt: number | null;
  range: string;
  /** Prompt Lens on: an eye per request row opens it on the Prompt Lens page. */
  onEye?: (requestId: string) => void;
}) {
  const [group, setGroup] = useState<GroupBy>("none");
  const [rowsMode, setRowsMode] = useState<"all" | "scope">("all"); // D9: everything by default
  const [show, setShow] = useState<"live" | "recent">("live");
  const [paused, setPaused] = useState(false);
  const frozen = useRef<string[] | null>(null);
  const withFin = useWithFinished(rows, show === "recent" ? RECENT_MS : FIN_SHOW_MS);
  const filtered = withFin.filter((r) => rowsMode === "all" || r.model === scopeModel);
  const live = filtered.filter((r) => !r._fin);
  const base = [...filtered].sort((a, b) => Number(!!a._fin) - Number(!!b._fin) || b.elapsed_s - a.elapsed_s);
  if (!paused) frozen.current = null;
  else if (!frozen.current) frozen.current = base.map((r) => r.id);
  const order = frozen.current;
  const ordered = order
    ? base.sort((a, b) => {
        const ia = order.indexOf(a.id);
        const ib = order.indexOf(b.id);
        return (ia < 0 ? 1e9 : ia) - (ib < 0 ? 1e9 : ib);
      })
    : base;

  const choose = (g: GroupBy) => {
    setGroup(g);
    onGroupChange(g);
  };

  const groupOpts = [
    { v: "none" as const, label: "None" },
    { v: "key" as const, label: "Key" },
    { v: "ip" as const, label: "Client IP" },
    ...(multiModel ? [{ v: "model" as const, label: "Model" }] : []),
  ];

  let table: React.ReactNode;
  if (group === "none") {
    const cols: Col<Fin<LiveRequestRow>>[] = [
      {
        head: "",
        width: "18px",
        cell: (r) => (
          <span
            className={cn("inline-block h-[7px] w-[7px] rounded-full", r._fin ? "bg-chat-dim" : r.phase === "queued" ? "bg-red-400" : "bg-emerald-400")}
          />
        ),
      },
      {
        head: "Key",
        width: "minmax(90px,1.1fr)",
        narrow: "minmax(0,1fr)",
        cell: (r) => (
          <span className="inline-flex max-w-full items-center gap-1.5">
            <Swatch color={hueOf(r.token_id ?? "")} />
            <span className="overflow-hidden text-ellipsis">{r.token_name ?? "anonymous"}</span>
          </span>
        ),
      },
      { head: "Client IP", width: "92px", cell: (r) => <span className="font-mono">{r.client_ip ?? "-"}</span>, opt: true },
      {
        head: "Session",
        width: "72px",
        opt: true,
        cell: (r) =>
          r.session_id ? (
            r.session_ref ? (
              <Link className="font-mono text-chat-accent" href={sessionHref(r.session_ref)}>
                {r.session_id.slice(0, 8)}
              </Link>
            ) : (
              <span className="font-mono">{r.session_id.slice(0, 8)}</span>
            )
          ) : (
            <span className="text-chat-dim">-</span>
          ),
      },
      {
        head: "Model",
        width: "minmax(80px,1fr)",
        opt: true,
        cell: (r) => (
          <span className="inline-flex max-w-full items-center gap-1.5">
            <Swatch color={hueOf(r.model)} />
            <span className="overflow-hidden text-ellipsis font-mono">{r.model}</span>
          </span>
        ),
      },
      {
        head: "Phase",
        width: "112px",
        narrow: "minmax(0,1fr)",
        cell: (r) =>
          noStream(r) ? (
            <PhasePill phase="running_no_stream" fin={!!r._fin} title={NO_STREAM_TIP} />
          ) : (
            <PhasePill phase={r.phase} fin={!!r._fin} />
          ),
      },
      {
        head: "Prompt (cached)",
        width: "104px",
        cls: "text-right",
        opt: true,
        cell: (r) => (
          <>
            {fmtTokens(r.prompt_tokens)}{" "}
            {r.cache_est_tokens != null && <span className="text-[10px] text-chat-dim">({fmtTokens(r.cache_est_tokens)} est.)</span>}
          </>
        ),
      },
      {
        head: "Out",
        width: "52px",
        cls: "text-right",
        opt: true,
        cell: (r) =>
          noStream(r) && !r._fin ? (
            r.engine_progress_tokens != null ? (
              <span title="the engine's own count (the client did not stream)">{fmtTokens(r.engine_progress_tokens)}</span>
            ) : (
              <span className="text-chat-dim" title={NO_STREAM_TIP}>
                -
              </span>
            )
          ) : (
            fmtTokens(r.out_tokens ?? r.completion_tokens)
          ),
      },
      {
        head: "TTFT",
        width: "58px",
        cls: "text-right",
        opt: true,
        cell: (r) =>
          noStream(r) && r.ttft_total_s == null ? (
            <span className="text-chat-dim" title={NO_STREAM_TIP}>
              no stream
            </span>
          ) : (
            ttftCell(r.ttft_s, r.ttft_total_s, r.queued_s)
          ),
      },
      {
        head: "tok/s",
        width: "52px",
        cls: "text-right",
        cell: (r) => (r.decode_tps_5s == null || r._fin ? <span className="text-chat-dim">-</span> : Math.round(r.decode_tps_5s)),
      },
      {
        head: "Context",
        width: "minmax(60px,.7fr)",
        opt: true,
        cell: (r) => <Ctx pct={r.context_pct} cachedPct={r.cache_est_tokens != null && r.max_model_len ? r.cache_est_tokens / r.max_model_len : null} />,
      },
      // With the eye on a phone, Elapsed gives it the room (spec 2026-11-22 section 6).
      { head: "Elapsed", width: "52px", cls: "text-right", opt: !!onEye, cell: (r) => fmtElapsed(r.elapsed_s) },
    ];
    if (onEye) cols.push(eyeCol(onEye));
    table = (
      <Table
        testid="live-requests-table"
        cols={cols}
        rows={ordered}
        rowKey={(r) => r.id}
        rowClass={(r) => (r._fin ? "opacity-45" : "")}
        empty={lastFinishedAt == null ? "No requests in flight · no requests yet" : `No requests in flight · last request ${fmtAgo(lastFinishedAt)}`}
        footLeft="Out and tok/s count real tokens, thinking and tool calls included"
      />
    );
  } else {
    const liveGroups = groupLive(live, group, multiModel);
    // Rows = Scope only narrows the window groups too, not only the live ones.
    const win = (windowGroups ?? []).filter((g) => rowsMode === "all" || g.model === scopeModel || group === "ip");
    const winKey = (g: SpeedGroup) => `${g.key ?? "-"}|${group === "key" && multiModel ? (g.model ?? "") : ""}`;
    const ids = new Map<string, { label: string; model: string | null }>();
    for (const g of win) ids.set(winKey(g), { label: g.label ?? "anonymous", model: group === "key" && multiModel ? g.model : null });
    for (const g of liveGroups) if (!ids.has(g.id)) ids.set(g.id, { label: g.label, model: g.model });
    const rowsG = [...ids.entries()].map(([id, meta]) => ({
      id,
      ...meta,
      live: liveGroups.find((g) => g.id === id),
      win: win.find((g) => winKey(g) === id),
    }));
    rowsG.sort((a, b) => (b.win?.requests ?? 0) - (a.win?.requests ?? 0) || a.label.localeCompare(b.label));
    type G = (typeof rowsG)[number];
    const first: Col<G> = {
      head: group === "key" ? "Key" : group === "ip" ? "Client IP" : "Model",
      width: "minmax(110px,1.3fr)",
      cell: (g) => (
        <span className="inline-flex max-w-full items-center gap-1.5">
          {group !== "ip" && <Swatch color={hueOf(g.id.split("|")[0])} />}
          <span className={cn("overflow-hidden text-ellipsis", group !== "key" && "font-mono")}>{g.label}</span>
        </span>
      ),
    };
    const cols: Col<G>[] = [first];
    if (group === "key" && multiModel) cols.push({ head: "Model", width: "minmax(110px,1.2fr)", opt: true, cell: (g) => <span className="font-mono">{g.model}</span> });
    cols.push(
      { head: "In flight", width: "64px", cls: "text-right", cell: (g) => g.live?.inFlight ?? 0 },
      {
        head: "Live tok/s",
        width: "72px",
        cls: "text-right",
        cell: (g) => (g.live?.liveTps != null ? Math.round(g.live.liveTps) : <span className="text-chat-dim">idle</span>),
      },
      { head: `Req (${range})`, width: "66px", cls: "text-right", opt: true, cell: (g) => g.win?.requests ?? 0 },
    );
    if (group === "ip") {
      cols.push({ head: "Context in flight", width: "110px", cls: "text-right", opt: true, cell: (g) => fmtTokens(g.live?.context ?? 0) });
    } else {
      cols.push(
        { head: "Prompt", width: "64px", cls: "text-right", opt: true, cell: (g) => fmtTokens(g.win?.prompt_tokens ?? 0) },
        { head: "p50 TTFT", width: "70px", cls: "text-right", opt: true, cell: (g) => p50(g.win?.p50_ttft ?? null, g.win?.n_ttft ?? 0, fmtSeconds) },
        { head: "p50 decode", width: "86px", cls: "text-right", cell: (g) => p50(g.win?.p50_decode ?? null, g.win?.n_decode ?? 0, (v) => `${fmtRate(v)} tok/s`) },
        { head: "p50 prefill", width: "86px", cls: "text-right", opt: true, cell: (g) => p50(g.win?.p50_prefill ?? null, g.win?.n_prefill ?? 0, (v) => `${fmtRate(v)} tok/s`) },
      );
    }
    table = (
      <Table
        testid="live-requests-table"
        cols={cols}
        rows={rowsG}
        rowKey={(g) => g.id}
        rowClass={() => "bg-chat-surface-2/30 font-medium"}
        empty={
          lastFinishedAt == null
            ? "No requests in flight or in the window · no requests yet"
            : `No requests in flight or in the window · last request ${fmtAgo(lastFinishedAt)}`
        }
        footLeft={multiModel && group === "key" ? "One row per group and model: speeds are never mixed across models" : `Live columns: now. Window columns: last ${range}`}
      />
    );
  }

  return (
    <div data-testid="requests-live">
      <div className="mb-2 flex flex-wrap items-center gap-x-3.5 gap-y-2">
        <span className="text-[10px] uppercase tracking-[0.08em] text-chat-dim">Group by</span>
        <Seg label="Group by" testid="group-by" value={group} options={groupOpts} onChange={choose} />
        {multiModel && (
          <>
            <span className="text-[10px] uppercase tracking-[0.08em] text-chat-dim">Rows</span>
            <Seg
              label="Rows"
              value={rowsMode}
              options={[
                { v: "all", label: "Everything" },
                { v: "scope", label: "Scope only" },
              ]}
              onChange={setRowsMode}
            />
          </>
        )}
        <span className="text-[10px] uppercase tracking-[0.08em] text-chat-dim">Show</span>
        <Seg
          label="Show"
          value={show}
          options={[
            { v: "live", label: "In flight" },
            { v: "recent", label: "+ recent" },
          ]}
          onChange={setShow}
        />
        <button
          type="button"
          aria-pressed={paused}
          onClick={() => setPaused((p) => !p)}
          title="Freeze row order; values keep updating"
          className={cn("rounded-md border border-chat-rule px-2.5 py-0.5 text-xs", paused ? "bg-chat-surface-2 text-chat-accent" : "text-chat-muted")}
        >
          {paused ? "Order paused" : "Pause order"}
        </button>
        <span className="ml-auto text-[11px] tabular-nums text-chat-dim">
          {live.length} in flight · {live.filter((r) => r.phase === "queued").length} queued
        </span>
      </div>
      {table}
    </div>
  );
}

// ---- key holder --------------------------------------------------------------

export function KeyRequestsTable({
  rows,
  multiModel,
  lastFinishedAt,
  onEye,
}: {
  rows: readonly MeLiveRow[];
  multiModel: boolean;
  lastFinishedAt: number | null;
  /** Prompt Lens on for this key: an eye per row opens it on the Prompt Lens page. */
  onEye?: (requestId: string) => void;
}) {
  const withFin = useWithFinished(rows, FIN_SHOW_MS);
  const ordered = [...withFin].sort((a, b) => Number(!!a._fin) - Number(!!b._fin));
  const cols: Col<Fin<MeLiveRow>>[] = [
    {
      head: "",
      width: "18px",
      cell: (r) => (
        <span
          className={cn(
            "inline-block h-[7px] w-[7px] rounded-full",
            r._fin ? "bg-chat-dim" : r.phase === "queued" || r.phase === "prefill" ? "bg-amber-400" : "bg-emerald-400",
          )}
        />
      ),
    },
    { head: "Session", width: "72px", opt: true, cell: (r) => <span className="font-mono">{r.session_id ? r.session_id.slice(0, 8) : "-"}</span> },
  ];
  if (multiModel) cols.push({ head: "Model", width: "minmax(80px,1fr)", opt: true, cell: (r) => <span className="font-mono">{r.model}</span> });
  cols.push(
    {
      head: "Phase",
      width: "120px",
      narrow: "minmax(0,1fr)",
      cell: (r) => {
        if (noStream(r)) return <PhasePill phase="running_no_stream" fin={!!r._fin} title={NO_STREAM_TIP} />;
        const waiting = !r._fin && (r.phase === "queued" || r.phase === "prefill");
        // D6: an estimate from the key's own requests only; never "N ahead".
        const left = r.expected_start_s != null ? Math.max(1, Math.round(r.expected_start_s - r.elapsed_s)) : null;
        return <PhasePill phase={r.phase} fin={!!r._fin} text={waiting ? (left != null ? `waiting ~${left} s` : "waiting") : undefined} />;
      },
    },
    { head: "Prompt", width: "70px", cls: "text-right", opt: true, cell: (r) => fmtTokens(r.prompt_tokens) },
    {
      head: "Out",
      width: "52px",
      cls: "text-right",
      opt: true,
      cell: (r) =>
        noStream(r) && !r._fin ? (
          r.engine_progress_tokens != null ? (
            fmtTokens(r.engine_progress_tokens)
          ) : (
            <span className="text-chat-dim" title={NO_STREAM_TIP}>
              -
            </span>
          )
        ) : (
          fmtTokens(r.out_tokens)
        ),
    },
    { head: "TTFT", width: "58px", cls: "text-right", opt: true, cell: (r) => ttftCell(r.ttft_s, r.ttft_total_s, r.queued_s) },
    { head: "tok/s", width: "52px", cls: "text-right", cell: (r) => (r.decode_tps_5s == null || r._fin ? <span className="text-chat-dim">-</span> : Math.round(r.decode_tps_5s)) },
    { head: "Context", width: "minmax(60px,.7fr)", opt: true, cell: (r) => <Ctx pct={r.context_pct} /> },
    { head: "Elapsed", width: "52px", cls: "text-right", opt: !!onEye, cell: (r) => fmtElapsed(r.elapsed_s) },
  );
  if (onEye) cols.push(eyeCol(onEye));
  return (
    <div data-testid="key-requests-live">
      <div className="mb-2 flex items-center gap-3">
        <span className="text-[10px] uppercase tracking-[0.08em] text-chat-dim">Your requests now</span>
        <span className="ml-auto text-[11px] tabular-nums text-chat-dim">{rows.length} of your requests in flight</span>
      </div>
      <Table
        testid="live-requests-table"
        cols={cols}
        rows={ordered}
        rowKey={(r) => r.id}
        rowClass={(r) => (r._fin ? "opacity-45" : "")}
        empty={
          lastFinishedAt == null
            ? "None of your requests are running · no requests from this key yet"
            : `None of your requests are running · your last request finished ${fmtAgo(lastFinishedAt)}`
        }
        footLeft="Waiting time is estimated from your own requests only"
      />
    </div>
  );
}
