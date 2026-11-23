"use client";

// The Prompt Lens request list (spec 2026-11-22 section 2.3, mockup states
// 1-2): the dock's requests grouped by session.
//
//   * Groups: by session (admin: by key and session). A sticky header carries
//     the key name and client IP (admin only), the session id8, the turns the
//     list holds and the conv id4. Groups are ordered by their newest row;
//     inside a group live rows come first, then newest first.
//   * Rows: a fixed 46 px. Status dot, turn and time, "non-streamed",
//     "kept", the phase chip or the finish, in/out tokens, live tok/s or the
//     duration, and an 84 px mini kvsig bar of the exact / loose / other-key
//     shares of the prompt. The other-key share is never drawn for a key
//     holder (G-R11): it reads as new.
//   * Pause: the caller passes the frozen order (`frozen`); rows keep it and
//     new ones wait behind the "N new" pill.
//
// "Finished" is `endedTs !== null`, never the phase (a kept row reads
// `queued` until it is filled).

import { memo } from "react";

import type { RequestView } from "@/lib/prompt-lens/store";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { serverNow } from "@/lib/prompt-lens/server-clock";
import { LiveNumber } from "@/components/ui/live-number";
import { cn } from "@/lib/utils";
import { STATE_CLASS, stateStyle } from "./kvsig-map";

export const LIST_ROW_PX = 46;
export const LIST_WIDTH_PX = 360;
export const MINI_BAR_PX = 84;

export const LIST_FOOTER =
  "Older requests left server memory. Their metrics stay in the Requests table on the Stats page.";

const BAD_FINISH = new Set(["length", "abort", "error", "runaway"]);

const PHASE_WORD: Record<string, string> = {
  queued: "queued",
  prefill: "prefill",
  thinking: "thinking",
  answering: "answering",
  tool_call: "tool call",
  running: "running",
};

export interface RowGroup {
  key: string;
  tokenId: string | null;
  sessionId: string | null;
  rows: RequestView[];
}

const live = (r: RequestView) => r.endedTs === null;

function groupKey(r: RequestView, scope: LensScope): string {
  const s = r.sessionId ?? "";
  return scope === "admin" ? `${r.tokenId ?? ""}|${s}` : s;
}

/** Rows grouped by session (admin: key and session). Unless `keepOrder`,
 *  groups go by their newest row and rows live first, then newest first;
 *  with it, rows keep the order given (a paused list). */
export function groupRows(rows: RequestView[], scope: LensScope, keepOrder = false): RowGroup[] {
  const byKey = new Map<string, RowGroup>();
  for (const r of rows) {
    const k = groupKey(r, scope);
    let g = byKey.get(k);
    if (!g) {
      g = { key: k, tokenId: r.tokenId, sessionId: r.sessionId, rows: [] };
      byKey.set(k, g);
    }
    g.rows.push(r);
  }
  const groups = [...byKey.values()];
  if (keepOrder) return groups;
  for (const g of groups) {
    g.rows.sort((a, b) => Number(live(b)) - Number(live(a)) || b.startedTs - a.startedTs);
  }
  const newest = (g: RowGroup) => Math.max(...g.rows.map((r) => r.startedTs));
  return groups.sort((a, b) => newest(b) - newest(a));
}

/** The list's order, flattened (what Pause freezes). */
export function listOrder(rows: RequestView[], scope: LensScope): string[] {
  return groupRows(rows, scope).flatMap((g) => g.rows.map((r) => r.reqId));
}

/** exact / loose / other-key shares of the prompt's chars (other: admin only). */
export function miniShares(r: RequestView, scope: LensScope): { exact: number; loose: number; other: number } {
  const parts = r.kvsig?.available ? r.kvsig.parts ?? [] : [];
  let total = 0;
  const by = { exact: 0, loose: 0, other: 0 };
  for (const p of parts) {
    const n = Math.max(0, p.chars ?? 0);
    total += n;
    if (p.state === "exact" || p.state === "loose") by[p.state] += n;
    else if (p.state === "other" && scope === "admin") by.other += n;
  }
  if (total <= 0) return { exact: 0, loose: 0, other: 0 };
  return { exact: by.exact / total, loose: by.loose / total, other: by.other / total };
}

/** Row counter slots, in characters: "≈99.9k" tokens, "≈999 tok/s" or
 *  "wait 9 min 59 s". */
const ROW_TOK_CH = 6;
const ROW_RATE_CH = 10;

const fmtTok = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n)));

function hms(ts: number): string {
  const d = new Date(ts * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function secs(x: number): string {
  if (!Number.isFinite(x) || x < 0) return "—";
  if (x < 10) return `${Number(x.toFixed(1))} s`;
  if (x < 60) return `${Math.round(x)} s`;
  const s = Math.floor(x);
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

function tokensIn(r: RequestView): string {
  if (r.end) return fmtTok(r.end.prompt_tokens);
  if (typeof r.keptTokens?.prompt === "number") return fmtTok(r.keptTokens.prompt);
  const chars = r.prompt?.total_chars || r.promptFlat.length;
  return chars ? `≈${fmtTok(chars / 4)}` : "—";
}

function tokensOut(r: RequestView): string {
  if (r.end) return fmtTok(r.end.completion_tokens);
  if (typeof r.keptTokens?.completion === "number") return fmtTok(r.keptTokens.completion);
  if (r.endedTs !== null) return "—";
  const frames = r.frames.reasoning + r.frames.content + r.frames.tool_call;
  return `≈${fmtTok(r.progressTokens ?? frames)}`;
}

/** Live: ≈ tok/s over thinking's and the reply's span (or the queue wait);
 *  finished: the duration. */
function rateOrDuration(r: RequestView): string {
  if (r.endedTs !== null) {
    const d = r.end?.duration_s ?? r.endedTs - r.startedTs;
    return d > 0 ? secs(d) : "—";
  }
  if (r.phase === "queued") return `wait ${secs(Math.max(0, serverNow() - r.startedTs))}`;
  const first = [r.thinking.firstTs, r.reply.firstTs].filter((x): x is number => x !== null);
  const last = [r.thinking.lastTs, r.reply.lastTs].filter((x): x is number => x !== null);
  const frames = r.frames.reasoning + r.frames.content;
  if (!first.length || !last.length) return "—";
  const span = Math.max(...last) - Math.min(...first);
  return span > 0 && frames > 1 ? `≈${Math.round(frames / span)} tok/s` : "—";
}

const CHIP = "rounded border px-[5px] text-[10px] leading-4";

const pct = (x: number) => `${Math.round(x * 100)}%`;

/** The mini bar's accessible name; the key holder's never mentions another key. */
export function miniLabel(sh: { exact: number; loose: number; other: number }, scope: LensScope): string {
  const base = `kvsig: ${pct(sh.exact)} exact, ${pct(sh.loose)} loose`;
  return scope === "admin" ? `${base}, ${pct(sh.other)} another key's` : base;
}

/** Memoised on the row's object: the reducer replaces only a changed one. */
const Row = memo(function Row({
  r,
  scope,
  selected,
  onSelect,
}: {
  r: RequestView;
  scope: LensScope;
  selected: boolean;
  onSelect: (reqId: string) => void;
}) {
  const ended = r.endedTs !== null;
  const bad = ended && r.finish !== null && BAD_FINISH.has(r.finish);
  const dot = !ended
    ? r.phase === "queued"
      ? "bg-vw-amber-fg"
      : "vw-live-dot"
    : bad
      ? "bg-red-400"
      : "bg-chat-dim";
  const status = !ended ? (r.phase === "queued" ? "queued" : "running") : bad ? "error" : "finished";
  const shares = miniShares(r, scope);
  return (
    <button
      type="button"
      data-testid={`prompt-lens-row-${r.reqId}`}
      data-req={r.reqId}
      aria-current={selected ? "true" : undefined}
      onClick={() => onSelect(r.reqId)}
      style={{ height: LIST_ROW_PX }}
      className={cn(
        "grid w-full flex-none cursor-pointer grid-cols-[10px_minmax(0,1fr)_auto] grid-rows-2 items-center gap-x-2 overflow-hidden border-b border-chat-rule/60 px-3 text-left text-[12px] hover:bg-chat-surface-2/40",
        selected && "bg-chat-surface-2/70",
      )}
    >
      <span
        role="img"
        aria-label={status}
        className={cn("row-span-2 inline-block h-2 w-2 rounded-full", dot)}
      />
      <span className="flex min-w-0 items-center gap-1.5 overflow-hidden whitespace-nowrap text-chat-fg">
        <span>{r.sessionId === null ? "no session" : r.turn !== null ? `turn ${r.turn}` : "turn ?"}</span>
        <span className="text-chat-dim">· {hms(r.startedTs)}</span>
        {!r.stream && r.hasStart && <span className={cn(CHIP, "border-chat-rule text-chat-muted")}>non-streamed</span>}
        {r.kept && <span className={cn(CHIP, "border-vw-amber-fg/40 text-vw-amber-fg")}>kept</span>}
      </span>
      <span className="justify-self-end whitespace-nowrap">
        {ended ? (
          <span className={cn(CHIP, bad ? "border-vw-amber-fg/50 text-vw-amber-fg" : "border-chat-rule text-chat-muted")}>
            {r.finish ?? "ended"}
          </span>
        ) : (
          <span className={cn(CHIP, "border-vw-ok-fg/50 text-vw-ok-fg")}>{PHASE_WORD[r.phase] ?? r.phase.replace(/_/g, " ")}</span>
        )}
      </span>
      <span className="flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap text-chat-muted">
        <span
          data-testid="prompt-lens-mini"
          role="img"
          aria-label={miniLabel(shares, scope)}
          title={scope === "admin" ? "kvsig: exact / loose / another key's shares of the prompt" : "kvsig: exact / loose shares of the prompt"}
          className="inline-flex h-1.5 flex-none overflow-hidden rounded-sm bg-chat-rule/60"
          style={{ width: MINI_BAR_PX }}
        >
          {shares.exact > 0 && <span data-share="exact" className={STATE_CLASS.exact} style={{ width: `${shares.exact * 100}%` }} />}
          {shares.loose > 0 && (
            <span data-share="loose" style={{ ...stateStyle("loose"), width: `${shares.loose * 100}%` }} />
          )}
          {scope === "admin" && shares.other > 0 && (
            <span data-share="other" className={STATE_CLASS.other} style={{ width: `${shares.other * 100}%` }} />
          )}
        </span>
        {/* Live counters in reserved slots (ui.live-numbers): "out" and the
            rate never push what follows them. */}
        <span>
          <LiveNumber resetKey={r.reqId}>{tokensIn(r)}</LiveNumber>{" "}
          in ·{" "}
          <LiveNumber reserve={ended ? 0 : ROW_TOK_CH} resetKey={r.reqId}>
            {tokensOut(r)}
          </LiveNumber>{" "}
          out
        </span>
      </span>
      <span className="justify-self-end whitespace-nowrap text-chat-muted">
        <LiveNumber reserve={ended ? 0 : ROW_RATE_CH} resetKey={r.reqId}>
          {rateOrDuration(r)}
        </LiveNumber>
      </span>
    </button>
  );
});

export interface RequestListProps {
  scope: LensScope;
  rows: RequestView[];
  /** Paused: the frozen order; rows not in it wait behind the pill. */
  frozen?: string[] | null;
  newCount?: number;
  onShowNew?: () => void;
  selected: string | null;
  onSelect: (reqId: string) => void;
  keyNames?: Record<string, string>;
}

export function RequestList({ scope, rows, frozen = null, newCount = 0, onShowNew, selected, onSelect, keyNames = {} }: RequestListProps) {
  let shown = rows;
  if (frozen) {
    const byId = new Map(rows.map((r) => [r.reqId, r]));
    shown = frozen.map((id) => byId.get(id)).filter((r): r is RequestView => r !== undefined);
  }
  const groups = groupRows(shown, scope, frozen !== null);
  return (
    <div className="flex min-h-0 flex-col">
      {newCount > 0 && (
        <div className="sticky top-0 z-[2] flex justify-center py-1">
          <button
            type="button"
            onClick={onShowNew}
            className="rounded-full border border-chat-accent/60 bg-chat-surface px-2.5 py-0.5 text-[11px] text-chat-accent-strong"
          >
            {`${newCount} new`}
          </button>
        </div>
      )}
      {groups.map((g) => {
        const newestKv = g.rows.find((r) => r.kvsig?.conv8)?.kvsig?.conv8 ?? null;
        const ip = g.rows.find((r) => r.clientIp)?.clientIp ?? null;
        const label = g.rows.find((r) => r.tokenLabel)?.tokenLabel ?? null;
        const name = (g.tokenId && keyNames[g.tokenId]) || label || (g.tokenId ? "unnamed key" : "no key");
        const n = g.rows.length;
        return (
          <div key={g.key} data-testid="prompt-lens-group" data-session={g.sessionId ?? ""}>
            <div
              data-testid="prompt-lens-group-head"
              className="sticky top-0 z-[1] flex h-7 items-center gap-1.5 overflow-hidden whitespace-nowrap border-b border-chat-rule bg-vw-dock px-3 text-[11px] text-chat-dim"
            >
              {scope === "admin" && <b className="font-semibold text-chat-fg">{name}</b>}
              <span>
                {g.sessionId ? (
                  <>
                    session <span className="font-mono text-chat-fg">{g.sessionId.slice(0, 8)}</span>
                  </>
                ) : (
                  "no session"
                )}
              </span>
              <span>· {`${n} turn${n === 1 ? "" : "s"} here`}</span>
              {scope === "admin" && ip && <span className="font-mono">· {ip}</span>}
              <span className="ml-auto">
                conv <span className="font-mono">{newestKv ? newestKv.slice(0, 4) : "—"}</span>
              </span>
            </div>
            {g.rows.map((r) => (
              <Row key={r.reqId} r={r} scope={scope} selected={r.reqId === selected} onSelect={onSelect} />
            ))}
          </div>
        );
      })}
      <p className="px-3 py-3 text-[11px] leading-snug text-chat-dim">{LIST_FOOTER}</p>
    </div>
  );
}
