"use client";

// The Sessions view's list (analysis spec 16.1): the scored sessions, newest
// first, each titled by its key (admin) and harness, with the overall and the
// three layers in words; a tap opens the scorecard. A session is scored ten minutes after its last request, so the
// newest ones are not here yet. A layer without enough data reads "not scored",
// never a number (A-7).

import { useEffect } from "react";

import { fmtDuration } from "@/components/sessions/format";
import { LAYER_NAME, meaning, useSessionList, type SessionListItem } from "@/lib/prompt-lens/scorecard";
import type { LensLayer } from "@/lib/prompt-lens/findings";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { S } from "@/lib/sessions/strings";
import { cn } from "@/lib/utils";

const LAYERS: readonly LensLayer[] = ["E", "C", "R"];

const LAYER_GIST: Record<LensLayer, string> = {
  E: "how much of each prompt the cache could reuse",
  C: "wasted or repeated prompt content",
  R: "whether replies came back complete and usable",
};

function when(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** The scorer's reason code, in words ("2 of 3 continuing turns" -> "2 of the 3 continuing turns it needs"). */
function reasonText(layer: LensLayer, reason: string | null | undefined): string {
  if (!reason) return "not enough data";
  if (layer === "E") return reason.replace(/^(\d+) of (\d+) continuing turns$/, "$1 of the $2 continuing turns it needs");
  if (layer === "R") return reason.replace(/^(\d+) of (\d+) assistant turns$/, "$1 of the $2 measured replies it needs");
  return reason;
}

function Layer({ layer, v, scope }: { layer: LensLayer; v: number | null | undefined; scope: LensScope }) {
  const scored = typeof v === "number";
  return (
    <span
      data-testid={`row-${layer}`}
      title={`${LAYER_NAME[layer]}: ${LAYER_GIST[layer]}. ${meaning(layer, scope)}`}
      className={cn("whitespace-nowrap tabular-nums", scored ? "text-chat-fg" : "text-chat-dim")}
    >
      {LAYER_NAME[layer]} {scored ? v : "—"}
    </span>
  );
}

function Row({ it, scope, selected, onOpen }: { it: SessionListItem; scope: LensScope; selected: boolean; onOpen: (id: string) => void }) {
  const overall = typeof it.overall === "number";
  const harness = S.harness[it.harness] ?? S.harness.other;
  const title = S.header.title(scope === "admin" ? (it.key_name ?? "unnamed key") : null, harness);
  const missing = LAYERS.filter((l) => typeof it[l.toLowerCase() as "e" | "c" | "r"] !== "number");
  const sub = [`${it.turns} turns`, fmtDuration(it.last_at - it.started)].join(" · ");
  return (
    <button
      type="button"
      data-testid="lens-session-row"
      data-sid={it.session_id}
      aria-current={selected ? "true" : undefined}
      onClick={() => onOpen(it.session_id)}
      className={cn(
        "flex w-full min-w-0 cursor-pointer flex-col gap-1 border-b border-chat-rule/60 bg-transparent px-3 py-2 text-left hover:bg-chat-surface",
        selected && "bg-chat-surface",
      )}
    >
      <span className="flex items-baseline gap-2">
        <b data-testid="row-title" className="min-w-0 flex-1 break-words text-[13px] font-semibold text-chat-fg">
          {title}
        </b>
        <span className="flex-none text-[11px] text-chat-dim">{when(it.last_at)}</span>
      </span>
      <span data-testid="row-sub" className="text-[11px] text-chat-dim">
        {sub}
      </span>
      <span className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[12px]">
        {overall && (
          <b data-testid="row-overall" className="whitespace-nowrap tabular-nums text-chat-fg">
            Overall {it.overall}
          </b>
        )}
        {LAYERS.map((l) => (
          <Layer key={l} layer={l} v={it[l.toLowerCase() as "e" | "c" | "r"]} scope={scope} />
        ))}
        {it.confidence && it.confidence !== "high" && <span className="text-[11px] text-chat-dim">{it.confidence} confidence</span>}
      </span>
      {missing.length > 0 && (
        <span data-testid="row-not-scored" className="text-[11px] leading-snug text-chat-dim">
          Not scored: {missing.map((l) => `${LAYER_NAME[l]} (${reasonText(l, it.reasons[l])})`).join(", ")}
        </span>
      )}
    </button>
  );
}

export interface SessionListProps {
  scope: LensScope;
  /** Unused now: the list API names each key itself (`key_name`). */
  keyNames?: Record<string, string>;
  /** Admin: only these keys' sessions (the page's Key filter); none: every key. */
  keys?: string[] | null;
  models?: string[] | null;
  harnesses?: string[] | null;
  /** Reports the number of sessions shown (null while loading or failed) and whether the server has more. */
  onCount?: (shown: number | null, more: boolean) => void;
  selected: string | null;
  onOpen: (sessionId: string) => void;
}

export function SessionList({ scope, keys = null, models = null, harnesses = null, onCount, selected, onOpen }: SessionListProps) {
  const { data, loading, error } = useSessionList(scope, keys, { models, harnesses });
  const shown = data && Array.isArray(data.sessions) ? data.sessions.length : null;
  const more = !!data?.next_cursor;
  useEffect(() => {
    onCount?.(shown, more);
  }, [onCount, shown, more]);
  if (loading) return <p className="p-3 text-[13px] text-chat-dim">Loading sessions…</p>;
  if (error || !data || !Array.isArray(data.sessions)) return <p data-testid="lens-sessions-error" className="p-3 text-[13px] text-vw-amber-fg">The scored sessions could not be loaded.</p>;
  if (data.sessions.length === 0) {
    return (
      <p data-testid="lens-sessions-empty" className="p-3 text-[13px] leading-snug text-chat-dim">
        No scored sessions yet. A session is scored ten minutes after its last request; the newest are not here yet.
      </p>
    );
  }
  return (
    <div data-testid="lens-session-list">
      {data.sessions.map((it) => (
        <Row key={it.session_id} it={it} scope={scope} selected={it.session_id === selected} onOpen={onOpen} />
      ))}
    </div>
  );
}
