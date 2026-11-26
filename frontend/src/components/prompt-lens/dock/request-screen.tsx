"use client";

// The Prompt Lens request screen (spec 2026-11-22 section 3, mockup states
// 3-7): one request of the dock, composed from the prompt map (Task 10), the
// section boxes (Task 11) and the pop-up (Task 12).
//
//   header   ‹ Requests · title · phase or finish chip · model
//   facts    queued · first token · elapsed or duration · prompt · out · rate
//            · client IP (admin only); one line that scrolls sideways
//   lines    key holder: the fixed privacy line; non-streamed: the banner
//   findings once the request ended: "Findings on this request" (analysis
//            spec 16.1, Task 1.8); each finding also sits beside its part
//            (parts table) or under the tool calls it judged
//   map      kvsig prompt map, parts table (in a section box over 8 rows), footer
//   columns  Input (tool results, prompt by role) · Output (thinking, reply,
//            tool calls, usage); stacked on a phone
//
// Scope (G-R11). Everything a key holder could learn about another key is
// already gone from the events (prompt_lens_scope.for_key_holder) and from the map
// (`mapParts`); this screen adds no figure of its own that could imply one: a
// key holder's usage shows `own_reused_tokens` as "reused from your earlier
// requests", never `cached_tokens`, even if one arrived.
//
// "Finished" is `endedTs !== null`, never the phase: a kept capture row reads
// `queued` until it is filled. Total wait to the first token = ttft_s +
// queued_s (ttft counts from admission). Rates while streaming are SSE frames
// per second over the section's span, marked "≈" (Q6); the exact usage
// replaces them at the end.
//
// Section text is always text: Thinking and Reply stream through TailText
// (append-only DOM text nodes), keyed by the request so a new request never
// continues the old one's blocks. Reply is not rendered as markdown: there is
// no markdown renderer in the frontend (chat-ui 0.1.28 exports none).
//
// The pop-up (⤢) fetches the whole section itself and does not write it back
// to the store (see the Task 13 report: the boxes keep the dock's tail).

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { PromptMsg } from "@/lib/prompt-lens/events";
import { useRequestFindings } from "@/lib/prompt-lens/findings";
import { mapParts, type MapPart } from "@/lib/prompt-lens/kvsig";
import { prettyJson } from "@/lib/prompt-lens/pretty-json";
import { sectionMarks } from "@/lib/prompt-lens/section-marks";
import { noteServerTs, serverNow } from "@/lib/prompt-lens/server-clock";
import { trimmedOf, type RequestView, type Section } from "@/lib/prompt-lens/store";
import type { FetchKind, LensScope } from "@/lib/prompt-lens/stream";
import { K } from "@/lib/key-area/strings";
import { LiveNumber } from "@/components/ui/live-number";
import { cn } from "@/lib/utils";
import { MediaStrip } from "../prompt-lens-media";
import { FindingChip } from "../findings/finding-chip";
import { FindingsStrip } from "../findings/findings-strip";
import { KvsigFooter, KvsigMap } from "./kvsig-map";
import { PARTS_BOX_ROWS, PartsTable } from "./parts-table";
import { LIVE_RATE_CH, SectionBox, type SectionKind, type useCollapseMemory } from "./section-box";
import { SectionPopup } from "./section-popup";
import { TailText } from "./tail-text";
import { ToolCallBlocks, type CallBlock } from "./tool-call-blocks";

export interface RequestScreenProps {
  request: RequestView;
  scope: LensScope;
  phone: boolean;
  collapse: ReturnType<typeof useCollapseMemory>;
  onBack?: () => void;
  /** Key scope: an admin set this key to keep captures (GET /api/me/key
   *  `keep_captures`), so the privacy line says they are on disk (G-R14). */
  keepsCaptures?: boolean;
}

/** Lines of a tool result shown in the box (the pop-up shows it whole). */
export const RESULT_LINES = 8;

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

/** A duration: 0.5 s, 1.75 s, 14.6 s, 3 min 08 s. */
function secs(x: number): string {
  if (!Number.isFinite(x) || x < 0) return "—";
  if (x < 10) return `${Number(x.toFixed(2))} s`;
  if (x < 60) return `${Number(x.toFixed(1))} s`;
  return clock(x);
}

/** A ticking duration in whole seconds: 10 s, 3 min 08 s. */
function clock(x: number): string {
  const s = Math.max(0, Math.floor(x));
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

/** HH:MM:SS of a wall clock time in seconds (local time). */
function hms(ts: number): string {
  const d = new Date(ts * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** Frames per second over a section's span (≈ tok/s), or null. */
function spanRate(frames: number, s: Section): number | null {
  if (s.firstTs === null || s.lastTs === null) return null;
  const span = s.lastTs - s.firstTs;
  return span > 0 && frames > 1 ? frames / span : null;
}

const PHASE_WORD: Record<string, string> = {
  queued: "queued",
  prefill: "prefill",
  thinking: "thinking",
  answering: "answering",
  tool_call: "tool call",
  running: "running",
};

const BAD_FINISH = new Set(["length", "abort", "error", "runaway"]);

/** The server's "now" (seconds since the epoch), ticking once a second
 *  while `on`. */
function useNow(on: boolean): number {
  const [now, setNow] = useState(serverNow);
  useEffect(() => {
    if (!on) return;
    setNow(serverNow());
    const t = setInterval(() => setNow(serverNow()), 1000);
    return () => clearInterval(t);
  }, [on]);
  return now;
}

/** The newest server `ts` the request holds. */
function newestTs(r: RequestView): number {
  let t = Math.max(r.startedTs, r.lastTs);
  for (const x of [r.thinking.lastTs, r.reply.lastTs, r.toolCallFirstTs, r.kvsig?.ts ?? null, r.end?.ts ?? null]) {
    if (x !== null && x > t) t = x;
  }
  return t;
}

/** Feed the server clock (lib/prompt-lens/server-clock.ts) when the request gets
 *  a newer event: it has just arrived, so receive time minus its ts is the
 *  skew plus the delivery delay. Not on mount: an old request's newest event
 *  may be long past. */
function useNoteServerTs(r: RequestView) {
  const seen = useRef<{ id: string; ts: number } | null>(null);
  const ts = newestTs(r);
  useLayoutEffect(() => {
    const prev = seen.current;
    seen.current = { id: r.reqId, ts };
    if (prev && prev.id === r.reqId && ts > prev.ts) noteServerTs(ts);
  }, [r.reqId, ts]);
}

const PILL = "flex-none rounded border px-[5px] text-[10px] leading-4";
const NO_FINDINGS: never[] = [];
/** Live fact slots, in characters (ui.live-numbers): a ticking clock up to
 *  "59 min 59 s", a duration up to "99.9 s", tokens out up to "≈ 99,999". */
const LIVE_CLOCK_CH = 11;
const LIVE_SECS_CH = 6;
const LIVE_OUT_CH = 8;
const CARET =
  "[&>div:last-child]:after:ml-px [&>div:last-child]:after:inline-block [&>div:last-child]:after:h-[1em] [&>div:last-child]:after:w-[2px] [&>div:last-child]:after:animate-pulse [&>div:last-child]:after:bg-chat-accent [&>div:last-child]:after:align-text-bottom [&>div:last-child]:after:content-['']";

// ---------------------------------------------------------------------------

export function RequestScreen({ request: r, scope, phone, collapse, onBack, keepsCaptures = false }: RequestScreenProps) {
  const [isCollapsed, setCollapsed] = collapse;
  const isKey = scope === "key";
  const ended = r.endedTs !== null;
  const end = r.end;
  useNoteServerTs(r);
  const now = useNow(!ended);
  const view = useMemo(() => mapParts(r, scope), [r, scope]);
  // Findings come from the stored row: asked for once the request ended.
  const found = useRequestFindings(scope, ended ? r.reqId : null);
  const findings = found?.findings ?? NO_FINDINGS;
  const names = found?.names;
  const callFindings = useMemo(() => findings.filter((f) => f.part?.startsWith("call:")), [findings]);

  // --- pop-up --------------------------------------------------------------
  const [popup, setPopup] = useState<FetchKind | null>(null);
  useEffect(() => setPopup(null), [r.reqId]);

  // --- a clicked map segment ------------------------------------------------
  const [active, setActive] = useState<{ part: number; msg: string; n: number } | null>(null);
  useEffect(() => setActive(null), [r.reqId]);
  const promptRef = useRef<HTMLDivElement>(null);
  const onSegment = useCallback(
    (p: MapPart) => {
      const part = view.parts.indexOf(p);
      const msg = p.p === "nodes" ? String(p.from) : p.p;
      setActive((a) => ({ part, msg, n: (a?.n ?? 0) + 1 }));
      setCollapsed("prompt", false);
    },
    [view.parts, setCollapsed],
  );
  const promptOpen = !isCollapsed("prompt");
  const activeKey = active && promptOpen ? rowKey(r, active.msg) : null;
  useLayoutEffect(() => {
    if (!active || !promptOpen || activeKey === null) return;
    const row = promptRef.current?.querySelector<HTMLElement>(`[data-msg="${activeKey}"]`);
    if (!row) return;
    const box = row.closest<HTMLElement>("[data-scroll-box]");
    if (box) box.scrollTop += row.getBoundingClientRect().top - box.getBoundingClientRect().top - 8;
    box?.closest("section")?.scrollIntoView?.({ block: "nearest" });
  }, [active, promptOpen, activeKey]);

  // --- header and facts --------------------------------------------------
  const sid = r.sessionId ? `session ${r.sessionId.slice(0, 8)}` : "no session";
  const title = [
    ...(isKey ? [] : [r.tokenLabel || (r.tokenId ? "unnamed key" : "no key")]),
    sid,
    ...(r.turn !== null ? [`turn ${r.turn}`] : []),
  ].join(" · ");
  const chip = ended ? r.finish ?? "ended" : PHASE_WORD[r.phase] ?? r.phase.replace(/_/g, " ");
  const chipClass = ended
    ? r.finish && BAD_FINISH.has(r.finish)
      ? "border-vw-amber-fg/50 text-vw-amber-fg"
      : "border-chat-rule text-chat-muted"
    : "border-vw-ok-fg/50 text-vw-ok-fg";
  const trim = trimmedOf(r);
  const trimmed = trim.chars + trim.toolCallChars + trim.toolCalls > 0 || r.stripped;

  const firstOut = minTs(minTs(r.thinking.firstTs, r.reply.firstTs), r.toolCallFirstTs);
  const admittedWait = typeof r.kvsig?.queued_s === "number" ? r.kvsig.queued_s : null;
  // Values that change while the request runs sit in reserved slots
  // (ui.live-numbers): the facts after them never move.
  const live = (text: string, reserve: number) => (
    <LiveNumber reserve={ended ? 0 : reserve} resetKey={r.reqId} align="left">
      {text}
    </LiveNumber>
  );
  const facts: [string, ReactNode][] = [];
  facts.push([
    "queued",
    live(
      end && end.queued_s !== null
        ? secs(end.queued_s)
        : admittedWait !== null
          ? secs(admittedWait)
          : !ended && r.phase === "queued"
            ? clock(now - r.startedTs)
            : "—",
      LIVE_CLOCK_CH,
    ),
  ]);
  facts.push([
    "first token",
    live(
      !r.stream
        ? "n/a"
        : end && end.ttft_s !== null
          ? secs(end.ttft_s + (end.queued_s ?? 0))
          : firstOut !== null
            ? secs(firstOut - r.startedTs)
            : "—",
      LIVE_SECS_CH,
    ),
  ]);
  if (ended) {
    const d = end?.duration_s ?? (r.endedTs as number) - r.startedTs;
    facts.push(["duration", live(secs(d), LIVE_CLOCK_CH)]);
  } else {
    facts.push(["elapsed", live(clock(now - r.startedTs), LIVE_CLOCK_CH)]);
  }
  facts.push([
    "prompt",
    end ? fmt(end.prompt_tokens) : r.prompt?.total_chars ? `≈ ${fmt(r.prompt.total_chars / 4)}` : "—",
  ]);
  const framesOut = r.frames.reasoning + r.frames.content + r.frames.tool_call;
  facts.push(["out", live(end ? fmt(end.completion_tokens) : `≈ ${fmt(r.progressTokens ?? framesOut)}`, LIVE_OUT_CH)]);
  // The span is thinking's and the reply's (tool-call deltas carry no span),
  // so only their frames count.
  const framesText = r.frames.reasoning + r.frames.content;
  if (ended) {
    if (end?.decode_tps != null)
      facts.push([r.stream ? "decode" : "avg", <>{live(fmt(end.decode_tps), LIVE_RATE_CH)} tok/s</>]);
  } else if (r.stream) {
    const first = minTs(r.thinking.firstTs, r.reply.firstTs);
    const last = maxTs(r.thinking.lastTs, r.reply.lastTs);
    const span = first !== null && last !== null ? last - first : 0;
    if (span > 0 && framesText > 1) facts.push(["now", <>≈ {live(fmt(framesText / span), LIVE_RATE_CH)} tok/s</>]);
  }
  if (!isKey) facts.push(["client", r.clientIp || "—"]);

  // --- sections -------------------------------------------------------------
  const kinds = { thinking: r.thinking, reply: r.reply };
  const replyStarted = r.reply.chars > 0 || r.toolCalls.length > 0;
  const thinkingLive = r.stream && !ended && !replyStarted && r.phase !== "queued";
  const replyLive = r.stream && !ended && r.reply.chars > 0 && r.toolCalls.length === 0;
  const callsLive = r.stream && !ended && r.toolCalls.length > 0;

  let thinkingStatus: string | null = null;
  if (r.thinking.chars > 0 && r.thinking.firstTs !== null && (replyStarted || ended)) {
    const stop = minTs(r.reply.firstTs, r.toolCallFirstTs) ?? r.thinking.lastTs ?? r.thinking.firstTs;
    const took = secs(Math.max(0, stop - r.thinking.firstTs));
    if (ended) {
      const tok = typeof end?.reasoning_tokens === "number" ? end.reasoning_tokens : r.thinking.chars / 4;
      thinkingStatus = `≈ ${fmt(tok)} tok · ${took}`;
    } else {
      thinkingStatus = `done in ${took}`;
    }
  }
  const rateOf = (k: "thinking" | "reply") =>
    r.stream && kinds[k].chars > 0 ? spanRate(r.frames[k === "thinking" ? "reasoning" : "content"], kinds[k]) : null;

  const replyEmpty = ended
    ? !r.stream && r.finish === "runaway"
      ? "No reply text: the request was stopped as a runaway before its reply arrived (a non-streamed reply comes only at the end)."
      : r.toolCalls.length > 0
        ? "No reply text: this turn answers with tool calls."
        : "No reply text."
    : !r.stream
      ? "A non-streamed reply arrives in one piece when the request ends."
      : r.thinking.chars > 0
        ? "The reply starts when thinking ends."
        : "No reply so far.";

  const calls = useMemo(
    (): CallBlock[] =>
      r.toolCalls.map((c, i) => ({
        key: `${c.index}`,
        index: c.index,
        callId: c.callId,
        name: c.name,
        args: c.args,
        // Streaming arguments are partial JSON: parse only finished calls.
        pretty: !callsLive || i < r.toolCalls.length - 1 ? prettyJson(c.args) : null,
        elided: c.elided,
      })),
    [r.toolCalls, callsLive],
  );
  const callChars = r.toolCalls.reduce((n, c) => n + c.args.length, 0);

  const results = useMemo(() => toolResults(r, view.parts), [r, view.parts]);
  const resultChars = results.blocks.reduce((n, b) => n + b.chars, 0);

  const marks = (k: SectionKind) => sectionMarks(r, k);
  const box = (k: SectionKind, tok = 0) => ({
    collapsed: isCollapsed(k, tok),
    onToggle: () => setCollapsed(k, !isCollapsed(k, tok)),
    phone,
    resetKey: r.reqId,
    marks: marks(k),
  });

  const textSection = (k: "thinking" | "reply", live: boolean) => {
    const s = kinds[k];
    if (s.chars === 0 && s.text === "") return undefined;
    return (
      <div data-testid={live ? `caret-${k}` : undefined}>
        <TailText key={r.reqId} text={s.text} offset={s.clipped} className={cn(live && CARET)} />
      </div>
    );
  };

  const thinkingTok = end && typeof end.reasoning_tokens === "number" ? end.reasoning_tokens : undefined;
  const replyTok =
    end && typeof end.reasoning_tokens === "number" && r.toolCalls.length === 0
      ? Math.max(0, end.completion_tokens - end.reasoning_tokens)
      : undefined;

  const output = (
    <>
      <SectionBox
        kind="thinking"
        title="Thinking"
        chars={r.thinking.chars}
        tok={thinkingTok}
        live={thinkingLive}
        rate={rateOf("thinking")}
        status={thinkingStatus}
        onExpand={() => setPopup("thinking")}
        empty={ended ? "No thinking." : "No thinking so far."}
        {...box("thinking")}
      >
        {textSection("thinking", thinkingLive)}
      </SectionBox>
      <SectionBox
        kind="reply"
        title="Reply"
        chars={r.reply.chars}
        tok={replyTok}
        live={replyLive}
        rate={rateOf("reply")}
        status={replyLive ? "so far" : null}
        onExpand={() => setPopup("reply")}
        empty={replyEmpty}
        {...box("reply")}
      >
        {textSection("reply", replyLive)}
      </SectionBox>
      <SectionBox
        kind="tool_calls"
        title="Tool calls"
        chars={callChars}
        live={callsLive}
        status={r.toolCalls.length ? `${r.toolCalls.length} ${r.toolCalls.length === 1 ? "call" : "calls"}` : null}
        onExpand={() => setPopup("tool_calls")}
        empty={ended ? "No tool calls." : "No tool call so far."}
        contentKey={`${r.toolCalls.length}:${callChars}:${ended}`}
        {...box("tool_calls")}
      >
        {r.toolCalls.length > 0 ? <ToolCallBlocks calls={calls} live={callsLive} /> : undefined}
      </SectionBox>
      {callFindings.length > 0 && (
        <div data-testid="call-findings" className="-mt-1 mb-2.5 min-w-0">
          {callFindings.map((f, i) => {
            const k = Number(f.part?.slice(5));
            const call = r.toolCalls.find((c) => c.index === k);
            const lead = `tool call ${k + 1}${call?.name ? ` · ${call.name}` : ""}`;
            return <FindingChip key={`${f.rule}:${i}`} finding={f} names={names} scope={scope} lead={lead} />;
          })}
        </div>
      )}
      <Usage r={r} isKey={isKey} phone={phone} />
    </>
  );

  const promptTok = end ? end.prompt_tokens : undefined;
  const promptChars = r.prompt ? r.prompt.total_chars || promptSize(r) : r.promptFlat.length;
  const mediaCount = r.media.reduce((n, m) => n + (m.dropped === "count" && m.count ? m.count : 1), 0);

  const input = (
    <>
      <SectionBox
        kind="tool_results"
        title="Tool results in the prompt"
        chars={resultChars}
        live={false}
        status={results.blocks.length ? `${results.blocks.length} ${results.blocks.length === 1 ? "result" : "results"}` : null}
        onExpand={() => setPopup("tool_results")}
        empty="This prompt carries no tool results."
        {...box("tool_results", Math.round(resultChars / 4))}
      >
        {results.blocks.length > 0 || results.older > 0 ? (
          <ResultCards blocks={results.blocks} older={results.older} />
        ) : undefined}
      </SectionBox>
      {mediaCount > 0 &&
        (isKey ? (
          <div className="mb-2.5">
            <span className={cn(PILL, "border-chat-rule text-chat-muted")}>image · {mediaCount}</span>
          </div>
        ) : (
          <div className="mb-2.5 min-w-0 overflow-hidden rounded-lg border border-chat-rule">
            <MediaStrip media={r.media} />
          </div>
        ))}
      <SectionBox
        kind="prompt"
        title="Prompt by role"
        chars={promptChars}
        tok={promptTok}
        live={false}
        status={r.prompt ? `${r.prompt.messages.length + r.prompt.collapsed} messages` : null}
        onExpand={() => setPopup("prompt")}
        empty="No prompt captured."
        contentKey={`${promptChars}:${activeKey ?? ""}`}
        {...box("prompt")}
      >
        {r.prompt ? (
          <div ref={promptRef}>
            <PromptRows r={r} active={activeKey} />
          </div>
        ) : r.promptFlat ? (
          <div ref={promptRef}>{r.promptFlat}</div>
        ) : undefined}
      </SectionBox>
    </>
  );

  // --- prompt map ---------------------------------------------------------
  const partsChars = view.parts.reduce((n, p) => n + p.chars, 0);
  const partsInBox = !phone && view.parts.length > PARTS_BOX_ROWS;
  const activePart = active?.part ?? null;

  return (
    <div data-testid="request-screen" className="min-w-0 text-chat-fg">
      <div className={cn("flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap", phone ? "h-12" : "h-8")}>
        {onBack && (
          <button
            type="button"
            onClick={onBack}
            className={cn(
              "flex-none cursor-pointer rounded-md border border-chat-rule bg-transparent px-2 py-0.5 text-[12px] text-chat-fg hover:bg-chat-surface",
              phone && "inline-flex min-h-11 min-w-11 items-center px-3",
            )}
          >
            ‹ Requests
          </button>
        )}
        <span data-testid="rs-title" title={title} className="min-w-0 overflow-hidden text-ellipsis text-[14px] font-semibold">
          {title}
        </span>
        <span data-testid="rs-chip" aria-live="polite" className={cn(PILL, "text-[11px]", chipClass)}>
          {chip}
        </span>
        {!r.stream && <span className={cn(PILL, "border-chat-rule text-chat-muted")}>non-streamed</span>}
        {r.kept && (
          <span className={cn(PILL, "border-vw-info-fg/50 text-vw-info-fg")} title="A kept capture: one of the last 50 kept on disk for this key">
            kept
          </span>
        )}
        {r.gap && (
          <span className={cn(PILL, "border-vw-amber-fg/50 text-vw-amber-fg")} title="The stream fell behind; some events of this request may be missing">
            events missing
          </span>
        )}
        {trimmed && (
          <span
            data-testid="rs-trimmed"
            className={cn(PILL, "border-vw-amber-fg/40 text-vw-amber-fg")}
            title={trimTitle(trim, r.stripped)}
          >
            trimmed
          </span>
        )}
        <span className="ml-auto min-w-0 flex-shrink overflow-hidden text-ellipsis text-[11px] text-chat-dim">
          {r.servedName || r.model}
        </span>
      </div>

      <div
        data-testid="rs-facts"
        tabIndex={0}
        role="group"
        aria-label="Request facts"
        className="flex h-6 min-w-0 items-center gap-x-4 overflow-x-auto whitespace-nowrap text-[12px] text-chat-muted [scrollbar-width:none] focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent"
      >
        {facts.map(([k, v]) => (
          <span key={k} className="inline-flex flex-none items-baseline gap-1">
            {k}
            <b className={cn("font-medium tabular-nums text-chat-fg", k === "client" && "font-mono")}>{v}</b>
          </span>
        ))}
      </div>

      {isKey && (
        <p className="my-1.5 rounded-md border border-chat-rule bg-chat-surface px-2.5 py-1.5 text-[12px] text-chat-muted">
          {keepsCaptures || r.kept ? K.stats.captureKept : K.stats.capture}
        </p>
      )}
      {!r.stream && (
        <p
          data-testid="rs-banner"
          className="my-1.5 rounded-md border border-vw-info-fg/40 bg-vw-info-fg/10 px-2.5 py-1.5 text-[12px] text-chat-fg"
        >
          Non-streamed request (<span className="font-mono">stream: false</span>).{" "}
          {ended ? (
            <>
              The reply arrived in one piece at <b>{hms(r.endedTs as number)}</b>, so the sections below filled at once
              and show no live rate. Decode speed is the average over the whole request.
            </>
          ) : (
            <>The reply arrives in one piece when the request ends; the sections fill at once and show no live rate.</>
          )}
        </p>
      )}

      <FindingsStrip findings={findings} names={names} scope={scope} />

      <div className="mb-2.5 mt-2 min-w-0 rounded-lg border border-chat-rule bg-chat-surface px-3 py-2.5">
        <KvsigMap view={view} onSegment={onSegment} active={activePart} phone={phone} />
        {partsInBox ? (
          <SectionBox
            kind="parts"
            title="Parts"
            chars={partsChars}
            tok={view.totalTok}
            live={false}
            onExpand={() => setPopup("prompt")}
            {...box("parts")}
          >
            <PartsTable parts={view.parts} phone={false} active={activePart} bare findings={findings} names={names} scope={scope} />
          </SectionBox>
        ) : (
          <PartsTable parts={view.parts} phone={phone} active={activePart} findings={findings} names={names} scope={scope} />
        )}
        <KvsigFooter view={view} />
      </div>

      <div data-testid="rs-cols" className={cn("min-w-0", phone ? "flex flex-col" : "grid grid-cols-2 gap-3")}>
        <div className="min-w-0" data-testid="rs-input">
          {input}
        </div>
        <div className="min-w-0" data-testid="rs-output">
          {output}
        </div>
      </div>

      {popup && <SectionPopup scope={scope} kind={popup} request={r} phone={phone} onClose={() => setPopup(null)} />}
    </div>
  );
}

// ---------------------------------------------------------------------------

function minTs(a: number | null, b: number | null): number | null {
  if (a === null) return b;
  if (b === null) return a;
  return Math.min(a, b);
}

function maxTs(a: number | null, b: number | null): number | null {
  if (a === null) return b;
  if (b === null) return a;
  return Math.max(a, b);
}

function trimTitle(t: ReturnType<typeof trimmedOf>, stripped: boolean): string {
  const out: string[] = [];
  if (t.chars) out.push(`${fmt(t.chars)} text chars not kept`);
  if (t.toolCallChars) out.push(`${fmt(t.toolCallChars)} tool-call argument chars cut`);
  if (t.toolCalls) out.push(`${fmt(t.toolCalls)} tool calls dropped on the server`);
  if (stripped) out.push("text dropped to save memory");
  return `${out.join("; ")}. ⤢ on a section shows it whole from server memory.`;
}

function promptSize(r: RequestView): number {
  const p = r.prompt;
  if (!p) return 0;
  return (p.sys?.chars ?? 0) + p.tools.chars + p.messages.reduce((n, m) => n + m.chars, 0) + p.older.chars.reduce((n, c) => n + c, 0);
}

/** The prompt row a map part's first message lands on: its own row, the
 *  "counted only" row for a message before the listed ones, else the next
 *  listed one. */
function rowKey(r: RequestView, msg: string): string | null {
  const p = r.prompt;
  if (!p) return null;
  if (msg === "sys") return p.sys ? "sys" : null;
  if (msg === "tools") return p.tools.count > 0 ? "tools" : null;
  const i = Number(msg);
  if (!Number.isFinite(i)) return null;
  const o = p.older;
  if (i >= o.from && i < o.from + o.roles.length) return String(i);
  if (p.messages.some((m) => m.i === i)) return String(i);
  if (i < o.from && p.omitted > 0) return "omitted";
  const next = [...Array.from(o.roles, (_, k) => o.from + k), ...p.messages.map((m) => m.i)].filter((n) => n >= i).sort((a, b) => a - b)[0];
  return next !== undefined ? String(next) : null;
}

// ---------------------------------------------------------------------------
// Usage

function Usage({ r, isKey, phone }: { r: RequestView; isKey: boolean; phone: boolean }) {
  const e = r.end;
  const items: [string, string][] = [];
  if (e) {
    items.push(["prompt", fmt(e.prompt_tokens)]);
    if (isKey) {
      items.push(["reused from your earlier requests", typeof e.own_reused_tokens === "number" ? fmt(e.own_reused_tokens) : "—"]);
    } else {
      items.push(["engine cached", typeof e.cached_tokens === "number" ? fmt(e.cached_tokens) : "—"]);
    }
    items.push(["completion", fmt(e.completion_tokens)]);
    items.push(["of which thinking", typeof e.reasoning_tokens === "number" ? fmt(e.reasoning_tokens) : "—"]);
    items.push(["first token", !r.stream ? "n/a" : e.ttft_s !== null ? secs(e.ttft_s + (e.queued_s ?? 0)) : "—"]);
    items.push(["duration", e.duration_s !== null ? secs(e.duration_s) : "—"]);
    items.push(["decode", e.decode_tps !== null ? `${fmt(e.decode_tps)} tok/s${r.stream ? "" : " average"}` : "—"]);
    items.push(["finish", e.finish_reason ?? "—"]);
    if ((e.tool_calls_dropped ?? 0) > 0) items.push(["tool calls dropped", fmt(e.tool_calls_dropped as number)]);
    if ((e.tool_call_chars_dropped ?? 0) > 0) items.push(["tool-call chars dropped", fmt(e.tool_call_chars_dropped as number)]);
  }
  return (
    <section
      data-testid="usage"
      aria-label="Usage"
      className="mb-2.5 min-w-0 rounded-lg border border-l-[3px] border-chat-rule border-l-chat-muted bg-chat-surface px-3 py-2.5"
    >
      <div className="mb-2 flex min-w-0 items-baseline gap-2 overflow-hidden whitespace-nowrap">
        <h4 className="m-0 flex-none text-[11px] font-bold uppercase tracking-[.08em] text-chat-muted">Usage</h4>
        <span className="min-w-0 overflow-hidden text-ellipsis text-[11px] text-chat-dim">
          {e ? `finish reason ${e.finish_reason ?? "unknown"}` : "at the end of the stream"}
        </span>
      </div>
      {/* Fixed height: the figures replace the placeholder without moving anything below. */}
      <div
        data-testid="usage-box"
        tabIndex={0}
        role="group"
        aria-label="Usage figures"
        className={cn(
          "min-w-0 overflow-y-auto focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent",
          phone ? "h-[176px]" : "h-[112px]",
        )}
      >
        {e ? (
          <dl className={cn("m-0 grid gap-x-4 gap-y-1.5 text-[12px]", phone ? "grid-cols-2" : "grid-cols-3")}>
            {items.map(([k, v]) => (
              <div key={k} className="min-w-0">
                <dt className="truncate text-[10px] uppercase tracking-[.06em] text-chat-dim">{k}</dt>
                <dd className="m-0 truncate font-medium tabular-nums text-chat-fg">{v}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="m-0 text-[12px] italic text-chat-dim">Filled in when the stream ends (the engine&apos;s own usage frame).</p>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Tool results in the prompt

interface ResultCard {
  i: number;
  name: string;
  callId: string | null;
  chars: number;
  text: string;
  elided: boolean;
  state: string | null;
}

const isResult = (role: string) => role === "tool" || role === "function";

function stateWord(state: string): string {
  switch (state) {
    case "exact":
      return "cached";
    case "loose":
      return "loose match";
    case "other":
      return "cached by another key";
    default:
      return "new";
  }
}

/** The prompt's tool results, newest first, with the map state of their node
 *  (as this scope sees it), and how many older ones are collapsed. */
function toolResults(r: RequestView, parts: MapPart[]): { blocks: ResultCard[]; older: number } {
  const p = r.prompt;
  if (!p) return { blocks: [], older: 0 };
  const stateOf = (i: number) => {
    const part = parts.find((x) => x.p === "nodes" && x.from <= i && i <= x.to);
    return part ? stateWord(part.state) : null;
  };
  const blocks = p.messages
    .filter((m) => isResult(m.role))
    .sort((a, b) => b.i - a.i)
    .map((m) => ({
      i: m.i,
      name: m.tool_name || (m.role === "function" ? "function result" : "tool result"),
      callId: m.tool_call_id ?? null,
      chars: m.chars,
      text: m.text,
      elided: m.elided,
      state: stateOf(m.i),
    }));
  let older = 0;
  for (const c of p.older.roles) if (c === "t" || c === "f") older += 1;
  return { blocks, older };
}

function ResultCards({ blocks, older }: { blocks: ResultCard[]; older: number }) {
  return (
    <div className="not-italic">
      {blocks.map((b) => {
        const lines = b.text.split("\n");
        const more = lines.length - RESULT_LINES;
        return (
          <div
            key={b.i}
            data-testid="tool-result"
            data-msg={String(b.i)}
            className="mb-2 min-w-0 rounded-md border border-vw-ttft/40 bg-vw-ttft/10"
          >
            <div className="flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap px-2.5 py-1 text-[12px]">
              <span className="min-w-0 overflow-hidden text-ellipsis font-mono font-medium text-vw-ttft">{b.name}</span>
              {b.callId && <span className="min-w-0 overflow-hidden text-ellipsis font-mono text-chat-dim">{b.callId}</span>}
              {b.elided && <span className={cn(PILL, "border-vw-amber-fg/40 text-vw-amber-fg")}>trimmed</span>}
              <span className="ml-auto flex-none text-[11px] tabular-nums text-chat-dim">
                {fmt(b.chars)} chars · msg {b.i + 1}
                {b.state ? ` · ${b.state}` : ""}
              </span>
            </div>
            <pre className="m-0 whitespace-pre-wrap border-t border-vw-ttft/40 px-3 py-1.5 font-mono text-[12px] leading-[1.5] [overflow-wrap:anywhere]">
              {more > 0 ? lines.slice(0, RESULT_LINES).join("\n") : b.text || "(no text)"}
              {more > 0 && (
                <span className="block text-chat-dim">
                  … {fmt(more)} more {more === 1 ? "line" : "lines"}
                </span>
              )}
            </pre>
          </div>
        );
      })}
      {older > 0 && (
        <div className="text-[11px] text-chat-dim">
          {fmt(older)} older tool {older === 1 ? "result is" : "results are"} collapsed in the capture (sizes only).
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prompt by role

const OLDER_ROLE: Record<string, string> = { u: "user", a: "assistant", t: "tool", f: "function" };

function oneLine(s: string, max = 200): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max)} …` : t;
}

function msgExcerpt(m: PromptMsg): string {
  const parts: string[] = [];
  if (isResult(m.role)) parts.push(`${m.tool_name ?? m.role}${m.tool_call_id ? ` → ${m.tool_call_id}` : ""}`);
  if (m.text.trim()) parts.push(oneLine(m.text));
  else if (m.reasoning?.trim()) parts.push(`[reasoning] ${oneLine(m.reasoning)}`);
  if (m.tool_calls?.length) parts.push(`tool call ${m.tool_calls.map((c) => c.name).join(", ")}`);
  if (m.images) parts.push(`[${m.images} ${m.images === 1 ? "image" : "images"}]`);
  return parts.join(" · ") || "(no text)";
}

const ROW = "flex min-w-0 items-baseline gap-2.5 border-b border-vw-rule-soft/40 px-1 py-[3px] text-[12px] leading-[1.45]";

function Row({ msg, active, children }: { msg: string; active: string | null; children: ReactNode }) {
  return (
    <div
      data-testid="prompt-row"
      data-msg={msg}
      aria-current={active === msg ? "true" : undefined}
      className={cn(ROW, active === msg && "rounded bg-chat-surface-2 outline outline-1 outline-chat-accent")}
    >
      {children}
    </div>
  );
}

function FullRow({ msg, active, role, text, chars }: { msg: string; active: string | null; role: string; text: string; chars: number }) {
  return (
    <Row msg={msg} active={active}>
      <span className="w-[68px] flex-none overflow-hidden text-ellipsis font-mono text-[11px] text-chat-muted">{role}</span>
      <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">{text}</span>
      <span className="flex-none tabular-nums text-[11px] text-chat-dim">{fmt(chars)} chars</span>
    </Row>
  );
}

function PromptRows({ r, active }: { r: RequestView; active: string | null }) {
  const p = r.prompt;
  if (!p) return null;
  const older = p.older;
  const names = p.tools.names.join(", ");
  return (
    <div className="whitespace-normal">
      {p.sys && <FullRow msg="sys" active={active} role="system" text={oneLine(p.sys.text) || "(no text)"} chars={p.sys.chars} />}
      {p.tools.count > 0 && (
        <FullRow
          msg="tools"
          active={active}
          role="tools"
          text={`${p.tools.count} ${p.tools.count === 1 ? "tool" : "tools"}${names ? `: ${names}${p.tools.count > p.tools.names.length ? ", …" : ""}` : ""}`}
          chars={p.tools.chars}
        />
      )}
      {p.omitted > 0 && (
        <Row msg="omitted" active={active}>
          <span className="text-chat-dim">
            {fmt(p.omitted)} earlier {p.omitted === 1 ? "message" : "messages"} · counted only
          </span>
        </Row>
      )}
      {Array.from(older.roles).map((c, k) => {
        const i = older.from + k;
        return (
          <Row key={`o${i}`} msg={String(i)} active={active}>
            <span className="text-chat-dim">{`msg ${i + 1} · ${OLDER_ROLE[c] ?? "unknown"} · ${fmt(older.chars[k] ?? 0)} chars`}</span>
          </Row>
        );
      })}
      {p.messages.map((m) => (
        <FullRow key={`m${m.i}`} msg={String(m.i)} active={active} role={m.role} text={msgExcerpt(m)} chars={m.chars} />
      ))}
    </div>
  );
}
