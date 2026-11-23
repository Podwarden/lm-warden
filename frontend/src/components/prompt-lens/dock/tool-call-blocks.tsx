"use client";

// Prompt Lens pop-up blocks (spec 2026-11-22 §3.3, mockup states 11 and 12):
// one collapsible block per tool call (name, call id, size, "arguments
// streaming" on the live one, the arguments as pretty-printed JSON with
// coloured keys once they parse, raw text before that) and one per tool
// result in the prompt (tool name, call id, size, message index, new or
// cached). Model output is always rendered as text, never as HTML.
//
// Search marks come from the pop-up: per block, the hits in the text the
// block shows (the pretty JSON when it parses, else the raw arguments), the
// current one in amber. A closed block renders no body.

import { memo, type ReactNode } from "react";

import type { JsonToken, JsonTokenKind, PrettyJson } from "@/lib/prompt-lens/pretty-json";
import { cn } from "@/lib/utils";

export type Hit = readonly [number, number];
const NO_HITS: readonly Hit[] = [];

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

const TOKEN_CLASS: Record<JsonTokenKind, string> = {
  key: "text-vw-info-fg",
  str: "text-vw-amber-fg",
  num: "text-vw-ok-fg",
  lit: "text-vw-ok-fg",
};

const MARK = "rounded-[2px] bg-vw-amber-fg/25 px-px text-chat-fg";
const MARK_CUR = "rounded-[2px] bg-chat-accent px-px text-chat-on-accent";

/** `text` with search marks (`hits`, numbered from `hitBase`; `cur` is the
 *  current hit's global number) and optional JSON token colours. Plain text
 *  nodes only: nothing in `text` is ever parsed as HTML. */
export function Highlighted({
  text,
  hits = NO_HITS,
  hitBase = 0,
  cur = -1,
  tokens,
}: {
  text: string;
  hits?: readonly Hit[];
  hitBase?: number;
  cur?: number;
  tokens?: readonly JsonToken[];
}) {
  if (!hits.length && !tokens?.length) return <>{text}</>;
  const out: ReactNode[] = [];
  let k = 0; // token pointer
  let key = 0;
  const emit = (a: number, b: number, into: ReactNode[]) => {
    let pos = a;
    if (tokens) {
      while (k < tokens.length && tokens[k][1] <= a) k++;
      let j = k;
      while (j < tokens.length && tokens[j][0] < b) {
        const [ts, te, kind] = tokens[j];
        const s = Math.max(ts, a);
        const e = Math.min(te, b);
        if (s > pos) into.push(text.slice(pos, s));
        into.push(
          <span key={key++} className={TOKEN_CLASS[kind]}>
            {text.slice(s, e)}
          </span>,
        );
        pos = e;
        if (te > b) break;
        j++;
      }
    }
    if (b > pos) into.push(text.slice(pos, b));
  };
  let pos = 0;
  hits.forEach(([s, e], i) => {
    if (s < pos || e > text.length) return;
    emit(pos, s, out);
    const inner: ReactNode[] = [];
    emit(s, e, inner);
    const n = hitBase + i;
    out.push(
      <mark key={key++} data-hit={n} data-current={n === cur ? "true" : undefined} className={n === cur ? MARK_CUR : MARK}>
        {inner}
      </mark>,
    );
    pos = e;
  });
  emit(pos, text.length, out);
  return <>{out}</>;
}

export interface BlockSearch {
  hits: readonly Hit[];
  /** Global number of this block's first hit. */
  base: number;
}

const DETAILS =
  "mb-2.5 min-w-0 rounded-md border border-vw-queue/40 bg-vw-queue/10 [&[open]>summary_.chev]:rotate-90";
const SUMMARY =
  "flex min-w-0 cursor-pointer list-none items-center gap-2 overflow-hidden whitespace-nowrap px-2.5 py-1.5 text-[12px] [&::-webkit-details-marker]:hidden focus-visible:outline focus-visible:outline-2 focus-visible:outline-chat-accent";
const PRE = "m-0 border-t px-3 py-2 font-mono text-[12.5px] leading-[1.55] text-chat-fg";
const PILL = "flex-none rounded border border-vw-amber-fg/40 px-[5px] text-[10px] leading-4 text-vw-amber-fg";

export interface CallBlock {
  key: string;
  index: number;
  callId: string | null;
  name: string;
  args: string;
  /** The text shown: the pretty JSON when the arguments parse, else raw. */
  pretty: PrettyJson | null;
  /** Cut by the server's 256k budget or the dock's 200k bound. */
  elided: boolean;
}

/** The text a call block shows (what search runs over). */
export const callText = (c: Pick<CallBlock, "args" | "pretty">) => c.pretty?.text ?? c.args;

interface ToolCallBlocksProps {
  calls: CallBlock[];
  /** The request still streams: the last call's arguments are arriving.
   *  No call is "done" before request_end (ruling G-F4). */
  live: boolean;
  wrap?: boolean;
  open?: Record<string, boolean>;
  onOpenChange?: (key: string, open: boolean) => void;
  search?: BlockSearch[];
  cur?: number;
}

export function ToolCallBlocks({ calls, live, wrap = true, open, onOpenChange, search, cur = -1 }: ToolCallBlocksProps) {
  if (!calls.length) return <p className="m-0 text-[12px] italic text-chat-dim">No tool calls.</p>;
  const last = calls.length - 1;
  return (
    <>
      {calls.map((c, i) => (
        <CallDetails
          key={c.key}
          call={c}
          streaming={live && i === last}
          done={!live}
          wrap={wrap}
          isOpen={open?.[c.key] ?? true}
          onOpenChange={onOpenChange}
          search={search?.[i]}
          cur={cur}
        />
      ))}
    </>
  );
}

const CallDetails = memo(function CallDetails({
  call,
  streaming,
  done,
  wrap,
  isOpen,
  onOpenChange,
  search,
  cur,
}: {
  call: CallBlock;
  streaming: boolean;
  done: boolean;
  wrap: boolean;
  isOpen: boolean;
  onOpenChange?: (key: string, open: boolean) => void;
  search?: BlockSearch;
  cur: number;
}) {
  const text = callText(call);
  const base = search?.base ?? 0;
  const hits = search?.hits ?? NO_HITS;
  const curHere = cur >= base && cur < base + hits.length ? cur : -1;
  return (
    <details
      data-testid={`tool-call-${call.index}`}
      open={isOpen}
      onToggle={(e) => {
        const now = (e.currentTarget as HTMLDetailsElement).open;
        if (now !== isOpen) onOpenChange?.(call.key, now);
      }}
      className={DETAILS}
    >
      <summary className={SUMMARY}>
        <span aria-hidden="true" className="chev inline-block flex-none text-[10px] text-chat-dim transition-transform">
          ▸
        </span>
        <span className="min-w-0 overflow-hidden text-ellipsis font-mono font-medium text-vw-queue">
          {call.name || "(unnamed call)"}
        </span>
        <span className="min-w-0 overflow-hidden text-ellipsis font-mono text-chat-dim">{call.callId ?? "no call id"}</span>
        {call.elided && (
          <span className={PILL} title="Arguments cut: past 256k chars on the server, or past 200k in this browser">
            trimmed
          </span>
        )}
        <span className="ml-auto flex-none text-[11px] tabular-nums text-chat-dim">
          {streaming && <span className="text-vw-ok-fg">● arguments streaming · </span>}
          {done && !call.elided && <span>arguments done · </span>}
          {fmt(call.args.length)} chars
          {call.pretty === null && call.args && !streaming ? " · not JSON" : ""}
        </span>
      </summary>
      {isOpen && (
        <pre
          data-testid={`tool-call-args-${call.index}`}
          data-json={call.pretty ? "true" : "false"}
          className={cn(PRE, "border-vw-queue/40", wrap ? "whitespace-pre-wrap [overflow-wrap:anywhere]" : "overflow-x-auto whitespace-pre")}
        >
          <Highlighted text={text} hits={hits} hitBase={base} cur={curHere} tokens={call.pretty?.tokens} />
          {streaming && <span aria-hidden="true" className="ml-px inline-block h-[1em] w-[2px] animate-pulse bg-chat-accent align-text-bottom" />}
        </pre>
      )}
    </details>
  );
});

export interface ResultBlock {
  key: string;
  /** Tool name, or the role for a prompt message. */
  title: string;
  callId?: string | null;
  /** Message (node) index, 0-based (shown from 1, like the map's nodes);
   *  null for sys and tools. */
  msg: number | null;
  chars: number;
  text: string;
  elided: boolean;
  /** new / cached / loose match / cached by another key, from the kvsig
   *  state of the message's node; null when there is no prompt map. */
  state?: string | null;
}

interface ResultBlocksProps {
  blocks: ResultBlock[];
  wrap?: boolean;
  open?: Record<string, boolean>;
  onOpenChange?: (key: string, open: boolean) => void;
  search?: BlockSearch[];
  cur?: number;
  empty: string;
  /** Prompt messages (role titles) rather than tool results. */
  variant?: "results" | "prompt";
}

export function ResultBlocks({ blocks, wrap = true, open, onOpenChange, search, cur = -1, empty, variant = "results" }: ResultBlocksProps) {
  if (!blocks.length) return <p className="m-0 text-[12px] italic text-chat-dim">{empty}</p>;
  return (
    <>
      {blocks.map((b, i) => (
        <ResultDetails
          key={b.key}
          block={b}
          wrap={wrap}
          isOpen={open?.[b.key] ?? true}
          onOpenChange={onOpenChange}
          search={search?.[i]}
          cur={cur}
          variant={variant}
        />
      ))}
    </>
  );
}

const ResultDetails = memo(function ResultDetails({
  block,
  wrap,
  isOpen,
  onOpenChange,
  search,
  cur,
  variant,
}: {
  block: ResultBlock;
  wrap: boolean;
  isOpen: boolean;
  onOpenChange?: (key: string, open: boolean) => void;
  search?: BlockSearch;
  cur: number;
  variant: "results" | "prompt";
}) {
  const base = search?.base ?? 0;
  const hits = search?.hits ?? NO_HITS;
  const curHere = cur >= base && cur < base + hits.length ? cur : -1;
  return (
    <details
      data-testid={`result-block-${block.key}`}
      open={isOpen}
      onToggle={(e) => {
        const now = (e.currentTarget as HTMLDetailsElement).open;
        if (now !== isOpen) onOpenChange?.(block.key, now);
      }}
      className={cn(
        DETAILS,
        variant === "results" ? "border-vw-ttft/40 bg-vw-ttft/10" : "border-chat-rule bg-chat-surface",
      )}
    >
      <summary className={SUMMARY}>
        <span aria-hidden="true" className="chev inline-block flex-none text-[10px] text-chat-dim transition-transform">
          ▸
        </span>
        <span className={cn("min-w-0 overflow-hidden text-ellipsis font-mono font-medium", variant === "results" ? "text-vw-ttft" : "text-chat-fg")}>
          {block.title}
        </span>
        {block.callId && <span className="min-w-0 overflow-hidden text-ellipsis font-mono text-chat-dim">{block.callId}</span>}
        {block.elided && (
          <span className={PILL} title="Windowed in the capture: the middle of this message is not kept">
            trimmed
          </span>
        )}
        <span className="ml-auto flex-none text-[11px] tabular-nums text-chat-dim">
          {fmt(block.chars)} chars{block.msg !== null ? ` · msg ${block.msg + 1}` : ""}
          {block.state ? ` · ${block.state}` : ""}
        </span>
      </summary>
      {isOpen && (
        <pre
          className={cn(
            PRE,
            variant === "results" ? "border-vw-ttft/40" : "border-chat-rule",
            wrap ? "whitespace-pre-wrap [overflow-wrap:anywhere]" : "overflow-x-auto whitespace-pre",
          )}
        >
          {block.text ? (
            <Highlighted text={block.text} hits={hits} hitBase={base} cur={curHere} />
          ) : (
            <span className="italic text-chat-dim">(no text)</span>
          )}
        </pre>
      )}
    </details>
  );
});
