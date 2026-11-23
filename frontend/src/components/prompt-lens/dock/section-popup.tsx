"use client";

// The Prompt Lens section pop-up (spec 2026-11-22 §3.3, mockup states 10-12):
// one section of one request, whole, in a modal over the window (24 px
// margin; full screen on a phone).
//
// Whole section, then live (ruling G-F3). On open the pop-up fetches the
// whole section from server memory (`fetchSection`, ?from=0). While the
// request streams, the store's section (Task 9) keeps growing: what it holds
// past the fetched end, `offset + text.length`, is appended. Positions are
// absolute char offsets, never seq (a key holder gets `seq: null`). A store
// that joined mid-stream counts from a later start; it is aligned to the
// server text by content once per fetch. When the store has dropped text past
// the fetched end (its 200k bound), the pop-up "resumes": it refetches with
// ?from=<end> and appends.
//
// Tool calls: the server's calls (256k cap per request, G-R5) merged with the
// store's (cut at the tail past 200k); a call's arguments are the longer of
// the two when one continues the other. No call is done before request_end
// (G-F4). Tool results and the prompt come from the structured capture.
//
// A 404 (the same for unknown, released and another key's id) keeps the
// dock's copy and says the request left server memory; a 429 (kept capture
// reads are limited to 60 a minute per key) says to retry.
//
// Text is always rendered as text, never as HTML. Reply would render as
// markdown with a raw toggle, but @podwarden/chat-ui 0.1.28 exports no
// <Markdown>, so Reply shows as pre-wrapped text and there is no raw toggle.
//
// Search (Q13): plain, case-insensitive, Enter / Shift+Enter and ↑ ↓, "4 of
// 67". Hits are found over the whole text (appended text only, as it grows);
// marks are drawn only in the blocks on screen and in the current hit's block.
// A current hit stops following; "↓ Jump to latest" clears the search.

import { copyToClipboard } from "@podwarden/chat-ui";
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useBackLayer } from "@/lib/prompt-lens/back-layer";
import type { PromptMsg, SectionResponse } from "@/lib/prompt-lens/events";
import { mapParts } from "@/lib/prompt-lens/kvsig";
import { prettyJson, type PrettyJson } from "@/lib/prompt-lens/pretty-json";
import { copyText, findHits, findHitsFrom, MAX_HITS, resultHead, splitBlocks } from "@/lib/prompt-lens/search";
import type { RequestView, Section, ToolCallView } from "@/lib/prompt-lens/store";
import { fetchSection, fetchStatus, type FetchKind, type LensScope } from "@/lib/prompt-lens/stream";
import { LiveNumber } from "@/components/ui/live-number";
import { cn } from "@/lib/utils";
import { LIVE_CHARS_CH, LIVE_TOK_CH, SECTION_BIG_TOK } from "./section-box";
import { TailText } from "./tail-text";
import {
  Highlighted,
  ResultBlocks,
  ToolCallBlocks,
  callText,
  type BlockSearch,
  type CallBlock,
  type Hit,
  type ResultBlock,
} from "./tool-call-blocks";
import { FollowTailContext, useFollowTail } from "./use-follow-tail";

export interface SectionPopupProps {
  scope: LensScope;
  kind: FetchKind;
  /** The request as the dock's store holds it now (re-rendered as it streams). */
  request: RequestView;
  onClose: () => void;
  /** Phone width (spec section 6, mockup state 12): full screen, 44 px
   *  controls, the toolbar on two rows, and the back gesture closes it. */
  phone?: boolean;
}

const NAMES: Record<FetchKind, string> = {
  thinking: "Thinking",
  reply: "Reply",
  tool_calls: "Tool calls",
  tool_results: "Tool results in the prompt",
  prompt: "Prompt by role",
};

/** Kinds that grow while the request streams. */
const STREAMED = new Set<FetchKind>(["thinking", "reply", "tool_calls"]);
/** Search waits this long after typing over a section this large. */
const DEBOUNCE_CHARS = 200_000;
const DEBOUNCE_MS = 150;
/** Tool calls the store has cut are refetched at most this often while live. */
const CALLS_REFRESH_MS = 2000;
/** Blocks open by default up to this many; past it only the newest ones. */
const OPEN_ALL_MAX = 50;
const OPEN_NEWEST = 10;
const SAMPLE = 64;
/** The shortest head sample the alignment tries. */
const MIN_SAMPLE = 4;
/** A failed resume fetch is retried this often at most, waiting 0.5 s, 1 s, 2 s, ... */
const RESUME_TRIES = 5;
const RESUME_BACKOFF_MS = 500;

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");
const EMPTY_HITS: Hit[] = [];

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

// ---------------------------------------------------------------------------
// The fetched section

interface ServerCall {
  index: number;
  call_id: string | null;
  name: string;
  args: string;
  elided?: boolean;
}

interface Whole {
  /** Bumped on every fetch that does not continue the text. */
  gen: number;
  live: boolean;
  kept: boolean;
  /** Text kinds: absolute offset of `text`; `offset + text.length` is the end. */
  offset: number;
  text: string;
  dropped: number;
  calls: ServerCall[] | null;
  droppedCalls: number;
  messages: Record<string, unknown>[] | null;
  older: Record<string, unknown> | null;
}

type Load = { state: "loading" } | { state: "ok" } | { state: "error"; status: number };

function toCalls(raw: SectionResponse["calls"]): ServerCall[] | null {
  if (!Array.isArray(raw)) return null;
  return raw.map((c, k) => {
    const x = c as Record<string, unknown>;
    return {
      index: typeof x.index === "number" ? x.index : k,
      call_id: typeof x.call_id === "string" ? x.call_id : null,
      name: typeof x.name === "string" ? x.name : "",
      args: typeof x.args === "string" ? x.args : "",
      elided: x.elided === true,
    };
  });
}

function toWhole(r: SectionResponse, gen: number): Whole {
  return {
    gen,
    live: !!r.live,
    kept: !!r.kept,
    offset: Math.max(0, r.offset ?? 0),
    text: typeof r.text === "string" ? r.text : "",
    dropped: Math.max(0, r.dropped_chars ?? 0),
    calls: toCalls(r.calls),
    droppedCalls: Math.max(0, r.dropped_calls ?? 0),
    messages: Array.isArray(r.messages) ? (r.messages as Record<string, unknown>[]) : null,
    older: r.older && typeof r.older === "object" ? (r.older as Record<string, unknown>) : null,
  };
}

/** `w` continued by a ?from=<end> fetch. A response that starts past the end
 *  (the server dropped the text in between) starts the text over. */
function continued(w: Whole, r: SectionResponse): Whole {
  const end = w.offset + w.text.length;
  const next = toWhole(r, w.gen);
  if (next.offset > end) return { ...next, gen: w.gen + 1 };
  const add = next.text.slice(end - next.offset);
  return { ...w, live: next.live, kept: next.kept, dropped: next.dropped, text: add ? w.text + add : w.text };
}

// ---------------------------------------------------------------------------
// Store alignment (text kinds)

/** Does the store's text, moved by `shift`, agree with the server text where
 *  both have it (the last SAMPLE chars of the overlap)? */
function shiftValid(w: Whole, sec: Section, shift: number): boolean {
  const cl = sec.clipped + shift;
  const lo = Math.max(cl, w.offset);
  const hi = Math.min(cl + sec.text.length, w.offset + w.text.length);
  if (hi <= lo) return true;
  const n = Math.min(SAMPLE, hi - lo);
  return sec.text.slice(hi - n - cl, hi - cl) === w.text.slice(hi - n - w.offset, hi - w.offset);
}

/** How far the store's char counts are behind the server's absolute offsets,
 *  or null when the two texts cannot be aligned. Tried in turn:
 *   - 0, when the counts already agree;
 *   - the store's last chars found in the server text (the store is behind
 *     or level with the fetched end);
 *   - the store's first chars found in the server text, the rest checked
 *     (a store that joined mid-stream and is already past the fetched end:
 *     its last chars are newer than the fetch, so the first test misses);
 *   - the server's first chars found in the store (the server dropped a
 *     head the store still holds).
 *  Each candidate must agree with the server text over the overlap. */
function findShift(w: Whole, sec: Section): number | null {
  const sEnd = sec.clipped + sec.text.length;
  const wEnd = w.offset + w.text.length;
  const overlap = Math.min(sEnd, wEnd) > Math.max(sec.clipped, w.offset);
  if (!overlap || shiftValid(w, sec, 0)) return 0;
  if (!sec.text.length || !w.text.length) return 0;
  const n = Math.min(SAMPLE, sec.text.length);
  const tailAt = w.text.lastIndexOf(sec.text.slice(-n));
  if (tailAt >= 0) {
    const shift = w.offset + tailAt + n - sEnd;
    if (shiftValid(w, sec, shift)) return shift;
  }
  // The head sample must lie inside the server text; how far the store runs
  // past it is unknown, so a short store tries shorter samples.
  for (let k = n; k >= MIN_SAMPLE; k = Math.floor(k / 2)) {
    const head = sec.text.slice(0, k);
    for (let at = w.text.indexOf(head); at >= 0; at = w.text.indexOf(head, at + 1)) {
      const shift = w.offset + at - sec.clipped;
      if (shiftValid(w, sec, shift)) return shift;
    }
  }
  const m = Math.min(SAMPLE, w.text.length);
  const wHead = w.text.slice(0, m);
  for (let at = sec.text.indexOf(wHead); at >= 0; at = sec.text.indexOf(wHead, at + 1)) {
    const shift = w.offset - (sec.clipped + at);
    if (shiftValid(w, sec, shift)) return shift;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Tool calls

const continues = (long: string, short: string) => {
  if (long.length <= short.length) return false;
  const n = Math.min(SAMPLE, short.length);
  return long.slice(short.length - n, short.length) === short.slice(short.length - n);
};

interface MergedCall {
  index: number;
  callId: string | null;
  name: string;
  args: string;
  elided: boolean;
}

/** The merged calls, and `stale`: a store call is longer than the server's
 *  but does not continue it (the store joined mid-call), so only a refetch
 *  brings the server copy up to date. */
function mergeCalls(server: ServerCall[] | null, store: ToolCallView[]): { calls: MergedCall[]; stale: boolean } {
  const byIndex = new Map<number, MergedCall>();
  let stale = false;
  for (const s of server ?? []) {
    byIndex.set(s.index, { index: s.index, callId: s.call_id, name: s.name, args: s.args, elided: !!s.elided });
  }
  for (const c of store) {
    const s = byIndex.get(c.index);
    if (!s) {
      byIndex.set(c.index, { index: c.index, callId: c.callId, name: c.name, args: c.args, elided: c.elided });
    } else if (server === null || continues(c.args, s.args)) {
      byIndex.set(c.index, {
        ...s,
        callId: s.callId ?? c.callId,
        name: s.name || c.name,
        args: c.args,
        elided: s.elided || c.elided,
      });
    } else if (c.args.length > s.args.length) {
      stale = true;
    }
  }
  return { calls: [...byIndex.values()].sort((a, b) => a.index - b.index), stale };
}

// ---------------------------------------------------------------------------
// Tool results and the prompt

const isResult = (role: unknown) => role === "tool" || role === "function";

function stateLabel(state: string): string {
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

function promptText(m: PromptMsg): string {
  const out: string[] = [];
  if (m.reasoning) out.push(`[reasoning]\n${m.reasoning}`);
  if (m.text) out.push(m.text);
  for (const c of m.tool_calls ?? []) out.push(`→ ${c.name}(${c.args})`);
  if (m.calls_omitted) out.push(`[${m.calls_omitted} more tool calls not captured]`);
  if (m.images) out.push(`[${m.images} image${m.images === 1 ? "" : "s"}]`);
  return out.join("\n\n");
}

function asMsg(m: Record<string, unknown>): PromptMsg {
  return {
    ...(m as unknown as PromptMsg),
    i: typeof m.i === "number" ? m.i : -1,
    role: typeof m.role === "string" ? m.role : "?",
    chars: typeof m.chars === "number" ? m.chars : typeof m.text === "string" ? m.text.length : 0,
    text: typeof m.text === "string" ? m.text : "",
    elided: m.elided === true,
  };
}

// ---------------------------------------------------------------------------
// Search

/** Hits of `q` in `text`, extended (not recomputed) while `text` only grows
 *  within the same `version`. */
function useTextHits(text: string, q: string, version: string): { hits: Hit[]; ver: number } {
  const memo = useRef<{ text: string; q: string; version: string; hits: Hit[]; ver: number } | null>(null);
  return useMemo(() => {
    const p = memo.current;
    let hits: Hit[];
    let ver = p?.ver ?? 0;
    if (!q) hits = EMPTY_HITS;
    else if (p && p.q === q && p.version === version && text.length >= p.text.length) {
      hits = text.length === p.text.length ? p.hits : (findHitsFrom(text, q, p.hits as [number, number][], p.text.length) as Hit[]);
    } else {
      hits = findHits(text, q);
      ver += 1;
    }
    memo.current = { text, q, version, hits, ver };
    return { hits, ver };
  }, [text, q, version]);
}

/** Per-block hits for block lists (calls, results, prompt): a block's hits are
 *  kept while its text is unchanged and extended while it only grows. */
function useBlockHits(blocks: { key: string; text: string }[], q: string): { search: BlockSearch[]; total: number } {
  const cache = useRef(new Map<string, { text: string; q: string; hits: Hit[] }>());
  return useMemo(() => {
    const next = new Map<string, { text: string; q: string; hits: Hit[] }>();
    const search: BlockSearch[] = [];
    let total = 0;
    for (const b of blocks) {
      let hits: Hit[] = EMPTY_HITS;
      if (q) {
        const p = cache.current.get(b.key);
        if (p && p.q === q && p.text === b.text) hits = p.hits;
        else if (p && p.q === q && continues(b.text, p.text)) {
          hits = findHitsFrom(b.text, q, p.hits as [number, number][], p.text.length) as Hit[];
        } else hits = findHits(b.text, q) as Hit[];
        next.set(b.key, { text: b.text, q, hits });
      }
      search.push({ hits, base: total });
      total += hits.length;
    }
    cache.current = next;
    return { search, total };
  }, [blocks, q]);
}

function lowerBound(hits: readonly Hit[], pos: number): number {
  let lo = 0;
  let hi = hits.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (hits[mid][0] < pos) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Which blocks of `root`'s `[data-block]` children are on screen (with a
 *  margin), or null when the browser cannot tell (all count as visible). */
function useVisibleBlocks(container: React.RefObject<HTMLElement | null>, scrollRoot: React.RefObject<HTMLElement | null>) {
  const [visible, setVisible] = useState<Set<number> | null>(() =>
    typeof IntersectionObserver === "undefined" ? null : new Set(),
  );
  const io = useRef<IntersectionObserver | null>(null);
  const seen = useRef(new WeakSet<Element>());
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const obs = new IntersectionObserver(
      (entries) => {
        setVisible((prev) => {
          const out = new Set(prev ?? []);
          for (const e of entries) {
            const n = Number((e.target as HTMLElement).dataset.block);
            if (e.isIntersecting) out.add(n);
            else out.delete(n);
          }
          return out;
        });
      },
      { root: scrollRoot.current, rootMargin: "400px 0px" },
    );
    io.current = obs;
    return () => {
      obs.disconnect();
      io.current = null;
      seen.current = new WeakSet();
    };
  }, [scrollRoot]);
  useEffect(() => {
    const obs = io.current;
    const el = container.current;
    if (!obs || !el) return;
    el.querySelectorAll("[data-block]").forEach((b) => {
      if (!seen.current.has(b)) {
        seen.current.add(b);
        obs.observe(b);
      }
    });
  });
  return visible;
}

interface TextBlockProps {
  n: number;
  text: string;
  start: number;
  highlight: boolean;
  hits: readonly Hit[];
  hitFrom: number;
  hitTo: number;
  cur: number;
  /** Changes when `hits` was recomputed (another query or text). */
  ver: number;
}

const TextBlock = memo(
  function TextBlock({ n, text, start, highlight, hits, hitFrom, hitTo, cur }: TextBlockProps) {
    const local = useMemo(
      () => (highlight ? hits.slice(hitFrom, hitTo).map(([s, e]) => [s - start, e - start] as const) : EMPTY_HITS),
      [highlight, hits, hitFrom, hitTo, start],
    );
    return (
      <div data-block={n}>
        {highlight ? <Highlighted text={text} hits={local} hitBase={hitFrom} cur={cur} /> : text}
      </div>
    );
  },
  (a, b) =>
    a.n === b.n &&
    a.start === b.start &&
    a.text.length === b.text.length &&
    a.highlight === b.highlight &&
    a.hitFrom === b.hitFrom &&
    a.hitTo === b.hitTo &&
    a.ver === b.ver &&
    // the current hit matters only to the blocks it enters or leaves
    (a.cur >= a.hitFrom && a.cur < a.hitTo) === (b.cur >= b.hitFrom && b.cur < b.hitTo) &&
    (!(b.cur >= b.hitFrom && b.cur < b.hitTo) || a.cur === b.cur) &&
    (a.text === b.text || a.ver === b.ver),
);

/** A text section with search marks, in blocks; only the visible blocks and
 *  the current hit's block are marked. `version` names the text: blocks of
 *  the same version and range hold the same text. */
function SearchText({
  text,
  hits,
  ver,
  cur,
  scrollRoot,
}: {
  text: string;
  hits: readonly Hit[];
  ver: number;
  cur: number;
  scrollRoot: React.RefObject<HTMLElement | null>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const bounds = useMemo(() => splitBlocks(text), [text]);
  const visible = useVisibleBlocks(ref, scrollRoot);
  const blocks = [];
  for (let n = 0; n + 1 < bounds.length; n++) {
    const s = bounds[n];
    const e = bounds[n + 1];
    const hitFrom = lowerBound(hits, s);
    const hitTo = lowerBound(hits, e);
    const hasCur = cur >= hitFrom && cur < hitTo;
    const highlight = hitTo > hitFrom && (visible === null || visible.has(n) || hasCur);
    blocks.push(
      <TextBlock
        key={s}
        n={n}
        text={text.slice(s, e)}
        start={s}
        highlight={highlight}
        hits={hits}
        hitFrom={hitFrom}
        hitTo={hitTo}
        cur={cur}
        ver={ver}
      />,
    );
  }
  return <div ref={ref}>{blocks}</div>;
}

// ---------------------------------------------------------------------------
// Messages

function errorText(status: number, hasCopy: boolean): string {
  switch (status) {
    case 404:
      return `This request left server memory.${hasCopy ? " Showing what this page still holds." : " The section is not available."}`;
    case 429:
      return "Too many reads of kept captures for this key. Try again in a minute.";
    case 409:
      return "Prompt Lens is off on this server.";
    case 401:
      return "Your sign-in has expired: sign in again to load the whole section.";
    default:
      return `Could not load the whole section${status ? ` (HTTP ${status})` : ""}.`;
  }
}

const BTN =
  "flex-none cursor-pointer whitespace-nowrap rounded border border-chat-rule bg-transparent px-2 py-[3px] text-[12px] text-chat-muted hover:border-chat-muted hover:text-chat-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-chat-accent disabled:cursor-default disabled:opacity-40";
const BTN_ON = "border-chat-accent/70 text-chat-fg";
/** A phone's touch target (44 px). */
const BTN_PHONE = "inline-flex min-h-11 min-w-11 items-center justify-center px-3";
const PILL = "flex-none rounded border px-[5px] text-[10px] leading-4 tabular-nums";

// ---------------------------------------------------------------------------

export function SectionPopup({ scope, kind, request, onClose, phone = false }: SectionPopupProps) {
  const name = NAMES[kind];
  const reqId = request.reqId;
  const ended = request.endedTs !== null;

  // --- fetch -------------------------------------------------------------
  const [whole, setWhole] = useState<Whole | null>(null);
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [attempt, setAttempt] = useState(0);
  const genRef = useRef(0);
  const inFlight = useRef(false);

  useEffect(() => {
    let alive = true;
    setLoad({ state: "loading" });
    inFlight.current = true;
    fetchSection(scope, reqId, kind, 0).then(
      (r) => {
        if (!alive) return;
        genRef.current += 1;
        setWhole(toWhole(r, genRef.current));
        setLoad({ state: "ok" });
      },
      (e) => {
        if (alive) setLoad({ state: "error", status: fetchStatus(e) });
      },
    ).finally(() => {
      if (alive) inFlight.current = false;
    });
    return () => {
      alive = false;
      inFlight.current = false;
    };
  }, [scope, reqId, kind, attempt]);

  const live = !ended && (whole?.live ?? true);

  // --- text kinds: whole + live tail -------------------------------------
  const isText = kind === "thinking" || kind === "reply";
  const sec: Section | null = kind === "thinking" ? request.thinking : kind === "reply" ? request.reply : null;
  // The store's alignment to the fetched text, found once per fetch and again
  // only when it stops agreeing (the store lost a delta). When no alignment
  // is found, the store's text from that moment on is appended at the fetched
  // end and the shift is kept (`sticky`): re-deriving it on every delta would
  // pin the store's end to the fetched end and append nothing, which froze
  // the pop-up (owner bug 2026-11-23).
  const shiftRef = useRef<{ gen: number; shift: number; sticky: boolean }>({ gen: -1, shift: 0, sticky: false });
  let display = { offset: 0, text: "" };
  let resumeFrom: number | null = null;
  if (sec) {
    if (whole) {
      const sr = shiftRef.current;
      if (sr.gen !== whole.gen || (!sr.sticky && !shiftValid(whole, sec, sr.shift))) {
        const found = findShift(whole, sec);
        const sEnd = sec.clipped + sec.text.length;
        const wEnd = whole.offset + whole.text.length;
        shiftRef.current =
          found !== null ? { gen: whole.gen, shift: found, sticky: false } : { gen: whole.gen, shift: wEnd - sEnd, sticky: true };
      }
      const shift = shiftRef.current.shift;
      const cl = sec.clipped + shift;
      const sEnd = cl + sec.text.length;
      const wEnd = whole.offset + whole.text.length;
      if (cl > wEnd) resumeFrom = wEnd;
      const tail = sEnd > wEnd && cl <= wEnd ? sec.text.slice(wEnd - cl) : "";
      display = { offset: whole.offset, text: tail ? whole.text + tail : whole.text };
    } else {
      display = { offset: sec.clipped, text: sec.text };
    }
  }

  // Resume: the store dropped text past our end; fetch from it. A fetch that
  // was skipped (another in flight) or failed is retried with backoff,
  // RESUME_TRIES times at most.
  const [resumeTry, setResumeTry] = useState(0);
  const resumeFails = useRef(0);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (retryTimer.current) clearTimeout(retryTimer.current);
  }, []);
  useEffect(() => {
    if (resumeFrom === null || !whole) return;
    const retry = () => {
      if (resumeFails.current >= RESUME_TRIES) return;
      const wait = RESUME_BACKOFF_MS * 2 ** resumeFails.current;
      resumeFails.current += 1;
      if (retryTimer.current) clearTimeout(retryTimer.current);
      retryTimer.current = setTimeout(() => setResumeTry((n) => n + 1), wait);
    };
    if (inFlight.current) {
      retry();
      return;
    }
    let alive = true;
    inFlight.current = true;
    fetchSection(scope, reqId, kind, resumeFrom)
      .then(
        (r) => {
          if (!alive) return;
          resumeFails.current = 0;
          setWhole((w) => {
            if (!w) return w;
            const next = continued(w, r);
            if (next.gen !== w.gen) genRef.current = next.gen;
            return next;
          });
        },
        () => {
          if (alive) retry();
        },
      )
      .finally(() => {
        inFlight.current = false;
      });
    return () => {
      alive = false;
    };
    // `whole` changes with every append; only a new resume point or a retry refetches.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumeFrom, scope, reqId, kind, resumeTry]);

  // --- calls -------------------------------------------------------------
  const prettyCache = useRef(new Map<string, PrettyJson | null>());
  const { calls, stale: callsStale } = useMemo((): { calls: CallBlock[]; stale: boolean } => {
    if (kind !== "tool_calls") return { calls: [], stale: false };
    const { calls: merged, stale } = mergeCalls(whole?.calls ?? null, request.toolCalls);
    const next = new Map<string, PrettyJson | null>();
    const out = merged.map((c) => {
      const key = `c${c.index}`;
      let pretty = prettyCache.current.get(c.args);
      if (pretty === undefined) pretty = prettyJson(c.args);
      next.set(c.args, pretty);
      return { key, ...c, pretty };
    });
    prettyCache.current = next;
    return { calls: out, stale };
  }, [kind, whole?.calls, request.toolCalls]);

  // Tool calls the server copy may lag: the store has cut them, or a store
  // call does not continue the server's. Refetch at most every 2 s while live,
  // and once at request_end when the pop-up saw the request live.
  const needCalls = request.toolCallsClipped > 0 || callsStale;
  const sawLive = useRef(false);
  if (!ended) sawLive.current = true;
  const endRefetched = useRef(false);
  useEffect(() => {
    if (kind !== "tool_calls" || load.state !== "ok") return;
    if (ended) {
      if (endRefetched.current || !sawLive.current) return;
      endRefetched.current = true;
    } else if (!needCalls) return;
    let alive = true;
    const refetch = () => {
      fetchSection(scope, reqId, kind, 0).then(
        (r) => {
          if (!alive) return;
          genRef.current += 1;
          setWhole(toWhole(r, genRef.current));
        },
        () => {},
      );
    };
    // At the end: now (its result bumps `whole.gen`, so the effect re-runs, and
    // must not cancel it). While live: after CALLS_REFRESH_MS.
    if (ended) {
      refetch();
      return;
    }
    const t = setTimeout(refetch, CALLS_REFRESH_MS);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [kind, needCalls, ended, load.state, scope, reqId, whole?.gen]);

  // --- results / prompt --------------------------------------------------
  const scopeParts = useMemo(() => (kind === "tool_results" ? mapParts(request, scope).parts : []), [kind, request, scope]);
  const { blocks: results, older } = useMemo((): { blocks: ResultBlock[]; older: string | null } => {
    if (kind === "tool_results") {
      const raw = whole?.messages ?? (request.prompt?.messages as unknown as Record<string, unknown>[] | undefined) ?? null;
      const msgs = (raw ?? []).map(asMsg).filter((m) => isResult(m.role));
      const stateOf = (i: number) => {
        const p = scopeParts.find((x) => x.p === "nodes" && x.from <= i && i <= x.to);
        return p ? stateLabel(p.state) : null;
      };
      return {
        blocks: msgs.map((m) => ({
          key: String(m.i),
          title: m.tool_name || "tool result",
          callId: m.tool_call_id ?? null,
          msg: m.i,
          chars: m.chars,
          text: m.text,
          elided: m.elided,
          state: stateOf(m.i),
        })),
        older: null,
      };
    }
    if (kind === "prompt") {
      const out: ResultBlock[] = [];
      let collapsed = 0;
      if (whole?.messages) {
        for (const raw of whole.messages) {
          if (raw.role === "sys") {
            out.push({ key: "sys", title: "system", msg: null, chars: Number(raw.chars) || 0, text: String(raw.text ?? ""), elided: raw.elided === true });
          } else if (raw.role === "tools") {
            const names = Array.isArray(raw.names) ? raw.names.filter((n): n is string => typeof n === "string") : [];
            out.push({ key: "tools", title: `tools · ${Number(raw.count) || names.length}`, msg: null, chars: Number(raw.chars) || 0, text: names.join(", "), elided: false });
          } else {
            const m = asMsg(raw);
            out.push({ key: `m${m.i}`, title: m.role, callId: m.tool_call_id ?? null, msg: m.i, chars: m.chars, text: promptText(m), elided: m.elided });
          }
        }
        collapsed = Number(whole.older?.collapsed) || 0;
      } else if (request.prompt) {
        const p = request.prompt;
        if (p.sys) out.push({ key: "sys", title: "system", msg: null, chars: p.sys.chars, text: p.sys.text, elided: p.sys.elided });
        if (p.tools.count) out.push({ key: "tools", title: `tools · ${p.tools.count}`, msg: null, chars: p.tools.chars, text: p.tools.names.join(", "), elided: false });
        for (const m of p.messages) out.push({ key: `m${m.i}`, title: m.role, callId: m.tool_call_id ?? null, msg: m.i, chars: m.chars, text: promptText(m), elided: m.elided });
        collapsed = p.collapsed;
      }
      return { blocks: out, older: collapsed > 0 ? `${fmt(collapsed)} earlier message${collapsed === 1 ? "" : "s"} collapsed in the capture (sizes only)` : null };
    }
    return { blocks: [], older: null };
  }, [kind, whole?.messages, whole?.older, request.prompt, scopeParts]);

  const isBlocks = kind === "tool_calls" || kind === "tool_results" || kind === "prompt";
  const blockTexts = useMemo(
    () => (kind === "tool_calls" ? calls.map((c) => ({ key: c.key, text: callText(c) })) : results.map((b) => ({ key: b.key, text: b.text }))),
    [kind, calls, results],
  );

  // --- search ------------------------------------------------------------
  const [q, setQ] = useState("");
  const [activeQ, setActiveQ] = useState("");
  const [cur, setCur] = useState(0);
  const displayLen = isText ? display.offset + display.text.length : blockTexts.reduce((n, b) => n + b.text.length, 0);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const applyQ = useCallback((v: string) => {
    setActiveQ(v);
    setCur(0);
  }, []);
  const onQ = (v: string) => {
    setQ(v);
    if (debounce.current) clearTimeout(debounce.current);
    if (displayLen > DEBOUNCE_CHARS && v) debounce.current = setTimeout(() => applyQ(v), DEBOUNCE_MS);
    else applyQ(v);
  };
  useEffect(() => () => {
    if (debounce.current) clearTimeout(debounce.current);
  }, []);

  const textVersion = whole ? `w${whole.gen}` : `c${display.offset}`;
  const textHits = useTextHits(isText ? display.text : "", isText ? activeQ : "", textVersion);
  const blockHits = useBlockHits(isBlocks ? blockTexts : [], isBlocks ? activeQ : "");
  const total = isText ? textHits.hits.length : blockHits.total;
  const curIdx = total ? Math.min(cur, total - 1) : -1;
  const hasHit = activeQ !== "" && total > 0;
  const step = (d: number) => {
    if (!total) return;
    setCur((c) => (Math.min(c, total - 1) + d + total) % total);
  };

  // --- open state of blocks ----------------------------------------------
  const [openState, setOpenState] = useState<Record<string, boolean>>({});
  const keys = blockTexts.map((b) => b.key);
  const defaultOpen = (k: string, i: number) => keys.length <= OPEN_ALL_MAX || i >= keys.length - OPEN_NEWEST;
  const openMap: Record<string, boolean> = {};
  keys.forEach((k, i) => {
    openMap[k] = openState[k] ?? defaultOpen(k, i);
  });
  // The current hit's block opens.
  let curBlock: string | null = null;
  if (isBlocks && curIdx >= 0) {
    const i = blockHits.search.findIndex((s) => curIdx >= s.base && curIdx < s.base + s.hits.length);
    if (i >= 0) curBlock = keys[i];
  }
  // Only when the current hit changes, so Collapse all still collapses it.
  const openedFor = useRef("");
  const curOpen = curBlock ? openMap[curBlock] : true;
  useEffect(() => {
    const want = curBlock ? `${activeQ}:${curIdx}` : "";
    if (want === openedFor.current) return;
    openedFor.current = want;
    if (curBlock && !curOpen) setOpenState((o) => ({ ...o, [curBlock]: true }));
  }, [curBlock, activeQ, curIdx, curOpen]);
  const setOpen = useCallback((k: string, v: boolean) => setOpenState((o) => (o[k] === v ? o : { ...o, [k]: v })), []);
  const setAll = (v: boolean) => setOpenState(Object.fromEntries(keys.map((k) => [k, v])));

  // --- follow ------------------------------------------------------------
  const bodyRef = useRef<HTMLDivElement>(null);
  const followable = live && STREAMED.has(kind);
  const tail = useFollowTail(bodyRef, hasHit ? "hit" : displayLen, followable, `${reqId}:${kind}`);
  const { stop: stopFollow, noteChars, jump } = tail;
  useEffect(() => {
    if (hasHit) stopFollow();
  }, [hasHit, activeQ, stopFollow]);
  const prevLen = useRef(displayLen);
  useLayoutEffect(() => {
    const added = displayLen - prevLen.current;
    prevLen.current = displayLen;
    if (added > 0) noteChars(added);
  }, [displayLen, noteChars]);
  const jumpLatest = () => {
    setQ("");
    applyQ("");
    jump();
  };
  const ctx = useMemo(() => ({ nudge: tail.nudge }), [tail.nudge]);

  // Scroll the current hit into view when it changes.
  const scrolledTo = useRef<string>("");
  useLayoutEffect(() => {
    const box = bodyRef.current;
    if (!box || curIdx < 0 || !activeQ) {
      scrolledTo.current = "";
      return;
    }
    const want = `${activeQ}:${curIdx}`;
    if (scrolledTo.current === want) return;
    const el = box.querySelector<HTMLElement>('mark[data-current="true"]');
    if (!el) return;
    scrolledTo.current = want;
    const top = box.scrollTop + el.getBoundingClientRect().top - box.getBoundingClientRect().top - box.clientHeight / 3;
    box.scrollTop = Math.max(0, top);
  });

  // --- wrap / copy -------------------------------------------------------
  const [wrap, setWrap] = useState(true);
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (copyTimer.current) clearTimeout(copyTimer.current);
  }, []);
  const onCopy = async () => {
    let s: string;
    if (kind === "tool_calls") s = copyText(kind, { calls });
    else if (kind === "tool_results")
      s = copyText(kind, {
        messages: results.map((b) => ({ i: b.msg ?? -1, role: "tool", chars: b.chars, text: b.text, elided: b.elided, tool_name: b.title, tool_call_id: b.callId ?? undefined })),
      });
    else if (kind === "prompt")
      s = copyText(kind, {
        text: results.map((b) => `${resultHead({ i: b.msg ?? -1, role: b.title, chars: b.chars, text: "", elided: false, tool_call_id: b.callId ?? undefined }).replace(/ · msg -1$/, "")}\n${b.text}`).join("\n\n"),
      });
    else s = copyText(kind, { text: display.text });
    try {
      await copyToClipboard(s);
      setCopied(true);
      if (copyTimer.current) clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      /* the button stays "Copy" */
    }
  };

  // --- dialog: focus, Esc, trap ------------------------------------------
  const dialogRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  // Phone: the pop-up owns a history entry; the back gesture closes it.
  useBackLayer(phone, () => onCloseRef.current());
  const btn = phone ? cn(BTN, BTN_PHONE) : BTN;
  const titleId = `lens-popup-${reqId}-${kind}`;
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        // Window capture runs first; stopImmediatePropagation also keeps any
        // later window-capture listener (and everything after) from seeing it.
        e.preventDefault();
        e.stopImmediatePropagation();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const root = dialogRef.current;
      if (!root) return;
      const els = Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => !el.closest("[inert]"));
      if (!els.length) {
        e.preventDefault();
        root.focus();
        return;
      }
      const first = els[0];
      const last = els[els.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && (active === first || !root.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !root.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey, true);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const appRoot = document.getElementById("app-root");
    const wasInert = appRoot?.hasAttribute("inert") ?? false;
    appRoot?.setAttribute("inert", "");
    (searchRef.current ?? dialogRef.current)?.focus();
    return () => {
      window.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = overflow;
      if (!wasInert) appRoot?.removeAttribute("inert");
      if (prev && typeof prev.focus === "function" && prev.isConnected) prev.focus();
    };
  }, []);

  // --- header facts --------------------------------------------------------
  const sizeChars = isText
    ? display.offset + display.text.length
    : kind === "tool_calls"
      ? calls.reduce((n, c) => n + c.args.length, 0) + (whole?.dropped ?? request.end?.tool_call_chars_dropped ?? 0)
      : results.reduce((n, b) => n + b.chars, 0);
  const sizeTok = Math.round(sizeChars / 4);
  const ctxParts: string[] = [];
  if (scope === "admin") ctxParts.push(request.tokenLabel || (request.tokenId ? `key ${request.tokenId.slice(0, 8)}` : "no key"));
  if (request.sessionId) ctxParts.push(`session ${request.sessionId.slice(0, 8)}`);
  if (request.turn !== null) ctxParts.push(`turn ${request.turn}`);
  const model = request.servedName || request.model;
  if (model) ctxParts.push(model);
  const kept = whole?.kept ?? request.kept;

  const marks: string[] = [];
  if (isText && whole && whole.dropped > 0) marks.push(`first ${fmt(whole.dropped)} chars dropped on the server`);
  if (kind === "tool_calls") {
    const dc = whole ? whole.dropped : (request.end?.tool_call_chars_dropped ?? 0);
    const dn = whole ? whole.droppedCalls : (request.end?.tool_calls_dropped ?? 0);
    if (dc > 0 && dn > 0) marks.push(`${fmt(dc)} chars and ${fmt(dn)} calls dropped on the server`);
    else if (dc > 0) marks.push(`${fmt(dc)} chars dropped on the server`);
    else if (dn > 0) marks.push(`${fmt(dn)} calls dropped on the server`);
  }

  const hasCopy = isText ? (sec?.text.length ?? 0) > 0 : kind === "tool_calls" ? request.toolCalls.length > 0 : request.prompt !== null;
  let status: React.ReactNode = null;
  if (load.state === "loading" && !whole) status = "Loading the whole section from server memory…";
  else if (load.state === "error") {
    const retry = load.status !== 404 && load.status !== 409 && load.status !== 401;
    status = (
      <>
        {errorText(load.status, hasCopy)}
        {retry && (
          <button type="button" className={cn(btn, "ml-2")} onClick={() => setAttempt((a) => a + 1)}>
            Retry
          </button>
        )}
      </>
    );
  }

  const count = activeQ ? (total ? `${fmt(curIdx + 1)} of ${total >= MAX_HITS ? `${fmt(MAX_HITS)}+` : fmt(total)}` : "no matches") : "";
  const clientTrim = isText && !whole && sec && sec.clipped > 0 ? sec.clipped : 0;
  const wrapCls = wrap ? "whitespace-pre-wrap [overflow-wrap:anywhere]" : "whitespace-pre [overflow-wrap:normal]";

  let content: React.ReactNode;
  if (isText) {
    content = (
      <div className={cn("text-[13.5px] leading-[1.6] text-chat-fg", kind === "thinking" && "italic text-chat-fg/90", wrapCls)}>
        {clientTrim > 0 && (
          <div className="mb-1 text-[11px] not-italic text-chat-dim">
            … {fmt(clientTrim)} earlier chars are not in this page&apos;s copy
          </div>
        )}
        {display.text === "" ? (
          <span className="text-[12px] italic text-chat-dim">{load.state === "loading" ? "" : "Empty"}</span>
        ) : activeQ ? (
          <SearchText text={display.text} hits={textHits.hits} ver={textHits.ver} cur={curIdx} scrollRoot={bodyRef} />
        ) : (
          <TailText text={display.text} offset={display.offset} className={wrapCls} />
        )}
      </div>
    );
  } else if (kind === "tool_calls") {
    content = (
      <ToolCallBlocks
        calls={calls}
        live={live}
        wrap={wrap}
        open={openMap}
        onOpenChange={setOpen}
        search={activeQ ? blockHits.search : undefined}
        cur={curIdx}
      />
    );
  } else {
    content = (
      <>
        {older && <p className="mb-2 mt-0 text-[12px] italic text-chat-dim">{older}</p>}
        <ResultBlocks
          blocks={results}
          wrap={wrap}
          open={openMap}
          onOpenChange={setOpen}
          search={activeQ ? blockHits.search : undefined}
          cur={curIdx}
          variant={kind === "prompt" ? "prompt" : "results"}
          empty={kind === "prompt" ? "No prompt captured." : "This prompt carries no tool results."}
        />
      </>
    );
  }

  if (typeof document === "undefined") return null;
  const showFollow = followable;
  const following = tail.following && !hasHit;

  return createPortal(
    <div className="fixed inset-0 z-[90]" data-testid="section-popup">
      <div data-testid="popup-backdrop" aria-hidden="true" className="absolute inset-0 bg-black/70" onClick={() => onCloseRef.current()} />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        data-phone={phone ? "true" : undefined}
        className={
          phone
            ? "absolute inset-0 flex min-w-0 flex-col overflow-hidden bg-vw-dock pb-[env(safe-area-inset-bottom)] outline-none"
            : "absolute inset-6 flex min-w-0 flex-col overflow-hidden rounded-[10px] border border-chat-rule bg-vw-dock shadow-[0_20px_60px_rgba(0,0,0,.6)] outline-none max-sm:inset-0 max-sm:rounded-none max-sm:border-0"
        }
      >
        <div data-testid="popup-header" className="flex min-w-0 items-center gap-2.5 overflow-hidden whitespace-nowrap border-b border-chat-rule px-3.5 py-2.5">
          <h2 id={titleId} className="m-0 flex-none text-[13px] font-bold uppercase tracking-[.06em] text-chat-fg">
            {name}
          </h2>
          <span
            data-testid="popup-size"
            title="The whole section"
            className={cn(PILL, sizeTok > SECTION_BIG_TOK ? "border-vw-amber-fg/60 text-vw-amber-fg" : "border-chat-rule text-chat-fg")}
          >
            {sizeChars === 0 ? (
              "empty"
            ) : (
              <>
                <LiveNumber value={sizeChars} reserve={live ? LIVE_CHARS_CH : 0} /> chars ·{" "}
                <LiveNumber reserve={live ? LIVE_TOK_CH + 2 : 0}>{`≈ ${fmt(sizeTok)}`}</LiveNumber> tok
              </>
            )}
          </span>
          {marks.length > 0 && (
            <span data-testid="popup-marks" className={cn(PILL, "min-w-0 overflow-hidden text-ellipsis border-vw-amber-fg/40 text-vw-amber-fg")}>
              {marks.join(" · ")}
            </span>
          )}
          {kept && (
            <span className={cn(PILL, "border-vw-info-fg/50 text-vw-info-fg")} title="A kept capture: stored on disk for this key">
              kept
            </span>
          )}
          {!phone && (
            <span className="min-w-0 overflow-hidden text-ellipsis text-[12px] text-chat-dim max-sm:hidden">{ctxParts.join(" · ")}</span>
          )}
          <span className="ml-auto flex flex-none items-center gap-2">
            {live && <span className="text-[11px] font-semibold uppercase tracking-[.06em] text-vw-live">Live</span>}
            <button type="button" aria-label="Close" title="Close (Esc)" onClick={() => onCloseRef.current()} className={btn}>
              ✕{!phone && <span className="max-sm:hidden"> Close <span className="text-chat-dim">Esc</span></span>}
            </button>
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 border-b border-chat-rule px-3.5 py-2">
          <div className={cn("flex min-w-0 items-center gap-2", phone ? "w-full flex-auto" : "flex-[0_1_auto] max-sm:w-full max-sm:flex-auto")}>
            <input
              ref={searchRef}
              type="search"
              aria-label="Search this section"
              placeholder="Search this section"
              value={q}
              onChange={(e) => onQ(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  step(e.shiftKey ? -1 : 1);
                }
              }}
              className={cn(
                "min-w-0 rounded-md border border-chat-rule bg-chat-surface px-2 text-chat-fg placeholder:text-chat-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-chat-accent",
                // 16 px on a phone: smaller text makes iOS zoom the page on focus.
                phone ? "h-11 flex-1 text-[16px] placeholder:text-[13px]" : "w-[280px] py-[3px] text-[12px] max-sm:w-auto max-sm:flex-1",
              )}
            />
            <span data-testid="popup-count" aria-live="polite" className={cn("text-[12px] tabular-nums text-chat-muted", phone ? "min-w-[52px] flex-none" : "min-w-[70px]")}>
              {count}
            </span>
            <button type="button" className={btn} aria-label="Previous match (Shift+Enter)" title="Previous match (Shift+Enter)" disabled={!total} onClick={() => step(-1)}>
              ↑
            </button>
            <button type="button" className={btn} aria-label="Next match (Enter)" title="Next match (Enter)" disabled={!total} onClick={() => step(1)}>
              ↓
            </button>
          </div>
          {/* A phone keeps this second row to one line: short labels, and a
              sideways scroll if a long "Jump to latest" still does not fit. */}
          <div
            data-testid="popup-tools"
            className={cn(
              "flex items-center gap-2",
              phone ? "ml-0 w-full flex-nowrap overflow-x-auto [scrollbar-width:none]" : "ml-auto flex-wrap max-sm:ml-0",
            )}
          >
            {(kind === "tool_calls" || kind === "tool_results") && (
              <>
                <button type="button" className={btn} aria-label="Expand all" onClick={() => setAll(true)}>
                  {phone ? "Expand" : "Expand all"}
                </button>
                <button type="button" className={btn} aria-label="Collapse all" onClick={() => setAll(false)}>
                  {phone ? "Collapse" : "Collapse all"}
                </button>
              </>
            )}
            <button type="button" aria-pressed={wrap} className={cn(btn, wrap && BTN_ON)} onClick={() => setWrap((w) => !w)} title="Wrap long lines (off: scroll sideways)">
              Wrap
            </button>
            <button
              type="button"
              className={btn}
              onClick={onCopy}
              disabled={load.state !== "ok"}
              title={
                load.state === "loading"
                  ? "Available once the whole section finishes loading"
                  : load.state === "error"
                    ? "The whole section is not loaded, so Copy is off: a copy of this page's part could be taken for the whole"
                    : kind === "tool_calls"
                      ? "Copy as a JSON array of {name, arguments}"
                      : "Copy the whole section as plain text"
              }
            >
              {copied ? "Copied" : "Copy"}
            </button>
            {showFollow &&
              (following ? (
                <button type="button" aria-pressed="true" className={cn(btn, BTN_ON, "text-vw-ok-fg")} onClick={stopFollow} title="New text keeps the view at the end; click or scroll up to stop">
                  Following
                </button>
              ) : (
                <button type="button" className={cn(btn, "border-chat-accent bg-chat-accent font-semibold text-chat-on-accent")}
                  aria-label={phone ? `Jump to latest${tail.newChars > 0 ? `, ${fmt(tail.newChars)} new chars` : ""}` : undefined}
                  onClick={jumpLatest}
                >
                  {phone ? "↓ Latest" : "↓ Jump to latest"}
                  {tail.newChars > 0 && (
                    <>
                      {" · "}
                      <LiveNumber value={tail.newChars} reserve={LIVE_TOK_CH} />
                      {!phone && " new chars"}
                    </>
                  )}
                </button>
              ))}
          </div>
        </div>

        {status !== null && (
          <div role="status" className="border-b border-chat-rule px-3.5 py-1.5 text-[12px] text-vw-amber-fg">
            {status}
          </div>
        )}

        <FollowTailContext.Provider value={ctx}>
          <div
            ref={bodyRef}
            data-testid="popup-body"
            data-scroll-box=""
            data-wrap={String(wrap)}
            data-following={String(following)}
            tabIndex={0}
            aria-label={`${name}, whole section`}
            className={cn(
              "relative min-h-0 flex-1 overflow-auto overscroll-contain px-4 py-3 [overflow-anchor:none] [scrollbar-width:thin] focus-visible:outline focus-visible:outline-1 focus-visible:outline-chat-accent max-sm:px-3",
              phone && "px-3",
            )}
          >
            {content}
          </div>
        </FollowTailContext.Provider>

        <div className="flex gap-3.5 overflow-hidden whitespace-nowrap border-t border-chat-rule px-3.5 py-1.5 text-[11px] text-chat-dim">
          <span className="min-w-0 overflow-hidden text-ellipsis">
            {live && STREAMED.has(kind) ? "Still streaming: new text appends at the end; scrolling up stops following." : "Final content."}
          </span>
          <span className={cn("min-w-0 overflow-hidden text-ellipsis max-sm:hidden", phone && "hidden")}>
            {kept ? "Kept capture: stored on disk for this key." : "Whole section, fetched from server memory."}
          </span>
          <span className={cn("ml-auto max-sm:hidden", phone && "hidden")}>Esc closes</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
