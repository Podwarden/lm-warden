"use client";

// God-mode live viewer — real-time prompts + model output flowing through the
// LLM Warden proxy for ONE key (and optionally its earlier keys). Admin-only
// (the SSE endpoint gates on require_jwt). Its only consumer is the token
// page's dock (components/tokens/detail/godmode-dock.tsx, spec 2026-09-18 D7).
//
// <Virtuoso> + the shared `useStickyBottom` hook give live/explore scroll and
// "Jump to latest", exactly as models/log-stream.tsx. The stream is scoped by
// the SERVER (`?token_ids=`, spec §3.5): every event carries token_id, so the
// filter is exact and never splits a request from its deltas. That is why the
// old client-side model filter is gone.
//
// Rendering: events are grouped by `req_id` into request blocks. Each block is
// one Virtuoso row — a header (token label · model · client · finish_reason or
// "streaming") with the completion streaming in beneath it. Reasoning renders
// dimmed/italic. Every request gets a stable per-session colour (req-color.ts).

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";

import { useStickyBottom } from "@/components/shared/use-sticky-bottom";
import { btn, FONT_SANS, GUTTER_X } from "@/components/tokens/detail/styles";
import { useEventSource, MAX_RECONNECT, type SseState, type SseStatus } from "@/lib/sse";
import { cn } from "@/lib/utils";
import { reqColor } from "./req-color";
import { MediaStrip } from "./godmode-media";

const GODMODE_STREAM_PATH = "/api/admin/godmode/stream";

// Bounded in-memory FIFO ring, mirroring log-stream's MAX_LINES. A long-open
// god-mode tab can't bloat the heap: past this many events the oldest are
// evicted and surfaced via the "… N older events elided" banner. (The backend
// hub has its own ring; this is the client-side cap independent of it.)
export const MAX_EVENTS = 4000;

// ---------------------------------------------------------------------------
// Event shape (matches app/proxy/godmode.py — see spec §3)
// ---------------------------------------------------------------------------

export interface RequestStartEvent {
  seq: number;
  type: "request_start";
  req_id: string;
  ts: number;
  token_label: string;
  token_id: string;
  model: string;
  served_name: string;
  client_ip: string;
  stream: boolean;
  prompt: string;
  /** Backend marked the captured prompt's middle as elided (head+tail window).
   *  Optional — older backends omit it; the viewer also detects the elision
   *  marker in the text itself, so it must work whether or not this is set. */
  prompt_elided?: boolean;
  /** Captured image_url parts (spec 2026-08-03) — absent on text-only
   *  requests and on events from older backends. */
  media?: MediaEntry[];
}

export interface MediaEntry {
  kind: "image";
  /** Present for store-backed items — fetch /api/admin/godmode/media/{id}. */
  media_id?: string;
  mime?: string;
  /** base64 length of the stored payload (decoded bytes ≈ chars * 3/4). */
  chars?: number;
  /** Present for remote images — hotlinked directly. */
  url?: string;
  dropped?: "too_large" | "count";
  count?: number;
}

export interface DeltaEvent {
  seq: number;
  type: "delta";
  req_id: string;
  ts: number;
  channel: "content" | "reasoning";
  text: string;
}

export interface RequestEndEvent {
  seq: number;
  type: "request_end";
  req_id: string;
  ts: number;
  finish_reason: string | null;
  prompt_tokens: number;
  completion_tokens: number;
}

export type GodModeEvent = RequestStartEvent | DeltaEvent | RequestEndEvent;

// ---------------------------------------------------------------------------
// Ring state
// ---------------------------------------------------------------------------

export interface GmState {
  events: GodModeEvent[];
  /** Total events dropped by the FIFO so far — surfaced as a banner row. */
  elided: number;
  /** Stable per-session index for each req_id, assigned on first sight and
   *  never reclaimed, so a request's golden-angle color survives ring
   *  eviction. */
  reqIndex: Record<string, number>;
  /** Next session index to hand out. */
  nextIndex: number;
}

const INITIAL_STATE: GmState = { events: [], elided: 0, reqIndex: {}, nextIndex: 0 };

export function appendEvent(prev: GmState, ev: GodModeEvent): GmState {
  let reqIndex = prev.reqIndex;
  let nextIndex = prev.nextIndex;
  if (reqIndex[ev.req_id] === undefined) {
    reqIndex = { ...reqIndex, [ev.req_id]: nextIndex };
    nextIndex += 1;
  }

  if (prev.events.length >= MAX_EVENTS) {
    return {
      events: [...prev.events.slice(prev.events.length - MAX_EVENTS + 1), ev],
      elided: prev.elided + 1,
      reqIndex,
      nextIndex,
    };
  }
  return { events: [...prev.events, ev], elided: prev.elided, reqIndex, nextIndex };
}

/** The dock's Clear button: drop every request that has ended. Running ones
 *  stay (their deltas are still arriving). `reqIndex` is kept so colours stay
 *  stable. Returns `prev` itself when there is nothing to drop. */
export function clearFinished(prev: GmState): GmState {
  const ended = new Set(
    prev.events.filter((e) => e.type === "request_end").map((e) => e.req_id),
  );
  if (ended.size === 0) return prev;
  return { ...prev, events: prev.events.filter((e) => !ended.has(e.req_id)) };
}

// ---------------------------------------------------------------------------
// Grouping
// ---------------------------------------------------------------------------

export interface RequestBlockData {
  reqId: string;
  start?: RequestStartEvent;
  deltas: DeltaEvent[];
  end?: RequestEndEvent;
}

/** Fold the flat event ring into request blocks, preserving first-seen order
 *  (which is start-order in practice, since request_start is a request's
 *  first event). O(n) over the window. */
export function groupBlocks(events: GodModeEvent[]): RequestBlockData[] {
  const byId = new Map<string, RequestBlockData>();
  const order: string[] = [];
  for (const ev of events) {
    let block = byId.get(ev.req_id);
    if (!block) {
      block = { reqId: ev.req_id, deltas: [] };
      byId.set(ev.req_id, block);
      order.push(ev.req_id);
    }
    if (ev.type === "request_start") block.start = ev;
    else if (ev.type === "delta") block.deltas.push(ev);
    else if (ev.type === "request_end") block.end = ev;
  }
  return order.map((id) => byId.get(id)!);
}

// ---------------------------------------------------------------------------
// Repeated-system-prompt collapse
// ---------------------------------------------------------------------------
//
// A god-mode viewer watching an agent loop sees the SAME lengthy system prompt
// re-sent on every request. To keep the genuinely-new turn readable, we diff a
// request's prompt against the PREVIOUS request from the same token identity
// and collapse the shared leading portion behind a toggle, leaving only the
// divergent remainder expanded. The diff is line-aware (split on lines, take
// the longest matching leading run) so we never cut mid-line.

/** Backend's elision marker, e.g. `…[1234 chars elided]…`. When a capture was
 *  windowed (head+tail), we must NOT diff across the gap — restrict the
 *  comparable region to the head that precedes this marker. */
const ELISION_MARKER_RE = /…\[\d+ chars elided\]…/;

function headBeforeElision(s: string): string {
  const m = s.match(ELISION_MARKER_RE);
  return m && m.index !== undefined ? s.slice(0, m.index) : s;
}

/** Longest common leading run of whole lines between `a` and `b`, returned as
 *  the exact leading substring of `a` (so `a.slice(prefix.length)` is the
 *  divergent remainder). "" when the first line already differs. */
export function commonLinePrefix(a: string, b: string): string {
  if (!a || !b) return "";
  const al = a.split("\n");
  const bl = b.split("\n");
  const n = Math.min(al.length, bl.length);
  let matched = 0;
  while (matched < n && al[matched] === bl[matched]) matched += 1;
  if (matched === 0) return "";
  let prefix = al.slice(0, matched).join("\n");
  // Re-attach the newline that separates the matched run from `a`'s remainder,
  // but only when there IS more content after it — so prefix+remainder === a.
  if (matched < al.length) prefix += "\n";
  return prefix;
}

export interface PromptCollapse {
  /** Shared leading portion, hidden by default behind the toggle. */
  collapsed: string;
  /** Divergent NEW content, always shown. */
  remainder: string;
}

/** Decide whether a prompt's shared head with the previous same-token prompt is
 *  substantial enough to collapse. Guarded so a short or mostly-new prompt is
 *  left fully visible. Returns null when no collapse should happen. */
export function computePromptCollapse(
  current: string,
  previous: string | undefined,
  opts?: { minChars?: number; minRatio?: number },
): PromptCollapse | null {
  if (!current || !previous) return null;
  const minChars = opts?.minChars ?? 400;
  const minRatio = opts?.minRatio ?? 0.4;
  // Never diff past an elision gap — compare only the heads. A prefix of the
  // head is still a prefix of the full current prompt, so the marker + tail
  // fall into `remainder` and stay visible (never claimed as "unchanged").
  const prefix = commonLinePrefix(headBeforeElision(current), headBeforeElision(previous));
  if (prefix.length < minChars) return null;
  const shorter = Math.min(current.length, previous.length);
  if (shorter === 0 || prefix.length < shorter * minRatio) return null;
  return { collapsed: prefix, remainder: current.slice(prefix.length) };
}

// ---------------------------------------------------------------------------
// Status message
// ---------------------------------------------------------------------------

function renderStatusMessage(s: SseState): { text: string; tone: "info" | "warn" | "error" } | null {
  switch (s.status) {
    case "connected":
      return null;
    case "connecting":
      return { text: "Connecting to god-mode stream…", tone: "info" };
    case "reconnecting":
      return {
        text: `Connection lost — retrying (${s.attempts}/${MAX_RECONNECT})…`,
        tone: "warn",
      };
    case "terminal-error": {
      const code = s.errorCode;
      let text: string;
      if (code === 401 || code === 403) {
        text = "Stream unavailable — your session expired. Please re-login.";
      } else if (code !== null) {
        text = `Stream unavailable (HTTP ${code}). Please refresh.`;
      } else {
        text = `Stream unavailable after ${MAX_RECONNECT} retries. Please refresh.`;
      }
      return { text, tone: "error" };
    }
  }
}

// A terminal-error whose ticket mint returned 404/409 means the backend has
// god mode switched off (spec §5). Show a purpose-built placeholder rather
// than a generic "stream unavailable" so the operator knows it's a config
// flag, not a fault.
function isDisabledResponse(s: SseState): boolean {
  return s.status === "terminal-error" && (s.errorCode === 404 || s.errorCode === 409);
}

// The mockup's `.dock-empty` line: dim DM Sans, 18px top/bottom, inside the
// stream's gutter. Error text keeps the danger colour.
function EmptyLine({ tone = "info", role = "status", children }: {
  tone?: "info" | "warn" | "error"; role?: "status" | "alert"; children: React.ReactNode;
}) {
  return (
    <div
      role={role}
      className={cn(
        GUTTER_X,
        FONT_SANS,
        "py-[18px] text-[12px]",
        tone === "error" ? "text-vw-danger-fg" : tone === "warn" ? "text-chat-accent" : "text-chat-dim",
      )}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export interface GodModeCounts {
  requests: number;
  running: number;
}

export interface GodModeViewerHandle {
  clear: () => void;
  setFollowing: (on: boolean) => void;
}

export interface GodModeViewerProps {
  tokenIds: string[];
  className?: string;
  heightPx?: number;
  onCounts?: (c: GodModeCounts) => void;
  onFollowingChange?: (following: boolean) => void;
  onStatusChange?: (status: SseStatus) => void;
  ref?: React.Ref<GodModeViewerHandle>;
}

/** Virtuoso's atBottomThreshold, also the follow-on-growth tolerance. */
const AT_BOTTOM_PX = 64;

export function GodModeViewer({
  tokenIds,
  className,
  heightPx = 560,
  onCounts,
  onFollowingChange,
  onStatusChange,
  ref,
}: GodModeViewerProps) {
  const [state, setState] = useState<GmState>(INITIAL_STATE);

  const onMessage = useCallback((ev: GodModeEvent) => {
    // Defensive: ignore anything without the fields we key on. A malformed
    // frame shouldn't tear down the view.
    if (!ev || typeof ev.seq !== "number" || typeof ev.req_id !== "string") return;
    setState((prev) => appendEvent(prev, ev));
  }, []);

  // No tokens, no stream: an empty `token_ids=` is a 422, and the viewer
  // never streams unfiltered (#251).
  const sse = useEventSource<GodModeEvent>(GODMODE_STREAM_PATH, {
    onMessage,
    enabled: tokenIds.length > 0,
    query: { token_ids: tokenIds.join(",") },
  });

  const sticky = useStickyBottom("stick");
  // useStickyBottom returns a fresh object every render, but its callbacks
  // (jumpToLatest, onAtBottomStateChange) are individually stable (useCallback
  // with empty deps). Destructure them so effects/callbacks below can depend
  // on the stable pieces instead of the ever-changing `sticky` object —
  // otherwise every delta-driven re-render (streaming can be <50ms apart)
  // would tear down and reschedule the landing timer below, and it would
  // never get a quiet window to fire.
  const { jumpToLatest: stickJump, followOutput, onAtBottomStateChange, mode } = sticky;
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  // The dock's "Following" button can stop auto-scroll while the list is still
  // at the bottom — something the stick/free latch alone cannot express,
  // because it only flips on scroll.
  const [followPaused, setFollowPaused] = useState(false);
  const following = !followPaused && mode === "stick";
  // Virtuoso's followOutput reacts to new blocks; the last block growing delta
  // by delta only re-scrolls once it has pushed past atBottomThreshold, so the
  // tail lagged 1–4 lines behind. The mockup scrolls to the end on every delta
  // while following. Only when the view sat at the bottom before this growth,
  // so an operator scrolling up to read is never yanked back.
  const followingRef = useRef(following);
  followingRef.current = following;
  const scrollerEl = useRef<HTMLElement | null>(null);
  // Distance from the bottom as of the scroller's last scroll event, i.e.
  // measured against the scrollHeight the reader actually saw. Growth never
  // fires a scroll event, so this stays the "gap before this growth" whether
  // or not the DOM already includes the new block (recomputing it from the
  // current scrollHeight minus the growth underestimates it when it doesn't).
  const gapAtLastScroll = useRef(0);
  const listH = useRef(0);
  const onListHeight = useCallback((h: number) => {
    const grew = h - listH.current;
    listH.current = h;
    if (!followingRef.current || !scrollerEl.current || grew <= 0) return;
    // "At the bottom" uses the same tolerance as atBottomThreshold (the
    // landing's align:"end" leaves the stream's bottom padding below the fold).
    if (gapAtLastScroll.current <= AT_BOTTOM_PX) virtuosoRef.current?.scrollTo({ top: Number.MAX_SAFE_INTEGER });
  }, []);
  const onScrollerScroll = useCallback((e: Event) => {
    const el = e.currentTarget as HTMLElement;
    gapAtLastScroll.current = el.scrollHeight - el.scrollTop - el.clientHeight;
  }, []);
  const onScroller = useCallback((el: HTMLElement | Window | null) => {
    scrollerEl.current?.removeEventListener("scroll", onScrollerScroll);
    scrollerEl.current = el && !(el instanceof Window) ? el : null;
    scrollerEl.current?.addEventListener("scroll", onScrollerScroll, { passive: true });
  }, [onScrollerScroll]);

  const blocks = useMemo(() => groupBlocks(state.events), [state.events]);
  const running = useMemo(() => blocks.filter((b) => b.start && !b.end).length, [blocks]);

  // Callbacks go through refs so a parent passing inline arrows can't loop.
  const cbs = useRef({ onCounts, onFollowingChange, onStatusChange });
  cbs.current = { onCounts, onFollowingChange, onStatusChange };
  useEffect(() => {
    cbs.current.onCounts?.({ requests: blocks.length, running });
  }, [blocks.length, running]);
  useEffect(() => {
    cbs.current.onFollowingChange?.(following);
  }, [following]);
  useEffect(() => {
    cbs.current.onStatusChange?.(sse.status);
  }, [sse.status]);

  const jumpToLatest = useCallback(() => {
    setFollowPaused(false);
    if (blocks.length > 0) {
      virtuosoRef.current?.scrollToIndex({ index: blocks.length - 1, behavior: "smooth" });
    }
    stickJump();
  }, [blocks.length, stickJump]);

  // The replay arrives as one burst right after connect and Virtuoso leaves the
  // view at the top (and the burst latches the sticky state to "free"). The
  // mockup opens scrolled to the newest request, following. Once the burst has
  // been quiet for 50 ms, land on the last block and re-stick — once per mount.
  const landedRef = useRef(false);
  useEffect(() => {
    if (landedRef.current || blocks.length === 0) return;
    const t = setTimeout(() => {
      landedRef.current = true;
      virtuosoRef.current?.scrollToIndex({ index: blocks.length - 1, align: "end" });
      stickJump();
    }, 50);
    return () => clearTimeout(t);
  }, [blocks.length, stickJump]);

  useImperativeHandle(
    ref,
    () => ({
      clear: () => setState((prev) => clearFinished(prev)),
      setFollowing: (on: boolean) => {
        if (on) jumpToLatest();
        else setFollowPaused(true);
      },
    }),
    [jumpToLatest],
  );

  const collapseByReq = useMemo(() => {
    const out: Record<string, PromptCollapse | null> = {};
    const lastPromptByToken: Record<string, string> = {};
    for (const block of blocks) {
      const start = block.start;
      if (!start || !start.prompt) {
        if (start) out[block.reqId] = null;
        continue;
      }
      const identity = start.token_label || start.token_id || "";
      const prev = identity ? lastPromptByToken[identity] : undefined;
      out[block.reqId] = computePromptCollapse(start.prompt, prev);
      if (identity) lastPromptByToken[identity] = start.prompt;
    }
    return out;
  }, [blocks]);

  if (isDisabledResponse(sse)) {
    return (
      <div className={className}>
        <EmptyLine>
          God mode is disabled (<code className="!font-mono text-chat-muted">VW_GODMODE_ENABLED</code>).
        </EmptyLine>
      </div>
    );
  }

  const statusBar = renderStatusMessage(sse);

  if (sse.status === "connected" && state.events.length === 0) {
    return (
      <div className={className}>
        <EmptyLine>Waiting for requests… (no traffic through the proxy yet)</EmptyLine>
      </div>
    );
  }

  if (state.events.length === 0 && statusBar !== null) {
    return (
      <div className={className}>
        <EmptyLine tone={statusBar.tone} role={statusBar.tone === "error" ? "alert" : "status"}>
          {statusBar.text}
        </EmptyLine>
      </div>
    );
  }

  return (
    <div className={cn("relative", className)}>
      {statusBar !== null && sse.status !== "connected" && (
        <EmptyLine tone={statusBar.tone} role={statusBar.tone === "error" ? "alert" : "status"}>
          {statusBar.text}
        </EmptyLine>
      )}
      <Virtuoso
        ref={virtuosoRef}
        components={{ List: GodModeList, Header: StreamTop, Footer: StreamBottom }}
        style={{ height: heightPx }}
        className="!font-mono text-[12px] leading-[1.6]"
        data={blocks}
        followOutput={followPaused ? false : followOutput}
        // Match log-stream's generous tolerance so a mid-burst frame where a
        // freshly-appended block sits below the fold doesn't flap the
        // "Jump to latest" button (see use-sticky-bottom rationale).
        atBottomThreshold={AT_BOTTOM_PX}
        atBottomStateChange={onAtBottomStateChange}
        totalListHeightChanged={onListHeight}
        scrollerRef={onScroller}
        computeItemKey={(_idx, block) => block.reqId}
        itemContent={(_idx, block) => (
          <RequestBlock
            block={block}
            colorIndex={state.reqIndex[block.reqId]}
            collapse={collapseByReq[block.reqId]}
          />
        )}
      />
      {state.elided > 0 && (
        <div role="status" className={cn(GUTTER_X, FONT_SANS, "absolute left-0 right-0 top-0 z-10 bg-vw-dock text-[11px] text-chat-dim")}>
          … {state.elided} older event{state.elided === 1 ? "" : "s"} elided
        </div>
      )}
      {mode === "free" && blocks.length > 0 && (
        <button
          type="button"
          onClick={jumpToLatest}
          className={btn(
            "primary",
            "absolute bottom-3.5 right-[max(24px,calc((100vw_-_1250px)/2_+_8px))]",
          )}
        >
          Jump to latest
        </button>
      )}
    </div>
  );
}

// .stream padding-top 8px / padding-bottom 14px. Virtuoso owns the list's
// vertical padding for virtualisation, so the spacing lives in Header/Footer.
function StreamTop() {
  return <div style={{ height: 8 }} />;
}
function StreamBottom() {
  return <div style={{ height: 14 }} />;
}

// ---------------------------------------------------------------------------
// Request block row
// ---------------------------------------------------------------------------

interface RequestBlockProps {
  block: RequestBlockData;
  colorIndex?: number;
  collapse?: PromptCollapse | null;
}

function Sep() {
  return <span aria-hidden className="text-vw-rule-soft">·</span>;
}

function RequestBlock({ block, colorIndex, collapse }: RequestBlockProps) {
  const color = reqColor(block.reqId, colorIndex);
  const start = block.start;
  const label = start?.token_label || start?.token_id || block.reqId.slice(0, 8);

  return (
    <div
      data-testid="godmode-block"
      data-req-id={block.reqId}
      className="mb-2 rounded border-l-4 bg-chat-surface/55"
      style={{ borderLeftColor: color.accent }}
    >
      <div
        className={cn(FONT_SANS, "flex flex-wrap items-center gap-x-2 gap-y-1 rounded-tr px-3 py-[5px] text-[12px]")}
        style={{ backgroundColor: color.headerBg }}
      >
        <span className="font-semibold" style={{ color: color.text }}>
          {label}
        </span>
        {start?.model && (
          <>
            <Sep />
            <span className="!font-mono text-chat-muted">{start.model}</span>
          </>
        )}
        {start?.client_ip && (
          <>
            <Sep />
            <span className="!font-mono text-chat-dim">{start.client_ip}</span>
          </>
        )}
        <Sep />
        {block.end ? (
          <span
            data-testid="godmode-finish"
            className="rounded-[3px] bg-chat-surface-2 px-1.5 py-px !font-mono text-[10px] uppercase text-chat-muted"
          >
            {block.end.finish_reason ?? "ended"}
          </span>
        ) : (
          <span data-testid="godmode-running" className="text-[11px] text-vw-ok-fg">
            streaming
          </span>
        )}
      </div>

      {start?.media && start.media.length > 0 && <MediaStrip media={start.media} />}

      {start?.prompt && <PromptView prompt={start.prompt} collapse={collapse} />}

      <div className="whitespace-pre-wrap px-3 py-1.5 text-chat-fg">
        {block.deltas.map((d) => (
          <span
            key={d.seq}
            data-channel={d.channel}
            className={cn("!font-mono", d.channel === "reasoning" && "italic text-chat-dim")}
          >
            {d.text}
          </span>
        ))}
        {!block.end && <span aria-hidden data-testid="godmode-caret" className="vw-caret" />}
      </div>
    </div>
  );
}

function Tri() {
  return <span aria-hidden className="mr-1 select-none !font-mono text-vw-rule-soft">▸</span>;
}

// Prompt row. With a collapse, the mockup's `.sys` pill IS the toggle: it keeps
// the existing expand behaviour and the existing test id.
function PromptView({ prompt, collapse }: { prompt: string; collapse?: PromptCollapse | null }) {
  const [expanded, setExpanded] = useState(false);
  const rowClass = "whitespace-pre-wrap border-b border-chat-rule/60 px-3 py-[5px] text-chat-dim";

  if (!collapse) {
    return (
      <div className={rowClass}>
        <Tri />
        <span className="!font-mono text-chat-muted">{prompt}</span>
      </div>
    );
  }

  return (
    <div className={rowClass}>
      <button
        type="button"
        data-testid="godmode-prompt-toggle"
        aria-expanded={expanded}
        onClick={() => setExpanded((v) => !v)}
        className={cn(
          FONT_SANS,
          "mr-2 inline-block rounded-[3px] bg-chat-surface-2 px-1.5 text-[10.5px] text-chat-muted hover:text-chat-fg",
        )}
      >
        system prompt · {collapse.collapsed.length.toLocaleString()} chars · unchanged from previous
      </button>
      {expanded && (
        <div data-testid="godmode-prompt-collapsed" className="text-chat-dim">
          {collapse.collapsed}
        </div>
      )}
      <span data-testid="godmode-prompt-remainder">
        <Tri />
        <span className="!font-mono text-chat-muted">{collapse.remainder}</span>
      </span>
    </div>
  );
}

// Virtuoso List override — role="log" live region; carries the stream gutter.
const GodModeList = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { context?: unknown }
>(function GodModeListImpl(props, ref) {
  const { context: _context, className, ...rest } = props;
  return (
    <div
      ref={ref}
      role="log"
      aria-live="polite"
      aria-label="God mode live stream"
      className={cn(GUTTER_X, className)}
      {...rest}
    />
  );
});
