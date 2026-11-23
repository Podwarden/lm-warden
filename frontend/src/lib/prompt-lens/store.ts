// Prompt Lens request store: a pure reducer folding the Prompt Lens stream's
// events (./events.ts) into one view per request, with bounded memory.
//
// Bounds: at most CLIENT_MAX_REQUESTS requests (the oldest finished one goes
// first; a live one only when none has finished) and CLIENT_SECTION_CHARS
// chars per section (the oldest text goes first). What was dropped is
// counted (`Section.clipped`, `toolCallsClipped`, `promptClipped`,
// `DockState.trimmed`) so the UI can say "trimmed".
//
// Duplicates and replays. The admin stream's `seq` is the hub's, global and
// monotonic across connections: an event at or below a request's `lastSeq`
// was already applied. The key holder's `seq` is numbered per connection from
// 1 (prompt_lens_stream.py), so the stream hook dispatches `{type: "connected"}`
// on every (re)connect and the reducer opens a new connection epoch:
//   * a replayed request_start rebuilds the request from scratch (its key
//     window holds every later event of it);
//   * a RESTORED start (`restored: true`: the server's copy of the start of a
//     request still running, sent at connect because its own left the
//     replay) only merges metadata into a request the dock already has, on
//     either stream, and never resets text; a restored kvsig only fills a
//     missing map;
//   * otherwise, until the request catches up, a finished request takes no
//     events and a live one only those newer (by `ts`) than the last it took.
// Within one connection the seq rules again.
//
// The reducer is pure: it never mutates its input and returns new objects
// only for what changed.

import type {
  DeltaEvent,
  PromptLensEvent,
  KeptCapture,
  KeptSummary,
  KvsigEvent,
  MediaEntry,
  ProgressEvent,
  PromptMsg,
  PromptParts,
  RequestEndEvent,
  RequestStartEvent,
  RingEvent,
  SectionResponse,
} from './events';

export const CLIENT_MAX_REQUESTS = 300;
export const CLIENT_SECTION_CHARS = 200_000;
/** Text held across every request. Past it the oldest requests lose their
 *  text first (the rows stay, marked `stripped`, the chars counted as trimmed). */
export const CLIENT_TOTAL_CHARS = 32_000_000;
/** Evicted request ids remembered, so late or replayed events of one do not
 *  bring it back. */
export const GONE_MAX = 1000;

export interface Section {
  text: string;
  /** Every char received, the clipped ones included. */
  chars: number;
  /** Chars dropped from the head (client bound, or already dropped on the
   *  server when the section was fetched). */
  clipped: number;
  /** Deltas received. */
  frames: number;
  firstTs: number | null;
  lastTs: number | null;
}

export interface ToolCallView {
  index: number;
  callId: string | null;
  name: string;
  args: string;
  /** Only at request_end (ruling G-F4): the stream has no per-call end. */
  done: boolean;
  /** Arguments were cut (by the server's budget or the client's bound). */
  elided: boolean;
}

export type Frames = { reasoning: number; content: number; tool_call: number };

export interface RequestView {
  reqId: string;
  requestId: string | null;
  tokenId: string | null;
  tokenLabel?: string | null;
  clientIp?: string | null;
  model: string;
  servedName: string;
  stream: boolean;
  sessionId: string | null;
  turn: number | null;
  startedTs: number;
  endedTs: number | null;
  /** The server's raw phase (G-F6): `queued` at start, then the latest
   *  progress phase or delta channel (`thinking`, `answering`, `tool_call`). */
  phase: string;
  finish: string | null;
  prompt: PromptParts | null;
  promptFlat: string;
  promptElided: boolean;
  /** Chars of the flat prompt or the fetched prompt dropped by the client. */
  promptClipped: number;
  media: MediaEntry[];
  kvsig: KvsigEvent | null;
  thinking: Section;
  reply: Section;
  toolCalls: ToolCallView[];
  /** Tool-call argument chars the client bound dropped. */
  toolCallsClipped: number;
  /** tool_call deltas received. */
  toolCallFrames: number;
  /** `ts` of the first tool_call delta (the first token of a tool-call turn
   *  with no thinking). */
  toolCallFirstTs: number | null;
  end: RequestEndEvent | null;
  /** Per channel: the larger of the server's progress counts and the deltas
   *  received (approximate rates). */
  frames: Frames;
  /** completion_tokens of the latest progress event. */
  progressTokens: number | null;
  lastSeq: number;
  /** The newest `ts` of a kept (replayable) event applied. */
  lastTs: number;
  /** The connection epoch this request was last caught up in. */
  conn: number;
  /** request_start seen (false: a placeholder from a later event or a kept
   *  list row). */
  hasStart: boolean;
  /** Persisted by keep captures (shown with a "kept" mark). */
  kept: boolean;
  /** A kept list row's token counts (its summary), until the capture's
   *  request_end fills `end`. */
  keptTokens?: { prompt: number | null; completion: number | null };
  /** Live when the stream lagged: some of its events may be missing. */
  gap: boolean;
  /** Its text was dropped by CLIENT_TOTAL_CHARS. */
  stripped: boolean;
  /** Text chars it holds (sections, flat and structured prompt, tool args). */
  cost: number;
  /** The structured prompt's share of `cost`. */
  promptCost: number;
}

export interface DockState {
  byId: Record<string, RequestView>;
  /** Request ids by start time, oldest first. */
  order: string[];
  ring: RingEvent | null;
  /** The stream lagged at least once (events were dropped server side). */
  lagged: boolean;
  laggedCount: number;
  /** Requests evicted by CLIENT_MAX_REQUESTS. */
  trimmed: number;
  /** Connection epoch, bumped by `connected`. */
  conn: number;
  /** `global`: the admin stream's hub seq. `connection`: the key holder's. */
  seqScope: 'global' | 'connection';
  /** Recently evicted ids with the connection epoch they went in and whether
   *  they were live, oldest first, at most GONE_MAX. A live one's start
   *  replayed on a later connection brings it back; a finished one stays gone. */
  gone: { id: string; conn: number; live: boolean }[];
  /** Text chars held across every request (sum of `cost`). */
  chars: number;
  /** Requests whose text CLIENT_TOTAL_CHARS dropped. */
  stripped: number;
}

/** Dispatched by the stream hook on every (re)connect (client side only). */
export interface ConnectedAction {
  type: 'connected';
  seqScope: 'global' | 'connection';
}
/** The kept captures list (GET .../prompt-lens/kept). */
export interface KeptListAction {
  type: 'kept_list';
  kept: KeptSummary[];
}
/** One kept capture (GET .../prompt-lens/kept/{req_id}). */
export interface KeptCaptureAction {
  type: 'kept_capture';
  capture: KeptCapture;
}
/** One fetched section (GET .../requests/{req_id}/section/{kind}); a kept
 *  capture takes its prompt from section/prompt. */
export interface SectionFillAction {
  type: 'section_fill';
  req_id: string;
  section: SectionResponse;
}

export type DockAction =
  | PromptLensEvent
  | ConnectedAction
  | KeptListAction
  | KeptCaptureAction
  | SectionFillAction;

type ReqEvent = RequestStartEvent | DeltaEvent | KvsigEvent | ProgressEvent | RequestEndEvent;

// ---------------------------------------------------------------------------

const EMPTY_SECTION: Section = { text: '', chars: 0, clipped: 0, frames: 0, firstTs: null, lastTs: null };

export function emptyDockState(): DockState {
  return {
    byId: {},
    order: [],
    ring: null,
    lagged: false,
    laggedCount: 0,
    trimmed: 0,
    conn: 0,
    seqScope: 'global',
    gone: [],
    chars: 0,
    stripped: 0,
  };
}

function blank(reqId: string, ts: number, conn: number): RequestView {
  return {
    reqId,
    requestId: null,
    tokenId: null,
    model: '',
    servedName: '',
    stream: false,
    sessionId: null,
    turn: null,
    startedTs: ts,
    endedTs: null,
    phase: 'queued',
    finish: null,
    prompt: null,
    promptFlat: '',
    promptElided: false,
    promptClipped: 0,
    media: [],
    kvsig: null,
    thinking: EMPTY_SECTION,
    reply: EMPTY_SECTION,
    toolCalls: [],
    toolCallsClipped: 0,
    toolCallFrames: 0,
    toolCallFirstTs: null,
    end: null,
    frames: { reasoning: 0, content: 0, tool_call: 0 },
    progressTokens: null,
    lastSeq: 0,
    lastTs: 0,
    conn,
    hasStart: false,
    kept: false,
    gap: false,
    stripped: false,
    cost: 0,
    promptCost: 0,
  };
}

function tail(text: string, max: number): [string, number] {
  return text.length > max ? [text.slice(text.length - max), text.length - max] : [text, 0];
}

function withStart(r: RequestView, ev: RequestStartEvent): RequestView {
  const [flat, cut] = tail(typeof ev.prompt === 'string' ? ev.prompt : '', CLIENT_SECTION_CHARS);
  return {
    ...r,
    requestId: ev.request_id ?? null,
    tokenId: ev.token_id ?? null,
    tokenLabel: ev.token_label,
    clientIp: ev.client_ip,
    model: ev.model ?? '',
    servedName: ev.served_name ?? '',
    stream: !!ev.stream,
    sessionId: ev.session_id ?? null,
    turn: ev.turn ?? null,
    startedTs: ev.ts,
    prompt: ev.prompt_parts ?? r.prompt,
    promptFlat: flat,
    promptElided: !!ev.prompt_elided,
    promptClipped: cut,
    media: ev.media ?? [],
    hasStart: true,
  };
}

function appendText(s: Section, text: string, ts: number): Section {
  const [kept, cut] = tail(s.text + text, CLIENT_SECTION_CHARS);
  return {
    text: kept,
    chars: s.chars + text.length,
    clipped: s.clipped + cut,
    frames: s.frames + 1,
    firstTs: s.firstTs ?? ts,
    lastTs: ts,
  };
}

const CHANNEL_PHASE: Record<DeltaEvent['channel'], string> = {
  reasoning: 'thinking',
  content: 'answering',
  tool_call: 'tool_call',
};

function argsTotal(calls: ToolCallView[]): number {
  let n = 0;
  for (const c of calls) n += c.args.length;
  return n;
}

function withToolCall(r: RequestView, ev: DeltaEvent): RequestView {
  const index = typeof ev.index === 'number' ? ev.index : 0;
  const room = Math.max(0, CLIENT_SECTION_CHARS - argsTotal(r.toolCalls));
  const text = ev.text ?? '';
  const keep = text.slice(0, room);
  const cut = text.length - keep.length;
  const pos = r.toolCalls.findIndex((c) => c.index === index);
  const prev: ToolCallView =
    pos >= 0 ? r.toolCalls[pos] : { index, callId: null, name: '', args: '', done: false, elided: false };
  const call: ToolCallView = {
    ...prev,
    callId: ev.call_id ?? prev.callId,
    name: ev.name && !prev.name ? ev.name : prev.name,
    args: prev.args + keep,
    elided: prev.elided || !!ev.elided || cut > 0,
  };
  const calls = r.toolCalls.slice();
  if (pos >= 0) calls[pos] = call;
  else {
    calls.push(call);
    calls.sort((a, b) => a.index - b.index);
  }
  return {
    ...r,
    toolCalls: calls,
    toolCallsClipped: r.toolCallsClipped + cut,
    toolCallFrames: r.toolCallFrames + 1,
    toolCallFirstTs: r.toolCallFirstTs ?? ev.ts,
    frames: { ...r.frames, tool_call: Math.max(r.frames.tool_call, r.toolCallFrames + 1) },
  };
}

function withDelta(r: RequestView, ev: DeltaEvent): RequestView {
  let out: RequestView;
  if (ev.channel === 'tool_call') {
    out = withToolCall(r, ev);
  } else if (ev.channel === 'reasoning' || ev.channel === 'content') {
    const key = ev.channel === 'reasoning' ? 'thinking' : 'reply';
    const sec = appendText(r[key], ev.text ?? '', ev.ts);
    out = {
      ...r,
      [key]: sec,
      frames: { ...r.frames, [ev.channel]: Math.max(r.frames[ev.channel], sec.frames) },
    };
  } else {
    return r;
  }
  return r.endedTs === null ? { ...out, phase: CHANNEL_PHASE[ev.channel] } : out;
}

function withProgress(r: RequestView, ev: ProgressEvent): RequestView {
  const f = ev.frames ?? { reasoning: 0, content: 0, tool_call: 0 };
  return {
    ...r,
    phase: r.endedTs === null && typeof ev.phase === 'string' ? ev.phase : r.phase,
    frames: {
      reasoning: Math.max(r.frames.reasoning, f.reasoning ?? 0),
      content: Math.max(r.frames.content, f.content ?? 0),
      tool_call: Math.max(r.frames.tool_call, f.tool_call ?? 0),
    },
    progressTokens: ev.completion_tokens ?? r.progressTokens,
  };
}

function withEnd(r: RequestView, ev: RequestEndEvent): RequestView {
  return {
    ...r,
    end: ev,
    endedTs: ev.ts,
    finish: ev.finish_reason ?? null,
    toolCalls: r.toolCalls.map((c) => (c.done ? c : { ...c, done: true })),
  };
}

function applyEvent(r: RequestView, ev: ReqEvent): RequestView {
  switch (ev.type) {
    case 'request_start':
      return withStart(r, ev);
    case 'delta':
      return withDelta(r, ev);
    case 'kvsig':
      return { ...r, kvsig: ev };
    case 'progress':
      return withProgress(r, ev);
    case 'request_end':
      return withEnd(r, ev);
  }
}

// ---------------------------------------------------------------------------
// Order and eviction

function insertOrdered(order: string[], byId: Record<string, RequestView>, id: string): string[] {
  const ts = byId[id].startedTs;
  let k = order.length;
  while (k > 0 && byId[order[k - 1]].startedTs > ts) k--;
  const out = order.slice();
  out.splice(k, 0, id);
  return out;
}

/** Evict past CLIENT_MAX_REQUESTS. `revived` (a request brought back from
 *  `gone`) is not counted again if it goes at once. */
function evict(s: DockState, revived: string | null = null): DockState {
  if (s.order.length <= CLIENT_MAX_REQUESTS) return s;
  const byId = { ...s.byId };
  let order = s.order;
  let gone = s.gone;
  let trimmed = s.trimmed;
  let chars = s.chars;
  while (order.length > CLIENT_MAX_REQUESTS) {
    let k = order.findIndex((id) => byId[id].endedTs !== null);
    if (k < 0) k = 0;
    const id = order[k];
    order = order.slice(0, k).concat(order.slice(k + 1));
    chars -= byId[id].cost;
    gone = gone.concat({ id, conn: s.conn, live: byId[id].endedTs === null });
    delete byId[id];
    if (id !== revived) trimmed += 1;
  }
  if (gone.length > GONE_MAX) gone = gone.slice(gone.length - GONE_MAX);
  return { ...s, byId, order, gone, trimmed, chars };
}

function promptCost(p: PromptParts | null): number {
  if (!p) return 0;
  let n = p.sys?.text.length ?? 0;
  for (const m of p.messages) {
    n += m.text.length + (m.reasoning?.length ?? 0);
    if (m.tool_calls) for (const c of m.tool_calls) n += c.args.length;
  }
  return n;
}

function withCost(r: RequestView, prev: RequestView | undefined): RequestView {
  const pc = prev !== undefined && prev.prompt === r.prompt ? prev.promptCost : promptCost(r.prompt);
  const cost = r.thinking.text.length + r.reply.text.length + r.promptFlat.length + argsTotal(r.toolCalls) + pc;
  return { ...r, cost, promptCost: pc };
}

/** `r` with its text dropped (the global bound), counted as clipped. */
function strip(r: RequestView): RequestView {
  const sec = (x: Section): Section => (x.text ? { ...x, text: '', clipped: x.clipped + x.text.length } : x);
  const p = r.prompt;
  return {
    ...r,
    thinking: sec(r.thinking),
    reply: sec(r.reply),
    promptFlat: '',
    promptClipped: r.promptClipped + r.promptFlat.length + promptCost(p),
    prompt: p && {
      ...p,
      sys: p.sys && { ...p.sys, text: '', elided: true },
      messages: p.messages.map((m) => ({
        ...m,
        text: '',
        reasoning: undefined,
        elided: true,
        ...(m.tool_calls ? { tool_calls: m.tool_calls.map((c) => ({ ...c, args: '' })) } : {}),
      })),
    },
    toolCalls: r.toolCalls.map((c) => (c.args ? { ...c, args: '', elided: true } : c)),
    toolCallsClipped: r.toolCallsClipped + argsTotal(r.toolCalls),
    stripped: true,
    cost: 0,
    promptCost: 0,
  };
}

/** Drop the oldest requests' text until the total is within CLIENT_TOTAL_CHARS;
 *  `keep` (the request just changed) goes last. */
function bound(s: DockState, keep: string): DockState {
  if (s.chars <= CLIENT_TOTAL_CHARS) return s;
  const byId = { ...s.byId };
  let chars = s.chars;
  let stripped = s.stripped;
  for (const pass of [false, true]) {
    for (const id of s.order) {
      if (chars <= CLIENT_TOTAL_CHARS) break;
      const r = byId[id];
      if (r.cost === 0 || (id === keep) !== pass) continue;
      chars -= r.cost;
      if (!r.stripped) stripped += 1;
      byId[id] = strip(r);
    }
  }
  return { ...s, byId, chars, stripped };
}

/** Put `r` in the state: replace it in place, or add it and evict. */
function put(s: DockState, r0: RequestView): DockState {
  const prev = s.byId[r0.reqId];
  const r = withCost(r0, prev);
  const byId = { ...s.byId, [r.reqId]: r };
  const chars = s.chars + r.cost - (prev?.cost ?? 0);
  if (prev === undefined) {
    const was = s.gone.some((g) => g.id === r.reqId);
    const gone = was ? s.gone.filter((g) => g.id !== r.reqId) : s.gone;
    const added = { ...s, byId, gone, chars, order: insertOrdered(s.order, byId, r.reqId) };
    return bound(evict(added, was ? r.reqId : null), r.reqId);
  }
  if (prev.startedTs !== r.startedTs) {
    const order = insertOrdered(
      s.order.filter((id) => id !== r.reqId),
      byId,
      r.reqId,
    );
    return bound({ ...s, byId, order, chars }, r.reqId);
  }
  return bound({ ...s, byId, chars }, r.reqId);
}

// ---------------------------------------------------------------------------
// Stream events

/** apply: fold in. dup: already have it. rebuild: start over from a replayed
 *  start. merge: take a start's metadata only (no catch-up). */
type Verdict = 'apply' | 'dup' | 'rebuild' | 'merge';

function verdict(s: DockState, r: RequestView, ev: ReqEvent): Verdict {
  const seq = typeof ev.seq === 'number' ? ev.seq : null;
  if (ev.type === 'request_start' && ev.restored) return 'merge';
  if (s.seqScope === 'global' || r.conn === s.conn) {
    // A start behind its own deltas still fills the placeholder.
    if (ev.type === 'request_start' && !r.hasStart) return 'merge';
    return seq !== null && seq <= r.lastSeq ? 'dup' : 'apply';
  }
  // A key holder's request last caught up on an earlier connection.
  if (ev.type === 'request_start') {
    // Finished: what it holds is complete; the window may have dropped some
    // of it, so a rebuild could only lose text.
    if (r.endedTs !== null) return r.hasStart ? 'dup' : 'merge';
    return 'rebuild';
  }
  if (ev.type === 'progress') return 'apply'; // live only: never replayed
  if (r.endedTs !== null) return 'dup';
  return ev.ts <= r.lastTs ? 'dup' : 'apply';
}

function onRequestEvent(s: DockState, ev: ReqEvent): DockState {
  let r = s.byId[ev.req_id];
  if (r === undefined) {
    const g = s.gone.find((x) => x.id === ev.req_id);
    // Evicted: only a live one's start, replayed on a later connection, brings it back.
    if (g !== undefined && !(ev.type === 'request_start' && g.live && g.conn < s.conn)) return s;
    r = blank(ev.req_id, ev.ts, s.conn);
  } else if (ev.type === 'kvsig' && ev.restored) {
    // Fills a missing map only, and never counts as catching up: what this
    // request already holds decides which replayed events are new.
    return r.kvsig === null ? put(s, { ...r, kvsig: ev }) : s;
  } else {
    const v = verdict(s, r, ev);
    if (v === 'dup') return s;
    if (v === 'merge') {
      const m = withStart(r, ev as RequestStartEvent);
      const same = s.seqScope === 'global' || r.conn === s.conn;
      const seq = typeof ev.seq === 'number' ? ev.seq : 0;
      return put(s, same && seq > m.lastSeq ? { ...m, lastSeq: seq } : m);
    }
    if (v === 'rebuild') {
      r = { ...blank(r.reqId, ev.ts, s.conn), kept: r.kept };
    }
  }
  let next = applyEvent(r, ev);
  if (ev.type !== 'progress') {
    const caught = s.seqScope === 'connection' && next.conn !== s.conn;
    next = {
      ...next,
      lastTs: Math.max(next.lastTs, ev.ts),
      conn: s.conn,
      lastSeq: typeof ev.seq === 'number' && (caught || ev.seq > next.lastSeq) ? ev.seq : next.lastSeq,
    };
  } else if (typeof ev.seq === 'number' && next.conn === s.conn && ev.seq > next.lastSeq) {
    next = { ...next, lastSeq: ev.seq };
  }
  return put(s, next);
}

function onLagged(s: DockState): DockState {
  const byId: Record<string, RequestView> = { ...s.byId };
  for (const id of s.order) {
    const r = byId[id];
    if (r.endedTs === null && !r.gap) byId[id] = { ...r, gap: true };
  }
  return { ...s, byId, lagged: true, laggedCount: s.laggedCount + 1 };
}

// ---------------------------------------------------------------------------
// Kept captures

function onKeptList(s: DockState, kept: KeptSummary[]): DockState {
  let out = s;
  for (const k of kept) {
    if (!k || typeof k.req_id !== 'string') continue;
    const r = out.byId[k.req_id];
    if (r !== undefined) {
      if (!r.kept) out = put(out, { ...r, kept: true });
      continue;
    }
    const b = blank(k.req_id, k.started_ts ?? k.ended_ts ?? 0, out.conn);
    out = put(out, {
      ...b,
      requestId: k.request_id ?? null,
      tokenId: k.token_id ?? null,
      model: k.model ?? '',
      servedName: k.model ?? '',
      stream: !!k.stream,
      sessionId: k.session_id ?? null,
      turn: k.turn ?? null,
      endedTs: k.ended_ts ?? k.started_ts ?? 0,
      finish: k.finish_reason ?? null,
      kept: true,
      keptTokens: {
        prompt: typeof k.prompt_tokens === 'number' ? k.prompt_tokens : null,
        completion: typeof k.completion_tokens === 'number' ? k.completion_tokens : null,
      },
    });
  }
  return out;
}

function onKeptCapture(s: DockState, cap: KeptCapture): DockState {
  if (!cap || typeof cap.req_id !== 'string') return s;
  let r = s.byId[cap.req_id];
  const events = (Array.isArray(cap.events) ? cap.events : []) as unknown as ReqEvent[];
  const first = events.find((e) => e && e.type === 'request_start');
  if (r === undefined) r = blank(cap.req_id, first?.ts ?? 0, s.conn);
  let next: RequestView = { ...r, kept: true };
  for (const ev of events) {
    if (!ev || typeof ev !== 'object') continue;
    if (ev.type === 'request_start' && !next.hasStart) next = withStart(next, ev);
    else if (ev.type === 'kvsig' && next.kvsig === null) next = { ...next, kvsig: ev };
    else if (ev.type === 'request_end' && next.end === null) next = withEnd(next, ev);
  }
  return put(s, next);
}

function fillText(sec: Section, body: SectionResponse): Section {
  if (typeof body.text !== 'string') return sec;
  const offset = Math.max(0, body.offset ?? 0);
  const total = offset + body.text.length;
  if (total <= sec.chars) return sec;
  const [text, cut] = tail(body.text, CLIENT_SECTION_CHARS);
  return { ...sec, text, chars: total, clipped: offset + cut };
}

function fillCalls(r: RequestView, body: SectionResponse): RequestView {
  if (!Array.isArray(body.calls)) return r;
  let room = CLIENT_SECTION_CHARS;
  let clipped = 0;
  const calls: ToolCallView[] = [];
  for (const raw of body.calls) {
    const c = raw as { index?: unknown; call_id?: unknown; name?: unknown; args?: unknown; elided?: unknown };
    const args = typeof c.args === 'string' ? c.args : '';
    const keep = args.slice(0, room);
    room -= keep.length;
    clipped += args.length - keep.length;
    calls.push({
      index: typeof c.index === 'number' ? c.index : calls.length,
      callId: typeof c.call_id === 'string' ? c.call_id : null,
      name: typeof c.name === 'string' ? c.name : '',
      args: keep,
      done: !body.live || r.endedTs !== null,
      elided: c.elided === true || keep.length < args.length,
    });
  }
  if (argsTotal(calls) + clipped < argsTotal(r.toolCalls) + r.toolCallsClipped) return r;
  calls.sort((a, b) => a.index - b.index);
  return { ...r, toolCalls: calls, toolCallsClipped: clipped };
}

function fillPrompt(r: RequestView, body: SectionResponse): RequestView {
  if (!Array.isArray(body.messages)) return r;
  let sys: PromptParts['sys'] = null;
  let tools: PromptParts['tools'] = { count: 0, chars: 0, names: [] };
  const msgs: PromptMsg[] = [];
  for (const raw of body.messages) {
    const m = raw as Record<string, unknown>;
    if (m.role === 'sys') {
      sys = {
        chars: typeof m.chars === 'number' ? m.chars : 0,
        text: typeof m.text === 'string' ? m.text : '',
        elided: m.elided === true,
      };
    } else if (m.role === 'tools') {
      tools = {
        count: typeof m.count === 'number' ? m.count : 0,
        chars: typeof m.chars === 'number' ? m.chars : 0,
        names: Array.isArray(m.names) ? m.names.filter((n): n is string => typeof n === 'string') : [],
      };
    } else if (typeof m.role === 'string') {
      msgs.push({ ...(m as unknown as PromptMsg), text: typeof m.text === 'string' ? m.text : '' });
    }
  }
  // Bound the wide prompt: oldest messages go first, the newest's head last.
  let size = (sys?.text.length ?? 0) + msgs.reduce((n, m) => n + m.text.length + (m.reasoning?.length ?? 0), 0);
  let clipped = 0;
  let dropped = 0;
  while (size > CLIENT_SECTION_CHARS && msgs.length > 1) {
    const m = msgs.shift() as PromptMsg;
    const n = m.text.length + (m.reasoning?.length ?? 0);
    size -= n;
    clipped += n;
    dropped += 1;
  }
  if (size > CLIENT_SECTION_CHARS && msgs.length === 1) {
    const [text, cut] = tail(msgs[0].text, Math.max(0, msgs[0].text.length - (size - CLIENT_SECTION_CHARS)));
    msgs[0] = { ...msgs[0], text, elided: true };
    clipped += cut;
  }
  const o = (body.older ?? {}) as Record<string, unknown>;
  const older = (o.older ?? { from: 0, roles: '', chars: [] }) as PromptParts['older'];
  const prompt: PromptParts = {
    sys,
    tools,
    messages: msgs,
    older,
    collapsed: (typeof o.collapsed === 'number' ? o.collapsed : 0) + dropped,
    omitted: typeof o.omitted === 'number' ? o.omitted : 0,
    total_chars: typeof o.total_chars === 'number' ? o.total_chars : 0,
  };
  return { ...r, prompt, promptClipped: Math.max(r.promptClipped, clipped) };
}

function onSectionFill(s: DockState, a: SectionFillAction): DockState {
  const r = s.byId[a.req_id];
  const body = a.section;
  if (r === undefined || !body) return s;
  let next = r;
  switch (body.kind) {
    case 'thinking':
      next = { ...r, thinking: fillText(r.thinking, body) };
      break;
    case 'reply':
      next = { ...r, reply: fillText(r.reply, body) };
      break;
    case 'tool_calls':
      next = fillCalls(r, body);
      break;
    case 'prompt':
      next = fillPrompt(r, body);
      break;
    default:
      return s;
  }
  if (next === r || (next.thinking === r.thinking && next.reply === r.reply && next.toolCalls === r.toolCalls && next.prompt === r.prompt)) {
    return s;
  }
  return put(s, next);
}

// ---------------------------------------------------------------------------

export function reduce(s: DockState, ev: DockAction): DockState {
  if (!ev || typeof ev !== 'object') return s;
  switch (ev.type) {
    case 'request_start':
    case 'delta':
    case 'kvsig':
    case 'progress':
    case 'request_end':
      return typeof ev.req_id === 'string' ? onRequestEvent(s, ev) : s;
    case 'ring':
      return { ...s, ring: ev };
    case 'lagged':
      return onLagged(s);
    case 'connected':
      return { ...s, conn: s.conn + 1, seqScope: ev.seqScope };
    case 'kept_list':
      return Array.isArray(ev.kept) ? onKeptList(s, ev.kept) : s;
    case 'kept_capture':
      return onKeptCapture(s, ev.capture);
    case 'section_fill':
      return onSectionFill(s, ev);
    default:
      return s;
  }
}

export function findByRequestId(s: DockState, requestId: string): RequestView | null {
  for (const id of s.order) {
    const r = s.byId[id];
    if (r.requestId === requestId) return r;
  }
  return null;
}

/** What was dropped for a request, client and server side, for a "trimmed"
 *  mark: text chars (thinking, reply, prompt), tool-call argument chars and
 *  whole tool calls. */
export function trimmedOf(r: RequestView): { chars: number; toolCallChars: number; toolCalls: number } {
  return {
    chars: r.thinking.clipped + r.reply.clipped + r.promptClipped,
    toolCallChars: r.toolCallsClipped + (r.end?.tool_call_chars_dropped ?? 0),
    toolCalls: r.end?.tool_calls_dropped ?? 0,
  };
}
