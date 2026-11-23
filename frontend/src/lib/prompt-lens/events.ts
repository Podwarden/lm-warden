// Prompt Lens wire types: the events of the Prompt Lens SSE streams, field for
// field as the server builds them.
//
//   * admin:      GET /api/admin/prompt-lens/stream (app/proxy/routes_prompt_lens.py)
//   * key holder: GET /api/me/prompt-lens/stream    (app/tokens/routes_me_prompt_lens.py)
//
// Both come from one generator (app/proxy/prompt_lens_stream.py). The events are
// published in app/proxy/routes.py (request_start, kvsig, delta, progress,
// request_end); the structured prompt is app/proxy/prompt_lens_capture.py
// (`structured_prompt`), the prompt map app/proxy/prompt_lens_kvsig.py. A key
// holder receives each event through `prompt_lens_scope.for_key_holder`, which
// drops the fields typed optional below (`client_ip`, `token_label`,
// `reusable_any`, `cached_tokens`, `cache`) and numbers `seq` per connection.
//
// Phase and state values are the server's raw vocabulary (ruling G-F6:
// `tool_call`, `prefill`, ...); the UI maps them to labels.

import type { components } from '@/lib/api-types.generated';

/** Captured image parts of a request (spec 2026-08-03). */
export interface MediaEntry {
  kind: 'image';
  /** Store-backed: fetch /api/admin/prompt-lens/media/{id}. */
  media_id?: string;
  mime?: string;
  /** Base64 length of the stored payload. */
  chars?: number;
  /** A remote image, hotlinked. */
  url?: string;
  dropped?: 'too_large' | 'count';
  count?: number;
}

// ---------------------------------------------------------------------------
// The structured prompt (PromptParts, prompt_lens_capture.structured_prompt)
// ---------------------------------------------------------------------------

/** One tool call of an assistant message in the prompt. */
export interface PromptCall {
  id: string | null;
  name: string;
  /** Arguments, windowed (head + tail) to the per-message budget. */
  args: string;
}

/** One listed (kept) body message. `i` is its body index, which is its kvsig
 *  node index. `text` is always a string (windowed). */
export interface PromptMsg {
  i: number;
  role: string;
  /** The message's whole size: text, reasoning and tool-call arguments. */
  chars: number;
  text: string;
  elided: boolean;
  /** An assistant message's reasoning, windowed. */
  reasoning?: string;
  tool_calls?: PromptCall[];
  /** Tool calls past the per-message budget, not listed. */
  calls_omitted?: number;
  tool_call_id?: string;
  /** The name of the call a tool result answers, when resolvable. */
  tool_name?: string;
  /** Image parts counted (never copied). */
  images?: number;
}

/** One-char role codes of the collapsed list: user, assistant, tool,
 *  function, anything else. */
export type OlderRole = 'u' | 'a' | 't' | 'f' | '?';

/** The collapsed (text-less) body messages, as parallel arrays starting at
 *  body index `from`: `roles[k]` and `chars[k]` are message `from + k`. */
export interface PromptOlder {
  from: number;
  /** One char per message (`OlderRole`). */
  roles: string;
  chars: number[];
}

export interface PromptParts {
  /** The head (system/developer messages) joined by a blank line, windowed. */
  sys: { chars: number; text: string; elided: boolean } | null;
  tools: { count: number; chars: number; names: string[] };
  /** The kept messages, oldest first (the newest is always kept). */
  messages: PromptMsg[];
  older: PromptOlder;
  /** Body messages not in `messages`: `older`'s plus `omitted`. */
  collapsed: number;
  /** The oldest collapsed messages, counted only (not in `older`). */
  omitted: number;
  total_chars: number;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

interface EventBase {
  /** Admin: the hub's global seq. Key holder: per connection, from 1 (a
   *  reconnect renumbers the replay). */
  seq?: number;
  req_id: string;
  /** Wall clock seconds (server `time.time()`). */
  ts: number;
  token_id: string | null;
}

export interface RequestStartEvent extends EventBase {
  type: 'request_start';
  /** A copy the server replays at connect for a request still running whose
   *  own start already left the replay: metadata only, merged, never a reset. */
  restored?: boolean;
  /** Admin only. */
  token_label?: string | null;
  /** The request-history id (the requests table's row id). */
  request_id: string | null;
  model: string;
  served_name: string;
  /** Admin only. */
  client_ip?: string | null;
  stream: boolean;
  /** The flat prompt (kept for one release), windowed. */
  prompt: string;
  prompt_elided?: boolean;
  /** Null when the body is not a chat body. Absent in kept captures (their
   *  prompt comes from section/prompt). */
  prompt_parts?: PromptParts | null;
  session_id: string | null;
  turn: number | null;
  media?: MediaEntry[];
}

export type DeltaChannel = 'content' | 'reasoning' | 'tool_call';

export interface DeltaEvent extends EventBase {
  type: 'delta';
  channel: DeltaChannel;
  text: string;
  /** tool_call only: the call's slot in the reply. */
  index?: number;
  /** tool_call only, on the fragment that carries them. */
  call_id?: string;
  name?: string;
  /** tool_call only: the fragment was cut by the per-request budget. */
  elided?: boolean;
}

/** exact / loose: this session's own lineage knows the part. other: another
 *  key made it cached (admin only; key holders get `new`, ruling G-R11).
 *  new: none of those. */
export type KvsigState = 'exact' | 'loose' | 'new' | 'other';

export interface KvsigPart {
  p: 'sys' | 'tools' | 'nodes';
  /** Body node range, inclusive; -1 for sys and tools. */
  from: number;
  to: number;
  chars: number;
  /** Role name -> count (unknown roles counted as "unknown"). */
  roles: Record<string, number>;
  exact8: string;
  loose8: string;
  state: KvsigState;
}

export type KvsigReason = 'not_chat' | 'no_key' | 'error' | 'over_cap';
export type KvsigEdge = 'new' | 'continue' | 'fork' | 'compacted_from' | 'subagent';

export interface KvsigEvent extends EventBase {
  type: 'kvsig';
  /** Replayed with a restored start: fills a missing map, never replaces one. */
  restored?: boolean;
  session_id: string | null;
  turn: number | null;
  available: boolean;
  /** Set when `available` is false. */
  reason?: KvsigReason;
  profile?: string;
  head8?: string;
  conv8?: string | null;
  edge?: KvsigEdge | null;
  parent_request_id?: string | null;
  exact_depth?: number;
  /** max(exact_depth, loose depth). */
  loose_depth?: number;
  n_nodes?: number;
  reusable_own?: number | null;
  /** Admin only. */
  reusable_any?: number | null;
  parts: KvsigPart[];
  /** The admission wait (G-R14): the kvsig event is published at admission;
   *  request_end repeats it. Absent from servers before it. */
  queued_s?: number | null;
}

export interface ProgressEvent extends EventBase {
  type: 'progress';
  /** The live row's raw phase (`thinking`, `answering`, `tool_call`,
   *  `prefill`, `running`, ...). */
  phase: string;
  /** SSE frames with text per channel (approximate rates). */
  frames: { reasoning: number; content: number; tool_call: number };
  completion_tokens: number | null;
}

export interface RequestEndEvent extends EventBase {
  type: 'request_end';
  /** stop, length, tool_calls, abort, error, runaway, or the engine's own. */
  finish_reason: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number | null;
  /** Admin only. */
  cached_tokens?: number | null;
  queued_s: number | null;
  /** From admission: total wait to first token = ttft_s + queued_s. */
  ttft_s: number | null;
  duration_s: number | null;
  decode_tps: number | null;
  own_reused_tokens: number | null;
  /** Admin only: the cache observation. */
  cache?: Record<string, unknown> | null;
  /** Sent only when > 0 (ruling G-R5). */
  tool_call_chars_dropped?: number;
  tool_calls_dropped?: number;
}

/** Admin only: what the replay ring holds. No seq, no req_id. */
export interface RingEvent {
  type: 'ring';
  oldest_ts: number | null;
  events: number;
  chars: number;
  chars_max: number;
}

/** The client fell behind its queue; the server dropped events. No seq. */
export interface LaggedEvent {
  type: 'lagged';
}

export type PromptLensEvent =
  | RequestStartEvent
  | DeltaEvent
  | KvsigEvent
  | ProgressEvent
  | RequestEndEvent
  | RingEvent
  | LaggedEvent;

// ---------------------------------------------------------------------------
// REST shapes (generated from the server's models)
// ---------------------------------------------------------------------------

type Schemas = components['schemas'];
export type SectionResponse = Schemas['SectionResponse'];
export type SectionKind = SectionResponse['kind'];
/** A kept capture in a list; the admin's carries `token_id`. */
export type KeptSummary = Schemas['KeptSummary'] & { token_id?: string };
/** A kept capture: its start (no prompt_parts), kvsig and end events. */
export type KeptCapture = Schemas['KeptCapture'] & { token_id?: string };

/** Parse one SSE `data:` payload into an event, or null when it is not one. */
export function parsePromptLensEvent(data: string): PromptLensEvent | null {
  let v: unknown;
  try {
    v = JSON.parse(data);
  } catch {
    return null;
  }
  if (!v || typeof v !== 'object') return null;
  const t = (v as { type?: unknown }).type;
  switch (t) {
    case 'ring':
    case 'lagged':
      return v as PromptLensEvent;
    case 'request_start':
    case 'delta':
    case 'kvsig':
    case 'progress':
    case 'request_end':
      return typeof (v as { req_id?: unknown }).req_id === 'string' ? (v as PromptLensEvent) : null;
    default:
      return null;
  }
}
