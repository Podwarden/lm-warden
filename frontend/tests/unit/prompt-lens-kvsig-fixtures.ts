// Builders shared by the Prompt Lens prompt map tests (unit and component).
import type { KvsigEvent, KvsigPart, KvsigState, PromptParts, RequestEndEvent } from '@/lib/prompt-lens/events';
import type { RequestView } from '@/lib/prompt-lens/store';

const EMPTY = { text: '', chars: 0, clipped: 0, frames: 0, firstTs: null, lastTs: null };

export function part(
  p: KvsigPart['p'],
  from: number,
  to: number,
  state: KvsigState,
  over: Partial<KvsigPart> = {},
): KvsigPart {
  return {
    p,
    from,
    to,
    chars: 1000,
    roles: p === 'sys' ? { system: 1 } : p === 'tools' ? {} : { user: to - from + 1 },
    exact8: `e${from + 1}`.padEnd(8, '0'),
    loose8: `1${to + 1}`.padEnd(8, 'f'),
    state,
    ...over,
  };
}

export function kvsig(over: Partial<KvsigEvent> = {}): KvsigEvent {
  return {
    type: 'kvsig',
    req_id: 'r1',
    ts: 2,
    token_id: 'tok1',
    session_id: 's1',
    turn: 17,
    available: true,
    profile: 'loose/1',
    head8: 'aaaabbbb',
    conv8: 'e81d0b52',
    edge: 'continue',
    parent_request_id: '7f3c99aa11',
    exact_depth: 14,
    loose_depth: 15,
    n_nodes: 16,
    reusable_own: 30000,
    parts: [],
    ...over,
  };
}

export function end(over: Partial<RequestEndEvent> = {}): RequestEndEvent {
  return {
    type: 'request_end',
    req_id: 'r1',
    ts: 3,
    token_id: 'tok1',
    finish_reason: 'stop',
    prompt_tokens: 43920,
    completion_tokens: 10,
    reasoning_tokens: null,
    queued_s: 0,
    ttft_s: 1,
    duration_s: 2,
    decode_tps: 50,
    own_reused_tokens: null,
    ...over,
  };
}

export function reqWith(
  o: { parts?: KvsigPart[]; kvsig?: Partial<KvsigEvent> | null; end?: Partial<RequestEndEvent> | null; prompt?: PromptParts | null; phase?: string } = {},
): RequestView {
  const k = o.kvsig === null ? null : kvsig({ ...(o.kvsig ?? {}), ...(o.parts ? { parts: o.parts } : {}) });
  return {
    reqId: 'r1',
    requestId: 'h1',
    tokenId: 'tok1',
    model: 'm',
    servedName: 'm',
    stream: true,
    sessionId: 's1',
    turn: 17,
    startedTs: 1,
    endedTs: o.end === undefined || o.end === null ? null : 3,
    phase: o.phase ?? 'answering',
    finish: null,
    prompt: o.prompt ?? null,
    promptFlat: '',
    promptElided: false,
    promptClipped: 0,
    media: [],
    kvsig: k,
    thinking: EMPTY,
    reply: EMPTY,
    toolCalls: [],
    toolCallsClipped: 0,
    toolCallFrames: 0,
    toolCallFirstTs: null,
    end: o.end === undefined || o.end === null ? null : end(o.end),
    frames: { reasoning: 0, content: 0, tool_call: 0 },
    progressTokens: null,
    lastSeq: 0,
    lastTs: 0,
    conn: 0,
    hasStart: true,
    kept: false,
    gap: false,
    stripped: false,
    cost: 0,
    promptCost: 0,
  };
}

export const PROMPT: PromptParts = {
  sys: { chars: 1200, text: 'You are an agent', elided: false },
  tools: { count: 24, chars: 3600, names: ['read_file', 'edit', 'grep', 'bash', 'list_dir'] },
  messages: [
    { i: 14, role: 'assistant', chars: 700, text: 'I read the dock. It already scopes by token ids', elided: false },
    {
      i: 15,
      role: 'tool',
      chars: 900,
      text: '"use client"',
      elided: false,
      tool_call_id: 'call_8f2a',
      tool_name: 'read_file',
    },
    { i: 16, role: 'user', chars: 160, text: 'now wire the dock into /key/stats so it shows all my sessions', elided: false },
  ],
  older: { from: 0, roles: 'uauauauatuatut', chars: Array(14).fill(700) },
  collapsed: 14,
  omitted: 0,
  total_chars: 15760,
};

