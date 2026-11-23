import { describe, expect, it } from 'vitest';

import {
  parsePromptLensEvent,
  type DeltaEvent,
  type PromptLensEvent,
  type KeptCapture,
  type KvsigEvent,
  type ProgressEvent,
  type PromptParts,
  type RequestEndEvent,
  type RequestStartEvent,
  type SectionResponse,
} from '@/lib/prompt-lens/events';
import {
  CLIENT_MAX_REQUESTS,
  CLIENT_SECTION_CHARS,
  CLIENT_TOTAL_CHARS,
  type DockAction,
  type DockState,
  emptyDockState,
  findByRequestId,
  reduce,
  trimmedOf,
} from '@/lib/prompt-lens/store';

// ---------------------------------------------------------------------------
// Event builders. `seq` and `ts` count up across a test unless given.
// ---------------------------------------------------------------------------

let SEQ = 0;
let TS = 1_000;
const next = () => ({ seq: ++SEQ, ts: ++TS });

const PARTS: PromptParts = {
  sys: { chars: 12, text: 'You are dock', elided: false },
  tools: { count: 0, chars: 0, names: [] },
  messages: [{ i: 0, role: 'user', chars: 5, text: 'hello', elided: false }],
  older: { from: 0, roles: '', chars: [] },
  collapsed: 0,
  omitted: 0,
  total_chars: 17,
};

function start(id: string, over: Partial<RequestStartEvent> = {}): RequestStartEvent {
  return {
    type: 'request_start',
    ...next(),
    req_id: id,
    token_id: 'tok1',
    token_label: 'demo',
    request_id: `row-${id}`,
    model: 'm1',
    served_name: 'qwen',
    client_ip: '10.0.0.1',
    stream: true,
    prompt: 'hello',
    prompt_parts: PARTS,
    session_id: 's1',
    turn: 1,
    ...over,
  };
}

function delta(
  id: string,
  channel: 'content' | 'reasoning',
  text: string,
  over: Partial<DeltaEvent> = {},
): DeltaEvent {
  return { type: 'delta', ...next(), req_id: id, token_id: 'tok1', channel, text, ...over };
}

function tc(id: string, index: number, text: string, callId?: string, name?: string): DeltaEvent {
  return {
    type: 'delta',
    ...next(),
    req_id: id,
    token_id: 'tok1',
    channel: 'tool_call',
    index,
    text,
    ...(callId !== undefined ? { call_id: callId } : {}),
    ...(name !== undefined ? { name } : {}),
  };
}

function prog(id: string, phase: string, frames = { reasoning: 0, content: 0, tool_call: 0 }): ProgressEvent {
  return {
    type: 'progress',
    ...next(),
    req_id: id,
    token_id: 'tok1',
    phase,
    frames,
    completion_tokens: 3,
  };
}

function kvsig(id: string, over: Partial<KvsigEvent> = {}): KvsigEvent {
  return {
    type: 'kvsig',
    ...next(),
    req_id: id,
    token_id: 'tok1',
    session_id: 's1',
    turn: 1,
    available: true,
    exact_depth: 0,
    loose_depth: 0,
    parts: [],
    ...over,
  };
}

function end(id: string, finish: string | null = 'stop', over: Partial<RequestEndEvent> = {}): RequestEndEvent {
  return {
    type: 'request_end',
    ...next(),
    req_id: id,
    token_id: 'tok1',
    finish_reason: finish,
    prompt_tokens: 10,
    completion_tokens: 5,
    reasoning_tokens: null,
    cached_tokens: 0,
    queued_s: 0.1,
    ttft_s: 0.2,
    duration_s: 1,
    decode_tps: 5,
    own_reused_tokens: null,
    ...over,
  };
}

const run = (evs: DockAction[], s: DockState = emptyDockState()) => evs.reduce(reduce, s);

/** Strip the seq for a key-holder stream: numbered per connection from 1. */
function renumber<T extends PromptLensEvent>(evs: T[]): T[] {
  return evs.map((e, k) => ({ ...e, seq: k + 1 }));
}

function deepFreeze<T>(o: T): T {
  if (o && typeof o === 'object' && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o as object)) deepFreeze(v);
  }
  return o;
}

// ---------------------------------------------------------------------------

describe('Prompt Lens store: basics', () => {
  it('starts empty', () => {
    const s = emptyDockState();
    expect(s.byId).toEqual({});
    expect(s.order).toEqual([]);
    expect(s.ring).toBeNull();
    expect(s.lagged).toBe(false);
    expect(s.trimmed).toBe(0);
  });

  it('builds a request view from request_start', () => {
    const s = run([start('r')]);
    const r = s.byId.r;
    expect(r).toMatchObject({
      reqId: 'r',
      requestId: 'row-r',
      tokenId: 'tok1',
      tokenLabel: 'demo',
      clientIp: '10.0.0.1',
      model: 'm1',
      servedName: 'qwen',
      stream: true,
      sessionId: 's1',
      turn: 1,
      endedTs: null,
      phase: 'queued',
      finish: null,
      promptFlat: 'hello',
      kvsig: null,
      end: null,
      hasStart: true,
      kept: false,
    });
    expect(r.prompt).toEqual(PARTS);
    expect(r.thinking).toEqual({ text: '', chars: 0, clipped: 0, frames: 0, firstTs: null, lastTs: null });
    expect(s.order).toEqual(['r']);
  });

  it('appends content to reply and reasoning to thinking with timestamps', () => {
    const s = run([
      start('r'),
      delta('r', 'reasoning', 'hm', { ts: 5000 }),
      delta('r', 'reasoning', 'mm', { ts: 5001 }),
      delta('r', 'content', 'Hi', { ts: 5002 }),
    ]);
    expect(s.byId.r.thinking).toMatchObject({ text: 'hmmm', chars: 4, frames: 2, firstTs: 5000, lastTs: 5001 });
    expect(s.byId.r.reply).toMatchObject({ text: 'Hi', chars: 2, frames: 1, firstTs: 5002, lastTs: 5002 });
    expect(s.byId.r.frames).toEqual({ reasoning: 2, content: 1, tool_call: 0 });
  });

  it('keeps the kvsig event (key holders get "new" in place of "other")', () => {
    const st = start('r');
    const k = kvsig('r', {
      parts: [{ p: 'nodes', from: 0, to: 0, chars: 5, roles: { user: 1 }, exact8: 'aa', loose8: 'bb', state: 'new' }],
    });
    const s = run([st, k]);
    expect(s.byId.r.kvsig).toEqual(k);
  });

  it('is pure: never mutates its input, and leaves untouched requests as they were', () => {
    const s0 = deepFreeze(run([start('a'), start('b'), delta('a', 'content', 'x')]));
    const s1 = reduce(s0, delta('b', 'content', 'y'));
    expect(s1).not.toBe(s0);
    expect(s1.byId.a).toBe(s0.byId.a);
    expect(s1.byId.b).not.toBe(s0.byId.b);
    expect(s0.byId.b.reply.text).toBe('');
  });
});

describe('Prompt Lens store: tool calls', () => {
  it('assembles interleaved tool calls by index', () => {
    let s = reduce(emptyDockState(), start('r'));
    s = [tc('r', 0, '', 'c0', 'bash'), tc('r', 1, '', 'c1', 'grep'), tc('r', 0, '{"a"'), tc('r', 1, '{"b"')].reduce(
      reduce,
      s,
    );
    expect(s.byId.r.toolCalls.map((c) => [c.index, c.name, c.args])).toEqual([
      [0, 'bash', '{"a"'],
      [1, 'grep', '{"b"'],
    ]);
    expect(s.byId.r.toolCalls.map((c) => c.callId)).toEqual(['c0', 'c1']);
  });

  it('records when the first tool-call delta arrived (first token on a tool-call turn)', () => {
    const s = run([start('r'), tc('r', 0, '{', 'c0', 'bash'), tc('r', 0, '}')]);
    const first = s.byId.r.toolCallFirstTs;
    expect(first).not.toBeNull();
    expect(first).toBeLessThan(TS);
    expect(run([start('q')]).byId.q.toolCallFirstTs).toBeNull();
  });

  it('marks a call done only at request_end, not when a later index starts (G-F4)', () => {
    let s = run([start('r'), tc('r', 0, '{', 'c0', 'bash'), tc('r', 1, '{', 'c1', 'grep'), tc('r', 0, '}')]);
    expect(s.byId.r.toolCalls.map((c) => c.done)).toEqual([false, false]);
    expect(s.byId.r.phase).toBe('tool_call');
    s = reduce(s, end('r', 'tool_calls'));
    expect(s.byId.r.toolCalls.map((c) => c.done)).toEqual([true, true]);
    expect(s.byId.r.toolCalls[0].args).toBe('{}');
    expect(s.byId.r.finish).toBe('tool_calls');
  });

  it('sorts calls by index whatever order they open in', () => {
    const s = run([start('r'), tc('r', 2, 'x', 'c2', 'b'), tc('r', 0, 'y', 'c0', 'a')]);
    expect(s.byId.r.toolCalls.map((c) => c.index)).toEqual([0, 2]);
  });

  it('shows what the server cut (elided fragment, request_end drop counters) as trimmed', () => {
    const st = start('r');
    const cut: DeltaEvent = { ...tc('r', 0, 'abc', 'c0', 'bash'), elided: true };
    const s = run([
      st,
      cut,
      end('r', 'tool_calls', { tool_call_chars_dropped: 40, tool_calls_dropped: 2 }),
    ]);
    expect(s.byId.r.toolCalls[0].elided).toBe(true);
    expect(trimmedOf(s.byId.r)).toMatchObject({ toolCallChars: 40, toolCalls: 2 });
  });

  it('bounds tool-call arguments at CLIENT_SECTION_CHARS and counts the cut', () => {
    const big = 'x'.repeat(150_000);
    const s = run([start('r'), tc('r', 0, big, 'c0', 'a'), tc('r', 1, big, 'c1', 'b')]);
    const total = s.byId.r.toolCalls.reduce((n, c) => n + c.args.length, 0);
    expect(total).toBe(CLIENT_SECTION_CHARS);
    expect(s.byId.r.toolCallsClipped).toBe(100_000);
    expect(s.byId.r.toolCalls[1].elided).toBe(true);
    expect(trimmedOf(s.byId.r).toolCallChars).toBe(100_000);
  });
});

describe('Prompt Lens store: bounds', () => {
  it('clips a section past CLIENT_SECTION_CHARS and counts it', () => {
    const a = 'a'.repeat(150_000);
    const b = 'b'.repeat(150_000);
    const s = run([start('r'), delta('r', 'content', a), delta('r', 'content', b)]);
    const reply = s.byId.r.reply;
    expect(reply.text.length).toBe(CLIENT_SECTION_CHARS);
    expect(reply.clipped).toBe(100_000);
    expect(reply.chars).toBe(300_000);
    // The head went: the tail (all of b) survives.
    expect(reply.text.endsWith(b)).toBe(true);
    expect(reply.text.startsWith('a'.repeat(50_000))).toBe(true);
    expect(trimmedOf(s.byId.r).chars).toBe(100_000);
  });

  it('clips a huge flat prompt to its tail and counts it', () => {
    const s = run([start('r', { prompt: 'p'.repeat(CLIENT_SECTION_CHARS + 7) })]);
    expect(s.byId.r.promptFlat.length).toBe(CLIENT_SECTION_CHARS);
    expect(s.byId.r.promptClipped).toBe(7);
  });

  it('evicts the oldest finished request first', () => {
    const evs: DockAction[] = [start('live0')];
    for (let k = 1; k <= CLIENT_MAX_REQUESTS; k++) evs.push(start(`r${k}`), end(`r${k}`));
    const s = run(evs);
    expect(s.order).toHaveLength(CLIENT_MAX_REQUESTS);
    expect(s.byId.live0).toBeDefined();
    expect(s.byId.r1).toBeUndefined();
    expect(s.byId.r2).toBeDefined();
    expect(s.trimmed).toBe(1);
  });

  it('evicts the oldest live request only when none has finished', () => {
    const evs: DockAction[] = [];
    for (let k = 0; k <= CLIENT_MAX_REQUESTS; k++) evs.push(start(`r${k}`));
    const s = run(evs);
    expect(s.order).toHaveLength(CLIENT_MAX_REQUESTS);
    expect(s.byId.r0).toBeUndefined();
    expect(s.order[0]).toBe('r1');
  });

  it('ignores late events of an evicted request instead of resurrecting it', () => {
    const evs: DockAction[] = [];
    for (let k = 0; k <= CLIENT_MAX_REQUESTS; k++) evs.push(start(`r${k}`), end(`r${k}`));
    let s = run(evs);
    expect(s.byId.r0).toBeUndefined();
    s = reduce(s, delta('r0', 'content', 'late'));
    expect(s.byId.r0).toBeUndefined();
    expect(s.order).toHaveLength(CLIENT_MAX_REQUESTS);
  });
});

describe('Prompt Lens store: phase and finish', () => {
  it('phase follows progress, then end sets finish', () => {
    let s = run([start('r')]);
    expect(s.byId.r.phase).toBe('queued');
    s = reduce(s, prog('r', 'prefill'));
    expect(s.byId.r.phase).toBe('prefill');
    s = reduce(s, delta('r', 'reasoning', 'hm'));
    expect(s.byId.r.phase).toBe('thinking');
    s = reduce(s, prog('r', 'thinking', { reasoning: 4, content: 0, tool_call: 0 }));
    s = reduce(s, delta('r', 'content', 'ok'));
    expect(s.byId.r.phase).toBe('answering');
    s = reduce(s, prog('r', 'answering', { reasoning: 4, content: 9, tool_call: 0 }));
    s = reduce(s, end('r', 'stop'));
    expect(s.byId.r.finish).toBe('stop');
    expect(s.byId.r.endedTs).not.toBeNull();
    expect(s.byId.r.end?.ttft_s).toBe(0.2);
    expect(s.byId.r.frames).toEqual({ reasoning: 4, content: 9, tool_call: 0 });
    expect(s.byId.r.progressTokens).toBe(3);
  });

  it('keeps the raw phase values of the server (G-F6)', () => {
    const s = run([start('r'), prog('r', 'running_no_stream')]);
    expect(s.byId.r.phase).toBe('running_no_stream');
  });

  it('does not move the phase after the request ended', () => {
    const s = run([start('r'), delta('r', 'content', 'x'), end('r'), prog('r', 'thinking')]);
    expect(s.byId.r.phase).toBe('answering');
    expect(s.byId.r.finish).toBe('stop');
  });

  it('frames never go backwards (a late progress with fewer frames)', () => {
    const s = run([
      start('r'),
      prog('r', 'answering', { reasoning: 2, content: 8, tool_call: 0 }),
      prog('r', 'answering', { reasoning: 1, content: 3, tool_call: 0 }),
    ]);
    expect(s.byId.r.frames).toEqual({ reasoning: 2, content: 8, tool_call: 0 });
  });
});

describe('Prompt Lens store: global bound', () => {
  it('past CLIENT_TOTAL_CHARS the oldest requests lose their text first, counted as trimmed', () => {
    const big = 'x'.repeat(CLIENT_SECTION_CHARS);
    const n = Math.floor(CLIENT_TOTAL_CHARS / CLIENT_SECTION_CHARS) + 5;
    const evs: DockAction[] = [];
    for (let k = 0; k < n; k++) evs.push(start(`r${k}`, { prompt: '' }), delta(`r${k}`, 'content', big));
    const s = run(evs);
    expect(s.chars).toBeLessThanOrEqual(CLIENT_TOTAL_CHARS);
    expect(s.order).toHaveLength(n);
    expect(s.byId.r0.reply.text).toBe('');
    expect(s.byId.r0.stripped).toBe(true);
    expect(trimmedOf(s.byId.r0).chars).toBeGreaterThanOrEqual(CLIENT_SECTION_CHARS);
    expect(s.byId[`r${n - 1}`].reply.text).toBe(big);
    expect(s.byId[`r${n - 1}`].stripped).toBe(false);
    expect(s.stripped).toBeGreaterThan(0);
  });

  it('tracks the total text held', () => {
    const s = run([start('a', { prompt: 'pp' }), delta('a', 'content', 'abc'), tc('a', 0, 'xy', 'c', 'f')]);
    expect(s.chars).toBe(2 + 3 + 2 + PARTS.sys!.text.length + 5);
  });
});

describe('Prompt Lens store: lookup', () => {
  it('findByRequestId maps a table row id to the dock request', () => {
    const s = run([start('a', { request_id: 'row-1' }), start('b', { request_id: 'row-2' })]);
    expect(findByRequestId(s, 'row-2')?.reqId).toBe('b');
    expect(findByRequestId(s, 'row-9')).toBeNull();
  });
});

describe('Prompt Lens store: ordering, duplicates, replay', () => {
  it('a start that arrives after its delta (lower seq) still fills the placeholder', () => {
    const st = start('r');
    const d = delta('r', 'content', 'early');
    expect(st.seq).toBeLessThan(d.seq as number);
    let s = run([d]);
    expect(s.byId.r).toMatchObject({ hasStart: false, model: '', reply: { text: 'early' } });
    s = reduce(s, st);
    expect(s.byId.r).toMatchObject({ hasStart: true, model: 'm1', servedName: 'qwen', reply: { text: 'early' } });
    expect(s.byId.r.startedTs).toBe(st.ts);
    expect(s.byId.r.lastSeq).toBe(d.seq);
    // Once it has its start, a repeat of the start is a duplicate.
    const again = reduce(s, st);
    expect(again).toBe(s);
  });

  it('orders the list by start time', () => {
    const a = start('a');
    const b = start('b');
    const s = run([b, a]);
    expect(s.order).toEqual(['a', 'b']);
  });

  it('applies a duplicate event once (admin: global seq)', () => {
    const st = start('r');
    const d = delta('r', 'content', 'x');
    const s = run([st, d, d]);
    expect(s.byId.r.reply.text).toBe('x');
    expect(s.byId.r.lastSeq).toBe(d.seq);
  });

  it('admin reconnect: the ring replay is a no-op for what it already has', () => {
    const evs: PromptLensEvent[] = [start('r'), delta('r', 'content', 'ab'), tc('r', 0, '{}', 'c', 'f'), end('r')];
    const s1 = run(evs);
    const s2 = run([{ type: 'connected', seqScope: 'global' }, ...evs], s1);
    expect(s2.byId.r).toEqual(s1.byId.r);
  });

  it('admin reconnect: replay plus newer events appends only the newer ones', () => {
    const head: PromptLensEvent[] = [start('r'), delta('r', 'content', 'ab')];
    const s1 = run(head);
    const more = delta('r', 'content', 'cd');
    const s2 = run([{ type: 'connected', seqScope: 'global' }, ...head, more], s1);
    expect(s2.byId.r.reply.text).toBe('abcd');
  });

  it('key reconnect: a replay renumbered from 1 rebuilds from its start, never doubles text', () => {
    const evs = renumber<PromptLensEvent>([start('r'), delta('r', 'content', 'ab'), delta('r', 'reasoning', 'th')]);
    let s = run([{ type: 'connected', seqScope: 'connection' }, ...evs]);
    expect(s.byId.r.reply.text).toBe('ab');
    const more = { ...delta('r', 'content', 'cd'), seq: 4 };
    s = run([{ type: 'connected', seqScope: 'connection' }, ...evs, more], s);
    expect(s.byId.r.reply.text).toBe('abcd');
    expect(s.byId.r.thinking.text).toBe('th');
    expect(s.byId.r.reply.frames).toBe(2);
  });

  it('key reconnect: a replay whose start left the window skips what was already seen (by ts)', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'ab');
    const d2 = delta('r', 'content', 'cd');
    let s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([st, d1, d2])]);
    const d3 = delta('r', 'content', 'ef');
    // The key window evicted the start and d1; the replay is d2, then d3 live.
    s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([d2, d3])], s);
    expect(s.byId.r.reply.text).toBe('abcdef');
    // Within the new connection the seq rules again: a repeat is a duplicate.
    s = reduce(s, { ...d3, seq: 2 });
    expect(s.byId.r.reply.text).toBe('abcdef');
  });

  it('key reconnect: a finished request stays complete when the replay kept its start but not its deltas', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'ab');
    const d2 = delta('r', 'content', 'cd');
    const e = end('r');
    const s1 = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([st, d1, d2, e])]);
    // The window dropped the deltas; start and end survived.
    const s2 = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([st, e])], s1);
    expect(s2.byId.r.reply.text).toBe('abcd');
    expect(s2.byId.r.end).toEqual(s1.byId.r.end);
  });

  it('key reconnect: an evicted live request comes back when the replay carries its start', () => {
    const evs: DockAction[] = [{ type: 'connected', seqScope: 'connection' }];
    const first = start('old');
    evs.push(first);
    for (let k = 1; k <= CLIENT_MAX_REQUESTS; k++) evs.push(start(`r${k}`));
    let s = run(evs);
    expect(s.byId.old).toBeUndefined();
    // Make room (finish one), then reconnect: the replay has old's start again.
    s = reduce(s, end('r1'));
    const later = delta('old', 'content', 'back');
    s = run([{ type: 'connected', seqScope: 'connection' }, { ...first, seq: 1 }, { ...later, seq: 2 }], s);
    expect(s.byId.old?.reply.text).toBe('back');
    expect(s.byId.r1).toBeUndefined();
  });

  it('admin reconnect: an evicted finished request stays gone when the ring replays it', () => {
    const evs: DockAction[] = [];
    const first = start('r0');
    evs.push(first, end('r0'));
    for (let k = 1; k <= CLIENT_MAX_REQUESTS; k++) evs.push(start(`r${k}`), end(`r${k}`));
    let s = run(evs);
    expect(s.trimmed).toBe(1);
    s = run([{ type: 'connected', seqScope: 'global' }, first, end('r0')], s);
    expect(s.byId.r0).toBeUndefined();
    expect(s.byId.r1).toBeDefined();
    expect(s.trimmed).toBe(1);
    expect(s.order).toHaveLength(CLIENT_MAX_REQUESTS);
  });

  it('key reconnect: a finished request is not touched by a replay without its start', () => {
    const evs = [start('r'), delta('r', 'content', 'ab'), end('r')];
    const s1 = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>(evs)]);
    const s2 = run(
      [{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>(evs.slice(1))],
      s1,
    );
    expect(s2.byId.r.reply.text).toBe('ab');
    expect(s2.byId.r.end).toEqual(s1.byId.r.end);
  });

  it('key stream: a new request seen first on the new connection applies normally', () => {
    let s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([start('a')])]);
    s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([start('b'), delta('b', 'content', 'x')])], s);
    expect(s.byId.b.reply.text).toBe('x');
    expect(s.order).toEqual(['a', 'b']);
  });

  it('a stale event (seq below the last applied) is dropped, never reorders the text', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'one ');
    const d2 = delta('r', 'content', 'two');
    const s = run([st, d2, d1]);
    expect(s.byId.r.reply.text).toBe('two');
  });

  it('kvsig and end before the start still land on the request', () => {
    const st = start('r');
    const k = kvsig('r');
    const e = end('r');
    const s = run([k, e, st]);
    expect(s.byId.r).toMatchObject({ hasStart: true, finish: 'stop', model: 'm1' });
    expect(s.byId.r.kvsig).toEqual(k);
  });

  it('lagged marks the dock and every live request as having a gap', () => {
    const s = run([start('a'), start('b'), end('b'), { type: 'lagged' }]);
    expect(s.lagged).toBe(true);
    expect(s.laggedCount).toBe(1);
    expect(s.byId.a.gap).toBe(true);
    expect(s.byId.b.gap).toBe(false);
  });

  it('keeps the latest ring frame', () => {
    const ring = { type: 'ring' as const, oldest_ts: 900, events: 12, chars: 500, chars_max: 4_000_000 };
    const s = run([ring]);
    expect(s.ring).toEqual(ring);
  });
});

describe('Prompt Lens store: restored starts (a dock that connects mid-request)', () => {
  it('key holder, first connect: a restored start gives a running request its prompt, model and live boxes', () => {
    const st = { ...start('r'), restored: true };
    const kv = { ...kvsig('r'), restored: true };
    const d1 = delta('r', 'content', 'ab');
    const s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([st, kv, d1])]);
    const r = s.byId.r;
    expect(r.hasStart).toBe(true);
    expect(r.prompt).toEqual(PARTS);
    expect(r.servedName).toBe('qwen');
    expect(r.stream).toBe(true);
    expect(r.kvsig).not.toBeNull();
    expect(r.phase).toBe('answering');
    expect(r.endedTs).toBeNull();
    expect(r.reply.text).toBe('ab');
  });

  it('key reconnect: a restored start merges metadata and never resets text that arrived', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'ab');
    const d2 = delta('r', 'content', 'cd');
    let s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([st, d1, d2])]);
    const d3 = delta('r', 'content', 'ef');
    // New connection: the start left the window, so it comes back restored.
    s = run(
      [{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([{ ...st, restored: true }, d2, d3])],
      s,
    );
    expect(s.byId.r.reply.text).toBe('abcdef');
    expect(s.byId.r.reply.frames).toBe(3);
    expect(s.byId.r.hasStart).toBe(true);
  });

  it('a later normal start of the same request dedupes with the restored one', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'ab');
    let s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([{ ...st, restored: true }, d1])]);
    s = reduce(s, { ...st, seq: 3 });
    expect(s.byId.r.reply.text).toBe('ab');
    expect(s.order.filter((id) => id === 'r')).toHaveLength(1);
  });

  it('admin: a restored start fills a request known only from its deltas and keeps the text', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'ab');
    let s = run([d1]);
    expect(s.byId.r.hasStart).toBe(false);
    s = reduce(s, { ...st, restored: true });
    expect(s.byId.r.hasStart).toBe(true);
    expect(s.byId.r.servedName).toBe('qwen');
    expect(s.byId.r.reply.text).toBe('ab');
    s = reduce(s, delta('r', 'content', 'cd'));
    expect(s.byId.r.reply.text).toBe('abcd');
  });

  it('key reconnect: a restored kvsig never makes the replay apply text twice', () => {
    const st = start('r');
    const d1 = delta('r', 'content', 'ab');
    const d2 = delta('r', 'content', 'cd');
    let s = run([{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([st, d1, d2])]);
    const kv = { ...kvsig('r'), restored: true };
    s = run(
      [{ type: 'connected', seqScope: 'connection' }, ...renumber<PromptLensEvent>([{ ...st, restored: true }, kv, d2])],
      s,
    );
    expect(s.byId.r.reply.text).toBe('abcd');
    expect(s.byId.r.kvsig).not.toBeNull();
  });

  it('a restored kvsig fills a missing map and never replaces one', () => {
    const st = start('r');
    let s = run([st]);
    s = reduce(s, { ...kvsig('r', { exact_depth: 2 }), restored: true });
    expect(s.byId.r.kvsig?.exact_depth).toBe(2);
    s = reduce(s, { ...kvsig('r', { exact_depth: 9 }), restored: true });
    expect(s.byId.r.kvsig?.exact_depth).toBe(2);
  });
});

describe('Prompt Lens events: parse', () => {
  it('parses stream frames and rejects what is not an event', () => {
    expect(parsePromptLensEvent('{"type":"lagged"}')).toEqual({ type: 'lagged' });
    expect(parsePromptLensEvent('{"type":"delta","req_id":"r","ts":1,"channel":"content","text":"x"}')?.type).toBe(
      'delta',
    );
    expect(parsePromptLensEvent('{"type":"delta"}')).toBeNull();
    expect(parsePromptLensEvent('{"type":"other","req_id":"r"}')).toBeNull();
    expect(parsePromptLensEvent('not json')).toBeNull();
    expect(parsePromptLensEvent('[1]')).toBeNull();
  });
});

describe('Prompt Lens store: kept captures', () => {
  const summary = {
    req_id: 'k1',
    request_id: 'row-k1',
    model: 'qwen',
    stream: true,
    session_id: 's9',
    turn: 3,
    started_ts: 500,
    ended_ts: 510,
    finish_reason: 'stop',
    prompt_tokens: 10,
    completion_tokens: 4,
    kept: true,
  };

  it('a kept list adds finished rows marked kept, in start order', () => {
    const s = run([start('live'), { type: 'kept_list', kept: [summary] }]);
    expect(s.order).toEqual(['k1', 'live']);
    expect(s.byId.k1).toMatchObject({
      reqId: 'k1',
      requestId: 'row-k1',
      servedName: 'qwen',
      sessionId: 's9',
      turn: 3,
      startedTs: 500,
      endedTs: 510,
      finish: 'stop',
      kept: true,
      hasStart: false,
    });
  });

  it('a kept list row keeps the summary token counts (G-R15)', () => {
    const s = run([{ type: 'kept_list', kept: [summary] }]);
    expect(s.byId.k1.keptTokens).toEqual({ prompt: 10, completion: 4 });
    const bare = run([{ type: 'kept_list', kept: [{ ...summary, prompt_tokens: undefined, completion_tokens: null }] }]);
    expect(bare.byId.k1.keptTokens).toEqual({ prompt: null, completion: null });
  });

  it('a kept list marks a request already in the dock as kept and changes nothing else', () => {
    const s1 = run([start('k1'), delta('k1', 'content', 'live text')]);
    const s2 = reduce(s1, { type: 'kept_list', kept: [summary] });
    expect(s2.byId.k1.kept).toBe(true);
    expect(s2.byId.k1.reply).toEqual(s1.byId.k1.reply);
    expect(s2.byId.k1.startedTs).toBe(s1.byId.k1.startedTs);
  });

  it('a kept capture fills start, kvsig and end (its start has no prompt_parts)', () => {
    const st = start('k1', { ts: 500, prompt_parts: undefined, client_ip: undefined, token_label: undefined });
    delete (st as Partial<RequestStartEvent>).prompt_parts;
    const capture: KeptCapture = {
      req_id: 'k1',
      kept: true,
      events: [st, kvsig('k1'), end('k1', 'length')] as unknown as KeptCapture['events'],
      sections: {},
    };
    const s = run([{ type: 'kept_list', kept: [summary] }, { type: 'kept_capture', capture }]);
    expect(s.byId.k1).toMatchObject({ hasStart: true, kept: true, finish: 'length', promptFlat: 'hello' });
    expect(s.byId.k1.prompt).toBeNull();
    expect(s.byId.k1.kvsig?.available).toBe(true);
    expect(s.byId.k1.end?.finish_reason).toBe('length');
  });

  it('a kept capture never overwrites what the live stream already gave', () => {
    const s1 = run([start('k1'), delta('k1', 'content', 'live'), end('k1', 'stop')]);
    const capture: KeptCapture = {
      req_id: 'k1',
      kept: true,
      events: [start('k1', { prompt: 'other' }), end('k1', 'length')] as unknown as KeptCapture['events'],
      sections: {},
    };
    const s2 = reduce(s1, { type: 'kept_capture', capture });
    expect(s2.byId.k1.kept).toBe(true);
    expect(s2.byId.k1.finish).toBe('stop');
    expect(s2.byId.k1.promptFlat).toBe('hello');
    expect(s2.byId.k1.reply.text).toBe('live');
  });

  it('a section fill restores a kept reply, clipped to the client bound', () => {
    const section: SectionResponse = {
      kind: 'reply',
      seq: null,
      live: false,
      dropped_chars: 10,
      offset: 10,
      text: 'r'.repeat(CLIENT_SECTION_CHARS + 5),
      dropped_calls: 0,
      kept: true,
    };
    const s = run([{ type: 'kept_list', kept: [summary] }, { type: 'section_fill', req_id: 'k1', section }]);
    expect(s.byId.k1.reply.text.length).toBe(CLIENT_SECTION_CHARS);
    expect(s.byId.k1.reply.chars).toBe(CLIENT_SECTION_CHARS + 15);
    expect(s.byId.k1.reply.clipped).toBe(15);
  });

  it('a section fill never replaces a longer section the client already has', () => {
    const s1 = run([start('r'), delta('r', 'content', 'abcdef')]);
    const section: SectionResponse = {
      kind: 'reply',
      seq: null,
      live: true,
      dropped_chars: 0,
      offset: 0,
      text: 'abc',
      dropped_calls: 0,
      kept: false,
    };
    const s2 = reduce(s1, { type: 'section_fill', req_id: 'r', section });
    expect(s2.byId.r.reply.text).toBe('abcdef');
  });

  it('a prompt section fill gives a kept capture its structured prompt', () => {
    const section: SectionResponse = {
      kind: 'prompt',
      seq: null,
      live: false,
      dropped_chars: 0,
      offset: 0,
      messages: [
        { role: 'sys', chars: 3, text: 'sys', elided: false },
        { role: 'tools', count: 1, chars: 20, names: ['bash'] },
        { i: 4, role: 'user', chars: 2, text: 'hi', elided: false },
        { i: 5, role: 'tool', chars: 2, text: 'ok', elided: false, tool_call_id: 'c1' },
      ],
      older: { older: { from: 0, roles: 'uaua', chars: [1, 2, 3, 4] }, collapsed: 4, omitted: 0, total_chars: 37 },
      dropped_calls: 0,
      kept: true,
    };
    const s = run([{ type: 'kept_list', kept: [summary] }, { type: 'section_fill', req_id: 'k1', section }]);
    const p = s.byId.k1.prompt;
    expect(p?.sys).toEqual({ chars: 3, text: 'sys', elided: false });
    expect(p?.tools).toEqual({ count: 1, chars: 20, names: ['bash'] });
    expect(p?.messages.map((m) => [m.i, m.role])).toEqual([
      [4, 'user'],
      [5, 'tool'],
    ]);
    expect(p?.older.roles).toBe('uaua');
    expect(p?.collapsed).toBe(4);
    expect(p?.total_chars).toBe(37);
  });

  it('a tool_calls section fill restores the calls of a finished request as done', () => {
    const section: SectionResponse = {
      kind: 'tool_calls',
      seq: null,
      live: false,
      dropped_chars: 0,
      offset: 0,
      calls: [{ index: 0, call_id: 'c0', name: 'bash', args: '{"x":1}', elided: true }],
      dropped_calls: 1,
      kept: true,
    };
    const s = run([{ type: 'kept_list', kept: [summary] }, { type: 'section_fill', req_id: 'k1', section }]);
    expect(s.byId.k1.toolCalls).toEqual([
      { index: 0, callId: 'c0', name: 'bash', args: '{"x":1}', done: true, elided: true },
    ]);
  });

  it('a section fill for an unknown request is ignored', () => {
    const s0 = emptyDockState();
    const s1 = reduce(s0, {
      type: 'section_fill',
      req_id: 'nope',
      section: { kind: 'reply', live: false, dropped_chars: 0, offset: 0, text: 'x', dropped_calls: 0, kept: false },
    });
    expect(s1).toBe(s0);
  });
});
