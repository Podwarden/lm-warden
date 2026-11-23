import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { PromptParts } from '@/lib/prompt-lens/events';
import {
  activeCount,
  applyFilters,
  defaultFilters,
  FINISH_VALUES,
  useDockFilters,
} from '@/lib/prompt-lens/filters';
import { emptyDockState, reduce, type RequestView } from '@/lib/prompt-lens/store';

const parts = (over: Partial<PromptParts> = {}): PromptParts => ({
  sys: null,
  tools: { count: 0, chars: 0, names: [] },
  messages: [],
  older: { from: 0, roles: '', chars: [] },
  collapsed: 0,
  omitted: 0,
  total_chars: 0,
  ...over,
});

/** A RequestView built through the reducer, then patched. */
function row(id: string, over: Partial<RequestView> = {}): RequestView {
  const s = reduce(emptyDockState(), {
    type: 'request_start',
    seq: 1,
    ts: 100,
    req_id: id,
    token_id: 'tok1',
    request_id: `row-${id}`,
    model: 'm1',
    served_name: 'qwen',
    client_ip: '1.2.3.4',
    stream: true,
    prompt: '',
    prompt_parts: parts(),
    session_id: 's1',
    turn: 1,
  });
  return { ...s.byId[id], ...over };
}

const section = (text: string) => ({ text, chars: text.length, clipped: 0, frames: 1, firstTs: 1, lastTs: 1 });

const rows: RequestView[] = [
  row('r1', { promptFlat: 'plain words' }),
  row('r2', {
    servedName: 'llama',
    model: 'm2',
    clientIp: '5.6.7.8',
    sessionId: 's2',
    prompt: parts({ messages: [{ i: 0, role: 'user', chars: 12, text: 'fix the dock [x]', elided: false }] }),
    endedTs: 200,
    finish: 'tool_calls',
    toolCalls: [{ index: 0, callId: 'c', name: 'bash', args: '{"cmd":"ls"}', done: true, elided: false }],
  }),
  row('r3', {
    thinking: section('pondering'),
    endedTs: 300,
    finish: 'stop',
    prompt: parts({
      messages: [{ i: 2, role: 'tool', chars: 2, text: 'ok', elided: false, tool_call_id: 'c' }],
    }),
  }),
  row('r4', {
    reply: section('An Answer'),
    media: [{ kind: 'image', media_id: 'm' }],
    prompt: parts({ older: { from: 0, roles: 'ut', chars: [1, 2] }, collapsed: 2 }),
  }),
];

const ids = (rs: RequestView[]) => rs.map((r) => r.reqId);

afterEach(() => {
  localStorage.clear();
});

describe('Prompt Lens filters: defaults', () => {
  it('admin default presets the scope model and every key', () =>
    expect(defaultFilters('admin', 'qwen')).toMatchObject({ models: ['qwen'], keys: null }));

  it('a key preset names those keys', () =>
    expect(defaultFilters('admin', null, ['t1', 't2'])).toMatchObject({ models: null, keys: ['t1', 't2'] }));

  it('the plain default filters nothing', () => {
    expect(defaultFilters('key')).toEqual({
      keys: null,
      sessions: null,
      models: null,
      ips: null,
      phase: 'all',
      finish: null,
      has: [],
      q: '',
    });
    expect(applyFilters(rows, defaultFilters('admin'), 'admin')).toHaveLength(rows.length);
  });

  it('offers the six finish values', () =>
    expect(FINISH_VALUES).toEqual(['stop', 'length', 'tool_calls', 'abort', 'error', 'runaway']));
});

describe('Prompt Lens filters: apply', () => {
  it('key scope never filters by IP', () =>
    expect(applyFilters(rows, { ...defaultFilters('key'), ips: ['1.2.3.4'] }, 'key')).toHaveLength(rows.length));

  it('admin scope filters by IP', () =>
    expect(ids(applyFilters(rows, { ...defaultFilters('admin'), ips: ['5.6.7.8'] }, 'admin'))).toEqual(['r2']));

  it('does not filter by key (that is server side)', () =>
    expect(applyFilters(rows, { ...defaultFilters('admin'), keys: ['nobody'] }, 'admin')).toHaveLength(rows.length));

  it('search is plain and case-insensitive', () =>
    expect(applyFilters(rows, { ...defaultFilters('admin'), q: 'DOCK [' }, 'admin').map((r) => r.reqId)).toEqual(['r2']));

  it('search reads the flat prompt, thinking, reply and tool-call args', () => {
    const q = (text: string) => ids(applyFilters(rows, { ...defaultFilters('admin'), q: text }, 'admin'));
    expect(q('PLAIN')).toEqual(['r1']);
    expect(q('ponder')).toEqual(['r3']);
    expect(q('an answer')).toEqual(['r4']);
    expect(q('"cmd"')).toEqual(['r2']);
    expect(q('   ')).toHaveLength(rows.length);
  });

  it('filters by model (served name or model id) and session', () => {
    expect(ids(applyFilters(rows, { ...defaultFilters('admin', 'llama') }, 'admin'))).toEqual(['r2']);
    expect(ids(applyFilters(rows, { ...defaultFilters('admin', 'm2') }, 'admin'))).toEqual(['r2']);
    expect(ids(applyFilters(rows, { ...defaultFilters('admin'), sessions: ['s2'] }, 'admin'))).toEqual(['r2']);
  });

  it('filters live and finished', () => {
    expect(ids(applyFilters(rows, { ...defaultFilters('admin'), phase: 'live' }, 'admin'))).toEqual(['r1', 'r4']);
    expect(ids(applyFilters(rows, { ...defaultFilters('admin'), phase: 'finished' }, 'admin'))).toEqual(['r2', 'r3']);
  });

  it('filters by finish', () =>
    expect(ids(applyFilters(rows, { ...defaultFilters('admin'), finish: ['stop'] }, 'admin'))).toEqual(['r3']));

  it('has: tool_results matches a prompt with a tool message', () => {
    // r3 lists a tool message; r4 has one among its collapsed (older) ones.
    expect(ids(applyFilters(rows, { ...defaultFilters('admin'), has: ['tool_results'] }, 'admin'))).toEqual([
      'r3',
      'r4',
    ]);
  });

  it('has: tool_calls, thinking and images', () => {
    const has = (h: ('tool_calls' | 'thinking' | 'images' | 'tool_results')[]) =>
      ids(applyFilters(rows, { ...defaultFilters('admin'), has: h }, 'admin'));
    expect(has(['tool_calls'])).toEqual(['r2']);
    expect(has(['thinking'])).toEqual(['r3']);
    expect(has(['images'])).toEqual(['r4']);
    expect(has(['thinking', 'tool_results'])).toEqual(['r3']);
  });

  it('counts active filters', () => {
    expect(activeCount(defaultFilters('admin'))).toBe(0);
    expect(activeCount(defaultFilters('admin', 'qwen'))).toBe(1);
    expect(
      activeCount({ ...defaultFilters('admin'), phase: 'live', has: ['images'], q: 'x', finish: ['stop'] }),
    ).toBe(4);
    expect(activeCount({ ...defaultFilters('admin'), q: '  ', finish: [] })).toBe(0);
  });
});

describe('Prompt Lens filters: persistence', () => {
  const KEY = 'vw.prompt-lens.filters.stats';
  const OLD_KEY = 'vw.godmode.filters.stats';

  it('reads filters saved under the old god mode key, then saves under the new one', () => {
    localStorage.setItem(OLD_KEY, JSON.stringify({ models: ['m'], phase: 'finished' }));
    const { result } = renderHook(() => useDockFilters('stats', 'admin', {}));
    expect(result.current[0]).toMatchObject({ models: ['m'], phase: 'finished' });
    act(() => result.current[1]({ ...result.current[0], phase: 'live' }));
    expect(JSON.parse(localStorage.getItem(KEY) ?? '{}')).toMatchObject({ phase: 'live' });
    expect(localStorage.getItem(OLD_KEY)).toBeNull();
  });

  it('the new key wins over the old one', () => {
    localStorage.setItem(OLD_KEY, JSON.stringify({ phase: 'finished' }));
    localStorage.setItem(KEY, JSON.stringify({ phase: 'live' }));
    const { result } = renderHook(() => useDockFilters('stats', 'admin', {}));
    expect(result.current[0].phase).toBe('live');
  });

  it('starts from the defaults and persists a change', () => {
    const { result } = renderHook(() => useDockFilters('stats', 'admin', { model: 'qwen' }));
    expect(result.current[0].models).toEqual(['qwen']);
    act(() => result.current[1]({ ...result.current[0], phase: 'live' }));
    expect(result.current[0].phase).toBe('live');
    expect(JSON.parse(localStorage.getItem(KEY) ?? '{}')).toMatchObject({ phase: 'live', models: ['qwen'] });
  });

  it('restores what was persisted', () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({ models: ['m'], phase: 'finished', finish: ['stop'], has: ['images'] }),
    );
    const { result } = renderHook(() => useDockFilters('stats', 'admin', {}));
    expect(result.current[0]).toMatchObject({ models: ['m'], phase: 'finished', finish: ['stop'], has: ['images'] });
  });

  it('never stores or restores search text, sessions, IPs or keys', () => {
    const { result } = renderHook(() => useDockFilters('stats', 'admin', {}));
    act(() =>
      result.current[1]({ ...result.current[0], q: 'secret', sessions: ['s'], ips: ['1.1.1.1'], keys: ['k'] }),
    );
    expect(result.current[0].q).toBe('secret');
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}');
    expect(Object.keys(saved).sort()).toEqual(['finish', 'has', 'models', 'phase']);
    localStorage.setItem(KEY, JSON.stringify({ q: 'old', sessions: ['s'], ips: ['i'], keys: ['k'] }));
    const b = renderHook(() => useDockFilters('stats', 'admin', {}));
    expect(b.result.current[0]).toEqual(defaultFilters('admin'));
  });

  it('does not persist preset keys', () => {
    const { result } = renderHook(() => useDockFilters('stats', 'admin', { keys: ['t1'] }));
    expect(result.current[0].keys).toEqual(['t1']);
    act(() => result.current[1]({ ...result.current[0], keys: ['t2'], phase: 'live' }));
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}');
    expect(saved.phase).toBe('live');
    expect('keys' in saved).toBe(false);
  });

  it('a preset of keys wins over stored keys', () => {
    localStorage.setItem(KEY, JSON.stringify({ ...defaultFilters('admin'), keys: ['old'] }));
    const { result } = renderHook(() => useDockFilters('stats', 'admin', { keys: ['t1'] }));
    expect(result.current[0].keys).toEqual(['t1']);
  });

  it('G-R16: a Model preset wins over a stored Model filter; the other stored choices stay', () => {
    localStorage.setItem(KEY, JSON.stringify({ models: ['old-model'], phase: 'finished' }));
    const { result } = renderHook(() => useDockFilters('stats', 'admin', { model: 'scope-model' }));
    expect(result.current[0].models).toEqual(['scope-model']);
    expect(result.current[0].phase).toBe('finished');
    // A cleared Model filter is stored as null: the preset still wins next visit.
    localStorage.setItem(KEY, JSON.stringify({ models: null }));
    const b = renderHook(() => useDockFilters('stats', 'admin', { model: 'scope-model' }));
    expect(b.result.current[0].models).toEqual(['scope-model']);
  });

  it('ignores a corrupt or ill-typed stored value', () => {
    localStorage.setItem(KEY, '{not json');
    const a = renderHook(() => useDockFilters('stats', 'admin', {}));
    expect(a.result.current[0]).toEqual(defaultFilters('admin'));
    localStorage.setItem(KEY, JSON.stringify({ phase: 'bogus', has: ['nope', 'images'], models: 'x', q: 3 }));
    const b = renderHook(() => useDockFilters('stats', 'admin', {}));
    expect(b.result.current[0]).toEqual({ ...defaultFilters('admin'), has: ['images'] });
  });

  it('works when storage throws', () => {
    const real = globalThis.localStorage;
    const broken = {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    };
    Object.defineProperty(globalThis, 'localStorage', { value: broken, configurable: true, writable: true });
    try {
      const { result } = renderHook(() => useDockFilters('stats', 'key', {}));
      expect(result.current[0]).toEqual(defaultFilters('key'));
      act(() => result.current[1]({ ...result.current[0], q: 'y' }));
      expect(result.current[0].q).toBe('y');
    } finally {
      Object.defineProperty(globalThis, 'localStorage', { value: real, configurable: true, writable: true });
    }
  });
});

describe('search cache (Task 14 review I1)', () => {
  /** A row whose flat prompt counts its reads. */
  function counted(id: string, text: string): [RequestView, () => number] {
    let reads = 0;
    const r = { ...row(id) };
    Object.defineProperty(r, 'promptFlat', {
      get: () => {
        reads += 1;
        return text;
      },
      enumerable: true,
    });
    return [r, () => reads];
  }

  it('an unchanged row is not searched again for the same text', () => {
    const [r, reads] = counted('c1', 'needle in a haystack');
    const f = { ...defaultFilters('admin'), q: 'needle' };
    expect(applyFilters([r], f, 'admin')).toEqual([r]);
    const once = reads();
    expect(once).toBeGreaterThan(0);
    expect(applyFilters([r], f, 'admin')).toEqual([r]);
    expect(reads()).toBe(once);
    // Another needle searches again.
    expect(applyFilters([r], { ...f, q: 'hay' }, 'admin')).toEqual([r]);
    expect(reads()).toBeGreaterThan(once);
  });

  it('a changed row (a new object) is searched again', () => {
    const [r] = counted('c2', 'nothing here');
    const f = { ...defaultFilters('admin'), q: 'needle' };
    expect(applyFilters([r], f, 'admin')).toEqual([]);
    const r2 = { ...row('c2'), promptFlat: 'now a needle' };
    expect(applyFilters([r2], f, 'admin')).toEqual([r2]);
  });
});
