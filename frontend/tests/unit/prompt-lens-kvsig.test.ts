import { describe, expect, it } from 'vitest';

import { mapParts, shortId, type KvsigView } from '@/lib/prompt-lens/kvsig';

import { PROMPT, part, reqWith } from './prompt-lens-kvsig-fixtures';

const r = reqWith({
  parts: [
    part('sys', -1, -1, 'exact', { chars: 1200 }),
    part('tools', -1, -1, 'exact', { chars: 3600 }),
    part('nodes', 0, 13, 'exact', { chars: 9800, roles: { user: 5, assistant: 5, tool: 4 } }),
    part('nodes', 14, 14, 'loose', { chars: 700, roles: { assistant: 1 } }),
  ],
  end: { prompt_tokens: 15300, cached_tokens: 13000 },
  prompt: PROMPT,
});

const adminReq = reqWith({
  parts: [
    part('sys', -1, -1, 'other', { chars: 2610 }),
    part('tools', -1, -1, 'other', { chars: 6130 }),
    part('nodes', 0, 0, 'new', { chars: 5360 }),
  ],
  kvsig: { reusable_own: 0, reusable_any: 8020, edge: 'new', turn: 1, parent_request_id: null },
  end: { prompt_tokens: 14100, cached_tokens: 8740, cache: { cache_outcome: 'cold' } },
});

// ---------------------------------------------------------------------------

describe('mapParts', () => {
  it('key scope never shows other, uses own reuse span', () => {
    const v = mapParts(
      reqWith({ parts: [part('nodes', 0, 13, 'exact'), part('nodes', 14, 14, 'new')], end: { own_reused_tokens: 26400 } }),
      'key',
    );
    expect(v.parts.some((p) => p.state === 'other')).toBe(false);
    expect(v.cachedSpanLabel).toMatch(/^reused from your earlier requests/);
    expect(v.cachedSpanTok).toBe(26400);
    // G-R11: a key's new part is plain "new", never "not attributed".
    expect(v.parts[1].vsEarlier).toBe('new');
    expect(JSON.stringify(v)).not.toMatch(/attribut|another key|other key/i);
  });

  it('key scope rewrites a stray other part to new and merges it with its new neighbours', () => {
    const withOther = mapParts(
      reqWith({
        parts: [part('nodes', 0, 13, 'exact'), part('nodes', 14, 14, 'other'), part('nodes', 15, 15, 'new', { exact8: 'ffff0000', loose8: 'eeee1111' })],
        kvsig: { reusable_any: 99999 },
        end: { own_reused_tokens: 5000, cached_tokens: 9000, cache: { cache_outcome: 'hit', divergence_cause: 'harness_rewrite' } },
      }),
      'key',
    );
    const noOther = mapParts(
      reqWith({
        parts: [part('nodes', 0, 13, 'exact'), part('nodes', 14, 15, 'new', { chars: 2000, roles: { user: 2 }, exact8: 'ffff0000', loose8: 'eeee1111' })],
        end: { own_reused_tokens: 5000 },
      }),
      'key',
    );
    expect(withOther).toEqual(noOther);
  });

  it('key scope ignores admin-only figures even if present', () => {
    const v = mapParts(adminReq, 'key');
    expect(v.otherSpanTok).toBeNull();
    expect(v.cachedSpanTok).toBeNull();
    expect(v.footer.map(([k]) => k)).not.toContain('engine outcome');
    expect(v.parts.map((p) => p.state)).toEqual(['new', 'new', 'new']);
  });

  it('admin labels the other-key span', () => {
    const v = mapParts(adminReq, 'admin');
    expect(v.otherSpanTok).toBe(8020);
    expect(v.otherSpanLabel).toBe('prefix cached by another key (any-key index) 8,020 tok');
    expect(v.parts[0].vsEarlier).toBe('cached by another key');
  });

  it('admin has no other span when any-key reuse is not above own', () => {
    expect(mapParts(reqWith({ parts: [part('sys', -1, -1, 'exact')], kvsig: { reusable_own: 500, reusable_any: 500 } }), 'admin').otherSpanTok).toBeNull();
  });

  it('token widths sum to prompt_tokens', () => {
    const v = mapParts(r, 'admin');
    expect(v.totalTok).toBe(15300);
    expect(v.parts.reduce((a, p) => a + p.tok, 0)).toBe(15300);
    // chars x prompt_tokens / sum(chars), rounded
    expect(v.parts[0].tok).toBe(Math.round((1200 * 15300) / 15300));
  });

  it('estimates tokens as chars / 4 before request_end', () => {
    const v = mapParts(reqWith({ parts: [part('sys', -1, -1, 'exact', { chars: 400 }), part('nodes', 0, 0, 'new', { chars: 401 })] }), 'admin');
    expect(v.totalTok).toBe(200);
    expect(v.parts.reduce((a, p) => a + p.tok, 0)).toBe(200);
    expect(v.parts.every((p) => p.cache === 'pending')).toBe(true);
    expect(v.cachedSpanTok).toBeNull();
  });

  it('labels are 1-based runs', () => expect(mapParts(r, 'admin').parts.map((p) => p.label)).toEqual(['sys', 'tools', 'nodes 1–14', 'node 15']));

  it('vs earlier names the previous turn on a continue edge', () => {
    const v = mapParts(r, 'admin');
    expect(v.parts.map((p) => p.vsEarlier)).toEqual([
      'same as turn 16',
      'same as turn 16',
      'continues turn 16',
      'loose same, exact differs: the client dropped the reasoning',
    ]);
  });

  it('vs earlier without a known turn', () => {
    const v = mapParts(
      reqWith({ parts: [part('sys', -1, -1, 'exact'), part('nodes', 0, 1, 'exact'), part('nodes', 2, 2, 'loose')], kvsig: { edge: 'fork' } }),
      'admin',
    );
    expect(v.parts.map((p) => p.vsEarlier)).toEqual(['same as an earlier request', 'matches an earlier request', 'loose same, exact differs']);
  });

  it('admin cache column follows cached_tokens; key follows own reuse', () => {
    const a = mapParts(r, 'admin');
    expect(a.parts.map((p) => p.cache)).toEqual(['cached', 'cached', 'cached', 'computed']);
    expect(a.cachedSpanLabel).toBe('engine cached 13,000 tok');
    const k = mapParts(reqWith({ parts: r.kvsig!.parts, end: { prompt_tokens: 15300, own_reused_tokens: 4800 }, prompt: PROMPT }), 'key');
    expect(k.parts.map((p) => p.cache)).toEqual(['reused', 'reused', 'unknown', 'unknown']);
    expect(k.cachedSpanLabel).toBe('reused from your earlier requests 4,800 tok');
  });

  it('admin without an engine figure shows unknown cache', () => {
    const v = mapParts(reqWith({ parts: [part('sys', -1, -1, 'exact')], end: { cached_tokens: null } }), 'admin');
    expect(v.parts[0].cache).toBe('unknown');
    expect(v.cachedSpanTok).toBeNull();
  });

  it('covers come from the structured prompt', () => {
    const v = mapParts(
      reqWith({
        parts: [
          part('sys', -1, -1, 'exact', { roles: { system: 1, developer: 1 } }),
          part('tools', -1, -1, 'exact'),
          part('nodes', 0, 13, 'exact', { roles: { user: 5, assistant: 5, tool: 4 } }),
          part('nodes', 14, 14, 'loose', { roles: { assistant: 1 } }),
          part('nodes', 15, 15, 'new', { roles: { tool: 1 } }),
          part('nodes', 16, 16, 'new', { roles: { user: 1 } }),
        ],
        prompt: PROMPT,
      }),
      'admin',
    );
    expect(v.parts.map((p) => p.covers)).toEqual([
      'system prompt (2 messages: system + developer)',
      '24 tools (read_file, edit, grep, bash, …)',
      'messages 1–14: 5 user, 5 assistant, 4 tool results',
      'assistant: “I read the dock. It already …”',
      'tool result: read_file (call_8f2a)',
      'user: “now wire the dock into /key/stats …”',
    ]);
  });

  it('footer: edge, conv, loose profile, admin outcome', () => {
    const a = mapParts(
      reqWith({ parts: r.kvsig!.parts, end: { cached_tokens: 1, cache: { cache_outcome: 'hit', divergence_cause: 'harness_rewrite' } } }),
      'admin',
    );
    expect(a.footer).toEqual([
      ['edge', 'continue from turn 16 (request 7f3c…)'],
      ['conv', 'e81d·0b52'],
      ['profile', 'loose'],
      ['engine outcome', 'hit · cause harness rewrite'],
    ]);
    const k = mapParts(reqWith({ parts: r.kvsig!.parts, end: { cache: { cache_outcome: 'hit' } } }), 'key');
    expect(k.footer.map(([key]) => key)).toEqual(['edge', 'conv', 'profile']);
    expect(JSON.stringify(a)).not.toMatch(/loose\/1|cobs2/);
  });

  it('unavailable kvsig yields an empty state per reason', () => {
    const texts = (['not_chat', 'no_key', 'error', 'over_cap'] as const).map(
      (reason) => mapParts(reqWith({ kvsig: { available: false, reason, parts: [] } }), 'admin').empty,
    );
    expect(texts.every((t) => typeof t === 'string' && t.length > 0)).toBe(true);
    expect(new Set(texts).size).toBe(4);
    expect(texts[0]).toMatch(/not a chat request/);
  });

  it('a queued request without kvsig says the map comes after admission', () => {
    const v: KvsigView = mapParts(reqWith({ kvsig: null, phase: 'queued' }), 'key');
    expect(v.empty).toMatch(/prompt map after admission/i);
    expect(v.parts).toEqual([]);
  });

  it('short ids are 4·4', () => {
    expect(shortId('e3a19f02')).toBe('e3a1·9f02');
    expect(shortId('')).toBe('—');
  });
});
