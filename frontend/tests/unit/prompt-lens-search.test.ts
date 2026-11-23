import { describe, expect, it } from 'vitest';

import { copyText, findHits, findHitsFrom, foldCase, MAX_HITS, splitBlocks } from '@/lib/prompt-lens/search';
import { prettyJson } from '@/lib/prompt-lens/pretty-json';

describe('findHits (Q13: plain, case-insensitive)', () => {
  it('findHits is plain and case-insensitive', () => expect(findHits('a.b A.B', 'a.b')).toEqual([[0, 3], [4, 7]]));

  it('treats regex characters as plain text', () => {
    expect(findHits('x(1)+ y(1)+', '(1)+')).toEqual([[1, 5], [7, 11]]);
    expect(findHits('aaa', '.')).toEqual([]);
  });

  it('is non-overlapping', () => expect(findHits('aaaa', 'aa')).toEqual([[0, 2], [2, 4]]));

  it('an empty query finds nothing', () => expect(findHits('abc', '')).toEqual([]));

  it('keeps offsets right where lower-casing changes a char length', () => {
    // "İ" lower-cases to two code units; the fold must stay one per char.
    const t = 'İx Ab ab';
    expect(foldCase(t).length).toBe(t.length);
    expect(findHits(t, 'ab')).toEqual([[3, 5], [6, 8]]);
  });

  it('stops at the cap', () => {
    const hits = findHits('a'.repeat(50), 'a', 10);
    expect(hits).toHaveLength(10);
    expect(MAX_HITS).toBeGreaterThanOrEqual(10_000);
  });

  it('findHitsFrom extends hits over appended text, with a match across the old end', () => {
    const a = 'one dock two do';
    const b = a + 'ck three dock';
    const first = findHits(a, 'dock');
    expect(first).toEqual([[4, 8]]);
    const more = findHitsFrom(b, 'dock', first, a.length);
    expect(more).toEqual(findHits(b, 'dock'));
  });

  it('stays fast over 2M chars', () => {
    const big = 'lorem ipsum dolor sit amet, '.repeat(75_000); // 2.1M chars
    const t0 = performance.now();
    const hits = findHits(big, 'DOLOR');
    const ms = performance.now() - t0;
    expect(hits).toHaveLength(Math.min(75_000, MAX_HITS));
    expect(ms).toBeLessThan(500);
  });
});

describe('copyText', () => {
  it('copy of tool calls is a JSON array of name/arguments', () => {
    const v = { calls: [{ name: 'bash', args: '{"cmd":"ls"}' }] };
    expect(JSON.parse(copyText('tool_calls', v))).toEqual([{ name: 'bash', arguments: '{"cmd":"ls"}' }]);
  });

  it('copies other kinds as their plain text', () => {
    expect(copyText('thinking', { text: 'hm <b>x</b>' })).toBe('hm <b>x</b>');
    expect(copyText('reply', { text: '# hi' })).toBe('# hi');
  });

  it('copies tool results as one plain block per result', () => {
    const out = copyText('tool_results', {
      messages: [
        { i: 3, role: 'tool', chars: 2, text: 'ok', elided: false, tool_name: 'bash', tool_call_id: 'c1' },
        { i: 5, role: 'tool', chars: 3, text: 'two', elided: false },
      ],
    });
    expect(out).toBe('bash · c1 · msg 3\nok\n\ntool result · msg 5\ntwo');
  });
});

describe('splitBlocks', () => {
  it('cuts after a newline once a block holds the chunk size, and covers the text', () => {
    const t = 'a'.repeat(10) + '\n' + 'b'.repeat(3) + '\n' + 'c'.repeat(12);
    const b = splitBlocks(t, 8);
    expect(b[0]).toBe(0);
    expect(b[b.length - 1]).toBe(t.length);
    // every inner cut sits right after a newline
    for (const k of b.slice(1, -1)) expect(t[k - 1]).toBe('\n');
    expect(b).toEqual([0, 11, t.length]);
  });

  it('an empty text is one empty block', () => expect(splitBlocks('', 8)).toEqual([0, 0]));
});

describe('prettyJson', () => {
  it('pretty-prints and marks keys, strings and numbers', () => {
    const p = prettyJson('{"cmd":"ls","n":2,"ok":true}');
    expect(p).not.toBeNull();
    expect(p!.text).toBe('{\n  "cmd": "ls",\n  "n": 2,\n  "ok": true\n}');
    const kinds = p!.tokens.map(([s, e, k]) => [p!.text.slice(s, e), k]);
    expect(kinds).toEqual([
      ['"cmd"', 'key'],
      ['"ls"', 'str'],
      ['"n"', 'key'],
      ['2', 'num'],
      ['"ok"', 'key'],
      ['true', 'lit'],
    ]);
  });

  it('returns null for partial or invalid JSON (shown raw while streaming)', () => {
    expect(prettyJson('{"cmd": "l')).toBeNull();
    expect(prettyJson('not json')).toBeNull();
    expect(prettyJson('')).toBeNull();
  });

  it('handles escaped quotes inside strings', () => {
    const p = prettyJson('{"a":"say \\"hi\\": now"}');
    const kinds = p!.tokens.map(([s, e, k]) => [p!.text.slice(s, e), k]);
    expect(kinds).toEqual([
      ['"a"', 'key'],
      ['"say \\"hi\\": now"', 'str'],
    ]);
  });
});
