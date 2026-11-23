// Pretty-printed JSON for Prompt Lens tool-call arguments, with token ranges
// for colouring (keys, strings, numbers, true/false/null). Arguments stream
// in pieces, so a call is usually partial JSON until it ends: `prettyJson`
// returns null for anything that does not parse, and the caller shows the
// raw text.

export type JsonTokenKind = 'key' | 'str' | 'num' | 'lit';
export type JsonToken = [start: number, end: number, kind: JsonTokenKind];

export interface PrettyJson {
  text: string;
  tokens: JsonToken[];
}

const TOKEN = /"(?:[^"\\]|\\.)*"(\s*:)?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\btrue\b|\bfalse\b|\bnull\b/g;

/** `raw` pretty-printed (2 spaces) with its tokens, or null when it is not
 *  JSON (yet). */
export function prettyJson(raw: string): PrettyJson | null {
  if (!raw || !raw.trim()) return null;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return null;
  }
  const text = JSON.stringify(v, null, 2);
  if (typeof text !== 'string') return null;
  const tokens: JsonToken[] = [];
  TOKEN.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = TOKEN.exec(text)) !== null) {
    const s = m[0];
    if (s[0] === '"') {
      if (m[1] !== undefined) tokens.push([m.index, m.index + s.length - m[1].length, 'key']);
      else tokens.push([m.index, m.index + s.length, 'str']);
    } else if (s === 'true' || s === 'false' || s === 'null') {
      tokens.push([m.index, m.index + s.length, 'lit']);
    } else {
      tokens.push([m.index, m.index + s.length, 'num']);
    }
  }
  return { text, tokens };
}
