// Prompt Lens pop-up search and copy (spec 2026-11-22 §3.3, owner answer Q13:
// plain text, case-insensitive; no regex).
//
// Search must stay responsive over a whole section (up to ~2M chars): one
// case fold and an indexOf walk, capped at MAX_HITS; a growing live section
// is searched only over what was appended (`findHitsFrom`). The pop-up then
// highlights only the blocks on screen.

import type { PromptMsg } from './events';
import type { SectionKind } from './section-marks';

/** Hits past this are not counted ("100,000+"). */
export const MAX_HITS = 100_000;

/** The pop-up's text blocks close at the first newline past this many chars
 *  (TailText's rule, so both views break the text alike). */
export const BLOCK_CHARS = 4096;

/** `text` lower-cased one UTF-16 unit for one, so offsets into the fold are
 *  offsets into `text`. A char whose lower case has another length ("İ")
 *  keeps its own case. */
export function foldCase(text: string): string {
  const low = text.toLowerCase();
  if (low.length === text.length) return low;
  let out = '';
  for (let k = 0; k < text.length; k++) {
    const c = text[k];
    const l = c.toLowerCase();
    out += l.length === 1 ? l : c;
  }
  return out;
}

/** Plain, case-insensitive, non-overlapping hits of `q` in `text`, as
 *  [start, end) pairs, at most `max`. */
export function findHits(text: string, q: string, max = MAX_HITS): [number, number][] {
  return findHitsFrom(text, q, [], 0, max);
}

/** `prev` (the hits of an earlier, shorter `text` of `prevLen` chars) plus the
 *  hits in what was appended since. A hit can straddle the old end, so the
 *  walk starts `q.length - 1` chars before it (and after the last hit). */
export function findHitsFrom(
  text: string,
  q: string,
  prev: [number, number][],
  prevLen: number,
  max = MAX_HITS,
): [number, number][] {
  if (!q) return [];
  const out = prev.slice(0, max);
  if (out.length >= max) return out;
  const fq = foldCase(q);
  const lastEnd = out.length ? out[out.length - 1][1] : 0;
  const from = Math.max(lastEnd, prevLen - fq.length + 1, 0);
  if (text.length - from < fq.length) return out;
  // Fold only the part searched (a 2M-char section is not folded per delta).
  const hay = foldCase(text.slice(from));
  let at = hay.indexOf(fq);
  while (at !== -1 && out.length < max) {
    out.push([from + at, from + at + fq.length]);
    at = hay.indexOf(fq, at + fq.length);
  }
  return out;
}

/** Block boundaries of `text`: [0, b1, ..., text.length]. A block closes at
 *  the first newline once it holds `chunk` chars; a long line stays whole. */
export function splitBlocks(text: string, chunk = BLOCK_CHARS): number[] {
  const out = [0];
  let start = 0;
  while (text.length - start > chunk) {
    const nl = text.indexOf('\n', start + chunk - 1);
    if (nl === -1 || nl + 1 >= text.length) break;
    start = nl + 1;
    out.push(start);
  }
  out.push(text.length);
  return out;
}

/** What a pop-up copies from. */
export interface CopyView {
  text?: string;
  calls?: { name: string; args: string }[];
  messages?: PromptMsg[];
}

/** The plain-text header of one tool result or prompt message. */
export function resultHead(m: PromptMsg): string {
  const isResult = m.role === 'tool' || m.role === 'function';
  const name = isResult ? m.tool_name || 'tool result' : m.role;
  return [name, m.tool_call_id, `msg ${m.i}`].filter(Boolean).join(' · ');
}

/** The pop-up's Copy: tool calls as a JSON array of {name, arguments}; every
 *  other kind as its plain text (tool results and the prompt one block per
 *  message, a header line then the text). */
export function copyText(kind: Exclude<SectionKind, 'parts'>, view: CopyView): string {
  if (kind === 'tool_calls') {
    return JSON.stringify(
      (view.calls ?? []).map(({ name, args }) => ({ name, arguments: args })),
      null,
      2,
    );
  }
  if (kind === 'tool_results' || kind === 'prompt') {
    if (view.messages) return view.messages.map((m) => `${resultHead(m)}\n${m.text}`).join('\n\n');
  }
  return view.text ?? '';
}
