// Prompt Lens prompt map: the per-request parts event (`kvsig`, built by
// app/proxy/prompt_lens_kvsig.py) turned into what the map and the parts table
// draw. Pure: one RequestView and a scope in, display rows out.
//
// Scope rules (spec 2026-11-22 section 4, rulings G-R7, G-R11, Q3):
//   * key: a part another key cached reads "new" exactly like any new part.
//     The server already sends the key holder parts built as if no other key
//     existed; a stray `other` part is still rewritten to `new` and merged with
//     its `new` node neighbours (the server's backstop), and the admin-only
//     figures (`reusable_any`, `cached_tokens`, `cache`) are ignored even if
//     present, so a key render never depends on another key's traffic. The
//     cached span is "reused from your earlier requests" (own_reused_tokens);
//     the raw engine figure is never shown.
//   * admin: `other` reads "cached by another key"; the engine-cached span is
//     `cached_tokens`, plus a dashed any-key span when it reaches past own.
//   * the loose form is labelled "loose", never by its profile name.

import type { KvsigEvent, KvsigPart, KvsigReason, KvsigState, PromptMsg, PromptParts } from './events';
import type { RequestView } from './store';

export type KvsigScope = 'key' | 'admin';

/** cached / computed: admin, from the engine's cached_tokens. reused: key,
 *  inside own_reused_tokens. unknown: outside what this scope can know (shown
 *  as a dash). pending: before request_end. */
export type PartCache = 'cached' | 'computed' | 'reused' | 'unknown' | 'pending';

export interface MapPart extends KvsigPart {
  label: string;
  covers: string;
  tok: number;
  vsEarlier: string;
  cache: PartCache;
}

export interface KvsigView {
  scope: KvsigScope;
  /** Set when there is no map to draw: the empty state's text. */
  empty: string | null;
  parts: MapPart[];
  totalTok: number;
  cachedSpanTok: number | null;
  cachedSpanLabel: string;
  /** Admin only. */
  otherSpanTok: number | null;
  otherSpanLabel: string;
  footer: [string, string][];
}

const EMPTY_TEXT: Record<KvsigReason, string> = {
  not_chat: 'No prompt map: this is not a chat request.',
  no_key: 'No prompt map: the request carried no key, so it has no earlier requests to match.',
  error: 'No prompt map: it could not be built for this request.',
  over_cap: 'No prompt map: the prompt is too large to map.',
};

const fmt = (n: number) => n.toLocaleString('en-US');

/** An 8-hex short id as `xxxx·xxxx`. */
export function shortId(hex8: string | null | undefined): string {
  if (!hex8) return '—';
  return hex8.length > 4 ? `${hex8.slice(0, 4)}·${hex8.slice(4, 8)}` : hex8;
}

/** The cache column's text. */
export function cacheText(c: PartCache): string {
  return c === 'unknown' ? '—' : c;
}

// ---------------------------------------------------------------------------

function mergeRoles(a: Record<string, number>, b: Record<string, number>): Record<string, number> {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) out[k] = (out[k] ?? 0) + v;
  return out;
}

/** The key holder's parts: `other` reads `new`, and adjacent `new` node runs
 *  merge (a run's ids are its last node's), as prompt_lens_scope._backstop does. */
function keyParts(parts: KvsigPart[]): KvsigPart[] {
  const out: KvsigPart[] = [];
  for (const p0 of parts) {
    const p = p0.state === 'other' ? { ...p0, state: 'new' as KvsigState } : p0;
    const prev = out[out.length - 1];
    if (prev && prev.p === 'nodes' && p.p === 'nodes' && prev.state === 'new' && p.state === 'new') {
      out[out.length - 1] = { ...p, from: prev.from, chars: prev.chars + p.chars, roles: mergeRoles(prev.roles, p.roles) };
    } else {
      out.push(p);
    }
  }
  return out;
}

/** Split `total` over `weights` in proportion, rounding so the sum is exact
 *  (largest remainder). */
function apportion(weights: number[], total: number): number[] {
  const sum = weights.reduce((a, w) => a + Math.max(0, w), 0);
  if (sum <= 0 || total <= 0) return weights.map(() => 0);
  const raw = weights.map((w) => (Math.max(0, w) * total) / sum);
  const out = raw.map(Math.floor);
  let left = total - out.reduce((a, b) => a + b, 0);
  const order = raw.map((x, i) => [x - Math.floor(x), i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (let k = 0; left > 0 && k < order.length; k++, left--) out[order[k][1]] += 1;
  return out;
}

function partLabel(p: KvsigPart): string {
  if (p.p !== 'nodes') return p.p;
  return p.from === p.to ? `node ${p.from + 1}` : `nodes ${p.from + 1}–${p.to + 1}`;
}

const ROLE_WORD: Record<string, [string, string]> = {
  user: ['user', 'user'],
  assistant: ['assistant', 'assistant'],
  tool: ['tool result', 'tool results'],
  function: ['function result', 'function results'],
  system: ['system', 'system'],
  developer: ['developer', 'developer'],
  unknown: ['unknown role', 'unknown roles'],
};
const ROLE_ORDER = ['system', 'developer', 'user', 'assistant', 'tool', 'function', 'unknown'];

function roleList(roles: Record<string, number>): string {
  return Object.entries(roles)
    .filter(([, n]) => n > 0)
    .sort((a, b) => ROLE_ORDER.indexOf(a[0]) - ROLE_ORDER.indexOf(b[0]))
    .map(([r, n]) => {
      const [one, many] = ROLE_WORD[r] ?? ROLE_WORD.unknown;
      return `${n} ${n === 1 ? one : many}`;
    })
    .join(', ');
}

function firstWords(text: string, words = 6): string {
  const all = text.trim().split(/\s+/).filter(Boolean);
  const head = all.slice(0, words).join(' ');
  const cut = head.length > 48 ? `${head.slice(0, 48)} …` : all.length > words ? `${head} …` : head;
  return `“${cut}”`;
}

const OLDER_ROLE: Record<string, string> = { u: 'user', a: 'assistant', t: 'tool', f: 'function' };

function messageCovers(i: number, roles: Record<string, number>, prompt: PromptParts | null): string {
  const m: PromptMsg | undefined = prompt?.messages.find((x) => x.i === i);
  if (m) {
    if (m.role === 'tool' || m.role === 'function') {
      const name = m.tool_name ?? (m.role === 'tool' ? 'tool' : 'function');
      return `${m.role === 'tool' ? 'tool' : 'function'} result: ${name}${m.tool_call_id ? ` (${m.tool_call_id})` : ''}`;
    }
    if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0 && !m.text.trim()) {
      return `assistant: tool call ${m.tool_calls.map((c) => c.name).join(', ')}`;
    }
    return m.text.trim() ? `${m.role}: ${firstWords(m.text)}` : `${m.role} message`;
  }
  const older = prompt?.older;
  const k = older ? i - older.from : -1;
  const role = older && k >= 0 && k < older.roles.length ? OLDER_ROLE[older.roles[k]] : Object.keys(roles)[0];
  const [one] = ROLE_WORD[role ?? 'unknown'] ?? ROLE_WORD.unknown;
  return one === 'user' || one === 'assistant' || one === 'system' || one === 'developer' ? `${one} message` : one;
}

function covers(p: KvsigPart, prompt: PromptParts | null): string {
  if (p.p === 'sys') {
    const n = Object.values(p.roles).reduce((a, b) => a + b, 0);
    if (n <= 0) return 'system prompt';
    if (n === 1) return 'system prompt (1 message)';
    const kinds = ROLE_ORDER.filter((r) => (p.roles[r] ?? 0) > 0);
    return `system prompt (${n} messages: ${kinds.join(' + ')})`;
  }
  if (p.p === 'tools') {
    const t = prompt?.tools;
    if (!t) return 'tool definitions';
    const names = t.names.slice(0, 4).join(', ');
    const more = t.count > 4 ? ', …' : '';
    return `${t.count} ${t.count === 1 ? 'tool' : 'tools'}${names ? ` (${names}${more})` : ''}`;
  }
  if (p.from === p.to) return messageCovers(p.from, p.roles, prompt);
  const list = roleList(p.roles);
  return `messages ${p.from + 1}–${p.to + 1}${list ? `: ${list}` : ''}`;
}

function isAssistantMessage(p: KvsigPart): boolean {
  return p.p === 'nodes' && p.from === p.to && (p.roles.assistant ?? 0) === 1 && Object.keys(p.roles).length === 1;
}

function vsEarlier(p: KvsigPart, k: KvsigEvent, scope: KvsigScope): string {
  const prevTurn = k.edge === 'continue' && typeof k.turn === 'number' && k.turn > 1 ? k.turn - 1 : null;
  switch (p.state) {
    case 'exact':
      if (p.p !== 'nodes') return prevTurn !== null ? `same as turn ${prevTurn}` : 'same as an earlier request';
      return prevTurn !== null ? `continues turn ${prevTurn}` : 'matches an earlier request';
    case 'loose':
      return isAssistantMessage(p) ? 'loose same, exact differs: the client dropped the reasoning' : 'loose same, exact differs';
    case 'other':
      return scope === 'admin' ? 'cached by another key' : 'new';
    default:
      return 'new';
  }
}

function shortReq(id: string | null | undefined): string {
  return id ? ` (request ${id.slice(0, 4)}…)` : '';
}

function edgeText(k: KvsigEvent): string | null {
  const prevTurn = typeof k.turn === 'number' && k.turn > 1 ? k.turn - 1 : null;
  switch (k.edge) {
    case 'new':
      return 'new';
    case 'continue':
      return `continue from ${prevTurn !== null ? `turn ${prevTurn}` : 'an earlier turn'}${shortReq(k.parent_request_id)}`;
    case 'fork':
      return `fork of an earlier turn${shortReq(k.parent_request_id)}`;
    case 'compacted_from':
      return `compacted from an earlier request${shortReq(k.parent_request_id)}`;
    case 'subagent':
      return `subagent of an earlier request${shortReq(k.parent_request_id)}`;
    default:
      return null;
  }
}

const words = (s: string) => s.replace(/_/g, ' ');

function emptyView(scope: KvsigScope, text: string): KvsigView {
  return {
    scope,
    empty: text,
    parts: [],
    totalTok: 0,
    cachedSpanTok: null,
    cachedSpanLabel: '',
    otherSpanTok: null,
    otherSpanLabel: '',
    footer: [],
  };
}

/** The prompt map of one request, as `scope` may see it. */
export function mapParts(r: RequestView, scope: KvsigScope): KvsigView {
  const k = r.kvsig;
  if (!k) {
    if (r.phase === 'queued' && r.endedTs === null) return emptyView(scope, 'Queued: prompt map after admission.');
    return emptyView(scope, 'No prompt map for this request.');
  }
  if (!k.available) return emptyView(scope, EMPTY_TEXT[k.reason ?? 'error'] ?? EMPTY_TEXT.error);
  const raw = scope === 'key' ? keyParts(k.parts) : k.parts;
  if (raw.length === 0) return emptyView(scope, 'No prompt map for this request.');

  const end = r.end;
  const sumChars = raw.reduce((a, p) => a + Math.max(0, p.chars), 0);
  const totalTok = end && end.prompt_tokens > 0 ? end.prompt_tokens : Math.round(sumChars / 4);
  const toks = apportion(
    raw.map((p) => p.chars),
    totalTok,
  );

  let cachedSpanTok: number | null = null;
  if (end) {
    const v = scope === 'admin' ? end.cached_tokens : end.own_reused_tokens;
    cachedSpanTok = typeof v === 'number' && v > 0 ? Math.min(v, totalTok) : null;
  }
  const known = end ? (scope === 'admin' ? typeof end.cached_tokens === 'number' : true) : false;

  let at = 0;
  const parts: MapPart[] = raw.map((p, i) => {
    const tok = toks[i];
    const mid = at + tok / 2;
    at += tok;
    let cache: PartCache;
    if (!end) cache = 'pending';
    else if (!known) cache = 'unknown';
    else if (scope === 'admin') cache = cachedSpanTok !== null && mid < cachedSpanTok ? 'cached' : 'computed';
    else cache = cachedSpanTok !== null && mid < cachedSpanTok ? 'reused' : 'unknown';
    return { ...p, label: partLabel(p), covers: covers(p, r.prompt), tok, vsEarlier: vsEarlier(p, k, scope), cache };
  });

  let otherSpanTok: number | null = null;
  if (scope === 'admin' && typeof k.reusable_any === 'number' && k.reusable_any > (k.reusable_own ?? 0)) {
    otherSpanTok = Math.min(k.reusable_any, totalTok);
  }

  const footer: [string, string][] = [];
  const edge = edgeText(k);
  if (edge) footer.push(['edge', edge]);
  if (k.conv8) footer.push(['conv', shortId(k.conv8)]);
  footer.push(['profile', 'loose']);
  if (scope === 'admin') {
    const cache = end?.cache;
    const outcome = cache && typeof cache.cache_outcome === 'string' ? words(cache.cache_outcome) : null;
    const cause = cache && typeof cache.divergence_cause === 'string' ? words(cache.divergence_cause) : null;
    if (!end) footer.push(['engine outcome', 'pending']);
    else if (outcome) footer.push(['engine outcome', cause ? `${outcome} · cause ${cause}` : outcome]);
  }

  return {
    scope,
    empty: null,
    parts,
    totalTok,
    cachedSpanTok,
    cachedSpanLabel:
      cachedSpanTok === null
        ? ''
        : scope === 'admin'
          ? `engine cached ${fmt(cachedSpanTok)} tok`
          : `reused from your earlier requests ${fmt(cachedSpanTok)} tok`,
    otherSpanTok,
    otherSpanLabel: otherSpanTok === null ? '' : `prefix cached by another key (any-key index) ${fmt(otherSpanTok)} tok`,
    footer,
  };
}
