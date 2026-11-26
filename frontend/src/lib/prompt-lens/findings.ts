// Prompt Lens request findings (analysis spec 16.1, Task 1.8): what
//
//   admin:      GET /api/admin/prompt-lens/requests/{req_id}/findings
//   key holder: GET /api/me/prompt-lens/requests/{req_id}/findings
//
// answer, and the copy the request view shows for each finding.
//
// A finding is codes and numbers only (A-10): turn numbers, counts, a keyed
// tool hash, a cause code. `ruleCopy` turns it into words with the catalogue's
// copy (app/lens/catalogue.py, spec tables 11.1-11.3), mirrored below verbatim
// and pinned by tests/component/prompt-lens-findings.test.tsx. Tool names are
// joined here only, from the answer's `names` (the caller's own names).
//
// Conventions carried from the server:
//   * `counts.info === 1` is a notice (C-01 "rarely used", C-03 with the head
//     reused, R-03 at one stop, R-08): shown muted, never as a fix;
//   * E-01's `counts.class` indexes SPAN_CLASSES (app/lens/probes.py);
//   * R-04's `counts.code` indexes CALL_CODES (app/lens/reply_verdicts.py);
//   * C-06's fix is "as E-01" and R-02's "see C-05": resolved to that rule's fix.

import { useEffect, useState } from 'react';

import { authFetchJSON } from '@/lib/auth-fetch';
import { sessionsFetcher } from '@/lib/sessions/client';

import type { LensScope } from './stream';

export type LensLayer = 'E' | 'C' | 'R';

export interface LensFinding {
  rule: string;
  turns: number[];
  counts: Record<string, number>;
  tool: string | null;
  cause: string | null;
  saving: { tokens: number; prefill_s: number | null; decode_s: number | null; money: number | null };
  confidence: 'high' | 'medium' | 'low';
  muted?: boolean;
  /** Who marked the rule won't fix: "key" (the key holder) or "admin". */
  muted_by?: string | null;
  /** "sys" | "tools" | "node:<i>" | "call:<k>", or null (the strip only). */
  part: string | null;
  /** R-09 only: the guard rule ids and where they matched, from the guard's data. */
  guard?: { rules: string[]; where: string[] } | null;
}

export interface RequestFindings {
  turn: number | null;
  findings: LensFinding[];
  names: Record<string, string>;
}

/** app/lens/probes.py SPAN_CLASSES, in order (append-only). */
export const SPAN_CLASSES = [
  'date',
  'time',
  'path',
  'uuid_or_hex_id',
  'version',
  'number',
  'url',
  'provider_or_model_id',
  'user_name_or_email',
  'other',
] as const;

/** app/lens/reply_verdicts.py CALL_CODES, in order. */
export const CALL_CODES = ['ok', 'json', 'unknown_tool', 'missing_arg', 'type', 'leaked_markup'] as const;

/** The catalogue's message and fix per rule (app/lens/catalogue.py), verbatim. */
export const CATALOGUE: Record<string, { message: string; fix: string }> = {
  'E-01': {
    message:
      'System prompt line {n} changes every {turn/session} ({class}); the {m} lines after it are recomputed',
    fix: 'make the value stable (a day, not a time; a fixed path) or move it to the newest user message; re-check answers (A-6)',
  },
  'E-02': {
    message: 'The same tools were sent in a different order (or with re-ordered JSON keys)',
    fix: 'sort tools; serialise schemas deterministically',
  },
  'E-03': {
    message: 'Tool {name/#k} was added, removed or edited mid-list; everything after it is recomputed',
    fix: 'append new tools at the end; load deferred tools last',
  },
  'E-04': {
    message: 'Message {i}, sent before, came back different ({probe})',
    fix: 'append-only history; do not re-serialise old messages',
  },
  'E-05': {
    message: 'Reasoning is kept on some turns and dropped on others',
    fix: 'keep or drop it the same way every turn; fix the thinking setting per session',
  },
  'E-06': {
    message: 'Anthropic caches only at breakpoints; this session has none',
    fix: 'add `cache_control` marks or turn on automatic marks (#341)',
  },
  'E-07': {
    message: 'Compaction also changed the system prompt or tools, so nothing was reused after it',
    fix: 'keep sys and tools byte-identical, append the summary',
  },
  'E-08': {
    message: 'Your prompt was stable; the cache expired during pauses of {p}',
    fix: 'keep pauses inside the cache lifetime or expect a cold prefix',
  },
  'E-09': {
    message: 'An image or attachment in message {i} changed or moved',
    fix: 'stable placement and encoding',
  },
  'C-01': {
    message: '{k} of {n} tools were never called; their definitions add {t} tokens to every turn',
    fix: "drop them for this app, or load them on demand; LM Warden's prompt diet can strip them for this key",
  },
  'C-02': {
    message: 'The same {family} result ({t} tokens) was sent {k} times',
    fix: 'keep one copy; tell the model the file is already in context; cache reads in the harness',
  },
  'C-03': {
    message: 'The system prompt and tools are {s} % of every request',
    fix: 'shorten, or make sure it is cached (if E is high this is info: it costs context room and first-turn time, not money)',
  },
  'C-04': {
    message: 'Tool results are {s} % of the context; the largest ({family}) is {t} tokens',
    fix: 'truncate or summarise large results; paginate reads',
  },
  'C-05': {
    message:
      'Context grows {g} tokens per turn; the window fills by turn {n} / your harness compacts at {p} % of the window',
    fix: 'trim tool results, compact later or with a cache-aware summary',
  },
  'C-06': { message: 'Line {n} of {m} changes; the lines after it are stable', fix: 'as E-01' },
  'C-07': { message: 'Reminder and environment blocks are {s} % of the prompt', fix: 'shorten or stop repeating them' },
  'C-08': {
    message: "Tool {name/#k}'s definition is {t} tokens, {x} × the others",
    fix: 'shorten the description, move long docs out',
  },
  'C-09': { message: '{n} tools; OpenAI accepts at most 128', fix: 'split or load on demand' },
  'R-01': {
    message: "{k} replies were cut at max_tokens = {m}; the session's p90 reply is {p}",
    fix: 'raise the limit for this step, or lower the thinking budget',
  },
  'R-02': { message: 'The prompt filled the context window at turn {n}', fix: 'see C-05' },
  'R-03': {
    message: 'The warden stopped a {thinking/repetition} loop on turn {n}',
    fix: 'lower thinking budget; check the prompt does not ask to repeat',
  },
  'R-04': {
    message: '{tool} was called with {code} {k} times of {n}; {cause from 8.2}',
    fix: 'add the format and an example to the description; mark required parameters; type every parameter',
  },
  'R-05': {
    message: '{tool} was called {k} times with the same arguments and got the same result',
    fix: 'make the result say what changed or that the task is done; cap retries in the harness',
  },
  'R-06': {
    message: '{tool} failed {k} times in a row on turns {list}; the error results are {len} characters',
    fix: 'return errors that say what to change; describe valid arguments',
  },
  'R-07': { message: '{k} replies had no text and no tool call', fix: 'per 8.2 cause' },
  'R-08': {
    message: '{k} turns thought for {r} tokens to answer in under 40',
    fix: 'a lower thinking budget for this step, **with an answer check**',
  },
  'R-09': {
    message: '{k} values matched rule {rule} in {role}; the model saw placeholders',
    fix: 'keep secrets out of the prompt at the source',
  },
  'R-10': {
    message: 'First token took {s} s: {a} s recomputing what a stable prompt would have reused, {b} s queue',
    fix: 'the E finding if recompute dominates; otherwise none (server load, A-11)',
  },
};

/** Spec 8.2's words for a cause code (app/lens/response.py RESPONSE_CAUSES). */
export const CAUSE_TEXT: Record<string, string> = {
  limit_below_need: 'the limit is below what this step needs',
  thinking_used_budget: 'thinking used the budget',
  window_filled: 'the prompt filled the window',
  schema_untyped: 'a parameter has no type',
  schema_undocumented: 'a parameter has no description',
  nested_no_required: 'nested objects mark no required parameters',
  short_description: 'the description is short for a tool with 3 or more parameters',
  similar_tools: 'the model mixes up similar tools',
  parser_mismatch_admin: "the engine's tool parser did not match the model's format (an engine setting, not the prompt)",
  result_not_final: 'the result does not tell the model it is done',
  harness_no_result: 'the client did not run the tool between calls',
  error_too_short: 'the error does not say what to change',
  same_args_retry: 'the model is not told the arguments were the problem',
  template_swallowed: 'the template or the thinking setting swallowed the answer',
  no_after_tool_instruction: 'the prompt does not say what to do after a tool returns',
  thinking_for_short_step: 'consider a lower thinking budget for this step',
  secret_in_tool_results: 'a tool returns credentials into the prompt',
  secret_in_system: 'a secret is in the system prompt',
  recompute_dominates: 'most of it was recomputing the prompt',
  server_load: 'server load, not your prompt',
  none_found: 'no prompt cause found',
};

/** R-07's fix "per 8.2 cause". */
const R07_FIX: Record<string, string> = {
  template_swallowed: 'check the chat template and the thinking setting for this model',
  no_after_tool_instruction: 'say in the prompt what to do after a tool returns',
};

/** Where a guard match was (app/lens/response.py GUARD_WHERE). */
const WHERE_TEXT: Record<string, string> = {
  tool_result: 'tool results',
  system: 'the system prompt',
  user: 'user messages',
  assistant: 'assistant messages',
  tools: 'tool definitions',
};

const ROLE_OF_CAUSE: Record<string, string> = {
  secret_in_tool_results: 'tool results',
  secret_in_system: 'the system prompt',
};

export const layerOf = (rule: string): LensLayer => (rule[0] === 'E' ? 'E' : rule[0] === 'C' ? 'C' : 'R');

/** A notice, never a fix (`counts.info`, or a finding marked won't fix). */
export const isNotice = (f: LensFinding): boolean => f.counts.info === 1 || f.muted === true;

const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

/** A token count: 980, 5.8k, 23.9k, 1.2M. */
export function tok(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n < 1000) return fmt(n);
  if (n < 1_000_000) return `${Number((n / 1000).toFixed(1))}k`;
  return `${Number((n / 1_000_000).toFixed(1))}M`;
}

const secs = (ms: number) => `${Number((ms / 1000).toFixed(1))}`;

function pause(s: number): string {
  if (s < 90) return `${fmt(s)} s`;
  if (s < 5400) return `${fmt(s / 60)} min`;
  return `${Number((s / 3600).toFixed(1))} h`;
}

/** A tool's plain name, or "tool #<hash>" when the caller has no name for it. */
export function toolName(hash: string | null, names: Record<string, string> = {}): string | null {
  if (!hash) return null;
  return names[hash] ?? `tool #${hash.slice(0, 4)}`;
}

/** A tool at the start of a title: "Tool <name>", "Tool #<hash>" or "A tool".
 *  A name is never re-cased (M8): it is the caller's own identifier. */
function toolLead(hash: string | null, names: Record<string, string> = {}): string {
  if (!hash) return 'A tool';
  const name = names[hash];
  return name !== undefined ? `Tool ${name}` : `Tool #${hash.slice(0, 4)}`;
}

/** The fix text as shown: spec references dropped, markdown emphasis stripped. */
function clean(s: string): string {
  return s
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\s*\((?:A-\d+|#\d+)\)/g, '')
    .replace(/,\s*A-\d+\)/g, ')');
}

/** R-04 when the engine's tool parser missed the model's format: an engine
 *  setting, so the fix is the admin's, never a prompt edit (I2). */
const PARSER_FIX: Record<LensScope, string> = {
  admin: "set the engine's tool parser to match the model's tool-call format",
  key: "nothing to change in your prompt: the server's tool parser does not match this model's format",
};

function fixOf(rule: string, f: LensFinding, seen: Set<string>, scope: LensScope): string {
  const entry = CATALOGUE[rule];
  if (!entry) return '';
  const raw = entry.fix;
  const ref = /^(?:as|see) ([ECR]-\d\d)$/.exec(raw);
  if (ref && !seen.has(ref[1])) {
    seen.add(rule);
    return fixOf(ref[1], f, seen, scope);
  }
  if (rule === 'R-04' && f.cause === 'parser_mismatch_admin') return PARSER_FIX[scope];
  if (rule === 'R-07') return R07_FIX[f.cause ?? ''] ?? 'no prompt cause found: check the template and the thinking setting';
  if (rule === 'R-10') {
    if (f.cause === 'recompute_dominates') return 'fix the efficiency (E) finding of this session first';
    if (f.cause === 'server_load') return 'none: server load, not your prompt';
  }
  return clean(raw);
}

function titleOf(rule: string, f: LensFinding, names: Record<string, string>): string {
  const c = f.counts;
  const has = (k: string) => typeof c[k] === 'number';
  const tool = toolName(f.tool, names);
  switch (rule) {
    case 'E-01': {
      const cls = has('class') ? SPAN_CLASSES[c.class] : undefined;
      const every = (c.turns ?? 0) < 2 && (c.sessions ?? 1) >= 2 ? 'session' : 'turn';
      const line = has('line') ? `System prompt line ${c.line}` : 'A system prompt line';
      return `${line} changes every ${every}${cls && cls !== 'other' ? ` (${cls.replace(/_/g, ' ')})` : ''}; the lines after it are recomputed`;
    }
    case 'E-03':
      return `${toolLead(f.tool, names)} was added, removed or edited mid-list; everything after it is recomputed`;
    case 'E-04':
      return `Message ${has('message') ? c.message : '?'}, sent before, came back different`;
    case 'E-08':
      return has('pause_s')
        ? `Your prompt was stable; the cache expired during pauses of ${pause(c.pause_s)}`
        : 'Your prompt was stable; the prefix was below the provider\'s caching minimum';
    case 'E-09':
      return `An image or attachment in ${has('message') ? `message ${c.message}` : 'a message'} changed or moved`;
    case 'C-01':
      return c.info === 1
        ? `${c.rarely ?? 0} of ${c.offered ?? 0} tools were rarely used; their definitions add ${tok(c.tokens ?? 0)} tokens to every turn`
        : `${c.unused ?? 0} of ${c.offered ?? 0} tools were never called; their definitions add ${tok(c.tokens ?? 0)} tokens to every turn`;
    case 'C-02':
      return `The same ${tool ?? 'tool'} result (${tok(c.tokens ?? 0)} tokens) was sent ${c.repeats ?? 0} times`;
    case 'C-03':
      return `The system prompt and tools are ${c.share ?? 0} % of every request`;
    case 'C-04':
      return `Tool results are ${c.share ?? 0} % of the context; the largest (${tool ?? 'a tool'}) is ${tok(c.largest ?? 0)} tokens`;
    case 'C-05':
      return has('compact_at')
        ? `Your harness compacts at ${c.compact_at} % of the window`
        : `Context grows ${tok(c.growth ?? 0)} tokens per turn; the window fills by turn ${c.fill_turn ?? '?'}`;
    case 'C-06':
      return `Line ${has('line') ? c.line : '?'} changes; the lines after it are stable`;
    case 'C-07':
      return `Reminder and environment blocks are ${c.share ?? 0} % of the prompt`;
    case 'C-08':
      return `${toolLead(f.tool, names)}'s definition is ${tok(c.tokens ?? 0)} tokens, ${c.x ?? '?'} × the others${(c.used ?? 0) > 0 ? ` (called ${c.used} times)` : ''}`;
    case 'C-09':
      return `${c.tools ?? '?'} tools; OpenAI accepts at most ${c.limit ?? 128}`;
    case 'R-01':
      return `${c.cuts ?? 0} replies were cut at max_tokens = ${fmt(c.m ?? 0)}; the session's p90 reply is ${fmt(c.p90 ?? 0)}`;
    case 'R-02':
      return `The prompt filled the context window at turn ${c.turn ?? f.turns[0] ?? '?'}`;
    case 'R-03': {
      const kind = (c.think ?? 0) >= (c.repeat ?? 0) ? 'thinking' : 'repetition';
      return `The warden stopped a ${kind} loop on turn ${f.turns[0] ?? '?'}`;
    }
    case 'R-04': {
      const code = has('code') && c.code >= 0 ? CALL_CODES[c.code] : undefined;
      const what = code ? code.replace(/_/g, ' ') : 'bad arguments';
      return `${toolLead(f.tool, names)} was called with ${what} ${c.bad ?? 0} times of ${c.calls ?? 0}; ${CAUSE_TEXT[f.cause ?? 'none_found'] ?? CAUSE_TEXT.none_found}`;
    }
    case 'R-05':
      return `${toolLead(f.tool, names)} was called ${c.calls ?? 0} times with the same arguments and got the same result`;
    case 'R-06':
      return `${toolLead(f.tool, names)} failed ${c.failed ?? 0} times in a row on turns ${f.turns.join(', ')}; the error results are ${fmt(c.len ?? 0)} characters`;
    case 'R-07':
      return `${c.empty ?? 0} replies had no text and no tool call`;
    case 'R-08':
      return `${c.turns ?? 0} turns thought for ${tok(c.reasoning ?? 0)} tokens to answer in under 40`;
    case 'R-09': {
      const rules = f.guard?.rules ?? [];
      const rule = rules.length === 1 ? rules[0] : rules.length ? rules.join(', ') : `${c.rules ?? 0} rules`;
      const where = f.guard?.where?.length
        ? f.guard.where.map((w) => WHERE_TEXT[w] ?? w).join(' and ')
        : ROLE_OF_CAUSE[f.cause ?? ''] ?? 'the prompt';
      return `${c.values ?? 0} values matched rule ${rule} in ${where}; the model saw placeholders`;
    }
    case 'R-10':
      return `First token took ${secs(c.ttft_ms ?? 0)} s: ${secs(c.recompute_ms ?? 0)} s recomputing what a stable prompt would have reused, ${secs(c.queue_ms ?? 0)} s queue`;
    default: {
      const m = CATALOGUE[rule]?.message ?? rule;
      return m;
    }
  }
}

/** The finding in words: its title (the catalogue message, filled) and its fix
 *  (the catalogue fix; cross-references resolved, emphasis stripped). `names`
 *  maps tool hashes to the names the caller may see; `scope` picks the fix
 *  where it differs by reader (R-04's parser mismatch). */
export function ruleCopy(
  rule: string,
  f: LensFinding,
  names: Record<string, string> = {},
  scope: LensScope = 'admin',
): { title: string; fix: string } {
  return { title: titleOf(rule, f, names), fix: fixOf(rule, f, new Set(), scope) };
}

/** A few words for the findings strip (mockup `request-findings.html`). */
export function shortLabel(f: LensFinding, names: Record<string, string> = {}): string {
  const c = f.counts;
  const tool = toolName(f.tool, names);
  switch (f.rule) {
    case 'E-01':
      return typeof c.line === 'number' ? `line ${c.line} changes` : 'a system prompt line changes';
    case 'C-06':
      return 'volatile line before stable ones';
    case 'E-02':
      return 'tools re-ordered';
    case 'E-03':
      return 'tools edited mid-list';
    case 'E-04':
      return 'history rewritten';
    case 'E-05':
      return 'reasoning kept and dropped';
    case 'E-06':
      return 'no cache breakpoints';
    case 'E-07':
      return 'compaction changed the head';
    case 'E-08':
      return 'expired during pauses';
    case 'E-09':
      return 'attachment changed';
    case 'C-01':
      return c.info === 1 ? `${c.rarely ?? 0} rarely used tools` : `${c.unused ?? 0} unused tools`;
    case 'C-02':
      return 'repeated result';
    case 'C-03':
      return `head is ${c.share ?? 0} %`;
    case 'C-04':
      return `tool results ${c.share ?? 0} %`;
    case 'C-05':
      return 'context growth';
    case 'C-07':
      return `boilerplate ${c.share ?? 0} %`;
    case 'C-08':
      return `large tool definition${tool ? `: ${tool}` : ''}`;
    case 'C-09':
      return `${c.tools ?? '?'} tools`;
    case 'R-01':
      return 'replies cut';
    case 'R-02':
      return 'window filled';
    case 'R-03':
      return 'loop stopped';
    case 'R-04':
      return 'bad arguments';
    case 'R-05':
      return 'repeated calls';
    case 'R-06':
      return 'failed retries';
    case 'R-07':
      return 'empty replies';
    case 'R-08':
      return 'long thinking, short answer';
    case 'R-09':
      return 'secret in the prompt';
    case 'R-10':
      return `slow first token: ${secs(c.recompute_ms ?? 0)} of ${secs(c.ttft_ms ?? 0)} s recompute`;
    default:
      return f.rule;
  }
}

/** Key-holder copy: what the key reused is "reused from your earlier
 *  requests", never "cached" (spec 2026-11-22 G-R11). */
export function forScope(text: string, scope: LensScope): string {
  if (scope !== 'key') return text;
  return text.replace(/\bmake sure it is cached\b/g, 'make sure it is reused from your earlier requests').replace(/\bcached\b/g, 'reused from your earlier requests');
}

/** The part key a parts-table row stands for: "sys", "tools", or the
 *  "node:<i>" keys of the body nodes it covers. */
export function partKeys(p: { p: string; from: number; to: number }): string[] {
  if (p.p === 'sys' || p.p === 'tools') return [p.p];
  const out: string[] = [];
  for (let i = p.from; i <= p.to && out.length < 4096; i++) out.push(`node:${i}`);
  return out;
}

/** The findings that sit on a part (by its keys). */
export function findingsOn(findings: LensFinding[], keys: string[]): LensFinding[] {
  const want = new Set(keys);
  return findings.filter((f) => f.part !== null && want.has(f.part));
}

export function findingsUrl(scope: LensScope, reqId: string): string {
  const base = scope === 'admin' ? '/api/admin/prompt-lens' : '/api/me/prompt-lens';
  return `${base}/requests/${encodeURIComponent(reqId)}/findings`;
}

/** Reads after a 404 (the row is written as the request ends) before giving up. */
const RETRIES = 2;
const RETRY_MS = 1500;

function normalise(body: unknown): RequestFindings {
  const b = (body ?? {}) as Partial<RequestFindings>;
  return {
    turn: typeof b.turn === 'number' ? b.turn : null,
    findings: Array.isArray(b.findings) ? b.findings : [],
    names: b.names && typeof b.names === 'object' ? b.names : {},
  };
}

/** The findings on `reqId` (null: not asked yet, or none to show). Asked once
 *  per request, when `reqId` is set (the caller passes it once the request
 *  ended); a 404 right after the end is retried twice. Errors read as none. */
export function useRequestFindings(scope: LensScope, reqId: string | null): RequestFindings | null {
  const [state, setState] = useState<{ key: string; data: RequestFindings } | null>(null);
  const key = reqId ? `${scope}:${reqId}` : null;
  useEffect(() => {
    if (!reqId || !key) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const url = findingsUrl(scope, reqId);
    const get = scope === 'admin' ? (u: string) => authFetchJSON<unknown>(u) : sessionsFetcher<unknown>('key');
    const attempt = (left: number) => {
      get(url)
        .then((body) => {
          if (alive) setState({ key, data: normalise(body) });
        })
        .catch((e: unknown) => {
          const status = (e as { status?: unknown })?.status;
          if (alive && status === 404 && left > 0) timer = setTimeout(() => attempt(left - 1), RETRY_MS);
        });
    };
    attempt(RETRIES);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [scope, reqId, key]);
  return state && state.key === key ? state.data : null;
}
