'use client';

// Prompt Lens dock filters: pure matching over the store's request views,
// plus a hook that keeps them per page in localStorage.
//
// The key filter is applied by the server (the stream's `?token_ids=`), never
// here. The key holder's view has no client IPs, so the IP filter is ignored
// in the key scope.

import { useCallback, useState } from 'react';

import type { RequestView } from './store';

export type DockScope = 'key' | 'admin';
export type HasFlag = 'tool_calls' | 'thinking' | 'images' | 'tool_results';

export interface DockFilters {
  keys: string[] | null;
  sessions: string[] | null;
  models: string[] | null;
  ips: string[] | null;
  /** The client (harness): Sessions only, the live stream does not carry it. */
  harnesses: string[] | null;
  phase: 'all' | 'live' | 'finished';
  finish: string[] | null;
  has: HasFlag[];
  q: string;
}

/** The finish values the filter offers. */
export const FINISH_VALUES = ['stop', 'length', 'tool_calls', 'abort', 'error', 'runaway'] as const;
export const HAS_FLAGS: readonly HasFlag[] = ['tool_calls', 'thinking', 'images', 'tool_results'];
const PHASES = ['all', 'live', 'finished'] as const;

export const FILTERS_STORAGE_PREFIX = 'vw.prompt-lens.filters.';

export function defaultFilters(
  scope: DockScope,
  presetModel?: string | null,
  presetKeys?: string[] | null,
): DockFilters {
  void scope;
  return {
    keys: presetKeys && presetKeys.length ? [...presetKeys] : null,
    sessions: null,
    models: presetModel ? [presetModel] : null,
    ips: null,
    harnesses: null,
    phase: 'all',
    finish: null,
    has: [],
    q: '',
  };
}

const listed = (v: string[] | null): v is string[] => Array.isArray(v) && v.length > 0;

function hasToolResults(r: RequestView): boolean {
  const p = r.prompt;
  if (!p) return false;
  if (p.messages.some((m) => m.role === 'tool' || m.role === 'function')) return true;
  return /[tf]/.test(p.older?.roles ?? '');
}

function hasImages(r: RequestView): boolean {
  if (r.media.length > 0) return true;
  return !!r.prompt?.messages.some((m) => (m.images ?? 0) > 0);
}

function has(r: RequestView, flag: HasFlag): boolean {
  switch (flag) {
    case 'tool_calls':
      return r.toolCalls.length > 0;
    case 'thinking':
      return r.thinking.chars > 0;
    case 'images':
      return hasImages(r);
    case 'tool_results':
      return hasToolResults(r);
  }
}

function contains(text: string | undefined | null, needle: string): boolean {
  return !!text && text.toLowerCase().includes(needle);
}

/** Plain (not a pattern), case-insensitive search over the request's text. */
function matchesQuery(r: RequestView, needle: string): boolean {
  if (contains(r.promptFlat, needle)) return true;
  if (r.prompt) {
    if (contains(r.prompt.sys?.text, needle)) return true;
    for (const m of r.prompt.messages) {
      if (contains(m.text, needle)) return true;
    }
  }
  if (contains(r.thinking.text, needle) || contains(r.reply.text, needle)) return true;
  return r.toolCalls.some((c) => contains(c.args, needle));
}

/** The last search of each row: the reducer returns a new object only for a
 *  request that changed, so an unchanged row is never scanned again for the
 *  same needle (its text can be megabytes; a delta changes one row). */
const searched = new WeakMap<RequestView, { needle: string; hit: boolean }>();

function matchesCached(r: RequestView, needle: string): boolean {
  const c = searched.get(r);
  if (c && c.needle === needle) return c.hit;
  const hit = matchesQuery(r, needle);
  searched.set(r, { needle, hit });
  return hit;
}

export function applyFilters(rows: RequestView[], f: DockFilters, scope: DockScope): RequestView[] {
  const needle = f.q.trim().toLowerCase();
  return rows.filter((r) => {
    if (listed(f.sessions) && !(r.sessionId !== null && f.sessions.includes(r.sessionId))) return false;
    if (listed(f.models) && !f.models.includes(r.servedName) && !f.models.includes(r.model)) return false;
    if (scope === 'admin' && listed(f.ips) && !(r.clientIp && f.ips.includes(r.clientIp))) return false;
    if (f.phase === 'live' && r.endedTs !== null) return false;
    if (f.phase === 'finished' && r.endedTs === null) return false;
    if (listed(f.finish) && !(r.finish !== null && f.finish.includes(r.finish))) return false;
    for (const flag of f.has) if (!has(r, flag)) return false;
    if (needle && !matchesCached(r, needle)) return false;
    return true;
  });
}

/** How many filters narrow the list (for "Filters (n)"). */
export function activeCount(f: DockFilters): number {
  let n = 0;
  if (listed(f.keys)) n++;
  if (listed(f.sessions)) n++;
  if (listed(f.models)) n++;
  if (listed(f.ips)) n++;
  if (listed(f.harnesses)) n++;
  if (f.phase !== 'all') n++;
  if (listed(f.finish)) n++;
  if (f.has.length > 0) n++;
  if (f.q.trim()) n++;
  return n;
}

// ---------------------------------------------------------------------------
// Persistence

export interface FilterPresets {
  model?: string | null;
  keys?: string[] | null;
}

function strList(v: unknown): string[] | null | undefined {
  if (v === null) return null;
  if (!Array.isArray(v)) return undefined;
  return v.filter((x): x is string => typeof x === 'string');
}

/** A stored value over the defaults, field by field; ill-typed fields keep
 *  their default. */
function restore(raw: string | null, base: DockFilters): DockFilters {
  if (!raw) return base;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return base;
  }
  if (!v || typeof v !== 'object') return base;
  const o = v as Record<string, unknown>;
  const out: DockFilters = { ...base };
  for (const k of ['models', 'harnesses', 'finish'] as const) {
    const l = strList(o[k]);
    if (l !== undefined) out[k] = l;
  }
  if (typeof o.phase === 'string' && (PHASES as readonly string[]).includes(o.phase)) {
    out.phase = o.phase as DockFilters['phase'];
  }
  if (Array.isArray(o.has)) {
    out.has = o.has.filter((x): x is HasFlag => (HAS_FLAGS as readonly unknown[]).includes(x));
  }
  return out;
}

/** The dock's filters for one page. Models, clients, phase, finish and has are kept
 *  in `localStorage["vw.prompt-lens.filters." + pageKey]`; keys (the page's
 *  presets, or none), sessions, IPs and the search text never are. A Model
 *  preset overrides the stored models (G-R16). */
export function useDockFilters(
  pageKey: string,
  scope: DockScope,
  presets: FilterPresets,
): [DockFilters, (f: DockFilters) => void] {
  const storageKey = FILTERS_STORAGE_PREFIX + pageKey;
  const presetKeys = presets.keys && presets.keys.length ? presets.keys : null;
  const [filters, setFilters] = useState<DockFilters>(() => {
    const base = defaultFilters(scope, presets.model, presetKeys);
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(storageKey);
    } catch {
      raw = null;
    }
    const out = restore(raw, base);
    // G-R16: the page's Model preset (its scope model) wins on every visit; a
    // stored Model choice applies only to a page without one.
    if (presets.model) out.models = [presets.model];
    return out;
  });

  const update = useCallback(
    (f: DockFilters) => {
      setFilters(f);
      // Only the coarse choices: never the search text (it can hold prompt
      // text or a pasted secret), session ids, IPs or keys.
      const stored = { models: f.models, harnesses: f.harnesses, phase: f.phase, finish: f.finish, has: f.has };
      try {
        localStorage.setItem(storageKey, JSON.stringify(stored));
      } catch {
        // Private window, blocked storage: the filters still apply this visit.
      }
    },
    [storageKey],
  );

  return [filters, update];
}
