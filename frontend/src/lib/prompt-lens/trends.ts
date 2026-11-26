// Prompt Lens trends, prompt versions and won't fix (analysis spec 11.4 and 13,
// Tasks 3.2-3.4):
//
//   key holder: GET /api/me/prompt-lens/trends | versions | compare | wontfix
//               PUT|DELETE /api/me/prompt-lens/wontfix/{rule}
//   admin:      the same under /api/admin/prompt-lens/ with a required key=
//
// Numbers and codes only (A-10). A bucket with fewer than three scored sessions
// has null medians and no band (A-7): the chart draws a gap there, never a 0. A
// key holder's answers hold no other key's figure (D-4): this module has no
// fleet call, and the key route takes no key.

import { authFetch } from '@/lib/auth-fetch';
import { meWrite } from '@/lib/key-area/api';

import { useFetched, type Loaded } from './scorecard';
import type { LensScope } from './stream';

export type TrendBy = 'day' | 'week' | 'version';
export type Metric = 'overall' | 'E' | 'C' | 'R';
export const METRICS: readonly Metric[] = ['overall', 'E', 'C', 'R'];
export const METRIC_NAME: Record<Metric, string> = {
  overall: 'Overall',
  E: 'Efficiency',
  C: 'Construction',
  R: 'Response',
};

export interface TrendPoint {
  /** The local day, the local week's Monday, or the version id. */
  bucket: string;
  n: number;
  medians: Record<Metric, number | null>;
  /** 25th-75th percentile of the overall; null with no median. */
  band?: [number, number] | null;
  start: number;
  label?: string | null;
}

export interface VersionInfo {
  id: string;
  label: string;
  since: number;
  last_at: number;
  n: number;
  medians: Record<Metric, number | null>;
  /** Rule -> sessions of this version it fired in. */
  rules: Record<string, number>;
  /** Fired on the version before and not here. */
  resolved: string[];
  new: string[];
}

export interface Trends {
  by: TrendBy;
  model?: string | null;
  models: string[];
  points: TrendPoint[];
  versions: VersionInfo[];
  weights: Record<string, number>;
  score_version: number;
}

export type Verdict = 'better_clear' | 'better_unclear' | 'no_change' | 'worse_clear' | 'worse_unclear' | 'not_enough';

export interface Compare {
  model?: string | null;
  a: string;
  b: string;
  a_label: string;
  b_label: string;
  n_a: number;
  n_b: number;
  diffs: Record<Metric, number | null>;
  verdict: Verdict;
  like_for_like: boolean;
  note?: string | null;
  resolved: string[];
  new: string[];
  p?: number | null;
}

export interface WontfixMark {
  rule: string;
  /** null: every prompt version. */
  version: string | null;
  by: string;
  at: number;
}

const base = (scope: LensScope) => (scope === 'admin' ? '/api/admin/prompt-lens' : '/api/me/prompt-lens');

function query(scope: LensScope, key: string | null, o: Record<string, string | number | null | undefined>): string {
  const q = new URLSearchParams();
  if (scope === 'admin' && key) q.set('key', key);
  for (const [k, v] of Object.entries(o)) if (v !== null && v !== undefined && v !== '') q.set(k, String(v));
  const s = q.toString();
  return s ? `?${s}` : '';
}

/** The minutes east of UTC the browser is at, for day and week buckets. */
export const tzMinutes = (): number => -new Date().getTimezoneOffset();

export function trendsUrl(scope: LensScope, o: { key?: string | null; by: TrendBy; model?: string | null; tz?: number }): string {
  return `${base(scope)}/trends${query(scope, o.key ?? null, { by: o.by, model: o.model, tz: o.tz ?? 0, since: 0 })}`;
}

export function compareUrl(scope: LensScope, o: { key?: string | null; a: string; b: string; model?: string | null }): string {
  return `${base(scope)}/compare${query(scope, o.key ?? null, { a: o.a, b: o.b, model: o.model, since: 0 })}`;
}

/** The admin reads one key's trends; without a key there is nothing to ask. */
const needsKey = (scope: LensScope, key: string | null) => scope === 'key' || !!key;

export function useTrends(scope: LensScope, key: string | null, by: TrendBy, model: string | null, tz: number = tzMinutes()): Loaded<Trends> & { reload: () => void } {
  return useFetched<Trends>(scope, needsKey(scope, key) ? trendsUrl(scope, { key, by, model, tz }) : null);
}

/** Version `b` against the earlier `a` (either null: nothing asked). */
export function useCompare(scope: LensScope, key: string | null, a: string | null, b: string | null, model: string | null): Loaded<Compare> & { reload: () => void } {
  return useFetched<Compare>(scope, a && b && needsKey(scope, key) ? compareUrl(scope, { key, a, b, model }) : null);
}

/** Mark (`on`) or unmark a rule won't fix for the key: every version when
 *  `version` is null, else that one. The key holder's mark names no key; the
 *  admin's names the session's. Resolves true when the server accepted it. */
export async function setWontfix(scope: LensScope, o: { rule: string; on: boolean; version: string | null; key?: string | null }): Promise<boolean> {
  if (scope === 'admin') {
    if (!o.key) return false;
    const url = `/api/admin/prompt-lens/wontfix/${encodeURIComponent(o.rule)}?${new URLSearchParams({ key: o.key, ...(!o.on && o.version ? { version: o.version } : {}) }).toString()}`;
    const r = await authFetch(url, {
      method: o.on ? 'PUT' : 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: o.on ? JSON.stringify({ version: o.version }) : undefined,
    });
    return r.ok;
  }
  const url = `/api/me/prompt-lens/wontfix/${encodeURIComponent(o.rule)}${!o.on && o.version ? `?version=${encodeURIComponent(o.version)}` : ''}`;
  const r = await meWrite(url, o.on ? 'PUT' : 'DELETE', o.on ? { version: o.version } : undefined);
  return r.ok;
}

// -- words -----------------------------------------------------------------------------

/** The verdict in the card's words ("Better, clear · 23 vs 31 sessions"). */
export function verdictText(c: Pick<Compare, 'verdict' | 'n_a' | 'n_b'>): string {
  const sessions = `${c.n_b} vs ${c.n_a} sessions`;
  switch (c.verdict) {
    case 'better_clear':
      return `Better, clear · ${sessions}`;
    case 'better_unclear':
      return `Better, not yet clear · ${sessions}`;
    case 'no_change':
      return `No change · ${sessions}`;
    case 'worse_clear':
      return `Worse, clear · ${sessions}`;
    case 'worse_unclear':
      return `Worse, not yet clear · ${sessions}`;
    default:
      return `Not enough sessions yet · ${sessions}`;
  }
}

/** Waiting (amber) for a verdict that is not settled, good for a clear gain. */
export const verdictTone = (v: Verdict): 'ok' | 'wait' | 'bad' => (v === 'better_clear' ? 'ok' : v === 'worse_clear' ? 'bad' : 'wait');

/** "+11", "-3", "0", or "—" when the layer has too few sessions. */
export function diffText(d: number | null | undefined): string {
  if (typeof d !== 'number') return '—';
  return d > 0 ? `+${d}` : String(d);
}

export function bucketLabel(p: TrendPoint, by: TrendBy): string {
  if (by === 'version') return p.label ?? p.bucket;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(p.bucket)) return p.bucket;
  const [, m, d] = p.bucket.split('-');
  const names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${Number(d)} ${names[Number(m) - 1] ?? m}`;
}

export function percent(n: number, of: number): string {
  return of > 0 ? `${Math.round((100 * n) / of)} %` : '—';
}

/** A few words per rule for the findings-by-version table (the catalogue's
 *  messages carry per-finding placeholders, so they cannot stand alone). */
export const RULE_LABEL: Record<string, string> = {
  'E-01': 'A system prompt line changes',
  'E-02': 'Tools sent in a different order',
  'E-03': 'A tool added, removed or edited mid-list',
  'E-04': 'Earlier messages came back different',
  'E-05': 'Reasoning kept on some turns, dropped on others',
  'E-06': 'No cache breakpoints (relayed route)',
  'E-07': 'Compaction changed the system prompt or tools',
  'E-08': 'The cache expired during pauses',
  'E-09': 'An image or attachment changed or moved',
  'C-01': 'Tools never called',
  'C-02': 'The same result sent again and again',
  'C-03': 'The system prompt and tools are most of every request',
  'C-04': 'Tool results dominate the context',
  'C-05': 'The context grows fast',
  'C-06': 'One line changes, the rest is stable',
  'C-07': 'Reminder and environment blocks',
  'C-08': 'One tool definition is much larger than the others',
  'C-09': 'Many tools',
  'R-01': 'Replies cut at max_tokens',
  'R-02': 'The prompt filled the context window',
  'R-03': 'A loop was stopped',
  'R-04': 'A tool called with bad arguments',
  'R-05': 'Repeated identical calls',
  'R-06': 'A tool failed several times in a row',
  'R-07': 'Empty replies',
  'R-08': 'Long thinking, short answer',
  'R-09': 'A secret in the prompt',
  'R-10': 'Slow first token',
};

/** How a rule stands across versions: "open" (fires on the newest), "resolved
 *  in vN" (stopped firing at that version and has not come back), "improved"
 *  (fires on fewer of the newest version's sessions than the first that had it). */
export function ruleStatus(rule: string, versions: VersionInfo[]): { text: string; tone: 'ok' | 'bad' | 'muted' } {
  const share = (v: VersionInfo) => (v.n > 0 ? (v.rules[rule] ?? 0) / v.n : 0);
  const last = versions[versions.length - 1];
  if (!last) return { text: '', tone: 'muted' };
  if (share(last) > 0) {
    const first = versions.find((v) => (v.rules[rule] ?? 0) > 0);
    return first && first !== last && share(last) < share(first) - 0.05 ? { text: 'improved', tone: 'muted' } : { text: 'open', tone: 'bad' };
  }
  const lastFired = [...versions].reverse().find((v) => (v.rules[rule] ?? 0) > 0);
  const at = lastFired ? versions[versions.indexOf(lastFired) + 1] : undefined;
  return { text: at ? `resolved in ${at.label}` : 'resolved', tone: 'ok' };
}
