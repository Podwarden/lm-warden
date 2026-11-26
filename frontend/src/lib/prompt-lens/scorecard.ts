// Prompt Lens scores and the scorecard (analysis spec 10 and 12, Task 2.5):
//
//   key holder: GET /api/me/prompt-lens/sessions
//               GET /api/me/prompt-lens/sessions/{ref}/scorecard
//   admin:      the same under /api/admin/prompt-lens/ (the list takes key=)
//
// Everything here is numbers and codes (A-10). A layer that was not scored has
// no number at all (A-7): `scoreLine` and `ScoreOut.scored` are how the UI tells
// them apart, never a zero. Key holder copy says "reused from your earlier
// requests", never "cached" (`forScope`).

import { useCallback, useEffect, useState } from 'react';

import { authFetchJSON } from '@/lib/auth-fetch';
import { sessionsFetcher } from '@/lib/sessions/client';

import { forScope, type LensFinding, type LensLayer } from './findings';
import type { LensScope } from './stream';

/** The page's views (spec 16.1): Trends and Leaderboard arrive with their routes. */
export type LensView = 'requests' | 'sessions' | 'trends' | 'leaderboard';
export const LENS_VIEWS: readonly LensView[] = ['requests', 'sessions', 'trends', 'leaderboard'];

export function parseView(v: string | null | undefined): LensView {
  return LENS_VIEWS.includes(v as LensView) ? (v as LensView) : 'requests';
}

export type Confidence = 'high' | 'medium' | 'low';

/** One score, or the reason there is none: a score that was not computed has no
 *  `value`, `lo` or `hi` at all (A-7). */
export interface ScoreOut {
  layer: LensLayer | 'overall';
  scored: boolean;
  turns: number;
  reason?: string | null;
  value?: number | null;
  lo?: number | null;
  hi?: number | null;
  confidence?: Confidence | null;
  inputs?: Record<string, unknown> | null;
  excluded_rules?: string[] | null;
}

/** The score with won't-fix findings counted again (`inputs.with_marks`), or null. */
export function withMarks(s: ScoreOut | undefined): number | null {
  const v = s?.inputs?.with_marks;
  return typeof v === 'number' ? v : null;
}

export interface ScorecardFinding extends LensFinding {
  layer: LensLayer;
}

export interface TurnBar {
  n: number;
  req_id: string;
  reused: number;
  recomputed: number;
  new: number;
  markers: string[];
}

export interface Scorecard {
  session: {
    harness: string;
    model: string;
    route: string;
    turns: number;
    duration_s: number;
    /** The prompt version's label ("v4", or the client's own name). */
    version?: string | null;
    /** The stored version id: what a version-limited won't-fix mark names. */
    version_id?: string | null;
    version_since?: number | null;
    /** Admin scorecards only: the session's key. */
    key_id?: string | null;
  };
  scores: Record<string, ScoreOut>;
  top: ScorecardFinding[];
  notice?: ScorecardFinding | null;
  findings: ScorecardFinding[];
  not_scored: Record<string, string | null>;
  turns: TurnBar[];
  cost: {
    prompt_tokens: number;
    reused_tokens: number;
    recomputed_tokens: number;
    fixable_lost_tokens: number;
    card?: { provider: string; model: string; currency: string; effective_from: string } | null;
    actual_input?: number | null;
    saving?: number | null;
    saving_pct?: number | null;
  };
  score_version: number;
  weights: Record<string, number>;
  review?: Record<string, unknown> | null;
  names: Record<string, string>;
}

export interface SessionListItem {
  session_id: string;
  model: string;
  route: string;
  harness: string;
  turns: number;
  started: number;
  last_at: number;
  overall?: number | null;
  e?: number | null;
  c?: number | null;
  r?: number | null;
  confidence?: Confidence | null;
  reasons: Record<string, string | null>;
  score_version: number;
  key_id?: string | null;
  /** The API key's name; null once the key is deleted. */
  key_name?: string | null;
}

export interface SessionList {
  sessions: SessionListItem[];
  next_cursor?: string | null;
}

const base = (scope: LensScope) => (scope === 'admin' ? '/api/admin/prompt-lens' : '/api/me/prompt-lens');

export const scorecardUrl = (scope: LensScope, ref: string) =>
  `${base(scope)}/sessions/${encodeURIComponent(ref)}/scorecard`;

/** What the Sessions list is filtered by; each list matches any of its values. */
export interface SessionListFilter {
  /** Admin only: one key id or several (a key holder's list is its own key's). */
  key?: string | string[] | null;
  models?: string[] | null;
  /** The client: `claude_code`, `codex`, `opencode`, `openai_sdk`, `other`. */
  harnesses?: string[] | null;
}

export function sessionListUrl(scope: LensScope, o: SessionListFilter & { limit?: number; cursor?: string | null } = {}): string {
  const q = new URLSearchParams({ since: '0', limit: String(o.limit ?? 50) });
  if (o.cursor) q.set('cursor', o.cursor);
  if (scope === 'admin') for (const k of typeof o.key === 'string' ? [o.key] : o.key ?? []) if (k) q.append('key', k);
  for (const m of o.models ?? []) if (m) q.append('model', m);
  for (const h of o.harnesses ?? []) if (h) q.append('harness', h);
  return `${base(scope)}/sessions?${q.toString()}`;
}

function getter(scope: LensScope): (url: string) => Promise<unknown> {
  return scope === 'admin' ? (u) => authFetchJSON<unknown>(u) : sessionsFetcher<unknown>('key');
}

export interface Loaded<T> {
  data: T | null;
  /** 404: unknown, forgotten or not yours (one answer, A-1). */
  missing: boolean;
  error: boolean;
  loading: boolean;
}

export function useFetched<T>(scope: LensScope, url: string | null): Loaded<T> & { reload: () => void } {
  const [state, setState] = useState<{ url: string; data: T | null; status: number | null } | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!url) return;
    let alive = true;
    getter(scope)(url).then(
      (body) => {
        if (alive) setState({ url, data: body as T, status: 200 });
      },
      (e: unknown) => {
        const status = (e as { status?: unknown })?.status;
        if (alive) setState({ url, data: null, status: typeof status === 'number' ? status : 0 });
      },
    );
    return () => {
      alive = false;
    };
  }, [scope, url, tick]);
  const mine = state && state.url === url ? state : null;
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return {
    data: mine?.data ?? null,
    missing: mine?.status === 404,
    error: mine !== null && mine.data === null && mine.status !== 404,
    loading: url !== null && mine === null,
    reload,
  };
}

/** The scorecard of session `ref` (null `ref`: nothing asked). */
export function useScorecard(scope: LensScope, ref: string | null): Loaded<Scorecard> & { reload: () => void } {
  return useFetched<Scorecard>(scope, ref ? scorecardUrl(scope, ref) : null);
}

/** The scored sessions, newest first (admin: every key, or `key`). */
export function useSessionList(
  scope: LensScope,
  key: string | string[] | null = null,
  more: Pick<SessionListFilter, 'models' | 'harnesses'> = {},
): Loaded<SessionList> & { reload: () => void } {
  return useFetched<SessionList>(scope, sessionListUrl(scope, { key, ...more }));
}

// -- words and numbers ------------------------------------------------------------------

/** The tile's number: "62", or "not scored: 2 of 3 continuing turns". Never a
 *  number for an unscored layer (A-7). */
export function scoreLine(s: ScoreOut | undefined): string {
  if (!s) return 'not scored';
  if (!s.scored || typeof s.value !== 'number') return `not scored: ${s.reason ?? 'not enough data'}`;
  return String(s.value);
}

export function rangeText(s: ScoreOut): string | null {
  return s.scored && typeof s.lo === 'number' && typeof s.hi === 'number' ? `${s.lo}-${s.hi}` : null;
}

export const LAYER_NAME: Record<LensLayer, string> = { E: 'Efficiency', C: 'Construction', R: 'Response' };

/** What 100 means per layer (spec 10.1). */
export function meaning(layer: LensLayer, scope: LensScope): string {
  const t = {
    E: '100 = no turn lost cache reuse to something you control',
    C: '100 = no dead weight; 0 = half of every prompt',
    R: '100 = no reply failed in a visible way',
  }[layer];
  return forScope(t, scope);
}

export function fmtTok(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1_000)}k`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(Math.round(n));
}

const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** The numbers a score is computed from, in a sentence ("inputs" toggle). */
export function inputsText(s: ScoreOut, scope: LensScope): string {
  const i = s.inputs ?? {};
  switch (s.layer) {
    case 'E':
      return forScope(`lost ${fmtTok(num(i.lost))} of ${fmtTok(num(i.ideal))} reusable tokens · ${s.turns} turns`, scope);
    case 'C':
      return `dead weight ${fmtTok(num(i.waste))} of ${fmtTok(num(i.prompt))} prompt tokens · ${s.turns} requests`;
    case 'R':
      return `${num(i.bad)} of ${num(i.turns)} turns failed · ${fmtTok(num(i.waste_out))} of ${fmtTok(num(i.completion))} output tokens wasted`;
    default: {
      const from = Array.isArray(i.from) ? (i.from as string[]) : [];
      return from.length ? `from ${from.join(', ')}` : '';
    }
  }
}

/** A saving in words: tokens always, seconds with a rate, money with a card. */
export function savingText(f: LensFinding, currency: string | null, mult = 1): string {
  const s = f.saving;
  const parts = [`${fmtTok(s.tokens * mult)} tokens`];
  const secs = (s.prefill_s ?? 0) + (s.decode_s ?? 0);
  if (s.prefill_s !== null || s.decode_s !== null) parts.push(`${(secs * mult).toFixed(secs * mult < 10 ? 1 : 0)} s`);
  if (s.money !== null && currency) parts.push(`${(s.money * mult).toFixed(s.money * mult < 1 ? 4 : 2)} ${currency}`);
  return parts.join(' · ');
}

export function turnsText(turns: number[]): string {
  const t = turns.slice(0, 6).join(', ');
  return turns.length > 6 ? `${t} and ${turns.length - 6} more` : t;
}

/** The scored sessions by session id, for the Sessions page's Score column: an
 *  empty map while loading, when Prompt Lens is off, or on any error (the
 *  column just stays empty). */
export function useSessionScores(scope: LensScope): Record<string, SessionListItem> {
  const { data } = useSessionList(scope);
  const out: Record<string, SessionListItem> = {};
  for (const s of data?.sessions ?? []) out[s.session_id] = s;
  return out;
}
