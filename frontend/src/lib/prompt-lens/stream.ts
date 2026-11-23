// Prompt Lens fetches. `fetchSection` reads one whole section of a request
// (spec 2026-11-22 G9; Task 8's routes):
//
//   admin:      GET /api/admin/prompt-lens/requests/{req_id}/section/{kind}?from=
//   key holder: GET /api/me/prompt-lens/requests/{req_id}/section/{kind}?from=
//
// The key holder's route is bound to the key of the sign-in (no key
// parameter), answers one 404 for an unknown, released or another key's id,
// and 429 past 60 DB-backed (kept capture) reads a minute for the key. A key
// holder's response has `seq: null`: the pop-up tracks its place by
// `offset + text.length` (ruling G-F3), never by seq.
//
// The stream hook (`usePromptLensStream`) opens one of the two live streams and
// folds its events into a DockState (./store.ts):
//
//   admin:      GET /api/admin/prompt-lens/stream[?token_ids=a,b]  (ticket from
//               POST /api/auth/sse-ticket, the admin session)
//   key holder: GET /api/me/prompt-lens/stream  (ticket from
//               POST /api/me/prompt-lens/ticket with the key's sign-in; the
//               stream is bound to that key on the server: no key parameter)
//
// Every EventSource open dispatches `{type: "connected"}` before that
// connection's first event: the key holder's seq restarts per connection, the
// admin's is the hub's (global). Every event with a server `ts` feeds the
// server clock (./server-clock.ts). The key stream allows 4 open streams and
// 30 ticket mints a minute per key (429): the mint's 429 waits for its
// Retry-After (lib/sse.ts).

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';

import { authFetchJSON } from '@/lib/auth-fetch';
import { KeyAuthError, SessionsHttpError, sessionsFetcher } from '@/lib/sessions/client';
import { clearKeyToken, readKeyToken } from '@/lib/sessions/key-token';
import { useEventSource, type SseState } from '@/lib/sse';

import type { PromptLensEvent, KeptCapture, KeptSummary, SectionResponse } from './events';
import type { SectionKind } from './section-marks';
import { noteServerTs } from './server-clock';
import { emptyDockState, reduce, type DockAction, type DockState } from './store';

export type { SectionResponse };
export type FetchKind = Exclude<SectionKind, 'parts'>;
export type LensScope = 'key' | 'admin';

/** A section fetch that failed: `status` is the HTTP status (401 when the
 *  key holder's sign-in is gone, 0 for a network error). */
export class SectionFetchError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string | null = null,
  ) {
    super(status ? `HTTP ${status}` : 'network error');
    this.name = 'SectionFetchError';
  }
}

export function sectionUrl(scope: LensScope, reqId: string, kind: FetchKind, from = 0): string {
  const base = scope === 'admin' ? '/api/admin/prompt-lens' : '/api/me/prompt-lens';
  const q = from > 0 ? `?from=${Math.floor(from)}` : '';
  return `${base}/requests/${encodeURIComponent(reqId)}/section/${kind}${q}`;
}

/** One whole section, text from the absolute offset `from`. Throws a
 *  SectionFetchError. */
export async function fetchSection(
  scope: LensScope,
  reqId: string,
  kind: FetchKind,
  from = 0,
): Promise<SectionResponse> {
  const url = sectionUrl(scope, reqId, kind, from);
  try {
    if (scope === 'admin') return await authFetchJSON<SectionResponse>(url);
    return await sessionsFetcher<SectionResponse>('key')(url);
  } catch (e) {
    if (e instanceof SectionFetchError) throw e;
    if (e instanceof KeyAuthError) throw new SectionFetchError(401);
    if (e instanceof SessionsHttpError) throw new SectionFetchError(e.status, e.detail);
    const status = (e as { status?: unknown })?.status;
    throw new SectionFetchError(typeof status === 'number' ? status : 0);
  }
}

/** The status of a failed fetch (0 when unknown). */
export function fetchStatus(e: unknown): number {
  return e instanceof SectionFetchError ? e.status : 0;
}

// ---------------------------------------------------------------------------
// The live stream

export const ADMIN_STREAM_PATH = '/api/admin/prompt-lens/stream';
export const KEY_STREAM_PATH = '/api/me/prompt-lens/stream';
export const KEY_TICKET_PATH = '/api/me/prompt-lens/ticket';

/** The key holder's ticket mint (ruling G-F12): a POST with the key's sign-in
 *  token that hands lib/sse.ts the raw Response to classify. No sign-in held:
 *  a synthetic 401 (terminal). */
export async function keyPostTicket(): Promise<Response> {
  const token = readKeyToken();
  if (!token) return new Response(JSON.stringify({ detail: 'not signed in' }), { status: 401 });
  const r = await fetch(KEY_TICKET_PATH, {
    method: 'POST',
    credentials: 'omit',
    headers: { Authorization: `Bearer ${token}` },
  });
  // A dead sign-in is forgotten at once, as sessionsFetcher does, so the rest
  // of the key area does not keep using it.
  if (r.status === 401 || r.status === 403) clearKeyToken();
  return r;
}

type StreamAction = { type: 'reset' } | { type: 'batch'; actions: DockAction[] };

function streamReduce(s: DockState, a: StreamAction): DockState {
  if (a.type === 'reset') return emptyDockState();
  let out = s;
  for (const x of a.actions) out = reduce(out, x);
  return out;
}

/** Past this many queued actions the queue is applied at once. */
const MAX_QUEUED = 2_000;
/** A hidden tab runs no animation frames: flush on a timer instead. */
const HIDDEN_FLUSH_MS = 250;

export interface PromptLensStream {
  state: DockState;
  sse: SseState;
  /** For kept captures and fetched sections (kept_list, kept_capture,
   *  section_fill). Queued with the stream's events, in order. */
  dispatch: (a: DockAction) => void;
  /** Drop the connection and connect again (the "can't connect" retry). */
  restart: () => void;
}

/**
 * The dock's live stream. `tokenIds` (admin only) narrows it on the server
 * (`?token_ids=`); a new set reconnects with an empty state. The key scope
 * ignores `tokenIds`: its stream is bound to the signed-in key.
 *
 * Events are queued and applied once per animation frame in one reducer pass
 * (one render a frame however fast the deltas come); a hidden tab, which runs
 * no frames, flushes on a timer.
 */
export function usePromptLensStream(
  scope: LensScope,
  opts: { enabled: boolean; tokenIds: string[] | null },
): PromptLensStream {
  const [state, dispatchRaw] = useReducer(streamReduce, undefined, emptyDockState);
  const ids = scope === 'admin' && opts.tokenIds && opts.tokenIds.length ? opts.tokenIds.join(',') : '';

  const queue = useRef<DockAction[]>([]);
  const pending = useRef<{ kind: 'raf' | 'timer'; id: number } | null>(null);
  const cancel = useCallback(() => {
    const p = pending.current;
    pending.current = null;
    if (!p) return;
    if (p.kind === 'raf') cancelAnimationFrame(p.id);
    else clearTimeout(p.id);
  }, []);
  const flush = useCallback(() => {
    pending.current = null;
    const actions = queue.current;
    if (!actions.length) return;
    queue.current = [];
    dispatchRaw({ type: 'batch', actions });
  }, []);
  const enqueue = useCallback(
    (a: DockAction) => {
      queue.current.push(a);
      if (queue.current.length >= MAX_QUEUED) {
        cancel();
        flush();
        return;
      }
      if (pending.current) return;
      const hidden = typeof document !== 'undefined' && document.visibilityState === 'hidden';
      if (!hidden && typeof requestAnimationFrame === 'function') {
        pending.current = { kind: 'raf', id: requestAnimationFrame(flush) };
      } else {
        pending.current = { kind: 'timer', id: setTimeout(flush, HIDDEN_FLUSH_MS) as unknown as number };
      }
    },
    [cancel, flush],
  );
  useEffect(() => cancel, [cancel]);
  // A frame scheduled while visible never runs once the tab hides: switch it
  // to the timer, or the queue would wait for the tab to come back (or 2000).
  useEffect(() => {
    if (typeof document === 'undefined') return;
    const onVis = () => {
      const p = pending.current;
      if (document.visibilityState !== 'hidden' || !p || p.kind !== 'raf') return;
      cancelAnimationFrame(p.id);
      pending.current = { kind: 'timer', id: setTimeout(flush, HIDDEN_FLUSH_MS) as unknown as number };
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [flush]);

  // A new key set (or scope) starts over: the old set's requests must not stay.
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    cancel();
    queue.current = [];
    dispatchRaw({ type: 'reset' });
  }, [scope, ids, cancel]);

  const seqScope = scope === 'admin' ? 'global' : 'connection';
  const onOpen = useCallback(() => enqueue({ type: 'connected', seqScope }), [enqueue, seqScope]);
  const onMessage = useCallback(
    (ev: PromptLensEvent) => {
      if (!ev || typeof ev !== 'object') return;
      const ts = (ev as { ts?: unknown }).ts;
      // Noted on arrival, not at the flush: the flush delay is not skew.
      if (typeof ts === 'number') noteServerTs(ts);
      enqueue(ev);
    },
    [enqueue],
  );

  const [attempt, setAttempt] = useState(0);
  const sse = useEventSource<PromptLensEvent>(scope === 'admin' ? ADMIN_STREAM_PATH : KEY_STREAM_PATH, {
    enabled: opts.enabled,
    onMessage,
    onOpen,
    query: ids ? { token_ids: ids } : undefined,
    mint: scope === 'key' ? keyPostTicket : undefined,
    restartKey: attempt,
  });

  const restart = useCallback(() => setAttempt((n) => n + 1), []);
  return { state, sse, dispatch: enqueue, restart };
}

// ---------------------------------------------------------------------------
// Kept captures (keep captures, G-R4): GET .../prompt-lens/kept[/{req_id}]

export function keptListUrl(scope: LensScope, tokenIds: string[] | null): string {
  if (scope === 'key') return '/api/me/prompt-lens/kept';
  const ids = tokenIds && tokenIds.length ? tokenIds.join(',') : '';
  return ids ? `/api/admin/prompt-lens/kept?token_ids=${encodeURIComponent(ids)}` : '/api/admin/prompt-lens/kept';
}

async function getJson<T>(scope: LensScope, url: string): Promise<T> {
  try {
    if (scope === 'admin') return await authFetchJSON<T>(url);
    return await sessionsFetcher<T>('key')(url);
  } catch (e) {
    if (e instanceof KeyAuthError) throw new SectionFetchError(401);
    if (e instanceof SessionsHttpError) throw new SectionFetchError(e.status, e.detail);
    const status = (e as { status?: unknown })?.status;
    throw new SectionFetchError(typeof status === 'number' ? status : 0);
  }
}

/** The kept captures of the key (key scope) or of `tokenIds` (admin; all
 *  keys when null). Throws a SectionFetchError. */
export async function fetchKeptList(scope: LensScope, tokenIds: string[] | null): Promise<KeptSummary[]> {
  const body = await getJson<{ kept?: KeptSummary[] }>(scope, keptListUrl(scope, tokenIds));
  return Array.isArray(body?.kept) ? body.kept : [];
}

/** One kept capture: its start (no prompt_parts), kvsig and end events. */
export async function fetchKeptCapture(scope: LensScope, reqId: string): Promise<KeptCapture> {
  const base = scope === 'admin' ? '/api/admin/prompt-lens' : '/api/me/prompt-lens';
  return getJson<KeptCapture>(scope, `${base}/kept/${encodeURIComponent(reqId)}`);
}
