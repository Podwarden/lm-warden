'use client';
import { useEffect, useRef, useState } from 'react';
import { authFetch, isLoginRedirectInFlight } from './auth-fetch';

// ---------------------------------------------------------------------------
// SSE connection state machine — v2026.05.15.2 hardening
// ---------------------------------------------------------------------------
// The previous useEventSource silently exp-backoff'd forever on every error,
// which meant a 404 or 403 from the backend looked indistinguishable from a
// transient network blip in the UI: the spinner just kept spinning. The
// EventSource API itself only exposes a generic `error` event with no HTTP
// status, so we can't tell terminal-from-transient by looking at the SSE
// connection alone.
//
// The fix: before each connect attempt we mint the SSE ticket via
// `POST /api/auth/sse-ticket`. That request DOES surface a real HTTP status,
// and serves as a HEAD-ish probe for the path itself (the backend rejects
// the ticket mint if the caller has no access to the underlying stream).
// We use the ticket-mint status to classify:
//
//   * 4xx (except 429) → terminal-error  — the operator needs to act
//     (re-login, or the stream genuinely doesn't exist). Stop reconnecting.
//   * 5xx, network, 429, or EventSource onerror after at least one onopen →
//     reconnecting with exp backoff, capped at MAX_RECONNECT attempts so
//     a broken backend doesn't pin the tab to retry forever.
//
// State is surfaced via the hook's return value so consumers (e.g.
// log-stream.tsx) can render distinct UI: "connecting…", "live", "lost,
// retrying (N/5)", "stream unavailable — please re-login or refresh".

export type SseStatus =
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'terminal-error';

export interface SseState {
  status: SseStatus;
  /** HTTP status of the most recent ticket-mint failure, if any. Useful
   *  for distinguishing 401/403 (auth) from 404 (stream gone) in the UI. */
  errorCode: number | null;
  /** How many reconnect attempts have been made since the last successful
   *  open. Resets to 0 on a successful EventSource `onopen`. */
  attempts: number;
}

/** Cap on automatic reconnect attempts before we give up and surface a
 *  terminal-error to the consumer. Five attempts at exp backoff
 *  (250ms, 500ms, 1s, 2s, 4s) cap at MAX_BACKOFF_MS — the cumulative
 *  ~8s window is intentionally tight so an operator who clicks "Pull"
 *  doesn't sit through 30s+ of "reconnecting" while the supervisor is
 *  still bootstrapping the subprocess. */
export const MAX_RECONNECT = 5;

/** Exponential backoff ceiling — issue #53. Lowered from 30s to 5s to
 *  shorten the operator's wait during the pulled→loading transition,
 *  where the Next.js rewrite-proxy may briefly return 503s as the
 *  upstream socket flips. Capping at 5s means a transient blip is at
 *  most one MAX_BACKOFF_MS wait away from a recovery attempt. */
const MAX_BACKOFF_MS = 5_000;

/** Initial backoff — also the value we reset to on a successful onopen.
 *  Issue #53. Lowered from 1s to 250ms so the FIRST retry after a
 *  transient failure is sub-second, hiding the proxy's status-flip blip
 *  from the operator. The doubling sequence then ramps as before. */
const INITIAL_BACKOFF_MS = 250;

interface UseEventSourceOptions<T> {
  onMessage: (m: T) => void;
  enabled?: boolean;
  /**
   * Extra query parameters for the stream URL (e.g. god mode's `token_ids`).
   *
   * The SSE ticket is still minted for the BARE `path`: the ticket check on the
   * server compares `request.url.path` only, and the god-mode 409 check in
   * `POST /api/auth/sse-ticket` compares the body's path exactly. The ticket is
   * appended last, so the URL is `${path}?${query}&ticket=…`. Changing the
   * query tears the stream down and reconnects.
   */
  query?: Record<string, string>;
}

export function useEventSource<T>(
  path: string,
  opts: UseEventSourceOptions<T>,
): SseState {
  const [state, setState] = useState<SseState>({
    status: 'connecting',
    errorCode: null,
    attempts: 0,
  });

  // Stash the latest onMessage in a ref so the effect can pick it up
  // without re-subscribing on every parent rerender. The pre-fix hook
  // relied on consumers memoising onMessage; we keep that contract via
  // the deps array (omitted from the lint rule) but also defend against
  // a non-memoised handler by always reading from the ref.
  const onMessageRef = useRef(opts.onMessage);
  onMessageRef.current = opts.onMessage;

  // Serialised once per render so the effect can depend on a string: a fresh
  // object literal from the caller must not reconnect on every render.
  const qs = opts.query ? new URLSearchParams(opts.query).toString() : '';

  useEffect(() => {
    if (opts.enabled === false) {
      setState({ status: 'connecting', errorCode: null, attempts: 0 });
      return;
    }
    let stopped = false;
    let es: EventSource | null = null;
    let backoffMs = INITIAL_BACKOFF_MS;
    let attempts = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const scheduleReconnect = (httpStatus: number | null) => {
      if (stopped) return;
      attempts += 1;
      if (attempts > MAX_RECONNECT) {
        // Out of retries — surface a terminal-error and stop. We pick
        // the most recent HTTP status if we have one; otherwise null
        // signals "EventSource gave up on us with no further detail".
        setState({
          status: 'terminal-error',
          errorCode: httpStatus,
          attempts,
        });
        return;
      }
      setState({
        status: 'reconnecting',
        errorCode: httpStatus,
        attempts,
      });
      const delay = Math.min(backoffMs, MAX_BACKOFF_MS);
      backoffMs *= 2;
      timer = setTimeout(connect, delay);
    };

    const setTerminal = (httpStatus: number) => {
      // Terminal-error short-circuits the reconnect counter — these are
      // failures that no amount of retrying will fix (auth revoked,
      // stream URL gone). The operator needs to act.
      setState({
        status: 'terminal-error',
        errorCode: httpStatus,
        attempts,
      });
    };

    /** Classify a ticket-mint failure status. 4xx (except 429) is terminal;
     *  5xx/429/network is transient. Exported semantics for the test. */
    const isTerminalStatus = (status: number): boolean => {
      // 429 (Too Many Requests) is rate-limiting and worth retrying.
      if (status === 429) return false;
      return status >= 400 && status < 500;
    };

    async function connect() {
      if (stopped) return;

      // v2026.05.15.3 — if a peer authFetch already triggered the
      // /login redirect, skip the preflight entirely. The navigation
      // is in flight; making a POST now races the unload and (pre-fix)
      // left the panel blank because authFetch resolved with a 401 the
      // hook never got to see. Surface terminal-error up front so the
      // operator sees an explicit "session expired" banner.
      if (isLoginRedirectInFlight()) {
        setTerminal(401);
        return;
      }

      let ticket: string;
      try {
        const r = await authFetch('/api/auth/sse-ticket', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path }),
        });
        if (!r.ok) {
          if (isTerminalStatus(r.status)) {
            // 401/403/404/etc — give up, tell the UI.
            setTerminal(r.status);
            return;
          }
          // 5xx or 429 — schedule a retry. Throwing into the catch
          // below keeps the backoff path consolidated.
          throw new Error(String(r.status));
        }
        // Body parse is best-effort: a missing/garbled ticket is NOT
        // treated as terminal. The previous hook constructed the
        // EventSource with whatever value it got (including
        // `ticket=undefined`) and let the server reject if needed;
        // we preserve that for downstream tests / endpoints that don't
        // care about the ticket round-trip. The 4xx/5xx classification
        // above is the real safety net.
        let body: unknown;
        try {
          body = await r.json();
        } catch {
          body = null;
        }
        const t = (body as { ticket?: unknown } | null)?.ticket;
        ticket = typeof t === 'string' ? t : '';
      } catch (e) {
        // Network failure or thrown 5xx/429 — schedule another attempt.
        // We have no HTTP status on a pure network drop; pass null.
        // v2026.05.15.3: if a peer 401 fired the login redirect WHILE
        // our preflight was in flight, authFetch may have rejected with
        // a network error from a torn-down page. Re-check the flag and
        // map that to terminal "session expired" rather than looping
        // into reconnect attempts that will all fail the same way
        // during the unload window. (A preflight that resolves with a
        // 401 Response is handled by the `if (!r.ok)` branch above, not
        // here — control only reaches this catch on a thrown error.)
        if (isLoginRedirectInFlight()) {
          setTerminal(401);
          return;
        }
        const status =
          e instanceof Error && /^\d+$/.test(e.message) ? Number(e.message) : null;
        scheduleReconnect(status);
        return;
      }

      // The effect may have been torn down while the ticket was minting.
      if (stopped) return;
      es = new EventSource(
        `${path}?${qs ? `${qs}&` : ''}ticket=${encodeURIComponent(ticket)}`,
      );
      es.onopen = () => {
        // Successful connection — reset the backoff and attempts. Note
        // we don't reset `attempts` to 0 until we've actually opened;
        // a ticket that mints OK but whose EventSource immediately
        // errors should NOT silently reset the counter.
        backoffMs = INITIAL_BACKOFF_MS;
        attempts = 0;
        setState({ status: 'connected', errorCode: null, attempts: 0 });
      };
      es.onmessage = (e) => {
        try {
          onMessageRef.current(JSON.parse(e.data) as T);
        } catch {
          // Swallow malformed payloads — a single bad line shouldn't
          // tear down the stream. The backend pin guarantees JSON, so
          // this is purely a defence against future regressions.
        }
      };
      es.onerror = () => {
        // EventSource errors don't carry an HTTP status — could be a
        // network drop, the server closing the stream, or anything in
        // between. Schedule a reconnect with null status; the next
        // ticket-mint will re-classify if it turns out to be terminal.
        es?.close();
        es = null;
        scheduleReconnect(null);
      };
    }

    setState({ status: 'connecting', errorCode: null, attempts: 0 });
    connect();
    return () => {
      stopped = true;
      if (timer !== null) clearTimeout(timer);
      es?.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, opts.enabled, qs]);

  return state;
}
