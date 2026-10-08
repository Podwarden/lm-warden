'use client';
// Shared live-engine SSE stream for /api/stats/live — one EventSource per
// browser tab, ref-counted across every subscriber.
//
// This is a direct sibling of `header-metrics-stream.ts`: same module-level
// singleton + subscriber-set pattern, same ticket-mint preflight, same
// exp-backoff reconnect, same terminal-error (401/403/404) classification.
// The merged /ui/stats page mounts a single <StatsPage>, but React
// StrictMode dev re-renders and fast-refresh can briefly double-mount —
// collapsing to one underlying stream keeps the ticket-mint + /metrics-scrape
// traffic to exactly one connection per tab regardless.
//
// See header-metrics-stream.ts for the full rationale on why this lives
// outside `useEventSource` (frontend/src/lib/sse.ts).
import { useEffect, useState } from 'react';
import { authFetch, isLoginRedirectInFlight } from './auth-fetch';
import type { LiveEngineFrame } from './live-stats';

const STREAM_PATH = '/api/stats/live';
const TICKET_PATH = '/api/auth/sse-ticket';
const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 8_000;

export type LiveStatsStatus =
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'terminal-error';

export interface LiveStatsState {
  status: LiveStatsStatus;
  frame: LiveEngineFrame | null;
  /** HTTP status of the most recent ticket-mint failure, when applicable —
   *  distinguishes 401 (re-auth) from 404 (endpoint gone). */
  errorCode: number | null;
}

type Subscriber = (s: LiveStatsState) => void;

interface Stream {
  es: EventSource | null;
  state: LiveStatsState;
  subscribers: Set<Subscriber>;
  backoffMs: number;
  timer: ReturnType<typeof setTimeout> | null;
  stopped: boolean;
}

// Module-level singleton. ``null`` when no subscribers exist.
let stream: Stream | null = null;

function broadcast() {
  if (!stream) return;
  for (const sub of stream.subscribers) sub(stream.state);
}

function setState(patch: Partial<LiveStatsState>) {
  if (!stream) return;
  stream.state = { ...stream.state, ...patch };
  broadcast();
}

function isTerminalStatus(status: number): boolean {
  if (status === 429) return false;
  return status >= 400 && status < 500;
}

async function connect() {
  if (!stream || stream.stopped) return;

  // Mirror sse.ts: short-circuit if a peer authFetch already triggered
  // /login. Avoids the race where the preflight resolves a 401 mid-unload.
  if (isLoginRedirectInFlight()) {
    setState({ status: 'terminal-error', errorCode: 401 });
    return;
  }

  let ticket = '';
  try {
    const r = await authFetch(TICKET_PATH, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: STREAM_PATH }),
    });
    if (!r.ok) {
      if (isTerminalStatus(r.status)) {
        setState({ status: 'terminal-error', errorCode: r.status });
        return;
      }
      throw new Error(String(r.status));
    }
    try {
      const body = (await r.json()) as { ticket?: unknown };
      ticket = typeof body.ticket === 'string' ? body.ticket : '';
    } catch {
      ticket = '';
    }
  } catch (e) {
    if (isLoginRedirectInFlight()) {
      setState({ status: 'terminal-error', errorCode: 401 });
      return;
    }
    const status =
      e instanceof Error && /^\d+$/.test(e.message) ? Number(e.message) : null;
    scheduleReconnect(status);
    return;
  }

  if (!stream || stream.stopped) return;

  const es = new EventSource(
    `${STREAM_PATH}?ticket=${encodeURIComponent(ticket)}`,
  );
  stream.es = es;

  es.onopen = () => {
    if (!stream) return;
    stream.backoffMs = INITIAL_BACKOFF_MS;
    setState({ status: 'connected', errorCode: null });
  };

  es.onmessage = (e) => {
    if (!stream) return;
    try {
      const frame = JSON.parse(e.data) as LiveEngineFrame;
      setState({ status: 'connected', errorCode: null, frame });
    } catch {
      // Malformed payload — swallow. Backend pins the JSON shape.
    }
  };

  es.onerror = () => {
    if (!stream) return;
    stream.es?.close();
    stream.es = null;
    scheduleReconnect(null);
  };
}

function scheduleReconnect(httpStatus: number | null) {
  if (!stream || stream.stopped) return;
  setState({ status: 'reconnecting', errorCode: httpStatus });
  const delay = Math.min(stream.backoffMs, MAX_BACKOFF_MS);
  stream.backoffMs *= 2;
  stream.timer = setTimeout(() => {
    if (!stream || stream.stopped) return;
    void connect();
  }, delay);
}

function teardown() {
  if (!stream) return;
  stream.stopped = true;
  if (stream.timer !== null) clearTimeout(stream.timer);
  stream.es?.close();
  stream = null;
}

/**
 * Subscribe to the shared live-engine stream. Returns an unsubscribe
 * function; the stream opens on the first subscriber and closes when the
 * last unsubscribes. The cached state is handed to a new subscriber
 * synchronously so it renders the last frame instead of "connecting".
 */
export function subscribeLiveStats(sub: Subscriber): () => void {
  if (!stream) {
    stream = {
      es: null,
      state: { status: 'connecting', frame: null, errorCode: null },
      subscribers: new Set(),
      backoffMs: INITIAL_BACKOFF_MS,
      timer: null,
      stopped: false,
    };
    void connect();
  }
  stream.subscribers.add(sub);
  sub(stream.state);

  return () => {
    if (!stream) return;
    stream.subscribers.delete(sub);
    if (stream.subscribers.size === 0) teardown();
  };
}

/** React hook wrapping `subscribeLiveStats`. `enabled=false` decouples from
 *  the stream entirely (e.g. when the tab-hidden guard should pause it). */
export function useLiveStats(enabled = true): LiveStatsState {
  const [state, setLocalState] = useState<LiveStatsState>({
    status: 'connecting',
    frame: null,
    errorCode: null,
  });

  useEffect(() => {
    if (!enabled) return;
    return subscribeLiveStats(setLocalState);
  }, [enabled]);

  return state;
}

// Test-only escape hatch — mirrors __resetHeaderMetricsStreamForTests.
// Production code MUST NOT call this.
export function __resetLiveStatsStreamForTests(): void {
  teardown();
}
