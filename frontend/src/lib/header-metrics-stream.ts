'use client';
// Shared header-metrics SSE stream — one EventSource per browser tab,
// ref-counted across every <HeaderMetrics /> mount.
//
// Why this lives outside `useEventSource`:
//   - `useEventSource` (frontend/src/lib/sse.ts) spins up a fresh
//     EventSource per hook instance. That's correct for the model-log
//     pages where every mount streams a *different* path. The header
//     widget mounts once at the layout level today, but tests and
//     future fast-refresh sessions can double-mount briefly during
//     hot reload — opening two parallel streams against the same
//     endpoint, doubling the ticket-mint traffic + nvidia-smi probes.
//   - The nav-bar widget must reuse a single underlying connection
//     even if the component re-mounts (route change, theme switch,
//     React.StrictMode dev re-render). A module-level singleton with
//     a subscriber set is the simplest pattern that matches the
//     existing module-state idiom in `auth-fetch.ts` (which carries
//     a similar singleton for `loginRedirectInFlight` + token).
//
// Lifecycle:
//   - First subscriber: mint ticket, open EventSource, broadcast each
//     parsed message to all subscribers.
//   - Subsequent subscribers (within the same tab): attach to the
//     existing stream, immediately receive the last-known frame if any.
//   - Last subscriber unmounts: close EventSource, clear state.
//
// Reconnect: simple exp-backoff capped at MAX_BACKOFF_MS. Terminal-error
// classification (401/403/404) matches `useEventSource` semantics.
import { useEffect, useState } from 'react';
import { authFetch, isLoginRedirectInFlight } from './auth-fetch';
import type { HeaderActiveModel, HeaderModelStatus } from './header-models';

const STREAM_PATH = '/api/header/metrics/stream';
const TICKET_PATH = '/api/auth/sse-ticket';
const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 8_000;

export interface HeaderMetricsGpu {
  index: number;
  name: string | null;
  memory_used_mib: number;
  memory_total_mib: number;
  utilization_pct: number;
}

export interface HeaderMetricsFrame {
  ts: string;
  gpus: HeaderMetricsGpu[];
  /** Pooled across every card: used / total. A whole-box question.
   *  The four numbers are null when nothing was measured — the GPU probe
   *  failed (probe_error says why) or no card answered. Never a stand-in 0
   *  (#255). An API older than that sends 0 instead. */
  vram_used_mib: number | null;
  vram_total_mib: number | null;
  vram_pct: number | null;
  /** The BUSIEST card, not an average — see app/header/routes_api.py. */
  gpu_util_pct: number | null;
  // Every model the box is serving, starting, or has crashed, most
  // significant first. Optional at the type level for the same skew reason as
  // active_model_status below: a UI newer than its API must still render, and
  // an API that predates multi-model sends only the singular fields.
  active_models?: HeaderActiveModel[];
  active_model: string | null;
  active_model_id: string | null;
  // Optional at the type level on purpose: the UI and API ship as separate
  // images and can skew, so a UI newer than its API must still render. When
  // the key is absent the widget falls back to deriving 'loaded' from a
  // non-null active_model — the old contract's only meaning.
  active_model_status?: HeaderModelStatus | null;
  probe_error: string | null;
}

export type HeaderMetricsStatus =
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'terminal-error';

export interface HeaderMetricsState {
  status: HeaderMetricsStatus;
  frame: HeaderMetricsFrame | null;
  /** HTTP status of the most recent ticket-mint failure, when applicable.
   *  Useful for distinguishing 401 (re-auth) from 404 (endpoint gone). */
  errorCode: number | null;
}

type Subscriber = (s: HeaderMetricsState) => void;

interface Stream {
  es: EventSource | null;
  state: HeaderMetricsState;
  subscribers: Set<Subscriber>;
  backoffMs: number;
  timer: ReturnType<typeof setTimeout> | null;
  stopped: boolean;
}

// Module-level singleton. ``null`` when no subscribers exist.
let stream: Stream | null = null;

function broadcast(s: Stream) {
  for (const sub of s.subscribers) sub(s.state);
}

function setState(s: Stream, patch: Partial<HeaderMetricsState>) {
  s.state = { ...s.state, ...patch };
  broadcast(s);
}

function isTerminalStatus(status: number): boolean {
  if (status === 429) return false;
  return status >= 400 && status < 500;
}

/** True once `s` has been torn down — including when a NEW stream has since
 *  replaced it. Every async step re-checks this against the stream it started
 *  for, never against the module-level `stream`: an unmount + remount during
 *  a ticket mint must not let the old connect() open a second EventSource on
 *  the new stream (#251). */
function dead(s: Stream): boolean {
  return s.stopped || stream !== s;
}

async function connect(s: Stream) {
  if (dead(s)) return;

  // Mirror sse.ts: short-circuit if a peer authFetch already triggered
  // /login. Avoids the same race where the preflight resolves a 401
  // mid-unload.
  if (isLoginRedirectInFlight()) {
    setState(s, { status: 'terminal-error', errorCode: 401 });
    return;
  }

  let ticket = '';
  try {
    const r = await authFetch(TICKET_PATH, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: STREAM_PATH }),
    });
    if (dead(s)) return;
    if (!r.ok) {
      if (isTerminalStatus(r.status)) {
        setState(s, { status: 'terminal-error', errorCode: r.status });
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
    if (dead(s)) return;
    if (isLoginRedirectInFlight()) {
      setState(s, { status: 'terminal-error', errorCode: 401 });
      return;
    }
    const status =
      e instanceof Error && /^\d+$/.test(e.message) ? Number(e.message) : null;
    scheduleReconnect(s, status);
    return;
  }

  if (dead(s)) return;

  const es = new EventSource(
    `${STREAM_PATH}?ticket=${encodeURIComponent(ticket)}`,
  );
  s.es = es;

  es.onopen = () => {
    if (dead(s)) return;
    s.backoffMs = INITIAL_BACKOFF_MS;
    setState(s, { status: 'connected', errorCode: null });
  };

  es.onmessage = (e) => {
    if (dead(s)) return;
    try {
      const frame = JSON.parse(e.data) as HeaderMetricsFrame;
      setState(s, { status: 'connected', errorCode: null, frame });
    } catch {
      // Malformed payload — swallow. Backend pins JSON shape.
    }
  };

  es.onerror = () => {
    es.close();
    if (dead(s)) return;
    s.es = null;
    scheduleReconnect(s, null);
  };
}

function scheduleReconnect(s: Stream, httpStatus: number | null) {
  if (dead(s)) return;
  setState(s, { status: 'reconnecting', errorCode: httpStatus });
  const delay = Math.min(s.backoffMs, MAX_BACKOFF_MS);
  s.backoffMs *= 2;
  s.timer = setTimeout(() => {
    if (dead(s)) return;
    void connect(s);
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
 * Subscribe to the shared header-metrics stream. Returns an unsubscribe
 * function; the stream opens on the first subscriber and closes when the
 * last unsubscribes. ``initial`` is broadcast synchronously so the caller
 * can render the cached state before the next emit.
 *
 * Exposed primarily for the hook below; tests may use it directly.
 */
export function subscribeHeaderMetrics(sub: Subscriber): () => void {
  if (!stream) {
    stream = {
      es: null,
      state: { status: 'connecting', frame: null, errorCode: null },
      subscribers: new Set(),
      backoffMs: INITIAL_BACKOFF_MS,
      timer: null,
      stopped: false,
    };
    void connect(stream);
  }
  stream.subscribers.add(sub);
  // Hand the new subscriber the cached state synchronously so they
  // don't briefly render "connecting" if a frame already arrived.
  sub(stream.state);

  return () => {
    if (!stream) return;
    stream.subscribers.delete(sub);
    if (stream.subscribers.size === 0) teardown();
  };
}

/**
 * React hook wrapping ``subscribeHeaderMetrics``. ``enabled=false``
 * decouples from the stream entirely (e.g. when mounted on /login —
 * but in practice the parent component returns null on those paths,
 * so this is belt-and-suspenders).
 */
export function useHeaderMetrics(enabled = true): HeaderMetricsState {
  const [state, setLocalState] = useState<HeaderMetricsState>({
    status: 'connecting',
    frame: null,
    errorCode: null,
  });

  useEffect(() => {
    if (!enabled) return;
    return subscribeHeaderMetrics(setLocalState);
  }, [enabled]);

  return state;
}

// Test-only escape hatch — see frontend/tests/setup.ts pattern in
// auth-fetch.ts. Production code MUST NOT call this; it tears down
// the shared stream so vitest module-state leak between files is
// neutralised.
export function __resetHeaderMetricsStreamForTests(): void {
  teardown();
}
