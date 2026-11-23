// Stats page redesign (2026-11-22): wire types of the new endpoints and the pure
// helpers the panels share. One scope per page (one model on this install); the
// bottom Requests table is the only place for breakdowns.
//
//   /api/stats/v2/speed            per-request latency and speed of one model (B3)
//   /api/stats/requests            in-flight rows with ttft / out tokens / 5 s decode (B1, B2)
//   /api/stats/live (SSE)          per-model block gains `now` and `history_5m` (B5)
//   /api/me/stats/*                the key holder's own view (B8, B9)
//   (/api/me/requests/{id}/capture stays one release, deprecated: the UI uses
//   the Prompt Lens dock, components/prompt-lens/dock, instead.)

import type { LiveEngineFrame } from "./live-stats";
import { type GpuLiveResponse, throttleVerdict } from "./system-info";

/** The server's low-volume summary (app/stats/speed.py `summarize`). */
export type SpeedSummary =
  | { n: 0; mode: "none" }
  | { n: number; mode: "list"; values: number[] }
  | { n: number; mode: "max"; p50: number; max: number }
  | { n: number; mode: "p95"; p50: number; p95: number; p99: number | null };

export interface SpeedPoint {
  id: string;
  finished_at: number;
  model_id: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  finish_reason: string | null;
  queued_s: number | null;
  ttft: number | null;
  decode: number | null;
  duration: number | null;
  /** Admin only. */
  token_id?: string | null;
  token_name?: string | null;
  client_ip?: string | null;
  prefill?: number | null;
  prefill_alone?: boolean;
  /** Key holder only: prompt / TTFT, cache effects included. */
  prompt_speed?: number | null;
}

export interface SpeedGroup {
  key: string | null;
  label: string | null;
  model_id: string | null;
  model: string | null;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  n_ttft: number;
  p50_ttft: number | null;
  n_decode: number;
  p50_decode: number | null;
  n_prefill: number;
  p50_prefill: number | null;
}

export interface SpeedResponse {
  range: string;
  model_id: string | null;
  last_finished_at: number | null;
  count: number;
  stride: number;
  summary: {
    ttft: SpeedSummary;
    decode: SpeedSummary;
    duration: SpeedSummary;
    warden_queue: SpeedSummary;
    prefill?: SpeedSummary;
    prompt?: SpeedSummary;
  } | null;
  points: SpeedPoint[];
  groups?: SpeedGroup[] | null;
  /** Key holder only: the models this key used in the window. */
  models?: { model_id: string; model: string; requests: number }[];
  /** Key holder only: when this key's newest request finished (any model,
   *  whatever the window), and the anchored end of the window, if any. */
  last_request_at?: number | null;
  until?: number | null;
}

/** The epoch the key holder's window should end at: the key's last request
 *  when the window up to now holds none of its requests (and the holder did not
 *  choose "Back to now"); null = end now. A key never used stays at now. */
export function anchoredEnd(now: SpeedResponse | undefined, backToNow: boolean): number | null {
  if (backToNow || !now || now.count > 0 || now.last_request_at == null) return null;
  return now.last_request_at + 1;
}

/** B5: the server-side ~10 s rates and the warden's own queue. */
export interface LiveNow {
  generation_tps_10s: number | null;
  prefill_computed_tps_10s: number | null;
  window_s: number | null;
  warden_queue: number;
  warden_running: number;
}

export interface LiveHistoryPoint {
  age_s: number;
  gen: number | null;
  prefill: number | null;
}

/** The key holder's in-flight row (/api/me/stats/requests): no key, IP or fleet figure. */
export interface MeLiveRow {
  id: string;
  model: string;
  model_id: string;
  phase: string;
  is_stream: boolean;
  prompt_tokens: number;
  completion_tokens: number;
  out_tokens: number;
  context_tokens: number;
  max_model_len: number | null;
  context_pct: number | null;
  elapsed_s: number;
  queued_s: number | null;
  ttft_s: number | null;
  ttft_total_s: number | null;
  decode_tps_5s: number | null;
  /** The engine's own count for a non-streamed request (optional). */
  engine_progress_tokens?: number | null;
  session_id: string | null;
  forest_session: string | null;
  expected_start_s: number | null;
  expected_start_basis: "own" | null;
}

export interface MeRequests {
  count: number;
  requests: MeLiveRow[];
  state: Record<string, "waiting" | "running">;
}

export interface MeTokens {
  range: string;
  model_id: string | null;
  since_minute: number;
  now_minute: number;
  minutes: { minute: number; requests: number; prompt_tokens: number; completion_tokens: number }[];
}

export interface MeCache {
  range: string;
  model_id: string | null;
  cache: {
    requests: number;
    prompt_tokens: number;
    own_reused_tokens: number;
    own_known_prompt_tokens: number;
    own_known_requests: number;
    outcomes: Record<string, number>;
    prefill_saved_s: number | null;
    prefill_saved_basis: "configured";
    prefill_rate_tok_s: number | null;
  };
}

// ---- fixed sizes (design 3.0) ----------------------------------------------
//
// Every panel above the bottom Requests block has a fixed height in every
// state; the live table is a fixed number of rows. Tests pin these.
export const FIXED = {
  scopeBar: 56,
  notice: 28,
  nowCard: 206,
  /** Phone (< 768 px) heights where a card's columns stack: fixed per breakpoint. */
  nowCardPhone: 330,
  cacheCardPhone: 262,
  engineCardPhone: 150,
  /** The latency chart's plot, in real pixels inside its fixed card. */
  latencyPlot: 216,
  tile: 184,
  latencyChart: 300,
  tokensChart: 412,
  cacheCard: 150,
  engineCard: 96,
  hostCard: 104,
  tableRows: 8,
  rowH: 38,
  tabW: 280,
  /** Phone tab width: the state chip stays on screen in the key shell. */
  tabWPhone: 220,
  chipW: 78,
} as const;

// ---- formatting --------------------------------------------------------------

export function fmtSeconds(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "-";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 10) return `${s.toFixed(1)} s`;
  return `${Math.round(s)} s`;
}

export function fmtRate(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "-";
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`;
}

export function fmtTokens(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "-";
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e4) return `${Math.round(v / 1e3)}k`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return `${Math.round(v)}`;
}

export function fmtAgo(epochS: number | null | undefined, nowMs = Date.now()): string {
  if (epochS == null) return "no requests yet";
  const s = Math.max(0, nowMs / 1000 - epochS);
  if (s < 90) return `${Math.max(1, Math.round(s))} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function fmtElapsed(s: number): string {
  const t = Math.max(0, Math.floor(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}

/** What a tile shows, from the server's summary. Same skeleton for every mode. */
export interface TileText {
  /** The large value, or null when `list` / `empty` text replaces it. */
  big: string | null;
  /** Replaces the large value: the values themselves, or "No requests in window". */
  list: string | null;
  sub: string;
  n: number;
}

export function tileText(
  s: SpeedSummary | undefined | null,
  fmt: (v: number) => string,
  lastFinishedAt: number | null,
  /** Requests in the window (all of them, not only those this figure counts). */
  windowCount = 0,
  /** What to say when requests exist but none qualifies for this figure. */
  noneQualify?: { list: string; sub: (n: number) => string },
): TileText {
  if (!s || s.mode === "none") {
    if (windowCount > 0 && noneQualify) {
      // Say what is true: there were requests, none counts for this figure.
      return { big: null, list: noneQualify.list, sub: noneQualify.sub(windowCount), n: 0 };
    }
    return {
      big: null,
      list: "No requests in window",
      sub: lastFinishedAt ? `last request ${fmtAgo(lastFinishedAt)}` : "no requests yet",
      n: 0,
    };
  }
  if (s.mode === "list") {
    return {
      big: null,
      list: s.values.map(fmt).join(" · "),
      sub: "too few for percentiles · each value shown",
      n: s.n,
    };
  }
  if (s.mode === "max") {
    return { big: fmt(s.p50), list: null, sub: `p50 · max ${fmt(s.max)} (p95 from n ≥ 20)`, n: s.n };
  }
  return {
    big: fmt(s.p50),
    list: null,
    sub: `p50 · p95 ${fmt(s.p95)}${s.p99 != null ? ` · p99 ${fmt(s.p99)}` : ""}`,
    n: s.n,
  };
}

/** Model state chip in the scope bar. Admin: Queueing (warden or engine queue),
 *  Busy (running), Idle. Key holder: their own Waiting / Running / Idle. */
export type ScopeState = "Queueing" | "Busy" | "Idle" | "Waiting" | "Running" | "Asleep" | "No data";

export function adminState(opts: {
  running: number | null;
  engineWaiting: number | null;
  wardenQueue: number | null;
  asleep?: boolean;
  noData?: boolean;
}): ScopeState {
  if (opts.noData) return "No data";
  if (opts.asleep) return "Asleep";
  if ((opts.engineWaiting ?? 0) > 0 || (opts.wardenQueue ?? 0) > 0) return "Queueing";
  if ((opts.running ?? 0) > 0) return "Busy";
  return "Idle";
}

/** Asleep only from a real non-awake sleep state, and never while the engine
 *  serves: requests running beat any sleep reading (vLLM 0.32 labels its
 *  sleep gauge per state, and an awake engine once read as "Asleep"). */
export function isAsleep(b: LiveEngineFrame): boolean {
  if ((b.engine?.num_requests_running ?? 0) > 0 || (b.now?.warden_running ?? 0) > 0) return false;
  const sleeping = b.engine?.engine_sleeping;
  if (sleeping !== undefined) return sleeping === true; // null: not exported, never Asleep
  return (b.engine?.engine_sleep_state ?? 0) > 0; // an api older than the field
}

/** One notice for GPUs held back by heat: it costs real throughput, so it goes
 *  on the reserved notice line rather than only on the per-GPU cards. */
export function thermalNotice(g: GpuLiveResponse | undefined): { strong: string; rest?: string } | null {
  const hot = (g?.gpus ?? [])
    .map((c) => ({ c, v: throttleVerdict(c.telemetry.throttle, c.telemetry.sm_clock_mhz, c.telemetry.sm_clock_max_mhz) }))
    .filter((x) => x.v && x.v.reasons.some((r) => /thermal/i.test(r)));
  if (!hot.length) return null;
  const pcts = hot.map((x) => x.v!.heldPct).filter((p): p is number => p != null);
  const temps = hot.map((x) => x.c.telemetry.temperature_c).filter((t): t is number => t != null);
  const span = (v: number[], unit: string) =>
    v.length ? (Math.min(...v) === Math.max(...v) ? `${Math.min(...v)}${unit}` : `${Math.min(...v)}-${Math.max(...v)}${unit}`) : null;
  const n = g?.gpus.length ?? hot.length;
  const parts = [
    span(pcts, " % of max clock"),
    span(temps, " °C"),
    "this costs throughput; check cooling",
  ].filter(Boolean);
  return { strong: `Thermal slowdown on ${hot.length} of ${n} GPU${n === 1 ? "" : "s"}`, rest: parts.join(" · ") };
}

// ---- live table grouping (client side, over the rows the table shows) ------

export type GroupBy = "none" | "key" | "ip" | "model";

export interface LiveLike {
  token_id?: string | null;
  token_name?: string | null;
  client_ip?: string | null;
  model: string;
  phase: string;
  context_tokens: number;
  decode_tps_5s?: number | null;
}

export interface LiveGroup {
  id: string;
  key: string | null;
  label: string;
  model: string | null;
  inFlight: number;
  queued: number;
  context: number;
  liveTps: number | null;
}

/** One group per key and model (speeds never mixed across models); ip and model
 *  groups carry no extra model split. */
export function groupLive(rows: readonly LiveLike[], by: Exclude<GroupBy, "none">, multiModel: boolean): LiveGroup[] {
  const acc = new Map<string, LiveGroup>();
  for (const r of rows) {
    let key: string | null;
    let label: string;
    let model: string | null = null;
    if (by === "key") {
      key = r.token_id ?? null;
      label = r.token_name ?? (key ? key.slice(0, 8) : "anonymous");
      if (multiModel) model = r.model;
    } else if (by === "ip") {
      key = r.client_ip ?? null;
      label = key ?? "unknown";
    } else {
      key = r.model;
      label = r.model;
    }
    const id = `${key ?? "-"}|${model ?? ""}`;
    let g = acc.get(id);
    if (!g) {
      g = { id, key, label, model, inFlight: 0, queued: 0, context: 0, liveTps: null };
      acc.set(id, g);
    }
    g.inFlight += 1;
    if (r.phase === "queued") g.queued += 1;
    g.context += r.context_tokens ?? 0;
    if (r.decode_tps_5s != null) g.liveTps = (g.liveTps ?? 0) + r.decode_tps_5s;
  }
  return [...acc.values()].sort((a, b) => b.inFlight - a.inFlight || a.label.localeCompare(b.label));
}

/** Rolling median over the last `win` values, from the 5th value on. */
export function rollingMedian(values: readonly number[], win = 20): (number | null)[] {
  return values.map((_, i) => {
    if (i < 4) return null;
    const w = values.slice(Math.max(0, i - win + 1), i + 1).slice().sort((a, b) => a - b);
    const m = (w.length - 1) / 2;
    return (w[Math.floor(m)] + w[Math.ceil(m)]) / 2;
  });
}
