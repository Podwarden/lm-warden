// Token details page: response types for GET /api/tokens/{id} and
// GET /api/tokens/{id}/series (spec 2026-09-18 §3.3/§3.4 — field names exactly),
// the series fetcher, the URL range state and the bin label.
//
// The URL holds the selection (`?range=7d` or `?from=<epoch>&to=<epoch>`) so a
// reload or a shared link reopens the same period (spec §4.3).

import { authFetch } from "@/lib/auth-fetch";
import type { TokenItem } from "@/components/tokens/token-row";

export interface TokenLineageEntry {
  id: string;
  name: string;
  created_at: string;
  rotated_at: string | null;
  is_revoked: boolean;
  in_grace: boolean;
  is_self: boolean;
}

export interface TokenDetail extends TokenItem {
  paused_at: string | null;
  is_paused: boolean;
  /** Rotation chain, oldest first; includes this token (`is_self`) and its
   *  successor if one exists. */
  lineage: TokenLineageEntry[];
}

export interface SeriesBin {
  minute: number;
  requests_per_min: number;
  prompt_per_min: number;
  completion_per_min: number;
  peak_prompt: number;
  peak_completion: number;
  requests: number;
  n: number;
  queue_p50: number | null;
  queue_p95: number | null;
  ttft_p50: number | null;
  ttft_p95: number | null;
  duration_p50: number | null;
  duration_p95: number | null;
}

/** One variant of a model (migration 0036): the exact configuration it was
 *  served as. The descriptor fields are a summary; other identity fields
 *  (repo, files, extra args) split variants apart without being shown. */
export interface ModelVariantUsage {
  variant_id: string;
  backend: string | null;
  engine_channel: string | null;
  engine_vllm_version: string | null;
  /** The in-container engine's baked version at launch; null when an image
   *  ran (its tag says it) or unknown. */
  engine_version: string | null;
  /** Tag of the image the engine actually ran (the pin, or the driver's
   *  default). */
  engine_image_tag: string | null;
  quantization: string | null;
  dtype: string | null;
  hf_revision: string | null;
  /** The commit `hf_revision` resolved to at launch, or the ref itself when
   *  the cache had no entry for it. */
  hf_commit: string | null;
  max_model_len: number | null;
  /** Epoch seconds the variant was first served; null if never recorded. */
  first_seen: number | null;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** Of the key's total within the per-model data, 0..1. */
  share: number;
}

/** One model's usage for the window and token set, busiest first. */
export interface ModelUsage {
  model_id: string;
  /** Served name; the stored id once the model is deleted. */
  model: string;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** Of the key's total within the per-model data, 0..1. */
  share: number;
  /** Busiest first; at least one. */
  variants: ModelVariantUsage[];
}

export interface ModelRates {
  prompt_per_min: number;
  completion_per_min: number;
}

/** Sparse: only bins with per-model data, and in each only the models that
 *  had traffic. Same keying and sum ÷ width rule as `bins`. */
export interface ModelBin {
  minute: number;
  models: Record<string, ModelRates>;
}

export interface TokenSeries {
  token_ids: string[];
  from_minute: number;
  to_minute: number;
  bin_minutes: number;
  /** Epoch seconds of the oldest request_history row that carries a token_id;
   *  null when there is none. Timings before it were never recorded. */
  latency_since: number | null;
  /** Timings are read from at most TIMING_ROW_CAP rows: every `stride`-th of
   *  `total` (spec §3.4 sampling guard). Bin `n` counts sampled rows. */
  timing_sample: { total: number; used: number; stride: number };
  totals: { requests: number; prompt_tokens: number; completion_tokens: number };
  /** Sparse: only bins with data. `fillBins` makes it dense. */
  bins: SeriesBin[];
  /** Epoch seconds of the first per-model row store-wide (0036 has no
   *  backfill); null when there is none, or when `by_model=0`. */
  by_model_since: number | null;
  by_model: ModelUsage[];
  model_bins: ModelBin[];
}

// ---- range selection --------------------------------------------------------

export type Preset = "1h" | "6h" | "24h" | "7d";
export const PRESETS: readonly Preset[] = ["1h", "6h", "24h", "7d"];
export const PRESET_MINUTES: Record<Preset, number> = { "1h": 60, "6h": 360, "24h": 1440, "7d": 10080 };

export type RangeSel =
  | { kind: "preset"; preset: Preset }
  | { kind: "custom"; from: number; to: number };

export const DEFAULT_RANGE: RangeSel = { kind: "preset", preset: "7d" };

export function parseRange(sp: { get(name: string): string | null } | null): RangeSel {
  if (!sp) return DEFAULT_RANGE;
  const rawFrom = sp.get("from");
  const rawTo = sp.get("to");
  if (rawFrom !== null && rawTo !== null) {
    const from = Number(rawFrom);
    const to = Number(rawTo);
    if (Number.isInteger(from) && Number.isInteger(to) && from > 0 && from < to) {
      return { kind: "custom", from, to };
    }
  }
  const r = sp.get("range");
  if (r !== null && (PRESETS as readonly string[]).includes(r)) return { kind: "preset", preset: r as Preset };
  return DEFAULT_RANGE;
}

export function rangeQuery(sel: RangeSel): string {
  return sel.kind === "preset" ? `range=${sel.preset}` : `from=${sel.from}&to=${sel.to}`;
}

/** A preset covers exactly `span` whole minutes ending with the current,
 *  partial one (the mockup's charts include minute NOW). The server turns
 *  `{from, to}` into `[floor(from/60), ceil(to/60))`, so `from` is the start of
 *  the minute `span - 1` minutes before the current one and `to` is now.
 *  A second-precision `from = now - span` would make that 1441 minutes for
 *  24h and flip the tick labels from `HH:MM` to `Sep 18 HH:MM`; `to` is not
 *  rounded up because a `to` past the server's own clock is merely clamped
 *  to it (clock-skew ruling, 2026-09-18), which would silently shrink the
 *  window rather than widen it as a rounded-up `to` intends. */
export function resolveWindow(sel: RangeSel, nowSec: number): { from: number; to: number } {
  if (sel.kind === "custom") return { from: sel.from, to: sel.to };
  // ceil(now/60) is floor(now/60) + 1 except on an exact minute, where the
  // current minute has no elapsed second yet and the window ends at `now`.
  return { from: Math.ceil(nowSec / 60) * 60 - PRESET_MINUTES[sel.preset] * 60, to: nowSec };
}

/** Spec §4.4: 10 s for 1h/6h, 60 s for 24h/7d, none for Custom. */
export function pollIntervalMs(sel: RangeSel): number {
  if (sel.kind === "custom") return 0;
  return sel.preset === "1h" || sel.preset === "6h" ? 10_000 : 60_000;
}

const SPAN_WORDS: Record<Preset, string> = {
  "1h": "last hour",
  "6h": "last 6 hours",
  "24h": "last 24 hours",
  "7d": "last 7 days",
};

export function spanWords(sel: RangeSel): string | null {
  return sel.kind === "preset" ? SPAN_WORDS[sel.preset] : null;
}

/** Mockup: w < 60 ? `${w} min bins` : w < 1440 ? `${w/60} h bins` : w === 1440 ? "1 day bins" : "1 week bins". */
export function binLabel(w: number): string {
  if (w < 60) return `${w} min bins`;
  if (w < 1440) return `${w / 60} h bins`;
  if (w === 1440) return "1 day bins";
  return "1 week bins";
}

// ---- timing sample footnote (spec §4.4, amendment d913ddd) -------------------

export function ordinal(n: number): string {
  const t = n % 100;
  if (t >= 11 && t <= 13) return `${n}th`;
  switch (n % 10) {
    case 1: return `${n}st`;
    case 2: return `${n}nd`;
    case 3: return `${n}rd`;
    default: return `${n}th`;
  }
}

/** Spec: "Timings from every <stride>th request (<used> of <total>)." — shown
 *  only when the server sampled (stride > 1). The spec's "<stride>th" is a
 *  template; English ordinals are used so stride 2 reads "2nd", not "2th". */
export function timingSampleLine(ts: TokenSeries["timing_sample"] | undefined): string | null {
  if (!ts || ts.stride <= 1) return null;
  return `Timings from every ${ordinal(ts.stride)} request (${ts.used.toLocaleString("en-US")} of ${ts.total.toLocaleString("en-US")}).`;
}

// ---- fetching ---------------------------------------------------------------

export interface SeriesQuery {
  from: number;
  to: number;
  chain: boolean;
  maxBins: number;
  /** `false` asks the server to skip the latency timings (`timings=0`). The
   *  history strip draws only token volume, and each strip request covers the
   *  key's whole life (#251). Omitted → the server's default (timings on). An
   *  API that predates the parameter ignores it. */
  timings?: boolean;
  /** `false` asks the server to skip the per-model split (`by_model=0`);
   *  the history strip never shows it. Omitted → on. */
  byModel?: boolean;
}

export function seriesUrl(id: string, q: SeriesQuery): string {
  return (
    `/api/tokens/${encodeURIComponent(id)}/series` +
    `?from=${Math.floor(q.from)}&to=${Math.floor(q.to)}&chain=${q.chain ? 1 : 0}&max_bins=${q.maxBins}` +
    (q.timings === false ? "&timings=0" : "") +
    (q.byModel === false ? "&by_model=0" : "")
  );
}

export class SeriesError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = "SeriesError";
    this.status = status;
    this.detail = detail;
  }
}

// Handles every `detail` shape the backend can return. F7 reuses this rather
// than duplicating it (controller ruling), and so does everything that talks to
// a route which can answer 422 — the model write path (#262) made that list a
// lot longer, and each site that re-derived this got one of the three shapes
// wrong:
//
//   * a plain string           — `HTTPException(409, "...")`
//   * a list of field errors   — any 422, from FastAPI's body parser or from
//                                `ModelChangeRefused`. `String(detail)` on one
//                                of these renders `[object Object]`.
//   * an object with `message` — the structured refusals (`session_only`,
//                                the stress route's `_conflict` bodies)
export function detailOf(body: unknown): string | null {
  if (!body || typeof body !== "object" || !("detail" in body)) return null;
  const d = (body as { detail: unknown }).detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    const msgs = d
      .map((x) => (x && typeof x === "object" && "msg" in x ? String((x as { msg: unknown }).msg) : null))
      .filter((m): m is string => m !== null);
    return msgs.length ? msgs.join("; ") : null;
  }
  if (d !== null && typeof d === "object" && "message" in d) {
    const m = (d as { message: unknown }).message;
    if (typeof m === "string") return m;
  }
  return null;
}

export async function fetchSeries(url: string): Promise<TokenSeries> {
  const r = await authFetch(url);
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new SeriesError(r.status, detailOf(body) ?? `HTTP ${r.status}`);
  }
  return (await r.json()) as TokenSeries;
}

// ---- shaping ------------------------------------------------------------------

export function emptyBin(minute: number): SeriesBin {
  return {
    minute,
    requests_per_min: 0,
    prompt_per_min: 0,
    completion_per_min: 0,
    peak_prompt: 0,
    peak_completion: 0,
    requests: 0,
    n: 0,
    queue_p50: null,
    queue_p95: null,
    ttft_p50: null,
    ttft_p95: null,
    duration_p50: null,
    duration_p95: null,
  };
}

/** The server returns only bins with data (spec §3.4); the charts need every
 *  bin so an idle stretch draws as zero and a gap in timings draws as a gap.
 *  Bins are UTC-aligned: floor(minute / w) * w. */
export function fillBins(s: TokenSeries): SeriesBin[] {
  const w = s.bin_minutes;
  const byMinute = new Map(s.bins.map((b) => [b.minute, b]));
  const out: SeriesBin[] = [];
  for (let m = Math.floor(s.from_minute / w) * w; m < s.to_minute; m += w) {
    out.push(byMinute.get(m) ?? emptyBin(m));
  }
  return out;
}

// ---- per model (0036) ---------------------------------------------------------

/** Per-bin, per-model rates keyed by bin minute. */
export function modelRatesByBin(s: TokenSeries): Map<number, Record<string, ModelRates>> {
  return new Map(s.model_bins.map((b) => [b.minute, b.models]));
}

/** A bin carries per-model data only if it starts at or after the first
 *  per-model row: a bin straddling it would under-count every model. */
export function binIsSplit(minute: number, bySinceSec: number | null): boolean {
  return bySinceSec != null && minute * 60 >= bySinceSec;
}

/** The window starts before per-model recording began. */
export function startsBeforeByModel(s: TokenSeries): boolean {
  return s.by_model_since != null && s.from_minute * 60 < s.by_model_since;
}

const BACKEND_NAMES: Record<string, string> = { vllm: "vLLM", llamacpp: "llama.cpp" };

/** "32k ctx" for a power-of-two-ish context, else the compact count. */
function ctxLabel(n: number): string {
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k ctx`;
  if (n >= 1000) return `${Math.round(n / 1000)}k ctx`;
  return `${n} ctx`;
}

const HEX_SHA = /^[0-9a-f]{40}$/;

/** The card's compact variant line, e.g. "vLLM 0.11.0 · awq · rev a1b2c3d ·
 *  32k ctx". The engine version is the pinned vLLM version, else the version
 *  the in-container engine had at launch, else the tag of the image that ran;
 *  the revision is the commit it resolved to (short), else a non-default ref.
 *  `main` and `auto` are defaults and left out. */
export function variantLabel(v: ModelVariantUsage): string {
  const parts: string[] = [];
  const backend = v.backend ? (BACKEND_NAMES[v.backend] ?? v.backend) : null;
  const version = v.engine_vllm_version ?? v.engine_version ?? v.engine_image_tag;
  const engine = [backend, version].filter(Boolean).join(" ");
  if (engine) parts.push(engine);
  if (v.quantization) parts.push(v.quantization);
  if (v.dtype && v.dtype !== "auto") parts.push(v.dtype);
  const rev = v.hf_commit && HEX_SHA.test(v.hf_commit) ? v.hf_commit : v.hf_revision;
  if (rev && rev !== "main") parts.push(`rev ${rev.slice(0, 7)}`);
  if (v.max_model_len) parts.push(ctxLabel(v.max_model_len));
  return parts.length ? parts.join(" · ") : "default settings";
}

/** Variant lines for one model, made unique: when two variants summarise to
 *  the same line (they differ only in fields the line leaves out, such as
 *  extra args), each gets its short id appended. */
export function variantLabels(vs: ModelVariantUsage[]): string[] {
  const base = vs.map(variantLabel);
  return base.map((l, i) => (base.indexOf(l) !== base.lastIndexOf(l) ? `${l} · #${vs[i].variant_id.slice(0, 7)}` : l));
}
