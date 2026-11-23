// Wire types + pure helpers for prompt-cache observation
// (GET /api/stats/v2/cache, GET /api/tokens/{id}/cache; spec 2026-10-07 §5).
// Null means unknown and renders "—", never 0. DOM-free so it is testable alone.

// The last two are cobs2 miss causes (app/cache_obs/classify.py), appended in the server's order.
export type CacheOutcome =
  | "hit" | "partial" | "lost" | "misrouted" | "diverged" | "cold" | "compacted" | "harness_rewrite";
export type DivergenceCause = "compacted" | "harness_rewrite";
export const OUTCOMES: CacheOutcome[] = [
  "hit", "partial", "lost", "misrouted", "diverged", "cold", "compacted", "harness_rewrite",
];

export interface CacheSummary {
  model_id: string;
  model: string;
  backend: string | null;
  requests: number;
  prompt_tokens: number;
  /** Sum of every known C (measured or estimated). */
  cached_tokens: number;
  /** Prompt tokens of the rows whose C is known: the "from cache" denominator. */
  known_prompt_tokens: number;
  measured_requests: number;
  estimated_requests: number;
  /** B4: engine-measured cached tokens and the prompts of those same rows (no
   *  estimate mixed in). Absent from an older server. */
  measured_cached_tokens?: number;
  measured_prompt_tokens?: number;
  /** Efficiency numerator: C over rows where both C and R are known. */
  reused_tokens: number;
  /** Efficiency denominator: max(R, C) over those same rows. */
  reusable_tokens: number;
  outcomes: Record<CacheOutcome, number>;
  /** Hit or partial requests whose prefix still broke deep inside the prompt
   *  (own lens: matched against this key's prompts only). */
  deep_diverged?: number;
  /** cobs2: rows whose broken prefix lineage explains -- relabelled misses and
   *  deep-diverged hits/partials alike. Absent from an older server. */
  divergence_causes?: Record<DivergenceCause, number>;
  by_rank: {
    dp_rank: number;
    requests: number;
    cached_tokens: number;
    reused_tokens: number;
    reusable_tokens: number;
  }[];
  prefill_saved_s: number | null;
  rate_source: "learned" | "hint";
  /** Operator lens only. */
  top_diverging?: { token_id: string; token_name: string | null; requests: number }[];
  /** The engine's prefix-cache block in tokens when it is large enough to
   *  change the outcomes (a hybrid model on vLLM), else null. Absent from an
   *  older server. */
  engine_block_tokens?: number | null;
  /** The engine also drops the last matched block of every hit (EAGLE/MTP
   *  speculative decoding); outcomes already allow for it. */
  engine_drops_last_block?: boolean;
  /** Tokens per prefix-match step (vLLM --prefix-match-unit); equals
   *  engine_block_tokens when unset. Absent from an older server. */
  engine_match_unit_tokens?: number | null;
}

export interface CacheResponse {
  range: string;
  since_epoch: number;
  models: CacheSummary[];
}

export interface Hint {
  outcome: CacheOutcome;
  share: number;
  text: string;
}

export const HINT_SHARE = 0.15;

const known = (s: CacheSummary) => s.measured_requests + s.estimated_requests > 0;

/** C / P over the requests whose C is known; unknown C never counts as 0. */
export function fromCache(s: CacheSummary): number | null {
  return known(s) && s.known_prompt_tokens > 0 ? s.cached_tokens / s.known_prompt_tokens : null;
}

/** Reused / reusable over the same rows (the server applies R := max(R, C) for measured C
 * and caps an estimated C at R, so this never exceeds 1). */
export function efficiency(s: CacheSummary): number | null {
  return known(s) && s.reusable_tokens > 0 ? s.reused_tokens / s.reusable_tokens : null;
}

/** Requests that carry cache figures (measured or estimated). */
export function coveredRequests(s: CacheSummary): number {
  return s.measured_requests + s.estimated_requests;
}

/** Seconds of prefill saved; null (unknown) when no request was observed. */
export function prefillSaved(s: CacheSummary): number | null {
  return known(s) ? s.prefill_saved_s : null;
}

export function isEstimated(s: CacheSummary): boolean {
  return s.estimated_requests > 0;
}

export function fmtPct(x: number | null, estimated: boolean): string {
  if (x === null) return "—";
  return `${estimated ? "≈" : ""}${Math.round(x * 100)}%`;
}

export function outcomeLabel(o: CacheOutcome): string {
  return {
    hit: "hit", partial: "partial", lost: "lost", misrouted: "other replica", diverged: "prefix changed",
    cold: "nothing to reuse", compacted: "compacted", harness_rewrite: "harness rewrote history",
  }[o];
}

function lostText(backend: string | null): string {
  if (backend === "llamacpp")
    return "Cache lost: the slot was overwritten or the request landed in a different slot. More --parallel slots, or fewer concurrent sessions, keep prefixes alive.";
  if (backend === "mlx") return "Cache lost: the prefix was evicted from the engine's prompt cache.";
  return "Cache lost to LRU eviction: the KV pool needed the space. Watch KV usage on this model.";
}

const CAUSE_TEXT: Record<DivergenceCause, string> = {
  harness_rewrite: "most of these follow the harness rewriting earlier messages",
  compacted: "most of these follow a compaction",
};

/** The cause behind most broken prefixes (prefix changed, deep-diverged hits, and the
 *  relabelled ones), or null when unknown, tied or not a majority. The server counts
 *  divergence_causes over that same population, so the share is like for like. */
export function dominantCause(s: CacheSummary): DivergenceCause | null {
  const c = s.divergence_causes;
  if (!c) return null;
  const o = s.outcomes;
  const broke = (s.deep_diverged ?? 0) + (o.diverged ?? 0) + (o.compacted ?? 0) + (o.harness_rewrite ?? 0);
  const h = c.harness_rewrite ?? 0;
  const k = c.compacted ?? 0;
  // A tie names no cause: neither is "most".
  if (h === k) return null;
  const top: DivergenceCause = h > k ? "harness_rewrite" : "compacted";
  const n = Math.max(h, k);
  return n * 2 > broke ? top : null;
}

/** ";"-joined suffix for a line about broken prefixes, or "" when no cause dominates. */
export function causeSuffix(s: CacheSummary): string {
  const c = dominantCause(s);
  return c ? `; ${CAUSE_TEXT[c]}` : "";
}

const HARNESS_TEXT =
  "The client rewrote earlier messages (reminders, dates, dropped reasoning); the cache can't reuse past that point.";

export function topProblem(s: CacheSummary, opts: { lens: "operator" | "own" }): Hint | null {
  const total = OUTCOMES.reduce((a, o) => a + (s.outcomes[o] ?? 0), 0);
  if (total === 0) return null;
  // "compacted" is informational (expected after a compaction), never a problem.
  const order: CacheOutcome[] = opts.lens === "own"
    ? ["diverged", "harness_rewrite", "lost"]
    : ["lost", "diverged", "harness_rewrite", "misrouted"];
  let best: Hint | null = null;
  for (const o of order) {
    const share = (s.outcomes[o] ?? 0) / total;
    if (share < HINT_SHARE || (best && share <= best.share)) continue;
    let text: string;
    if (o === "lost") text = opts.lens === "own"
      ? "Your requests often find their cached prefix gone: other traffic or long pauses let the engine evict it."
      : lostText(s.backend);
    else if (o === "diverged") {
      if (opts.lens === "own") {
        text = "Your prompt changes near its start between calls. Move changing data (time, ids) to the end.";
      } else {
        const who = s.top_diverging?.[0];
        text = who
          ? `Key "${who.token_name ?? who.token_id}" rewrites the start of its prompt in ${who.requests} requests. Move changing data (time, ids) to the end.`
          : "A client rewrites the start of its prompt. Move changing data (time, ids) to the end.";
      }
      const cause = dominantCause(s);
      if (cause) text += ` ${CAUSE_TEXT[cause][0].toUpperCase()}${CAUSE_TEXT[cause].slice(1)}.`;
    } else if (o === "harness_rewrite") text = HARNESS_TEXT;
    else text = "The prefix was cached on another replica.";
    best = { outcome: o, share, text };
  }
  return best;
}

/** Plain wording for the engine's cache granularity, or null when it does not
 *  shape the outcomes. `short` fits one card line; `long` is the tooltip. */
export function engineBlockNote(s: CacheSummary | undefined): { short: string; long: string } | null {
  const b = s?.engine_block_tokens;
  if (!b) return null;
  const n = (x: number) => x.toLocaleString("en-US");
  const u = s?.engine_match_unit_tokens;
  const drops = !!s?.engine_drops_last_block;
  let short: string;
  let long: string;
  if (u && u < b) {
    short = `engine reuses in ${u}-token steps`;
    long = `The engine matches prompts every ${n(u)} tokens (its cache blocks are ${n(b)} tokens): the part of a prompt after the last ${n(u)}-token step is recomputed.`;
    if (drops) long += ` With speculative decoding it also recomputes the last ${n(u)} tokens it matched.`;
  } else {
    short = `engine caches in ${b >= 1000 ? `~${(b / 1000).toFixed(1)}k` : String(b)}-token blocks`;
    long = `The engine caches prompts in ${n(b)}-token blocks: the part of a prompt after its last whole block is always recomputed.`;
    if (drops)
      long += " With speculative decoding it also recomputes the last whole block it matched, so a prompt needs two blocks before anything is reused.";
  }
  long += " Requests that got everything the engine could serve count as hits.";
  return { short, long };
}
