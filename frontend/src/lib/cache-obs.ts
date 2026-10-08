// Wire types + pure helpers for prompt-cache observation
// (GET /api/stats/v2/cache, GET /api/tokens/{id}/cache; spec 2026-10-07 §5).
// Null means unknown and renders "—", never 0. DOM-free so it is testable alone.

export type CacheOutcome = "hit" | "partial" | "lost" | "misrouted" | "diverged" | "cold";
export const OUTCOMES: CacheOutcome[] = ["hit", "partial", "lost", "misrouted", "diverged", "cold"];

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
  /** Efficiency numerator: C over rows where both C and R are known. */
  reused_tokens: number;
  /** Efficiency denominator: max(R, C) over those same rows. */
  reusable_tokens: number;
  outcomes: Record<CacheOutcome, number>;
  /** Hit or partial requests whose prefix still broke deep inside the prompt
   *  (own lens: matched against this key's prompts only). */
  deep_diverged?: number;
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
  return { hit: "hit", partial: "partial", lost: "lost", misrouted: "other replica", diverged: "prefix changed", cold: "nothing to reuse" }[o];
}

function lostText(backend: string | null): string {
  if (backend === "llamacpp")
    return "Cache lost: the slot was overwritten or the request landed in a different slot. More --parallel slots, or fewer concurrent sessions, keep prefixes alive.";
  if (backend === "mlx") return "Cache lost: the prefix was evicted from the engine's prompt cache.";
  return "Cache lost to LRU eviction: the KV pool needed the space. Watch KV usage on this model.";
}

export function topProblem(s: CacheSummary, opts: { lens: "operator" | "own" }): Hint | null {
  const total = OUTCOMES.reduce((a, o) => a + s.outcomes[o], 0);
  if (total === 0) return null;
  const order: CacheOutcome[] = opts.lens === "own" ? ["diverged", "lost"] : ["lost", "diverged", "misrouted"];
  let best: Hint | null = null;
  for (const o of order) {
    const share = s.outcomes[o] / total;
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
    } else text = "The prefix was cached on another replica.";
    best = { outcome: o, share, text };
  }
  return best;
}
