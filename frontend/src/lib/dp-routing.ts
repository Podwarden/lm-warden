// GET /api/models/{model_id}/dp-routing (#286). The types come from the
// generated client so the card cannot drift from the endpoint; every engine-side
// field is `number | null` (null: not loaded / scrape failed / rank absent),
// `since` is null until the first routed request, kv_cache_usage_perc and
// prefix_cache_hit_rate are fractions 0..1.

import type { components } from "@/lib/api-types.generated";

export type DpRank = components["schemas"]["DpRankRow"];
export type DpTotals = components["schemas"]["DpRoutingTotals"];
export type DpRoutingResponse = components["schemas"]["DpRoutingResponse"];

/** sticky / (sticky + spilled), or null when nothing has been routed. */
export function stickyShare(totals: Pick<DpTotals, "sticky" | "spilled">): number | null {
  const n = totals.sticky + totals.spilled;
  return n > 0 ? totals.sticky / n : null;
}
