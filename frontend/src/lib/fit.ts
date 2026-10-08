// VRAM-fit classification — client mirror of `app/models/fit.py` (#85).
//
// The Add Model wizard re-runs this classifier on every GPU-checkbox tick
// so the row colour updates without a server round-trip. The first row
// render seeds budgets from `POST /api/models/fit-preview`; subsequent
// recomputes reuse the cached `weights_budget` from that response and
// only swap in a different `file_size` (or a recomputed budget when the
// caller has the per-GPU VRAM math handy).
//
// CONTRACT: constants and the verdict ladder MUST match
// `app/models/fit.py:21-23` and `app/models/fit.py:classify_fit` exactly.
// The contract test in `tests/contract/fit-classifier.test.ts` round-trips
// representative payloads through the server's `POST /api/models/fit-preview`
// and asserts that `classifyFit(file_size, weights_budget)` returns the
// same verdict as `response.verdict`. Drift in either direction breaks
// that test loudly.

export type FitVerdict = "green" | "yellow" | "orange" | "red";

// Thresholds locked by #82. Anything < 0.55 is "comfortably fits" (green),
// < 0.80 is "probably fits with some KV headroom" (yellow), < 1.0 is
// "tight — only short context will fit" (orange), >= 1.0 won't load (red).
export const GREEN_RATIO = 0.55;
export const YELLOW_RATIO = 0.80;
export const ORANGE_RATIO = 1.0;

// Sentinel the backend returns when the budget is non-positive (KV alone
// overflows VRAM). JSON can't carry +Infinity, so the FE recognises this
// magic number as "the ratio is effectively infinite" — see
// `app/models/routes_api.py` near the FitPreviewResponse construction.
export const RATIO_OVERFLOW_SENTINEL = 1e9;

// Target ratio for the client-side `recommendMaxModelLen` fallback solver:
// aim for "yellow" (well-inside-budget) rather than the orange boundary so a
// small misestimate of the weight footprint doesn't immediately tip the
// user back into "tight". Mirrors `app/models/fit.py:RECOMMENDATION_TARGET_RATIO`.
export const RECOMMENDATION_TARGET_RATIO = 0.70;

/**
 * Compute a `max_model_len` that lands the fit ratio in the yellow band,
 * client-side fallback for when the backend's `recommended_max_model_len`
 * field is absent. Mirrors `app/models/fit.py:recommend_max_model_len`
 * exactly so the two implementations stay in lockstep.
 *
 * `bytesPerToken` is the per-token KV footprint at the model's torch_dtype
 * and the operator's chosen `max_batch_size` (i.e. it already encodes the
 * batch multiplier — caller is responsible for that). This matches the way
 * the modal's tooltip math derives `bytes_per_token` from the backend's
 * `kv_reserve / max_model_len_used`, so the values flow through unchanged.
 *
 * Returns null when no positive L achieves the target ratio (model too big
 * for the selected GPUs at any context length).
 */
export function recommendMaxModelLen(args: {
  fileSize: number;
  capBytes: number;
  bytesPerToken: number;
  targetRatio?: number;
}): number | null {
  const target = args.targetRatio ?? RECOMMENDATION_TARGET_RATIO;
  if (target <= 0) return null;
  if (args.bytesPerToken <= 0) return null;
  const neededWeights = args.fileSize / target;
  const numerator = args.capBytes - neededWeights;
  if (numerator <= 0) return null;
  const L = Math.floor(numerator / args.bytesPerToken);
  return L >= 1 ? L : null;
}

/**
 * Verdict for a single candidate weights file.
 *
 * Mirrors `app.models.fit.classify_fit`:
 * - `budgetBytes <= 0`  → KV alone overflows the allowed VRAM → "red"
 * - `ratio < 0.55`      → "green"
 * - `ratio < 0.80`      → "yellow"
 * - `ratio < 1.0`       → "orange"
 * - otherwise           → "red"
 */
export function classifyFit(fileSizeBytes: number, budgetBytes: number): FitVerdict {
  if (budgetBytes <= 0) return "red";
  const ratio = fileSizeBytes / budgetBytes;
  if (ratio < GREEN_RATIO) return "green";
  if (ratio < YELLOW_RATIO) return "yellow";
  if (ratio < ORANGE_RATIO) return "orange";
  return "red";
}

/** Map a fit verdict to a Tailwind text/badge colour token. */
export function verdictBadgeClass(verdict: FitVerdict): string {
  switch (verdict) {
    case "green":
      return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300";
    case "yellow":
      return "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300";
    case "orange":
      // Tailwind doesn't ship "orange" badge tokens in our palette by
      // default; reuse amber with a stronger accent to keep the four-way
      // visual distinction.
      return "bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300";
    case "red":
      return "bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300";
  }
}

/** Short, human-readable label for the fit badge. */
export function verdictLabel(verdict: FitVerdict): string {
  switch (verdict) {
    case "green":
      return "fits";
    case "yellow":
      return "fits (tight headroom)";
    case "orange":
      return "tight";
    case "red":
      return "won't fit";
  }
}

/** Format a byte count as a short human string (e.g. "12.3 GiB", "412 MiB"). */
export function formatBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  if (n <= 0) return "0 B";
  const KIB = 1024;
  const MIB = KIB * 1024;
  const GIB = MIB * 1024;
  const TIB = GIB * 1024;
  if (n >= TIB) return `${(n / TIB).toFixed(2)} TiB`;
  if (n >= GIB) return `${(n / GIB).toFixed(2)} GiB`;
  if (n >= MIB) return `${(n / MIB).toFixed(1)} MiB`;
  if (n >= KIB) return `${(n / KIB).toFixed(0)} KiB`;
  return `${n} B`;
}
