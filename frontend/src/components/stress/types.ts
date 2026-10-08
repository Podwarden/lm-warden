/**
 * Narrow, local view of the stress-test API.
 *
 * Deliberately NOT sourced from `@/lib/api-types.generated.ts`: that file is
 * generated from `openapi.json` and regenerated in CI, and the stress
 * endpoints land on a different slice of this branch. The model-detail page
 * already carries a hand-written `ModelDetail` interface for exactly this
 * reason — declare the fields this UI reads, tolerate the rest.
 *
 * Everything the UI does not strictly need is optional, so a response that is
 * one field ahead of (or behind) this file renders rather than throws. The
 * fields that are NOT optional are the ones without which there is nothing to
 * show: a run's `id` and `status`.
 *
 * Mirrors, on the server side:
 *   - `app/db/sql/0029_stress_runs.sql`   (run row: status, truncated_by,
 *     traffic_observed, non_monotone, limits, observations, recommended_config)
 *   - `app/stress/outcomes.py::Class`     (per-probe outcome classes)
 *   - `app/stress/sweep.py::Recommendation` (max_model_len, limited_by,
 *     next_failed_at, gate_tripped)
 *   - `app/stress/gates.py::Gate`         (which quality gate ended the climb)
 */

/** Design §8. Estimates are in `MODE_CHOICES` below, next to the mode. */
export type StressMode = "conservative" | "quick" | "thorough";

/** `stress_runs.status` CHECK constraint, verbatim. */
export type StressRunStatus =
  | "running"
  | "completed"
  | "aborted"
  | "interrupted"
  | "failed_unrecovered";

/** `app/stress/outcomes.py::Class` — "values are published, so they are wire contract". */
export type StressOutcomeClass =
  | "pass"
  | "degraded"
  | "refused"
  | "timeout"
  | "crashed"
  | "errored"
  | "inconclusive"
  | "backpressure";

/** `app/stress/sweep.py::Recommendation.limited_by`. */
export type LimitedBy = "quality" | "load" | "crash" | "model_ceiling";

/**
 * The replicated-oracle parameters (design §3.3), published as structured
 * fields rather than prose so a client can tell a 21%-contour number from a
 * 9%-contour one.
 */
export interface StressOracle {
  n_consecutive?: number | null;
  contour_p?: number | null;
  confirm_m?: number | null;
  safety_factor?: number | null;
}

/**
 * One measured limit. Three numbers, not one: `value` is the recommendation
 * (confirmed × safety factor), `raw_confirmed` is what actually passed M/M,
 * and `first_observed_failure` is the smallest value ever seen to fail. The
 * design publishes all three precisely because a probabilistic edge is not a
 * scalar — the UI must never collapse them back into one.
 */
export interface StressLimit {
  value: number | null;
  unit?: string | null;
  provenance?: string | null;
  confidence?: string | null;
  limited_by?: LimitedBy | string | null;
  gate_tripped?: string | null;
  oracle?: StressOracle | null;
  raw_confirmed?: number | null;
  first_observed_failure?: number | null;
}

/**
 * A configuration that measured better than the running one (design §6.2).
 *
 * Advisory. It describes a config the model is NOT currently running, which is
 * why it is kept in its own field with its own fingerprint rather than merged
 * into `limits`.
 */
export interface StressRecommendedConfig {
  max_model_len?: number | null;
  limited_by?: LimitedBy | string | null;
  next_failed_at?: number | null;
  gate_tripped?: string | null;
  fingerprint?: string | null;
}

/** One probe result, as rendered in the running-phase attempt list. */
export interface StressObservation {
  /** "length" | "concurrency" — which axis this probe moved. */
  axis?: string | null;
  /** The value probed on that axis (prompt tokens, or concurrent requests). */
  value?: number | null;
  cls: StressOutcomeClass | string;
  gate_tripped?: string | null;
  note?: string | null;
}

export interface StressRunProgress {
  phase?: string | null;
  completed?: number | null;
  total?: number | null;
  note?: string | null;
  /**
   * Seconds left IN THIS PHASE, from that phase's own measured pace.
   *
   * Never a whole-run estimate: a length rung is one HTTP probe while a sweep
   * candidate is an engine reload of up to ten minutes, so a pace carried
   * across phases would be wrong by orders of magnitude. Null whenever the
   * phase cannot bound its own remainder — the same rule the published limits
   * follow.
   */
  eta_s?: number | null;
  phase_elapsed_s?: number | null;
}

export interface StressRun {
  id: string;
  model_id?: string | null;
  mode?: StressMode | string | null;
  status: StressRunStatus;
  /** Why a run stopped short. A *completed* run can still be truncated. */
  truncated_by?: string | null;
  traffic_observed?: boolean | null;
  non_monotone?: boolean | null;
  limits?: Record<string, StressLimit> | null;
  /** Array, or `{probes: [...]}` — normalised by `normaliseObservations`. */
  observations?: unknown;
  recommended_config?: StressRecommendedConfig | null;
  started_at?: string | null;
  finished_at?: string | null;
  last_error?: string | null;
  progress?: StressRunProgress | null;
}

/** `GET /api/models/{id}/capabilities`. */
export interface CapabilitiesResponse {
  limits?: Record<string, StressLimit> | null;
  recommended_config?: StressRecommendedConfig | null;
  runs?: StressRun[] | null;
  /**
   * Which run the top-level `limits` / `recommended_config` came from.
   *
   * The published record is keyed on the FINGERPRINT, not on the latest run,
   * so a run that measured nothing publishable leaves an older run's numbers
   * standing here. Attributing those to the run just finished would be a lie
   * of exactly the kind this feature exists to avoid, so they are only ever
   * merged onto a run whose id matches.
   */
  record_run_id?: string | null;
}

/** `POST /api/models/{id}/stress` → 202. */
export interface StressStartResponse {
  run_id: string;
}

export interface ModeChoice {
  mode: StressMode;
  label: string;
  estimate: string;
  detail: string;
}

/**
 * Design §8. Note what none of these say: v1 called the first mode `safe` and
 * promised it "never provokes a crash", which its own evidence refutes — a
 * model died at ~2.2k tokens with no degradation phase at all. Conservative
 * *aborts* on the first crash; it does not prevent one, and the copy here
 * must never imply otherwise.
 */
export const MODE_CHOICES: ModeChoice[] = [
  {
    mode: "conservative",
    label: "Conservative",
    estimate: "~15 min",
    detail:
      "Quality ladder on prompt length; concurrency up to the first preemption. Aborts on the first crash rather than spending a crash budget.",
  },
  {
    mode: "quick",
    label: "Quick",
    estimate: "~30–45 min",
    detail:
      "Adds crash-seeking refinement on both axes and verifies the concurrency frontier. Tolerates up to 2 crashes.",
  },
  {
    mode: "thorough",
    label: "Thorough",
    estimate: "2.5–4 h",
    detail:
      "Adds a reload sweep: the model is reloaded at several max_model_len values to find the best one. This is the only mode that can recommend a different setting. Tolerates up to 4 crashes.",
  },
];

/**
 * The remedy differs by cause, and collapsing them throws away the actionable
 * half — a memory wall wants more VRAM or a smaller quant, a quality wall
 * wants a different model. Mirrors the reasoning in
 * `app/stress/sweep.py::choose_recommendation`.
 */
export const LIMITED_BY_REMEDY: Record<string, { label: string; remedy: string }> = {
  quality: {
    label: "quality",
    remedy:
      "Answers went bad above this point while the engine stayed healthy. More VRAM will not move it — a different model, or a less aggressive quantisation, will.",
  },
  load: {
    label: "memory at load",
    remedy:
      "The engine could not allocate its KV pool above this point, so it never served. More VRAM, a smaller quantisation, or fewer concurrent sequences.",
  },
  crash: {
    label: "crash",
    remedy:
      "The engine died above this point instead of refusing cleanly. Treat this as a hard ceiling for this model on this hardware; a different engine build or GPU may move it.",
  },
  model_ceiling: {
    label: "model ceiling",
    remedy:
      "Nothing failed. This is the largest context the model itself declares, so the hardware was never the binding constraint.",
  },
};

const OUTCOME_VARIANTS: Record<
  string,
  "default" | "success" | "warning" | "error" | "info"
> = {
  pass: "success",
  degraded: "warning",
  refused: "info",
  backpressure: "info",
  timeout: "warning",
  crashed: "error",
  errored: "error",
  inconclusive: "default",
};

export function badgeVariantForOutcome(
  cls: string,
): "default" | "success" | "warning" | "error" | "info" {
  return OUTCOME_VARIANTS[cls] ?? "default";
}

/**
 * `observations` is machine-written JSON owned by the runner slice, and its
 * envelope is not pinned down on this branch yet. Accept the two shapes it
 * can plausibly take — a bare array, or `{probes: [...]}` — and the two
 * spellings of the class field (`cls` matches the Python dataclass;
 * `class` is what a naive serialiser would emit).
 *
 * Anything unrecognised yields an empty list, which renders as "no probes
 * recorded yet" rather than a crashed modal on top of a crashed engine.
 */
export function normaliseObservations(raw: unknown): StressObservation[] {
  const rows = Array.isArray(raw)
    ? raw
    : raw && typeof raw === "object" && Array.isArray((raw as { probes?: unknown }).probes)
      ? ((raw as { probes: unknown[] }).probes)
      : [];
  const out: StressObservation[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const r = row as Record<string, unknown>;
    const cls = r.cls ?? r.class ?? r.outcome;
    if (typeof cls !== "string") continue;
    out.push({
      axis: typeof r.axis === "string" ? r.axis : null,
      value: typeof r.value === "number" ? r.value : null,
      cls,
      gate_tripped: typeof r.gate_tripped === "string" ? r.gate_tripped : null,
      note: typeof r.note === "string" ? r.note : null,
    });
  }
  return out;
}

/** Human labels for the limit keys the design names (§4.5, §7.5). */
export const LIMIT_LABELS: Record<string, string> = {
  quality_context: "Quality context",
  needle_recall_context: "Needle-recall context",
  capacity_context: "Capacity context",
  comfortable_concurrency: "Comfortable concurrency",
  recommended_concurrency: "Recommended concurrency",
  max_tokens_headroom: "max_tokens headroom",
};

export function limitLabel(key: string): string {
  return LIMIT_LABELS[key] ?? key.replace(/_/g, " ");
}
