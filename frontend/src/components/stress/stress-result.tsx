"use client";

/**
 * Done and failed phases of the stress-test modal.
 *
 * The whole feature exists to answer one operator question — "I downloaded a
 * model; what settings should I use for it on this hardware?" — so the result
 * view LEADS with the setting and treats the breaking points as supporting
 * detail, not the other way round.
 *
 * Two things this view must never do:
 *
 *   1. Present `recommended_config` as if it were live. It describes a
 *      configuration the model is NOT currently running (design §6.2);
 *      publishing it as live would tell clients they can send 96k-token
 *      prompts to an engine allocated for 32k, which the engine will then
 *      correctly refuse.
 *   2. Show a limit as a single scalar. The design publishes `value`,
 *      `raw_confirmed` and `first_observed_failure` together precisely
 *      because a probabilistic edge is not one number (§3.4).
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  LIMITED_BY_REMEDY,
  limitLabel,
  type StressLimit,
  type StressRun,
} from "./types";

function num(n: number | null | undefined): string {
  return typeof n === "number" ? n.toLocaleString("en-US") : "—";
}

function LimitedByNote({ limitedBy }: { limitedBy: string | null | undefined }) {
  if (!limitedBy) return null;
  const known = LIMITED_BY_REMEDY[limitedBy];
  return (
    <div className="mt-3 rounded-md border border-slate-700 bg-slate-900/40 p-3">
      <p className="text-xs text-slate-400">
        Limited by{" "}
        <span
          className="font-medium text-slate-200"
          data-testid="stress-limited-by"
        >
          {known?.label ?? limitedBy}
        </span>
      </p>
      {known && (
        <p className="mt-1 text-xs text-slate-400">{known.remedy}</p>
      )}
    </div>
  );
}

/** One limit, always as three numbers plus the oracle that produced them. */
function LimitRow({ name, limit }: { name: string; limit: StressLimit }) {
  const o = limit.oracle ?? null;
  return (
    <div
      className="rounded-md border border-slate-700 p-3"
      data-testid={`stress-limit-${name}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-slate-100">
          {limitLabel(name)}
        </span>
        <div className="flex items-center gap-2">
          {limit.provenance && (
            <Badge variant={limit.provenance === "stale" ? "warning" : "info"}>
              {limit.provenance}
            </Badge>
          )}
          <span className="font-mono tabular-nums text-slate-100">
            {num(limit.value)}
            {limit.unit ? ` ${limit.unit}` : ""}
          </span>
        </div>
      </div>
      {/* Three numbers, never one. `value` already carries the safety factor;
          showing it alone would hide how much of it is measurement and how
          much is judgement. */}
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-400">
        <dt>Confirmed at</dt>
        <dd className="text-right font-mono tabular-nums text-slate-300">
          {num(limit.raw_confirmed)}
        </dd>
        <dt>First observed failure</dt>
        <dd className="text-right font-mono tabular-nums text-slate-300">
          {limit.first_observed_failure == null
            ? "none observed"
            : num(limit.first_observed_failure)}
        </dd>
        {limit.gate_tripped && (
          <>
            <dt>Gate tripped</dt>
            <dd className="text-right font-mono text-slate-300">
              {limit.gate_tripped}
            </dd>
          </>
        )}
        {o?.n_consecutive != null && (
          <>
            <dt>Oracle</dt>
            <dd className="text-right font-mono text-slate-300">
              {o.n_consecutive}/{o.n_consecutive} consecutive
              {o.contour_p != null
                ? `, p ≈ ${Math.round(o.contour_p * 100)}%`
                : ""}
              {o.safety_factor != null ? `, ×${o.safety_factor}` : ""}
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}

export function StressResult({
  run,
  currentMaxModelLen,
  onClose,
  onRerun,
  onApply,
  applyPending,
  applyError,
}: {
  run: StressRun;
  /** The model's live max_model_len, so the headline can say what changes. */
  currentMaxModelLen: number | null;
  onClose: () => void;
  /**
   * Apply the recommended `max_model_len` and reload (design §6.4).
   *
   * Offered only where there is a recommendation AND it differs from what is
   * running: reloading to arrive at the setting already in force costs the
   * model its availability for the length of a load and changes nothing.
   * Optional, so the component still renders where applying is not possible.
   */
  onApply?: () => void;
  /** True while the apply request is in flight; disables the button. */
  applyPending?: boolean;
  /**
   * Why the last apply failed, if it did.
   *
   * Rendered rather than swallowed: an apply that silently does nothing leaves
   * the operator believing the model was reloaded with a setting it never
   * received.
   */
  applyError?: string | null;
  /**
   * Discard these results and measure again.
   *
   * Offered rather than taken. A run that merely FAILED must not withdraw a
   * good older measurement — that is why the published record falls back past
   * a valueless run. An operator who has READ the results and asked to replace
   * them is the opposite case, and only they can tell the two apart. Optional,
   * so the component still renders where no rerun is possible.
   */
  onRerun?: () => void;
}) {
  const rec = run.recommended_config ?? null;
  const limits = run.limits ?? {};
  const quality = limits.quality_context ?? null;
  const recLen = rec?.max_model_len ?? null;

  return (
    <div className="space-y-4 text-sm" data-testid="stress-done">
      {/* --- the headline: a setting, not a report ------------------------ */}
      {recLen != null ? (
        <div className="rounded-md border border-emerald-700 bg-emerald-900/20 p-4">
          <p
            className="text-base font-semibold text-emerald-200"
            data-testid="stress-recommendation"
          >
            Set max_model_len to {num(recLen)}
          </p>
          {currentMaxModelLen != null && (
            <p className="mt-1 text-xs text-emerald-300/80">
              {currentMaxModelLen === recLen
                ? "That is the current setting — no change needed."
                : `Currently ${num(currentMaxModelLen)}.`}
            </p>
          )}
          {/* Non-negotiable caveat: this is a config the model is not
              running. Applying it means a reload, and a reload invalidates
              the measurement that produced it (design §6.4). */}
          <p
            className="mt-2 text-xs text-slate-300"
            data-testid="stress-not-live-caveat"
          >
            This describes a configuration the model is{" "}
            <span className="font-semibold">not currently running</span>.
            Applying it reloads the model, which invalidates this measurement —
            re-run the test afterwards.
          </p>
          {rec?.next_failed_at != null && (
            <p className="mt-1 text-xs text-slate-400">
              The next candidate up, {num(rec.next_failed_at)}, failed
              {rec.gate_tripped ? ` (${rec.gate_tripped})` : ""}.
            </p>
          )}
          <LimitedByNote limitedBy={rec?.limited_by} />
          {onApply && currentMaxModelLen !== recLen ? (
            <div className="mt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={onApply}
                disabled={applyPending}
                data-testid="stress-apply"
              >
                {applyPending ? "Applying…" : "Apply and reload"}
              </Button>
              <p className="mt-1 text-xs text-slate-400">
                Unloads the model, saves {num(recLen)}, and loads it again. It
                will not serve until the load completes.
              </p>
              {applyError ? (
                <p
                  className="mt-1 text-xs text-red-300"
                  data-testid="stress-apply-error"
                >
                  Could not apply: {applyError}
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : quality?.value != null ? (
        <div className="rounded-md border border-slate-600 bg-slate-900/40 p-4">
          <p
            className="text-base font-semibold text-slate-100"
            data-testid="stress-recommendation"
          >
            Keep prompts under {num(quality.value)} tokens
          </p>
          <p className="mt-2 text-xs text-slate-400">
            This run measured the model as it is loaded now
            {currentMaxModelLen != null
              ? ` (max_model_len ${num(currentMaxModelLen)})`
              : ""}
            , so it can say how much of that context is usable but not whether a
            different setting would be better. Run <em>thorough</em> mode for a
            recommended max_model_len — it reloads the model at several context
            sizes.
          </p>
          <LimitedByNote limitedBy={quality.limited_by} />
        </div>
      ) : (
        <div
          className="rounded-md border border-amber-700 bg-amber-900/20 p-4 text-amber-200"
          data-testid="stress-recommendation"
        >
          <p className="font-semibold">No recommendation.</p>
          <p className="mt-1 text-xs text-amber-200/80">
            Nothing met the bar for publication. A fabricated setting would be
            worse than silence — see the run detail below.
          </p>
        </div>
      )}

      {/* --- caveats that survive a *completed* run ----------------------- */}
      {run.truncated_by && (
        <p
          className="rounded-md border border-amber-700 bg-amber-900/20 p-3 text-xs text-amber-200"
          data-testid="stress-truncated"
        >
          Partial: the run stopped early ({run.truncated_by}). Everything below
          is a lower bound — the real limit may be higher.
        </p>
      )}
      {run.traffic_observed && (
        <p
          className="rounded-md border border-amber-700 bg-amber-900/20 p-3 text-xs text-amber-200"
          data-testid="stress-tainted"
        >
          Tainted: this model was serving traffic during the run, so its KV pool
          was shared. These numbers are recorded but not published.
        </p>
      )}

      {/* --- supporting numbers ------------------------------------------ */}
      {Object.keys(limits).length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs uppercase tracking-wide text-slate-500">
            Measured limits
          </h3>
          {Object.entries(limits).map(([name, limit]) => (
            <LimitRow key={name} name={name} limit={limit} />
          ))}
        </div>
      )}

      <div className="flex items-center justify-between gap-3">
        {onRerun ? (
          <div className="min-w-0">
            <Button
              variant="outline"
              size="sm"
              onClick={onRerun}
              data-testid="stress-rerun"
            >
              Wipe and run again
            </Button>
            <p className="mt-1 text-xs text-slate-500">
              Discards the results above and starts a fresh measurement.
            </p>
          </div>
        ) : (
          <span />
        )}
        <Button variant="outline" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  );
}

interface Failure {
  title: string;
  detail: string;
}

/**
 * Which failure it was, because the operator's next move differs: a crash
 * budget exhausted means "the numbers you have are a lower bound", a neighbour
 * impact means "move the model off that GPU first", and a non-monotone verdict
 * means the measurement itself was unsound and nothing was published.
 */
export function describeFailure(run: StressRun): Failure {
  if (run.non_monotone) {
    return {
      title: "Measurement unsound — nothing published",
      detail:
        "A probe at the top of the ladder passed while a smaller one failed, so the failure region is not upward-closed and the search's assumption does not hold for this model. Publishing a value here could understate the real limit several-fold while carrying full confidence.",
    };
  }
  const by = run.truncated_by ?? "";
  if (by === "crash_budget") {
    return {
      title: "Crash budget exhausted",
      detail:
        "The engine died more times than this mode allows. Whatever was confirmed before the budget ran out is a lower bound, not the limit.",
    };
  }
  if (by === "wedged" || by === "engine_wedged") {
    return {
      title: "Engine wedged",
      detail:
        "Three probes in a row timed out while the engine still answered /health — a scheduler deadlock rather than a capacity limit. The engine was restarted and the axis abandoned.",
    };
  }
  if (by.includes("neighbour")) {
    return {
      title: "A co-resident model was affected",
      detail:
        "Another model sharing this GPU became unhealthy, so nothing measured here can be attributed to this model. Nothing was published. Move the models apart, or wait until the GPU is exclusively yours, and re-run.",
    };
  }
  if (by.includes("traffic") || run.traffic_observed) {
    return {
      title: "Traffic detected on this model",
      detail:
        "Requests were being served by this model during the run. They share the KV pool and confound every latency gate, so the measurement is not sound. Nothing was published.",
    };
  }
  if (run.status === "interrupted") {
    return {
      title: "Interrupted",
      detail:
        "The warden restarted while the run was in flight. The run is in-process by design — there is no resume — so it was marked interrupted and never published. Re-run it.",
    };
  }
  if (run.status === "failed_unrecovered") {
    return {
      title: "The engine did not come back",
      detail:
        "Recovery was attempted twice and the model is still not serving. This is not looped further on purpose — a restart loop is exactly what the crash budget exists to prevent. Check the model's logs before re-running.",
    };
  }
  return {
    title: "Aborted",
    detail:
      "The run stopped before it could confirm anything, so nothing was published.",
  };
}

export function StressFailure({
  run,
  onClose,
}: {
  run: StressRun;
  onClose: () => void;
}) {
  const { title, detail } = describeFailure(run);
  return (
    <div className="space-y-4 text-sm" data-testid="stress-failed">
      <div className="rounded-md border border-red-700 bg-red-900/20 p-4">
        <p
          className="font-semibold text-red-200"
          data-testid="stress-failure-title"
        >
          {title}
        </p>
        <p className="mt-1 text-xs text-red-200/80">{detail}</p>
      </div>
      {run.last_error && (
        <p
          className="rounded-md border border-slate-700 p-3 font-mono text-xs text-slate-300"
          data-testid="stress-failure-error"
        >
          {run.last_error}
        </p>
      )}
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  );
}
