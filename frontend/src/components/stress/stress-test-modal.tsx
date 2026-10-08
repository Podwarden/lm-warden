"use client";

/**
 * Stress-test modal for the model-detail page.
 *
 * Answers one operator question: "I downloaded a model — what settings should
 * I use for it on this hardware?" The headline output is a recommended
 * `max_model_len`; the breaking points are supporting detail.
 *
 * Four phases, following the phase machine in
 * `@/components/stats/cache-gc-button` (`:54`) — the closest precedent for a
 * long, phased, destructive operation:
 *
 *   confirm  — what this is about to do to a production engine, and an
 *              explicit acknowledgement. Not pre-ticked.
 *   running  — progress by POLLING the run row (design §9). Not SSE.
 *   done     — the recommendation first, the numbers second.
 *   failed   — which failure, because the operator's next move differs.
 *
 * Phase state is reset on close so the next open starts fresh, with one
 * deliberate exception: on open we ask the server whether a run is already
 * in flight for this model and adopt it. Closing this modal does not cancel
 * anything — the run is server-side — so a modal that forgot about it would
 * strand the operator with no way back to the progress they started.
 *
 * API (implemented in the routes slice of this branch):
 *   POST /api/models/{id}/stress       {mode, acknowledge_disruption, force}
 *                                      → 202 {run_id}
 *   GET  /api/models/{id}/capabilities → {limits, recommended_config, runs}
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { detailOf as detailFromBody } from "@/lib/token-series";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";
import { StressRunProgress } from "./stress-run-progress";
import { StressFailure, StressResult } from "./stress-result";
import {
  MODE_CHOICES,
  type CapabilitiesResponse,
  type StressMode,
  type StressRun,
  type StressStartResponse,
} from "./types";

type Phase = "confirm" | "running" | "done" | "failed";

interface StressTestModalProps {
  open: boolean;
  onClose: () => void;
  modelId: string;
  servedModelName: string;
  /** The live setting, so the result can say what would actually change. */
  maxModelLen: number | null;
  /** Poll cadence for the run row. Matches the detail page's SWR interval. */
  pollIntervalMs?: number;
}

/** Pull a human message out of an error body, whatever shape it arrived in.
 *
 * Delegates to the shared `detailOf` rather than re-deriving the shapes. This
 * helper used to know about strings and `{message}` only, so a 422 -- which
 * `POST /{id}/stress/apply` can now answer, because the write validates the
 * whole merged row -- rendered as a bare `HTTP 422` with nothing naming the
 * column to fix, on the one request that touches a production engine. */
async function detailOf(r: Response): Promise<string> {
  try {
    return detailFromBody(await r.json()) ?? `HTTP ${r.status}`;
  } catch {
    /* non-JSON body — fall back to the status code */
    return `HTTP ${r.status}`;
  }
}


/**
 * Attach the published record to the run it actually came from.
 *
 * `GET /capabilities` returns `limits` and `recommended_config` at the TOP
 * level, while `runs[]` carries only the summary — no limits, no
 * recommendation. `StressResult` reads them off the run, so without this
 * merge every completed run, successful ones included, renders "No
 * recommendation. Nothing met the bar for publication."
 *
 * The id guard is not defensive padding: the record is keyed on the
 * fingerprint, so after a run that published nothing it still holds an
 * EARLIER run's numbers, and merging unconditionally would present those as
 * this run's findings.
 */
export function withRecord(run: StressRun, body: CapabilitiesResponse): StressRun {
  if (!body.record_run_id || body.record_run_id !== run.id) return run;
  return {
    ...run,
    limits: body.limits ?? run.limits ?? null,
    recommended_config: body.recommended_config ?? run.recommended_config ?? null,
  };
}

export function StressTestModal({
  open,
  onClose,
  modelId,
  servedModelName,
  maxModelLen,
  pollIntervalMs = 2000,
}: StressTestModalProps) {
  const [phase, setPhase] = useState<Phase>("confirm");
  const [mode, setMode] = useState<StressMode>("conservative");
  // Explicit, never pre-ticked: this deliberately crashes a production
  // engine. A default-on acknowledgement is not an acknowledgement.
  const [acknowledged, setAcknowledged] = useState(false);
  const [force, setForce] = useState(false);
  const [resetRequested, setResetRequested] = useState(false);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  // Only revealed after the server refuses on a precondition, so `force` is
  // not an ambient option the operator can tick out of habit.
  const [conflict, setConflict] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<StressRun | null>(null);
  const [submitting, setSubmitting] = useState(false);
  // Synchronous double-submit guard, same reasoning as DeleteModelModal: the
  // `disabled` prop does not land in the DOM until the next render, and the
  // second POST would start (or 409 against) a second run.
  const inflight = useRef(false);
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  const adopt = useCallback((found: StressRun) => {
    setRun(found);
    setRunId(found.id);
    if (found.status === "running") setPhase("running");
    else if (found.status === "completed") setPhase("done");
    else setPhase("failed");
  }, []);

  // On open, adopt a run that is already in flight. A run started from this
  // modal and then closed out of, or one started in another tab, is still
  // burning GPU time; showing the confirm screen over the top of it would
  // invite a second run against the same engine.
  useEffect(() => {
    if (!open || runId !== null) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await authFetch(`/api/models/${modelId}/capabilities`);
        if (!r.ok || cancelled) return;
        const body = (await r.json()) as CapabilitiesResponse;
        const active = (body.runs ?? []).find((x) => x.status === "running");
        if (active && !cancelled) adopt(active);
      } catch {
        /* No in-flight run to adopt, or the endpoint is unreachable. Either
           way the confirm screen is the right thing to show; the POST will
           surface a real error if the API is genuinely down. */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, runId, modelId, adopt]);

  // Poll the run row while it is running. Deliberately a self-rescheduling
  // timeout rather than an interval: a slow poll must not stack up behind
  // itself while the engine is restarting after a crash.
  useEffect(() => {
    if (!open || phase !== "running" || !runId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function tick() {
      try {
        const r = await authFetch(`/api/models/${modelId}/capabilities`);
        if (cancelled) return;
        if (!r.ok) throw new Error(await detailOf(r));
        const body = (await r.json()) as CapabilitiesResponse;
        if (cancelled) return;
        const raw = (body.runs ?? []).find((x) => x.id === runId) ?? null;
        const found = raw ? withRecord(raw, body) : null;
        setPollError(null);
        if (found) {
          setRun(found);
          if (found.status !== "running") {
            setPhase(found.status === "completed" ? "done" : "failed");
            return;
          }
        }
      } catch (e) {
        if (cancelled) return;
        // Not fatal. The warden restarts the engine after a crash and the API
        // can blip; the run continues server-side either way. Surface it and
        // keep polling rather than declaring a failure we cannot see.
        setPollError(e instanceof Error ? e.message : String(e));
      }
      if (!cancelled) timer = setTimeout(tick, pollIntervalMs);
    }

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [open, phase, runId, modelId, pollIntervalMs]);

  async function start() {
    if (inflight.current) return;
    inflight.current = true;
    setSubmitting(true);
    setError(null);
    setConflict(null);
    try {
      const r = await authFetch(`/api/models/${modelId}/stress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          acknowledge_disruption: acknowledged,
          force,
          // Set only by "Wipe and run again": the server clears this model's
          // finished runs before starting, and skips the cooldown, which
          // otherwise protects a measurement the operator has just chosen to
          // discard.
          reset: resetRequested,
        }),
      });
      if (r.status === 409) {
        // A precondition the operator can knowingly override (traffic on this
        // model, another run on this GPU set). Reveal `force` and let them
        // decide, with the taint spelled out.
        setConflict(await detailOf(r));
        return;
      }
      if (!r.ok) {
        setError(await detailOf(r));
        return;
      }
      const body = (await r.json()) as StressStartResponse;
      setRunId(body.run_id);
      setRun(null);
      setPhase("running");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      inflight.current = false;
      setSubmitting(false);
    }
  }

  function handleClose() {
    onClose();
    // Fresh on next open. The adoption effect re-attaches to a still-running
    // run, so resetting here loses no state that matters.
    setPhase("confirm");
    setMode("conservative");
    setAcknowledged(false);
    setForce(false);
    setConflict(null);
    setError(null);
    setPollError(null);
    setRunId(null);
    setRun(null);
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Stress test"
      size="lg"
      initialFocusRef={phase === "confirm" ? cancelRef : undefined}
    >
      {error && (
        <p
          role="alert"
          className="mb-3 rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          data-testid="stress-error"
        >
          {error}
        </p>
      )}

      {phase === "confirm" && (
        <div className="space-y-4 text-sm">
          <div
            className="rounded-md border border-amber-700 bg-amber-900/20 p-3 text-amber-100"
            data-testid="stress-disruption-warning"
          >
            <p className="font-semibold">This will break the engine.</p>
            <p className="mt-1 text-xs text-amber-200/90">
              The test escalates prompt size and concurrency against{" "}
              <span className="font-mono">{servedModelName}</span> until it
              stops producing usable output. The engine may crash and be
              restarted; requests served by this model will fail while that
              happens. No mode can promise otherwise — a model has been
              measured dying at ~2,200 tokens with no degradation and no
              warning first.
            </p>
          </div>

          <fieldset className="space-y-2">
            <legend className="text-xs uppercase tracking-wide text-slate-500">
              Mode
            </legend>
            {MODE_CHOICES.map((c) => (
              <label
                key={c.mode}
                className="flex cursor-pointer items-start gap-3 rounded-md border border-slate-700 bg-slate-900/40 p-3"
              >
                <input
                  type="radio"
                  name="stress-mode"
                  value={c.mode}
                  checked={mode === c.mode}
                  onChange={() => setMode(c.mode)}
                  disabled={submitting}
                  className="mt-0.5 h-4 w-4 cursor-pointer border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-400"
                  data-testid={`stress-mode-${c.mode}`}
                />
                <span className="flex-1">
                  <span className="flex items-center justify-between gap-2">
                    <span className="font-medium text-slate-100">
                      {c.label}
                    </span>
                    <span className="font-mono text-xs text-slate-400">
                      {c.estimate}
                    </span>
                  </span>
                  <span className="mt-1 block text-xs text-slate-400">
                    {c.detail}
                  </span>
                </span>
              </label>
            ))}
          </fieldset>

          <label className="flex items-start gap-3 rounded-md border border-slate-700 bg-slate-900/40 p-3">
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
              disabled={submitting}
              className="mt-0.5 h-4 w-4 cursor-pointer rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-400"
              data-testid="stress-acknowledge"
            />
            <span className="flex-1 text-xs text-slate-300">
              I understand this may crash the engine and interrupt anything
              this model is serving.
            </span>
          </label>

          {conflict && (
            <div
              role="alert"
              className="space-y-2 rounded-md border border-amber-700 bg-amber-900/20 p-3 text-xs text-amber-100"
              data-testid="stress-conflict"
            >
              <p>{conflict}</p>
              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={force}
                  onChange={(e) => setForce(e.target.checked)}
                  disabled={submitting}
                  className="mt-0.5 h-4 w-4 cursor-pointer rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-400"
                  data-testid="stress-force"
                />
                <span>
                  Run anyway. The result is recorded as tainted and is not
                  published — a shared KV pool confounds every number this test
                  produces.
                </span>
              </label>
            </div>
          )}

          <div className="flex justify-end gap-2 pt-1">
            <Button
              ref={cancelRef}
              variant="ghost"
              size="sm"
              onClick={handleClose}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={start}
              disabled={!acknowledged || submitting || (conflict !== null && !force)}
              data-testid="stress-start"
            >
              {submitting ? "Starting…" : "Start stress test"}
            </Button>
          </div>
        </div>
      )}

      {phase === "running" && (
        <div className="space-y-4">
          <StressRunProgress run={run} pollError={pollError} />
          <div className="flex justify-end">
            <Button variant="outline" size="sm" onClick={handleClose}>
              Close
            </Button>
          </div>
          <p className="text-xs text-slate-500">
            Closing this does not stop the run. Reopen to pick the progress back
            up.
          </p>
        </div>
      )}

      {phase === "done" && run && (
        <StressResult
          run={run}
          currentMaxModelLen={maxModelLen}
          onClose={handleClose}
          applyPending={applying}
          applyError={applyError}
          onApply={async () => {
            setApplyError(null);
            setApplying(true);
            try {
              const r = await authFetch(`/api/models/${modelId}/stress/apply`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  run_id: run.id,
                  // The operator pressed a button labelled "Apply and reload"
                  // on a panel that says the model stops serving. That IS the
                  // acknowledgement; asking twice for the same consent trains
                  // people to click through it.
                  acknowledge_disruption: true,
                }),
              });
              if (!r.ok) throw new Error(await detailOf(r));
              // The measurement described the configuration just replaced, so
              // leaving it on screen would present it as describing what is
              // now running. Close instead.
              handleClose();
            } catch (e) {
              setApplyError(e instanceof Error ? e.message : String(e));
            } finally {
              setApplying(false);
            }
          }}
          onRerun={() => {
            // Back to the confirm screen, not straight to a POST: having read
            // a result does not imply the disruption acknowledgement, and this
            // run can still take the engine down.
            setResetRequested(true);
            setRun(null);
            setRunId(null);
            setPhase("confirm");
          }}
        />
      )}

      {phase === "failed" && run && (
        <StressFailure run={run} onClose={handleClose} />
      )}
    </Modal>
  );
}
