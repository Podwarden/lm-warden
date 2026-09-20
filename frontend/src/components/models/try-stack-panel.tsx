"use client";

// Try-stack panel (#162) — per-model trial-and-error engine loop.
//
// An operator picks an engine combo (CUDA channel + vLLM version) and clicks
// Load. The panel issues two POSTs back-to-back: try-stack to pin the combo on
// the model + record a "pending" attempt, then /load to start the engine. The
// operator then reports whether the model came up (ok/failed); on failure the
// backend's classifier annotates the attempt with a category + a suggested
// next combo. Once an attempt is "ok", the combo can be saved as a reusable
// template.
//
// The merged Load click (was two steps: "Try combo" here + the Load button on
// the model card) keeps the operator inside the picker — they pick a combo,
// hit Load, and the engine starts. If the (channel, version) pair matches a
// previously-failed attempt, we surface a yellow banner above the button so
// the operator doesn't silently re-run an experiment that already failed.
//
// Two exports:
//   - TryStackHistory: presentational attempt list (unit-tested in isolation)
//   - TryStackPanel: container — SWR fetch + the load / report / save round-trips
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";
import { useBackendCapability } from "@/lib/system-backends";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Combobox, type ComboboxSuggestion } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import type { ModelStatus } from "./model-card";

// Only the CUDA channels resolve to an image today (app/templates/resolver.py);
// the others 400 on try-stack, so we don't offer them in the picker.
const CHANNELS = ["cuda-stable", "cuda-edge", "cuda-legacy"] as const;

export interface StackAttempt {
  id: string;
  channel: string;
  vllm_version: string;
  image: string | null;
  result: "pending" | "ok" | "failed";
  error: string | null;
  category: string | null;
  suggested_next: { suggestion: string } | null;
  created_at: string | null;
}

function resultVariant(r: StackAttempt["result"]): "success" | "error" | "info" {
  if (r === "ok") return "success";
  if (r === "failed") return "error";
  return "info";
}

export function TryStackHistory({
  attempts,
  onRetry,
}: {
  attempts: StackAttempt[];
  // Optional retry hook: when present, each historical row that isn't the
  // current pending attempt grows a small "Retry" link that refills the
  // picker with this row's combo. Hidden on a still-pending latest row
  // (retrying a not-yet-reported attempt makes no sense).
  onRetry?: (channel: string, version: string) => void;
}) {
  if (attempts.length === 0) {
    return <p className="text-sm text-slate-500">No attempts yet.</p>;
  }
  // The latest attempt sits at the end of the list (chronological). When it's
  // still pending, the operator hasn't reported a result yet — offering Retry
  // on it would re-pin the same combo with no new signal, so suppress.
  const latestIdx = attempts.length - 1;
  return (
    <ul className="space-y-2" data-testid="try-stack-history">
      {attempts.map((a, i) => {
        const isLatestPending = i === latestIdx && a.result === "pending";
        return (
          <li
            key={a.id}
            className="rounded-md border border-slate-700 bg-slate-900/40 p-3 text-sm"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-slate-200">
                {a.channel} · vLLM {a.vllm_version}
              </span>
              <div className="flex items-center gap-2">
                {onRetry && !isLatestPending && (
                  <button
                    type="button"
                    className="text-xs text-emerald-400 underline-offset-2 hover:underline focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-0"
                    onClick={() => onRetry(a.channel, a.vllm_version)}
                    data-testid={`try-stack-retry-${a.id}`}
                  >
                    Retry
                  </button>
                )}
                <Badge variant={resultVariant(a.result)}>{a.result}</Badge>
              </div>
            </div>
            {a.error && <p className="mt-1 text-red-300">{a.error}</p>}
            {a.suggested_next?.suggestion && (
              <p className="mt-1 text-amber-300">
                Suggested next: {a.suggested_next.suggestion}
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

interface TryStackResponse {
  attempts: StackAttempt[];
}

// #177: GET /api/templates/engine-versions?channel= — the published
// vllm/vllm-openai semver tags that resolve to a real image for a channel.
// Empty for non-resolvable channels or a Docker Hub hiccup (never 500s).
interface EngineVersionsResponse {
  channel: string;
  family: string | null;
  versions: string[];
  error: string | null;
}

export function TryStackPanel({
  modelId,
  hfRepo,
  maxModelLen,
  tensorParallelSize,
  modelStatus,
  backend,
}: {
  modelId: string;
  hfRepo: string;
  maxModelLen: number | null;
  tensorParallelSize: number | null;
  // Current engine status of the model — drives the Load button's busy lock
  // so the operator can't fire /load while the engine is already starting,
  // running, or shutting down.
  modelStatus: ModelStatus;
  // The row's own backend. This panel is vLLM's image catalogue, so which
  // engine serves the row decides whether it renders at all — the model detail
  // page makes that call via versionPinControlApplies(); here it only selects
  // WHICH capability row to read the disabled-reason from.
  backend: string | null | undefined;
}) {
  const key = `/api/models/${modelId}/try-stack`;
  const { data, mutate } = useSWR<TryStackResponse>(key, authFetchJSON);
  // Memoize the fallback so downstream useMemos that depend on `attempts`
  // don't see a fresh `[]` identity on every render (lint warns otherwise).
  const attempts = useMemo(() => data?.attempts ?? [], [data]);
  const latest = attempts.length > 0 ? attempts[attempts.length - 1] : null;

  const [channel, setChannel] = useState<string>(CHANNELS[0]);
  const [version, setVersion] = useState("");

  // #177: published vLLM versions that resolve to an image for the selected
  // channel. SWR-keyed on `channel` so switching channels refetches; the
  // backend serves this from a 6h family-keyed cache, so it's cheap to call
  // and keeps the field a typeable combobox (escape hatch for an unpublished
  // version or a pinned digest the dropdown won't list). While loading or on
  // an empty/error result, the field still works as free text.
  const { data: engineVersions } = useSWR<EngineVersionsResponse>(
    `/api/templates/engine-versions?channel=${encodeURIComponent(channel)}`,
    authFetchJSON,
  );
  const versionSuggestions: ComboboxSuggestion[] = (engineVersions?.versions ?? []).map(
    (v) => ({ value: v, label: v }),
  );

  // #177: whether this deployment can honour a version pin FOR THIS ROW'S
  // ENGINE. Was /api/system/engine, which answers only for vLLM and only about
  // the driver; /api/system/backends answers per backend, resolved against the
  // active driver, and carries the sentence explaining a refusal.
  //
  // While loading (capability === undefined) we do NOT disable — the common
  // case is the capable docker driver, and disabling-then-enabling would flash
  // the controls. We only lock the selector once we KNOW it's false.
  const capability = useBackendCapability(backend);
  const versionSelectDisabled = capability?.version_pin_available === false;
  // Rendered verbatim. The client cannot write this sentence: one boolean does
  // not say whether the obstacle is the driver or the backend, and the two have
  // different remedies (routes_engine.py::_version_pin_obstacle).
  const versionSelectReason = capability?.version_pin_reason ?? null;
  const engineVersion = capability?.version ?? null;

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failureDetail, setFailureDetail] = useState("");
  // Duplicate-failed-attempt guard: when the selected (channel, version) pair
  // matches an existing failed attempt, the first Load click shows a warning
  // and arms `confirmRepeat`; the SECOND click proceeds. Changing channel or
  // version clears confirmRepeat — picking a different combo is a fresh ask.
  const [confirmRepeat, setConfirmRepeat] = useState(false);

  const trimmedVersion = version.trim();

  // Engine-busy lock: the model card's load is gated on status; mirror those
  // rules here so the panel's Load button doesn't fire /load while the
  // engine is already loading, loaded, or unloading. The operator must
  // unload first to try a different combo.
  const engineBusy =
    modelStatus === "loading" ||
    modelStatus === "loaded" ||
    modelStatus === "unloading";

  // Find a previously-failed attempt that exactly matches the selected combo.
  // The warning is per-combo (channel + vllm_version), not per-attempt, so a
  // version typo that happens to look like a past failure is fine.
  const existingFailedAttempt = useMemo(() => {
    if (!trimmedVersion) return null;
    for (let i = attempts.length - 1; i >= 0; i -= 1) {
      const a = attempts[i];
      if (
        a.result === "failed" &&
        a.channel === channel &&
        a.vllm_version === trimmedVersion
      ) {
        return a;
      }
    }
    return null;
  }, [attempts, channel, trimmedVersion]);

  // Any change to the picked combo clears the confirm gate — the warning the
  // gate exists for is keyed on the exact (channel, version) pair.
  useEffect(() => {
    setConfirmRepeat(false);
  }, [channel, trimmedVersion]);

  // Unified Load: pin the combo (POST try-stack), then start the engine
  // (POST /load). On any duplicate-failed combo the first click only arms
  // the confirmation banner — a second click actually submits.
  async function loadCombo() {
    if (busy || !trimmedVersion || engineBusy) return;
    if (existingFailedAttempt && !confirmRepeat) {
      setConfirmRepeat(true);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(key, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ channel, vllm_version: trimmedVersion }),
      });
      if (!r.ok) {
        // `detailOf`, not `String(d.detail)`: pinning an engine goes through
        // the model write path now (#265), so this can answer 422 with
        // FastAPI's LIST of field errors -- which stringifies to
        // `[object Object]` -- as well as the 409 it already could.
        const d = await r.json().catch(() => null);
        setError(detailOf(d) ?? `Failed (HTTP ${r.status})`);
        return;
      }
      // Pin succeeded — refresh attempts so the new "pending" row appears,
      // then issue /load. If /load fails the pending attempt stays pending
      // and the operator can still report it via Mark failed (no cleanup
      // needed; the backend's classifier will pick it up on the next pass).
      await mutate();
      const lr = await authFetch(`/api/models/${modelId}/load`, { method: "POST" });
      if (!lr.ok) {
        const d = await lr.json().catch(() => null);
        setError(detailOf(d) ?? `Failed to start engine (HTTP ${lr.status})`);
        return;
      }
      setConfirmRepeat(false);
    } finally {
      setBusy(false);
    }
  }

  // Report whether the pinned combo actually brought the model up. Drives the
  // backend's record_try_stack_result, which on "failed" runs the classifier
  // to annotate the attempt with a category + suggested next combo.
  async function reportResult(result: "ok" | "failed") {
    if (busy || !latest || latest.result !== "pending") return;
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { result };
      if (result === "failed" && failureDetail.trim()) {
        body.error = failureDetail.trim();
      }
      const r = await authFetch(`${key}/${latest.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        setError((d && typeof d === "object" && "detail" in d ? String(d.detail) : null) ?? `Failed to report result (HTTP ${r.status})`);
        return;
      }
      setFailureDetail("");
      await mutate();
    } finally {
      setBusy(false);
    }
  }

  async function saveAsTemplate() {
    if (!latest || latest.result !== "ok") return;
    setBusy(true);
    setError(null);
    try {
      const id = `${modelId}-${latest.channel}-${latest.vllm_version}`;
      // Capture the model's actual tuning when known so the saved template
      // reproduces the combo that worked; omit when unset (backend defaults).
      const body: Record<string, unknown> = {
        id,
        label: `${hfRepo || modelId} on ${latest.channel} vLLM ${latest.vllm_version}`,
        hf_repo: hfRepo,
        // #170: hand the backend the live model id so it sources the model's
        // actual extra_args + gpu_memory_utilization from the row (the panel
        // doesn't carry those props), instead of saving schema defaults.
        model_id: modelId,
        engine: {
          channel: latest.channel,
          vllm_version: latest.vllm_version,
          image: latest.image,
        },
      };
      if (maxModelLen != null) body.max_model_len = maxModelLen;
      if (tensorParallelSize != null) body.tensor_parallel_size = tensorParallelSize;
      const r = await authFetch("/api/models/templates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        setError((d && typeof d === "object" && "detail" in d ? String(d.detail) : null) ?? `Failed to save template (HTTP ${r.status})`);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      {versionSelectDisabled && versionSelectReason && (
        <p className="text-xs text-amber-300" data-testid="try-stack-driver-note">
          {versionSelectReason}
          {engineVersion ? ` Running ${engineVersion}.` : ""}
        </p>
      )}
      <div className="flex flex-wrap items-end gap-2">
        <label className="space-y-1">
          <span className="block text-xs text-slate-400">Channel</span>
          <select
            data-testid="try-stack-channel"
            className="rounded-md border border-slate-700 bg-slate-900 p-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
            value={channel}
            onChange={(e) => setChannel(e.target.value)}
            disabled={versionSelectDisabled}
          >
            {CHANNELS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-xs text-slate-400">vLLM version</span>
          {/* #177: typeable dropdown of published image-resolving versions.
              Stays free-text so an operator can still enter an unpublished
              version or a pinned digest the catalog won't list.

              The placeholder was the literal "0.20.0" -- the template
              registry's hardcoded default, which nothing on a running
              deployment has any reason to match. It rendered one line under a
              note reading "(vLLM 0.26.0)", so the panel stated two different
              engine versions about itself, and on the subprocess driver the
              field is disabled, so the operator could not correct it. Show
              what the deployment actually runs, or show nothing. */}
          <Combobox
            suggestions={versionSuggestions}
            value={version}
            onChange={setVersion}
            placeholder={engineVersion ?? ""}
            className="w-32"
            ariaLabel="vLLM version"
            data-testid="try-stack-version"
            disabled={versionSelectDisabled}
          />
        </label>
        <Button
          size="sm"
          onClick={() => void loadCombo()}
          disabled={busy || !trimmedVersion || versionSelectDisabled || engineBusy}
          data-testid="try-stack-submit"
        >
          {busy ? "Working…" : "Load"}
        </Button>
        {engineBusy && (
          <p
            className="basis-full text-xs text-amber-300"
            data-testid="try-stack-busy-note"
          >
            Model is currently {modelStatus}. Unload first to try a different
            combo.
          </p>
        )}
        {latest?.result === "ok" && (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void saveAsTemplate()}
            disabled={busy}
            data-testid="try-stack-save-template"
          >
            Save working combo as template
          </Button>
        )}
      </div>

      {existingFailedAttempt && (
        <div
          className="rounded-md border border-amber-600 bg-amber-900/30 p-3 text-sm text-amber-100"
          data-testid="try-stack-dup-warning"
        >
          You already tried{" "}
          <span className="font-mono">
            {existingFailedAttempt.channel} · vLLM{" "}
            {existingFailedAttempt.vllm_version}
          </span>{" "}
          on{" "}
          <span className="font-mono">
            {existingFailedAttempt.created_at ?? "an earlier attempt"}
          </span>
          . It failed:{" "}
          <span className="font-mono">
            {existingFailedAttempt.error ||
              existingFailedAttempt.category ||
              "no detail recorded"}
          </span>
          . Loading again will repeat the same attempt.
          {confirmRepeat && (
            <p
              className="mt-1 font-semibold text-amber-200"
              data-testid="try-stack-dup-confirm"
            >
              Click Load again to confirm.
            </p>
          )}
        </div>
      )}

      {latest?.result === "pending" && (
        <div
          className="flex flex-wrap items-end gap-2 rounded-md border border-slate-700 bg-slate-900/40 p-3"
          data-testid="try-stack-report"
        >
          <label className="space-y-1">
            <span className="block text-xs text-slate-400">
              Did the model come up? (combo {latest.channel} · vLLM {latest.vllm_version})
            </span>
            <Input
              data-testid="try-stack-report-error"
              value={failureDetail}
              onChange={(e) => setFailureDetail(e.target.value)}
              placeholder="Failure detail (optional, for failed)"
              className="w-72"
            />
          </label>
          <Button
            size="sm"
            onClick={() => void reportResult("ok")}
            disabled={busy}
            data-testid="try-stack-report-ok"
          >
            Mark working
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void reportResult("failed")}
            disabled={busy}
            data-testid="try-stack-report-failed"
          >
            Mark failed
          </Button>
        </div>
      )}

      {error && (
        <p className="text-sm text-red-400" data-testid="try-stack-error">
          {error}
        </p>
      )}

      <TryStackHistory
        attempts={attempts}
        onRetry={(c, v) => {
          setChannel(c);
          setVersion(v);
        }}
      />
    </div>
  );
}
