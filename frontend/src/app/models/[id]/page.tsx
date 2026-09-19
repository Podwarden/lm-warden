"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import {
  backendDisplayName,
  fieldAppliesTo,
  versionPinControlApplies,
} from "@/lib/backend-fields";
import { useBackendCapability } from "@/lib/system-backends";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { LogStream } from "@/components/models/log-stream";
import { PullProgress } from "@/components/models/pull-progress";
import { DeleteModelModal } from "@/components/models/delete-model-modal";
import { ForceUnloadModal } from "@/components/models/force-unload-modal";
import { TryStackPanel } from "@/components/models/try-stack-panel";
import { StressTestModal } from "@/components/stress/stress-test-modal";
import { useBreadcrumb } from "@/lib/use-breadcrumb";

interface ModelDetail {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number | null;
  /** Which engine serves this model. NULL decodes to vLLM (D6). */
  backend: string | null;
  mmproj_filename: string | null;
  n_gpu_layers: number | null;
  dtype: string | null;
  max_model_len: number | null;
  gpu_memory_utilization: number | null;
  trust_remote_code: boolean;
  extra_args: string[];
  extra_env: Record<string, string>;
  status:
    | "registered"
    | "pulling"
    | "pulled"
    | "loading"
    | "loaded"
    | "unloading"
    | "failed";
  pulled_bytes: number | null;
  pulled_total: number | null;
  last_error: string | null;
}

type ModelStatus = ModelDetail["status"];

function badgeVariantForStatus(
  status: ModelStatus,
): "default" | "success" | "warning" | "error" | "info" {
  switch (status) {
    case "loaded":
    case "pulled":
      return "success";
    case "loading":
    case "pulling":
    case "unloading":
      return "info";
    case "failed":
      return "error";
    case "registered":
    default:
      return "default";
  }
}

// Backend gates on these in app/models/routes_api.py:
//  - load:   row.status must be in ("pulled", "failed")  → 409 otherwise
//  - unload: `_unloadable_statuses(force)`:
//              plain        → ("loaded", "failed")
//              ?force=true  → + ("loading", "unloading")          (#236)
//  - delete: row.status must NOT be in ACTIVE_STATUSES
//            ("loaded", "loading", "unloading", "pulling") → 409
//
// Mirroring those rules here keeps disabled buttons in sync with the server
// contract; the actual response still wins (we surface errors below) but a
// disabled button is the right UX hint before the round-trip.
//
// #244 — the mirror is a copy, and copies drift: canUnload kept the pre-#236
// gate after the server widened it, which left a row stranded in `loading`
// with no control at all. When one of these changes on the server, change
// it HERE in the same commit.
function canLoad(s: ModelStatus): boolean {
  return s === "pulled" || s === "failed";
}
function canUnload(s: ModelStatus): boolean {
  return s === "loaded" || s === "failed";
}
// The transient statuses the server accepts only with ?force=true. Offered
// as a distinct, confirmed action (ForceUnloadModal) rather than folded into
// canUnload: from these statuses it kills a process that may be mid-startup.
function needsForceUnload(s: ModelStatus): s is "loading" | "unloading" {
  return s === "loading" || s === "unloading";
}
function canDelete(s: ModelStatus): boolean {
  return !["loaded", "loading", "unloading", "pulling"].includes(s);
}
// The stress test probes a live engine over HTTP, so there has to be one:
// `loaded` and nothing else. `failed` is not enough (that row may have no
// process at all), and a `loading` row has no port assigned yet.
function canStressTest(s: ModelStatus): boolean {
  return s === "loaded";
}

export default function ModelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  // Next.js 15 dynamic params are exposed as a Promise so the framework can
  // suspend the page until they're known. `use()` unwraps inside a client
  // component without forcing the whole subtree to be a server component.
  const { id } = use(params);
  const router = useRouter();
  const { mutate } = useSWRConfig();
  const key = `/api/models/${id}`;
  const { data, error, isLoading } = useSWR<ModelDetail>(key, authFetchJSON, {
    // 2s aligns with the cadence the plan calls for. Pause polling while
    // the tab is hidden (mirror the list page's visibility guard) so the
    // detail tab doesn't keep hammering the API while the operator is in
    // another tab. Coming back triggers SWR's focus-revalidate so the
    // first frame after returning is fresh.
    refreshInterval: () =>
      typeof document !== "undefined" && document.hidden ? 0 : 2000,
  });
  // The breadcrumb strip and the back button name the model by its served
  // name, not its id. A model that failed to load (404 included) falls back
  // to the id so the crumb isn't stuck on the italic "Model" placeholder.
  useBreadcrumb({ title: data?.served_model_name ?? (error ? id : undefined) });
  // Does a version-pin control mean anything for the engine that serves this
  // row? The rule is @/lib/backend-fields' (hide what the engine has no
  // concept of; disable-and-explain what only this deployment cannot do); the
  // INPUT to it is the server's version_pin_reason_code, because whether a pin
  // is possible depends on the driver as well as the engine.
  //
  // `undefined` while the capability fetch is in flight, and
  // versionPinControlApplies(undefined) is true — so the card renders and then
  // disappears if it must, rather than appearing late on the deployments where
  // it belongs. Mirrors the panel's own "only lock once we KNOW" default.
  const backendCapability = useBackendCapability(data?.backend);
  const showTryStack = versionPinControlApplies(
    backendCapability?.version_pin_reason_code,
  );
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<null | "load" | "unload">(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [forceUnloadOpen, setForceUnloadOpen] = useState(false);
  const [stressOpen, setStressOpen] = useState(false);
  // Synchronous guard against double-clicks. setBusy is React state — it
  // doesn't take effect until the next render, which means a fast second
  // click can fire a concurrent POST before the disabled prop lands in
  // the DOM. The backend returns 409 on the duplicate call and we'd
  // flash a misleading error on what was actually a successful action.
  // A useRef flag is mutated in the same task as the first click, so the
  // second click sees the in-flight marker and bails out.
  //
  // S6 (epic/overhaul, #105): delete now goes through DeleteModelModal
  // which handles its own in-flight state, error surface, and the
  // optional "free cache" chained DELETE. The runAction path here only
  // handles load/unload.
  const inflight = useRef<"load" | "unload" | null>(null);

  async function runAction(action: "load" | "unload") {
    if (inflight.current) return;
    inflight.current = action;
    setActionError(null);
    setBusy(action);
    try {
      const r = await authFetch(`/api/models/${id}/${action}`, {
        method: "POST",
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = await r.json();
          if (body && typeof body.detail === "string") {
            detail = body.detail;
          } else if (
            body &&
            typeof body.detail === "object" &&
            body.detail !== null &&
            typeof body.detail.message === "string"
          ) {
            detail = body.detail.message;
          }
        } catch {
          /* non-JSON error body — fall back to status code */
        }
        setActionError(detail);
        return;
      }
      // For load/unload, just refresh the row so the new status reflects.
      await mutate(key);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      inflight.current = null;
      setBusy(null);
    }
  }

  async function handleDeleted() {
    // Modal's own busy guard prevented a double-click; the row is
    // gone, and we may have also cleared the cache row. Refresh the
    // list cache and navigate back. The modal stays mounted until the
    // navigation lands so its cache-failed notice can render.
    setDeleteOpen(false);
    await mutate("/api/models");
    router.push("/models");
  }

  // 404 — model was deleted out from under us, or the URL is bogus.
  // SWR's authFetchJSON throws an Error with .status; surface a clean
  // dead-end page rather than a generic crash.
  const status = (error as (Error & { status?: number }) | undefined)?.status;
  if (status === 404) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Model not found</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-400">
            The model <span className="font-mono">{id}</span> does not exist.
            It may have been deleted.
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading || (!data && !error)) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="space-y-4">
        <div className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200">
          Failed to load model
          {error instanceof Error ? `: ${error.message}` : "."}
        </div>
      </div>
    );
  }

  // After the loading/404/error guards above, `data` is defined. The TS
  // narrowing flow doesn't track this through `!data && !error` so help
  // the checker with a runtime guard that doubles as defensive depth.
  if (!data) return null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate text-2xl font-semibold">
            {data.served_model_name}
          </h1>
          <p className="mt-1 truncate text-xs text-slate-400">
            <span className="font-mono">{data.hf_repo}</span>
            <span className="mx-1 text-slate-500">@</span>
            <span className="font-mono">{data.hf_revision}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={() => runAction("load")}
            disabled={!canLoad(data.status) || busy !== null}
          >
            {busy === "load" ? "Loading…" : "Load"}
          </Button>
          {/* One slot, two controls. From `loading` / `unloading` the plain
              unload is refused (409) and only ?force=true gets the row out,
              so the slot shows a Force unload that opens a confirmation
              instead of a disabled Unload — a disabled button here is what
              left operators stranded (#244). It does NOT go through
              runAction: the modal owns its in-flight state and error, as
              DeleteModelModal does, so a refusal is read next to the
              explanation of what the action does. */}
          {needsForceUnload(data.status) ? (
            <Button
              size="sm"
              variant="destructive"
              onClick={() => setForceUnloadOpen(true)}
              disabled={busy !== null}
              data-testid="force-unload-button"
            >
              Force unload
            </Button>
          ) : (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => runAction("unload")}
              disabled={!canUnload(data.status) || busy !== null}
              data-testid="unload-button"
            >
              {busy === "unload" ? "Unloading…" : "Unload"}
            </Button>
          )}
          {/* Conditionally render the Link wrapper. A wrapped <Link> with a
              disabled <Button> still navigates: Button applies
              `disabled:pointer-events-none` on the inner <button> so the
              click falls through to the enclosing <a>. Splitting the two
              cases keeps the disabled state honest. */}
          {busy !== null ? (
            <Button size="sm" variant="outline" disabled>
              Settings
            </Button>
          ) : (
            <Link href={`/models/${id}/settings`}>
              <Button size="sm" variant="outline">
                Settings
              </Button>
            </Link>
          )}
          {/* Stress test sits between Settings and Delete so the destructive
              action stays last in the row. It does NOT go through runAction:
              that helper is load/unload-only (its `inflight` ref is typed to
              those two actions and its success path just revalidates the row),
              and starting a stress run is neither. The modal owns its own
              in-flight state, exactly as DeleteModelModal does. */}
          <Button
            size="sm"
            variant="outline"
            onClick={() => setStressOpen(true)}
            disabled={!canStressTest(data.status) || busy !== null}
            data-testid="stress-test-button"
          >
            Stress test
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => setDeleteOpen(true)}
            disabled={!canDelete(data.status) || busy !== null}
          >
            Delete
          </Button>
        </div>
      </div>

      <DeleteModelModal
        open={deleteOpen}
        onClose={() => setDeleteOpen(false)}
        modelId={data.id}
        servedModelName={data.served_model_name}
        hfRepo={data.hf_repo}
        onDeleted={handleDeleted}
      />

      {/* Mounted only while the status calls for it, so the `status` prop is
          narrowed and the modal never has to render copy for a status it
          cannot act on. */}
      {needsForceUnload(data.status) && (
        <ForceUnloadModal
          open={forceUnloadOpen}
          onClose={() => setForceUnloadOpen(false)}
          modelId={data.id}
          servedModelName={data.served_model_name}
          status={data.status}
          onUnloaded={() => {
            setForceUnloadOpen(false);
            void mutate(key);
          }}
        />
      )}

      <StressTestModal
        open={stressOpen}
        onClose={() => setStressOpen(false)}
        modelId={data.id}
        servedModelName={data.served_model_name}
        maxModelLen={data.max_model_len}
      />

      {actionError && (
        <div
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
        >
          {actionError}
        </div>
      )}

      {/* S1 (epic/overhaul) collapsed the Benchmark tab; the detail page
          now renders a single Overview view inline. If a future slice
          re-introduces multiple tabs, restore the <Tabs> wrapper from
          the pre-overhaul history (commit before this one). */}
      <div className="space-y-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
            <CardTitle>Status</CardTitle>
            <Badge variant={badgeVariantForStatus(data.status)}>
              {data.status}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-300">
            {/* Why a Force unload is on offer (#244). The page cannot tell a
                load that is two minutes into reading weights from a row the
                warden restarted out from under — both read `loading` — so it
                says what distinguishes them (Live logs) and what the way out
                is, instead of leaving a red button next to a spinner. */}
            {needsForceUnload(data.status) && (
              <div
                role="status"
                data-testid="force-unload-hint"
                className="rounded-md border border-amber-700 bg-amber-900/30 p-3 text-xs text-amber-200"
              >
                {data.status === "loading" ? (
                  <>
                    <span className="font-semibold">Still loading.</span>{" "}
                    Large models take minutes; watch Live logs below. If the
                    log has gone quiet, or the warden restarted mid-load, the
                    engine may no longer exist and this row is stranded —{" "}
                    <span className="font-semibold">Force unload</span> is
                    the way out.
                  </>
                ) : (
                  <>
                    <span className="font-semibold">Still unloading.</span>{" "}
                    Engine teardown normally finishes within a minute, even
                    for a large multi-GPU engine. If the row stays here, the
                    unload did not complete —{" "}
                    <span className="font-semibold">Force unload</span> kills
                    what is left and releases the row.
                  </>
                )}
              </div>
            )}
            {/* "GPUs: 1" on a model pinned to GPU index 1 reads as a count.
                These are indices -- say so. Singular/plural comes off the
                length so a one-card model does not read "indices". */}
            <p>
              {data.gpu_indices.length === 1 ? "GPU index:" : "GPU indices:"}{" "}
              {data.gpu_indices.length > 0 ? (
                <span className="font-mono" data-testid="model-gpu-indices">
                  {data.gpu_indices.join(", ")}
                </span>
              ) : (
                <span className="text-slate-500">none</span>
              )}
            </p>
            <p>
              Engine:{" "}
              <span className="font-mono" data-testid="model-engine">
                {backendDisplayName(data.backend)}
              </span>
            </p>
            {/* Per-backend visibility, shared with the model settings page via
                @/lib/backend-fields. Everything a DIFFERENT backend owns is
                hidden, not disabled: gpu_memory_utilization on a llama.cpp row
                is not "unavailable", it is a concept that engine does not have.
                `backend` reaches us from GET /api/models/{id} -- before that
                response carried it, every row here rendered as vLLM. */}
            {fieldAppliesTo("mmproj_filename", data.backend) && data.mmproj_filename && (
              <p>
                Vision projector:{" "}
                <span className="font-mono">{data.mmproj_filename}</span>
              </p>
            )}
            {/* `!= null` on purpose: NULL means "omit the flag, let llama.cpp
                auto-fit", and an undefined key (an older API) must not render
                an empty value either -- which is exactly what it did. */}
            {fieldAppliesTo("n_gpu_layers", data.backend) && data.n_gpu_layers != null && (
              <p>
                n_gpu_layers:{" "}
                <span className="font-mono" data-testid="model-n-gpu-layers">
                  {data.n_gpu_layers}
                </span>
              </p>
            )}
            {fieldAppliesTo("tensor_parallel_size", data.backend) &&
              data.tensor_parallel_size !== null && (
                <p>
                  Tensor parallel size:{" "}
                  <span className="font-mono">
                    {data.tensor_parallel_size}
                  </span>
                </p>
              )}
            {fieldAppliesTo("dtype", data.backend) && data.dtype && (
              <p>
                dtype: <span className="font-mono">{data.dtype}</span>
              </p>
            )}
            {data.max_model_len !== null && (
              <p>
                max_model_len:{" "}
                <span className="font-mono">{data.max_model_len}</span>
              </p>
            )}
            {fieldAppliesTo("gpu_memory_utilization", data.backend) &&
              data.gpu_memory_utilization !== null && (
                <p>
                  gpu_memory_utilization:{" "}
                  <span className="font-mono">
                    {data.gpu_memory_utilization}
                  </span>
                </p>
              )}
          </CardContent>
        </Card>

        {/* Try-stack (#162): trial-and-error engine-combo loop. Pins a
            (channel, vLLM version) onto the model, records the attempt, lets
            the operator report ok/failed (classifier suggests the next combo
            on failure), and saves a working combo as a reusable template.

            Absent entirely for an engine with no image catalogue. Both controls
            in it are vLLM's — a CUDA channel and a vLLM version — and there is
            no llama.cpp resolver, so honouring a pin on a llama.cpp row would
            resolve a vLLM image and launch vLLM under the operator's model
            name. That is not a control that is unavailable here; it is one this
            engine has no concept of, and the file this rule lives in says hide,
            not disable. */}
        {showTryStack && (
          <Card>
            <CardHeader>
              <CardTitle>Try stack</CardTitle>
            </CardHeader>
            <CardContent>
              <TryStackPanel
                modelId={id}
                hfRepo={data.hf_repo}
                maxModelLen={data.max_model_len}
                tensorParallelSize={data.tensor_parallel_size}
                modelStatus={data.status}
                backend={data.backend}
              />
            </CardContent>
          </Card>
        )}

        {/* PullProgress decides internally whether to render — keeps the
            detail page's layout straightforward (no conditional Card). */}
        {(data.status === "pulling" || data.status === "registered") && (
          <Card>
            <CardHeader>
              <CardTitle>Pull progress</CardTitle>
            </CardHeader>
            <CardContent>
              <PullProgress
                status={data.status}
                pulledBytes={data.pulled_bytes}
                pulledTotal={data.pulled_total}
              />
            </CardContent>
          </Card>
        )}

        {data.last_error && (
          <div
            role="alert"
            className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          >
            <span className="font-semibold">Last error:</span>{" "}
            {data.last_error}
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Live logs</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Key on modelId only — issue #52/#53. The earlier
                `${id}:${data.status}` key (commit 7df62a3,
                v17.2) forced LogStream to fully remount on
                every status transition so each flip got a fresh
                SSE connect + 200-line backfill. Cost: every
                remount tore down EventSource, minted a new
                ticket, and re-opened — and the Next.js rewrite
                proxy surfaces the upstream socket teardown as
                503 during the `pulled → loading` window, which
                the SSE hook then drives into reconnect/backoff
                while the panel sits blank.

                devops empirically verified the in-pod tail path
                is fine (file grows, fd tracks, curling SSE from
                inside the API container during an active load
                delivers live vLLM startup lines). The browser-
                facing symptom is the proxy churn caused by the
                status-keyed remount, NOT a backend tail bug.
                De-keying lets LogStream keep a single
                EventSource open across the lifecycle so the
                proxy never sees the teardown.

                The genuine remount case (operator switches to a
                different model — different id, different key)
                still triggers a fresh connect; the backend's
                per-connect 200-line backfill replays anything
                produced before the new mount.

                We hand the current status down to LogStream so
                it can skip opening the EventSource for non-log-
                producing states (`registered`) up front. */}
            <LogStream key={id} modelId={id} status={data.status} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
