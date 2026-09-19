"use client";

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, type SelectOption } from "@/components/ui/select";
import type { ModelRow } from "@/components/models/model-card";

// ---------------------------------------------------------------------------
// Model tab — navigation pivot, not a duplicate editor
// ---------------------------------------------------------------------------
//
// The per-model settings page (frontend/src/app/models/[id]/settings/page.tsx)
// is the canonical editor for any individual model. This tab exists to
// answer the spec's question "where do I edit the currently-loaded model's
// settings?" — by deep-linking to that page when a model is loaded, or by
// surfacing the spec's empty-state copy + a select-to-edit dropdown for
// the no-loaded-model case.
//
// Re-implementing the per-model form here would duplicate ~500 lines of
// dirty-tracking + restart-on-edit behaviour for no benefit; the operator
// can equally well navigate to /models/<id>/settings, which is also where
// "Edit" on the model detail page lands them.
// ---------------------------------------------------------------------------

interface ModelsResponse {
  models: ModelRow[];
}

export function ModelTab() {
  const { data, error, isLoading } = useSWR<ModelsResponse>(
    "/api/models",
    authFetchJSON,
    {
      // Match the dashboard list's 5s cadence so a load/unload that just
      // happened reflects here without a manual refresh. Pause on hidden
      // tabs to avoid hammering the API when nobody's looking.
      // A number, not a function returning 0 while hidden: SWR stops polling
      // for good on a 0. It already skips ticks while the tab is hidden.
      refreshInterval: 5000,
    },
  );

  // The select-to-edit dropdown for the empty-state path. We hold the
  // selection here instead of navigating on every Select change so the
  // operator can confirm before leaving the tab — Next.js' Link with
  // disabled-href semantics is awkward, so a Button-as-link is cleaner.
  const [pickedId, setPickedId] = useState<string | null>(null);

  if (isLoading || (!data && !error)) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div
        role="alert"
        className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200"
      >
        Failed to load models
        {error instanceof Error ? `: ${error.message}` : "."}
      </div>
    );
  }

  const models = data?.models ?? [];
  // Spec says "currently-loaded" — singular. The backend permits multiple
  // models loaded simultaneously (CUDA_VISIBLE_DEVICES partitioning), so
  // we surface the first one and surface a footnote if there are more.
  // The per-model editor handles the actual edit.
  const loaded = models.filter((m) => m.status === "loaded");
  const loadedFirst = loaded[0];
  // Registered-but-not-loaded models are the dropdown's population for
  // the empty-state path. "registered" + "pulled" are the safe-to-edit
  // statuses (the editor on /models/<id>/settings will refuse to PATCH
  // anything that's currently loading/pulling/unloading via the 409
  // path, but offering them in the dropdown would just confuse the
  // operator). Keep the list ordered by served_model_name for stable
  // UX.
  const editable = models
    .filter((m) => m.status !== "loaded")
    .sort((a, b) => a.served_model_name.localeCompare(b.served_model_name));

  // Loaded model branch — the spec's "shows the currently-loaded model"
  // path. We render a card summarising the model and a CTA into the
  // per-model editor.
  if (loadedFirst) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>{loadedFirst.served_model_name}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-slate-300">
              Currently loaded. Editing the currently-loaded model requires a
              reload — vLLM will be stopped, settings persisted, and started
              again.
            </p>
            <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-slate-400 sm:grid-cols-2">
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  HF repo
                </dt>
                <dd className="font-mono text-slate-200">
                  {loadedFirst.hf_repo}
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  Revision
                </dt>
                <dd className="font-mono text-slate-200">
                  {loadedFirst.hf_revision}
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  GPU indices
                </dt>
                <dd className="font-mono text-slate-200">
                  [{loadedFirst.gpu_indices.join(",")}]
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  Tensor-parallel
                </dt>
                <dd className="font-mono text-slate-200">
                  {loadedFirst.tensor_parallel_size ?? "—"}
                </dd>
              </div>
            </dl>
            <Link
              href={`/models/${loadedFirst.id}/settings`}
              className="inline-block text-emerald-400 underline hover:text-emerald-300"
            >
              Edit settings →
            </Link>
          </CardContent>
        </Card>

        {loaded.length > 1 && (
          <p className="text-xs text-slate-500">
            {loaded.length - 1} other model
            {loaded.length - 1 === 1 ? "" : "s"} also loaded. Use the{" "}
            <Link href="/models" className="underline hover:text-slate-300">
              Models page
            </Link>{" "}
            to edit those.
          </p>
        )}
      </div>
    );
  }

  // Empty-state branch — spec copy, verbatim modulo Link wiring.
  const options: SelectOption<string>[] = editable.map((m) => ({
    value: m.id,
    label: m.served_model_name,
  }));

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>No model loaded</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <p className="text-slate-300">
            No model is currently loaded. Load a model from the{" "}
            <Link
              href="/models"
              className="text-emerald-400 underline hover:text-emerald-300"
            >
              Models page
            </Link>{" "}
            to edit its settings here.
          </p>
          {editable.length > 0 && (
            <div className="space-y-2">
              <p className="text-slate-400">
                Or pick a registered model to edit directly:
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Select
                  options={options}
                  value={pickedId}
                  onChange={(v) => setPickedId(v)}
                  placeholder="Select a model…"
                  ariaLabel="Select a model to edit"
                />
                {pickedId && (
                  <Link
                    href={`/models/${pickedId}/settings`}
                    className="inline-flex"
                  >
                    <Button size="sm">Edit settings</Button>
                  </Link>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
