"use client";

import { useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ModelCard, type ModelRow } from "@/components/models/model-card";
import { AddModelModal } from "@/components/models/add-model-modal";

interface ModelsResponse {
  models: ModelRow[];
}

export default function ModelsPage() {
  const [open, setOpen] = useState(false);
  const { data, error, isLoading } = useSWR<ModelsResponse>(
    "/api/models",
    authFetchJSON,
    // Poll the list every 5s so pull/load state transitions don't require
    // the operator to refresh manually. The detail page (Task 11.4) wires
    // a tighter SSE stream for pull progress on a single model.
    //
    // Pause polling while the tab is hidden — SWR accepts a function form
    // for refreshInterval and re-evaluates it per tick. Returning 0 here is
    // SWR's "do not poll" sentinel, which avoids hammering the API for a
    // tab nobody is looking at (and the visibilitychange-driven revalidate
    // on focus will catch the user up immediately when they come back).
    {
      // A number, not a function returning 0 while hidden: SWR stops polling
      // for good on a 0. It already skips ticks while the tab is hidden.
      refreshInterval: 5000,
    },
  );

  const models = data?.models ?? [];
  const hasModels = !isLoading && !error && models.length > 0;
  const isEmpty = !isLoading && !error && models.length === 0;

  return (
    <div className="space-y-4">
      {/* S4 design principle #2 (instrument-panel aesthetic): the Add-model
          control sits inline with the H1 only when the fleet is populated.
          On an empty fleet the CTA becomes a deliberate hero in the empty
          state below — operators land on /models the first time and need
          one obvious path forward, not a small button tucked in a corner. */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Models</h1>
        {hasModels && (
          <Button
            variant="outline"
            onClick={() => setOpen(true)}
            data-testid="add-model-button-inline"
          >
            Add model
          </Button>
        )}
      </div>

      {isLoading && (
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}

      {error && !isLoading && (
        <p className="text-sm text-red-500">
          Failed to load models{error instanceof Error ? `: ${error.message}` : "."}
        </p>
      )}

      {isEmpty && (
        <EmptyStateHero onAdd={() => setOpen(true)} />
      )}

      {hasModels && (
        <div className="grid gap-4 md:grid-cols-2">
          {models.map((m) => (
            <ModelCard key={m.id} model={m} />
          ))}
        </div>
      )}

      <AddModelModal open={open} onClose={() => setOpen(false)} />
    </div>
  );
}

/**
 * Empty-state hero (S4 #2). One generous CTA centered in a dashed-border card,
 * explaining what an operator gets out of adding a model. Replaces the
 * previous one-line "No models yet — add one." line which buried the action.
 *
 * Visual contract:
 *   - Outer dashed border + muted background — matches the existing card
 *     palette so theme overrides (retro / retro-dark) continue to apply.
 *   - Headline + one short body line + primary action button. No marketing
 *     copy — this is a tool, not a landing page.
 *   - Button is the default (filled) variant so it stands out against the
 *     dashed border without needing extra emphasis classes.
 */
function EmptyStateHero({ onAdd }: { onAdd: () => void }) {
  return (
    <div
      data-testid="models-empty-state"
      className="flex flex-col items-center justify-center gap-3 rounded-md border border-dashed border-slate-700 bg-slate-900/30 px-6 py-12 text-center"
    >
      <p className="text-base font-medium text-slate-200">No models registered yet.</p>
      <p className="max-w-md text-sm text-slate-400">
        Add a HuggingFace repo to begin — the wizard will discover files,
        suggest a starting configuration, and reserve GPUs.
      </p>
      <Button
        onClick={onAdd}
        data-testid="add-model-button-hero"
        className="mt-2"
      >
        Add your first model
      </Button>
    </div>
  );
}
