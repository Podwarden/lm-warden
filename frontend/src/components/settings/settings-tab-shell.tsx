"use client";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { UseRuntimeSettingsResult } from "@/components/settings/hooks/use-runtime-settings";

// ---------------------------------------------------------------------------
// SettingsTabShell — the common chrome each of the four runtime tabs wears:
//
//   * title row with [Edit] (idle) or [Cancel] [Reset] [Save] (editing)
//   * restart banner (status, polite live region) under the title
//   * inline error banner (alert) for PATCH 4xx / network failures
//   * loading skeleton / error fallback for the initial GET
//
// Pure plumbing — every meaningful piece of state lives on the
// `useRuntimeSettings` hook, threaded through `controls`. The tab body
// (sections + fields) is supplied as `children`.
//
// Why split the shell out of each tab: the four tabs (General / Networking /
// Sessions / Maintenance) all need identical chrome. The pre-redesign
// runtime-tab.tsx hardcoded this 80-line block; copy-pasting it into four
// tabs would create a four-way drift surface for any future fix (banner
// copy, button arrangement, error styling).
// ---------------------------------------------------------------------------

interface SettingsTabShellProps {
  title: string;
  controls: UseRuntimeSettingsResult;
  children: React.ReactNode;
}

export function SettingsTabShell({
  title,
  controls,
  children,
}: SettingsTabShellProps) {
  const {
    draft,
    dirty,
    isLoading,
    error,
    editing,
    saving,
    saveError,
    restartBanner,
    edit,
    cancel,
    reset,
    save,
  } = controls;

  if (isLoading || (!draft && !error)) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error && !draft) {
    return (
      <div
        role="alert"
        className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200"
      >
        Failed to load settings
        {error instanceof Error ? `: ${error.message}` : "."}
      </div>
    );
  }

  if (!draft) return null;

  const canSave = editing && !saving && dirty.length > 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="text-xl font-semibold">{title}</h2>
        <div className="flex gap-2">
          {!editing && (
            <Button size="sm" onClick={edit}>
              Edit
            </Button>
          )}
          {editing && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={cancel}
                disabled={saving}
              >
                Cancel
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={reset}
                disabled={saving || dirty.length === 0}
              >
                Reset
              </Button>
              <Button size="sm" onClick={() => void save()} disabled={!canSave}>
                {saving ? "Saving…" : "Save"}
              </Button>
            </>
          )}
        </div>
      </div>

      {restartBanner.length > 0 && (
        <div
          role="status"
          aria-live="polite"
          className="space-y-1 rounded-md border border-amber-700 bg-amber-900/30 p-3 text-sm text-amber-200"
        >
          {restartBanner.map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>
      )}

      {saveError && (
        <div
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
        >
          {saveError}
        </div>
      )}

      <div className="space-y-6">{children}</div>
    </div>
  );
}
