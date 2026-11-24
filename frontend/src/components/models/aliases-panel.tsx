"use client";

/**
 * The "Aliases" tab of /models (/models?tab=aliases) — a named, ordered list
 * of target models plus optional proxy-side settings (guard / thinking /
 * params). An alias is how clients survive a model swap: a stable name in
 * front of whatever is loaded, first-loaded-target fallback.
 *
 * Reads GET /api/aliases (targets arrive pre-resolved with served names and
 * status), and opens the create/edit modal (alias-editor) from the row and
 * the header button.
 */

import { useState } from "react";
import useSWR from "swr";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { AliasEditor } from "@/components/models/alias-editor";
import {
  AliasApiError,
  deleteAlias,
  listAliases,
  setAliasEnabled,
  type AliasOut,
} from "@/lib/aliases";
import { cn } from "@/lib/utils";

const STATUS_DOT: Record<string, string> = {
  loaded: "bg-emerald-400",
  pulled: "bg-emerald-400",
  loading: "bg-amber-400",
  pulling: "bg-amber-400",
  unloading: "bg-amber-400",
  failed: "bg-red-400",
  registered: "bg-slate-500",
};

function dotClass(status: string | null | undefined): string {
  return STATUS_DOT[status ?? ""] ?? "bg-slate-500";
}

function settingsSummary(a: AliasOut): string {
  const parts: string[] = [];
  if (a.guard !== "inherit") parts.push(`guard ${a.guard}`);
  if (a.thinking_policy !== "inherit") parts.push(`thinking ${a.thinking_policy}`);
  for (const [key, p] of Object.entries(a.params ?? {})) {
    const value = p?.value;
    const v =
      typeof value === "string" && value.length > 24 ? `${value.slice(0, 24)}…` : String(value);
    if (p?.mode === "cap") parts.push(`${key} ≤ ${v}`);
    else if (p?.mode === "force") parts.push(`${key} = ${v}`);
    else parts.push(`${key} = ${v} (if unset)`);
  }
  return parts.length > 0 ? parts.join(" · ") : "Inherits";
}

export function AliasesPanel() {
  const { data, error, isLoading, mutate } = useSWR(
    "/api/aliases",
    listAliases,
    {
      // A short poll so pull/load state transitions (the target's status
      // dot) never require a manual refresh.
      refreshInterval: 5000,
    },
  );
  const aliases = data ?? [];

  const [editor, setEditor] = useState<{ mode: "create" | "edit"; alias?: AliasOut } | null>(null);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function afterSave() {
    setEditor(null);
    void mutate();
  }

  async function toggleEnabled(a: AliasOut) {
    setActionError(null);
    try {
      await setAliasEnabled(a.id, !a.enabled);
      void mutate();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  }

  async function confirmDelete(a: AliasOut) {
    setActionError(null);
    try {
      await deleteAlias(a.id);
      setConfirmingDeleteId(null);
      void mutate();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold">Aliases</h2>
          <p className="mt-1 text-sm text-slate-400">
            Named entries clients call — each sends requests to the first loaded
            model in its target list, with optional settings.
          </p>
        </div>
        <Button variant="outline" onClick={() => setEditor({ mode: "create" })} data-testid="add-alias-button">
          Add alias
        </Button>
      </div>

      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )}

      {error && !isLoading && (
        <p className="text-sm text-red-500">
          Failed to load aliases{error instanceof Error ? `: ${error.message}` : "."}
        </p>
      )}

      {actionError && (
        <p role="alert" className="text-sm text-red-500">{actionError}</p>
      )}

      {!isLoading && !error && aliases.length === 0 && (
        <div
          data-testid="aliases-empty-state"
          className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-slate-700 bg-slate-900/30 px-6 py-10 text-center"
        >
          <p className="max-w-md text-sm text-slate-400">
            No aliases yet. An alias is a name clients call; it sends requests to the first loaded model in its list, with optional settings.
          </p>
        </div>
      )}

      {!isLoading && !error && aliases.length > 0 && (
        <div className="overflow-x-auto rounded-md border border-slate-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="px-3 py-2 font-medium">Name</th>
                <th className="px-3 py-2 font-medium">Targets</th>
                <th className="px-3 py-2 font-medium">Settings</th>
                <th className="px-3 py-2 font-medium">Enabled</th>
                <th className="px-3 py-2 font-medium"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {aliases.map((a) => (
                <tr key={a.id} data-testid={`alias-row-${a.id}`} className="border-b border-slate-800/60 last:border-0">
                  <td className="px-3 py-2 align-top font-mono text-slate-100">{a.name}</td>
                  <td className="px-3 py-2 align-top">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {a.targets.map((t) => (
                        <span
                          key={t.id}
                          title={t.status ?? undefined}
                          data-testid="alias-target"
                          className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900/60 px-2 py-0.5 text-xs text-slate-200"
                        >
                          <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", dotClass(t.status))} />
                          {t.served_model_name ?? t.id}
                        </span>
                      ))}
                      {a.targets.length === 0 && <span className="text-xs text-slate-500">no targets</span>}
                    </div>
                  </td>
                  <td className="px-3 py-2 align-top">
                    <span data-testid="alias-settings" className="text-xs text-slate-400">
                      {settingsSummary(a)}
                    </span>
                  </td>
                  <td className="px-3 py-2 align-top">
                    <button
                      type="button"
                      role="switch"
                      aria-checked={a.enabled}
                      aria-label={`Enabled for ${a.name}`}
                      data-testid={`switch-${a.id}`}
                      onClick={() => void toggleEnabled(a)}
                      className={cn(
                        "relative h-5 w-9 rounded-full transition-colors",
                        a.enabled ? "bg-emerald-600" : "bg-slate-700",
                      )}
                    >
                      <span
                        aria-hidden
                        className={cn(
                          "absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all",
                          a.enabled ? "left-4" : "left-0.5",
                        )}
                      />
                    </button>
                  </td>
                  <td className="px-3 py-2 align-top">
                    {confirmingDeleteId === a.id ? (
                      <div className="flex items-center gap-2 whitespace-nowrap">
                        <span className="text-xs text-red-300">Delete?</span>
                        <Button
                          size="sm"
                          variant="destructive"
                          data-testid={`confirm-delete-${a.id}`}
                          onClick={() => void confirmDelete(a)}
                        >
                          Delete
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          data-testid={`cancel-delete-${a.id}`}
                          onClick={() => setConfirmingDeleteId(null)}
                        >
                          Cancel
                        </Button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 whitespace-nowrap">
                        <Button
                          size="sm"
                          variant="ghost"
                          data-testid={`edit-alias-${a.id}`}
                          aria-label={`Edit alias ${a.name}`}
                          onClick={() => setEditor({ mode: "edit", alias: a })}
                        >
                          Edit
                        </Button>
                        <button
                          type="button"
                          aria-label={`Delete alias ${a.name}`}
                          data-testid={`delete-alias-${a.id}`}
                          onClick={() => setConfirmingDeleteId(a.id)}
                          className="rounded p-1.5 text-slate-400 hover:bg-slate-800 hover:text-red-400"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {editor && (
        <AliasEditor
          open
          mode={editor.mode}
          alias={editor.mode === "edit" ? editor.alias : null}
          onClose={() => setEditor(null)}
          onSaved={afterSave}
        />
      )}
    </div>
  );
}
