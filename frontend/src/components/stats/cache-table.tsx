"use client";

/**
 * Storage / HF cache table for the Stats page.
 *
 * Lists every ``models--*`` directory the backend scanner found, with
 * size, last-modified, and any DB rows that own the repo. Per-row
 * Delete button drives ``DELETE /api/cache/models/<repo>`` with a
 * two-stage confirm flow: the first 409 ("pulled-but-unloaded —
 * pass ?force=true") becomes a second confirm that re-submits with
 * ``?force=true``. Active rows are non-deletable (the backend
 * refuses regardless of ``force``) — the button is disabled with
 * an explanatory title.
 *
 * See vllm-warden#114 + ``docs/superpowers/specs/2026-05-20-hf-cache-management-design.md``.
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { authFetch } from "@/lib/auth-fetch";
import { formatBytes } from "@/lib/fit";

export interface MatchedModelRef {
  id: string;
  served_model_name: string;
  status: string;
}

export interface CachedRepoView {
  repo: string;
  path: string;
  size_bytes: number;
  last_modified: number;
  matched_models: MatchedModelRef[];
}

// Same set guarded by ``DELETE /api/cache/models/<repo>``. Surfaced here
// so the UI can disable the button up front instead of relying on the
// 409 round-trip — better UX for a destructive action.
const ACTIVE_STATUSES = new Set(["loaded", "loading", "unloading", "pulling"]);

function formatRelativeTime(epochSeconds: number): string {
  if (!Number.isFinite(epochSeconds) || epochSeconds <= 0) return "—";
  const ageMs = Date.now() - epochSeconds * 1000;
  if (ageMs < 0) return "just now";
  const minutes = Math.floor(ageMs / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

function rowStatus(matched: MatchedModelRef[]): { label: string; tone: "orphan" | "active" | "idle" | "failed" | "mixed" } {
  if (matched.length === 0) return { label: "orphan", tone: "orphan" };
  const statuses = new Set(matched.map((m) => m.status));
  if ([...statuses].some((s) => ACTIVE_STATUSES.has(s))) {
    return { label: "in use", tone: "active" };
  }
  if (statuses.size === 1 && statuses.has("failed")) {
    return { label: "failed", tone: "failed" };
  }
  if (statuses.size === 1 && (statuses.has("pulled") || statuses.has("idle"))) {
    return { label: "pulled", tone: "idle" };
  }
  return { label: [...statuses].join("/"), tone: "mixed" };
}

interface CacheTableProps {
  repos: CachedRepoView[];
  /** Called after a successful delete so the parent can revalidate
   *  the SWR cache and refresh the list. */
  onMutate: () => void;
}

export function CacheTable({ repos, onMutate }: CacheTableProps) {
  // Tracks the repo currently mid-flight for delete; null = idle.
  const [busyRepo, setBusyRepo] = useState<string | null>(null);
  // When a delete returns 409 with the force-required message, we
  // stash the repo + message here to render an inline confirm prompt
  // in place of the row's normal action button. Clearing back to null
  // either via Confirm (re-submit with force) or Cancel restores
  // the row to its normal state.
  const [forcePrompt, setForcePrompt] = useState<{ repo: string; message: string } | null>(null);
  const [errorByRepo, setErrorByRepo] = useState<Record<string, string>>({});

  const sorted = [...repos].sort((a, b) => b.size_bytes - a.size_bytes);

  async function doDelete(repo: string, force: boolean): Promise<void> {
    setBusyRepo(repo);
    setErrorByRepo((p) => {
      const { [repo]: _unused, ...rest } = p;
      return rest;
    });
    try {
      const qs = force ? "?force=true" : "";
      // ``encodeURI`` (not ``encodeURIComponent``) deliberately preserves
      // the ``/`` in ``<org>/<name>`` so the backend's ``{repo:path}``
      // matcher captures the whole id as one segment. HF repo names are
      // alnum + ``_-.`` only (no ``#?+`` or other URL-reserved chars), so
      // ``encodeURI``'s narrower escaping is safe — see CR feedback on
      // !MR for vllm-warden#114.
      const r = await authFetch(`/api/cache/models/${encodeURI(repo)}${qs}`, {
        method: "DELETE",
      });
      if (r.status === 204) {
        setForcePrompt(null);
        onMutate();
        return;
      }
      // 409 with a force-required message → flip into the force-prompt
      // state so the row renders a Confirm-with-force button. ALL other
      // statuses (including 409 for active rows) are terminal errors.
      let message = `HTTP ${r.status}`;
      try {
        const body = await r.json();
        if (typeof body?.detail === "string") message = body.detail;
      } catch {
        /* keep the HTTP fallback */
      }
      if (r.status === 409 && /force=true/i.test(message)) {
        setForcePrompt({ repo, message });
        return;
      }
      setErrorByRepo((p) => ({ ...p, [repo]: message }));
    } catch (err) {
      setErrorByRepo((p) => ({
        ...p,
        [repo]: err instanceof Error ? err.message : "delete failed",
      }));
    } finally {
      setBusyRepo(null);
    }
  }

  if (sorted.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No cached HuggingFace repos found under the configured cache root.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border border-slate-700">
      <table className="w-full text-sm">
        <thead className="bg-slate-800 text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th scope="col" className="px-3 py-2 text-left">Repo</th>
            <th scope="col" className="px-3 py-2 text-right">Size</th>
            <th scope="col" className="px-3 py-2 text-right">Last used</th>
            <th scope="col" className="px-3 py-2 text-left">Owned by</th>
            <th scope="col" className="px-3 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {sorted.map((row) => {
            const { label, tone } = rowStatus(row.matched_models);
            const isActive = row.matched_models.some((m) => ACTIVE_STATUSES.has(m.status));
            const promptOpen = forcePrompt?.repo === row.repo;
            const err = errorByRepo[row.repo];
            return (
              <tr key={row.path} data-testid={`cache-row-${row.repo}`} className="hover:bg-slate-800/50">
                <td className="px-3 py-2 font-mono text-xs text-slate-100">{row.repo}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-200">
                  {formatBytes(row.size_bytes)}
                </td>
                <td className="px-3 py-2 text-right text-xs text-slate-400">
                  {formatRelativeTime(row.last_modified)}
                </td>
                <td className="px-3 py-2 text-xs text-slate-300">
                  <Badge
                    variant={
                      tone === "active"
                        ? "info"
                        : tone === "failed"
                          ? "error"
                          : tone === "idle"
                            ? "success"
                            : "default"
                    }
                  >
                    {label}
                  </Badge>
                  {row.matched_models.length > 0 && (
                    <span className="ml-2 text-slate-400">
                      {row.matched_models.map((m) => m.served_model_name).join(", ")}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right">
                  {promptOpen ? (
                    <div className="flex flex-col items-end gap-1">
                      <span className="text-[11px] text-amber-400" data-testid={`cache-row-${row.repo}-force-msg`}>
                        {forcePrompt!.message}
                      </span>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setForcePrompt(null)}
                          disabled={busyRepo === row.repo}
                        >
                          Cancel
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => doDelete(row.repo, true)}
                          disabled={busyRepo === row.repo}
                          data-testid={`cache-row-${row.repo}-force`}
                        >
                          {busyRepo === row.repo ? "Deleting…" : "Delete anyway"}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-end gap-1">
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => doDelete(row.repo, false)}
                        disabled={isActive || busyRepo === row.repo}
                        title={isActive ? "row is active — unload first" : undefined}
                        data-testid={`cache-row-${row.repo}-delete`}
                      >
                        {busyRepo === row.repo ? "Deleting…" : "Delete"}
                      </Button>
                      {err && (
                        <span
                          className="text-[11px] text-red-400"
                          data-testid={`cache-row-${row.repo}-err`}
                        >
                          {err}
                        </span>
                      )}
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
