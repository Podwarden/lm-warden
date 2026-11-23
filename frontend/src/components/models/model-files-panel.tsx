"use client";

/**
 * "Model files" — the downloaded model weights on disk (the HuggingFace hub
 * cache under VW_HF_CACHE_DIR): one row per repo with its size, the models
 * that use it, per-row delete, and a garbage-collect sweep for orphans.
 *
 * The "Model files" tab of /models (/models?tab=files). It was the
 * top-level /cache page (epic/overhaul S6, #105) until 2026-11-23; the menu
 * no longer lists it and /cache redirects here. Named for what it holds, not
 * "Cache", so it is not confused with the KV cache store or the prompt cache.
 *
 * Endpoints (unchanged):
 *   GET /api/cache/models           — list of downloaded repos
 *   DELETE /api/cache/models/{repo} — per-row delete (via CacheTable)
 *   POST /api/cache/models/gc       — dry-run + run GC (via CacheGcButton)
 */

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { CacheTable, type CachedRepoView } from "@/components/stats/cache-table";
import { CacheGcButton } from "@/components/stats/cache-gc-button";
import { MetricSummaryPanel } from "@/components/panels/metric-summary-panel";
import { Skeleton } from "@/components/ui/skeleton";
import { formatBytes } from "@/lib/fit";

// Same set of statuses the backend treats as "active" — repos owned by
// an active model can't be deleted from this page. Mirrors the constant
// in cache-table.tsx; duplicated here so the summary can count orphans
// without crossing the component boundary.
const ACTIVE_STATUSES = new Set(["loaded", "loading", "unloading", "pulling"]);

export function ModelFilesPanel() {
  // 30s poll: filesystem du is comparatively expensive and the files change
  // on human-paced model pulls, not per-request timescale.
  const cache = useSWR<CachedRepoView[]>(
    "/api/cache/models",
    authFetchJSON,
    {
      // A number, not a function returning 0 while hidden: SWR stops polling
      // for good on a 0. It already skips ticks while the tab is hidden.
      refreshInterval: 30_000,
    },
  );

  const repos = cache.data ?? [];
  const totalBytes = repos.reduce((sum, r) => sum + r.size_bytes, 0);
  const orphanCount = repos.filter((r) => r.matched_models.length === 0).length;
  const inUseCount = repos.filter((r) =>
    r.matched_models.some((m) => ACTIVE_STATUSES.has(m.status)),
  ).length;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold">Model files</h2>
          <p className="mt-1 text-sm text-slate-400">
            Downloaded model weights on disk — every pull and load reads them.
            Free space, see which models use each repo, and collect orphans.
          </p>
        </div>
        <CacheGcButton onMutate={() => cache.mutate()} />
      </div>

      <MetricSummaryPanel
        title="Storage"
        metrics={[
          {
            label: "Total on disk",
            value: formatBytes(totalBytes),
            hint: cache.isLoading ? "loading…" : "across all repos",
          },
          {
            label: "Repos",
            value: repos.length.toLocaleString(),
            hint: inUseCount > 0 ? `${inUseCount} in use` : undefined,
          },
          {
            label: "Orphans",
            value: orphanCount.toLocaleString(),
            hint:
              orphanCount > 0
                ? "candidates for GC"
                : "none — clean",
          },
        ]}
      />

      <section className="space-y-3" data-testid="cache-section">
        {cache.isLoading ? (
          <Skeleton className="h-32 w-full" />
        ) : cache.error ? (
          <p className="text-sm text-red-500">
            Failed to load model files
            {cache.error instanceof Error
              ? `: ${cache.error.message}`
              : "."}
          </p>
        ) : repos.length === 0 ? (
          // First-class empty state — matches the /models empty-state hero
          // rhythm so the tab doesn't read as "broken" before the first pull.
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 bg-slate-900/40 p-10 text-center">
            <h3 className="text-base font-semibold text-slate-200">
              No model files yet
            </h3>
            <p className="mt-1 max-w-md text-sm text-slate-400">
              Pulled model weights live here under
              {" "}
              <span className="font-mono text-slate-300">~/.cache/huggingface</span>.
              Register a model and the first pull will populate this list.
            </p>
          </div>
        ) : (
          <CacheTable
            repos={repos}
            onMutate={() => cache.mutate()}
          />
        )}
      </section>
    </div>
  );
}
