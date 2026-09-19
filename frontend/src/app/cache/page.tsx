"use client";

/**
 * /cache — HuggingFace cache management surface (epic/overhaul S6, #105).
 *
 * Re-homed from the Storage section on /stats. The cache lives on the
 * model lifecycle pathway (it backs every pull / load / unload), so it
 * deserves a top-level destination rather than being buried below the
 * throughput chart. The page is intentionally single-column and reuses
 * the existing CacheTable + CacheGcButton components verbatim — only the
 * surrounding scaffold (header, summary tiles, empty-state hero) is new.
 *
 * Wires to the same endpoints the Storage section used:
 *   GET /api/cache/models           — list of cached repos
 *   DELETE /api/cache/models/{repo} — per-row delete (via CacheTable)
 *   POST /api/cache/models/gc       — dry-run + run GC (via CacheGcButton)
 *
 * No backend changes in this slice.
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

export default function CachePage() {
  // 30s poll matches the cadence the Storage section used on /stats —
  // filesystem du is comparatively expensive and the cache changes on
  // human-paced model pulls, not per-request timescale. Pause when the
  // tab is hidden so we don't hammer the backend in the background.
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
          <h1 className="text-2xl font-semibold">Cache</h1>
          <p className="mt-1 text-sm text-slate-400">
            HuggingFace weight cache — backs every model pull and load.
            Free space, audit ownership, and collect orphans.
          </p>
        </div>
        <CacheGcButton onMutate={() => cache.mutate()} />
      </div>

      <MetricSummaryPanel
        title="Storage"
        metrics={[
          {
            label: "Total cached",
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
            Failed to load cache
            {cache.error instanceof Error
              ? `: ${cache.error.message}`
              : "."}
          </p>
        ) : repos.length === 0 ? (
          // First-class empty state — matches the S4 /models empty-state
          // hero rhythm so the page doesn't read as "broken" before the
          // first pull lands. The bare CacheTable empty string still
          // renders below; this card is what the operator notices first.
          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 bg-slate-900/40 p-10 text-center">
            <h2 className="text-base font-semibold text-slate-200">
              Cache is empty
            </h2>
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
