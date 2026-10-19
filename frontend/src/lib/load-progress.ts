"use client";

// SWR hooks for the load-progress API (Task 6 / Task 8).
//
// ``useLoadProgress(id)`` polls ``GET /api/models/{id}/load-progress`` every
// 2 s while the model's status is ``loading``, and stops polling (returns
// ``undefined``) otherwise.  ``useActiveLoadProgress()`` polls the
// all-models endpoint ``GET /api/models/load-progress`` at the same cadence
// and returns the array of active loads (empty when nothing is loading).
//
// Both hooks use the shared ``authFetchJSON`` fetcher so they go through the
// same auth/CSRF/eager-refresh path as every other SWR consumer in the app.
//
// The 2 s cadence matches the model detail page's own status poll.  The API
// returns 404 when no active load exists for the id; SWR surfaces that as
// ``error`` with ``.status === 404`` and the hook treats it as "no progress"
// (``undefined``) rather than a hard failure — the load simply is not active.
//
// Types come from the generated ``api-types.generated.ts`` (regenerated from
// the FastAPI OpenAPI schema); we re-export the relevant ones for callers so
// they do not need to reach into ``components["schemas"]`` themselves.

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { components } from "@/lib/api-types.generated";

export type LoadProgressOut = components["schemas"]["LoadProgressOut"];
export type StepOut = components["schemas"]["StepOut"];
export type VitalsOut = components["schemas"]["VitalsOut"];
export type VitalsPointOut = components["schemas"]["VitalsPointOut"];

/** Refresh cadence while a load is active. Matches the detail page poll. */
const PROGRESS_REFRESH_MS = 2000;

/**
 * Format a duration in seconds for the load-progress UI.
 *
 * One decimal under 10 s (``5.2s``), rounded to an integer under a minute
 * (``14s``, ``30s``), then ``Xm`` / ``XmYs`` for minutes.  Shared by the
 * checklist and the compact card so the two views do not drift apart.
 */
export function formatDuration(s: number): string {
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return r === 0 ? `${m}m` : `${m}m${r}s`;
}

/**
 * Poll one model's load progress.
 *
 * Pass the model's current row ``status`` (``loading`` / ``loaded`` / etc.)
 * so the hook can poll only while a load is in flight — the endpoint 404s
 * for any id that has no active load, so polling it on a ``loaded`` row
 * would be wasted requests + a console full of 404s.
 *
 * Returns ``undefined`` while not loading (or before the first poll lands).
 * Returns the ``LoadProgressOut`` snapshot while loading.  A 404 mid-load
 * (the load finished between polls) is swallowed as "no longer active".
 */
export function useLoadProgress(
  modelId: string | undefined,
  status: string | undefined,
): LoadProgressOut | undefined {
  const loading = status === "loading";
  const key = loading && modelId ? `/api/models/${modelId}/load-progress` : null;
  const { data, error } = useSWR<LoadProgressOut>(key, authFetchJSON, {
    refreshInterval: loading ? PROGRESS_REFRESH_MS : 0,
    shouldRetryOnError: (err: unknown) => {
      // A 404 means "no active load" — not a transient error worth retrying.
      // SWR would otherwise re-fetch on focus/reconnect for a model that is
      // no longer loading, spamming 404s.  Let the hook go dormant.
      const status_ = (err as Error & { status?: number } | undefined)?.status;
      return status_ !== 404;
    },
  });
  if (error && (error as Error & { status?: number }).status === 404) {
    return undefined;
  }
  return data;
}

/**
 * Poll the all-models load-progress endpoint.
 *
 * Returns the array of active loads (empty when nothing is loading).  Used
 * by the models list / dashboard cards for the compact load-progress bar.
 *
 * Pass ``enabled = true`` only while at least one model in the list the
 * caller already has is in status ``loading`` — the endpoint returns
 * ``[]`` when nothing is loading so there is no 404 path to suppress, but
 * there is no point polling it when the fleet is idle.  When ``enabled`` is
 * false the SWR key is ``null`` (no fetch, no interval); when it flips to
 * true the first fetch fires immediately and the 2 s cadence starts.
 */
export function useActiveLoadProgress(
  enabled: boolean,
): LoadProgressOut[] | undefined {
  const key = enabled ? "/api/models/load-progress" : null;
  const { data } = useSWR<LoadProgressOut[]>(key, authFetchJSON, {
    refreshInterval: enabled ? PROGRESS_REFRESH_MS : 0,
  });
  return data;
}
