"use client";

import { useEffect, useMemo } from "react";
import useSWR from "swr";
import { LiveRequestsPanel } from "@/components/stats/live-panels";
import { ForestAuthError, forestFetcher } from "@/lib/forest/client";
import type { LiveRequestRow, LiveRequestsSnapshot } from "@/lib/live-stats";
import { GlassDetails } from "./glass";

export type ForestMode = "session" | "forest-token";

/**
 * Right, below the token bars (spec §6.4): the Stats page's in-flight table, live from `/api/stats/requests` (spec
 * §10.6: live view only), with the forest's auth mode. Row hover marks the session's limb; row click moves the camera.
 */
export function InFlight({
  mode,
  paused = false,
  isPaused,
  onRowHover,
  onRowClick,
  onAuthError,
}: {
  mode: ForestMode;
  /** Stop polling (an auth failure is latched by the view). */
  paused?: boolean;
  /** Checked before every revalidation: true stops the fetch even before `paused` re-renders. */
  isPaused?: () => boolean;
  onRowHover?: (row: LiveRequestRow | null) => void;
  onRowClick?: (row: LiveRequestRow) => void;
  onAuthError?: (err: ForestAuthError) => void;
}) {
  const fetcher = useMemo(() => {
    const get = forestFetcher<LiveRequestsSnapshot>(mode);
    return ([url]: readonly [string, ForestMode]) => get(url);
  }, [mode]);
  // keyed by mode: never shares the Stats page's "/api/stats/requests" entry (a different auth scope)
  const { data, error, isLoading } = useSWR<LiveRequestsSnapshot>(["/api/stats/requests", mode] as const, fetcher, {
    refreshInterval: paused ? 0 : 2000,
    isPaused: () => paused || (isPaused?.() ?? false),
    shouldRetryOnError: (e: unknown) => !(e instanceof ForestAuthError),
    keepPreviousData: true,
  });
  useEffect(() => {
    if (error instanceof ForestAuthError) onAuthError?.(error);
  }, [error, onAuthError]);
  const rows = data?.requests ?? [];
  return (
    <GlassDetails
      title="Requests in flight"
      note={data ? `${rows.length} now` : undefined}
      storageKey="forest.inflight.open"
      defaultOpen={() => true}
      testid="forest-inflight"
      className="flex min-h-0 flex-col"
      bodyClassName="min-h-0 overflow-auto"
    >
      <LiveRequestsPanel
        rows={rows}
        error={error}
        isLoading={isLoading && !data}
        onRowHover={onRowHover}
        onRowClick={onRowClick}
        compact
      />
    </GlassDetails>
  );
}
