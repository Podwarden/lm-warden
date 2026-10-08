"use client";

// /router/stats → "Activity". The decisions filter lives in `?route=` so the
// overview's strip can deep-link "See the fallbacks" (/router/stats?route=fallback).

import { Suspense, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { RouterStats } from "@/components/router/router-stats";
import { parseRouteFilter, type RouteFilter } from "@/components/router/decisions-table";

export default function RouterStatsPage() {
  // useSearchParams opts the tree into client-side rendering; Suspense is
  // required around it (same as tokens/page.tsx).
  return (
    <Suspense fallback={null}>
      <ActivityView />
    </Suspense>
  );
}

function ActivityView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const route = parseRouteFilter(searchParams.get("route"));
  const search = searchParams.toString();

  // replace, not push: a filter click is not a history step.
  const onRouteChange = useCallback(
    (r: Exclude<RouteFilter, "all"> | null) => {
      const q = new URLSearchParams(search);
      if (r) q.set("route", r);
      else q.delete("route");
      const qs = q.toString();
      router.replace(qs ? `/router/stats?${qs}` : "/router/stats", { scroll: false });
    },
    [router, search],
  );

  return <RouterStats route={route} onRouteChange={onRouteChange} />;
}
