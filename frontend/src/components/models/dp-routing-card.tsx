"use client";

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { stickyShare, type DpRoutingResponse } from "@/lib/dp-routing";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const DASH = "—";

const pct = (v: number | null | undefined): string =>
  v == null ? DASH : `${(v * 100).toFixed(1)} %`;

const num = (v: number | null | undefined): string =>
  v == null ? DASH : String(v);

const fixed1 = (v: number | null | undefined): string =>
  v == null ? DASH : v.toFixed(1);

/**
 * Live per-replica routing view for a data-parallel model (#286). Renders
 * nothing for a single-replica model. Polls only while the model is loaded:
 * the proxy counters do not move otherwise (SWR `refreshInterval` is a stable
 * number in both branches, never a function).
 */
export function DpRoutingCard({
  modelId,
  dataParallelSize,
  status,
}: {
  modelId: string;
  dataParallelSize: number;
  status: string;
}) {
  const enabled = dataParallelSize > 1;
  const { data, error } = useSWR<DpRoutingResponse>(
    enabled ? `/api/models/${modelId}/dp-routing` : null,
    authFetchJSON,
    { refreshInterval: status === "loaded" ? 2000 : 0 },
  );

  if (!enabled) return null;

  // SWR keeps the last good payload next to a later error; prefer the data.
  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Data-parallel routing</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-slate-400">
          {error ? (
            <p data-testid="dp-routing-unavailable">
              Routing statistics are unavailable
              {(error as { status?: number }).status === 404
                ? " (this server does not expose them)."
                : "."}
            </p>
          ) : (
            <p>Loading…</p>
          )}
        </CardContent>
      </Card>
    );
  }

  const share = stickyShare(data.totals);
  const em = data.engine_metrics;

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 space-y-0">
        <CardTitle>Data-parallel routing</CardTitle>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={data.affinity_enabled ? "success" : "default"}>
            {`Affinity ${data.affinity_enabled ? "on" : "off"}`}
          </Badge>
          <Badge variant="info">
            {`Spill threshold ${data.spill_threshold} (${data.spill_threshold_source})`}
          </Badge>
          <Badge>{`Sticky ${pct(share)}`}</Badge>
          <Badge>{`In flight ${data.totals.in_flight}`}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 text-sm text-slate-300">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs" data-testid="dp-rank-table">
            <thead className="text-slate-400">
              <tr>
                <th className="py-1 pr-3 font-medium">Rank</th>
                <th className="py-1 pr-3 font-medium">In flight</th>
                <th className="py-1 pr-3 font-medium">Sticky</th>
                <th className="py-1 pr-3 font-medium">Spilled in</th>
                <th className="py-1 pr-3 font-medium">Client-pinned</th>
                <th className="py-1 pr-3 font-medium">Running / waiting</th>
                <th className="py-1 pr-3 font-medium">KV used</th>
                <th className="py-1 font-medium">Prefix-cache hit</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {data.ranks.map((r) => (
                <tr
                  key={r.rank}
                  data-testid={`dp-rank-row-${r.rank}`}
                  className="border-t border-slate-800"
                >
                  <td className="py-1 pr-3">{r.rank}</td>
                  <td className="py-1 pr-3">{r.in_flight}</td>
                  <td className="py-1 pr-3">{r.sticky}</td>
                  <td className="py-1 pr-3">{r.spilled_in}</td>
                  <td className="py-1 pr-3">{r.client_pinned}</td>
                  <td className="py-1 pr-3">
                    {r.requests_running == null && r.requests_waiting == null
                      ? DASH
                      : `${fixed1(r.requests_running)} / ${fixed1(r.requests_waiting)}`}
                  </td>
                  <td className="py-1 pr-3">{pct(r.kv_cache_usage_perc)}</td>
                  <td className="py-1">{pct(r.prefix_cache_hit_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!em.available && (
          <p className="text-xs text-slate-400" data-testid="dp-engine-metrics-unavailable">
            {em.error ?? "engine does not report per-replica metrics"}
          </p>
        )}
        <p className="text-xs text-slate-500">
          {data.since
            ? `Counters since ${new Date(data.since).toLocaleString()} (warden start)`
            : "Counters: no requests yet"}
        </p>
      </CardContent>
    </Card>
  );
}
