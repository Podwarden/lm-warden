"use client";

// /key — what a key holder lands on after signing in with their API key: usage over the last 24 hours, by model, and
// how many values the guard hid.

import { KeyShell } from "@/components/key-area/key-shell";
import { type CacheResponse, type MeGuardEvents, type MeUsage, useMe } from "@/lib/key-area/api";
import { K } from "@/lib/key-area/strings";

const num = new Intl.NumberFormat("en-US");

function compact(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)} k`;
  return num.format(n);
}

function Kpi({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
      <b className="block text-xl">{value}</b>
      <span className="text-xs text-slate-400">{label}</span>
    </div>
  );
}

function Overview() {
  const usage = useMe<MeUsage>("/api/me/usage?range=24h");
  const cache = useMe<CacheResponse>("/api/me/cache?range=24h");
  const guard = useMe<MeGuardEvents>("/api/me/guard/events?limit=1");
  const t = usage.data?.totals;
  const models = cache.data?.models ?? [];
  const known = models.reduce((a, m) => a + m.known_prompt_tokens, 0);
  const cached = models.reduce((a, m) => a + m.cached_tokens, 0);
  const pct = known > 0 ? Math.round((100 * cached) / known) : null;
  const hidden = guard.data?.totals?.sanitized;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi value={t ? num.format(t.requests) : "—"} label={K.overview.requests} />
        <Kpi
          value={t ? compact(t.prompt_tokens) : "—"}
          label={pct !== null ? `${K.overview.prompt} · ${K.overview.fromCache(pct)}` : K.overview.prompt}
        />
        <Kpi value={t ? compact(t.completion_tokens) : "—"} label={K.overview.generated} />
        <Kpi value={hidden !== undefined ? num.format(hidden) : "—"} label={K.overview.hidden} />
      </div>
      <section>
        <h2 className="mb-2 text-sm text-slate-400">{K.overview.perModel}</h2>
        {cache.data && models.length === 0 ? (
          <p className="text-sm text-slate-400">{K.overview.noTraffic}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-left text-xs text-slate-500">
                  {K.overview.cols.map((c) => (
                    <th key={c} className="px-2 py-1.5 font-medium">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.model_id} className="border-b border-slate-800/60">
                    <td className="px-2 py-1.5">{m.model}</td>
                    <td className="px-2 py-1.5">{num.format(m.requests)}</td>
                    <td className="px-2 py-1.5">{compact(m.prompt_tokens)}</td>
                    <td className="px-2 py-1.5">
                      {m.known_prompt_tokens > 0 ? `${Math.round((100 * m.cached_tokens) / m.known_prompt_tokens)}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default function KeyOverviewPage() {
  return (
    <KeyShell>
      <Overview />
    </KeyShell>
  );
}
