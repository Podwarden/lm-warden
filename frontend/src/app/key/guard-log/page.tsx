"use client";

// /key/guard-log — the sensitive data guard's actions on this key's requests. No values: kind, place, length and
// the value's keyed id only (app/guard/events.py).

import { KeyShell } from "@/components/key-area/key-shell";
import { type MeGuardEvents, useMe } from "@/lib/key-area/api";
import { K } from "@/lib/key-area/strings";
import { fmtDateTime } from "@/lib/token-format";

function where(e: MeGuardEvents["events"][number]): string {
  const parts = [e.role, e.msg_index !== null ? `#${e.msg_index}` : null, e.kind].filter(Boolean);
  return parts.length ? parts.join(" ") : "—";
}

function GuardLog() {
  const { data } = useMe<MeGuardEvents>("/api/me/guard/events?limit=500");
  if (!data) return null;
  const t = data.totals;
  return (
    <section>
      <h2 className="font-semibold">{K.guardLog.title}</h2>
      <p className="mb-1 mt-1 text-sm text-slate-400">{K.guardLog.intro}</p>
      <p className="mb-3 text-xs text-slate-500">
        {K.guardLog.totals(t.sanitized ?? 0, t.blocked ?? 0, t.distinct_values ?? 0)}
      </p>
      {data.events.length === 0 ? (
        <p className="text-sm text-slate-400">{K.guardLog.empty}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-xs text-slate-500">
                {K.guardLog.cols.map((c) => (
                  <th key={c} className="px-2 py-1.5 font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.events.map((e, i) => (
                <tr key={`${e.ts}-${i}`} className="border-b border-slate-800/60">
                  <td className="whitespace-nowrap px-2 py-1.5">{fmtDateTime(e.ts)}</td>
                  <td className="px-2 py-1.5">{K.guardLog.actions[e.action] ?? e.action}</td>
                  <td className="px-2 py-1.5">{e.rule_id ?? "—"}</td>
                  <td className="px-2 py-1.5">{where(e)}</td>
                  <td className="px-2 py-1.5">{e.length ?? "—"}</td>
                  <td className="px-2 py-1.5 font-mono text-xs">{e.ph ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function KeyGuardLogPage() {
  return (
    <KeyShell>
      <GuardLog />
    </KeyShell>
  );
}
