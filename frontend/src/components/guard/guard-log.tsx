"use client";

// The guard's action log (GET /api/guard/events): one row per value per request, with the rule,
// where it was, its length and keyed fingerprint. Never a value.

import { useId, useState } from "react";
import useSWR from "swr";

import { authFetchJSON } from "@/lib/auth-fetch";
import { G } from "@/lib/guard/strings";
import type { GuardEventRow, GuardEvents } from "@/lib/guard/types";
import { fmtDateTime } from "@/lib/token-format";

export function whereText(e: Pick<GuardEventRow, "kind" | "msg_index">): string {
  if (!e.msg_index) return e.kind === "system" ? "system prompt" : e.kind === "tool definition" ? "tool definitions" : (e.kind ?? "—");
  return `${e.kind ?? "message"}, message ${e.msg_index}`;
}

export function GuardLog({ tokenId }: { tokenId?: string }) {
  const filterId = useId();
  const [action, setAction] = useState("");
  const q = new URLSearchParams({ limit: "200" });
  if (tokenId) q.set("token", tokenId);
  if (action) q.set("action", action);
  const { data } = useSWR<GuardEvents>(`/api/guard/events?${q.toString()}`, authFetchJSON, { refreshInterval: 30_000 });

  return (
    <section className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[13px] font-semibold">{G.log.title}</h3>
        <span className="flex items-center gap-1.5 text-[12px]">
          <label htmlFor={filterId} className="text-chat-dim">
            {G.log.filterLabel}
          </label>
          <select id={filterId} className="rounded-md border border-chat-rule bg-chat-page text-chat-fg px-1.5 py-0.5" value={action} onChange={(e) => setAction(e.target.value)}>
            <option value="">{G.log.all}</option>
            {Object.entries(G.log.actions).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </span>
      </div>
      <p className="text-[12px] text-chat-dim">{G.log.intro}</p>
      {data && <p className="text-[12px]">{G.log.totals(data.totals)}</p>}
      {data && data.events.length === 0 ? (
        <p className="text-[12px] text-chat-dim">{G.log.empty}</p>
      ) : (
        <div className="overflow-x-auto">
          <table aria-label={G.log.title} className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-chat-rule text-left text-chat-dim">
                {G.log.cols.map((c) => (
                  <th key={c} className="px-1.5 py-1 font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data?.events ?? []).map((e, i) => (
                <tr key={`${e.ts}-${i}`} className="border-b border-chat-rule/50">
                  <td className="whitespace-nowrap px-1.5 py-1">{fmtDateTime(e.ts)}</td>
                  <td className="px-1.5 py-1">{G.log.actions[e.action] ?? e.action}</td>
                  <td className="px-1.5 py-1">{e.rule_id ?? "—"}</td>
                  <td className="px-1.5 py-1">
                    {whereText(e)}
                    {e.relayed && <span className="text-chat-dim"> · {G.log.relayed}</span>}
                  </td>
                  <td className="whitespace-nowrap px-1.5 py-1">
                    {G.log.value(e.length, e.ph)} {e.ph && <span className="font-mono">{e.ph}</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
