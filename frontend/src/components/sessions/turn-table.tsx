"use client";

import { Fragment, useState } from "react";

import { Panel } from "@/components/stats/live-panels";
import { formatCompact } from "@/lib/live-stats";
import type { Selection } from "@/lib/sessions/layout";
import { S } from "@/lib/sessions/strings";
import type { Turn } from "@/lib/sessions/types";

import { AnatomyBar } from "./anatomy-bar";
import { fmtSecs, fmtShareCause } from "./format";
import { TurnShape } from "./turn-shape";

function tierText(t: Turn): string {
  const parts = (["gpu", "ram", "nvme", "remote"] as const)
    .filter((k) => t.tiers[k] !== null)
    .map((k) => `${k} ${formatCompact(t.tiers[k])}`);
  return parts.join(" · ") || "—";
}

export function TurnTable({
  turns,
  title,
  selection,
  onSelect,
}: {
  turns: Turn[];
  title: string;
  selection: Selection;
  onSelect: (t: Turn) => void;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const tiers = turns.some((t) => Object.values(t.tiers).some((v) => v !== null));
  const c = S.turn.cols;
  const span = tiers ? 10 : 9;
  return (
    <Panel title={title} testid="turn-table">
      <div className="max-h-[28rem] overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-chat-surface text-left text-xs uppercase text-chat-muted">
            <tr>
              <th className="py-1 pr-2">{c.turn}</th>
              <th className="py-1 pr-2">{c.edge}</th>
              <th className="py-1 pr-2 text-right">{c.prompt}</th>
              <th className="py-1 pr-2">{c.cache}</th>
              <th className="w-16 py-1 pr-2">{c.anatomy}</th>
              <th className="py-1 pr-2 text-right">{c.gen}</th>
              <th className="py-1 pr-2">{c.time}</th>
              <th className="py-1 pr-2">{c.replica}</th>
              {tiers && <th className="py-1 pr-2">{c.tier}</th>}
              <th />
            </tr>
          </thead>
          <tbody>
            {turns.map((t) => {
              const sel = selection.requestId === t.request_id;
              return (
                <Fragment key={t.request_id}>
                  {t.lineage_unknown && (
                    <tr>
                      <td colSpan={span} className="px-1 pt-1 text-xs text-amber-400">
                        {S.turn.lineageUnknown}
                      </td>
                    </tr>
                  )}
                  {t.evicted_on_resume && (
                    <tr>
                      <td colSpan={span} className="px-1 pt-1 text-xs text-amber-400">
                        {S.turn.evicted(t.pause_s ?? null)}
                      </td>
                    </tr>
                  )}
                  <tr
                    data-req={t.request_id}
                    aria-selected={sel}
                    onClick={() => onSelect(t)}
                    className={`cursor-pointer border-t border-chat-rule ${sel ? "bg-slate-700/60" : "hover:bg-slate-800/60"}`}
                  >
                    <td className="py-1 pr-2">{t.turn}</td>
                    <td className="py-1 pr-2">{t.edge ?? "—"}</td>
                    <td className="py-1 pr-2 text-right">{formatCompact(t.prompt_tokens)}</td>
                    <td className="py-1 pr-2">{fmtShareCause(t)}</td>
                    <td className="w-16 py-1 pr-2">{t.anatomy && <AnatomyBar a={t.anatomy} />}</td>
                    <td className="py-1 pr-2 text-right">{formatCompact(t.generated_tokens)}</td>
                    <td className="py-1 pr-2">
                      {fmtSecs(t.prefill_s)} + {fmtSecs(t.gen_s)}
                    </td>
                    <td className="py-1 pr-2">{t.relayed ? t.upstream ?? "—" : t.dp_rank ?? "—"}</td>
                    {tiers && <td className="py-1 pr-2">{tierText(t)}</td>}
                    <td className="py-1 text-right">
                      <button
                        type="button"
                        aria-label={S.turn.expand}
                        aria-expanded={open === t.request_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          setOpen(open === t.request_id ? null : t.request_id);
                        }}
                        className="px-1 text-chat-muted"
                      >
                        {open === t.request_id ? "▾" : "▸"}
                      </button>
                    </td>
                  </tr>
                  {open === t.request_id && (
                    <tr>
                      <td colSpan={span}>
                        <TurnShape t={t} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
