"use client";

import { useState } from "react";

import { Panel } from "@/components/stats/live-panels";
import { formatCompact } from "@/lib/live-stats";
import { S } from "@/lib/sessions/strings";
import type { Tools } from "@/lib/sessions/types";

const PAGE = 50;

export function ToolsPanel({ tools }: { tools: Tools | null }) {
  const [page, setPage] = useState(0);
  const has = tools !== null && tools.offered > 0;
  const rows = has ? tools.rows.slice(page * PAGE, page * PAGE + PAGE) : [];
  const c = S.tools.cols;
  return (
    <Panel title={S.tools.title} testid="tools-panel">
      {!has ? (
        <p className="text-sm text-chat-muted">{S.tools.none}</p>
      ) : (
        <div className="space-y-2">
          <p className="text-sm">
            {S.tools.headline({
              offered: tools.offered,
              called: tools.called,
              unused: tools.unused,
              perTurn: tools.unused_tokens_per_turn === null ? null : formatCompact(tools.unused_tokens_per_turn),
              share: tools.unused_share,
            })}
          </p>
          {tools.truncated && <p className="text-xs text-amber-400">{S.tools.truncated}</p>}
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase text-chat-muted">
                <tr>
                  <th className="py-1 pr-2">{c.tool}</th>
                  <th className="py-1 pr-2">{c.family}</th>
                  <th className="py-1 pr-2 text-right">{c.tokens}</th>
                  <th className="py-1 pr-2 text-right">{c.chars}</th>
                  <th className="py-1 pr-2 text-right">{c.called}</th>
                  <th className="py-1 text-right">{c.offered}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className={`border-t border-chat-rule ${r.called === 0 ? "text-chat-muted" : ""}`}>
                    <td className="py-1 pr-2 font-mono">{r.name ?? `${r.family} #${r.id}`}</td>
                    <td className="py-1 pr-2">{r.family}</td>
                    <td className="py-1 pr-2 text-right">≈ {formatCompact(r.tokens)}</td>
                    <td className="py-1 pr-2 text-right">{r.chars.toLocaleString()}</td>
                    <td className="py-1 pr-2 text-right">{r.called}</td>
                    <td className="py-1 text-right">{r.offered_turns}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {tools.rows.length > PAGE && (
            <div className="flex gap-2 text-xs">
              <button type="button" disabled={page === 0} onClick={() => setPage(page - 1)} className="underline">
                {S.tools.prev}
              </button>
              <button
                type="button"
                disabled={(page + 1) * PAGE >= tools.rows.length}
                onClick={() => setPage(page + 1)}
                className="underline"
              >
                {S.tools.next}
              </button>
            </div>
          )}
          <p className="text-xs text-chat-muted">{S.tools.costNote}</p>
        </div>
      )}
    </Panel>
  );
}
