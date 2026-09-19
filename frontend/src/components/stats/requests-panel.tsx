"use client";

// "Requests" — the chart that replaced the "Recently finished" table, with
// the table kept underneath as the detail view.
//
// The old table was ten identical-looking rows, kept 15 minutes, nothing
// aggregated: no question it could answer. This panel answers "what did
// this box serve over the window, for whom, how fast, and how did it end"
// in one picture, and honours the page's 1h/6h/24h/7d selector because the
// rows are persisted (app/stats/request_history.py) rather than ringed in
// memory. The table is subordinate: collapsed, newest first, for reading an
// exact value — and its depth is the window's, not a ring's.

import { useMemo, useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { formatCompact, formatInt, formatLatency } from "@/lib/live-stats";
import {
  FINISH_CLASS_LABEL,
  colourGroups,
  coverageNote,
  groupKeyOf,
  pickColourField,
  type RequestHistoryResponse,
  type RequestHistoryRow,
} from "@/lib/request-history";
import type { StatsRange } from "@/lib/stats-v2";
import { ColourLegend, RequestsChart, toPoints } from "./requests-chart";
import { Panel, ScopeNote } from "./live-panels";

/** Rows the detail table shows before "show all". */
export const TABLE_ROWS_COLLAPSED = 25;

function FinishPill({ reason }: { reason: string | null | undefined }) {
  if (!reason) return <span className="text-chat-dim">—</span>;
  const cls =
    reason === "stop"
      ? "rounded-md border border-chat-rule text-chat-muted"
      : reason === "length"
        ? "rounded-full border border-chat-warn/40 text-chat-warn"
        : "rounded-md border border-chat-negative/40 text-chat-negative";
  return (
    <span className={cn("inline-block px-2 py-0.5 font-mono text-[11px]", cls)}>{reason}</span>
  );
}

export function RequestsPanel({
  data,
  range,
  human,
  isLoading,
  error,
}: {
  data: RequestHistoryResponse | undefined;
  range: StatsRange;
  /** "last hour", from WINDOW_META. */
  human: string;
  isLoading: boolean;
  error: unknown;
}) {
  const rows = useMemo(() => data?.requests ?? [], [data]);
  const field = useMemo(() => pickColourField(rows), [rows]);
  const groups = useMemo(() => colourGroups(rows, field), [rows, field]);
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [expanded, setExpanded] = useState(false);

  const shown = useMemo(
    () => (hidden.size === 0 ? rows : rows.filter((r) => !hidden.has(groupKeyOf(r, field)))),
    [rows, hidden, field],
  );
  const points = useMemo(() => toPoints(shown, field, groups), [shown, field, groups]);
  const note = data
    ? coverageNote(data.coverage, data.since_epoch, data.now_epoch, human)
    : null;

  const tableRows = expanded ? shown : shown.slice(0, TABLE_ROWS_COLLAPSED);

  return (
    <Panel
      title="Requests"
      testid="requests-panel"
      right={
        <ScopeNote>
          {data ? (
            <span data-testid="requests-scope">
              {human} · {formatInt(data.total)} requests
              {data.stride > 1 ? ` · drawing every ${data.stride}th` : ""}
            </span>
          ) : (
            human
          )}
        </ScopeNote>
      }
    >
      {isLoading ? (
        <div className="h-72 w-full animate-pulse rounded bg-chat-surface-2/60" />
      ) : error && !data ? (
        <p className="text-sm text-chat-negative">
          Failed to load request history
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      ) : !data || rows.length === 0 ? (
        <p className="py-6 text-center text-sm text-chat-dim" data-testid="requests-empty">
          {note ?? `No requests finished in the ${human} for this selection.`}
        </p>
      ) : (
        <>
          {note && (
            <p className="mb-2 text-[11px] text-chat-warn" data-testid="requests-coverage">
              {note}
            </p>
          )}
          <RequestsChart
            points={points}
            range={range}
            domain={[data.since_epoch * 1000, data.now_epoch * 1000]}
          />
          <div className="mt-2 flex flex-wrap items-center justify-between gap-x-6 gap-y-1">
            <ColourLegend
              field={field}
              groups={groups}
              hidden={hidden}
              onToggle={(key) =>
                setHidden((h) => {
                  const next = new Set(h);
                  if (next.has(key)) next.delete(key);
                  else next.add(key);
                  return next;
                })
              }
            />
            <span className="text-[11px] text-chat-dim">
              y = duration (log) · size = generated tokens · shape = finish:{" "}
              {Object.values(FINISH_CLASS_LABEL).join(", ")}
            </span>
          </div>

          {/* Detail view, subordinate to the chart. `details` keeps the exact
              values one click away without a second panel. */}
          <details className="mt-4 group" data-testid="requests-table">
            <summary className="cursor-pointer select-none text-xs text-chat-muted hover:text-chat-fg">
              Table · newest {formatInt(tableRows.length)} of {formatInt(shown.length)} fetched
              {data.total > rows.length ? ` (${formatInt(data.total)} in window)` : ""}
            </summary>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[64rem] text-sm">
                <thead className="border-b border-chat-rule text-left text-[11px] uppercase tracking-wider text-chat-dim">
                  <tr>
                    <th className="px-3 py-2 font-medium">Finished</th>
                    <th className="px-3 py-2 font-medium">Token</th>
                    <th className="px-3 py-2 font-medium">Client IP</th>
                    <th className="px-3 py-2 font-medium">Model</th>
                    <th className="px-3 py-2 font-medium">Finish</th>
                    <th className="px-3 py-2 text-right font-medium">Prompt</th>
                    <th className="px-3 py-2 text-right font-medium">Generated</th>
                    <th className="px-3 py-2 text-right font-medium">TTFT</th>
                    <th className="px-3 py-2 text-right font-medium">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-chat-rule/70">
                  {tableRows.map((r: RequestHistoryRow) => (
                    <tr key={r.id} data-testid="finished-row" className="text-chat-fg">
                      <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-chat-muted">
                        {new Date(r.finished_at * 1000).toLocaleTimeString()}
                      </td>
                      <td className="px-3 py-2">
                        <span className="font-mono text-xs">
                          {r.token_id ? (
                            <Link
                              href={`/tokens/${encodeURIComponent(r.token_id)}`}
                              className="hover:underline underline-offset-2"
                            >
                              {r.token_name ?? r.token_id}
                            </Link>
                          ) : (
                            r.token_name ?? <span className="text-chat-dim">anonymous</span>
                          )}
                        </span>
                        {r.orphan && (
                          <span
                            title="The client had already disconnected when this request finished."
                            className="ml-2 rounded bg-chat-negative/15 px-1.5 py-0.5 text-[10px] font-medium uppercase text-chat-negative"
                          >
                            orphan
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-chat-muted">
                        {r.client_ip ?? "—"}
                      </td>
                      <td
                        className="max-w-[12rem] truncate px-3 py-2 font-mono text-xs text-chat-muted"
                        title={r.model}
                      >
                        {r.model}
                      </td>
                      <td className="px-3 py-2">
                        <FinishPill reason={r.finish_reason} />
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {formatCompact(r.prompt_tokens)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {formatCompact(r.completion_tokens)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {/* Null TTFT means no token ever arrived — a dash, not 0s. */}
                        {r.ttft_s === null ? "—" : formatLatency(r.ttft_s)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {formatLatency(r.duration_s)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {shown.length > TABLE_ROWS_COLLAPSED && (
              <button
                type="button"
                onClick={() => setExpanded((e) => !e)}
                className="mt-2 text-xs text-chat-accent hover:underline"
                data-testid="requests-table-toggle"
              >
                {expanded ? `Show newest ${TABLE_ROWS_COLLAPSED}` : `Show all ${formatInt(shown.length)} fetched`}
              </button>
            )}
          </details>
        </>
      )}
    </Panel>
  );
}
