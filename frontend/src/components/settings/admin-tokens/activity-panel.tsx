"use client";

import useSWRInfinite from "swr/infinite";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { authFetchJSON } from "@/lib/auth-fetch";
import { auditUrl, formatEpoch, type AdminAuditPage, type AdminToken } from "@/lib/admin-tokens";

interface ActivityPanelProps {
  token: AdminToken;
  onClose: () => void;
}

// The spec's "drawer" is the shared Modal at size="lg" (plan Rulings #10):
// it already traps focus, closes on Escape and makes the page inert.
export function AdminTokenActivity({ token, onClose }: ActivityPanelProps) {
  const getKey = (index: number, previous: AdminAuditPage | null): string | null => {
    if (index === 0) return auditUrl(token.id, null);
    if (previous === null || previous.next_before === null) return null;
    return auditUrl(token.id, previous.next_before);
  };
  // No refreshInterval: the trail is read on demand, page by page.
  const { data, error, size, setSize, isValidating } = useSWRInfinite<AdminAuditPage>(
    getKey,
    authFetchJSON,
    { revalidateFirstPage: false },
  );
  const rows = data?.flatMap((page) => page.items) ?? [];
  const last = data?.[data.length - 1];
  const hasMore = last !== undefined && last.next_before !== null;
  const loadingMore = isValidating && data !== undefined && size > data.length;

  return (
    <Modal open onClose={onClose} title={`Activity — ${token.name}`} size="lg">
      {error && !data ? (
        <p
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
        >
          Failed to load the activity.
        </p>
      ) : !data ? (
        <Skeleton className="h-24 w-full" />
      ) : rows.length === 0 ? (
        <p className="text-sm text-slate-400">
          No requests yet. Every request this token makes is listed here for 90 days.
        </p>
      ) : (
        <div className="space-y-3">
          <table className="w-full text-xs">
            <thead className="border-b border-slate-700 bg-slate-800 text-left uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-2 py-1">Time</th>
                <th className="px-2 py-1">Method</th>
                <th className="px-2 py-1">Route</th>
                <th className="px-2 py-1 text-right">Status</th>
                <th className="px-2 py-1 text-right">Duration</th>
                <th className="px-2 py-1">Client IP</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {rows.map((row) => {
                // `client_ip` honours X-Forwarded-For and can be forged;
                // `peer_ip` is the socket peer, which the client cannot
                // forge. Show client_ip as the primary value and surface
                // peer_ip alongside it only when it differs — same
                // request, different vantage point.
                const showPeer = row.peer_ip !== null && row.peer_ip !== row.client_ip;
                return (
                  <tr key={row.id} data-testid="audit-row" className="text-slate-400">
                    <td className="px-2 py-1 tabular-nums">{formatEpoch(row.ts)}</td>
                    <td className="px-2 py-1 font-mono">{row.method}</td>
                    <td className="px-2 py-1 font-mono text-slate-100">{row.path}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{row.status}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{row.duration_ms} ms</td>
                    <td
                      className="px-2 py-1 font-mono"
                      title={showPeer ? `via ${row.peer_ip}` : undefined}
                    >
                      <div>{row.client_ip ?? "—"}</div>
                      {showPeer && (
                        <div className="text-[10px] text-slate-500">via {row.peer_ip}</div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {hasMore && (
            <div className="flex justify-center">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void setSize(size + 1)}
                disabled={loadingMore}
              >
                {loadingMore ? "Loading…" : "Load more"}
              </Button>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
