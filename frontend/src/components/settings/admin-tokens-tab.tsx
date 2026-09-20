"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { authFetchJSON } from "@/lib/auth-fetch";
import { getPublicBaseUrl } from "@/lib/public-url";
import { cn, copyToClipboard } from "@/lib/utils";
import {
  ADMIN_TOKENS_URL,
  LIST_REFRESH_MS,
  STATUS_LABEL,
  curlExample,
  formatSqliteTs,
  isDead,
  type AdminToken,
  type AdminTokenList,
  type AdminTokenStatus,
} from "@/lib/admin-tokens";
import { IssueAdminTokenDialog } from "@/components/settings/admin-tokens/issue-dialog";
import { RefreshAdminTokenDialog } from "@/components/settings/admin-tokens/refresh-dialog";
import { RevokeAdminTokenDialog } from "@/components/settings/admin-tokens/revoke-dialog";
import { AdminTokenActivity } from "@/components/settings/admin-tokens/activity-panel";

// ---------------------------------------------------------------------------
// Settings → Admin tokens (spec docs/superpowers/specs/2026-09-19-admin-tokens-design.md).
//
// Not a runtime-settings form, so it does not wear SettingsTabShell. Every
// call here is session-only on the server: an admin token cannot manage
// admin tokens. Dialogs are mounted only while open, so a shown secret is
// discarded with its dialog.
// ---------------------------------------------------------------------------

// Same Badge component/palette the other Settings tabs use for status chips
// (e.g. SettingSection's restart badge, cache-table's row-status badge).
const STATUS_BADGE_VARIANT: Record<AdminTokenStatus, "success" | "warning" | "error" | "default"> = {
  active: "success",
  grace: "warning",
  revoked: "error",
  expired: "default",
};

type Open =
  | { kind: "issue" }
  | { kind: "refresh" | "revoke" | "activity"; token: AdminToken }
  | null;

export function AdminTokensTab() {
  const { data, error, isLoading, mutate } = useSWR<AdminTokenList>(
    ADMIN_TOKENS_URL,
    authFetchJSON,
    { refreshInterval: LIST_REFRESH_MS },
  );
  // The admin-token curl examples deliberately do NOT use the operator's
  // configured `public_url` (final-review I1, 2026-09-19): the operator is
  // looking at this warden right now, so its own origin is always the right
  // host, and no stored setting can steer a freshly revealed secret elsewhere.
  const base = getPublicBaseUrl(null);
  const [open, setOpen] = useState<Open>(null);
  const [copied, setCopied] = useState(false);

  function close() {
    setOpen(null);
    void mutate();
  }

  async function copyCurl() {
    try {
      await copyToClipboard(curlExample(base));
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  const items = data?.items ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="text-xl font-semibold">Admin tokens</h2>
        {items.length > 0 && (
          <Button size="sm" onClick={() => setOpen({ kind: "issue" })}>
            Issue token
          </Button>
        )}
      </div>

      <Card>
        <CardContent className="space-y-3 text-sm text-slate-300">
          <p>
            Admin tokens call the control API (<code>/api/*</code>) with the same power as signing
            in. The API is described at <code>/api/openapi.json</code>.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <code
              data-testid="admin-intro-curl"
              className="break-all rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-200"
            >
              {curlExample(base)}
            </code>
            <Button size="sm" variant="outline" onClick={() => void copyCurl()}>
              {copied ? "Copied" : "Copy curl example"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {isLoading && !data ? (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : error && !data ? (
        <div
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200"
        >
          Failed to load admin tokens{error instanceof Error ? `: ${error.message}` : "."}
        </div>
      ) : items.length === 0 ? (
        <Card>
          <CardContent className="space-y-3 p-8 text-center">
            <p className="text-sm text-slate-300">
              No admin tokens yet. Issue one to let a script, a CI job or an agent drive this
              warden — change settings, register, pull and load models, manage inference tokens,
              read statistics — without the admin password. Only a signed-in session can issue,
              refresh or revoke admin tokens.
            </p>
            <Button size="sm" onClick={() => setOpen({ kind: "issue" })}>
              Issue token
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="overflow-x-auto rounded-md border border-slate-700">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-700 bg-slate-800 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-2 py-2">Name</th>
                <th className="px-2 py-2">Prefix</th>
                <th className="px-2 py-2">Created by</th>
                <th className="px-2 py-2">Created</th>
                <th className="px-2 py-2">Last used</th>
                <th className="px-2 py-2">Expires</th>
                <th className="px-2 py-2">Status</th>
                <th className="px-2 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {items.map((t) => {
                const dead = isDead(t);
                return (
                  <tr
                    key={t.id}
                    data-testid="admin-token-row"
                    data-dead={dead ? "true" : "false"}
                    className={cn("text-slate-100", dead && "opacity-60")}
                  >
                    <td className="px-2 py-2 font-medium">{t.name}</td>
                    <td className="px-2 py-2 font-mono text-xs text-slate-200">{t.prefix}</td>
                    <td className="px-2 py-2 text-xs text-slate-400">{t.created_by ?? "—"}</td>
                    <td className="px-2 py-2 text-xs text-slate-400">{formatSqliteTs(t.created_at)}</td>
                    <td className="px-2 py-2 text-xs text-slate-400">{formatSqliteTs(t.last_used_at)}</td>
                    <td className="px-2 py-2 text-xs text-slate-400">
                      {t.expires_at ? formatSqliteTs(t.expires_at) : "Never"}
                    </td>
                    <td className="px-2 py-2">
                      <Badge variant={STATUS_BADGE_VARIANT[t.status]}>{STATUS_LABEL[t.status]}</Badge>
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          aria-label={`Activity ${t.name}`}
                          onClick={() => setOpen({ kind: "activity", token: t })}
                        >
                          Activity
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          aria-label={`Refresh ${t.name}`}
                          disabled={t.status !== "active"}
                          onClick={() => setOpen({ kind: "refresh", token: t })}
                        >
                          Refresh
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          aria-label={`Revoke ${t.name}`}
                          disabled={dead}
                          onClick={() => setOpen({ kind: "revoke", token: t })}
                        >
                          Revoke
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {open?.kind === "issue" && <IssueAdminTokenDialog base={base} onClose={close} />}
      {open?.kind === "refresh" && (
        <RefreshAdminTokenDialog token={open.token} base={base} onClose={close} />
      )}
      {open?.kind === "revoke" && (
        <RevokeAdminTokenDialog
          token={open.token}
          predecessor={items.find((t) => t.id === open.token.rotated_from)}
          onClose={close}
        />
      )}
      {open?.kind === "activity" && (
        <AdminTokenActivity token={open.token} onClose={() => setOpen(null)} />
      )}
    </div>
  );
}
