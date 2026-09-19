"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RotateTokenDialog } from "./rotate-token-dialog";
import { useTokenActions } from "./use-token-actions";
import type { components } from "@/lib/api-types.generated";

// The shape of one `GET /api/tokens` item, generated from the endpoint's
// response model (`TokenListItem` in app/tokens/routes_api.py, which mirrors
// `_enrich` field for field). Notable fields:
//   - is_revoked: server-computed `revoked_at <= now` (#185) -- `revoked_at`
//     alone cannot tell an open grace window from a completed cut, and
//     comparing in the browser would tie the badge to the workstation clock.
//   - paused_at / is_paused: a paused key gets 403 `token paused`.
//   - priority: 0..9 STRICT; 9 always served first, 0 may starve.
//   - usage_24h: rollup totals from token_usage_minute over the trailing 24h.
export type TokenUsage24h = components["schemas"]["TokenUsage24h"];
export type TokenItem = components["schemas"]["TokenListItem"];

interface StatusInfo {
  label: string;
  variant: "default" | "success" | "warning" | "error" | "info";
  // Optional native tooltip — used by the rotated branches to answer
  // "grace until when?" / "cut off when?" without opening the API.
  title?: string;
}

// Status precedence is defined in the §11.6 spec — encoded here as a
// short-circuit ladder so callers can't accidentally surface two states at
// once. The "Revoked" branch is a safety net: the backend filter strips
// revoked-and-not-rotated rows from the list response, but if the contract
// ever changes the row component still renders something sane.
function deriveStatus(item: TokenItem): StatusInfo {
  // Paused outranks every other state (spec §4.2 badge order: Paused →
  // Revoked → Expired → Grace → Active).
  if (item.is_paused) {
    return { label: "Paused", variant: "warning", title: `Paused since ${formatTs(item.paused_at)}. New requests get 403.` };
  }
  if (item.revoked_at != null && item.rotated_at == null) {
    return { label: "Revoked", variant: "error" };
  }
  if (item.is_expired) {
    return { label: "Expired", variant: "error" };
  }
  if (item.rotated_at != null && item.successor_deleted) {
    return { label: "Rotated (orphan)", variant: "error" };
  }
  // #185 — a rotated row whose grace window has closed (including the
  // grace_hours=0 hard cut) is REJECTED by auth. Without this branch it kept
  // reading amber "Rotated (grace)" forever, telling an operator who had just
  // hard-revoked a leaked token that a grace window was still open.
  if (item.rotated_at != null && item.is_revoked) {
    return {
      label: "Rotated (revoked)",
      variant: "error",
      title: `Rejected since ${formatTs(item.revoked_at)}. Requests already running on the engine were not cancelled.`,
    };
  }
  if (item.rotated_at != null) {
    return {
      label: "Rotated (grace)",
      variant: "warning",
      title: `Old token keeps working until ${formatTs(item.revoked_at)}.`,
    };
  }
  if (item.is_near_expiry) {
    return { label: "Expiring soon", variant: "warning" };
  }
  return { label: "Active", variant: "success" };
}

// SQLite emits naive UTC strings like "2026-01-01 00:00:00" (see
// `sqlite_utc_now` in app/db/repos/tokens.py). `new Date(s)` parses these
// as local time on some browsers — append "Z" so it's unambiguously UTC.
function formatTs(value: string | null): string {
  if (!value) return "—";
  const isoish = value.includes("T") ? value : value.replace(" ", "T") + "Z";
  const d = new Date(isoish);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}

// Compact human number — operators scan the 24h column for "is this token
// alive?" not for accountancy precision. 12345 → "12.3k", 1_234_567 → "1.2M".
function formatCompact(n: number): string {
  if (n < 1_000) return n.toString();
  if (n < 1_000_000) return `${(n / 1_000).toFixed(n < 10_000 ? 1 : 0)}k`;
  if (n < 1_000_000_000) return `${(n / 1_000_000).toFixed(n < 10_000_000 ? 1 : 0)}M`;
  return `${(n / 1_000_000_000).toFixed(1)}B`;
}

// Priority tier → Badge variant. 0..3 muted (default slate), 4..6 normal
// (info), 7..8 elevated (warning amber), 9 critical (error red). Visual
// loudness scales with starvation risk to the priorities BELOW this row,
// not the urgency of the row itself — a deliberate "this token outranks
// others" cue rather than a status indicator.
function priorityVariant(p: number): "default" | "info" | "warning" | "error" {
  if (p >= 9) return "error";
  if (p >= 7) return "warning";
  if (p >= 4) return "info";
  return "default";
}

// Tooltip text MUST warn about starvation under STRICT scheduling — this is
// a contract surface for the operator UI (see dispatch + docs/operating.md).
// Kept short enough to fit in a native title attribute; long-form lives in
// the docs page linked from the table header.
const PRIORITY_TOOLTIP =
  "STRICT scheduler: higher priority is ALWAYS served first. " +
  "A priority-0 token can wait indefinitely behind a steady stream of " +
  "priority-9 traffic (starvation by design).";

interface TokenRowProps {
  item: TokenItem;
  onChange: () => void;
}

export function TokenRow({ item, onChange }: TokenRowProps) {
  const actions = useTokenActions(item, onChange);
  const { busy, deleteError, testResult, testing } = actions;
  const status = deriveStatus(item);
  const totalTokens24h =
    item.usage_24h.prompt_tokens + item.usage_24h.completion_tokens;

  async function onDelete() {
    // Revalidate on success AND failure — on failure the row reappears from
    // the server's perspective, which is the correct state. Not when nothing
    // was sent (the confirm was cancelled, or a delete is already running).
    if ((await actions.remove()) !== null) onChange();
  }

  return (
    <>
      <tr className={deleteError || testResult ? "" : "border-b border-slate-800"}>
        <td className="px-2 py-2 font-medium">
          <Link href={`/tokens/${encodeURIComponent(item.id)}`} className="hover:underline">
            {item.name}
          </Link>
        </td>
        <td className="px-2 py-2 font-mono text-xs">{item.prefix}</td>
        <td className="px-2 py-2 text-xs text-slate-400" data-testid="token-created">
          {formatTs(item.created_at)}
        </td>
        <td className="px-2 py-2 text-xs text-slate-400">{formatTs(item.expires_at)}</td>
        <td className="px-2 py-2 text-xs text-slate-400">{formatTs(item.last_used_at)}</td>
        <td className="px-2 py-2" data-testid="token-priority">
          <Badge
            variant={priorityVariant(item.priority)}
            title={PRIORITY_TOOLTIP}
            // tabIndex so keyboard users can also surface the native tooltip
            // via focus on browsers that honour `title` on focus.
            tabIndex={0}
            aria-label={`Priority ${item.priority}. ${PRIORITY_TOOLTIP}`}
          >
            P{item.priority}
          </Badge>
        </td>
        <td
          className="px-2 py-2 text-right font-mono text-xs tabular-nums text-slate-300"
          // Tooltip exposes the exact request count + prompt/completion split
          // for operators who need numbers (e.g. cost reconciliation). The
          // visible cell stays compact so the row doesn't blow up.
          title={
            `${item.usage_24h.requests.toLocaleString()} requests · ` +
            `${item.usage_24h.prompt_tokens.toLocaleString()} prompt + ` +
            `${item.usage_24h.completion_tokens.toLocaleString()} completion tokens`
          }
          data-testid="token-usage-24h"
        >
          {item.usage_24h.requests === 0 ? (
            <span className="text-slate-500">—</span>
          ) : (
            <>
              {formatCompact(item.usage_24h.requests)}
              <span className="text-slate-500"> req · </span>
              {formatCompact(totalTokens24h)}
              <span className="text-slate-500"> tok</span>
            </>
          )}
        </td>
        <td className="px-2 py-2">
          <Badge variant={status.variant} title={status.title}>
            {status.label}
          </Badge>
        </td>
        <td className="px-2 py-2 text-right">
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              aria-label="Test token authentication"
              onClick={() => { void actions.runTest(); }}
              disabled={testing}
              data-testid="token-test"
            >
              {testing ? "Testing…" : "Test"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              aria-label="Rotate"
              onClick={actions.openRotate}
              disabled={busy || item.rotated_at != null}
              // Disable rotate on already-rotated rows — rotating the
              // predecessor would chain a second grace period. If we
              // want that, we'd add it as an explicit "extend grace"
              // action rather than overloading rotate.
            >
              Rotate
            </Button>
            <Button
              type="button"
              variant="destructive"
              size="sm"
              onClick={onDelete}
              disabled={busy}
            >
              Delete
            </Button>
          </div>
        </td>
      </tr>
      {testResult && (
        // Sub-row hosting the test result. Same pattern as deleteError —
        // separate <tr> so a long error message doesn't reflow the actions.
        // colSpan covers all 9 columns of the table from page.tsx.
        <tr className={deleteError ? "" : "border-b border-slate-800"}>
          <td colSpan={9} className="px-2 pb-2 text-right">
            <p
              role="status"
              className={`text-sm ${testResult.ok ? "text-emerald-400" : "text-red-500"}`}
            >
              {testResult.ok
                ? `OK — ${testResult.detail ?? "verified"} (${testResult.ms} ms)`
                : `FAIL — HTTP ${testResult.status || "n/a"} (${testResult.ms} ms)` +
                  (testResult.detail ? `: ${testResult.detail}` : "")}
            </p>
          </td>
        </tr>
      )}
      {deleteError && (
        // Sub-row hosting the delete error. colSpan covers the 9-column
        // table from page.tsx.
        // We render it as a separate <tr> rather than overflowing the
        // Actions cell because the error message can be long enough to push
        // the action buttons around.
        <tr className="border-b border-slate-800">
          <td colSpan={9} className="px-2 pb-2 text-right">
            <p role="alert" className="text-sm text-red-500">
              {deleteError}
            </p>
          </td>
        </tr>
      )}
      <RotateTokenDialog
        open={actions.rotateOpen}
        tokenId={item.id}
        onClose={actions.closeRotate}
      />
    </>
  );
}
