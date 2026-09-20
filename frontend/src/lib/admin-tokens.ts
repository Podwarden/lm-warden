// Settings → Admin tokens (spec docs/superpowers/specs/2026-09-19-admin-tokens-design.md).
// The types come from the generated OpenAPI types, so a backend field rename
// fails `npm run typecheck` instead of rendering `undefined`.
import type { components } from "@/lib/api-types.generated";
import { detailOf } from "@/lib/token-series";

export type AdminToken = components["schemas"]["AdminToken"];
export type AdminTokenIssued = components["schemas"]["AdminTokenIssued"];
export type AdminTokenList = components["schemas"]["AdminTokenList"];
export type AdminAuditRow = components["schemas"]["AdminAuditRow"];
export type AdminAuditPage = components["schemas"]["AdminAuditPage"];
export type AdminTokenStatus = AdminToken["status"];

export const ADMIN_TOKENS_URL = "/api/admin-tokens";
export const AUDIT_PAGE_SIZE = 50;
// A number, never a function (SWR 2.3 stops polling for good on a 0).
export const LIST_REFRESH_MS = 10_000;

export const EXPIRY_DAYS = [30, 90, 365] as const;
export type ExpiryChoice = (typeof EXPIRY_DAYS)[number] | "never";
export const DEFAULT_EXPIRY: ExpiryChoice = 90;

export const GRACE_OPTIONS: readonly { hours: 0 | 1 | 24; label: string }[] = [
  { hours: 0, label: "None — the old secret stops working now" },
  { hours: 1, label: "1 hour" },
  { hours: 24, label: "24 hours" },
];
export const DEFAULT_GRACE_HOURS = 1;

export const STATUS_LABEL: Record<AdminTokenStatus, string> = {
  active: "Active",
  grace: "Grace",
  revoked: "Revoked",
  expired: "Expired",
};

/** Revoked or expired: dimmed, and no longer refreshable or revocable. */
export function isDead(t: Pick<AdminToken, "status">): boolean {
  return t.status === "revoked" || t.status === "expired";
}

/** The example never embeds a secret: the token lives in $VW_ADMIN_TOKEN. */
export function curlExample(base: string): string {
  return `curl -s -H "Authorization: Bearer $VW_ADMIN_TOKEN" ${base}/api/openapi.json`;
}

export function auditUrl(tokenId: string, before: number | null): string {
  const q = new URLSearchParams({ limit: String(AUDIT_PAGE_SIZE) });
  if (before !== null) q.set("before", String(before));
  return `${ADMIN_TOKENS_URL}/${encodeURIComponent(tokenId)}/audit?${q.toString()}`;
}

/** SQLite's naive UTC "YYYY-MM-DD HH:MM:SS" → local time; "—" for null. */
export function formatSqliteTs(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

export function formatEpoch(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

/** The operator-facing reason from an error response: a session_only
 *  `{detail: {message}}`, a string detail, or a 422's `[{msg}]`. */
export async function responseError(r: Response, fallback: string): Promise<string> {
  const body: unknown = await r.json().catch(() => null);
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail) && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return detailOf(body) ?? fallback;
}
