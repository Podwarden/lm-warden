// Wire types and fetching for the key holder's area (/api/me/*, app/tokens/routes_me.py). Every request carries the
// key holder's sign-in token (sessionsFetcher("key")); a 401/403 clears it and throws KeyAuthError.

import useSWR from "swr";

import type { CacheResponse } from "@/lib/cache-obs";
import { KeyAuthError, sessionsFetcher, sessionsKey } from "@/lib/sessions/client";
import { clearKeyToken, readKeyToken } from "@/lib/sessions/key-token";

export interface MeKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  expires_at: string | null;
  is_revoked: boolean;
  is_expired: boolean;
  is_paused: boolean;
  allowed_models: string[] | null;
  priority: number | null;
  thinking_policy: string | null;
  anthropic_relay: boolean;
  guard_effective?: { mode?: string; packs?: string[]; rules?: number; source?: { mode?: string } } | null;
}

export interface MeUsage {
  buckets: { minute: number; requests: number; prompt_tokens: number; completion_tokens: number }[];
  totals: { requests: number; prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface GuardEventRow {
  ts: number;
  request_id: string | null;
  sess_ref: string | null;
  action: string;
  rule_id: string | null;
  role: string | null;
  msg_index: number | null;
  kind: string | null;
  length: number | null;
  ph: string | null;
  relayed: boolean;
}

export interface MeGuardEvents {
  events: GuardEventRow[];
  totals: Record<string, number>;
}

const fetchKey = sessionsFetcher<unknown>("key");

export function useMe<T>(url: string | null) {
  return useSWR<T>(sessionsKey("key", url), ([, , u]: readonly [string, string, string]) => fetchKey(u) as Promise<T>);
}

/** A write as the key holder. Throws KeyAuthError on 401 (the sign-in is gone); returns the response otherwise. */
export async function meWrite(url: string, method: "PATCH" | "POST", body: unknown): Promise<Response> {
  const token = readKeyToken();
  if (!token) throw new KeyAuthError();
  const r = await fetch(url, {
    method,
    credentials: "omit",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.status === 401) {
    clearKeyToken();
    throw new KeyAuthError();
  }
  return r;
}

/** End the sign-in on the server, then forget it here. A failed call still clears the local token. */
export async function signOutKey(): Promise<void> {
  const token = readKeyToken();
  if (token) {
    try {
      await fetch("/api/forest/logout", {
        method: "POST",
        credentials: "omit",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch {
      /* offline: the token still expires on its own */
    }
  }
  clearKeyToken();
}

export type { CacheResponse };
