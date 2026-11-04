// Fetching for the sessions pages. Admin mode uses the admin session (authFetchJSON); key mode sends the key holder's
// read token as Bearer and never touches the admin session. A 401/403 in key mode clears the token and throws
// KeyAuthError (the page routes to /sessions/login); no token means no request at all.

import { usePathname } from "next/navigation";

import { authFetch } from "@/lib/auth-fetch";
import { clearKeyToken, keyMode, readKeyToken } from "@/lib/sessions/key-token";

export type SessionsMode = "admin" | "key";

export class KeyAuthError extends Error {
  constructor() {
    super("sessions key token rejected");
    this.name = "KeyAuthError";
  }
}

/** A non-2xx answer, with the API's `detail` (e.g. "not recorded yet") when it sent one. */
export class SessionsHttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string | null,
  ) {
    super(`HTTP ${status}`);
    this.name = "SessionsHttpError";
  }
}

async function readError(r: Response): Promise<SessionsHttpError> {
  let detail: string | null = null;
  try {
    const body = (await r.json()) as { detail?: unknown };
    detail = typeof body.detail === "string" ? body.detail : null;
  } catch {
    /* not JSON */
  }
  return new SessionsHttpError(r.status, detail);
}

export function sessionsFetcher<T>(mode: SessionsMode): (url: string) => Promise<T> {
  if (mode === "admin") {
    return async (url: string) => {
      const r = await authFetch(url);
      if (!r.ok) throw await readError(r);
      return (await r.json()) as T;
    };
  }
  return async (url: string) => {
    const token = readKeyToken();
    if (!token) throw new KeyAuthError();
    const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` }, credentials: "omit" });
    if (r.status === 401 || r.status === 403) {
      clearKeyToken();
      throw new KeyAuthError();
    }
    if (!r.ok) throw await readError(r);
    return (await r.json()) as T;
  };
}

export interface ListArgs {
  range?: string;
  model?: string | null;
  token?: string | null;
  q?: string | null;
  cursor?: string | null;
  limit?: number;
  group?: string | null;
  groupKey?: string | null;
  view?: string | null;
  where?: string | null;
  sort?: string | null;
  since?: number | null;
  until?: number | null;
  offset?: number | null;
  perGroup?: number | null;
  tzOffsetMin?: number | null;
}

export function sessionsListUrl(args: ListArgs = {}): string {
  const p = new URLSearchParams();
  p.set("range", args.range ?? "24h");
  if (args.model) p.set("model", args.model);
  if (args.token) p.set("token", args.token);
  if (args.q) p.set("q", args.q);
  if (args.cursor) p.set("cursor", args.cursor);
  if (args.limit) p.set("limit", String(args.limit));
  if (args.group) p.set("group", args.group);
  if (args.groupKey) p.set("group_key", args.groupKey);
  if (args.view && args.view !== "all") p.set("view", args.view);
  if (args.where && args.where !== "all") p.set("where", args.where);
  if (args.sort) p.set("sort", args.sort);
  if (args.since != null) p.set("since", String(args.since));
  if (args.until != null) p.set("until", String(args.until));
  if (args.offset != null) p.set("offset", String(args.offset));
  if (args.perGroup != null) p.set("per_group", String(args.perGroup));
  if (args.tzOffsetMin != null) p.set("tz_offset_min", String(args.tzOffsetMin));
  return `/api/sessions?${p.toString()}`;
}

export function sessionUrl(ref: string): string {
  return `/api/sessions/${encodeURIComponent(ref)}`;
}

/** SWR key for a sessions request, tagged by mode so key-mode entries can be dropped together. */
export function sessionsKey(mode: SessionsMode, url: string | null): readonly [string, SessionsMode, string] | null {
  return url === null ? null : (["sessions", mode, url] as const);
}

export function useSessionsMode(): SessionsMode {
  const path = usePathname() ?? "";
  return keyMode(path) ? "key" : "admin";
}
