// The key holder's sign-in token. A key holder signs in on /login (the API key tab) with an inference API key; the
// backend (POST /api/forest/login) answers with a short-lived token that reaches only the key's own routes (sessions,
// forest, /api/me/*), scoped to that key. Only the token and its expiry are kept, in this tab's sessionStorage. The API key itself is never
// stored anywhere.

export const SESSIONS_TOKEN_KEY = "vw-sessions-token";
export const SESSIONS_EXP_KEY = "vw-sessions-exp";
/** Where a key holder signs in: the API key tab of the one login page. */
export const SESSIONS_LOGIN = "/login?key=1";

export type KeyTokenState = "valid" | "expired" | "absent";

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

export function clearKeyToken(): void {
  const s = storage();
  if (!s) return;
  try {
    s.removeItem(SESSIONS_TOKEN_KEY);
    s.removeItem(SESSIONS_EXP_KEY);
  } catch {
    /* storage blocked: nothing held */
  }
}

/**
 * Keep the token and its expiry (`now + expires_in` seconds, as epoch ms). The expiry goes in first, then the token,
 * so a token is never held without one. On any failure (storage blocked, quota) neither is kept: returns false.
 */
export function storeKeyToken(token: string, expiresInS: number): boolean {
  const s = storage();
  if (!s) return false;
  try {
    s.setItem(SESSIONS_EXP_KEY, String(Date.now() + expiresInS * 1000));
    s.setItem(SESSIONS_TOKEN_KEY, token);
    return true;
  } catch {
    clearKeyToken();
    return false;
  }
}

/**
 * The held token's state. A token past its stored expiry, or with an unreadable or missing one, counts as expired and
 * is cleared here, so no request is ever made with it.
 */
export function keyTokenState(): KeyTokenState {
  const s = storage();
  if (!s) return "absent";
  let token: string | null = null;
  let rawExp: string | null = null;
  try {
    token = s.getItem(SESSIONS_TOKEN_KEY);
    rawExp = s.getItem(SESSIONS_EXP_KEY);
  } catch {
    return "absent";
  }
  if (!token) return "absent";
  const exp = rawExp === null ? NaN : Number(rawExp);
  if (!Number.isFinite(exp) || exp <= Date.now()) {
    clearKeyToken();
    return "expired";
  }
  return "valid";
}

/** The token when it is held and unexpired; otherwise null (an expired one is cleared). */
export function readKeyToken(): string | null {
  if (keyTokenState() !== "valid") return null;
  try {
    return storage()?.getItem(SESSIONS_TOKEN_KEY) ?? null;
  } catch {
    return null;
  }
}

/** `?key=1`: a key holder's URL (the key view, never the admin one; on /login, the API key tab). */
export function keyRequested(): boolean {
  try {
    return new URLSearchParams(window.location.search).get("key") === "1";
  } catch {
    return false;
  }
}

/** The sessions pages a key token may open: the list and the detail page, exactly (never an admin page). */
const SESSIONS_EXACT_PATHS: ReadonlySet<string> = new Set([
  "/sessions",
  "/sessions/",
  "/sessions/detail",
  "/sessions/detail/",
]);

export function isSessionsPath(path: string): boolean {
  return SESSIONS_EXACT_PATHS.has(path);
}

/** The key holder's own area: opens only with a key token, never with an admin session. */
const KEY_AREA_PATHS: ReadonlySet<string> = new Set(
  ["/key", "/key/settings", "/key/manage", "/key/guard-log"].flatMap((p) => [p, `${p}/`]),
);

export function isKeyAreaPath(path: string): boolean {
  return KEY_AREA_PATHS.has(path);
}

/** Every page a key token may open: the sessions pages and the key area. */
export function isKeyTokenPath(path: string): boolean {
  return isSessionsPath(path) || isKeyAreaPath(path);
}

/** The page runs on the key token (no admin chrome, no admin requests). */
export function keyMode(path: string): boolean {
  return isKeyTokenPath(path) && keyTokenState() === "valid";
}

/** SWR keys of the key-token view are tagged `["sessions", "key", …]`: dropped on sign-in and sign-out. */
export function isKeyCacheKey(key: unknown): boolean {
  return Array.isArray(key) && key[0] === "sessions" && key[1] === "key";
}
