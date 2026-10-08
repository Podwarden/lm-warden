// The key holder's forest token (Task 8). A key holder signs in at /forest/login with an inference API key; the
// backend answers with a forest-only JWT that reaches only the forest routes, scoped to that key. Only the token and
// its expiry are kept, in this tab's sessionStorage. The API key itself is never stored anywhere.

export const FOREST_TOKEN_KEY = "vw-forest-token";
export const FOREST_EXP_KEY = "vw-forest-exp";

export type ForestTokenState = "valid" | "expired" | "absent";

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

export function clearForestToken(): void {
  const s = storage();
  if (!s) return;
  try {
    s.removeItem(FOREST_TOKEN_KEY);
    s.removeItem(FOREST_EXP_KEY);
  } catch {
    /* storage blocked: nothing held */
  }
}

/**
 * Keep the token and its expiry (`now + expires_in` seconds, as epoch ms). The expiry goes in first, then the token,
 * so a token is never held without one. On any failure (storage blocked, quota) neither is kept: returns false.
 */
export function storeForestToken(token: string, expiresInS: number): boolean {
  const s = storage();
  if (!s) return false;
  try {
    s.setItem(FOREST_EXP_KEY, String(Date.now() + expiresInS * 1000));
    s.setItem(FOREST_TOKEN_KEY, token);
    return true;
  } catch {
    clearForestToken();
    return false;
  }
}

/**
 * The held token's state. A token past its stored expiry, or with an unreadable or missing one, counts as expired and
 * is cleared here: it is treated as absent, so no request is ever made with it. (The login always writes both keys,
 * expiry first: a token without one is corrupt.)
 */
export function forestTokenState(): ForestTokenState {
  const s = storage();
  if (!s) return "absent";
  let token: string | null = null;
  let rawExp: string | null = null;
  try {
    token = s.getItem(FOREST_TOKEN_KEY);
    rawExp = s.getItem(FOREST_EXP_KEY);
  } catch {
    return "absent";
  }
  if (!token) return "absent";
  const exp = rawExp === null ? NaN : Number(rawExp);
  if (!Number.isFinite(exp) || exp <= Date.now()) {
    clearForestToken();
    return "expired";
  }
  return "valid";
}

/** The token when it is held and unexpired; otherwise null (an expired one is cleared). */
export function readForestToken(): string | null {
  if (forestTokenState() !== "valid") return null;
  try {
    return storage()?.getItem(FOREST_TOKEN_KEY) ?? null;
  } catch {
    return null;
  }
}

/** `/forest?key=1`: the URL a key holder lands on after signing in (the key-holder view, never the admin one). */
export function forestKeyRequested(): boolean {
  try {
    return new URLSearchParams(window.location.search).get("key") === "1";
  } catch {
    return false;
  }
}

/**
 * The SWR cache entries of the forest-token view, exactly: the forest poll (`forest-view:forest-token`, and
 * `forest-view:forest-token:<range>` for a range other than the default, ForestView) and
 * the in-flight table (`["/api/stats/requests", "forest-token"]`, InFlight). Dropped on sign-in and sign-out, so a
 * remounted view never sees the previous token's cached ForestAuthError (or its data).
 */
export function isForestTokenCacheKey(key: unknown): boolean {
  // every range's entry: the default range's key has no suffix, the others end in ":<range>"
  if (typeof key === "string") return key === "forest-view:forest-token" || key.startsWith("forest-view:forest-token:");
  return Array.isArray(key) && key.length === 2 && key[0] === "/api/stats/requests" && key[1] === "forest-token";
}
