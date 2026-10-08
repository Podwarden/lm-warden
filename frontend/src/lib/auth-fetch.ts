// ---------------------------------------------------------------------------
// Result of an attempted `refresh()`.
// ---------------------------------------------------------------------------
// Pre-v2026.05.20 (#97), `refresh()` returned `string | null` and the
// 401-replay handler treated EVERY null as "session is dead — redirect to
// /login". That conflated three very different failure modes:
//
//   - HTTP 401/403 from /api/auth/refresh: the refresh token cookie is
//     actually rejected by the backend → session IS dead, redirect is correct.
//   - HTTP 5xx from /api/auth/refresh: the backend is restarting / unhealthy
//     → the cookie is still valid, retry on the next user action.
//   - Network error (TCP RST, DNS, abort): same as 5xx — transient.
//   - HTTP 429: rate-limited (unlikely on this endpoint but covers all bases).
//
// On a flaky link or during a backend bounce, every authenticated SWR fetcher
// hitting refresh in parallel would resolve null → the first one would fire
// `window.location.replace('/login')` and the loginRedirectInFlight guard
// would keep the rest silent — but the user got bounced to /login mid-session
// for no reason. Issue #97 manifested as "vllm-warden logs me out every few
// minutes": the proactive refresh added below (Fix #1) collides with backend
// restarts often enough that the transient/terminal distinction is required.
type RefreshResult =
  | { token: string }
  | { error: 'invalid' | 'transient' };

let accessToken: string | null = null;
let refreshing: Promise<RefreshResult> | null = null;

// ---------------------------------------------------------------------------
// Proactive refresh scheduler (#97).
// ---------------------------------------------------------------------------
// The backend mints access tokens with a 15-minute default TTL. Before this
// scheduler, the FE only refreshed when an authed request returned 401 —
// which means every concurrent SWR fetcher that crossed the expiry boundary
// would burn one 401 + one /api/auth/refresh + one replay round-trip,
// flooding the console with 401s and (on slow links) leaving a visible
// "page is broken" window. Worse, if /api/auth/refresh itself was hit by a
// transient 5xx in that exact window, the user got punted to /login.
//
// The scheduler runs a single timer per page lifetime, set on every
// successful `setAccessToken(token, expiresIn)` call — both at login
// (login/page.tsx) and inside `refresh()` itself. Fires at 80% of
// `expires_in` (e.g. 12 min for the default 15 min TTL), well before the
// token expires. A failed proactive refresh does NOT schedule a retry and
// does NOT redirect — it just leaves the timer un-scheduled; the next
// user-driven request hits the 401-replay path normally and either succeeds
// (transient outage cleared) or terminates the session (refresh token
// genuinely rejected).
//
// Why "let refresh() schedule itself recursively": after a successful
// refresh, refresh() calls setAccessToken(token, expiresIn), which in turn
// installs a fresh timer. No external bookkeeping required — the chain
// terminates whenever a refresh fails or the user logs out.

let refreshTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Schedule the next proactive refresh. Cancels any prior pending timer.
 * Pass `expiresIn` in seconds (matches backend response shape).
 *
 * Refresh percentage is 80% — a balance between firing often enough to
 * keep the session alive across normal latency / clock skew, and not
 * firing so often that we double the auth load on the backend.
 */
function scheduleProactiveRefresh(expiresIn: number): void {
  if (refreshTimer !== null) {
    clearTimeout(refreshTimer);
    refreshTimer = null;
  }
  if (!Number.isFinite(expiresIn) || expiresIn <= 0) return;
  const delayMs = Math.floor(expiresIn * 0.8 * 1000);
  refreshTimer = setTimeout(() => {
    refreshTimer = null;
    // Proactive refresh — fire and forget. `refresh()` itself will call
    // setAccessToken on success, which re-schedules the next tick.
    // Transient failure here is silent (caller will retry on next 401);
    // terminal failure here MUST NOT trigger a redirect from the timer
    // callback, because nothing is waiting on it. The next user-driven
    // request will hit 401-replay, refresh again, and that path's
    // terminal-failure branch will own the redirect under the existing
    // loginRedirectInFlight invariant.
    void refresh().catch(() => {
      // refresh() never throws — it converts all failures into
      // RefreshResult — but guard anyway so an unexpected exception
      // can't crash the page lifecycle.
    });
  }, delayMs);
}

/**
 * Set the in-memory access token. Pass `expiresIn` (seconds) to also
 * schedule the next proactive refresh at 80% of that TTL. Passing
 * `null` clears the token and cancels any pending proactive refresh.
 *
 * The two-argument form is consumed at exactly two sites:
 *   - login/page.tsx after a successful POST /api/auth/login
 *   - inside refresh() after a successful POST /api/auth/refresh
 *
 * Existing callers that only set the token (notably tests) keep the
 * single-argument form — they leave the timer untouched. */
export function setAccessToken(t: string | null, expiresIn?: number) {
  accessToken = t;
  if (t === null) {
    if (refreshTimer !== null) {
      clearTimeout(refreshTimer);
      refreshTimer = null;
    }
    return;
  }
  if (typeof expiresIn === 'number') {
    scheduleProactiveRefresh(expiresIn);
  }
}
export function getAccessToken() { return accessToken; }

/** Test-only: returns true iff a proactive-refresh timer is currently
 *  scheduled. Production code MUST NOT consult this — it exists so
 *  vitest can assert that login / refresh / setAccessToken(null) all
 *  manage the timer correctly. */
export function __hasProactiveRefreshScheduledForTests(): boolean {
  return refreshTimer !== null;
}

// ---------------------------------------------------------------------------
// Login-redirect de-duplication (v2026.05.15.3)
// ---------------------------------------------------------------------------
// When the session token expires, every in-flight authenticated request
// returns 401 in parallel. Pre-fix, each one ran the 401 branch below and called
// `window.location.replace('/login')` independently — multiple overlapping
// navigations confused the browser and, critically, stranded any peer code
// path (notably `useEventSource`'s `POST /api/auth/sse-ticket` preflight)
// that was waiting for `authFetch` to resolve. The preflight saw `authFetch`
// resolve to a value it didn't expect (or, in some race orderings, never
// resolve at all before the page navigated) and left LogStream blank.
//
// The fix is a single module-level flag: the first 401 that exhausts refresh
// fires the redirect, all subsequent 401s short-circuit. The flag is also
// readable via `isLoginRedirectInFlight()` so callers like the SSE preflight
// can detect "auth is being torn down, don't bother making the network call"
// and transition to a terminal-error state up front.
//
// The flag is NEVER reset within a page lifetime. The browser navigation
// itself is the reset — once `/login` loads, this module is re-imported
// with a fresh `false`. Resetting it programmatically would re-open the
// double-redirect race the flag is here to close.

let loginRedirectInFlight = false;

/** True iff `authFetch` has already triggered a `/login` redirect during
 *  this page lifetime. Callers can short-circuit network work when the
 *  session is being torn down. */
export function isLoginRedirectInFlight(): boolean {
  return loginRedirectInFlight;
}

/** Test-only: reset the redirect flag between vitest cases. Production
 *  code MUST NOT call this — clearing the flag mid-session re-opens the
 *  parallel-redirect race the flag is here to close. */
export function __resetLoginRedirectInFlightForTests(): void {
  loginRedirectInFlight = false;
}

/**
 * Send the browser to /login, at most once per page lifetime.
 *
 * Extracted from `authFetch`'s 401 branch so `SessionGate` can use exactly
 * the same navigation — same de-dup flag, same same-page guard — rather than
 * open a second, subtly different route to /login.
 *
 * Same-page guard: `replace('/login')` from /login is a full-page reload that
 * re-imports this module with `loginRedirectInFlight = false`, so without it
 * a 401-eliciting fetch on /login loops. Exact-matched (not `startsWith`) so
 * a hypothetical /login-help page keeps normal bounce-to-login behaviour.
 *
 * Note the bare '/login' target: the UI is served under basePath '/ui', and
 * the front door 308s /login → /ui/login (deploy/caddy/Caddyfile). That
 * indirection is deliberate and long-standing; do not "fix" it to /ui/login,
 * which would break the dev server where no such redirect exists.
 */
export function redirectToLogin(): void {
  const pathname =
    typeof window !== 'undefined' ? window.location?.pathname ?? '' : '';
  const onLogin = pathname === '/login' || pathname === '/login/';
  if (loginRedirectInFlight || onLogin) return;
  loginRedirectInFlight = true;
  window.location.replace('/login');
}

/** What a session check concluded. Mirrors `RefreshResult`, flattened. */
export type SessionCheck = 'ok' | 'invalid' | 'transient';

/**
 * Resolve whether this browser currently holds a usable session, without
 * issuing an application request.
 *
 * `SessionGate` calls this before rendering the app shell. Pre-gate, the
 * shell rendered immediately and the page's own SWR fetchers discovered the
 * dead session a round trip later — so a logged-out visitor to /ui/ was shown
 * a Models page with skeleton cards before being bounced to /login, which
 * reads as "the product is broken" rather than "you are signed out".
 *
 * Costs nothing extra: `authFetch`'s eager-refresh guard was already going to
 * make this exact call before its first request, and `refresh()` de-dupes
 * concurrent callers through the shared `refreshing` promise.
 */
export async function ensureSession(): Promise<SessionCheck> {
  if (accessToken !== null) return 'ok';
  const result = await refresh();
  return 'error' in result ? result.error : 'ok';
}

// ---------------------------------------------------------------------------
// CSRF token caching
// ---------------------------------------------------------------------------
// The backend mints a per-session CSRF token (HMAC of session/csrf cookie ID
// with cookie_secret) and exposes it via `GET /api/csrf` → `{"csrf": "..."}`.
// Any mutating request to a non-bypassed path must echo the token back as
// `X-CSRF-Token`. We cache the token in this module so a single fetch covers
// every subsequent unsafe call until the cookie identity rotates (login,
// logout) — at which point the next request will get a 403 and trigger a
// one-shot refetch + retry below.
//
// Paths mirrored from app/auth/csrf.py:_BYPASS_PREFIXES so we don't bother
// the server with a token fetch for routes that wouldn't validate it anyway.

const CSRF_BYPASS_PREFIXES: readonly string[] = [
  "/v1/",
  "/login",
  "/logout",
  "/healthz",
  "/static",
  "/api/auth",
  "/api/setup",
  "/api/csrf",
];

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

let csrfToken: string | null = null;
let csrfFetching: Promise<string | null> | null = null;

export function setCsrfToken(t: string | null) { csrfToken = t; }
export function getCsrfToken() { return csrfToken; }

async function fetchCsrfToken(): Promise<string | null> {
  if (csrfFetching) return csrfFetching;
  csrfFetching = (async () => {
    try {
      const r = await fetch("/api/csrf", { credentials: "include" });
      if (!r.ok) return null;
      let body: unknown;
      try { body = await r.json(); } catch { return null; }
      const tok = (body as { csrf?: unknown } | null)?.csrf;
      if (typeof tok !== "string") return null;
      csrfToken = tok;
      return tok;
    } finally {
      csrfFetching = null;
    }
  })();
  return csrfFetching;
}

function pathFromInput(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  // URL or Request — Request has .url, URL stringifies.
  //
  // NB the parameter is `RequestInfo | URL`, not plain `RequestInfo`: the DOM
  // lib defines `RequestInfo = Request | string`, so `URL` is NOT a member of
  // it, even though this branch has always handled one and the global `fetch`
  // has always accepted one. The declared type was simply narrower than the
  // real behaviour. Widened v2026.08.25 so `authFetch` is assignable to
  // `@podwarden/chat-ui`'s `Adapters['fetch']`, which takes
  // `RequestInfo | URL` — under `strictFunctionTypes` a narrower parameter
  // makes the whole function unassignable. Widening a parameter is safe for
  // every existing caller.
  if (input instanceof URL) return input.toString();
  return (input as Request).url ?? "";
}

function needsCsrfToken(method: string, input: RequestInfo | URL): boolean {
  if (SAFE_METHODS.has(method.toUpperCase())) return false;
  const path = pathFromInput(input);
  // Anchor against the path portion. Absolute URLs (rare for same-origin
  // API calls but valid) still match because the prefixes appear after the
  // origin; startsWith on the raw URL would miss those — strip origin first.
  let pathname = path;
  try {
    if (path.startsWith("http://") || path.startsWith("https://")) {
      pathname = new URL(path).pathname;
    }
  } catch { /* fall back to raw path */ }
  return !CSRF_BYPASS_PREFIXES.some((p) => pathname.startsWith(p));
}

/**
 * Attempt to refresh the access token. De-duped via the module-level
 * `refreshing` promise so concurrent callers share one /api/auth/refresh
 * round-trip.
 *
 * Returns a tagged result so callers can distinguish "session is actually
 * dead (redirect)" from "backend hiccupped (try again later)":
 *   - { token }            — success; accessToken is set; timer rescheduled.
 *   - { error: 'invalid' } — refresh token rejected (HTTP 401/403). The
 *                            session IS dead; caller should redirect once.
 *   - { error: 'transient' } — network error, HTTP 5xx, HTTP 429, or a
 *                              malformed response body. Caller MUST NOT
 *                              redirect — the next user action retries.
 *
 * The pre-#97 single-null return collapsed both error cases, causing
 * backend bounces to evict the user. See the RefreshResult comment above.
 */
async function refresh(): Promise<RefreshResult> {
  if (refreshing) return refreshing;
  refreshing = (async () => {
    try {
      let r: Response;
      try {
        r = await fetch('/api/auth/refresh', {
          method: 'POST', credentials: 'include',
          headers: { 'Origin': window.location.origin },
        });
      } catch (err) {
        // Network-layer failure (TCP RST, DNS, abort, offline). Treat as
        // transient — the cookie hasn't been rejected, we just couldn't
        // ask the backend. Log to console so a sustained outage is at
        // least visible in DevTools without bouncing the user.
        // eslint-disable-next-line no-console
        console.warn('[auth] /api/auth/refresh network error:', err);
        return { error: 'transient' };
      }
      // 401 or 403 = backend explicitly rejected the refresh token.
      // Everything else (5xx, 429, malformed body) is transient.
      if (r.status === 401 || r.status === 403) {
        return { error: 'invalid' };
      }
      if (!r.ok) {
        // eslint-disable-next-line no-console
        console.warn(`[auth] /api/auth/refresh transient status ${r.status}`);
        return { error: 'transient' };
      }
      let body: unknown;
      try {
        body = await r.json();
      } catch {
        // 2xx with a non-JSON body is a backend protocol bug, not a
        // token-validity problem — treat as transient so we don't punt
        // the user to /login over a single corrupted response.
        return { error: 'transient' };
      }
      const { access_token, expires_in } =
        (body ?? {}) as { access_token?: unknown; expires_in?: unknown };
      if (typeof access_token !== 'string') return { error: 'transient' };
      // Single source of truth for "token is now valid + schedule next
      // refresh": setAccessToken installs the timer when expires_in is
      // provided. Pre-#97 we just assigned `accessToken = …` here and
      // never scheduled anything.
      setAccessToken(
        access_token,
        typeof expires_in === 'number' ? expires_in : undefined,
      );
      return { token: access_token };
    } finally { refreshing = null; }
  })();
  return refreshing;
}

// Convert Headers → plain object so tests can match on { Authorization: ... }
// via expect.objectContaining. Headers.forEach yields lowercase keys
// (per WHATWG spec / jsdom), so we drop the lowercase variants for keys
// we want surfaced with canonical capitalization. Leaving both
// "authorization" and "Authorization" in the object made real `fetch`
// concatenate them into a single `Authorization: Bearer x, Bearer x`
// header on the wire — the API then sliced past "Bearer " and saw the
// rest as a malformed token, returning 401 forever.
function headersToObject(h: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  h.forEach((v, k) => {
    if (k === 'authorization' || k === 'x-csrf-token') return;
    out[k] = v;
  });
  if (h.get('Authorization')) out['Authorization'] = h.get('Authorization')!;
  if (h.get('X-CSRF-Token')) out['X-CSRF-Token'] = h.get('X-CSRF-Token')!;
  return out;
}

// Paths that MUST NOT trigger an eager refresh in the guard below.
//   - /api/auth/* is the auth subsystem itself. `refresh()` POSTs
//     /api/auth/refresh; gating that on a refresh would deadlock.
//   - /api/csrf is the CSRF prefetch fired by `fetchCsrfToken`, which
//     deliberately does not call authFetch — but a future caller might
//     authFetch it directly, and the same recursion would arise if a
//     /api/csrf request triggered a refresh that the refresh handler
//     itself depended on.
//
// Anchored as prefixes, mirroring CSRF_BYPASS_PREFIXES so the
// path-matching semantics stay identical. We don't reuse
// CSRF_BYPASS_PREFIXES directly because the two lists have different
// intents — CSRF's bypass also covers `/v1/`, `/login`, `/logout`,
// `/healthz`, `/static`, `/api/setup` which are NOT auth-subsystem
// paths and we DO want eager refresh on them (any of them might be
// reached via authFetch with a 200-on-anonymous response that the
// guard would otherwise short-circuit incorrectly). Keep this list
// minimal to the deadlock cases.
//
// Trailing-slash asymmetry (intentional):
//   - "/api/auth/"  has a trailing slash because /api/auth is a subtree
//     with multiple endpoints (/api/auth/refresh, /api/auth/login,
//     /api/auth/sse-ticket, …). The slash anchors the match so a
//     hypothetical sibling like "/api/authoritative" couldn't bypass.
//   - "/api/csrf"   has NO trailing slash because /api/csrf is the
//     ONLY endpoint at that path (app/main.py:144 registers exactly
//     one route). Every internal caller (`fetchCsrfToken` above) and
//     e2e caller hits the bare "/api/csrf" — adding a trailing slash
//     here would silently drop the match and re-introduce the
//     recursion bug this list exists to prevent. There is no risk of
//     an accidental sibling match because the backend never exposes
//     "/api/csrf-anything".
const UNAUTH_BYPASS_PREFIXES: readonly string[] = [
  "/api/auth/",
  "/api/csrf",
];

// ...except the one endpoint under /api/auth/ that is called without a user
// action on every page load and requires the JWT: the SSE ticket mint
// (`require_jwt` in app/auth/routes.py). No recursion is possible — it is
// not part of `refresh()`, which calls `fetch` directly. Bypassing it made
// the header-metrics stream (opened by NavBar, outside SessionGate, before
// the session is known) POST its first ticket with no bearer on every hard
// load: a 401, a refresh, a replay — two ticket mints per page load (#251).
const EAGER_REFRESH_AUTH_PATHS: ReadonlySet<string> = new Set([
  "/api/auth/sse-ticket",
]);

function shouldEagerRefresh(input: RequestInfo | URL): boolean {
  const path = pathFromInput(input);
  let pathname = path;
  try {
    if (path.startsWith("http://") || path.startsWith("https://")) {
      pathname = new URL(path).pathname;
    }
  } catch { /* fall back to raw path */ }
  if (EAGER_REFRESH_AUTH_PATHS.has(pathname)) return true;
  return !UNAUTH_BYPASS_PREFIXES.some((p) => pathname.startsWith(p));
}

export async function authFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const method = (init.method ?? 'GET').toString();
  const headers = new Headers(init.headers);

  // Eager-refresh guard (Option E1, #50).
  //
  // On hard reload / new tab / deep link, the session cookie is valid
  // but the in-memory `accessToken` (module-level `let` at the top of
  // this file) is null. Pre-fix, several SWR fetchers fired in
  // parallel — `/api/version`, `/api/models`, `/api/stats/*` — all
  // got a 401 (no Authorization header), and authFetch's existing
  // single-shot refresh-and-replay path below recovered each of them.
  // That worked, but logged a flurry of console 401 errors on every
  // first page load AND wasted N-1 RTTs (one wasted 401 per parallel
  // fetcher minus the one that won the `refreshing` promise race).
  //
  // The eager guard promotes the refresh from after-401-recovery to
  // before-first-request when accessToken is null. /api/auth/* and
  // /api/csrf are bypassed to avoid recursion during refresh() itself
  // (it POSTs /api/auth/refresh). The 401-replay path below remains
  // in place for mid-session expiry — refresh() de-dupes via the
  // `refreshing` promise, so a concurrent burst still produces
  // exactly one /api/auth/refresh call.
  if (accessToken === null && shouldEagerRefresh(input)) {
    await refresh();
  }

  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);

  const csrfRequired = needsCsrfToken(method, input);
  if (csrfRequired) {
    const tok = csrfToken ?? await fetchCsrfToken();
    if (tok) headers.set('X-CSRF-Token', tok);
  }

  let r = await fetch(input, { ...init, headers: headersToObject(headers) });

  // 401 → try refreshing the JWT once and replay.
  if (r.status === 401) {
    const result = await refresh();
    if ('error' in result) {
      // Distinguish "session is dead" from "backend hiccupped" (#97).
      //
      // - 'invalid' (HTTP 401/403 from /api/auth/refresh) — the refresh
      //   token cookie was rejected by the backend. The session is
      //   genuinely dead and the user must re-authenticate. Fall through
      //   to the existing loginRedirectInFlight + same-page guard so the
      //   first 401 in this page lifetime fires exactly one redirect.
      //
      // - 'transient' (network error, 5xx, 429, malformed body) — the
      //   refresh COULD NOT BE EVALUATED. The cookie may still be valid;
      //   the user's session is not necessarily dead. Surface the
      //   original 401 to the caller (so its error branch runs) but do
      //   NOT touch loginRedirectInFlight and do NOT redirect. The next
      //   user action (or the proactive scheduler) will retry refresh
      //   when the backend is back. Pre-#97 this branch silently
      //   bounced users to /login during every backend bounce.
      if (result.error === 'transient') {
        return r;
      }

      // Refresh failed — the session is dead. Only the FIRST 401 in this
      // page lifetime gets to fire `replace('/login')`; subsequent 401s
      // (which arrive in parallel when every concurrent authed fetch
      // expires at once) skip the navigation but still hand the 401
      // Response back to their caller so error handling can run. See
      // the loginRedirectInFlight comment at the top of this module.
      //
      // The de-dup flag and the same-page guard both live in
      // `redirectToLogin` above (v2026.05.15.4 + #39), so this path and
      // `SessionGate`'s cannot drift apart. The 401 Response is still
      // returned either way, so the caller's error branch runs normally.
      redirectToLogin();
      return r;
    }
    headers.set('Authorization', `Bearer ${result.token}`);
    r = await fetch(input, { ...init, headers: headersToObject(headers) });
  }

  // 403 on an unsafe path → likely a stale CSRF token (session rotated,
  // server restarted, etc.). Drop the cache, refetch, and replay once.
  //
  // Assumption: the backend uses 401 for authn/authz failures and reserves
  // 403 for CSRF rejection on unsafe methods — see app/auth/csrf.py:csrf_check
  // which is the only middleware that emits 403 for mutating routes. This
  // heuristic would misclassify a genuine permission error as a stale-token
  // signal and trigger a useless replay. Revisit if backend adds role-based
  // 403s (e.g. viewer attempting an admin POST) — at that point we'd need to
  // distinguish CSRF-403 from authz-403, probably via response body shape.
  if (r.status === 403 && csrfRequired) {
    csrfToken = null;
    const tok = await fetchCsrfToken();
    if (tok) {
      headers.set('X-CSRF-Token', tok);
      r = await fetch(input, { ...init, headers: headersToObject(headers) });
    }
  }

  return r;
}

// Convenience JSON fetcher for SWR. Mirrors authFetch's error-on-not-ok
// behaviour so SWR's `error` slot surfaces non-2xx responses instead of
// silently passing through an HTML/error body that would later blow up at
// `.json()` time inside a render.
export async function authFetchJSON<T = unknown>(input: RequestInfo | URL): Promise<T> {
  const r = await authFetch(input);
  if (!r.ok) {
    const err = new Error(`HTTP ${r.status}`) as Error & { status?: number };
    err.status = r.status;
    throw err;
  }
  return (await r.json()) as T;
}
