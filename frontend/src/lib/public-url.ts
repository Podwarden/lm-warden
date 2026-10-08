// ---------------------------------------------------------------------------
// getPublicBaseUrl — single source of truth for "what URL do clients
// outside this browser tab see this warden at?".
// ---------------------------------------------------------------------------
//
// Used by user-facing snippets (curl examples, OpenAI client configs, install
// hints) so they render the right hostname even when the warden is behind a
// reverse proxy whose external URL differs from `window.location.origin`.
//
// Resolution rules:
//   1. If `settings.public_url` is set and parses as an http(s) URL, use it.
//   2. Otherwise fall back to `window.location.origin`.
// Both branches strip a trailing slash so the caller can safely do
// `${base}/v1/completions` without doubling the slash.
//
// The `settings.public_url` value is fetched at the call-site via SWR on
// the same key the Networking tab uses (`/api/settings/runtime`), so an
// edit on the Networking tab propagates to every snippet on the next
// SWR mutate. This module is intentionally pure (no SWR, no fetch) — it
// takes the resolved value as an argument so callers control caching.
//
// IMPORTANT: this helper is NOT for CSRF / auth-fetch internals. The
// CSRF middleware compares `Origin` against `window.location.origin`,
// not against the operator-configured `public_url`; mixing those would
// break CSRF when the configured `public_url` happens to be wrong.
// Keep this strictly for cosmetic snippet rendering.
// ---------------------------------------------------------------------------

/**
 * Pure resolver — exported for unit tests. Given the raw `public_url`
 * setting (or null/undefined when unset) and the current origin, return
 * the canonical base URL with no trailing slash.
 */
export function _resolvePublicBaseUrl(
  publicUrl: string | null | undefined,
  origin: string,
): string {
  const candidate = typeof publicUrl === 'string' ? publicUrl.trim() : '';
  if (candidate.length > 0) {
    // Validate scheme to match the backend `_url` coercer's contract.
    // Defensive: a misconfigured row that snuck past validation (e.g.
    // hand-edited SQL) shouldn't render `ftp://x` into a curl example.
    try {
      const parsed = new URL(candidate);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
        return candidate.replace(/\/+$/, '');
      }
    } catch {
      /* fall through to origin */
    }
  }
  return origin.replace(/\/+$/, '');
}

/**
 * Public helper — resolves against `window.location.origin` in the
 * browser, falls back to an empty string for SSR (`typeof window`
 * guard). The latter never renders into the DOM in practice because
 * every consumer of this helper is inside a "use client" component, but
 * the guard prevents a ReferenceError during SSR module evaluation.
 */
export function getPublicBaseUrl(publicUrl: string | null | undefined): string {
  const origin =
    typeof window !== 'undefined' && window.location?.origin
      ? window.location.origin
      : '';
  return _resolvePublicBaseUrl(publicUrl, origin);
}
