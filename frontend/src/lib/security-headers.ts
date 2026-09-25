// Security headers for every UI response, applied by next.config.ts
// `headers()`. The same set the api sends (app/utils/security_headers.py):
// the UI is reached through whatever front door a deployment has (our
// Caddyfile, a Hub stack's own, Traefik, or the published port directly), so
// the application sets them itself rather than trusting the proxy to.
//
// Sources are relative to basePath ('/ui'): '/:path*' covers /ui and
// everything under it, _next assets included.
//
// HSTS is sent only when the request says it arrived over HTTPS
// (X-Forwarded-Proto: https, which Caddy/Traefik set). The UI server cannot
// see VW_TRUST_PROXY_ORIGIN, and it does not need to: a browser ignores HSTS
// received over plain HTTP, so a forged header on an http request is inert,
// while an https request can only have come through a TLS front door.
//
// No Content-Security-Policy restricting scripts yet. Next.js inlines its
// bootstrap and RSC payload scripts, so a useful script-src needs a per-request
// nonce from middleware (which also forces dynamic rendering of every page),
// and nobody has verified that SSE streams, chat2 and god-mode media survive
// it. Until then the UI's CSP carries only the directives that cannot break a
// page: no framing, no <base> hijack, no plugins. The static landing page,
// served by the api, does have a strict hash-based policy.

export interface HeaderRule {
  source: string;
  headers: { key: string; value: string }[];
  has?: { type: 'header'; key: string; value?: string }[];
}

export const PERMISSIONS_POLICY =
  'accelerometer=(), camera=(), geolocation=(), gyroscope=(), ' +
  'magnetometer=(), microphone=(), payment=(), usb=(), browsing-topics=()';

export const UI_CSP = "frame-ancestors 'none'; base-uri 'self'; object-src 'none'";

export const HSTS = 'max-age=31536000';

export function securityHeaders(): HeaderRule[] {
  return [
    {
      source: '/:path*',
      headers: [
        { key: 'X-Content-Type-Options', value: 'nosniff' },
        { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
        { key: 'X-Frame-Options', value: 'DENY' },
        { key: 'Permissions-Policy', value: PERMISSIONS_POLICY },
        { key: 'Content-Security-Policy', value: UI_CSP },
      ],
    },
    {
      source: '/:path*',
      // The client-most entry of a possibly comma-joined chain.
      has: [{ type: 'header', key: 'x-forwarded-proto', value: 'https(?:\\s*,.*)?' }],
      headers: [{ key: 'Strict-Transport-Security', value: HSTS }],
    },
  ];
}
