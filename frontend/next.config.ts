import type { NextConfig } from 'next';
import withBundleAnalyzer from '@next/bundle-analyzer';
import { securityHeaders } from './src/lib/security-headers';

// BACKLOG-macos P4: LMW_UI_EXPORT=1 (exactly '1') builds the static export the
// macOS app mounts at /ui inside its single uvicorn process; anything else
// (unset, '', '0') builds the standalone server the Linux product ships.
// headers() and rewrites() are illegal under `output: 'export'`, so the export
// leaves the keys out entirely; FastAPI's /ui mount sets the same security
// headers there.
const isExport = process.env.LMW_UI_EXPORT === '1';

// An id segment that is not the literal `detail` page segment.
const ID = ':id((?!detail/|detail$)[^/]+)';
const DETAIL_BOOKMARK_REDIRECTS = [
  { source: `/models/${ID}/settings`, destination: '/models/detail/settings?id=:id', permanent: true },
  { source: `/models/${ID}`, destination: '/models/detail?id=:id', permanent: true },
  { source: `/tokens/${ID}`, destination: '/tokens/detail?id=:id', permanent: true },
];

const config: NextConfig = {
  // #155 unified-port: serve the entire UI under /ui so a single Caddy
  // listener can fan out / → landing, /ui/* → next, /api+/v1 → fastapi.
  // basePath is BAKED IN at build time (Next inlines it into asset URLs,
  // <Link> targets, and the routes manifest), so this MUST live in
  // next.config.ts rather than an env var; changing it requires a rebuild.
  basePath: '/ui',
  // Disable Next.js's default response compression. The standalone server's
  // built-in gzip middleware (default `compress: true`) silently compresses
  // `text/event-stream` responses; because SSE is sparse, the gzip encoder
  // buffers indefinitely waiting for a flush block, so no bytes ever reach
  // the browser. The Live Logs panel surfaced this as a stream that
  // connected but emitted zero events. Ingress-side compression (Caddy /
  // Traefik) still handles bulk HTML/JSON responses, so disabling the
  // Next.js layer has no perceptible bandwidth cost. Issue #52.
  compress: false,
  // Long-running non-streaming completions (e.g. max_tokens=32768) routinely
  // exceed Next.js's 30s rewrite default, which cuts the upstream socket and
  // surfaces as ECONNRESET. Issue #13.
  experimental: {
    proxyTimeout: 600_000,
  },
  ...(isExport
    ? {
        output: 'export' as const,
        // Every page becomes <route>/index.html, which StaticFiles(html=True)
        // serves for /ui/<route>/.
        trailingSlash: true,
        // No image optimisation server exists in a static export.
        images: { unoptimized: true },
      }
    : {
        output: 'standalone' as const,
        // nosniff, Referrer-Policy, no framing, Permissions-Policy, HSTS over
        // HTTPS only -- see src/lib/security-headers.ts. Response headers only:
        // nothing here touches buffering or compression, so SSE is unaffected.
        async headers() {
          return securityHeaders();
        },
        // Old detail bookmarks (/ui/models/<id>, /ui/models/<id>/settings,
        // /ui/tokens/<id>) -> the query-id pages the static export can build
        // (BACKLOG-macos P4, src/lib/routes.ts). 308 like the Mac's FastAPI
        // twin, app/webui/redirects.py (export mode has no redirects(): that
        // router answers there). The literal `detail` segment is the new page
        // itself and never matches (no loop).
        async redirects() {
          return DETAIL_BOOKMARK_REDIRECTS;
        },
      }),
  // No rewrites in either mode. #155 unified-port DELETED the /api/:path* and
  // /v1/:path* rewrites: Caddy owns those routes and forwards them directly to
  // FastAPI; routing them through Next.js a second time would double the proxy
  // hop, break SSE flush semantics, and bake a server URL into the standalone
  // build. BACKLOG-macos P4 deleted the last one, the #48 /healthz ->
  // /api/health alias, with its Next API route (illegal under export):
  // FastAPI serves /healthz (Caddy routes it to the api on Linux) and aliases
  // /ui/healthz on the Mac.
};

export default process.env.ANALYZE === 'true'
  ? withBundleAnalyzer({ enabled: true })(config)
  : config;
