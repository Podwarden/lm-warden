import type { NextConfig } from 'next';
import withBundleAnalyzer from '@next/bundle-analyzer';
import { securityHeaders } from './src/lib/security-headers';

const config: NextConfig = {
  output: 'standalone',
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
  // nosniff, Referrer-Policy, no framing, Permissions-Policy, HSTS over
  // HTTPS only -- see src/lib/security-headers.ts. Response headers only:
  // nothing here touches buffering or compression, so SSE is unaffected.
  async headers() {
    return securityHeaders();
  },
  async rewrites() {
    // /healthz is the k8s/operator-convention liveness path; alias it to
    // the Next.js-internal /api/health route so curl, runbooks, and uptime
    // monitors hitting /healthz (via Caddy → /ui/healthz under basePath,
    // or hitting Next.js directly during e2e) get the same { ok: true }
    // response. Issue #48.
    //
    // #155 unified-port: the /api/:path* and /v1/:path* rewrites are
    // DELETED. Caddy owns those routes now and forwards them directly to
    // FastAPI; routing them through Next.js a second time would double
    // the proxy hop, break SSE flush semantics, and bake a server URL into
    // the standalone build that the unified-port deployment doesn't use.
    return [{ source: '/healthz', destination: '/api/health' }];
  },
};

export default process.env.ANALYZE === 'true'
  ? withBundleAnalyzer({ enabled: true })(config)
  : config;
