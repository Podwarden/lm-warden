// Issue #48 — `/healthz` is the k8s/operator-convention liveness path.
// We alias it to the Next.js-internal `/api/health` route via next.config.ts
// `rewrites()`. This test pins that contract so a future refactor of the
// rewrite block (e.g. adding `beforeFiles` / `fallback` keys) can't silently
// drop the alias.
//
// History note: pre-#155 this file also asserted that `/api/:path*` and
// `/v1/:path*` rewrote through Next.js to the upstream backend (the
// BACKEND_URL env was read at module load). #155 moved to a unified-port
// architecture: Caddy now fronts everything on :8080 and proxies the api/v1
// paths DIRECTLY to FastAPI, so routing them a second time through Next
// would re-engage the gzip path #52 had to disable for SSE. The BACKEND_URL
// branch on the rewrites table is gone (BACKEND_URL is now SSR-only). We
// keep both `BACKEND_URL` setup/teardown blocks here to prove the absence
// of those rewrites is unconditional — see also
// tests/contract/next-config-basepath.test.ts which asserts the same
// negative + the basePath positive in one place.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import nextConfig from '../../next.config';

type Rewrite = { source: string; destination: string };

async function getRewrites(): Promise<Rewrite[]> {
  // next.config.ts exports either a NextConfig or the bundle-analyzer wrapper.
  // Both shapes expose `rewrites` as the same async function we authored.
  const cfg = nextConfig as { rewrites?: () => Promise<Rewrite[] | unknown> };
  expect(typeof cfg.rewrites).toBe('function');
  const result = (await cfg.rewrites!()) as Rewrite[];
  expect(Array.isArray(result)).toBe(true);
  return result;
}

describe('next.config rewrites — /healthz alias', () => {
  const originalBackendUrl = process.env.BACKEND_URL;

  beforeEach(() => {
    delete process.env.BACKEND_URL;
  });

  afterEach(() => {
    if (originalBackendUrl === undefined) {
      delete process.env.BACKEND_URL;
    } else {
      process.env.BACKEND_URL = originalBackendUrl;
    }
  });

  it('aliases /healthz → /api/health when BACKEND_URL is unset', async () => {
    const rewrites = await getRewrites();
    const healthz = rewrites.find((r) => r.source === '/healthz');
    expect(healthz).toBeDefined();
    expect(healthz!.destination).toBe('/api/health');
  });

  it('keeps /healthz alias even when BACKEND_URL is set, and does NOT proxy /api or /v1 through Next (#155)', async () => {
    // BACKEND_URL is no longer read by the rewrites table — Caddy owns the
    // /api and /v1 forward. We still set it here to prove the rewrite list
    // is unaffected by the env var.
    process.env.BACKEND_URL = 'http://api:8080';
    const rewrites = await getRewrites();
    const healthz = rewrites.find((r) => r.source === '/healthz');
    expect(healthz).toBeDefined();
    expect(healthz!.destination).toBe('/api/health');
    // #155 unified-port: Caddy proxies these now; Next.js MUST NOT shadow.
    expect(rewrites.some((r) => r.source === '/api/:path*')).toBe(false);
    expect(rewrites.some((r) => r.source === '/v1/:path*')).toBe(false);
  });
});
