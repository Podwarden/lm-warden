// next.config.ts rewrites — Next.js must never shadow the api's routes.
//
// History note: this file was born for issue #48, pinning a `/healthz` →
// `/api/health` rewrite to a Next.js-internal route. BACKLOG-macos P4
// retired that alias in lockstep with deleting `src/app/api/health/route.ts`
// (an API route is illegal under the Mac's `output: 'export'` build, and
// FastAPI already serves `/healthz`: Caddy routes it to the api on Linux,
// and the Mac's uvicorn serves it directly and aliases `/ui/healthz`).
//
// Pre-#155 this file also asserted that `/api/:path*` and `/v1/:path*`
// rewrote through Next.js to the upstream backend (the BACKEND_URL env was
// read at module load). #155 moved to a unified-port architecture: Caddy now
// fronts everything on :8080 and proxies the api/v1 paths DIRECTLY to
// FastAPI, so routing them a second time through Next would re-engage the
// gzip path #52 had to disable for SSE. The BACKEND_URL branch on the
// rewrites table is gone (BACKEND_URL is now SSR-only). We keep both
// `BACKEND_URL` setup/teardown blocks here to prove the absence of those
// rewrites is unconditional — see also
// tests/contract/next-config-basepath.test.ts which asserts the same
// negative + the basePath positive + the export-mode legs in one place.
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import nextConfig from '../../next.config';

type Rewrite = { source: string; destination: string };

async function getRewrites(): Promise<Rewrite[]> {
  // next.config.ts exports either a NextConfig or the bundle-analyzer wrapper.
  // With the /healthz alias retired the config has no rewrites at all; a
  // future rewrites() must still return a plain array we can scan.
  const cfg = nextConfig as { rewrites?: () => Promise<Rewrite[] | unknown> };
  if (cfg.rewrites === undefined) return [];
  expect(typeof cfg.rewrites).toBe('function');
  const result = (await cfg.rewrites()) as Rewrite[];
  expect(Array.isArray(result)).toBe(true);
  return result;
}

describe('next.config rewrites — no shadowing of api routes', () => {
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

  it('does NOT proxy /api or /v1 through Next when BACKEND_URL is unset (#155)', async () => {
    const rewrites = await getRewrites();
    expect(rewrites.some((r) => r.source === '/api/:path*')).toBe(false);
    expect(rewrites.some((r) => r.source === '/v1/:path*')).toBe(false);
  });

  it('does NOT proxy /api or /v1 through Next even when BACKEND_URL is set (#155)', async () => {
    // BACKEND_URL is no longer read by the rewrites table — Caddy owns the
    // /api and /v1 forward. We still set it here to prove the rewrite list
    // is unaffected by the env var.
    process.env.BACKEND_URL = 'http://api:8080';
    const rewrites = await getRewrites();
    // #155 unified-port: Caddy proxies these now; Next.js MUST NOT shadow.
    expect(rewrites.some((r) => r.source === '/api/:path*')).toBe(false);
    expect(rewrites.some((r) => r.source === '/v1/:path*')).toBe(false);
  });
});
