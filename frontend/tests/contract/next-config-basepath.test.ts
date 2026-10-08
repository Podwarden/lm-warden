// Contract test — Next.js config for #155 unified-port deployment.
//
// The unified-port architecture moves the UI under `basePath: '/ui'` so a
// single Caddy listener on :8080 can fan out:
//   `/`            → FastAPI /_landing (public landing)
//   `/ui/*`        → ui container (Next.js standalone, basePath '/ui')
//   `/_next/*`     → ui container (assets — Next still serves at root)
//   `/api/*`       → api container (FastAPI, JWT-gated)
//   `/v1/*`        → api container (OpenAI-compat proxy)
//
// Three properties of `next.config.ts` are load-bearing for that topology:
//
//   1. `basePath` MUST equal `/ui`. Next bakes this into every asset URL,
//      <Link> target, and the routes manifest AT BUILD TIME. Dropping it
//      or changing it without rebuilding the image silently breaks every
//      page redirect under Caddy. We assert the literal string.
//
//   2. The rewrites table MUST NOT proxy `/api/:path*` or `/v1/:path*` to
//      FastAPI. Caddy owns those routes in the unified-port topology;
//      double-proxying via Next.js would (a) break SSE flush semantics
//      because Next's rewrite uses the same gzip code path that #52 had to
//      disable for SSE, and (b) bake a hardcoded BACKEND_URL into the
//      standalone build that wouldn't survive a Caddy reconfigure.
//
//   3. The `/healthz` → `/api/health` alias MUST stay. That route lives
//      INSIDE Next.js (not FastAPI) and is the cheap liveness path that
//      uptime monitors hit; #48 added it specifically because k8s probes
//      can't easily traverse basePath rewrites.
//
// We import the config object and exercise rewrites() directly rather than
// pattern-matching the file text so a future config refactor that keeps the
// same runtime behavior continues to pass.

import { describe, it, expect } from "vitest";
import nextConfig from "../../next.config";

// next.config.ts's `export default` is wrapped by withBundleAnalyzer only
// when `process.env.ANALYZE === 'true'`. In the vitest environment ANALYZE
// is unset, so the default export is the bare NextConfig object.

interface RewriteRule {
  source: string;
  destination: string;
}

async function readRewrites(): Promise<RewriteRule[]> {
  // `rewrites` is declared as an async function on NextConfig. Allow both
  // the bare-array shape and the {beforeFiles, afterFiles, fallback} shape
  // Next supports; we only ever return the bare array in this config but
  // the type allows both, so the assertion is the safe path.
  const rewritesFn = (nextConfig as { rewrites?: () => Promise<unknown> })
    .rewrites;
  if (typeof rewritesFn !== "function") return [];
  const result = await rewritesFn();
  if (Array.isArray(result)) return result as RewriteRule[];
  // Object form: flatten beforeFiles + afterFiles + fallback into a single
  // list for the purposes of asserting absence/presence of patterns.
  const obj = result as {
    beforeFiles?: RewriteRule[];
    afterFiles?: RewriteRule[];
    fallback?: RewriteRule[];
  };
  return [
    ...(obj.beforeFiles ?? []),
    ...(obj.afterFiles ?? []),
    ...(obj.fallback ?? []),
  ];
}

describe("next.config — #155 unified-port topology", () => {
  it("bakes basePath '/ui' so Caddy can route /ui/* to the standalone server", () => {
    expect(nextConfig.basePath).toBe("/ui");
  });

  it("keeps /healthz aliased to /api/health for cheap uptime checks (#48)", async () => {
    const rules = await readRewrites();
    const healthz = rules.find((r) => r.source === "/healthz");
    expect(healthz).toBeDefined();
    expect(healthz?.destination).toBe("/api/health");
  });

  it("does NOT proxy /api/* through Next — Caddy owns that route now", async () => {
    const rules = await readRewrites();
    for (const r of rules) {
      // Reject any rule whose source is the bare /api/* catch-all the
      // pre-#155 config used. Other /api/... rules (notably the /healthz
      // → /api/health internal alias) are still allowed because they
      // resolve to Next's own /api/health handler, not to FastAPI.
      expect(r.source).not.toBe("/api/:path*");
    }
  });

  it("does NOT proxy /v1/* through Next — Caddy owns the OpenAI-compat route now", async () => {
    const rules = await readRewrites();
    for (const r of rules) {
      expect(r.source).not.toBe("/v1/:path*");
    }
  });
});
