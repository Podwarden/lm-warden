// Contract test — Next.js config for #155 unified-port deployment, and the
// opt-in static export the macOS app serves (BACKLOG-macos P4).
//
// The unified-port architecture moves the UI under `basePath: '/ui'` so a
// single Caddy listener on :8080 can fan out:
//   `/`            → FastAPI /_landing (public landing)
//   `/ui/*`        → ui container (Next.js standalone, basePath '/ui')
//   `/_next/*`     → ui container (assets — Next still serves at root)
//   `/api/*`       → api container (FastAPI, JWT-gated)
//   `/v1/*`        → api container (OpenAI-compat proxy)
//   `/healthz`     → api container (FastAPI liveness)
//
// Properties of `next.config.ts` that are load-bearing for that topology:
//
//   1. `basePath` MUST equal `/ui` — in BOTH build modes. Next bakes this
//      into every asset URL, <Link> target, and the routes manifest AT
//      BUILD TIME. Dropping it or changing it without rebuilding the image
//      silently breaks every page redirect under Caddy (Linux) and under
//      the FastAPI `/ui` StaticFiles mount (Mac). We assert the literal.
//
//   2. The rewrites table MUST NOT proxy `/api/:path*` or `/v1/:path*` to
//      FastAPI. Caddy owns those routes in the unified-port topology;
//      double-proxying via Next.js would (a) break SSE flush semantics
//      because Next's rewrite uses the same gzip code path that #52 had to
//      disable for SSE, and (b) bake a hardcoded BACKEND_URL into the
//      standalone build that wouldn't survive a Caddy reconfigure.
//
//   3. `compress` MUST be false in both modes (#52: Next's gzip buffers
//      sparse `text/event-stream` forever).
//
// History: until BACKLOG-macos P4 this file also pinned a `/healthz` →
// `/api/health` rewrite to a Next-internal route (#48, added because k8s
// probes couldn't easily traverse basePath rewrites). That alias is GONE:
// an API route is illegal under `output: 'export'`, FastAPI already serves
// `/healthz` (Caddy sends it to the api on Linux; the Mac's single uvicorn
// process serves it directly and aliases `/ui/healthz`), so the Next route
// was dead weight. The assertion now runs the other way: the route file is
// deleted and no rewrite targets it.
//
// 4. `LMW_UI_EXPORT=1` (exactly '1') switches to the static export the Mac
//    app mounts: `output: 'export'`, `trailingSlash: true`,
//    `images.unoptimized: true`, and NO `headers` / `rewrites` keys (both are
//    illegal under export; FastAPI sets the UI's security headers on `/ui`).
//    Any other value — unset, '', '0' — keeps `output: 'standalone'` with the
//    security headers, exactly as the Linux product ships.
//
// We import the config object and exercise rewrites()/headers() directly
// rather than pattern-matching the file text so a future config refactor
// that keeps the same runtime behavior continues to pass. The mode is read
// once at module load, so each leg re-imports the config fresh.

import fs from "node:fs";
import path from "node:path";
import { describe, it, expect, afterEach, vi } from "vitest";
import type { NextConfig } from "next";

// next.config.ts's `export default` is wrapped by withBundleAnalyzer only
// when `process.env.ANALYZE === 'true'`. In the vitest environment ANALYZE
// is unset, so the default export is the bare NextConfig object.

interface RewriteRule {
  source: string;
  destination: string;
}

const FRONTEND_DIR = path.resolve(__dirname, "../..");

async function loadConfig(exportMode: string | undefined): Promise<NextConfig> {
  vi.resetModules();
  // stubEnv records the original value so afterEach's unstubAllEnvs
  // restores it; the unset leg then deletes the stubbed value outright.
  vi.stubEnv("LMW_UI_EXPORT", exportMode ?? "");
  if (exportMode === undefined) delete process.env.LMW_UI_EXPORT;
  return (await import("../../next.config")).default;
}

async function readRewrites(cfg: NextConfig): Promise<RewriteRule[]> {
  // `rewrites` is declared as an async function on NextConfig. Allow both
  // the bare-array shape and the {beforeFiles, afterFiles, fallback} shape
  // Next supports; the assertion is the safe path for either.
  const rewritesFn = (cfg as { rewrites?: () => Promise<unknown> }).rewrites;
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

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("next.config — #155 unified-port topology (standalone, the default)", () => {
  it("bakes basePath '/ui' so Caddy can route /ui/* to the standalone server", async () => {
    const cfg = await loadConfig(undefined);
    expect(cfg.basePath).toBe("/ui");
  });

  it("keeps compress:false so SSE is never gzip-buffered (#52)", async () => {
    const cfg = await loadConfig(undefined);
    expect(cfg.compress).toBe(false);
  });

  it("does NOT proxy /api/* through Next — Caddy owns that route now", async () => {
    const rules = await readRewrites(await loadConfig(undefined));
    for (const r of rules) {
      // Reject any rule whose source is the bare /api/* catch-all the
      // pre-#155 config used.
      expect(r.source).not.toBe("/api/:path*");
    }
  });

  it("does NOT proxy /v1/* through Next — Caddy owns the OpenAI-compat route now", async () => {
    const rules = await readRewrites(await loadConfig(undefined));
    for (const r of rules) {
      expect(r.source).not.toBe("/v1/:path*");
    }
  });

  it("has no /healthz alias and no rewrite into a Next API route (P4: FastAPI serves /healthz)", async () => {
    const rules = await readRewrites(await loadConfig(undefined));
    expect(rules.find((r) => r.source === "/healthz")).toBeUndefined();
    expect(rules.some((r) => r.destination === "/api/health")).toBe(false);
  });

  it("deletes the Next-internal /api/health route (illegal under static export)", () => {
    expect(fs.existsSync(path.join(FRONTEND_DIR, "src/app/api/health/route.ts"))).toBe(false);
  });

  it("outputs 'standalone' and keeps the security headers when LMW_UI_EXPORT is unset", async () => {
    const cfg = await loadConfig(undefined);
    expect(cfg.output).toBe("standalone");
    expect(typeof cfg.headers).toBe("function");
    const { securityHeaders } = await import("../../src/lib/security-headers");
    expect(await cfg.headers!()).toEqual(securityHeaders());
    expect(cfg.trailingSlash).not.toBe(true);
  });

  it.each(["", "0", "true", "1 "])(
    "stays standalone for LMW_UI_EXPORT=%j (only exactly '1' turns export on)",
    async (value) => {
      const cfg = await loadConfig(value);
      expect(cfg.output).toBe("standalone");
      expect(typeof cfg.headers).toBe("function");
    },
  );
});

describe("next.config — LMW_UI_EXPORT=1 static export (macOS app, BACKLOG-macos P4)", () => {
  it("outputs 'export' with trailingSlash and unoptimized images", async () => {
    const cfg = await loadConfig("1");
    expect(cfg.output).toBe("export");
    expect(cfg.trailingSlash).toBe(true);
    expect(cfg.images?.unoptimized).toBe(true);
  });

  it("keeps basePath '/ui' (the FastAPI StaticFiles mount point)", async () => {
    const cfg = await loadConfig("1");
    expect(cfg.basePath).toBe("/ui");
  });

  it("keeps compress:false (#52)", async () => {
    const cfg = await loadConfig("1");
    expect(cfg.compress).toBe(false);
  });

  it("has NO headers or rewrites keys at all (both are illegal under export)", async () => {
    const cfg = await loadConfig("1");
    expect("headers" in cfg).toBe(false);
    expect("rewrites" in cfg).toBe(false);
  });

  it("has NO redirects key (illegal under export; app/webui/redirects.py answers old bookmarks there)", async () => {
    const cfg = await loadConfig("1");
    expect("redirects" in cfg).toBe(false);
  });
});

// Old detail bookmarks (BACKLOG-macos P4): the detail pages moved to fixed
// query-id URLs (src/lib/routes.ts), and the standalone build redirects the
// old /models/<id>, /models/<id>/settings and /tokens/<id> paths to them —
// the same table app/webui/redirects.py serves on the Mac. Matched with
// Next's own route matcher so the lookahead that spares the literal `detail`
// segment is exercised the way the server runs it; the built server is
// curl-checked in the gauntlet too.
describe("next.config — detail bookmark redirects (standalone)", () => {
  interface Redirect {
    source: string;
    destination: string;
    permanent?: boolean;
  }

  async function redirectFor(p: string): Promise<string | null> {
    const cfg = await loadConfig(undefined);
    const rules = (await cfg.redirects!()) as Redirect[];
    const { getPathMatch } = await import("next/dist/shared/lib/router/utils/path-match");
    for (const r of rules) {
      const params = getPathMatch(r.source, { strict: true, removeUnnamedParams: true })(p);
      if (!params) continue;
      expect(r.permanent).toBe(true); // 308, like the FastAPI twin
      return r.destination.replace(":id", String(params.id));
    }
    return null;
  }

  it.each([
    ["/models/abc", "/models/detail?id=abc"],
    ["/models/abc/settings", "/models/detail/settings?id=abc"],
    ["/tokens/t1", "/tokens/detail?id=t1"],
    ["/models/detailx", "/models/detail?id=detailx"],
  ])("redirects %s -> %s", async (from, to) => {
    expect(await redirectFor(from)).toBe(to);
  });

  it.each(["/models/detail", "/models/detail/settings", "/tokens/detail", "/models", "/tokens", "/models/a/b"])(
    "leaves %s alone (the new pages themselves, lists, unknown depth)",
    async (p) => {
      expect(await redirectFor(p)).toBeNull();
    },
  );
});
