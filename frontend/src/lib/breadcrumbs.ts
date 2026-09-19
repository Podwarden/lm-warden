/**
 * Breadcrumb route registry — maps URL patterns to human-readable labels.
 *
 * Ported from PodWarden Core (`frontend/src/lib/breadcrumbs.ts`, #1187 /
 * #1687) and adapted to LLM Warden's routes.
 *
 * Static segments (e.g. `/settings`) get a fixed label here. Dynamic
 * segments (e.g. `/tokens/[id]`) need a resolved name from the page itself;
 * that override comes through `useBreadcrumb({title})` and is wired up via
 * `NavStackProvider.setBreadcrumbTitle`.
 *
 * Paths here are app-router paths WITHOUT the `/ui` basePath — they are what
 * `usePathname()` returns and what `<Link href>` / `router.push` take (Next
 * prefixes the basePath itself).
 *
 * --- Navigability ---
 *
 * A breadcrumb entry is only rendered as a clickable <Link> when the
 * corresponding URL has an actual page.tsx in the app-router tree. An entry
 * that exists only to label an intermediate segment is marked
 * `navigable: false` and renders as a plain <span>. Any intermediate
 * segment with NO matching route at all is also non-navigable (see
 * `buildBreadcrumbs`).
 *
 * Rule: a STATIC_ROUTES entry is navigable unless explicitly `navigable: false`.
 *
 * Verified page.tsx inventory (2026-09-18, frontend/src/app):
 *   /                       — page.tsx (redirects to /models)  navigable
 *   /models                 — page.tsx  navigable
 *   /models/[id]            — page.tsx  navigable
 *   /models/[id]/settings   — page.tsx  navigable
 *   /tokens                 — page.tsx  navigable
 *   /tokens/[id]            — page.tsx  navigable
 *   /stats                  — page.tsx  navigable
 *   /chat                   — page.tsx (redirects to /chat2)  navigable
 *   /chat2                  — page.tsx  navigable
 *   /cache                  — page.tsx  navigable
 *   /settings               — page.tsx  navigable
 *   /templates              — page.tsx  navigable
 *   /login, /setup, /setup/* — no app shell → breadcrumb hidden
 * No intermediate segment without a page.tsx exists today, so no entry
 * carries `navigable: false`; the flag stays for the next one that does.
 */

export interface BreadcrumbEntry {
  /** Resolved label rendered in the breadcrumb. */
  label: string;
  /** href to navigate to when the segment is clicked (no basePath). */
  href: string;
  /** Optional max chars before truncating with ellipsis. */
  maxChars?: number;
  /** True if this is a dynamic segment whose label is provisional / loading. */
  pending?: boolean;
  /**
   * False when the entry's href has no real navigable page.tsx.
   * Non-current entries with navigable === false are rendered as plain <span>
   * elements rather than clickable <Link> elements. Defaults to true.
   */
  navigable?: boolean;
}

/**
 * Where "Home" actually navigates. `/` has a page.tsx, but all it does is
 * redirect to /models, so linking to it would round-trip to the Models page
 * through a redirect. The registry keeps `/` as Home's *pattern* (every
 * chain still starts at `/`); only the navigation target is /models.
 */
export const HOME_HREF = "/models";

/** The URL a crumb or back entry navigates to (Home → HOME_HREF). */
export function navTarget(href: string): string {
  return href === "/" ? HOME_HREF : href;
}

export interface BreadcrumbOverride {
  title: string;
  /** Optional alternative parent path (default: the URL parent). */
  parent?: string;
}

export type BreadcrumbOverrides = Record<string, BreadcrumbOverride>;

interface StaticRoute {
  /** Path pattern; `[name]` marks a dynamic segment. Not a regex. */
  pattern: string;
  /** Static label. */
  label: string;
  /** Optional max chars. */
  maxChars?: number;
  /**
   * False when this pattern has no real page.tsx — it exists only to give a
   * human-readable label to an intermediate segment. Defaults to true.
   */
  navigable?: boolean;
}

/**
 * Static routes. Keep this list in sync with the app-router tree and the
 * menu in `components/nav-bar.tsx` (labels follow the menu, except Tokens,
 * whose pages call themselves "API tokens").
 */
const STATIC_ROUTES: StaticRoute[] = [
  { pattern: "/", label: "Home" },
  { pattern: "/models", label: "Models" },
  { pattern: "/templates", label: "Templates" },
  { pattern: "/chat", label: "Chat" },
  { pattern: "/chat2", label: "Chat" },
  { pattern: "/tokens", label: "API tokens" },
  { pattern: "/stats", label: "Stats" },
  { pattern: "/cache", label: "Cache" },
  { pattern: "/settings", label: "Settings" },

  // Dynamic detail pages — the label is a placeholder; the real name comes
  // from the page via `useBreadcrumb`. Model served names and token names
  // run long, so they get a wider budget than the default.
  { pattern: "/models/[id]", label: "Model", maxChars: 40 },
  { pattern: "/models/[id]/settings", label: "Settings" },
  { pattern: "/tokens/[id]", label: "Token", maxChars: 40 },
];

/**
 * Paths where the breadcrumb header must NOT render: the pages you reach
 * without a session. Mirrors `isUnauthRoute` in components/session-gate.tsx
 * and nav-bar.tsx (exact /login and /setup; the wizard owns /setup/*) — a
 * test pins the three together. Kept as data here so this module stays free
 * of the auth-fetch import chain.
 */
const HIDE_EXACT_PATHS: ReadonlySet<string> = new Set(["/login", "/login/", "/setup", "/setup/"]);
const HIDE_PREFIXES = ["/setup/"];

export function shouldHideBreadcrumb(pathname: string): boolean {
  if (HIDE_EXACT_PATHS.has(pathname)) return true;
  return HIDE_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

/**
 * Match a concrete pathname against the static route table. Returns `null`
 * when no entry matches. Among same-depth matches a static segment beats a
 * dynamic one, so a future `/tokens/new` would win over `/tokens/[id]`.
 */
function matchRoute(pathname: string): StaticRoute | null {
  const segments = pathname.split("/").filter(Boolean);
  let best: { route: StaticRoute; dynamic: number } | null = null;
  for (const route of STATIC_ROUTES) {
    const patSegs = route.pattern.split("/").filter(Boolean);
    if (patSegs.length !== segments.length) continue;
    let ok = true;
    let dynamic = 0;
    for (let i = 0; i < patSegs.length; i++) {
      const p = patSegs[i];
      if (p.startsWith("[") && p.endsWith("]")) {
        dynamic++;
      } else if (p !== segments[i]) {
        ok = false;
        break;
      }
    }
    if (ok && (!best || dynamic < best.dynamic)) best = { route, dynamic };
  }
  return best?.route ?? null;
}

/**
 * True when the pattern's own (last) segment is dynamic, i.e. its registry
 * label is only a placeholder for a name the page supplies. `/models/[id]`
 * is; `/models/[id]/settings` is not — "Settings" is its real label.
 * (PodWarden checks for any dynamic segment, which styled such pages as
 * still loading forever.)
 */
function isDynamicPattern(pattern: string): boolean {
  const last = pattern.split("/").pop() ?? "";
  return last.startsWith("[") && last.endsWith("]");
}

function safeDecode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

/**
 * Build the breadcrumb chain for a given pathname.
 *
 * @param pathname - the current URL path (no query/hash, no basePath).
 * @param overrides - per-path title overrides indexed by *exact pathname*,
 *   populated by pages calling `useBreadcrumb({title})`.
 */
export function buildBreadcrumbs(
  pathname: string,
  overrides: BreadcrumbOverrides = {},
): BreadcrumbEntry[] {
  const chain: BreadcrumbEntry[] = [];
  const visited = new Set<string>();

  // Walk up the URL path one segment at a time, prepending each ancestor.
  function walk(path: string) {
    if (!path || visited.has(path)) return;
    visited.add(path);

    const override = overrides[path];
    if (override?.parent) {
      walk(override.parent);
    } else if (path !== "/") {
      const cut = path.lastIndexOf("/");
      walk(cut <= 0 ? "/" : path.slice(0, cut));
    }

    const route = matchRoute(path);
    const fallbackLabel = route?.label ?? safeDecode(path.split("/").pop() ?? "");
    chain.push({
      label: override?.title ?? fallbackLabel,
      href: path,
      maxChars: route?.maxChars,
      pending: !override && !!route && isDynamicPattern(route.pattern),
      // Navigable only when the path matches a route that has a page.tsx.
      navigable: path === "/" || (!!route && route.navigable !== false),
    });
  }

  walk(pathname);
  if (chain.length === 0 || chain[0].href !== "/") {
    chain.unshift({ label: "Home", href: "/", navigable: true });
  }
  return chain;
}

/**
 * Resolve the title of a page (the last breadcrumb segment): the page's own
 * override if it set one, else the static route label.
 */
export function resolvePageTitle(
  pathname: string,
  overrides: BreadcrumbOverrides = {},
): string | null {
  const override = overrides[pathname];
  if (override?.title) return override.title;
  return matchRoute(pathname)?.label ?? null;
}
