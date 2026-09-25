// Breadcrumb route registry (src/lib/breadcrumbs.ts) — ported from PodWarden
// Core's lib/breadcrumbs.test.ts (#1187 / #1687) and adapted to LM Warden's
// routes: resolution, labels, the navigable rule, per-page overrides, and the
// routes where the strip is hidden.
import { describe, expect, it } from 'vitest';
import { readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import {
  HOME_HREF,
  navTarget,
  buildBreadcrumbs,
  resolvePageTitle,
  shouldHideBreadcrumb,
  type BreadcrumbEntry,
} from '@/lib/breadcrumbs';
import { isUnauthRoute } from '@/components/session-gate';

const nav = (e: BreadcrumbEntry) => e.navigable !== false;
const trail = (p: string, o = {}) => buildBreadcrumbs(p, o).map((e) => [e.href, e.label]);

describe('buildBreadcrumbs — the route table', () => {
  it.each([
    ['/', [['/', 'Home']]],
    ['/models', [['/', 'Home'], ['/models', 'Models']]],
    ['/models/abc', [['/', 'Home'], ['/models', 'Models'], ['/models/abc', 'Model']]],
    [
      '/models/abc/settings',
      [['/', 'Home'], ['/models', 'Models'], ['/models/abc', 'Model'], ['/models/abc/settings', 'Settings']],
    ],
    ['/tokens', [['/', 'Home'], ['/tokens', 'API tokens']]],
    ['/tokens/tok-1', [['/', 'Home'], ['/tokens', 'API tokens'], ['/tokens/tok-1', 'Token']]],
    ['/stats', [['/', 'Home'], ['/stats', 'Stats']]],
    ['/chat', [['/', 'Home'], ['/chat', 'Chat']]],
    ['/chat2', [['/', 'Home'], ['/chat2', 'Chat']]],
    ['/cache', [['/', 'Home'], ['/cache', 'Cache']]],
    ['/settings', [['/', 'Home'], ['/settings', 'Settings']]],
    ['/templates', [['/', 'Home'], ['/templates', 'Templates']]],
  ])('%s', (p, expected) => {
    expect(trail(p)).toEqual(expected);
  });

  it('hrefs are app-router paths — never carry the /ui basePath (next/link adds it)', () => {
    for (const e of buildBreadcrumbs('/models/abc/settings')) expect(e.href.startsWith('/ui')).toBe(false);
  });

  it('marks an un-titled dynamic segment pending (placeholder label), static ones not', () => {
    const chain = buildBreadcrumbs('/models/abc/settings');
    expect(chain.map((e) => !!e.pending)).toEqual([false, false, true, false]);
  });

  it('gives the long-name detail pages a wider truncation budget', () => {
    expect(buildBreadcrumbs('/models/abc').at(-1)!.maxChars).toBe(40);
    expect(buildBreadcrumbs('/tokens/abc').at(-1)!.maxChars).toBe(40);
    expect(buildBreadcrumbs('/stats').at(-1)!.maxChars).toBeUndefined();
  });

  it('decodes an unknown segment for its fallback label, and survives a malformed escape', () => {
    expect(buildBreadcrumbs('/stats/live%20view').at(-1)!.label).toBe('live view');
    expect(buildBreadcrumbs('/stats/%E0%A4%A').at(-1)!.label).toBe('%E0%A4%A');
  });
});

describe('buildBreadcrumbs — navigability (PodWarden #1687)', () => {
  it('Home is always first and navigable', () => {
    const [home] = buildBreadcrumbs('/tokens');
    expect(home).toMatchObject({ href: '/', label: 'Home' });
    expect(nav(home)).toBe(true);
  });

  it('every real page in the chain is navigable, including the dynamic ones', () => {
    for (const e of buildBreadcrumbs('/models/abc/settings')) expect(nav(e)).toBe(true);
    for (const e of buildBreadcrumbs('/tokens/tok-1')) expect(nav(e)).toBe(true);
  });

  it('a segment with no route (and so no page.tsx) is NOT navigable', () => {
    // /stats/live was removed (not-found.tsx); the 404 page still gets a
    // trail, but "live" must not be offered as a link.
    const chain = buildBreadcrumbs('/stats/live');
    expect(chain.map((e) => [e.href, nav(e)])).toEqual([
      ['/', true],
      ['/stats', true],
      ['/stats/live', false],
    ]);
    const deep = buildBreadcrumbs('/nope/deeper');
    expect(deep.find((e) => e.href === '/nope')!.navigable).toBe(false);
  });

  // Contract with the filesystem: every directory under src/app that holds a
  // page.tsx must resolve to a navigable, labelled crumb (not the raw
  // segment), or be one of the no-shell routes where the strip is hidden.
  // Adding a page without a registry entry fails here.
  it('covers every page.tsx in src/app', () => {
    const appDir = path.resolve(__dirname, '../../src/app');
    const pages: string[] = [];
    (function walk(dir: string, route: string) {
      for (const name of readdirSync(dir)) {
        const full = path.join(dir, name);
        if (statSync(full).isDirectory()) walk(full, `${route}/${name}`);
        else if (name === 'page.tsx') pages.push(route || '/');
      }
    })(appDir, '');
    expect(pages.length).toBeGreaterThan(10);
    for (const route of pages) {
      const concrete = route.replace(/\[[^\]]+\]/g, 'x');
      if (shouldHideBreadcrumb(concrete)) continue;
      const chain = buildBreadcrumbs(concrete);
      const last = chain.at(-1)!;
      expect(last.href, route).toBe(concrete);
      expect(nav(last), route).toBe(true);
      for (const e of chain) expect(e.label, `${route} → ${e.href}`).not.toBe('x');
    }
  });
});

describe('buildBreadcrumbs — per-page overrides (useBreadcrumb)', () => {
  it('a title override replaces the placeholder and clears pending', () => {
    const chain = buildBreadcrumbs('/tokens/tok-1', { '/tokens/tok-1': { title: 'ci-runner' } });
    expect(chain.at(-1)).toMatchObject({ label: 'ci-runner', pending: false, navigable: true });
  });

  it("an ancestor's override names it inside a child's trail", () => {
    const chain = buildBreadcrumbs('/models/abc/settings', { '/models/abc': { title: 'Qwen3-8B' } });
    expect(chain.map((e) => e.label)).toEqual(['Home', 'Models', 'Qwen3-8B', 'Settings']);
  });

  it('a parent override re-roots the chain', () => {
    const chain = buildBreadcrumbs('/tokens/tok-1', {
      '/tokens/tok-1': { title: 'ci-runner', parent: '/stats' },
    });
    expect(chain.map((e) => e.href)).toEqual(['/', '/stats', '/tokens/tok-1']);
  });

  it('a parent cycle terminates', () => {
    const chain = buildBreadcrumbs('/tokens/a', {
      '/tokens/a': { title: 'a', parent: '/tokens/b' },
      '/tokens/b': { title: 'b', parent: '/tokens/a' },
    });
    expect(chain[0].href).toBe('/');
    expect(chain.map((e) => e.label)).toEqual(['Home', 'b', 'a']);
  });
});

describe('navTarget — where a crumb navigates', () => {
  it('Home (`/`, which only redirects) navigates to /models; everything else to itself', () => {
    expect(HOME_HREF).toBe('/models');
    expect(navTarget('/')).toBe('/models');
    expect(navTarget('/tokens')).toBe('/tokens');
    expect(navTarget('/models/abc')).toBe('/models/abc');
  });

  it('the registry still matches Home on `/`', () => {
    expect(buildBreadcrumbs('/')).toEqual([expect.objectContaining({ href: '/', label: 'Home' })]);
    expect(buildBreadcrumbs('/models')[0].href).toBe('/');
  });
});

describe('resolvePageTitle', () => {
  it('prefers the override, then the registry label, else null', () => {
    expect(resolvePageTitle('/tokens/t', { '/tokens/t': { title: 'ci' } })).toBe('ci');
    expect(resolvePageTitle('/tokens/t')).toBe('Token');
    expect(resolvePageTitle('/tokens')).toBe('API tokens');
    expect(resolvePageTitle('/nope')).toBeNull();
  });
});

describe('shouldHideBreadcrumb', () => {
  it.each(['/login', '/login/', '/setup', '/setup/', '/setup/welcome', '/setup/gpus', '/setup/hf-token', '/setup/admin', '/setup/done'])(
    'hides on %s (no app shell)',
    (p) => expect(shouldHideBreadcrumb(p)).toBe(true),
  );

  it.each(['/', '/models', '/models/abc/settings', '/tokens/x', '/login-help', '/setupx', '/stats'])(
    'shows on %s',
    (p) => expect(shouldHideBreadcrumb(p)).toBe(false),
  );

  // The strip must vanish exactly where the nav bar and SessionGate treat a
  // page as "for someone not signed in" — the three lists are copies.
  it('agrees with isUnauthRoute (session-gate / nav-bar) everywhere', () => {
    for (const p of ['/', '/login', '/login/', '/login-help', '/setup', '/setup/', '/setup/x', '/setupx', '/models', '/tokens/a']) {
      expect(shouldHideBreadcrumb(p), p).toBe(isUnauthRoute(p));
    }
  });
});
