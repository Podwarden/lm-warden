import { describe, it, expect } from 'vitest';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import {
  CONSOLE_CSP,
  defaultTargets,
  pagesWithoutCsp,
  run,
  inlineScriptHashes,
  patchDirectory,
  patchDocument,
  unprerenderedRoutes,
} from '../../scripts/patch-csp-meta.mjs';

// conventions: security.headers@2 -- the console's script Content-Security-
// Policy. Next.js inlines its bootstrap/RSC scripts, so the policy is a <meta>
// tag patched into each prerendered page after `next build`, carrying the
// sha256 of that page's own inline scripts (Crawl Warden's pattern). An
// injected <script> matches none of the hashes and does not run.

const FRONTEND = path.resolve(__dirname, '../..');
const SCRIPT = path.join(FRONTEND, 'scripts', 'patch-csp-meta.mjs');

function directives(policy: string): Map<string, string[]> {
  return new Map(
    policy
      .split(';')
      .map((d) => d.trim().split(/\s+/))
      .filter((p) => p[0])
      .map(([name, ...values]) => [name, values]),
  );
}

const sha = (body: string) => `'sha256-${createHash('sha256').update(body, 'utf8').digest('base64')}'`;

const PAGE =
  '<!DOCTYPE html><html><head><meta charset="utf-8"/>' +
  '<script src="/ui/_next/static/chunks/a.js" async=""></script>' +
  '<script src="/ui/_next/static/chunks/polyfills.js" noModule=""></script>' +
  '</head><body><div>hi</div>' +
  '<script>(self.__next_f=self.__next_f||[]).push([0])</script>' +
  '<script>self.__next_f.push([1,"x"])</script>' +
  '<script>self.__next_f.push([1,"x"])</script>' +
  '<script id="__x">window.y=1</script>' +
  '<script type="application/json">{"a":1}</script>' +
  '</body></html>';

describe('console CSP policy', () => {
  const policy = CONSOLE_CSP([sha('a')]);
  const d = directives(policy);

  it('restricts scripts: script-src is present, self plus hashes only', () => {
    const script = d.get('script-src');
    expect(script, 'script-src must stay in the policy').toBeDefined();
    expect(script).toContain("'self'");
    expect(script).toContain(sha('a'));
    for (const v of script!) expect(v).toMatch(/^('self'|'sha256-[A-Za-z0-9+/]+=*')$/);
  });

  it("never allows 'unsafe-inline' or 'unsafe-eval' (or a wildcard) for scripts", () => {
    for (const name of ['default-src', 'script-src', 'script-src-elem', 'script-src-attr']) {
      const v = d.get(name) ?? [];
      expect(v, name).not.toContain("'unsafe-inline'");
      expect(v, name).not.toContain("'unsafe-eval'");
      expect(v, name).not.toContain("'unsafe-hashes'");
      expect(v, name).not.toContain('*');
      expect(v, name).not.toContain('data:');
      expect(v, name).not.toContain('blob:');
      expect(v, name).not.toContain('http:');
      expect(v, name).not.toContain('https:');
    }
  });

  it('carries the rest of the conventions policy', () => {
    expect(d.get('default-src')).toEqual(["'self'"]);
    expect(d.get('style-src')).toEqual(["'self'", "'unsafe-inline'"]);
    expect(d.get('font-src')).toEqual(["'self'"]);
    expect(d.get('img-src')).toEqual(["'self'", 'data:', 'blob:']);
    expect(d.get('connect-src')).toEqual(["'self'"]);
    expect(d.get('object-src')).toEqual(["'none'"]);
    expect(d.get('base-uri')).toEqual(["'none'"]);
    expect(d.get('form-action')).toEqual(["'self'"]);
  });

  it('leaves out the directives a <meta> CSP may not carry (the header sends frame-ancestors)', () => {
    for (const name of ['frame-ancestors', 'report-uri', 'sandbox']) expect(d.has(name)).toBe(false);
  });
});

describe('patchDocument', () => {
  it('hashes every executable inline script, once each, and no external or data script', () => {
    expect(inlineScriptHashes(PAGE)).toEqual([
      sha('(self.__next_f=self.__next_f||[]).push([0])'),
      sha('self.__next_f.push([1,"x"])'),
      sha('window.y=1'),
    ]);
  });

  it('puts the meta tag after the charset meta, ahead of every script, and is idempotent', () => {
    const { html, patched } = patchDocument(PAGE);
    expect(patched).toBe(true);
    expect(
      html.startsWith(
        '<!DOCTYPE html><html><head><meta charset="utf-8"/><meta http-equiv="Content-Security-Policy" content="',
      ),
    ).toBe(true);
    expect(html.indexOf('Content-Security-Policy')).toBeLessThan(html.indexOf('<script'));
    expect(patchDocument('<html><head><title>x</title></head></html>').html).toMatch(
      /^<html><head><meta http-equiv="Content-Security-Policy"/,
    );
    const content = html.match(/http-equiv="Content-Security-Policy" content="([^"]*)"/)![1];
    expect(content).toBe(CONSOLE_CSP(inlineScriptHashes(PAGE)));
    expect(patchDocument(html)).toEqual({ html, patched: false });
  });

  it('refuses a document without <head>', () => {
    expect(() => patchDocument('<html><body></body></html>')).toThrow(/head/);
  });
});

describe('patchDirectory', () => {
  function tree(files: Record<string, string>) {
    const dir = mkdtempSync(path.join(tmpdir(), 'csp-'));
    for (const [rel, body] of Object.entries(files)) {
      mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
      writeFileSync(path.join(dir, rel), body);
    }
    return dir;
  }

  it('patches every html file under the directory', () => {
    const dir = tree({ 'index.html': PAGE, 'models/index.html': PAGE, 'a.js': 'x' });
    expect(patchDirectory(dir)).toEqual({ patched: 2, total: 2 });
    expect(readFileSync(path.join(dir, 'models/index.html'), 'utf8')).toContain('Content-Security-Policy');
  });

  it('refuses a server build whose page route has no prerendered html (it would ship with no CSP)', () => {
    const dir = tree({ 'page.js': '', 'index.html': PAGE, 'models/page.js': '', 'tokens/page.js': '', 'tokens.html': PAGE });
    expect(unprerenderedRoutes(dir)).toEqual(['models']);
    expect(() => patchDirectory(dir)).toThrow(/models/);
    expect(readFileSync(path.join(dir, 'index.html'), 'utf8')).toBe(PAGE);
  });

  it('refuses a directory with no html at all', () => {
    expect(() => patchDirectory(tree({ 'a.js': 'x' }))).toThrow(/no \.html/);
  });

  it('the CLI exits non-zero when a target is missing', () => {
    const r = spawnSync(process.execPath, [SCRIPT, path.join(tmpdir(), 'no-such-dir-csp')], { encoding: 'utf8' });
    expect(r.status).not.toBe(0);
  });
});

describe('build wiring (both builds get the policy)', () => {
  it('targets out/ for the static export and every server html tree for standalone', () => {
    const fe = mkdtempSync(path.join(tmpdir(), 'csp-fe-'));
    expect(defaultTargets({ LMW_UI_EXPORT: '1' }, fe)).toEqual([path.join(fe, 'out')]);
    // Not a standalone build (no copy of the pages to patch): refuse.
    expect(() => defaultTargets({}, fe)).toThrow(/standalone/);
    // Docker layout, plus a nested one (a local build under a parent lockfile);
    // node_modules is never searched. server/pages holds the pages-router
    // 404.html / 500.html Next still emits, which the server can answer with.
    for (const rel of [
      '.next/server/app',
      '.next/server/pages',
      '.next/standalone/.next/server/app',
      '.next/standalone/.next/server/pages',
      '.next/standalone/apps/x/frontend/.next/server/app',
      '.next/standalone/apps/x/frontend/.next/server/pages',
      '.next/standalone/node_modules/y/.next/server/app',
    ]) {
      mkdirSync(path.join(fe, rel), { recursive: true });
    }
    const server = [
      path.join(fe, '.next/server/app'),
      path.join(fe, '.next/server/pages'),
      path.join(fe, '.next/standalone/.next/server/app'),
      path.join(fe, '.next/standalone/.next/server/pages'),
      path.join(fe, '.next/standalone/apps/x/frontend/.next/server/app'),
      path.join(fe, '.next/standalone/apps/x/frontend/.next/server/pages'),
    ];
    expect(defaultTargets({}, fe)).toEqual(server);
    expect(defaultTargets({ LMW_UI_EXPORT: '0' }, fe)).toEqual(server);
  });

  it('a standalone run patches the pages-router 404/500 too, so no served html lacks the policy', () => {
    const fe = mkdtempSync(path.join(tmpdir(), 'csp-run-'));
    const files: Record<string, string> = {
      '.next/server/app/index.html': PAGE,
      '.next/server/app/page.js': '',
      '.next/server/pages/404.html': PAGE,
      '.next/server/pages/500.html': PAGE,
      '.next/standalone/.next/server/app/index.html': PAGE,
      '.next/standalone/.next/server/app/page.js': '',
      '.next/standalone/.next/server/pages/404.html': PAGE,
    };
    for (const [rel, body] of Object.entries(files)) {
      mkdirSync(path.dirname(path.join(fe, rel)), { recursive: true });
      writeFileSync(path.join(fe, rel), body);
    }
    run({}, fe, [], () => {});
    for (const rel of Object.keys(files).filter((f) => f.endsWith('.html'))) {
      expect(readFileSync(path.join(fe, rel), 'utf8'), rel).toContain('http-equiv="Content-Security-Policy"');
    }
  });

  it('pagesWithoutCsp lists every html under a tree that lacks the meta tag', () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'csp-check-'));
    mkdirSync(path.join(dir, 'x'));
    writeFileSync(path.join(dir, 'a.html'), patchDocument(PAGE).html);
    writeFileSync(path.join(dir, 'x', 'b.html'), PAGE);
    writeFileSync(path.join(dir, 'c.js'), PAGE);
    expect(pagesWithoutCsp(dir)).toEqual([path.join(dir, 'x', 'b.html')]);
  });

  it('the build fails when, after patching, a page still lacks a real CSP meta tag', () => {
    // The marker text without the tag (here in a comment) makes patchDocument
    // skip the page as "already patched"; the post-check still catches it.
    const fe = mkdtempSync(path.join(tmpdir(), 'csp-fail-'));
    mkdirSync(path.join(fe, 'out', 'x'), { recursive: true });
    writeFileSync(path.join(fe, 'out', 'index.html'), PAGE);
    writeFileSync(
      path.join(fe, 'out', 'x', 'index.html'),
      PAGE.replace('<head>', '<head><!-- http-equiv="Content-Security-Policy" -->'),
    );
    expect(() => run({ LMW_UI_EXPORT: '1' }, fe, [], () => {})).toThrow(/without the CSP meta tag.*x\/index\.html/);
  });

  it('npm run build (the Docker image build) runs the patcher after next build', () => {
    const pkg = JSON.parse(readFileSync(path.join(FRONTEND, 'package.json'), 'utf8'));
    expect(pkg.scripts.build).toMatch(/^next build && node scripts\/patch-csp-meta\.mjs$/);
  });

  it('the macOS bundle step patches the export after its next build', () => {
    const sh = readFileSync(path.join(FRONTEND, '..', 'scripts', 'macos', 'bundle-webui.sh'), 'utf8');
    const build = sh.indexOf('./node_modules/.bin/next build');
    const patch = sh.indexOf('&& node scripts/patch-csp-meta.mjs out)');
    expect(build).toBeGreaterThan(-1);
    expect(patch).toBeGreaterThan(build);
  });
});
