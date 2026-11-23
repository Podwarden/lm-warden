#!/usr/bin/env node
// The console's script Content-Security-Policy, patched into the prerendered
// HTML after `next build`.
// conventions: security.headers@2 (pattern copied from Crawl Warden's
// frontend/scripts/patch-csp-meta.mjs and the conventions scaffold).
//
// WHY A META TAG, NOT A HEADER. The Next 15 app router puts its bootstrap and
// RSC flight data in INLINE <script> tags in every page, and has no nonce
// support for prerendered pages (a nonce needs middleware and forces every
// page to render per request, which the static export cannot do at all). A
// header carrying script-src would block those scripts: white screen. So each
// page gets a <meta http-equiv="Content-Security-Policy"> whose script-src is
// 'self' plus the sha256 of THAT page's own inline scripts. The HTML is fixed
// at build time, so its hashes are too, and an injected <script> (inline or
// from another host) matches none of them and does not run. The response
// header (src/lib/security-headers.ts, app/webui/mount.py on the Mac) keeps
// only directives that cannot break a page, frame-ancestors among them: a
// meta tag may not carry frame-ancestors.
//
// BOTH BUILDS. The Docker image runs `npm run build` = `next build && node
// scripts/patch-csp-meta.mjs`: with no arguments the targets are the server's
// html trees, .next/server/app (prerendered pages) and .next/server/pages (the
// pages-router 404/500), and their copies under .next/standalone (the
// standalone server answers with those files). After patching, any html in a
// target without the policy fails the build. The macOS app's
// static export (LMW_UI_EXPORT=1) targets out/; scripts/macos/bundle-webui.sh
// runs `node scripts/patch-csp-meta.mjs out` after its own `next build` and
// refuses an export page without the tag.
//
// Every console route is prerendered (`○ (Static)` in the build table). A page
// route rendered per request would have no .html to patch and would ship with
// no script policy, so the server-build targets are refused, before anything
// is written, when any page.js has no prerendered .html beside it.

import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The policy, minus what a <meta> CSP may not carry (frame-ancestors,
 * report-uri, sandbox). `hashes` are CSP source expressions
 * ("'sha256-...'"). tests/unit/patch-csp-meta.test.ts fails if script-src
 * goes missing or gains 'unsafe-inline' / 'unsafe-eval'.
 *
 * - style-src 'unsafe-inline': React style={{...}} attributes, recharts,
 *   chat-ui and KaTeX set inline styles; the conventions allow it for styles.
 * - connect-src 'self': fetch, EventSource (/api/... SSE) and the /v1 calls
 *   are all same-origin. The console opens no WebSocket.
 * - img-src data: blob:: inline SVG icons and Prompt Lens images shown through
 *   object URLs.
 * - worker-src 'self' blob:: no worker today; listed so a bundled worker
 *   does not silently fall back to script-src.
 */
export const CONSOLE_CSP = (hashes) =>
  [
    "default-src 'self'",
    ["script-src", "'self'", ...hashes].join(' '),
    "style-src 'self' 'unsafe-inline'",
    "font-src 'self'",
    "img-src 'self' data: blob:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
  ].join('; ');

const META_MARKER = 'http-equiv="Content-Security-Policy"';

// An inline script is a <script> with no src. Next emits them attribute-less;
// any other attribute is hashed too unless it marks non-executable data
// (type="application/json" etc.), which script-src does not govern.
const SCRIPT_RE = /<script(\s[^>]*)?>([\s\S]*?)<\/script>/g;
const EXECUTABLE_TYPE = /^(|text\/javascript|application\/javascript|module)$/i;

function isInlineExecutable(attrs) {
  if (/\ssrc\s*=/i.test(attrs)) return false;
  const type = attrs.match(/\stype\s*=\s*["']?([^"'\s>]*)/i);
  return !type || EXECUTABLE_TYPE.test(type[1]);
}

/** CSP sources ('sha256-<base64>') of each executable inline script, deduped, in order. */
export function inlineScriptHashes(html) {
  const out = [];
  for (const m of html.matchAll(SCRIPT_RE)) {
    const attrs = m[1] ?? '';
    const body = m[2];
    if (!isInlineExecutable(attrs) || body.length === 0) continue;
    const h = `'sha256-${createHash('sha256').update(body, 'utf8').digest('base64')}'`;
    if (!out.includes(h)) out.push(h);
  }
  return out;
}

// Next puts <meta charSet="utf-8"/> first in <head>. It must stay within the
// first 1024 bytes for the browser's encoding sniffing, and a CSP meta tag
// with a page's worth of hashes is close to that long, so ours goes right
// after it: still ahead of every <script>, which is what the policy needs.
const CHARSET_META = /^<meta charset="[^"]*"\s*\/?>/i;

/** Insert the page's CSP meta tag at the top of <head> (after a leading charset meta). Idempotent. */
export function patchDocument(html) {
  if (html.includes(META_MARKER)) return { html, patched: false };
  const i = html.indexOf('<head>');
  if (i === -1) throw new Error('no <head> in the document: not a Next page?');
  let at = i + '<head>'.length;
  const charset = html.slice(at).match(CHARSET_META);
  if (charset) at += charset[0].length;
  const meta = `<meta ${META_MARKER} content="${CONSOLE_CSP(inlineScriptHashes(html))}"/>`;
  return { html: html.slice(0, at) + meta + html.slice(at), patched: true };
}

function htmlFiles(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) htmlFiles(p, out);
    else if (e.name.endsWith('.html')) out.push(p);
  }
  return out;
}

/**
 * Page routes of a server build with no prerendered document. A nested
 * <route>/page.js has <route>.html beside the <route>/ directory; the root
 * page.js has index.html beside it. Empty for a static export (no page.js).
 */
export function unprerenderedRoutes(dir) {
  const out = [];
  const walk = (d) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      if (e.isDirectory()) walk(join(d, e.name));
      else if (e.name === 'page.js') {
        const root = d === dir;
        const html = root ? join(d, 'index.html') : `${d}.html`;
        if (!existsSync(html)) out.push(root ? '/' : d.slice(dir.length + 1));
      }
    }
  };
  walk(dir);
  return out.sort();
}

/** Patch every .html under `dir` in place; refuses (writing nothing) on an unprerendered page route. */
export function patchDirectory(dir) {
  if (!existsSync(dir)) throw new Error(`no such directory: ${dir}`);
  const missing = unprerenderedRoutes(dir);
  if (missing.length > 0) {
    throw new Error(`page routes with no prerendered html (they would have no CSP): ${missing.join(', ')}`);
  }
  const files = htmlFiles(dir);
  if (files.length === 0) throw new Error(`no .html files under ${dir}: nothing to patch`);
  let patched = 0;
  for (const f of files) {
    const r = patchDocument(readFileSync(f, 'utf8'));
    if (r.patched) {
      writeFileSync(f, r.html, 'utf8');
      patched += 1;
    }
  }
  return { patched, total: files.length };
}

// Where a server build keeps html: the app router's prerendered pages, and the
// pages-router 404.html / 500.html Next still emits (the server can answer an
// unknown route or an error with them).
const SERVER_HTML_DIRS = [join('.next', 'server', 'app'), join('.next', 'server', 'pages')];

/**
 * The standalone server's copies of the server html trees: every
 * `.next/server/{app,pages}` under .next/standalone (outside node_modules). In
 * the Docker build that is .next/standalone/.next/server/...; a local build
 * under a parent directory with its own lockfile nests it deeper
 * (.next/standalone/<path from the inferred workspace root>/.next/server/...).
 */
export function standaloneHtmlDirs(frontend) {
  const root = join(frontend, '.next', 'standalone');
  const out = [];
  const walk = (d) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      if (!e.isDirectory() || e.name === 'node_modules') continue;
      const p = join(d, e.name);
      if (SERVER_HTML_DIRS.some((s) => p.endsWith(s))) out.push(p);
      else walk(p);
    }
  };
  if (existsSync(root)) walk(root);
  return out.sort();
}

/**
 * What `node scripts/patch-csp-meta.mjs` (no arguments) patches for this
 * build mode: out/ for the static export; for the standalone build the
 * server's html trees and the standalone copies of them (which must exist).
 */
export function defaultTargets(env, frontend) {
  if (env.LMW_UI_EXPORT === '1') return [join(frontend, 'out')];
  const copies = standaloneHtmlDirs(frontend);
  if (!copies.some((c) => c.endsWith(SERVER_HTML_DIRS[0]))) {
    throw new Error(`no .next/server/app under ${join(frontend, '.next', 'standalone')}: not a standalone build?`);
  }
  const own = SERVER_HTML_DIRS.map((d) => join(frontend, d)).filter((d, i) => i === 0 || existsSync(d));
  return [...own, ...copies];
}

// A real policy tag: the marker inside a <meta> that carries a script-src.
const CSP_META_RE = /<meta http-equiv="Content-Security-Policy" content="[^"]*\bscript-src [^"]*"\s*\/?>/;

/** Every .html under `dir` without a CSP meta tag carrying script-src. */
export function pagesWithoutCsp(dir) {
  return htmlFiles(dir)
    .filter((f) => !CSP_META_RE.test(readFileSync(f, 'utf8')))
    .sort();
}

/**
 * Patch `args` (or the build mode's default targets), then refuse the build if
 * any html in those trees still has no policy: no page ships without one, even
 * if a future Next layout puts html somewhere patchDocument skips.
 */
export function run(env, frontend, args, log = console.log) {
  const targets = args.length > 0 ? args : defaultTargets(env, frontend);
  for (const t of targets) {
    const { patched, total } = patchDirectory(t);
    log(`patch-csp-meta: ${t}: CSP meta tag added to ${patched} of ${total} pages`);
  }
  const missing = targets.flatMap((t) => pagesWithoutCsp(t));
  if (missing.length > 0) {
    throw new Error(`pages without the CSP meta tag after patching: ${missing.join(', ')}`);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run(process.env, resolve(dirname(fileURLToPath(import.meta.url)), '..'), process.argv.slice(2));
}
