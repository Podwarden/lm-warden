// lib/routes.ts — the detail pages' query-id URLs (BACKLOG-macos P4). A
// static export cannot pre-render /models/[id] per runtime id, so the pages
// live at /models/detail?id=, /models/detail/settings?id= and
// /tokens/detail?id=; the breadcrumb registry and the back stack keep
// speaking the path-shaped logical form, mapped both ways here.
import { describe, expect, it } from 'vitest';
import {
  detailHref,
  isQueryDetailPath,
  logicalPath,
  modelHref,
  modelSettingsHref,
  tokenHref,
} from '@/lib/routes';

describe('detail hrefs', () => {
  it('put the id in the query, percent-encoded', () => {
    expect(modelHref('abc')).toBe('/models/detail?id=abc');
    expect(modelSettingsHref('abc')).toBe('/models/detail/settings?id=abc');
    expect(tokenHref('tok-1')).toBe('/tokens/detail?id=tok-1');
    // '&', '=', '#', '?', '/' and spaces cannot add a parameter or a segment.
    expect(modelHref('a&x=1#f?g/h i')).toBe('/models/detail?id=a%26x%3D1%23f%3Fg%2Fh%20i');
  });
});

describe('logicalPath', () => {
  it.each([
    ['/models/detail', 'id=abc', '/models/abc'],
    ['/models/detail/', 'id=abc', '/models/abc'], // trailingSlash export
    ['/models/detail/settings', 'id=abc', '/models/abc/settings'],
    ['/models/detail/settings/', 'id=abc&x=1', '/models/abc/settings'],
    ['/tokens/detail', 'id=tok-1&range=7d', '/tokens/tok-1'],
    ['/tokens/detail', 'id=a%2Fb', '/tokens/a%2Fb'],
  ])('%s?%s -> %s', (path, search, logical) => {
    expect(logicalPath(path, search)).toBe(logical);
    expect(logicalPath(path, new URLSearchParams(search))).toBe(logical);
  });

  it.each([
    ['/models/detail', ''],
    ['/tokens/detail', 'range=7d'],
    ['/models', 'id=abc'],
    ['/stats', 'id=abc'],
  ])('leaves %s?%s alone (no id, or not a detail page)', (path, search) => {
    expect(logicalPath(path, search)).toBe(path);
  });

  it('isQueryDetailPath names exactly the three detail pages', () => {
    for (const p of ['/models/detail', '/models/detail/', '/models/detail/settings', '/tokens/detail']) {
      expect(isQueryDetailPath(p)).toBe(true);
    }
    for (const p of ['/models', '/models/abc', '/tokens/detail/x', '/stats']) expect(isQueryDetailPath(p)).toBe(false);
  });
});

describe('detailHref (logical -> real), the inverse of logicalPath', () => {
  it.each([
    ['/models/abc', '/models/detail?id=abc'],
    ['/models/abc/settings', '/models/detail/settings?id=abc'],
    ['/tokens/tok-1', '/tokens/detail?id=tok-1'],
    ['/tokens/a%2Fb', '/tokens/detail?id=a%2Fb'],
  ])('%s -> %s', (logical, real) => {
    expect(detailHref(logical)).toBe(real);
    const u = new URL(real, 'http://x');
    expect(logicalPath(u.pathname, u.search.slice(1))).toBe(logical);
  });

  it.each(['/', '/models', '/tokens', '/stats', '/router/stats', '/models/detail', '/models/a/b', '/stats/live/x'])(
    'leaves %s alone',
    (p) => expect(detailHref(p)).toBe(p),
  );
});
