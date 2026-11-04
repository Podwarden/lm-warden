import { describe, expect, it, beforeEach } from 'vitest';

import { detailHref, isQueryDetailPath, logicalPath, sessionHref, SESSION_DETAIL_PATH } from '@/lib/routes';
import { isSessionsPath, keyMode, isKeyCacheKey, storeKeyToken, keyTokenState, readKeyToken, clearKeyToken } from '@/lib/sessions/key-token';
import { buildBreadcrumbs, shouldHideBreadcrumb } from '@/lib/breadcrumbs';

describe('sessions routes', () => {
  beforeEach(() => sessionStorage.clear());

  it('builds and maps the detail href', () => {
    expect(SESSION_DETAIL_PATH).toBe('/sessions/detail');
    expect(sessionHref('s_ab')).toBe('/sessions/detail?id=s_ab');
    expect(isQueryDetailPath('/sessions/detail/')).toBe(true);
    expect(logicalPath('/sessions/detail', 'id=s_ab')).toBe('/sessions/s_ab');
    expect(detailHref('/sessions/s_ab')).toBe('/sessions/detail?id=s_ab');
  });

  it('labels the sessions breadcrumbs and hides the login', () => {
    const crumbs = buildBreadcrumbs('/sessions/s_ab');
    expect(crumbs.map((c) => c.label)).toContain('Sessions');
    expect(shouldHideBreadcrumb('/sessions/login')).toBe(true);
  });

  it('knows the sessions paths exactly', () => {
    for (const p of ['/sessions', '/sessions/', '/sessions/detail', '/sessions/detail/']) expect(isSessionsPath(p)).toBe(true);
    for (const p of ['/sessions/login', '/sessionsx', '/stats']) expect(isSessionsPath(p)).toBe(false);
  });

  it('holds only a valid unexpired token; key mode needs one', () => {
    expect(keyTokenState()).toBe('absent');
    expect(keyMode('/sessions')).toBe(false);
    expect(storeKeyToken('jwt', 60)).toBe(true);
    expect(keyTokenState()).toBe('valid');
    expect(readKeyToken()).toBe('jwt');
    expect(keyMode('/sessions')).toBe(true);
    expect(keyMode('/stats')).toBe(false);
    sessionStorage.setItem('vw-sessions-exp', String(Date.now() - 1));
    expect(keyTokenState()).toBe('expired');
    expect(readKeyToken()).toBeNull();
    clearKeyToken();
    expect(keyTokenState()).toBe('absent');
  });

  it('tags key-mode cache keys', () => {
    expect(isKeyCacheKey(['sessions', 'key', '/api/sessions'])).toBe(true);
    expect(isKeyCacheKey(['sessions', 'admin', '/api/sessions'])).toBe(false);
    expect(isKeyCacheKey('/api/sessions')).toBe(false);
  });
});
