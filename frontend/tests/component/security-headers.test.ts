import { describe, it, expect } from 'vitest';
import config from '../../next.config';
import { HSTS, securityHeaders } from '@/lib/security-headers';

// next.config.ts applies these to every /ui response. Pin the set, that it
// is wired, and that HSTS is conditional on the request having been HTTPS.

function matchesHas(value: string, pattern: string): boolean {
  // Next compiles `has[].value` as new RegExp('^' + value + '$').
  return new RegExp('^' + pattern + '$').test(value);
}

describe('UI security headers', () => {
  it('is what next.config.ts serves', async () => {
    expect(await config.headers!()).toEqual(securityHeaders());
  });

  it('sends the base set on every path', () => {
    const [base] = securityHeaders();
    expect(base.source).toBe('/:path*');
    expect(base.has).toBeUndefined();
    const byKey = Object.fromEntries(base.headers.map((h) => [h.key, h.value]));
    expect(byKey).toMatchObject({
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'X-Frame-Options': 'DENY',
    });
    expect(byKey['Permissions-Policy']).toContain('camera=()');
    expect(byKey['Content-Security-Policy']).toContain("frame-ancestors 'none'");
    expect(byKey['Strict-Transport-Security']).toBeUndefined();
  });

  it('sends HSTS only for a request forwarded as https', () => {
    const hsts = securityHeaders().find((r) =>
      r.headers.some((h) => h.key === 'Strict-Transport-Security'),
    )!;
    expect(hsts.headers).toEqual([{ key: 'Strict-Transport-Security', value: HSTS }]);
    const cond = hsts.has![0];
    expect(cond.key).toBe('x-forwarded-proto');
    expect(matchesHas('https', cond.value!)).toBe(true);
    expect(matchesHas('https, http', cond.value!)).toBe(true);
    expect(matchesHas('http', cond.value!)).toBe(false);
    expect(matchesHas('http, https', cond.value!)).toBe(false);
  });
});
