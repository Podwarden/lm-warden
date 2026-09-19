import { describe, it, expect } from 'vitest';
// Local-time formatters: pin the zone so the expectations are stable. Node
// re-reads TZ on change, and nothing below touches Date before this line runs.
process.env.TZ = 'UTC';
import {
  compact, secs, day, hhmm, fmtWhen, fmtDateTime, parseSqliteUtc, relativeAgo,
  toInputValue, fromInputValue,
} from '@/lib/token-format';

const at = (y: number, mo: number, d: number, h: number, mi: number) => Date.UTC(y, mo - 1, d, h, mi) / 1000;

describe('token-format (ported from the mockup)', () => {
  it('compact', () => {
    expect([0, 0.5, 1.5, 42, 1500, 12345, 1_500_000, 110_000_000, 2.5e9].map(compact))
      .toEqual(['0', '0.50', '1.5', '42', '1.5k', '12k', '1.5M', '110M', '2.5B']);
  });
  it('secs', () => {
    expect([null, 0, 0.25, 2.5, 90].map(secs)).toEqual(['–', '0 ms', '250 ms', '2.5 s', '1.5 min']);
  });
  it('day / hhmm / fmtWhen', () => {
    const t = at(2026, 9, 6, 13, 59);
    expect(day(t)).toBe('Sep 6');
    expect(hhmm(t)).toBe('13:59');
    expect(fmtWhen(t, 60)).toBe('13:59');
    expect(fmtWhen(t, 1440)).toBe('13:59');
    expect(fmtWhen(t, 2880)).toBe('Sep 6 13:59');
    expect(fmtWhen(t, 4000)).toBe('Sep 6 13:59');
    expect(fmtWhen(t, 10080)).toBe('Sep 6');
  });
  it('fmtDateTime adds the year only when it is not this year', () => {
    const now = at(2026, 9, 18, 13, 21);
    expect(fmtDateTime(at(2026, 9, 6, 13, 59), now)).toBe('Sep 6, 13:59');
    expect(fmtDateTime(at(2025, 9, 6, 13, 59), now)).toBe('Sep 6 2025, 13:59');
  });
  it('parseSqliteUtc reads naive UTC', () => {
    expect(parseSqliteUtc('2026-09-06 13:59:00')).toBe(at(2026, 9, 6, 13, 59));
    expect(parseSqliteUtc(null)).toBeNull();
    expect(parseSqliteUtc('garbage')).toBeNull();
  });
  it('relativeAgo', () => {
    const now = at(2026, 9, 18, 13, 21);
    expect(relativeAgo(null, now)).toBe('Never');
    expect(relativeAgo(now - 20, now)).toBe('just now');
    expect(relativeAgo(now - 120, now)).toBe('2 min ago');
    expect(relativeAgo(now - 3 * 3600, now)).toBe('3 h ago');
    expect(relativeAgo(now - 86400, now)).toBe('1 day ago');
    expect(relativeAgo(now - 5 * 86400, now)).toBe('5 days ago');
  });
  it('datetime-local round trip, minute precision', () => {
    const t = at(2026, 9, 12, 9, 0);
    expect(toInputValue(t)).toBe('2026-09-12T09:00');
    expect(fromInputValue('2026-09-12T09:00')).toBe(t);
    expect(fromInputValue('')).toBeNull();
  });
});
