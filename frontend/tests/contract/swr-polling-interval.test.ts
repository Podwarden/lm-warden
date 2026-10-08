import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const SRC = path.resolve(__dirname, '../../src');

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) ? [p] : [];
  });
}

// SWR 2.3 only reschedules its polling loop for a non-zero interval, so a
// refreshInterval function that returns 0 while the tab is hidden stops
// polling for good the first time the tab is hidden. SWR already skips ticks
// on a hidden tab (refreshWhenHidden defaults to false): pass a number.
describe('SWR polling intervals', () => {
  it('no refreshInterval turns to 0 on a hidden tab', () => {
    const offenders = sources(SRC).filter((f) =>
      /document\.hidden\s*\?\s*0\b/.test(readFileSync(f, 'utf8')),
    );
    expect(offenders.map((f) => path.relative(SRC, f))).toEqual([]);
  });
});
