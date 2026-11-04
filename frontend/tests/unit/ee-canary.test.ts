import { describe, it, expect } from 'vitest';
import { mkdtempSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { findCanary, EE_CANARY } from '../../scripts/check-ee-canary.mjs';

// A community build must not contain the enterprise canary. The string is
// built here (not written literally) so this file is never a false positive.
describe('ee canary check', () => {
  it('detects the canary and passes when absent', () => {
    const d = mkdtempSync(path.join(tmpdir(), 'canary-'));
    mkdirSync(path.join(d, 'static'));
    writeFileSync(path.join(d, 'static', 'a.js'), 'console.log(1)');
    expect(findCanary(d)).toEqual([]);
    writeFileSync(path.join(d, 'static', 'b.js'), `x="${EE_CANARY}"`);
    expect(findCanary(d)).toHaveLength(1);
  });

  it('a community build output (if one exists) is canary-free', () => {
    const next = path.resolve(__dirname, '../../.next');
    if (process.env.LM_EDITION === 'enterprise' || !existsSync(next)) return;
    expect(findCanary(next)).toEqual([]);
  });

  it('--present passes only when the canary is there (the Enterprise build)', () => {
    const script = path.resolve(__dirname, '../../scripts/check-ee-canary.mjs');
    const run = (d: string, ...a: string[]) =>
      spawnSync(process.execPath, [script, ...a, d], { encoding: 'utf8' }).status;
    const d = mkdtempSync(path.join(tmpdir(), 'canary-'));
    writeFileSync(path.join(d, 'a.js'), 'console.log(1)');
    expect(run(d)).toBe(0);
    expect(run(d, '--present')).toBe(1);
    writeFileSync(path.join(d, 'b.js'), `x="${EE_CANARY}-0123abcd"`);
    expect(run(d)).toBe(1);
    expect(run(d, '--present')).toBe(0);
  });
});
