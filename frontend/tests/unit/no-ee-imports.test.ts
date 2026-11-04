import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import path from 'node:path';

// Only src/lib/extensions.ts may import `@ee`; nothing outside src/ee may
// reach into src/ee directly (the community build has no such directory).

const SRC = path.resolve(__dirname, '../../src');
const ALLOWED = path.join(SRC, 'lib', 'extensions.ts');

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = path.join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : /\.(tsx?|mjs|jsx?)$/.test(n) ? [p] : [];
  });
}

const IMPORT_RE = /(?:from|import|require\()\s*['"]((?:@ee|@\/ee\/|(?:\.\.?\/)+ee\/|(?:\.\.?\/)+ee['"])[^'"]*)/;

describe('ee import boundary', () => {
  const files = walk(SRC).filter((f) => !f.startsWith(path.join(SRC, 'ee') + path.sep));
  it('scans a non-trivial tree', () => expect(files.length).toBeGreaterThan(50));
  it('only lib/extensions.ts imports @ee', () => {
    const bad = files.filter((f) => f !== ALLOWED && /['"]@ee['"]/.test(readFileSync(f, 'utf8')));
    expect(bad).toEqual([]);
  });
  it('nobody outside src/ee imports from src/ee directly', () => {
    const bad = files.filter((f) => f !== ALLOWED && IMPORT_RE.test(readFileSync(f, 'utf8')));
    expect(bad).toEqual([]);
  });
  it('extensions.ts does import @ee', () => {
    expect(readFileSync(ALLOWED, 'utf8')).toMatch(/from ['"]@ee['"]/);
  });
  it('the community stub holds no canary', () => {
    const stub = path.join(SRC, 'ee-stub');
    for (const f of walk(stub)) expect(readFileSync(f, 'utf8')).not.toContain('LMW-EE-CANARY');
    expect(existsSync(stub)).toBe(true);
  });
});
