import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const TOKENS = [
  'page', 'surface', 'surface-2', 'rule', 'fg', 'muted', 'dim', 'accent',
  'accent-strong', 'user', 'warn', 'warn-dim', 'negative', 'negative-dim',
  'positive', 'code', 'on-accent',
];

describe('chat theme variables', () => {
  const css = readFileSync(path.resolve(__dirname, '../../../src/app/globals.css'), 'utf8');

  it('imports the package theme defaults before the host overrides', () => {
    expect(css.indexOf("@import '@podwarden/chat-ui/theme.css'")).toBeGreaterThanOrEqual(0);
    expect(css.indexOf("@import '@podwarden/chat-ui/theme.css'")).toBeLessThan(css.indexOf('[data-theme="retro"]'));
  });

  for (const theme of ['retro', 'retro-dark']) {
    it(`defines all 17 --chat-* triplets under [data-theme="${theme}"]`, () => {
      const start = css.indexOf(`[data-theme="${theme}"] {`);
      const block = css.slice(start, css.indexOf('\n}\n', start));
      for (const t of TOKENS) expect(block).toMatch(new RegExp(`--chat-${t}:\\s*\\d+ \\d+ \\d+;`));
    });
  }
});
