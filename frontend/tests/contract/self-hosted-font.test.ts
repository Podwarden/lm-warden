import { describe, it, expect } from 'vitest';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

// conventions: ui.typeface@1 -- the console ships DM Sans itself and loads
// nothing from a font CDN. A request to Google Fonts shares every console
// visitor's IP address with Google, and fails on an air-gapped install (the
// retro themes then fell back to system-ui).

const FRONTEND = path.resolve(__dirname, '../..');
const SRC = path.join(FRONTEND, 'src');
const FONTS = path.join(SRC, 'fonts');
const CDN = /fonts\.googleapis\.com|fonts\.gstatic\.com/;

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = path.join(dir, name);
    return statSync(p).isDirectory() ? files(p) : [p];
  });
}

describe('console typeface (ui.typeface)', () => {
  it('no source file references a font CDN', () => {
    const hits = files(SRC)
      .filter((f) => /\.(tsx?|css|mjs|js|json|html)$/.test(f))
      .filter((f) => CDN.test(readFileSync(f, 'utf8')))
      .map((f) => path.relative(FRONTEND, f));
    expect(hits).toEqual([]);
  });

  it('declares DM Sans from font files committed beside the stylesheet', () => {
    const css = readFileSync(path.join(FONTS, 'dm-sans.css'), 'utf8');
    expect(css).toMatch(/font-family:\s*'DM Sans'/);
    const urls = [...css.matchAll(/url\(([^)]+)\)/g)].map((m) => m[1].replace(/['"]/g, ''));
    expect(urls.length).toBeGreaterThan(0);
    for (const u of urls) {
      expect(u).toMatch(/^\.\/dm-sans\/[\w.-]+\.woff2$/);
      expect(existsSync(path.join(FONTS, u)), u).toBe(true);
    }
    // Normal and italic, so <em> in the retro themes is a real italic.
    expect(css).toMatch(/font-style:\s*italic/);
    expect(css).toMatch(/font-style:\s*normal/);
  });

  it('ships the SIL Open Font License beside the font files', () => {
    const licence = readFileSync(path.join(FONTS, 'dm-sans', 'OFL.txt'), 'utf8');
    expect(licence).toContain('SIL OPEN FONT LICENSE Version 1.1');
    expect(licence).toContain('The DM Sans Project Authors');
  });

  it('the root layout imports the self-hosted font', () => {
    const layout = readFileSync(path.join(SRC, 'app', 'layout.tsx'), 'utf8');
    expect(layout).toMatch(/import\s+['"]@\/fonts\/dm-sans\.css['"]/);
  });

  // Build output check: whichever of the two builds is present locally
  // (standalone .next/, static export out/) must not reference a font CDN
  // either. Absent in a clean checkout, so each is checked only if it exists.
  for (const dir of ['.next/static', '.next/server/app', 'out']) {
    const abs = path.join(FRONTEND, dir);
    it.runIf(existsSync(abs))(`built output ${dir} references no font CDN`, () => {
      const hits = files(abs)
        .filter((f) => /\.(html|js|css|rsc|txt|json)$/.test(f))
        .filter((f) => CDN.test(readFileSync(f, 'utf8')))
        .map((f) => path.relative(FRONTEND, f));
      expect(hits).toEqual([]);
    });
  }
});
