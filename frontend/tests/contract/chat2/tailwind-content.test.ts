import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

// REGRESSION (2026-08-24): tailwind.config only scanned src/components and
// src/app, so every utility class used exclusively under src/features/ (all
// of chat2 at the time) generated no CSS — "-m-6" on the chat root silently
// became margin:0 (page scrolled, composer under the fold), "-translate-x-1/2"
// on jump-to-latest vanished (button stuck left), and the sidebar's own
// overflow-y-auto scroll went with it. Every directory that contains
// className strings MUST be in the content globs.
//
// src/features/ itself is gone (2026-08-25): chat2 now renders from the
// @podwarden/chat-ui package, so the glob for it was dropped along with the
// rest of the in-tree feature. Keep this list in sync with the directories
// that actually exist under src/.
describe('tailwind content globs', () => {
  it('scan every className-bearing source directory', () => {
    const cfg = readFileSync(path.resolve(__dirname, '../../../tailwind.config.ts'), 'utf8');
    for (const dir of ['components', 'app', 'lib']) {
      expect(cfg).toContain(`./src/${dir}/**/*.{js,ts,jsx,tsx,mdx}`);
    }
    expect(cfg).not.toContain('./src/features');
  });

  it('scans the package dist and applies its preset', () => {
    const cfg = readFileSync(path.resolve(__dirname, '../../../tailwind.config.ts'), 'utf8');
    expect(cfg).toContain('./node_modules/@podwarden/chat-ui/dist/**/*.js');
    expect(cfg).toContain("require('@podwarden/chat-ui/tailwind-preset')");
  });
});
