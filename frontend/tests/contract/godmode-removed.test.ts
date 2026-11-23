import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

// Spec D7: the docks are the only Prompt Lens UI (since 2026-11-22 the shared
// dock on /stats, /key/stats and the token page). The old /godmode page is
// removed, not redirected, so /ui/godmode falls through to the app 404.
const src = (p: string) => path.resolve(__dirname, '../../src', p);

describe('/godmode removal (D7)', () => {
  it('has no /godmode route', () => {
    expect(existsSync(src('app/godmode'))).toBe(false);
  });
  it('/stats does not link to /godmode', () => {
    // Any "/godmode" route segment ("/godmode", "/ui/godmode", "/godmode/x"),
    // on lines other than imports: the dock's module path
    // ("@/components/godmode/dock/...") is not a link.
    const lines = readFileSync(src('app/stats/page.tsx'), 'utf8')
      .split('\n')
      .filter((l) => !/^\s*import\b|\bfrom\s+["']@\//.test(l));
    for (const l of lines) expect(l).not.toMatch(/\/godmode(?![\w-])/);
  });
  it('no comment still claims a selection is shared with /godmode', () => {
    for (const f of ['lib/model-selection.ts', 'components/stats/model-selector.tsx']) {
      if (!existsSync(src(f))) continue; // the model selector left with the stats redesign
      expect(readFileSync(src(f), 'utf8')).not.toContain('/godmode');
    }
  });
});
