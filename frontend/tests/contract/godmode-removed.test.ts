import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

// Spec D7: the token page's dock is the only god-mode UI. The old page is
// removed, not redirected, so /ui/godmode falls through to the app 404.
const src = (p: string) => path.resolve(__dirname, '../../src', p);

describe('/godmode removal (D7)', () => {
  it('has no /godmode route', () => {
    expect(existsSync(src('app/godmode'))).toBe(false);
  });
  it('/stats does not link to /godmode', () => {
    expect(readFileSync(src('app/stats/page.tsx'), 'utf8')).not.toContain('/godmode');
  });
  it('no comment still claims a selection is shared with /godmode', () => {
    for (const f of ['lib/model-selection.ts', 'components/stats/model-selector.tsx']) {
      expect(readFileSync(src(f), 'utf8')).not.toContain('/godmode');
    }
  });
});
