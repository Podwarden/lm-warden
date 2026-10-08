import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

// @podwarden/chat-ui's Modal sets `inert` on document.getElementById(rootInertId)
// while a modal is open (frontend/src/components/ui/modal.tsx does the same
// for our own modals). chat2/page.tsx passes rootInertId="app-root" to
// ChatApp, so the root layout's <main> MUST carry id="app-root" or the inert
// behaviour silently no-ops (getElementById returns null).
describe('root layout app-root id', () => {
  it('layout.tsx gives <main> id="app-root"', () => {
    const layout = readFileSync(
      path.resolve(__dirname, '../../src/app/layout.tsx'),
      'utf8',
    );
    expect(layout).toContain('id="app-root"');
  });
});
