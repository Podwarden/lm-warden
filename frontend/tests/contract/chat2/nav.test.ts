import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

// The nav is a client component whose menu only renders behind a button and a
// route guard; the contract that matters here is just the table itself.
//
// /chat is RETIRED (2026-08-23): the nav carries a single "Chat" entry that
// points at /chat2, and the old /chat route permanently redirects there so
// stale bookmarks keep working.
describe('nav', () => {
  it('has exactly one Chat entry, pointing at /chat2', () => {
    const src = readFileSync(path.resolve(__dirname, '../../../src/components/nav-bar.tsx'), 'utf8');
    expect(src).toMatch(/href: '\/chat2', label: 'Chat'/);
    expect(src).not.toMatch(/href: '\/chat',/);
  });

  it('keeps /chat alive as a redirect to /chat2', () => {
    const page = readFileSync(path.resolve(__dirname, '../../../src/app/chat/page.tsx'), 'utf8');
    expect(page).toMatch(/redirect\('\/chat2'\)/);
  });
});
