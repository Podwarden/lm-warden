// Metal design §1 (P4, frontend part 2): the three redirect pages must be
// client components. A server-side `redirect()` throws during render, which
// is legal in a server component but impossible in a statically exported
// build (next.config gains `output: 'export'`); `router.replace()` in
// useEffect is the export-safe equivalent, and the targets are
// byte-identical to today's redirect() targets:
//   /            -> /stats
//   /chat        -> /chat2
//   /setup       -> /setup/welcome
//
// The render legs mock next/navigation's useRouter (vi.mock is hoisted above
// the in-test page imports) and assert the exact replace target; the
// source-text legs pin what the render cannot see: no `redirect` import from
// next/navigation remains, and each file's first statement is 'use client'.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createElement } from 'react';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const replace = vi.fn();
vi.mock('next/navigation', () => ({ useRouter: () => ({ replace }) }));

const src = (p: string) => path.resolve(__dirname, '../../src', p);

describe('client-side redirect pages (export-safe)', () => {
  const pages = [
    { file: 'app/page.tsx', target: '/stats' },
    { file: 'app/chat/page.tsx', target: '/chat2' },
    { file: 'app/setup/page.tsx', target: '/setup/welcome' },
  ];

  beforeEach(() => {
    replace.mockClear();
  });

  it.each(pages)('$file renders nothing and router.replace($target)', async ({ file, target }) => {
    const { render } = await import('@testing-library/react');
    const { default: Page } = await import(`../../src/${file}`);
    const { container } = render(createElement(Page));
    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith(target);
    expect(container.innerHTML).toBe('');
  });

  it.each(pages)('$file no longer imports redirect from next/navigation', ({ file }) => {
    expect(readFileSync(src(file), 'utf8')).not.toMatch(
      /import\s+.*redirect.*from\s+["']next\/navigation["']/,
    );
  });

  it.each(pages)("$file's first statement is 'use client'", ({ file }) => {
    const first = readFileSync(src(file), 'utf8').split('\n', 1)[0].trim();
    expect(first).toBe("'use client';");
  });
});
