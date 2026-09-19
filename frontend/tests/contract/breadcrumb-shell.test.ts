import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { BREADCRUMB_BAR_PX } from '@/components/breadcrumb-header';

const read = (p: string) => readFileSync(path.resolve(__dirname, '../../src', p), 'utf8');

// The breadcrumb strip lives in the root layout, between the nav bar and
// <main>. Pages reach the provider through useBreadcrumb, so it has to wrap
// <main> too — outside it, useBreadcrumb silently does nothing.
describe('breadcrumb strip in the app shell', () => {
  const layout = read('app/layout.tsx');

  it('renders the strip right under the nav bar, inside the provider that also wraps <main>', () => {
    const open = layout.indexOf('<NavStackProvider>');
    const nav = layout.indexOf('<NavBar />');
    const strip = layout.indexOf('<BreadcrumbHeader />');
    const main = layout.indexOf('<main id="app-root"');
    const close = layout.indexOf('</NavStackProvider>');
    expect(open).toBeGreaterThan(-1);
    expect(open).toBeLessThan(nav);
    expect(nav).toBeLessThan(strip);
    expect(strip).toBeLessThan(main);
    expect(main).toBeLessThan(close);
  });

  // chat2 is a full-height shell sized off the viewport; it must subtract the
  // strip or the page grows a scrollbar the height of the strip.
  it("chat2's full-height calc subtracts the strip's height", () => {
    expect(read('components/breadcrumb-header.tsx')).toContain(`h-[${BREADCRUMB_BAR_PX}px]`);
    expect(read('app/chat2/page.tsx')).toContain(`h-[calc(100vh-3.5rem-${BREADCRUMB_BAR_PX}px)]`);
  });

  // One trail only: the pages that used to draw their own crumbs or back
  // links now name themselves through useBreadcrumb instead.
  it.each([
    'components/tokens/detail/token-header.tsx',
    'app/tokens/[id]/page.tsx',
    'app/models/[id]/page.tsx',
    'app/models/[id]/settings/page.tsx',
  ])('%s draws no breadcrumb of its own', (file) => {
    const src = read(file);
    expect(src).not.toContain('aria-label="Breadcrumb"');
    expect(src).not.toMatch(/← Back to/);
  });

  it.each(['app/tokens/[id]/page.tsx', 'app/models/[id]/page.tsx', 'app/models/[id]/settings/page.tsx'])(
    '%s calls useBreadcrumb',
    (file) => expect(read(file)).toMatch(/useBreadcrumb\(\{/),
  );
});
