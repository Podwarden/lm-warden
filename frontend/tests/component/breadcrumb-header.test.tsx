// BreadcrumbHeader (src/components/breadcrumb-header.tsx) — ported from
// PodWarden Core's components/breadcrumb-header.test.tsx (#1687) and adapted:
// the real NavStackProvider drives it, and next/link is the real one with the
// app's `/ui` basePath switched on, so the hrefs asserted here are the ones a
// browser gets.
import React from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';

// next/link reads the basePath from this env var when its module loads
// (Next inlines it at build time from next.config.ts `basePath: '/ui'`).
// Vitest runs each test file in its own worker, so this does not leak.
const navState = vi.hoisted(() => {
  process.env.__NEXT_ROUTER_BASEPATH = '/ui';
  return { pathname: '/models', push: vi.fn() };
});
vi.mock('next/navigation', () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: navState.push, replace: vi.fn(), refresh: vi.fn(), prefetch: vi.fn() }),
}));

import { BreadcrumbHeader, truncate } from '@/components/breadcrumb-header';
import { NavStackProvider } from '@/lib/nav-stack';
import { useBreadcrumb } from '@/lib/use-breadcrumb';

function Title({ title, path }: { title?: string; path?: string }) {
  useBreadcrumb({ title, path });
  return null;
}

function tree(page?: React.ReactNode) {
  return (
    <NavStackProvider>
      <BreadcrumbHeader />
      {page}
    </NavStackProvider>
  );
}

function mount(path: string, page?: React.ReactNode) {
  navState.pathname = path;
  const r = render(tree(page));
  const go = async (to: string, next?: React.ReactNode) => {
    await act(async () => {
      navState.pathname = to;
      r.rerender(tree(next));
    });
  };
  return { ...r, go };
}

// The desktop <ol> — the mobile one duplicates the ends of the chain.
const desktop = () => screen.getByRole('navigation', { name: 'Breadcrumb' }).querySelector('ol')!;

afterEach(() => {
  cleanup();
  navState.pathname = '/models';
  navState.push.mockReset();
});

describe('BreadcrumbHeader — hidden routes', () => {
  it.each(['/login', '/setup', '/setup/welcome', '/setup/gpus', '/setup/admin'])('renders nothing on %s', (p) => {
    const { container } = mount(p);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders on every app-shell page', () => {
    mount('/stats');
    expect(screen.getByTestId('breadcrumb-header')).toBeInTheDocument();
  });
});

describe('BreadcrumbHeader — trail', () => {
  it('Home › API tokens › <token name>, links carry the /ui basePath', () => {
    mount('/tokens/tok-1', <Title title="ci-runner" />);
    const ol = within(desktop());
    // Home is `/` in the registry, but `/` only redirects to /models — the
    // link goes straight there.
    expect(ol.getByRole('link', { name: 'Home' })).toHaveAttribute('href', '/ui/models');
    expect(ol.getByRole('link', { name: 'API tokens' })).toHaveAttribute('href', '/ui/tokens');
    const current = ol.getByText('ci-runner');
    expect(current).toHaveAttribute('aria-current', 'page');
    expect(current.closest('a')).toBeNull();
  });

  it('Home › Models › <served name> › Settings on the model settings page', () => {
    mount('/models/abc/settings', <Title path="/models/abc" title="Qwen3-8B" />);
    const ol = within(desktop());
    expect(ol.getByRole('link', { name: 'Models' })).toHaveAttribute('href', '/ui/models');
    expect(ol.getByRole('link', { name: 'Qwen3-8B' })).toHaveAttribute('href', '/ui/models/abc');
    expect(ol.getByText('Settings')).toHaveAttribute('aria-current', 'page');
  });

  it('shows the placeholder in the pending style until the page names itself', async () => {
    const { go } = mount('/models/abc');
    const placeholder = within(desktop()).getByText('Model');
    expect(placeholder).toHaveAttribute('aria-current', 'page');
    expect(placeholder.className).toContain('italic');
    await go('/models/abc', <Title title="Qwen3-8B" />);
    const named = within(desktop()).getByText('Qwen3-8B');
    expect(named.className).not.toContain('italic');
    expect(named.className).toContain('text-chat-fg');
  });

  it('a segment with no page is plain text, not a link (PodWarden #1687)', () => {
    mount('/stats/live/x');
    const ol = within(desktop());
    expect(ol.getByRole('link', { name: 'Stats' })).toHaveAttribute('href', '/ui/stats');
    const live = ol.getByText('live');
    expect(live.tagName).toBe('SPAN');
    expect(live.closest('a')).toBeNull();
  });

  it('the current segment is never a link, in both renderings', () => {
    mount('/cache');
    const all = screen.getAllByText('Cache');
    expect(all.length).toBe(2); // desktop + mobile
    for (const el of all) {
      expect(el).toHaveAttribute('aria-current', 'page');
      expect(el.closest('a')).toBeNull();
    }
  });

  it('truncates long names and keeps the full one in the tooltip', () => {
    const long = 'x'.repeat(60);
    mount('/tokens/tok-1', <Title title={long} />);
    const el = within(desktop()).getByTitle(long);
    expect(el.textContent).toBe('x'.repeat(39) + '…');
    expect(truncate('abc', 3)).toBe('abc');
    expect(truncate('abcd', 3)).toBe('ab…');
    expect(truncate('y'.repeat(29), undefined)).toBe('y'.repeat(27) + '…');
  });

  it('mobile collapses the middle behind … and expands on tap', () => {
    mount('/models/abc/settings', <Title path="/models/abc" title="Qwen3-8B" />);
    const nav = screen.getByRole('navigation', { name: 'Breadcrumb' });
    const mobile = nav.querySelectorAll('ol')[1] as HTMLElement;
    expect(within(mobile).queryByText('Qwen3-8B')).toBeNull();
    expect(within(mobile).getByRole('link', { name: 'Home' })).toHaveAttribute('href', '/ui/models');
    fireEvent.click(within(mobile).getByRole('button', { name: 'Show hidden breadcrumb segments' }));
    expect(within(mobile).getByRole('link', { name: 'Qwen3-8B' })).toHaveAttribute('href', '/ui/models/abc');
  });

  it('uses only theme tokens — no hex colours in class names', () => {
    const { container } = mount('/tokens/tok-1', <Title title="ci-runner" />);
    for (const el of container.querySelectorAll('[class]')) {
      expect(el.getAttribute('class')).not.toMatch(/#[0-9a-fA-F]{3,8}\b|slate-/);
    }
  });
});

describe('BreadcrumbHeader — back button', () => {
  it('with no history: "← Home", a link to /ui/models (where `/` would redirect)', () => {
    mount('/stats');
    const back = screen.getByRole('link', { name: 'Back to Home' });
    expect(back).toHaveAttribute('href', '/ui/models');
    expect(back).toHaveTextContent('Home');
  });

  it('on a fresh /models there is no dead "← Home" and no Home link to the same page', () => {
    mount('/models');
    expect(screen.queryByRole('link', { name: 'Back to Home' })).toBeNull();
    expect(screen.queryByRole('button', { name: /^Back to/ })).toBeNull();
    const nav = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(within(nav).queryByRole('link')).toBeNull();
    // Still reads Home › Models, with Models the current page.
    expect(desktop()).toHaveTextContent(/^HomeModels$/);
    expect(within(desktop()).getByText('Models')).toHaveAttribute('aria-current', 'page');
    expect(within(desktop()).getByText('Home').closest('a')).toBeNull();
  });

  it('`/` itself (before its redirect lands) offers "← Home" to /models', () => {
    mount('/');
    expect(screen.getByRole('link', { name: 'Back to Home' })).toHaveAttribute('href', '/ui/models');
  });

  it('on /models with history, the back button is there and names the previous page', async () => {
    const { go } = mount('/stats');
    await go('/models');
    expect(screen.getByRole('button', { name: 'Back to Stats' })).toBeInTheDocument();
  });

  it('after navigating: "← <previous page title>", which goes back through the stack', async () => {
    const { go } = mount('/tokens/tok-1', <Title title="ci-runner" />);
    await go('/stats');
    const back = screen.getByRole('button', { name: 'Back to ci-runner' });
    expect(back).toHaveTextContent('ci-runner');
    fireEvent.click(back);
    expect(navState.push).toHaveBeenCalledWith('/tokens/tok-1', { scroll: false });
    // Popped: with nothing left the button falls back to Home.
    expect(screen.getByRole('link', { name: 'Back to Home' })).toBeInTheDocument();
  });

  it('truncates a long previous title', async () => {
    const long = 'a-very-long-model-served-name-for-testing';
    const { go } = mount('/models/abc', <Title title={long} />);
    await go('/stats');
    const back = screen.getByRole('button', { name: `Back to ${long}` });
    expect(back.textContent).toBe('a'.concat(long.slice(1, 23), '…'));
  });
});
