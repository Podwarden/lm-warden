// NavStackProvider (src/lib/nav-stack.tsx) + useBreadcrumb — ported from
// PodWarden Core's contexts/NavStackContext.test.tsx (#1187 AC-3) and extended
// for the LLM Warden adaptations: goBack (pop, restore scroll, fall back to
// Home), the no-shell routes, the stack cap and override clean-up.
import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render } from '@testing-library/react';

// next/navigation with a mutable pathname: a test "navigates" by writing it
// and re-rendering the provider.
const navState = vi.hoisted(() => ({ pathname: '/', push: vi.fn() }));
vi.mock('next/navigation', () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: navState.push, replace: vi.fn(), refresh: vi.fn() }),
}));

import { MAX_STACK, NavStackProvider, useNavStack, type NavStackContextValue } from '@/lib/nav-stack';
import { useBreadcrumb } from '@/lib/use-breadcrumb';

function harness() {
  const ref: { ctx: NavStackContextValue | null } = { ctx: null };
  function Probe() {
    ref.ctx = useNavStack();
    return null;
  }
  const tree = (extra?: React.ReactNode) => (
    <NavStackProvider>
      <Probe />
      {extra}
    </NavStackProvider>
  );
  return { ref, tree };
}

async function start(path: string, extra?: React.ReactNode) {
  const h = harness();
  navState.pathname = path;
  const r = render(h.tree(extra));
  const go = async (to: string, next?: React.ReactNode) => {
    await act(async () => {
      navState.pathname = to;
      r.rerender(h.tree(next ?? extra));
    });
  };
  return { ...h, go, rerender: r.rerender };
}

beforeEach(() => {
  vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
    cb(0);
    return 0;
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  navState.pathname = '/';
  navState.push.mockReset();
  window.scrollY = 0;
});

const stackOf = (c: NavStackContextValue | null) => c!.stack.map((e) => ({ href: e.href, title: e.title }));

describe('NavStackProvider — back-button label capture (PodWarden #1187 AC-3)', () => {
  it('A → B → C: the top of the stack on C is B, titled as B (not C)', async () => {
    const { ref, go } = await start('/models');
    expect(ref.ctx!.stack).toEqual([]); // a direct load pushes nothing
    expect(ref.ctx!.canGoBack).toBe(false);

    await go('/tokens');
    expect(ref.ctx!.topOfStack).toMatchObject({ href: '/models', title: 'Models' });

    await go('/stats');
    expect(ref.ctx!.topOfStack).toMatchObject({ href: '/tokens', title: 'API tokens' });
    expect(stackOf(ref.ctx)).toEqual([
      { href: '/models', title: 'Models' },
      { href: '/tokens', title: 'API tokens' },
    ]);
    expect(ref.ctx!.canGoBack).toBe(true);
  });

  it("keeps a page's useBreadcrumb title on the entry it leaves", async () => {
    function TokenPage() {
      useBreadcrumb({ title: 'ci-runner' });
      return null;
    }
    const { ref, go } = await start('/tokens/tok-1', <TokenPage />);
    expect(ref.ctx!.overrides['/tokens/tok-1']).toEqual({ title: 'ci-runner', parent: undefined });
    await go('/stats', <></>);
    expect(ref.ctx!.topOfStack).toMatchObject({ href: '/tokens/tok-1', title: 'ci-runner' });
  });

  it('an unknown path is remembered under its decoded last segment', async () => {
    const { ref, go } = await start('/stats/live%20view');
    await go('/models');
    expect(ref.ctx!.topOfStack!.title).toBe('live view');
  });

  it('collapses a duplicate consecutive entry into the fresher snapshot', async () => {
    const { ref, go } = await start('/models');
    await go('/tokens');
    await go('/models');
    await go('/tokens');
    // /models → /tokens → /models → /tokens pushes /models, /tokens, /models:
    // no two equal neighbours.
    expect(stackOf(ref.ctx).map((e) => e.href)).toEqual(['/models', '/tokens', '/models']);
  });

  it(`caps the stack at ${MAX_STACK} entries, dropping the oldest`, async () => {
    const { ref, go } = await start('/tokens/t0');
    for (let i = 1; i <= MAX_STACK + 5; i++) await go(`/tokens/t${i}`);
    expect(ref.ctx!.stack).toHaveLength(MAX_STACK);
    expect(ref.ctx!.stack[0].href).toBe('/tokens/t5');
    expect(ref.ctx!.topOfStack!.href).toBe(`/tokens/t${MAX_STACK + 4}`);
  });
});

describe('NavStackProvider — no-shell routes', () => {
  it('never pushes /login or /setup/*, and entering one clears the trail', async () => {
    const { ref, go } = await start('/models');
    await go('/tokens');
    expect(ref.ctx!.stack).toHaveLength(1);
    await go('/login'); // sign-out
    expect(ref.ctx!.stack).toEqual([]);
    await go('/models'); // signed back in: no "← login"
    expect(ref.ctx!.stack).toEqual([]);
    await go('/setup/gpus');
    await go('/setup/admin');
    await go('/models');
    expect(ref.ctx!.stack).toEqual([]);
  });
});

describe('NavStackProvider — goBack', () => {
  it('pops the top entry, navigates to it without Next scrolling, then restores its scroll', async () => {
    const { ref, go } = await start('/models');
    window.scrollY = 640;
    act(() => ref.ctx!.captureScroll());
    await go('/models/abc');
    expect(ref.ctx!.topOfStack).toMatchObject({ href: '/models', scrollY: 640 });

    await act(async () => ref.ctx!.goBack());
    expect(navState.push).toHaveBeenCalledWith('/models', { scroll: false });
    expect(ref.ctx!.stack).toEqual([]);

    await go('/models'); // the router lands
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 640, behavior: 'auto' });
    // Arriving by goBack pushes nothing — the back button now says Home.
    expect(ref.ctx!.stack).toEqual([]);

    // The restore slot is spent: the next forward navigation is an ordinary push.
    await go('/tokens');
    expect(stackOf(ref.ctx)).toEqual([{ href: '/models', title: 'Models' }]);
    expect(window.scrollTo).toHaveBeenCalledTimes(1);
  });

  it('with an empty stack, falls back to Home — /models, not the redirecting `/`', async () => {
    const { ref } = await start('/stats');
    await act(async () => ref.ctx!.goBack());
    expect(navState.push).toHaveBeenCalledWith('/models');
  });
});

describe('NavStackProvider — overrides', () => {
  it('keeps overrides for the current page and its ancestors, drops the rest', async () => {
    const { ref, go } = await start('/models/abc');
    act(() => {
      ref.ctx!.setBreadcrumbTitle('/models/abc', 'Qwen3-8B');
      ref.ctx!.setBreadcrumbTitle('/tokens/t1', 'stale');
    });
    await go('/models/abc/settings');
    expect(Object.keys(ref.ctx!.overrides)).toEqual(['/models/abc']);
    await go('/stats');
    expect(ref.ctx!.overrides).toEqual({});
  });

  it('setBreadcrumbTitle(path, null) removes an override; an identical set is a no-op', async () => {
    const { ref } = await start('/tokens/t1');
    act(() => ref.ctx!.setBreadcrumbTitle('/tokens/t1', 'ci'));
    const before = ref.ctx!.overrides;
    act(() => ref.ctx!.setBreadcrumbTitle('/tokens/t1', 'ci'));
    expect(ref.ctx!.overrides).toBe(before);
    act(() => ref.ctx!.setBreadcrumbTitle('/tokens/t1', null));
    expect(ref.ctx!.overrides).toEqual({});
  });
});

describe('useBreadcrumb', () => {
  it('waits while the title is undefined (loading), then registers it for the current path', async () => {
    let setTitle: (t: string) => void = () => {};
    function Page() {
      const [t, set] = React.useState<string | undefined>(undefined);
      setTitle = set;
      useBreadcrumb({ title: t });
      return null;
    }
    const { ref } = await start('/tokens/t1', <Page />);
    expect(ref.ctx!.overrides).toEqual({});
    await act(async () => setTitle('ci-runner'));
    expect(ref.ctx!.overrides).toEqual({ '/tokens/t1': { title: 'ci-runner', parent: undefined } });
  });

  it('`path` names another page — a child naming its dynamic parent', async () => {
    function SettingsPage() {
      useBreadcrumb({ path: '/models/abc', title: 'Qwen3-8B' });
      return null;
    }
    const { ref } = await start('/models/abc/settings', <SettingsPage />);
    expect(Object.keys(ref.ctx!.overrides)).toEqual(['/models/abc']);
  });

  it('is a no-op outside a provider (a page unit-tested on its own)', () => {
    function Page() {
      useBreadcrumb({ title: 'x' });
      return <p>ok</p>;
    }
    expect(() => render(<Page />)).not.toThrow();
  });

  it('useNavStack outside a provider throws', () => {
    function Bad() {
      useNavStack();
      return null;
    }
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => render(<Bad />)).toThrow(/NavStackProvider/);
  });
});
