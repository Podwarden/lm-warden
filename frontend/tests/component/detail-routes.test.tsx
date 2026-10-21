// The query-id detail routes (BACKLOG-macos P4). `LMW_UI_EXPORT=1 next build`
// cannot pre-render /models/[id] for runtime ids, so the pages are fixed
// paths that read `?id=`: they render the same page bodies as before, keyed
// by the id, and send an id-less URL back to the list.
import fs from 'node:fs';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ReactElement } from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';

const nav = vi.hoisted(() => ({ search: '', replace: vi.fn(), mounts: [] as string[] }));
vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => ({ push: vi.fn(), replace: nav.replace, refresh: vi.fn() }),
}));
// The bodies have their own suites; here only the id hand-off matters.
vi.mock('@/components/models/model-detail-page', async () => {
  const { useEffect } = await import('react');
  return {
    ModelDetailPage: ({ id }: { id: string }) => {
      // Records each mount, so a remount on an id change is observable.
      useEffect(() => {
        nav.mounts.push(id);
      }, []); // eslint-disable-line react-hooks/exhaustive-deps
      return <p>model detail {id}</p>;
    },
  };
});
vi.mock('@/components/models/model-settings-page', () => ({
  ModelSettingsPage: ({ id }: { id: string }) => <p>model settings {id}</p>,
}));
vi.mock('@/components/tokens/token-detail-page', () => ({
  TokenDetailPage: ({ id }: { id: string }) => <p>token detail {id}</p>,
}));

import ModelDetailRoute from '@/app/models/detail/page';
import ModelSettingsRoute from '@/app/models/detail/settings/page';
import TokenDetailRoute from '@/app/tokens/detail/page';

afterEach(() => {
  cleanup();
  nav.search = '';
  nav.replace.mockReset();
  nav.mounts.length = 0;
});

const ROUTES = [
  ['/models/detail', ModelDetailRoute, 'model detail', '/models'],
  ['/models/detail/settings', ModelSettingsRoute, 'model settings', '/models'],
  ['/tokens/detail', TokenDetailRoute, 'token detail', '/tokens'],
] as const;

describe.each(ROUTES)('%s', (_path, Route, body, list) => {
  it('renders the page body for ?id=', async () => {
    nav.search = 'id=a%2Fb%20c&range=7d';
    await act(async () => {
      render(<Route />);
    });
    expect(screen.getByText(`${body} a/b c`)).toBeInTheDocument();
    expect(nav.replace).not.toHaveBeenCalled();
  });

  it(`sends an id-less URL to ${list}`, async () => {
    let container!: HTMLElement;
    await act(async () => {
      ({ container } = render(<Route />));
    });
    expect(container).toBeEmptyDOMElement();
    expect(nav.replace).toHaveBeenCalledWith(list);
  });
});

describe('id change', () => {
  // The old /models/[id] segment remounted the page body when the id
  // changed (fresh state per model); the gate keys the body by id to match.
  it('remounts the page body when ?id= changes', async () => {
    nav.search = 'id=a';
    let rerender!: (ui: ReactElement) => void;
    await act(async () => {
      ({ rerender } = render(<ModelDetailRoute />));
    });
    nav.search = 'id=b';
    await act(async () => {
      rerender(<ModelDetailRoute />);
    });
    expect(screen.getByText('model detail b')).toBeInTheDocument();
    expect(nav.mounts).toEqual(['a', 'b']);
  });
});

describe('static-export shape', () => {
  // A dynamic segment without generateStaticParams fails `output: 'export'`;
  // ids are runtime data, so no page may hide behind one.
  it('no app route has a dynamic [segment]', () => {
    const app = path.resolve(__dirname, '../../src/app');
    const dynamic: string[] = [];
    const walk = (dir: string) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        if (!e.isDirectory()) continue;
        const p = path.join(dir, e.name);
        if (/^\[.*\]$/.test(e.name)) dynamic.push(path.relative(app, p));
        walk(p);
      }
    };
    walk(app);
    expect(dynamic).toEqual([]);
  });
});
