// /models carries two tabs: the fleet ("Models") and the downloaded weights
// on disk ("Model files", formerly the top-level /cache page). The tab lives
// in ?tab= so /cache and the delete modal can deep-link to it.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

const nav = vi.hoisted(() => ({ replace: vi.fn(), search: '', sp: new URLSearchParams() }));
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => {
    if (nav.sp.toString() !== nav.search) nav.sp = new URLSearchParams(nav.search);
    return nav.sp;
  },
  usePathname: () => '/models',
}));

import ModelsPage from '@/app/models/page';

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

const calls: string[] = [];
function installFetch() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      if (url === '/api/auth/refresh') return json({ access_token: 'jwt' });
      if (url === '/api/csrf') return json({ csrf: 'csrf' });
      if (url === '/api/models') return json({ models: [] });
      if (url === '/api/cache/models') {
        return json([
          {
            repo: 'facebook/opt-125m',
            path: '/hf/hub/models--facebook--opt-125m',
            size_bytes: 250_000_000,
            last_modified: 1_793_000_000,
            matched_models: [],
          },
        ]);
      }
      return json({});
    }),
  );
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ModelsPage />
    </SWRConfig>,
  );
}

describe('/models tabs', () => {
  beforeEach(() => {
    setAccessToken('jwt', 900);
    setCsrfToken('csrf');
    nav.replace.mockReset();
    nav.search = '';
    calls.length = 0;
    installFetch();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows the fleet by default, with a Model files tab next to it', async () => {
    renderPage();
    expect(screen.getByRole('heading', { level: 1, name: 'Models' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Models' })).toHaveAttribute('aria-current', 'true');
    expect(screen.getByRole('button', { name: 'Model files' })).not.toHaveAttribute('aria-current');
    await waitFor(() => expect(screen.getByTestId('models-empty-state')).toBeInTheDocument());
    expect(screen.queryByTestId('cache-section')).toBeNull();
    expect(calls).not.toContain('/api/cache/models');
  });

  it('?tab=files renders the downloaded model files with their sizes', async () => {
    nav.search = 'tab=files';
    renderPage();
    expect(screen.getByRole('button', { name: 'Model files' })).toHaveAttribute('aria-current', 'true');
    expect(screen.getByTestId('cache-section')).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText('facebook/opt-125m').length).toBeGreaterThan(0));
    expect(screen.queryByTestId('models-empty-state')).toBeNull();
    // The fleet's Add-model controls belong to the Models tab only.
    expect(screen.queryByTestId('add-model-button-inline')).toBeNull();
  });

  it('clicking a tab writes it to the URL with replace (not a history step)', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Model files' }));
    expect(nav.replace).toHaveBeenCalledWith('/models?tab=files', { scroll: false });
    nav.search = 'tab=files';
    cleanup();
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Models' }));
    expect(nav.replace).toHaveBeenLastCalledWith('/models', { scroll: false });
  });

  it('an unknown ?tab= falls back to the fleet', () => {
    nav.search = 'tab=bogus';
    renderPage();
    expect(screen.getByRole('button', { name: 'Models' })).toHaveAttribute('aria-current', 'true');
  });
});
