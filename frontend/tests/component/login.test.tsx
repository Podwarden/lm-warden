import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import LoginPage from '@/app/login/page';

// Hoisted so each test can assert on the SAME router.replace spy that the
// mocked next/navigation hook returns.
const replaceMock = vi.hoisted(() => vi.fn());
vi.mock('next/navigation', () => ({ useRouter: () => ({ replace: replaceMock }) }));

// The login page now fetches /api/setup/state on mount (entry gate). A helper
// builds a fetch mock that answers the setup probe one way and the login POST
// another, dispatching on URL so a single global stub covers both calls.
function fetchMockFor(opts: {
  setup?: { status?: number; body?: unknown };
  login?: Response;
}) {
  const setupStatus = opts.setup?.status ?? 200;
  const setupBody = opts.setup?.body ?? { step: 'done', done: true };
  return vi.fn((url: string) => {
    if (typeof url === 'string' && url.includes('/api/setup/state')) {
      return Promise.resolve(
        new Response(JSON.stringify(setupBody), { status: setupStatus }),
      );
    }
    return Promise.resolve(
      opts.login ?? new Response('{"access_token":"abc","expires_in":900}', { status: 200 }),
    );
  });
}

describe('LoginPage', () => {
  afterEach(() => {
    // Explicit unmount: this project runs vitest with globals:false and
    // tests/setup.ts does not import @testing-library/react, so RTL's
    // auto-cleanup is NOT registered. Several tests here mount LoginPage,
    // so without this each leaves a stale DOM tree and getByRole matches
    // duplicate elements across tests.
    cleanup();
    vi.unstubAllGlobals();
    replaceMock.mockReset();
  });

  it('redirects a fresh install to the setup wizard', async () => {
    vi.stubGlobal('fetch', fetchMockFor({ setup: { body: { step: 'welcome', done: false } } }));
    render(<LoginPage />);
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith('/setup/welcome'));
    // Sign-in form must NOT render for a fresh install.
    expect(screen.queryByRole('button', { name: /log in/i })).not.toBeInTheDocument();
  });

  it('redirects to the CURRENT wizard step mid-setup (step past welcome, not done)', async () => {
    // Regression: hard-coding /setup/welcome here stranded half-finished
    // setups — the welcome POST was rejected ("not at welcome step") and the
    // welcome page offered no way forward. Funnel to the server's step.
    vi.stubGlobal('fetch', fetchMockFor({ setup: { body: { step: 'gpus', done: false } } }));
    render(<LoginPage />);
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith('/setup/gpus'));
    expect(screen.queryByRole('button', { name: /log in/i })).not.toBeInTheDocument();
  });

  it('funnels a malformed setup-state body to /setup/welcome', async () => {
    vi.stubGlobal('fetch', fetchMockFor({ setup: { body: { nonsense: 1 } } }));
    render(<LoginPage />);
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith('/setup/welcome'));
  });

  it('says how long to wait when the login is throttled (429 + Retry-After)', async () => {
    vi.stubGlobal(
      'fetch',
      fetchMockFor({
        login: new Response('{"detail":"too many failed login attempts; retry later"}', {
          status: 429,
          headers: { 'Retry-After': '60' },
        }),
      }),
    );
    render(<LoginPage />);
    const btn = await screen.findByRole('button', { name: /log in/i });
    fireEvent.click(btn);
    await waitFor(() =>
      expect(screen.getByText('Too many failed attempts. Try again in 60 seconds.')).toBeInTheDocument(),
    );
    expect(screen.queryByText('Invalid credentials')).not.toBeInTheDocument();
  });

  it('renders the sign-in form once setup is done', async () => {
    vi.stubGlobal('fetch', fetchMockFor({ setup: { body: { step: 'done', done: true } } }));
    render(<LoginPage />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /log in/i })).toBeInTheDocument(),
    );
    expect(replaceMock).not.toHaveBeenCalledWith('/setup/welcome');
  });

  it('falls back to the login form when the setup probe fails', async () => {
    vi.stubGlobal('fetch', fetchMockFor({ setup: { status: 503, body: {} } }));
    render(<LoginPage />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /log in/i })).toBeInTheDocument(),
    );
    expect(replaceMock).not.toHaveBeenCalledWith('/setup/welcome');
  });

  it('posts credentials and stores access token', async () => {
    const fetchMock = fetchMockFor({ setup: { body: { step: 'done', done: true } } });
    vi.stubGlobal('fetch', fetchMock);
    render(<LoginPage />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /log in/i })).toBeInTheDocument(),
    );
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'admin' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'pw' } });
    fireEvent.click(screen.getByRole('button', { name: /log in/i }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/auth/login',
        expect.objectContaining({ method: 'POST', credentials: 'include' }),
      ),
    );
    // Successful login still navigates to /models (preserved behavior).
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith('/stats'));
  });
});
