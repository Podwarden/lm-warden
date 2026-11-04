/**
 * SessionGate and the key holder's sessions view: a valid key token opens /sessions (and its detail page) with no
 * admin check; it never opens an admin page; without one, sessions paths go to the API key sign-in, never /login.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

let mockPath = '/sessions';
const mockRouterReplace = vi.fn();
vi.mock('next/navigation', () => ({
  usePathname: () => mockPath,
  useRouter: () => ({ replace: mockRouterReplace }),
}));

import { SessionGate } from '@/components/session-gate';
import { setAccessToken, __resetLoginRedirectInFlightForTests } from '@/lib/auth-fetch';

function stubWindowLocation(pathname: string, search = ''): ReturnType<typeof vi.fn> {
  const replace = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { replace, pathname, search, origin: 'http://localhost' },
    writable: true,
    configurable: true,
  });
  return replace;
}

function stubRefresh(status: number, body = '{}') {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url.includes('/api/auth/refresh')) return new Response(body, { status });
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function holdToken(ms = 60_000) {
  sessionStorage.setItem('vw-sessions-token', 'jwt');
  sessionStorage.setItem('vw-sessions-exp', String(Date.now() + ms));
}

const CHILD = <p data-testid="page-body">page</p>;

describe('SessionGate — sessions key mode', () => {
  beforeEach(() => {
    mockPath = '/sessions';
    mockRouterReplace.mockReset();
    sessionStorage.clear();
    setAccessToken(null);
    __resetLoginRedirectInFlightForTests();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it.each(['/sessions/login', '/sessions/login/'])('%s is public', (p) => {
    mockPath = p;
    render(<SessionGate>{CHILD}</SessionGate>);
    expect(screen.getByTestId('page-body')).toBeTruthy();
  });

  it.each(['/sessions', '/sessions/detail'])('opens %s in key mode with a valid token and makes no session check', async (p) => {
    mockPath = p;
    holdToken();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(screen.getByTestId('page-body')).toBeTruthy());
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('sends an expired token to the API key sign-in', async () => {
    holdToken(-1);
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalledWith('/login?key=1'));
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.queryByTestId('page-body')).toBeNull();
  });

  it('sends ?key=1 with no token to the API key sign-in', async () => {
    stubWindowLocation('/sessions', '?key=1');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalledWith('/login?key=1'));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('sends /sessions to the API key sign-in (not /login) with neither token nor session', async () => {
    const replace = stubWindowLocation('/sessions');
    stubRefresh(401, 'Unauthorized');
    render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalledWith('/login?key=1'));
    expect(replace).not.toHaveBeenCalled();
    expect(screen.queryByTestId('page-body')).toBeNull();
  });

  it.each(['/key', '/key/settings', '/key/manage', '/key/guard-log'])(
    'opens %s with a valid token and makes no session check',
    async (p) => {
      mockPath = p;
      holdToken();
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);
      render(<SessionGate>{CHILD}</SessionGate>);
      await waitFor(() => expect(screen.getByTestId('page-body')).toBeTruthy());
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it('never opens the key area on an admin session: no token goes to the API key sign-in', async () => {
    mockPath = '/key';
    setAccessToken('admin-jwt');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalledWith('/login?key=1'));
    expect(screen.queryByTestId('page-body')).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('never opens an admin page with a key token', async () => {
    holdToken();
    const replace = stubWindowLocation('/sessions');
    const fetchMock = stubRefresh(401, 'Unauthorized');
    const view = render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(screen.getByTestId('page-body')).toBeTruthy());
    expect(fetchMock).not.toHaveBeenCalled();
    mockPath = '/stats';
    view.rerender(<SessionGate>{CHILD}</SessionGate>);
    expect(screen.queryByTestId('page-body')).toBeNull();
    await waitFor(() => expect(replace).toHaveBeenCalledWith('/login'));
    expect(screen.queryByTestId('page-body')).toBeNull();
  });
});
