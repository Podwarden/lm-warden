/**
 * SessionGate — the app shell waits until the session is known.
 *
 * The defect: opening /ui/ while signed out rendered the Models page —
 * heading and skeleton cards — for one round trip before bouncing to
 * /ui/login. A first-time visitor reads that as a broken product, not as
 * "you are signed out".
 *
 * What must hold:
 *   - signed out  → nothing behind the gate ever paints; we go to /login
 *   - signed in   → the page paints, once, with no flash of anything else
 *   - backend down→ the page paints anyway; a bounce must not wedge the UI
 *   - /login and the setup wizard are never gated (they are the pages you
 *     reach WITHOUT a session; gating them would deadlock first run)
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

let mockPath = '/models';
vi.mock('next/navigation', () => ({
  usePathname: () => mockPath,
}));

import { SessionGate } from '@/components/session-gate';
import {
  setAccessToken,
  __resetLoginRedirectInFlightForTests,
} from '@/lib/auth-fetch';

function stubWindowLocation(pathname: string): ReturnType<typeof vi.fn> {
  const replace = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { replace, pathname, origin: 'http://localhost' },
    writable: true,
    configurable: true,
  });
  return replace;
}

/** `POST /api/auth/refresh` answers with `status`; nothing else is called. */
function stubRefresh(status: number, body = '{}') {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url.includes('/api/auth/refresh')) {
      return new Response(body, { status });
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const CHILD = <p data-testid="page-body">models page</p>;

describe('SessionGate', () => {
  beforeEach(() => {
    mockPath = '/models';
    sessionStorage.clear();
    setAccessToken(null);
    __resetLoginRedirectInFlightForTests();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders nothing from the page while the session is unknown', () => {
    stubRefresh(200, JSON.stringify({ access_token: 't', expires_in: 900 }));
    render(<SessionGate>{CHILD}</SessionGate>);
    // The synchronous first paint — the exact moment the Models page used
    // to appear to a logged-out visitor.
    expect(screen.queryByTestId('page-body')).toBeNull();
    expect(screen.getByTestId('session-gate-pending')).toBeTruthy();
  });

  it('opens once the refresh succeeds', async () => {
    stubRefresh(200, JSON.stringify({ access_token: 't', expires_in: 900 }));
    render(<SessionGate>{CHILD}</SessionGate>);
    await waitFor(() => expect(screen.getByTestId('page-body')).toBeTruthy());
    expect(screen.queryByTestId('session-gate-pending')).toBeNull();
  });

  it('redirects to /login and never opens when the session is dead', async () => {
    const replace = stubWindowLocation('/models');
    stubRefresh(401, 'Unauthorized');
    render(<SessionGate>{CHILD}</SessionGate>);

    await waitFor(() => expect(replace).toHaveBeenCalledWith('/login'));
    // The crux of the defect: the page body must not have rendered at any
    // point, not merely be gone by now.
    expect(screen.queryByTestId('page-body')).toBeNull();
    expect(screen.getByTestId('session-gate-pending')).toBeTruthy();
  });

  it('opens anyway on a transient backend failure', async () => {
    // 5xx means the refresh could not be EVALUATED — the cookie was never
    // rejected. Holding the whole UI behind a spinner during a backend
    // bounce would be a worse failure than the one being fixed.
    const replace = stubWindowLocation('/models');
    stubRefresh(503, 'bad gateway');
    render(<SessionGate>{CHILD}</SessionGate>);

    await waitFor(() => expect(screen.getByTestId('page-body')).toBeTruthy());
    expect(replace).not.toHaveBeenCalled();
  });

  it('does not gate a client-side navigation (token already in memory)', () => {
    setAccessToken('already-signed-in');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    render(<SessionGate>{CHILD}</SessionGate>);
    expect(screen.getByTestId('page-body')).toBeTruthy();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(['/login', '/setup', '/setup/welcome'])(
    'renders %s immediately without any session check',
    (path) => {
      mockPath = path;
      const fetchMock = vi.fn();
      vi.stubGlobal('fetch', fetchMock);
      render(<SessionGate>{CHILD}</SessionGate>);
      expect(screen.getByTestId('page-body')).toBeTruthy();
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it('does not treat /login-help as an unauthenticated route', () => {
    // Mirrors the #39 tightening in nav-bar.tsx: exact match, not
    // startsWith, so a sibling page keeps normal bounce-to-login behaviour.
    mockPath = '/login-help';
    stubRefresh(200, JSON.stringify({ access_token: 't', expires_in: 900 }));
    render(<SessionGate>{CHILD}</SessionGate>);
    expect(screen.queryByTestId('page-body')).toBeNull();
  });
});
