/**
 * The API key tab of /login (and the old /sessions/login address, which forwards to it): failures map to their
 * message and never echo the key; blocked storage is reported. The happy path is in key-area.test.tsx.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { SWRConfig } from 'swr';

const mockReplace = vi.fn();
vi.mock('next/navigation', () => ({
  usePathname: () => '/login',
  useRouter: () => ({ replace: mockReplace }),
}));

import LoginPage from '@/app/login/page';
import SessionsLoginRedirect from '@/app/sessions/login/page';
import { K } from '@/lib/key-area/strings';

const KEY = 'vw_secretkey1234567890';

function stub(login: () => Response) {
  const f = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url.includes('/api/setup/state')) return new Response(JSON.stringify({ done: true, step: 'done' }));
    return login();
  });
  vi.stubGlobal('fetch', f);
  return f;
}

async function submitKey() {
  render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <LoginPage />
    </SWRConfig>,
  );
  fireEvent.change(await screen.findByLabelText(K.login.keyLabel), { target: { value: KEY } });
  fireEvent.click(screen.getByRole('button', { name: K.login.keySubmit }));
}

describe('API key sign-in', () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    mockReplace.mockReset();
    window.history.replaceState({}, '', '/login?key=1');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('the old /sessions/login address forwards to the API key tab', () => {
    render(<SessionsLoginRedirect />);
    expect(mockReplace).toHaveBeenCalledWith('/login?key=1');
  });

  it.each([401, 403, 429])('maps %i to its message and never echoes the key', async (status) => {
    stub(() => new Response(`{"detail":"${KEY}"}`, { status }));
    await submitKey();
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toBe(K.login.keyErrors[status]);
    expect(document.body.textContent).not.toContain(KEY);
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it('reports blocked storage', async () => {
    stub(() => new Response(JSON.stringify({ token: 'jwt-1', expires_in: 600 }), { status: 200 }));
    // jsdom's Storage methods cannot be spied on: swap the whole object for one that refuses writes.
    const real = window.sessionStorage;
    const blocked = { getItem: () => null, removeItem: () => undefined, setItem: () => { throw new Error('blocked'); } };
    Object.defineProperty(window, 'sessionStorage', { value: blocked, configurable: true });
    try {
      await submitKey();
      expect((await screen.findByRole('alert')).textContent).toBe(K.login.keyStorage);
      expect(mockReplace).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(window, 'sessionStorage', { value: real, configurable: true });
    }
  });
});
