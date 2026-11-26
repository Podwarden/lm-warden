/**
 * The unified login (Login | API key) and the key holder's area: overview, settings (rename), manage (rotate,
 * revoke) and sign out. The key is sent once and never stored; only the key holder's sign-in token is kept.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

let mockPath = '/login';
const mockReplace = vi.fn();
vi.mock('next/navigation', () => ({
  usePathname: () => mockPath,
  useRouter: () => ({ replace: mockReplace, push: mockReplace }),
}));

import LoginPage from '@/app/login/page';
import KeyOverviewPage from '@/app/key/page';
import KeyManagePage from '@/app/key/manage/page';
import KeySettingsPage from '@/app/key/settings/page';
import { K } from '@/lib/key-area/strings';

const KEY = 'vw_secretkey1234567890';
const DETAIL = {
  id: 'tok1', name: 'farm-07', prefix: 'vw_k2x7a', created_at: '2026-09-12 10:00:00', expires_at: null,
  is_revoked: false, is_expired: false, is_paused: false, allowed_models: null, priority: 5,
  thinking_policy: 'client', anthropic_relay: false, capture_until: null, prompt_diet: null,
  guard_effective: { mode: 'sanitize', packs: ['cloud-api-keys'], rules: 2, source: { mode: 'platform' } },
};

function signedIn() {
  sessionStorage.setItem('vw-sessions-token', 'jwt-1');
  sessionStorage.setItem('vw-sessions-exp', String(Date.now() + 600_000));
}

function stub(routes: Record<string, (init?: RequestInit) => Response>) {
  const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const hit = Object.keys(routes).find((k) => url.includes(k));
    if (!hit) return new Response('{}', { status: 404 });
    return routes[hit](init);
  });
  vi.stubGlobal('fetch', f);
  return f;
}

const json = (b: unknown, status = 200) => () => new Response(JSON.stringify(b), { status });

function wrap(el: React.ReactElement) {
  return render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{el}</SWRConfig>);
}

describe('unified login', () => {
  beforeEach(() => {
    mockPath = '/login';
    window.history.replaceState({}, '', '/login');
    sessionStorage.clear();
    mockReplace.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('offers Login and API key; the key goes once as Bearer and lands on the key area', async () => {
    const f = stub({
      '/api/setup/state': json({ done: true, step: 'done' }),
      '/api/forest/login': json({ token: 'jwt-1', expires_in: 600 }),
    });
    wrap(<LoginPage />);
    const tabs = await screen.findByRole('tablist');
    expect(within(tabs).getByRole('tab', { name: K.login.tabLogin })).toBeTruthy();
    fireEvent.click(within(tabs).getByRole('tab', { name: K.login.tabKey }));
    fireEvent.change(screen.getByLabelText(K.login.keyLabel), { target: { value: KEY } });
    fireEvent.click(screen.getByRole('button', { name: K.login.keySubmit }));
    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/key'));
    const call = f.mock.calls.find((c) => String(c[0]).includes('/api/forest/login'))!;
    expect((call[1]!.headers as Record<string, string>).Authorization).toBe(`Bearer ${KEY}`);
    expect(sessionStorage.getItem('vw-sessions-token')).toBe('jwt-1');
    expect(JSON.stringify({ ...sessionStorage }) + JSON.stringify({ ...localStorage })).not.toContain(KEY);
  });

  it('opens on the API key tab with ?key=1', async () => {
    window.history.replaceState({}, '', '/login?key=1');
    stub({ '/api/setup/state': json({ done: true, step: 'done' }) });
    wrap(<LoginPage />);
    expect(await screen.findByLabelText(K.login.keyLabel)).toBeTruthy();
  });
});

describe('key area', () => {
  beforeEach(() => {
    sessionStorage.clear();
    mockReplace.mockReset();
    signedIn();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('overview shows the key and its usage', async () => {
    mockPath = '/key';
    stub({
      '/api/me/key': json(DETAIL),
      '/api/me/usage': json({ buckets: [], totals: { requests: 2481, prompt_tokens: 186_000_000, completion_tokens: 1_900_000, total_tokens: 0 } }),
      '/api/me/cache': json({ models: [] }),
    });
    wrap(<KeyOverviewPage />);
    expect(await screen.findByText('farm-07')).toBeTruthy();
    expect(await screen.findByText('2,481')).toBeTruthy();
  });

  it('settings: admin settings read-only, the name editable', async () => {
    mockPath = '/key/settings';
    const f = stub({ '/api/me/key': (init) => new Response(JSON.stringify(init?.method === 'PATCH' ? { ...DETAIL, name: 'mine' } : DETAIL)) });
    wrap(<KeySettingsPage />);
    const name = (await screen.findByLabelText(K.settings.nameLabel)) as HTMLInputElement;
    expect(screen.getAllByText(K.settings.byAdmin).length).toBeGreaterThan(2);
    fireEvent.change(name, { target: { value: 'mine' } });
    fireEvent.click(screen.getByRole('button', { name: K.settings.save }));
    await waitFor(() => expect(f.mock.calls.some((c) => c[1]?.method === 'PATCH')).toBe(true));
    const patch = f.mock.calls.find((c) => c[1]?.method === 'PATCH')!;
    expect(JSON.parse(String(patch[1]!.body))).toEqual({ name: 'mine' });
  });

  it('settings: the key holder sees both permissions read-only, with a reason when Prompt Lens is off', async () => {
    mockPath = '/key/settings';
    stub({ '/api/me/key': json({ ...DETAIL, prompt_lens: false, engine_facts: true, prompt_lens_view_available: false }) });
    wrap(<KeySettingsPage />);
    expect(await screen.findByText(K.settings.promptLensOff)).toBeTruthy();
    expect(screen.getByText(K.settings.engineFactsOn)).toBeTruthy();
    // No control for either: the admin sets them.
    expect(screen.queryByRole('switch')).toBeNull();
    cleanup();
    stub({ '/api/me/key': json({ ...DETAIL, prompt_lens: true, engine_facts: false, prompt_lens_view_available: true }) });
    wrap(<KeySettingsPage />);
    expect(await screen.findByText(K.settings.promptLensOn)).toBeTruthy();
    expect(screen.getByText(K.settings.engineFactsOff)).toBeTruthy();
  });

  it('rotate shows the new key once and switches the sign-in to it', async () => {
    mockPath = '/key/manage';
    const f = stub({
      '/api/me/key/rotate': json({ id: 'tok2', plaintext: 'vw_newkey', token: 'jwt-2', expires_in: 600 }, 201),
      '/api/me/key': json(DETAIL),
    });
    wrap(<KeyManagePage />);
    fireEvent.change(await screen.findByLabelText(K.manage.graceLabel), { target: { value: '1' } });
    fireEvent.click(screen.getByRole('button', { name: K.manage.rotate }));
    expect(await screen.findByText('vw_newkey')).toBeTruthy();
    const call = f.mock.calls.find((c) => String(c[0]).includes('/rotate'))!;
    expect(JSON.parse(String(call[1]!.body))).toEqual({ grace_hours: 1 });
    expect(sessionStorage.getItem('vw-sessions-token')).toBe('jwt-2');
  });

  it('revoke needs the name, then signs out', async () => {
    mockPath = '/key/manage';
    const f = stub({ '/api/me/key/revoke': () => new Response(null, { status: 204 }), '/api/me/key': json(DETAIL) });
    wrap(<KeyManagePage />);
    const btn = (await screen.findByRole('button', { name: K.manage.revoke })) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText(K.manage.confirmLabel), { target: { value: 'farm-07' } });
    expect(btn.disabled).toBe(false);
    fireEvent.click(btn);
    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/login?key=1'));
    expect(f.mock.calls.some((c) => String(c[0]).includes('/revoke'))).toBe(true);
    expect(sessionStorage.getItem('vw-sessions-token')).toBeNull();
  });

  it('sign out ends the sign-in on the server and clears the token', async () => {
    mockPath = '/key';
    const f = stub({
      '/api/forest/logout': () => new Response(null, { status: 204 }),
      '/api/me/key': json(DETAIL),
      '/api/me/usage': json({ buckets: [], totals: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } }),
      '/api/me/cache': json({ models: [] }),
    });
    wrap(<KeyOverviewPage />);
    fireEvent.click(await screen.findByRole('button', { name: K.shell.signOut }));
    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith('/login?key=1'));
    const call = f.mock.calls.find((c) => String(c[0]).includes('/api/forest/logout'))!;
    expect((call[1]!.headers as Record<string, string>).Authorization).toBe('Bearer jwt-1');
    expect(sessionStorage.getItem('vw-sessions-token')).toBeNull();
  });
});
