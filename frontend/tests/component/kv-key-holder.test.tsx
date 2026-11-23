/**
 * The key holder's cache controls on /key/settings (plan 2, task 14): a shared key can
 * keep its cache separate (one way: no way back to sharing), the warning shows before
 * anything is saved, reset asks first, and the usage card never shows who dropped the
 * key's parts.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

vi.mock('next/navigation', () => ({
  usePathname: () => '/key/settings',
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));

import KeySettingsPage from '@/app/key/settings/page';
import { KEY_REPORT, KEY_STORE } from './kv-fixtures';

const DETAIL = {
  id: 'tok1', name: 'dana-laptop', prefix: 'vw_k2x7a', created_at: '2026-09-12 10:00:00', expires_at: null,
  is_revoked: false, is_expired: false, is_paused: false, allowed_models: null, priority: 5,
  thinking_policy: 'client', anthropic_relay: true, kv_isolation: 'sharing',
  guard_effective: { mode: 'off' },
  models: [
    { id: 'qwen3-235b-a22b', cache_isolation: 'isolated' },
    { id: 'gemma-3-27b', cache_isolation: 'unsupported' },
  ],
};

type Route = (init?: RequestInit) => Response;

function stub(routes: Record<string, Route>) {
  const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const path = url.split('?')[0];
    const h = routes[`${(init?.method ?? 'GET').toUpperCase()} ${path}`];
    return h ? h(init) : new Response('{}', { status: 404 });
  });
  vi.stubGlobal('fetch', f);
  return f;
}

const json = (b: unknown, status = 200) => () => new Response(JSON.stringify(b), { status });
const calls = (f: ReturnType<typeof vi.fn>, method: string, path: string) =>
  f.mock.calls.filter(([u, i]) => String(u).split('?')[0] === path && ((i as RequestInit | undefined)?.method ?? 'GET') === method);

function page() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <KeySettingsPage />
    </SWRConfig>,
  );
}

beforeEach(() => {
  sessionStorage.clear();
  sessionStorage.setItem('vw-sessions-token', 'jwt-1');
  sessionStorage.setItem('vw-sessions-exp', String(Date.now() + 600_000));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('key holder: prompt cache', () => {
  it('an isolated key has no way back to sharing', async () => {
    stub({
      'GET /api/me/key': json({ ...DETAIL, kv_isolation: 'isolated' }),
      'GET /api/me/key/cache': json(KEY_REPORT),
    });
    page();
    expect(await screen.findByText('isolated, only this key uses its cache')).toBeInTheDocument();
    expect(screen.getByText(/Only an admin can switch this key back to shared/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /separate|isolate|shar/i })).toBeNull();
    expect(screen.queryByRole('radio')).toBeNull();
  });

  it('a shared key: the warning shows before anything is saved', async () => {
    const f = stub({
      'GET /api/me/key': json(DETAIL),
      'GET /api/me/key/cache': json({ ...KEY_REPORT, isolation: 'sharing' }),
      'PATCH /api/me/key': json({ ...DETAIL, kv_isolation: 'isolated' }),
    });
    page();
    fireEvent.click(await screen.findByRole('button', { name: 'Keep my cache separate' }));
    const warn = screen.getByTestId('kv-holder-warning');
    expect(within(warn).getByText("Keep this key's cache separate?")).toBeInTheDocument();
    expect(within(warn).getByText("This key's next turn computes its prompt again.")).toBeInTheDocument();
    expect(within(warn).getByText('You cannot switch back yourself. Only an admin can.')).toBeInTheDocument();
    expect(within(warn).getByText(/gemma-3-27b cannot keep a separate cache, so it will refuse this key/)).toBeInTheDocument();
    expect(calls(f, 'PATCH', '/api/me/key')).toHaveLength(0);
    fireEvent.click(within(warn).getByRole('button', { name: 'Keep it separate' }));
    await waitFor(() => expect(calls(f, 'PATCH', '/api/me/key')).toHaveLength(1));
    expect(JSON.parse(calls(f, 'PATCH', '/api/me/key')[0][1]!.body as string)).toEqual({ kv_isolation: 'isolated' });
  });

  it('cancel saves nothing', async () => {
    const f = stub({ 'GET /api/me/key': json(DETAIL), 'GET /api/me/key/cache': json(KEY_REPORT) });
    page();
    fireEvent.click(await screen.findByRole('button', { name: 'Keep my cache separate' }));
    fireEvent.click(within(screen.getByTestId('kv-holder-warning')).getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByTestId('kv-holder-warning')).toBeNull();
    expect(calls(f, 'PATCH', '/api/me/key')).toHaveLength(0);
  });

  it('reset asks for confirmation', async () => {
    const f = stub({
      'GET /api/me/key': json({ ...DETAIL, kv_isolation: 'isolated' }),
      'GET /api/me/key/cache': json(KEY_REPORT),
      'POST /api/me/key/kv-reset': json({ ...DETAIL, kv_isolation: 'isolated' }),
    });
    page();
    fireEvent.click(await screen.findByRole('button', { name: 'Reset' }));
    const dlg = await screen.findByRole('dialog');
    expect(within(dlg).getByText('Reset your cache?')).toBeInTheDocument();
    expect(calls(f, 'POST', '/api/me/key/kv-reset')).toHaveLength(0);
    fireEvent.click(within(dlg).getByRole('button', { name: 'Reset cache' }));
    await waitFor(() => expect(calls(f, 'POST', '/api/me/key/kv-reset')).toHaveLength(1));
  });

  it('a shared key has no reset', async () => {
    stub({ 'GET /api/me/key': json(DETAIL), 'GET /api/me/key/cache': json(KEY_REPORT) });
    page();
    expect(await screen.findByText('Your key shares its cache, so there is nothing of its own to reset.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
  });

  it("the usage card never renders who dropped the key's parts", async () => {
    stub({
      'GET /api/me/key': json({ ...DETAIL, kv_isolation: 'isolated' }),
      // Even a server that wrongly sent evicted_by must not get it on screen.
      'GET /api/me/key/cache': json({
        ...KEY_REPORT,
        store: { ...KEY_STORE, evicted_by: [{ token_id: 't9', name: 'nightly-evals', count: 301 }] },
      }),
    });
    page();
    const card = await screen.findByTestId('kv-holder-usage');
    await within(card).findByText('3.4 GiB');
    expect(within(card).getByText('1,208')).toBeInTheDocument(); // own total only
    expect(screen.queryByText(/nightly-evals/)).toBeNull();
    expect(screen.queryByText(/dropped by/i)).toBeNull();
    expect(screen.queryByText('301')).toBeNull();
    expect(within(card).getByText(/since the KV cache store started/)).toBeInTheDocument();
  });

  it('a store that does not answer says so; the windows stay', async () => {
    stub({
      'GET /api/me/key': json({ ...DETAIL, kv_isolation: 'isolated' }),
      'GET /api/me/key/cache': json({ ...KEY_REPORT, store: null }),
    });
    page();
    expect(await screen.findByText(/The KV cache store is not answering, so its figures are unavailable/)).toBeInTheDocument();
    expect(screen.getByTestId('kv-hit-headline')).toHaveTextContent('64 %');
  });

  it("an isolated key's models list marks the models that cannot isolate", async () => {
    stub({ 'GET /api/me/key': json({ ...DETAIL, kv_isolation: 'isolated' }), 'GET /api/me/key/cache': json(KEY_REPORT) });
    page();
    const row = (await screen.findByText('gemma-3-27b')).closest('li')!;
    expect(row).toHaveTextContent("cannot keep this key's cache separate: refused");
    expect(screen.getByText('qwen3-235b-a22b').closest('li')).not.toHaveTextContent('refused');
    cleanup();
    stub({ 'GET /api/me/key': json(DETAIL), 'GET /api/me/key/cache': json(KEY_REPORT) });
    page();
    await screen.findByText('gemma-3-27b');
    expect(screen.queryByText(/cannot keep this key's cache separate/)).toBeNull();
  });
});
