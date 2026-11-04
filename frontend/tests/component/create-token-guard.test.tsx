/**
 * Create API token → Sensitive data: the new key's guard is chosen in the same dialog. Default is
 * the platform setting (sends no guard); "Off for this key" and "Own settings" send what PATCH
 * would; a refused setting shows the server's reason and creates nothing.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { G } from '@/lib/guard/strings';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

const PACKS = {
  default: ['cloud-api-keys', 'private-keys'],
  packs: [
    { id: 'cloud-api-keys', label: 'Cloud and API keys', default: true, rules: [{ id: 'aws-access-key', label: 'AWS access key', public_prefix: 'AKIA' }] },
    { id: 'private-keys', label: 'Private keys', default: true, rules: [{ id: 'pem-private-key', label: 'PEM private key', public_prefix: null }] },
  ],
};
const PLATFORM = { mode: 'sanitize', packs: ['cloud-api-keys'], rules: [], overrides: {}, restore: 'all', trailer: 'new_only', fail: 'closed' };
const CREATED = { id: 'x', name: 'n', plaintext: 'vw_x', prefix: 'vw_', preview: 'vw_', expires_at: null };

function stub(platform: unknown, create: (body: unknown) => Response = () => new Response(JSON.stringify(CREATED), { status: 201 })) {
  const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (init?.method === 'POST' && url.includes('/api/tokens')) return create(JSON.parse(String(init.body)));
    if (url.includes('/api/guard/packs')) return new Response(JSON.stringify(PACKS));
    if (url.includes('/api/guard/settings')) return new Response(JSON.stringify({ guard: platform }));
    return new Response('{}', { status: 404 });
  });
  vi.stubGlobal('fetch', f);
  return f;
}

function sent(f: ReturnType<typeof stub>) {
  const call = f.mock.calls.find((c) => c[1]?.method === 'POST')!;
  return JSON.parse(String(call[1]!.body));
}

function open() {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <CreateTokenDialog open onClose={() => {}} />
    </SWRConfig>,
  );
  fireEvent.change(screen.getByPlaceholderText('ci-bot'), { target: { value: 'agent-1' } });
}

beforeEach(() => {
  setAccessToken('jwt');
  setCsrfToken('csrf');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('create dialog: sensitive data', () => {
  it('defaults to the platform setting, shows what that is, and sends no guard', async () => {
    const f = stub(PLATFORM);
    open();
    expect(((await screen.findByRole('radio', { name: G.key.inherit })) as HTMLInputElement).checked).toBe(true);
    expect(await screen.findByText('Sanitize · Cloud and API keys')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toBeTruthy());
    expect('guard' in sent(f)).toBe(false);
  });

  it('says so when the platform has no guard', async () => {
    stub(null);
    open();
    expect(await screen.findByText(G.key.platformOff)).toBeTruthy();
  });

  it('off for this key sends {"off": true}', async () => {
    const f = stub(PLATFORM);
    open();
    fireEvent.click(await screen.findByRole('radio', { name: G.key.off }));
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toBeTruthy());
    expect(sent(f).guard).toEqual({ off: true });
  });

  it('own settings start from the platform and send an override', async () => {
    const f = stub(PLATFORM);
    open();
    fireEvent.click(await screen.findByRole('radio', { name: G.key.own }));
    fireEvent.click(screen.getByRole('radio', { name: G.mode.block }));
    fireEvent.click(screen.getByRole('checkbox', { name: /Private keys/ }));
    fireEvent.change(screen.getByLabelText(G.packs.actionLabel('Cloud and API keys')), { target: { value: 'sanitize' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toBeTruthy());
    const g = sent(f).guard.override;
    expect(g.mode).toBe('block');
    expect(g.packs).toEqual(['cloud-api-keys', 'private-keys']);
    expect(g.overrides).toEqual({ 'cloud-api-keys': 'sanitize' });
  });

  it('a refused setting shows the reason and creates no key', async () => {
    stub(PLATFORM, () => new Response(JSON.stringify({ detail: 'nested repetition can take exponential time' }), { status: 422 }));
    open();
    fireEvent.click(await screen.findByRole('radio', { name: G.key.own }));
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(await screen.findByText(/exponential time/)).toBeTruthy();
    expect(screen.queryByTestId('new-token')).toBeNull();
  });
});
