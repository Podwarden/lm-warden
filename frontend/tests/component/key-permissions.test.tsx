/**
 * Per-key permissions (owner decision 2026-11-26): the create dialog sends both, a new key starts
 * without Prompt Lens and with engine facts; the key page's card saves each on its own; the key
 * holder sees the state read-only and is told why Prompt Lens is off.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { PermissionsCard } from '@/components/tokens/detail/permissions-card';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

const CREATED = { id: 'x', name: 'n', plaintext: 'vw_x', prefix: 'vw_', preview: 'vw_', expires_at: null };

function stub() {
  const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (init?.method === 'POST' && url.includes('/api/tokens')) return new Response(JSON.stringify(CREATED), { status: 201 });
    if (url.includes('/api/guard/packs')) return new Response(JSON.stringify({ default: [], packs: [] }));
    if (url.includes('/api/guard/settings')) return new Response(JSON.stringify({ guard: null }));
    return new Response('{}', { status: 404 });
  });
  vi.stubGlobal('fetch', f);
  return f;
}
const posted = (f: ReturnType<typeof stub>) => JSON.parse(String(f.mock.calls.find((c) => c[1]?.method === 'POST')![1]!.body));

function open() {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <CreateTokenDialog open onClose={() => {}} />
    </SWRConfig>,
  );
  fireEvent.change(screen.getByPlaceholderText('ci-bot'), { target: { value: 'agent-1' } });
}

beforeEach(() => {
  setAccessToken('t');
  setCsrfToken('c');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('create dialog permissions', () => {
  it('starts with Prompt Lens off and engine facts on, with one-line help', () => {
    stub();
    open();
    expect(screen.getByTestId('token-prompt-lens')).not.toBeChecked();
    expect(screen.getByTestId('token-engine-facts')).toBeChecked();
    expect(document.body.textContent).toMatch(/may use Prompt Lens on their own traffic/);
    expect(document.body.textContent).toMatch(/cache block size, match unit and context length/);
  });

  it('sends the default: prompt_lens false, engine_facts true', async () => {
    const f = stub();
    open();
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(f.mock.calls.some((c) => c[1]?.method === 'POST')).toBe(true));
    expect(posted(f)).toMatchObject({ prompt_lens: false, engine_facts: true });
  });

  it('sends what the admin ticked', async () => {
    const f = stub();
    open();
    fireEvent.click(screen.getByTestId('token-prompt-lens'));
    fireEvent.click(screen.getByTestId('token-engine-facts'));
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(f.mock.calls.some((c) => c[1]?.method === 'POST')).toBe(true));
    expect(posted(f)).toMatchObject({ prompt_lens: true, engine_facts: false });
  });
});

describe('permissions card', () => {
  it('shows both and saves each switch on its own', async () => {
    const onChange = vi.fn().mockResolvedValue(null);
    render(<PermissionsCard promptLens={false} engineFacts onChange={onChange} />);
    expect(screen.getByRole('switch', { name: 'Prompt Lens' })).toHaveAttribute('aria-checked', 'false');
    expect(screen.getByRole('switch', { name: 'Engine facts' })).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(screen.getByRole('switch', { name: 'Prompt Lens' }));
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('prompt_lens', true));
    fireEvent.click(screen.getByRole('switch', { name: 'Engine facts' }));
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('engine_facts', false));
  });

  it('shows an error and says when the platform switch is off', async () => {
    const onChange = vi.fn().mockResolvedValue('nope');
    render(<PermissionsCard promptLens engineFacts onChange={onChange} platformOn={false} />);
    expect(screen.getByTestId('perm-platform-off')).toHaveTextContent('Prompt Lens is off on this server');
    fireEvent.click(screen.getByTestId('perm-prompt_lens'));
    expect(await screen.findByRole('alert')).toHaveTextContent('nope');
    expect(document.body.textContent).not.toMatch(/god[ -]?mode/i);
  });
});
