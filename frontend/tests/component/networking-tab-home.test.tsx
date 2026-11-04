import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { NetworkingTab } from '@/components/settings/networking-tab';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// The home page control (spec 2026-11-04): Off / Built-in + what Off answers.

function runtime(overrides: Record<string, string> = {}): Record<string, string> {
  return {
    admin_username: 'admin',
    public_url: '',
    analytics_ga4_id: '',
    analytics_google_ads_id: '',
    search_console_verification: '',
    home_page_mode: 'builtin',
    home_page_off_action: 'console',
    ...overrides,
  };
}

function setup(initial: Record<string, string>, patched: { body: Record<string, unknown> | null }) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (method === 'GET' && url === '/api/settings/runtime') {
        return new Response(JSON.stringify(initial), { status: 200 });
      }
      if (method === 'PATCH' && url === '/api/settings/runtime') {
        patched.body = JSON.parse((init?.body as string) ?? '{}');
        return new Response(
          JSON.stringify({ ok: true, requires_restart: [], requires_restart_kinds: [] }),
          { status: 200 },
        );
      }
      return new Response('unmocked', { status: 404 });
    }),
  );
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <NetworkingTab />
    </SWRConfig>,
  );
}

describe('NetworkingTab: home page control', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows Built-in selected and no website switch', async () => {
    setup(runtime(), { body: null });
    const builtin = await screen.findByRole('radio', { name: 'Built-in' });
    expect(builtin).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Off' })).not.toBeChecked();
    expect(screen.queryByText('Public website')).toBeNull();
    // Off action only matters while Off.
    expect(screen.getByLabelText('When the home page is off')).toBeDisabled();
  });

  it('saves Off with the 404 action', async () => {
    const patched: { body: Record<string, unknown> | null } = { body: null };
    setup(runtime(), patched);
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    fireEvent.click(await screen.findByRole('radio', { name: 'Off' }));
    const select = screen.getByLabelText('When the home page is off');
    await waitFor(() => expect(select).not.toBeDisabled());
    expect(screen.getByRole('option', { name: 'Open the console' })).toBeInTheDocument();
    fireEvent.change(select, { target: { value: '404' } });
    const save = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(save).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(save);
    });
    await waitFor(() => expect(patched.body).not.toBeNull());
    expect(patched.body).toEqual({ home_page_mode: 'off', home_page_off_action: '404' });
  });
});
