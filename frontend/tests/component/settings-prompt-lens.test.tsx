import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import SettingsPage from '@/app/settings/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// Settings > General > Privacy: the Prompt Lens switch (runtime setting
// `prompt_lens_enabled`, applied live). It explains what turning it on means,
// says which source decides the value in effect (the setting, the env
// variable under its new or deprecated name, or the default), and that kept
// captures on disk stay when it is turned off.

type Status = {
  enabled: boolean;
  source: 'setting' | 'env' | 'default';
  setting: boolean | null;
  env_var: string | null;
  env_value: boolean | null;
  last_change?: Record<string, unknown> | null;
};

function runtime(extra: Record<string, string | null> = {}) {
  return {
    admin_username: 'admin',
    admin_password: '***',
    hf_token: null,
    default_gpu_indices: '[]',
    prompt_lens_enabled: 'false',
    ...extra,
  };
}

const CARD = {
  id: 'c1', provider: 'upstream.example', model: 'big', currency: 'USD', input_per_m_micros: 100_000,
  output_per_m_micros: 400_000, cache_read_per_m_micros: 10_000, cache_write_per_m_micros: null, effective_from: '2026-11-21',
};
const DEFAULT = {
  provider: 'openrouter', model: 'anthropic/claude-haiku-5.5', currency: 'USD', input_per_m_micros: 100_000,
  output_per_m_micros: null, cache_read_per_m_micros: 10_000, cache_write_per_m_micros: 125_000,
  effective_from: '2026-11-21', source: 'a test source',
};

function install(status: Status, rt = runtime(), cards: Record<string, unknown> = { cards: [], defaults: [], reference: {} }) {
  const calls: { method: string; url: string; body?: string }[] = [];
  let current = { ...status };
  let currentRt = { ...rt };
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const method = (init?.method ?? 'GET').toUpperCase();
    calls.push({ method, url, body: typeof init?.body === 'string' ? init.body : undefined });
    if (method === 'GET' && url === '/api/settings/runtime') return new Response(JSON.stringify(currentRt));
    if (method === 'GET' && url === '/api/system/gpus') return new Response(JSON.stringify({ gpus: [] }));
    if (method === 'GET' && url === '/api/admin/prompt-lens/status') return new Response(JSON.stringify(current));
    if (method === 'GET' && url === '/api/admin/lens/rate-cards') return new Response(JSON.stringify(cards));
    if (method === 'GET' && url === '/api/models')
      return new Response(JSON.stringify({ models: [{ id: 'm1', served_model_name: 'local-model' }] }));
    if (method === 'PUT' && url.startsWith('/api/admin/lens/')) return new Response(JSON.stringify({ ok: true }));
    if (method === 'PATCH' && url === '/api/settings/runtime') {
      const on = JSON.parse(String(init?.body)).prompt_lens_enabled as boolean | null;
      if (on === null) {
        const envOn = current.env_value === true;
        current = { ...current, enabled: envOn, source: current.env_var ? 'env' : 'default', setting: null };
        currentRt = { ...currentRt, prompt_lens_enabled: envOn ? 'true' : 'false' };
        return new Response(JSON.stringify({ ok: true, requires_restart: [], requires_restart_kinds: [] }));
      }
      current = { ...current, enabled: on, source: 'setting', setting: on };
      currentRt = { ...currentRt, prompt_lens_enabled: on ? 'true' : 'false' };
      return new Response(JSON.stringify({ ok: true, requires_restart: [], requires_restart_kinds: [] }));
    }
    return new Response('unmocked', { status: 404 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
}

async function renderPage() {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <SettingsPage />
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe('Settings — Prompt Lens switch', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows the switch in a Privacy section with what it does', async () => {
    install({ enabled: false, source: 'default', setting: null, env_var: null, env_value: null });
    await renderPage();
    expect(await screen.findByText(/^Privacy$/)).toBeInTheDocument();
    const box = screen.getByLabelText(/Prompt Lens/);
    expect(box).not.toBeChecked();
    const text = document.body.textContent ?? '';
    expect(text).toMatch(/prompts and replies in memory/i);
    expect(text).toMatch(/key holders see their own/i);
    expect(text).toMatch(/admins see all/i);
    expect(text).toMatch(/kept captures.*stay/i);
    expect(text).toMatch(/admins can still open the captures kept on disk/i);
    expect(text).toMatch(/key holders cannot/i);
    // Nothing saved: nothing to reset.
    expect(screen.queryByRole('button', { name: /use server default/i })).toBeNull();
    expect(await screen.findByTestId('prompt-lens-source')).toHaveTextContent(/Off by default/);
  });

  it('names the env variable in effect', async () => {
    install(
      { enabled: true, source: 'env', setting: null, env_var: 'VW_PROMPT_LENS_ENABLED', env_value: true },
      runtime({ prompt_lens_enabled: 'true' }),
    );
    await renderPage();
    const note = await screen.findByTestId('prompt-lens-source');
    await waitFor(() => expect(note).toHaveTextContent(/On, from VW_PROMPT_LENS_ENABLED/));
    expect(note).not.toHaveTextContent(/deprecated/);
    expect(screen.getByLabelText(/Prompt Lens/)).toBeChecked();
  });

  it('says when the setting overrides the env', async () => {
    install(
      { enabled: false, source: 'setting', setting: false, env_var: 'VW_PROMPT_LENS_ENABLED', env_value: true },
    );
    await renderPage();
    const note = await screen.findByTestId('prompt-lens-source');
    await waitFor(() => expect(note).toHaveTextContent(/Off, set here/));
    expect(note).toHaveTextContent(/overrides VW_PROMPT_LENS_ENABLED \(on\)/);
  });

  it('saves the switch alone and shows the new source', async () => {
    const calls = install({ enabled: false, source: 'default', setting: null, env_var: null, env_value: null });
    await renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    fireEvent.click(screen.getByLabelText(/Prompt Lens/));
    const save = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(save).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(save);
    });
    const patch = calls.find((c) => c.method === 'PATCH');
    expect(patch && JSON.parse(patch.body ?? '{}')).toEqual({ prompt_lens_enabled: true });
    await waitFor(() => expect(screen.getByTestId('prompt-lens-source')).toHaveTextContent(/On, set here/));
  });

  it('"Use server default" clears the saved switch so the env value applies again', async () => {
    const calls = install({
      enabled: false,
      source: 'setting',
      setting: false,
      env_var: 'VW_PROMPT_LENS_ENABLED',
      env_value: true,
    });
    await renderPage();
    const reset = await screen.findByRole('button', { name: /use server default/i });
    await act(async () => {
      fireEvent.click(reset);
    });
    const patch = calls.find((c) => c.method === 'PATCH');
    expect(patch && JSON.parse(patch.body ?? '{}')).toEqual({ prompt_lens_enabled: null });
    await waitFor(() =>
      expect(screen.getByTestId('prompt-lens-source')).toHaveTextContent(/On, from VW_PROMPT_LENS_ENABLED/),
    );
    await waitFor(() => expect(screen.getByLabelText(/Prompt Lens/)).toBeChecked());
  });

  it('shows who changed it last', async () => {
    install({
      enabled: true,
      source: 'setting',
      setting: true,
      env_var: null,
      env_value: null,
      last_change: {
        ts: 1795400000,
        actor: 'admin',
        via: 'session',
        old_enabled: false,
        new_enabled: true,
        old_source: 'default',
        new_source: 'setting',
      },
    } as Status);
    await renderPage();
    await waitFor(() =>
      expect(screen.getByTestId('prompt-lens-source')).toHaveTextContent(/Last changed by admin \(session\), off to on/),
    );
  });

  // Owner report 2026-11-23: with the value in effect coming from the env, the
  // checkbox already shows it, so ticking it changes nothing and Save (dirty
  // fields only) sends nothing: "On" could not be stored as a setting.
  it('env source: "Save as a setting here" stores the value in effect (in edit mode)', async () => {
    const calls = install(
      { enabled: true, source: 'env', setting: null, env_var: 'VW_PROMPT_LENS_ENABLED', env_value: true },
      runtime({ prompt_lens_enabled: 'true' }),
    );
    await renderPage();
    const store = await screen.findByRole('button', { name: /save as a setting here/i });
    // Same rules as the switch: only in edit mode.
    expect(store).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: /^edit$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /save as a setting here/i })).toBeEnabled());
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /save as a setting here/i }));
    });
    const patch = calls.find((c) => c.method === 'PATCH');
    expect(patch && JSON.parse(patch.body ?? '{}')).toEqual({ prompt_lens_enabled: true });
    await waitFor(() => expect(screen.getByTestId('prompt-lens-source')).toHaveTextContent(/On, set here/));
    expect(screen.queryByRole('button', { name: /save as a setting here/i })).toBeNull();
    expect(screen.getByRole('button', { name: /use server default/i })).toBeInTheDocument();
  });

  it('env source: a refused save (admin token, session_only) says why', async () => {
    install(
      { enabled: true, source: 'env', setting: null, env_var: 'VW_PROMPT_LENS_ENABLED', env_value: true },
      runtime({ prompt_lens_enabled: 'true' }),
    );
    const base = globalThis.fetch as unknown as (i: RequestInfo, n?: RequestInit) => Promise<Response>;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo, init?: RequestInit) =>
      (init?.method ?? 'GET').toUpperCase() === 'PATCH'
        ? new Response(JSON.stringify({ detail: { code: 'session_only', message: 'Only a signed-in session can change this.' } }), { status: 403 })
        : base(input, init)));
    await renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: /save as a setting here/i }));
    });
    expect(await screen.findByRole('alert')).toHaveTextContent(/signed-in session/i);
  });

  it('outside edit mode the greyed switch says how to change it', async () => {
    install({ enabled: false, source: 'default', setting: null, env_var: null, env_value: null });
    await renderPage();
    expect(await screen.findByTestId('prompt-lens-edit-hint')).toHaveTextContent(/Press Edit to change/);
    fireEvent.click(screen.getByRole('button', { name: /^edit$/i }));
    await waitFor(() => expect(screen.queryByTestId('prompt-lens-edit-hint')).toBeNull());
  });
});


describe('Settings — Prompt Lens price cards', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });
  const status: Status = { enabled: true, source: 'setting', setting: true, env_var: null, env_value: null };

  it('lists the cards with their effective date, and the shipped list prices apart', async () => {
    install(status, runtime({ prompt_lens_enabled: 'true' }), { cards: [CARD], defaults: [DEFAULT], reference: {} });
    await renderPage();
    const list = await screen.findByTestId('price-card-list');
    expect(list).toHaveTextContent('upstream.example · big: in 0.1 · out 0.4 · read 0.01 USD, effective 2026-11-21');
    expect(screen.getByTestId('price-defaults')).toHaveTextContent('anthropic/claude-haiku-5.5 (2026-11-21)');
    expect(screen.getByTestId('price-defaults')).toHaveTextContent('not used until you confirm');
  });

  it('saves a card in micro-units per million tokens; an empty output price is refused', async () => {
    const calls = install(status, runtime({ prompt_lens_enabled: 'true' }), { cards: [], defaults: [DEFAULT], reference: {} });
    await renderPage();
    await screen.findByTestId('price-cards');
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    fireEvent.click(screen.getByRole('button', { name: /anthropic\/claude-haiku-5.5/ })); // loads the shipped prices
    expect((screen.getByLabelText('Input price') as HTMLInputElement).value).toBe('0.1');
    fireEvent.click(screen.getByRole('button', { name: 'Add or replace price card' }));
    expect(await screen.findByText('The output price must be a number.')).toBeInTheDocument();
    expect(calls.some((c) => c.method === 'PUT')).toBe(false);
    fireEvent.change(screen.getByLabelText('Output price'), { target: { value: '0.5' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Add or replace price card' }));
    });
    const put = calls.find((c) => c.method === 'PUT' && c.url === '/api/admin/lens/rate-cards');
    expect(put).toBeTruthy();
    expect(JSON.parse(put?.body ?? '{}').cards[0]).toMatchObject({
      provider: 'openrouter', model: 'anthropic/claude-haiku-5.5', input_per_m_micros: 100_000,
      output_per_m_micros: 500_000, cache_read_per_m_micros: 10_000, cache_write_per_m_micros: 125_000,
      effective_from: '2026-11-21',
    });
  });

  it('picks a reference card per own model', async () => {
    const calls = install(status, runtime({ prompt_lens_enabled: 'true' }), { cards: [CARD], defaults: [], reference: {} });
    await renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const pick = await screen.findByLabelText('Reference card for local-model');
    await act(async () => {
      fireEvent.change(pick, { target: { value: 'upstream.example\u0000big' } });
    });
    const put = calls.find((c) => c.method === 'PUT' && c.url === '/api/admin/lens/reference-card/m1');
    expect(JSON.parse(put?.body ?? '{}')).toEqual({ provider: 'upstream.example', model: 'big' });
  });
});
