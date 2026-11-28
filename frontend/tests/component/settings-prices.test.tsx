import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, act, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import SettingsPage from '@/app/settings/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// Settings > Prices (#347): a provider section per provider in use with each price's source
// and date, an override that is saved in micro-units per million tokens, models with no known
// price, reference prices for local models, and the refresh state.

const SONNET = {
  model: 'claude-sonnet-4-5-20250929',
  price: { currency: 'USD', input_per_m_micros: 3_000_000, output_per_m_micros: 15_000_000, cache_read_per_m_micros: 300_000, cache_write_per_m_micros: 3_750_000 },
  source: 'openrouter',
  fetched_at: '2026-11-26T19:19:32Z',
  requests: 12,
};
const UNKNOWN = { model: 'mystery-1', price: null, source: null, fetched_at: null, requests: 2 };

const ANSWER = {
  providers: [
    { host: 'api.anthropic.com', label: 'Anthropic', reasons: ['router upstream', 'relayed requests'], catalogue_models: 16, models: [SONNET, UNKNOWN] },
  ],
  local: [{ id: 'm1', name: 'local-llm', reference: null, priced_as: null }],
  refresh: {
    interval_hours: 24, network_allowed: true, last_attempt: 1795719572, last_success: 1795719572, last_error: null,
    last_source: 'openrouter', catalogue_rows: 579, snapshot: { source: 'openrouter', fetched_at: '2026-11-26T19:19:32Z', rows: 579 },
    sources: ['openrouter.ai (primary)', 'raw.githubusercontent.com, LiteLLM (fallback)'],
  },
};

function install(answer: Record<string, unknown> = ANSWER) {
  const calls: { method: string; url: string; body?: string }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      calls.push({ method, url, body: typeof init?.body === 'string' ? init.body : undefined });
      if (url === '/api/settings/runtime') return new Response(JSON.stringify({ admin_username: 'admin', admin_password: '***', hf_token: null, default_gpu_indices: '[]' }));
      if (url === '/api/system/gpus') return new Response(JSON.stringify({ gpus: [] }));
      if (url === '/api/admin/prompt-lens/status') return new Response(JSON.stringify({ enabled: true, source: 'default', setting: null, env_var: null, env_value: null }));
      if (method === 'GET' && url === '/api/admin/prices') return new Response(JSON.stringify(answer));
      if (method === 'GET' && url === '/api/admin/prices/audit') return new Response(JSON.stringify({ entries: [{ ts: 1795719572, actor: 'admin', action: 'override_set', provider: 'api.anthropic.com', model: 'x', old: null, new: {} }] }));
      if (url.startsWith('/api/admin/prices') || url.startsWith('/api/admin/lens/')) return new Response(JSON.stringify({ ok: true }));
      return new Response('unmocked', { status: 404 });
    }),
  );
  return calls;
}

async function openTab() {
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <SettingsPage />
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  fireEvent.click(await screen.findByRole('button', { name: 'Prices' }));
  return screen.findByTestId('prices-tab');
}

describe('Settings — Prices tab', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows each provider in use with every price, its source and its date', async () => {
    install();
    await openTab();
    const section = await screen.findByTestId('price-provider-api.anthropic.com');
    expect(section).toHaveTextContent('Anthropic');
    expect(section).toHaveTextContent('router upstream, relayed requests');
    const rows = within(section).getAllByTestId('price-model');
    expect(rows[0]).toHaveTextContent('in 3 · out 15 · cache read 0.3 · cache write 3.75 USD/M');
    expect(rows[0]).toHaveTextContent('OpenRouter, 2026-11-26');
    expect(rows[1]).toHaveTextContent('no price known: Prompt Lens shows tokens');
    expect(screen.getByTestId('price-refresh')).toHaveTextContent('Refreshed every 24 h');
    expect(screen.getByTestId('price-refresh')).toHaveTextContent('bundled snapshot of 2026-11-26 (579 prices, openrouter)');
  });

  it('saves a per-price override in micro-units per million tokens', async () => {
    const calls = install();
    await openTab();
    const section = await screen.findByTestId('price-provider-api.anthropic.com');
    fireEvent.click(within(section).getAllByRole('button', { name: 'override' })[0]);
    expect((screen.getByLabelText('Input price') as HTMLInputElement).value).toBe('3');
    fireEvent.change(screen.getByLabelText('Input price'), { target: { value: '2.5' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Save override' }));
    });
    const put = calls.find((c) => c.method === 'PUT' && c.url === '/api/admin/prices/override');
    expect(put).toBeTruthy();
    expect(JSON.parse(put?.body ?? '{}')).toMatchObject({
      provider: 'api.anthropic.com',
      model: 'claude-sonnet-4-5-20250929',
      input_per_m_micros: 2_500_000,
      output_per_m_micros: 15_000_000,
      cache_read_per_m_micros: 300_000,
    });
  });

  it('refuses a price that is not a number and sends nothing', async () => {
    const calls = install();
    await openTab();
    const section = await screen.findByTestId('price-provider-api.anthropic.com');
    fireEvent.click(within(section).getAllByRole('button', { name: 'set price' })[0]);
    fireEvent.change(screen.getByLabelText('Input price'), { target: { value: 'abc' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save override' }));
    expect(await screen.findByText('The input price must be a number.')).toBeInTheDocument();
    expect(calls.some((c) => c.method === 'PUT')).toBe(false);
  });

  it('prices a local model as a provider model, and shows the refresh error with the kept prices', async () => {
    const calls = install({ ...ANSWER, refresh: { ...ANSWER.refresh, last_error: 'RuntimeError: HTTP 500' } });
    await openTab();
    expect(await screen.findByTestId('price-refresh-error')).toHaveTextContent('the previous prices are kept: RuntimeError: HTTP 500');
    const local = screen.getByTestId('price-local');
    expect(local).toHaveTextContent('no reference price');
    fireEvent.change(screen.getByLabelText('Model to price local-llm as'), { target: { value: 'claude-sonnet-4.5' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Price as this' }));
    });
    const put = calls.find((c) => c.method === 'PUT' && c.url === '/api/admin/lens/reference-card/m1');
    expect(JSON.parse(put?.body ?? '{}')).toEqual({ provider: 'api.anthropic.com', model: 'claude-sonnet-4.5' });
  });

  it('lists recent changes', async () => {
    install();
    await openTab();
    expect(await screen.findByTestId('price-audit')).toHaveTextContent('admin · override set · api.anthropic.com');
  });
});
