/**
 * The cache store for the admin (plan 2, task 14): its use on the Models page, a
 * model's weight and cache store setting, and the server's Prompt cache settings tab.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

import { KvUsagePanel } from '@/components/kv/kv-usage-panel';
import { KvStoreSection } from '@/components/kv/kv-store-section';
import { PromptCacheTab } from '@/components/settings/prompt-cache-tab';
import { setAccessToken } from '@/lib/auth-fetch';
import { KV_USAGE } from './kv-fixtures';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const NAMES = {
  'm-qwen': { name: 'qwen3-235b-a22b', kv_store: 'ram+nvme' as const },
  'm-coder': { name: 'coder-30b', kv_store: 'ram' as const },
};

describe('models page: KV cache store use', () => {
  it('shows the tiers since the store started and one row per model, keys on click', () => {
    render(<KvUsagePanel usage={KV_USAGE} models={NAMES} storeOn isolated={new Set(['tok1'])} />);
    expect(screen.getByText(/since the store started, 3 d ago/)).toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('11.6 GiB')).toBeInTheDocument();
    expect(screen.getByText('231 GB')).toBeInTheDocument();
    const table = screen.getByTestId('kv-usage-table');
    expect(within(table).getByText('qwen3-235b-a22b')).toBeInTheDocument();
    expect(within(table).getByText('RAM only')).toBeInTheDocument();
    // The first model opens; the second lists its keys on click.
    expect(within(table).getByText('dana-laptop')).toBeInTheDocument();
    expect(within(table).getByText('412 · by nightly-evals')).toBeInTheDocument();
    expect(within(table).getByText('isolated')).toBeInTheDocument();
    fireEvent.click(within(table).getByRole('button', { name: /qwen3-235b-a22b/ }));
    expect(within(table).queryByText('dana-laptop')).toBeNull();
  });

  it('a store that does not answer, and no model with the store on', () => {
    render(<KvUsagePanel usage={{ ...KV_USAGE, available: false, tiers: null, models: [] }} models={NAMES} storeOn />);
    expect(screen.getByText(/Usage is unavailable. Requests keep working/)).toBeInTheDocument();
    cleanup();
    render(<KvUsagePanel usage={{ ...KV_USAGE, available: false, tiers: null, models: [] }} models={NAMES} storeOn={false} />);
    expect(screen.getByText(/No model has the KV cache store on/)).toBeInTheDocument();
  });
});

function section(over: Partial<Parameters<typeof KvStoreSection>[0]> = {}) {
  const onStore = vi.fn();
  const onWeight = vi.fn();
  render(
    <KvStoreSection
      backend="vllm"
      loaded={false}
      kvStore="ram"
      weight={3}
      nvmeDir="/kv"
      cacheIsolation="isolated"
      onStore={onStore}
      onWeight={onWeight}
      part={0.75}
      policy="fair"
      {...over}
    />,
  );
  return { onStore, onWeight };
}

describe('model settings: KV cache store and weight', () => {
  it('weight is a live field; the store choice is an engine one', () => {
    const { onStore, onWeight } = section();
    fireEvent.change(screen.getByLabelText(/^Weight/), { target: { value: '5' } });
    expect(onWeight).toHaveBeenCalledWith(5);
    fireEvent.click(screen.getByRole('radio', { name: /RAM \+ disk/ }));
    expect(onStore).toHaveBeenCalledWith('ram+nvme');
    expect(screen.getByText(/Isolated keys:/)).toHaveTextContent('Isolated keys: served.');
  });

  it('RAM + disk is refused while no disk folder is set, pointing to the server settings', () => {
    section({ nvmeDir: '' });
    expect(screen.getByRole('radio', { name: /RAM \+ disk/ })).toBeDisabled();
    expect(screen.getByText(/No disk folder is set/)).toHaveTextContent('Settings, Prompt cache');
  });

  it('the weight waits for fair share, like the caps', () => {
    section({ policy: 'lru' });
    expect(screen.getByLabelText(/^Weight/)).toBeDisabled();
    expect(screen.getByText(/weights work only with the fair eviction policy/)).toBeInTheDocument();
  });

  it('locked while loaded', () => {
    section({ loaded: true });
    expect(screen.getByRole('radio', { name: /^Off/ })).toBeDisabled();
    expect(screen.getByLabelText(/^Weight/)).toBeEnabled();
    expect(screen.getByText(/Unload the model to change this/)).toBeInTheDocument();
  });

  it('llama.cpp: no store, isolated keys refused, no weight', () => {
    section({ backend: 'llamacpp', kvStore: 'off', cacheIsolation: 'unsupported' });
    expect(screen.getByRole('radio', { name: /^RAM$|^RAM vLLM/ })).toBeDisabled();
    expect(screen.queryByLabelText(/^Weight/)).toBeNull();
    expect(screen.getByText(/Isolated keys:/)).toHaveTextContent('Isolated keys: refused.');
  });
});

const RUNTIME = {
  kv_store_ram_gib: '16', kv_store_nvme_dir: '/kv', kv_store_nvme_cap_gb: '250',
  kv_evict_policy: null, kv_idle_ram_min: null, kv_idle_nvme_h: null, kv_isolation_default: null,
};

describe('settings: Prompt cache tab', () => {
  beforeEach(() => setAccessToken('admin'));
  afterEach(() => setAccessToken(null));

  function tab(runtime: Record<string, string | null> = RUNTIME) {
    const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith('/api/settings/runtime') && init?.method === 'PATCH') return new Response(JSON.stringify({ ok: true, requires_restart: [] }));
      if (url.startsWith('/api/settings/runtime')) return new Response(JSON.stringify(runtime));
      if (url.startsWith('/api/system/kv-store'))
        return new Response(JSON.stringify({ daemon: { healthy: true, generation: 1, ram_requested_bytes: 16 * 1024 ** 3, ram_granted_bytes: 16 * 1024 ** 3 }, models: {} }));
      if (url.startsWith('/api/kv/usage')) return new Response(JSON.stringify(KV_USAGE));
      return new Response('{}', { status: 404 });
    });
    vi.stubGlobal('fetch', f);
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <PromptCacheTab />
      </SWRConfig>,
    );
    return f;
  }

  it('the idle times wait for fair share; save sends only what changed', async () => {
    const f = tab();
    const ramIdle = await screen.findByLabelText(/Keep recent parts in RAM for/);
    expect(ramIdle).toBeDisabled();
    expect(screen.getByText(/The two times, key caps and model weights only work with fair share/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('radio', { name: /Fair share/ }));
    expect(ramIdle).toBeEnabled();
    fireEvent.change(ramIdle, { target: { value: '90' } });
    fireEvent.click(screen.getByRole('radio', { name: /Isolated/ }));
    expect(screen.getByText('Unsaved')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(f.mock.calls.some(([, i]) => (i as RequestInit | undefined)?.method === 'PATCH')).toBe(true));
    const patch = f.mock.calls.find(([, i]) => (i as RequestInit | undefined)?.method === 'PATCH')!;
    expect(JSON.parse((patch[1] as RequestInit).body as string)).toEqual({
      kv_evict_policy: 'fair', kv_idle_ram_min: 90, kv_isolation_default: 'isolated',
    });
  });

  it('store size carries the restart chip and the status line', async () => {
    tab();
    expect(await screen.findByText('Requires warden restart')).toBeInTheDocument();
    expect(await screen.findByText('KV cache store running')).toBeInTheDocument();
    expect(screen.getByLabelText(/Disk folder/)).toHaveValue('/kv');
  });
});

describe('settings: Prompt cache tab radio groups', () => {
  beforeEach(() => setAccessToken('admin'));
  afterEach(() => setAccessToken(null));
  it('each group has a name', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith('/api/settings/runtime')) return new Response(JSON.stringify(RUNTIME));
      return new Response('{}', { status: 404 });
    }));
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <PromptCacheTab />
      </SWRConfig>,
    );
    expect(await screen.findByRole('radiogroup', { name: 'Eviction policy' })).toBeInTheDocument();
    expect(screen.getByRole('radiogroup', { name: 'Cache for new keys' })).toBeInTheDocument();
  });
});
