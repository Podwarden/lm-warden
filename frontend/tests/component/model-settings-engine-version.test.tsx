import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { ModelSettingsPage } from '@/components/models/model-settings-page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// #343: the engine-version control on the model settings page.

const ENGINE = '0.34.0rc1.dev1202-pr69165';

function renderPage(id: string) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div>loading</div>}>
        <ModelSettingsPage id={id} />
      </Suspense>
    </SWRConfig>,
  );
}

function fakeSettings(overrides: Record<string, unknown> = {}) {
  return {
    id: 'abc',
    served_model_name: 'inkling',
    hf_repo: 'org/inkling',
    hf_revision: 'main',
    gpu_indices: [0],
    tensor_parallel_size: 1,
    data_parallel_size: 1,
    dp_affinity_enabled: 1,
    dp_spill_threshold: null,
    dtype: null,
    max_model_len: null,
    gpu_memory_utilization: 0.9,
    trust_remote_code: false,
    extra_args: [],
    extra_env: {},
    supports_tools: null,
    supports_vision: null,
    supports_reasoning: null,
    status: 'pulled',
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
    backend: 'vllm',
    n_gpu_layers: null,
    mmproj_filename: null,
    engine_channel: null,
    engine_vllm_version: null,
    ...overrides,
  };
}

function setup(settings: Record<string, unknown>, bundled: string[] = [ENGINE]) {
  const patches: unknown[] = [];
  const loads: string[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const method = (init?.method ?? 'GET').toUpperCase();
    if (url === '/api/models/abc/settings' && method === 'GET')
      return new Response(JSON.stringify(settings), { status: 200 });
    if (url === '/api/models/abc/settings' && method === 'PATCH') {
      patches.push(JSON.parse(String(init?.body)));
      return new Response('{"ok":true}', { status: 200 });
    }
    if (url === '/api/models/abc/load' && method === 'POST') {
      loads.push(url);
      return new Response('{}', { status: 202 });
    }
    if (url === '/api/system/engine')
      return new Response(
        JSON.stringify({
          driver: 'subprocess',
          supports_version_select: false,
          vllm_version: '0.32.0',
          bundled_vllm: bundled,
        }),
        { status: 200 },
      );
    if (url === '/api/system/gpus')
      return new Response(
        JSON.stringify({
          gpus: [{ index: 0, name: 'A', memory_total_mib: 16000, memory_used_mib: 0, utilization_pct: 0 }],
          probed_at: new Date().toISOString(),
          probe_error: null,
        }),
        { status: 200 },
      );
    if (url.endsWith('/effective-argv')) return new Response('{"argv":[]}', { status: 200 });
    if (url === '/api/presets') return new Response('{"presets":[]}', { status: 200 });
    return new Response('{}', { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return { patches, loads };
}

async function load() {
  renderPage('abc');
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByLabelText(/HF revision/i);
}

const trigger = () =>
  screen.getByTestId('section-engine-version').querySelector('button[aria-haspopup="listbox"]') as HTMLButtonElement;

describe('ModelSettingsPage — engine version (#343)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows the default engine and lists the bundled engines', async () => {
    setup(fakeSettings());
    await load();
    expect(trigger()).toHaveTextContent(/default/i);
    fireEvent.click(trigger());
    expect(await screen.findByRole('option', { name: ENGINE })).toBeInTheDocument();
  });

  it('shows the model\'s current pin', async () => {
    setup(fakeSettings({ engine_vllm_version: ENGINE }));
    await load();
    await waitFor(() => expect(trigger()).toHaveTextContent(ENGINE));
  });

  it('saves a picked engine, then says it applies on the next load and offers Load now', async () => {
    const { patches, loads } = setup(fakeSettings());
    await load();
    fireEvent.click(trigger());
    fireEvent.mouseDown(await screen.findByRole('option', { name: ENGINE }));
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(patches).toEqual([{ engine_vllm_version: ENGINE }]));
    const notice = await screen.findByTestId('engine-version-saved');
    expect(notice).toHaveTextContent(/next load/i);
    fireEvent.click(screen.getByRole('button', { name: /load now/i }));
    await waitFor(() => expect(loads).toHaveLength(1));
  });

  it('goes back to the default engine with null', async () => {
    const { patches } = setup(fakeSettings({ engine_vllm_version: ENGINE }));
    await load();
    fireEvent.click(trigger());
    fireEvent.mouseDown(await screen.findByRole('option', { name: /default/i }));
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    await waitFor(() => expect(patches).toEqual([{ engine_vllm_version: null }]));
  });

  it('is read-only while the model is loaded', async () => {
    setup(fakeSettings({ status: 'loaded', engine_vllm_version: ENGINE }));
    await load();
    await waitFor(() => expect(trigger()).toBeDisabled());
    expect(trigger()).toHaveTextContent(ENGINE);
  });

  it('is read-only text when a channel pins the engine image', async () => {
    setup(fakeSettings({ engine_channel: 'stable', engine_vllm_version: '0.9.2' }));
    await load();
    expect(screen.getByTestId('section-engine-version').querySelector('button[aria-haspopup="listbox"]')).toBeNull();
    expect(screen.getByTestId('engine-version-readonly')).toHaveTextContent('0.9.2');
  });

  it('keeps a pin the image no longer carries visible', async () => {
    setup(fakeSettings({ engine_vllm_version: ENGINE }), []);
    await load();
    await waitFor(() => expect(trigger()).toHaveTextContent(ENGINE));
    expect(screen.getByTestId('engine-version-missing')).toBeInTheDocument();
  });

  it('is absent for a llama.cpp model', async () => {
    setup(fakeSettings({ backend: 'llamacpp' }));
    await load();
    expect(screen.queryByTestId('section-engine-version')).toBeNull();
  });
});
