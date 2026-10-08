import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import useSWR, { SWRConfig } from 'swr';
import ModelSettingsPage from '@/app/models/[id]/settings/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = 'fulfilled';
  p.value = value;
  return p;
}

function renderPage(id: string) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div>loading</div>}>
        <ModelSettingsPage params={syncResolved({ id })} />
      </Suspense>
    </SWRConfig>,
  );
}

const GPUS = {
  gpus: [0, 1, 2, 3, 4, 5, 6].map((index) => ({
    index,
    name: 'A4000',
    memory_total_mib: 16376,
    memory_used_mib: 0,
    utilization_pct: 0,
  })),
  probed_at: new Date().toISOString(),
  probe_error: null,
};

function fakeSettings(overrides: Record<string, unknown> = {}) {
  return {
    id: 'abc',
    served_model_name: 'qwen',
    hf_repo: 'org/qwen',
    hf_revision: 'main',
    gpu_indices: [0, 1, 2, 3],
    tensor_parallel_size: 4,
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
    ...overrides,
  };
}

const deletes: string[] = [];

function setup(settings: Record<string, unknown>) {
  const patches: unknown[] = [];
  deletes.length = 0;
  let current = settings;
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const method = (init?.method ?? 'GET').toUpperCase();
    if (url === '/api/models/abc/layout-notice' && method === 'DELETE') {
      deletes.push(url);
      current = { ...current, layout_notice_level: null, layout_notice: null };
      return new Response(null, { status: 204 });
    }
    if (url === '/api/models/abc/settings' && method === 'GET')
      return new Response(JSON.stringify(current), { status: 200 });
    if (url === '/api/models/abc/settings' && method === 'PATCH') {
      patches.push(JSON.parse(String(init?.body)));
      return new Response('{"ok":true}', { status: 200 });
    }
    if (url === '/api/system/gpus') return new Response(JSON.stringify(GPUS), { status: 200 });
    if (url.endsWith('/effective-argv')) return new Response('{"argv":[]}', { status: 200 });
    if (url === '/api/presets') return new Response('{"presets":[]}', { status: 200 });
    return new Response('{}', { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return patches;
}

async function load() {
  renderPage('abc');
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByLabelText(/HF revision/i);
  await waitFor(() => expect(document.getElementById('gpu-checklist-0')).toBeTruthy());
}

const gpu = (i: number) => document.getElementById(`gpu-checklist-${i}`) as HTMLInputElement;
const dpInput = () => screen.getByLabelText(/data-parallel replicas/i) as HTMLInputElement;
const tpInput = () => screen.getByLabelText(/tensor-parallel size/i) as HTMLInputElement;
const save = () => screen.getByRole('button', { name: /save/i });

describe('ModelSettingsPage — data-parallel (#286)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders the replica field for a vLLM row and not for llama.cpp', async () => {
    setup(fakeSettings());
    await load();
    expect(dpInput().value).toBe('1');
    cleanup();
    setup(fakeSettings({ backend: 'llamacpp', gpu_indices: [0], tensor_parallel_size: 1 }));
    await load();
    expect(screen.queryByLabelText(/data-parallel replicas/i)).toBeNull();
    expect(screen.queryByTestId('section-dp-routing')).toBeNull();
  });

  it('hides the routing section at dp=1 and shows it at dp=2', async () => {
    setup(fakeSettings());
    await load();
    expect(screen.queryByTestId('section-dp-routing')).toBeNull();
    expect(screen.getByTestId('dp-routing-needs-replicas')).toBeInTheDocument();
    cleanup();
    setup(fakeSettings({ data_parallel_size: 2, tensor_parallel_size: 2, dp_affinity_enabled: true }));
    await load();
    expect(screen.getByTestId('section-dp-routing')).toBeInTheDocument();
    expect(screen.queryByTestId('dp-routing-needs-replicas')).toBeNull();
    expect(screen.getByLabelText(/replica affinity/i)).toBeChecked();
    expect(screen.getByLabelText(/spill threshold/i)).toBeInTheDocument();
  });

  it('reads dp_affinity_enabled=0 (raw int) as off', async () => {
    setup(fakeSettings({ data_parallel_size: 2, tensor_parallel_size: 2, dp_affinity_enabled: 0 }));
    await load();
    expect(screen.getByLabelText(/replica affinity/i)).not.toBeChecked();
  });

  it('defaults a missing data_parallel_size key to 1 (older server)', async () => {
    const s = fakeSettings();
    delete (s as Record<string, unknown>).data_parallel_size;
    delete (s as Record<string, unknown>).dp_affinity_enabled;
    delete (s as Record<string, unknown>).dp_spill_threshold;
    setup(s);
    await load();
    expect(dpInput().value).toBe('1');
    expect(screen.queryByTestId('section-dp-routing')).toBeNull();
    expect(save()).toBeDisabled(); // not spuriously dirty
  });

  it('tp follows dp and the GPU count: tp x dp always equals the selected GPUs', async () => {
    setup(fakeSettings({ gpu_indices: [0, 1, 2, 3, 4, 5], tensor_parallel_size: 6 }));
    await load();
    fireEvent.change(dpInput(), { target: { value: '3' } });
    await waitFor(() => expect(tpInput().value).toBe('2'));
    fireEvent.change(dpInput(), { target: { value: '2' } });
    await waitFor(() => expect(tpInput().value).toBe('3'));
    // Adding a 7th GPU: 7 is not divisible by 2, so the layout falls back to
    // one replica across all seven (tp = 7) instead of leaving a stale pair.
    await act(async () => {
      fireEvent.click(gpu(6));
    });
    await waitFor(() => expect(dpInput().value).toBe('1'));
    expect(tpInput().value).toBe('7');
  });

  it('dp that does not divide the GPU count shows an error and blocks Save', async () => {
    setup(fakeSettings());
    await load();
    fireEvent.change(dpInput(), { target: { value: '3' } });
    await waitFor(() => expect(screen.getByTestId('dp-invalid-error')).toBeInTheDocument());
    expect(screen.getByTestId('dp-invalid-error').textContent).toMatch(/divide.*\(4\)/i);
    expect(save()).toBeDisabled();
  });

  it('setting dp=2 derives tp=2 and PATCHes the layout', async () => {
    const patches = setup(fakeSettings());
    await load();
    fireEvent.change(dpInput(), { target: { value: '2' } });
    await waitFor(() => expect(tpInput().value).toBe('2'));
    await waitFor(() => expect(save()).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(save());
    });
    await waitFor(() => expect(patches.length).toBe(1));
    expect(patches[0]).toEqual({ data_parallel_size: 2, tensor_parallel_size: 2 });
  });

  it('changing the GPU set so dp no longer divides resets dp to 1 and tp to the count', async () => {
    const patches = setup(
      fakeSettings({ gpu_indices: [0, 1, 2, 3], tensor_parallel_size: 2, data_parallel_size: 2 }),
    );
    await load();
    await act(async () => {
      fireEvent.click(gpu(3));
    });
    await waitFor(() => expect(dpInput().value).toBe('1'));
    expect(tpInput().value).toBe('3');
    await act(async () => {
      fireEvent.click(save());
    });
    await waitFor(() => expect(patches.length).toBe(1));
    expect(patches[0]).toEqual({
      gpu_indices: [0, 1, 2],
      data_parallel_size: 1,
      tensor_parallel_size: 3,
    });
  });

  it('on a loaded row, changing only the spill threshold is saveable and sends only that key', async () => {
    const patches = setup(
      fakeSettings({ status: 'loaded', data_parallel_size: 2, tensor_parallel_size: 2 }),
    );
    await load();
    expect(dpInput()).toBeDisabled();
    const spill = screen.getByLabelText(/spill threshold/i);
    expect(spill).not.toBeDisabled();
    fireEvent.change(spill, { target: { value: '12' } });
    await waitFor(() => expect(save()).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(save());
    });
    await waitFor(() => expect(patches.length).toBe(1));
    expect(patches[0]).toEqual({ dp_spill_threshold: 12 });
  });

  it('on a loaded row, toggling affinity sends a JSON boolean', async () => {
    const patches = setup(
      fakeSettings({ status: 'loaded', data_parallel_size: 2, tensor_parallel_size: 2 }),
    );
    await load();
    fireEvent.click(screen.getByLabelText(/replica affinity/i));
    await waitFor(() => expect(save()).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(save());
    });
    await waitFor(() => expect(patches.length).toBe(1));
    expect(patches[0]).toEqual({ dp_affinity_enabled: false });
  });

  it('shows the boot reconcile notice, as a warning when the operator must act (#286)', async () => {
    setup(
      fakeSettings({
        layout_notice_level: 'warning',
        layout_notice: 'Remove the parallel flags from extra_args.',
      }),
    );
    await load();
    const el = screen.getByTestId('layout-notice');
    expect(el).toHaveAttribute('data-level', 'warning');
    expect(el).toHaveAttribute('role', 'alert');
    expect(el).toHaveTextContent('Remove the parallel flags from extra_args.');
  });

  it('shows no notice when the row has none', async () => {
    setup(fakeSettings());
    await load();
    expect(screen.queryByTestId('layout-notice')).toBeNull();
  });

  it('dismisses an info notice through DELETE /layout-notice (#286 re-review)', async () => {
    setup(
      fakeSettings({ layout_notice_level: 'info', layout_notice: 'Layout moved to columns.' }),
    );
    await load();
    expect(screen.getByTestId('layout-notice')).toHaveTextContent('Layout moved to columns.');
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    });
    await waitFor(() => expect(screen.queryByTestId('layout-notice')).toBeNull());
    expect(deletes).toEqual(['/api/models/abc/layout-notice']);
  });

  it('revalidates the model detail and list keys after a dismiss', async () => {
    setup(fakeSettings({ layout_notice_level: 'info', layout_notice: 'Layout moved to columns.' }));
    const detailFetches: string[] = [];
    const inner = globalThis.fetch as unknown as (i: RequestInfo, o?: RequestInit) => Promise<Response>;
    vi.stubGlobal('fetch', async (i: RequestInfo, o?: RequestInit) => {
      const url = typeof i === 'string' ? i : (i as Request).url;
      if (url === '/api/models/abc') detailFetches.push(url);
      return inner(i, o);
    });
    function Probe() {
      useSWR('/api/models/abc', (u: string) => fetch(u).then((r) => r.json()), { revalidateOnFocus: false, dedupingInterval: 0 });
      return null;
    }
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <Probe />
        <Suspense fallback={<div>loading</div>}>
          <ModelSettingsPage params={syncResolved({ id: 'abc' })} />
        </Suspense>
      </SWRConfig>,
    );
    await screen.findByLabelText(/HF revision/i);
    await waitFor(() => expect(detailFetches.length).toBe(1));
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    });
    await waitFor(() => expect(detailFetches.length).toBe(2));
  });

  it('offers no dismiss on a warning', async () => {
    setup(fakeSettings({ layout_notice_level: 'warning', layout_notice: 'Fix extra_args.' }));
    await load();
    expect(screen.getByTestId('layout-notice')).toBeTruthy();
    expect(screen.queryByRole('button', { name: /dismiss/i })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Settings redesign (plan 2026-10-04-settings-redesign §2, §4, Task 4): the
// band you can act on comes first, locked engine sections collapse to one-line
// summaries, the save bar carries the total / live split / invalid reason, and
// the reload rule is stated once per band instead of once per field.
// ---------------------------------------------------------------------------
describe('ModelSettingsPage — attention flow (settings redesign)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  const precedes = (a: Element, b: Element) =>
    Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

  it('renders the Live band before the Engine band on a loaded model', async () => {
    setup(fakeSettings({ status: 'loaded', data_parallel_size: 2, tensor_parallel_size: 2 }));
    await load();
    expect(precedes(screen.getByTestId('band-live'), screen.getByTestId('band-engine'))).toBe(true);
  });

  it('renders the Engine band first when the model is not loaded', async () => {
    setup(fakeSettings());
    await load();
    expect(precedes(screen.getByTestId('band-engine'), screen.getByTestId('band-live'))).toBe(true);
    // Not loaded is not an alert: the strip is a polite status.
    expect(screen.getByTestId('settings-status-strip')).toHaveAttribute('role', 'status');
  });

  it('collapses the engine sections to value summaries when loaded', async () => {
    setup(fakeSettings({ status: 'loaded', data_parallel_size: 2, tensor_parallel_size: 2 }));
    await load();
    for (const id of ['section-compute', 'section-memory', 'section-identity', 'section-advanced']) {
      expect(screen.getByTestId(`${id}-toggle`)).toHaveAttribute('aria-expanded', 'false');
      expect(screen.getByTestId(`${id}-summary`)).toBeInTheDocument();
    }
    expect(screen.getByTestId('section-compute-summary').textContent).toMatch(/2 replicas × 2 GPUs/);
    expect(screen.getByTestId('section-memory-summary').textContent).toMatch(/0\.90 of VRAM/);
    // The live sections stay open.
    expect(screen.getByTestId('section-dp-routing-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByTestId('section-capabilities-toggle')).toHaveAttribute('aria-expanded', 'true');
  });

  it('opens Layout and Memory by default when not loaded, and keeps identity collapsed', async () => {
    setup(fakeSettings());
    await load();
    expect(screen.getByTestId('section-compute-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByTestId('section-memory-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByTestId('section-identity-toggle')).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByTestId('section-advanced-toggle')).toHaveAttribute('aria-expanded', 'false');
  });

  it('counts two live changes on a loaded row and says Save applies now', async () => {
    const patches = setup(
      fakeSettings({ status: 'loaded', data_parallel_size: 2, tensor_parallel_size: 2 }),
    );
    await load();
    fireEvent.change(screen.getByLabelText(/spill threshold/i), { target: { value: '48' } });
    fireEvent.click(screen.getByRole('button', { name: 'Vision' }));
    fireEvent.mouseDown(screen.getByRole('option', { name: 'Yes' }));
    await waitFor(() =>
      expect(screen.getByTestId('settings-save-summary').textContent).toMatch(/2 unsaved changes/),
    );
    expect(screen.getByTestId('save-bar-tag-live').textContent).toBe('2 live');
    expect(screen.queryByTestId('save-bar-tag-next')).toBeNull();
    expect(save()).toHaveAccessibleName('Save — applies now');
    await act(async () => {
      fireEvent.click(save());
    });
    await waitFor(() => expect(patches.length).toBe(1));
    expect(patches[0]).toEqual({ dp_spill_threshold: 48, supports_vision: true });
  });

  it('splits a mixed draft into next-load and live changes', async () => {
    setup(fakeSettings());
    await load();
    fireEvent.change(dpInput(), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Vision' }));
    fireEvent.mouseDown(screen.getByRole('option', { name: 'Yes' }));
    // dp=2 also derives tp=2: three keys, two of them next-load.
    await waitFor(() => expect(screen.getByTestId('save-bar-tag-next').textContent).toBe('2 on next load'));
    expect(screen.getByTestId('save-bar-tag-live').textContent).toBe('1 live');
    expect(save()).toHaveAccessibleName('Save');
    // Section counts sit beside their section.
    expect(screen.getByTestId('section-compute-dirty').textContent).toBe('2 unsaved');
  });

  it('shows no per-field "requires model-reload" badge anywhere', async () => {
    setup(fakeSettings());
    await load();
    expect(document.body.textContent).not.toMatch(/requires model-reload/i);
  });

  it('repeats the invalid replicas reason in the save bar, and Fix focuses the Replicas input', async () => {
    setup(fakeSettings());
    await load();
    fireEvent.change(dpInput(), { target: { value: '3' } });
    await waitFor(() =>
      expect(screen.getByTestId('settings-save-summary').textContent).toMatch(
        /Can't save: replicas must divide the 4 selected GPUs/,
      ),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Fix' }));
    expect(document.activeElement).toBe(dpInput());
    expect(dpInput().id).toBe('dp');
    // The layout picture shows the same problem, without becoming an alert.
    expect(screen.getByTestId('layout-diagram')).toHaveAttribute('data-invalid', 'true');
  });

  it('blocks Save when tp × dp ≠ GPUs, and Fix focuses the tensor-parallel input', async () => {
    setup(fakeSettings({ gpu_indices: [0, 1], tensor_parallel_size: 2 }));
    await load();
    fireEvent.change(tpInput(), { target: { value: '1' } });
    await waitFor(() =>
      expect(screen.getByTestId('settings-save-summary').textContent).toMatch(
        /Can't save: tensor-parallel size × replicas must equal the 2 selected GPUs/,
      ),
    );
    expect(save()).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Fix' }));
    expect(document.activeElement).toBe(tpInput());
  });

  it('shows the GiB readout under GPU memory utilization', async () => {
    setup(fakeSettings());
    await load();
    // 0.9 × 16376 MiB = 14.4 GiB of 16 GiB.
    const readout = screen.getByTestId('gpu-memory-readout');
    expect(readout.textContent).toMatch(/14\.4 GiB of 16 GiB per GPU/);
    const described = screen.getByLabelText(/gpu memory utilization/i).getAttribute('aria-describedby') ?? '';
    expect(described.split(' ')).toContain(readout.id);
  });

  it('draws the layout picture from the selected GPUs', async () => {
    setup(fakeSettings({ data_parallel_size: 2, tensor_parallel_size: 2 }));
    await load();
    expect(screen.getByRole('img', { name: '4 GPUs = 2 replicas × 2 GPUs each (tensor-parallel 2)' })).toBeInTheDocument();
  });

  it('gives focus back to the field after a keyboard save (T5)', async () => {
    // Fields are disabled while saving; a real browser drops focus to <body>
    // when the focused input becomes disabled, so after Ctrl+S the operator
    // lost their place. jsdom does not blur on disable, so the test does it.
    const patches = setup(
      fakeSettings({ status: 'loaded', data_parallel_size: 2, tensor_parallel_size: 2 }),
    );
    const base = globalThis.fetch as unknown as (i: RequestInfo, init?: RequestInit) => Promise<Response>;
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo, init?: RequestInit) => {
        if ((init?.method ?? 'GET').toUpperCase() === 'PATCH') await gate;
        return base(input, init);
      }),
    );
    await load();
    const spill = screen.getByLabelText(/spill threshold/i) as HTMLInputElement;
    spill.focus();
    fireEvent.change(spill, { target: { value: '7' } });
    fireEvent.keyDown(spill, { key: 's', ctrlKey: true });
    await waitFor(() => expect(spill).toBeDisabled());
    // Move focus off the input the way the browser would (jsdom ignores
    // blur() on a disabled element): focus a throwaway node, then drop it.
    const sink = document.body.appendChild(document.createElement('button'));
    sink.focus();
    sink.remove();
    expect(document.activeElement).toBe(document.body);
    await act(async () => {
      release();
    });
    await waitFor(() => expect(patches.length).toBe(1));
    await waitFor(() => expect(spill).not.toBeDisabled());
    await waitFor(() => expect(document.activeElement).toBe(spill));
  });

  it('expands a collapsed section that a preset changes', async () => {
    // A preset that touches Advanced opens Advanced (plan §3).
    setup(fakeSettings());
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : (input as Request).url;
        const method = (init?.method ?? 'GET').toUpperCase();
        if (url === '/api/models/abc/settings' && method === 'GET')
          return new Response(JSON.stringify(fakeSettings()), { status: 200 });
        if (url === '/api/system/gpus') return new Response(JSON.stringify(GPUS), { status: 200 });
        if (url === '/api/presets')
          return new Response(
            JSON.stringify({
              presets: [
                {
                  id: 'p',
                  name: 'P',
                  description: 'd',
                  target_archetype: 'any',
                  settings: { extra_env: { HF_HUB_OFFLINE: '1' } },
                },
              ],
            }),
            { status: 200 },
          );
        return new Response('{"argv":[]}', { status: 200 });
      }),
    );
    await load();
    expect(screen.getByTestId('section-advanced-toggle')).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(await screen.findByTestId('preset-chip-p'));
    fireEvent.click(await screen.findByTestId('preset-apply'));
    await waitFor(() =>
      expect(screen.getByTestId('section-advanced-toggle')).toHaveAttribute('aria-expanded', 'true'),
    );
  });
});

describe('ModelSettingsPage — review fixes (settings redesign)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  /** setup() plus a switch that makes later GETs of the row report `loaded`. */
  function setupFlippable(settings: Record<string, unknown>) {
    const patches = setup(settings);
    const base = globalThis.fetch as unknown as (i: RequestInfo, init?: RequestInit) => Promise<Response>;
    const state = { loaded: false };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo, init?: RequestInit) => {
        const r = await base(input, init);
        const url = typeof input === 'string' ? input : (input as Request).url;
        if (state.loaded && url === '/api/models/abc/settings' && (init?.method ?? 'GET') === 'GET') {
          const j = await r.json();
          return new Response(JSON.stringify({ ...j, status: 'loaded' }), { status: 200 });
        }
        return r;
      }),
    );
    return { patches, state };
  }

  it('a loaded row holding an unsaved engine edit says why it cannot save, and Reset clears it', async () => {
    // Review #1/#2: the draft re-seeds only when patchable fields change, so a
    // refetch that merely flips status to loaded (here: dismissing the layout
    // notice) keeps an engine edit made while unloaded.
    const { patches, state } = setupFlippable(
      fakeSettings({
        data_parallel_size: 2,
        tensor_parallel_size: 2,
        layout_notice_level: 'info',
        layout_notice: 'Promoted to replicas.',
      }),
    );
    await load();
    fireEvent.change(screen.getByLabelText(/HF revision/i), { target: { value: 'v2' } });
    // A live edit too: "every key is live" must fail on the mix (kills some()).
    fireEvent.change(screen.getByLabelText(/spill threshold/i), { target: { value: '8' } });
    state.loaded = true;
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    await waitFor(() =>
      expect(screen.getByTestId('settings-status-pill').textContent).toMatch(/Loaded/),
    );
    expect(save()).toBeDisabled();
    expect(save()).toHaveAccessibleName('Save');
    expect(screen.getByTestId('settings-save-summary').textContent).toMatch(
      /Can't save: engine changes can't be saved while loaded/,
    );
    fireEvent.keyDown(window, { key: 's', ctrlKey: true });
    expect(patches).toEqual([]);
    const reset = screen.getByTestId('settings-reset');
    expect(reset).not.toBeDisabled();
    fireEvent.click(reset);
    await waitFor(() =>
      expect(screen.getByTestId('settings-save-summary').textContent).toMatch(/No unsaved changes/),
    );
    expect((screen.getByLabelText(/HF revision/i) as HTMLInputElement).value).toBe('main');
  });

  it('Fix opens a collapsed Layout section before focusing Replicas', async () => {
    // Review #3: Fix focused #dp inside a hidden section body.
    setup(fakeSettings());
    await load();
    fireEvent.change(dpInput(), { target: { value: '3' } });
    fireEvent.click(screen.getByTestId('section-compute-toggle'));
    expect(screen.getByTestId('section-compute-toggle')).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(screen.getByRole('button', { name: 'Fix' }));
    expect(screen.getByTestId('section-compute-toggle')).toHaveAttribute('aria-expanded', 'true');
    expect(document.activeElement).toBe(dpInput());
  });

  it('opens Layout on a loaded row whose saved layout is invalid', async () => {
    // The auto-open rule (plan §3): Layout is collapsed by default when
    // loaded, but a section holding an error must not stay collapsed.
    setup(fakeSettings({ status: 'loaded', gpu_indices: [0, 1, 2], data_parallel_size: 2, tensor_parallel_size: 1 }));
    await load();
    await waitFor(() =>
      expect(screen.getByTestId('section-compute-toggle')).toHaveAttribute('aria-expanded', 'true'),
    );
  });

  it('applies every key a preset diff shows, including live and llama.cpp keys', async () => {
    // Review minor: computeDiff listed supports_* / dp_* / n_gpu_layers /
    // mmproj_filename, but Apply silently dropped them.
    setup(fakeSettings({ data_parallel_size: 2, tensor_parallel_size: 2 }));
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo, init?: RequestInit) => {
        const url = typeof input === 'string' ? input : (input as Request).url;
        const method = (init?.method ?? 'GET').toUpperCase();
        if (url === '/api/models/abc/settings' && method === 'GET')
          return new Response(
            JSON.stringify(fakeSettings({ data_parallel_size: 2, tensor_parallel_size: 2 })),
            { status: 200 },
          );
        if (url === '/api/system/gpus') return new Response(JSON.stringify(GPUS), { status: 200 });
        if (url === '/api/presets')
          return new Response(
            JSON.stringify({
              presets: [
                {
                  id: 'p',
                  name: 'P',
                  description: 'd',
                  target_archetype: 'any',
                  settings: {
                    supports_vision: true,
                    dp_spill_threshold: 8,
                    dp_affinity_enabled: false,
                    n_gpu_layers: 20,
                    mmproj_filename: 'mm.gguf',
                  },
                },
              ],
            }),
            { status: 200 },
          );
        return new Response('{"argv":[]}', { status: 200 });
      }),
    );
    await load();
    fireEvent.click(await screen.findByTestId('preset-chip-p'));
    const shown = screen
      .getAllByTestId('preset-diff-row')
      .map((r) => r.getAttribute('data-diff-key'));
    fireEvent.click(await screen.findByTestId('preset-apply'));
    await waitFor(() => expect(screen.getAllByTestId('save-bar-diff-row').length).toBeGreaterThan(0));
    const applied = screen
      .getAllByTestId('save-bar-diff-row')
      .map((r) => r.getAttribute('data-diff-key'));
    expect(shown.sort()).toEqual(
      ['dp_affinity_enabled', 'dp_spill_threshold', 'mmproj_filename', 'n_gpu_layers', 'supports_vision'].sort(),
    );
    expect(applied.sort()).toEqual(shown.sort());
  });
});
