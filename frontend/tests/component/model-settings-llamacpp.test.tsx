import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import ModelSettingsPage from '@/app/models/[id]/settings/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// ---------------------------------------------------------------------------
// The llama.cpp settings page was vLLM's page with two fields swapped. Its
// genuinely important knobs -- flash attention and the two KV cache types --
// had no control at all, so the only way to set them was to hand-type a flag
// into Extra args.
//
// They now have controls that store into `extra_args` rather than into new
// columns. The properties this file pins:
//
//   * the controls exist for a llama.cpp row and are ABSENT for a vLLM one --
//     the same per-backend rule as every other field, from the same table;
//   * a value set in a control never duplicates or conflicts with the same
//     flag typed by hand: it has exactly one home, and Extra args shows only
//     what the controls do not own;
//   * "unset" stays reachable, so an untouched control changes no argv.
// ---------------------------------------------------------------------------

function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = 'fulfilled';
  p.value = value;
  return p;
}

const DEFAULT_GPU_PROBE = {
  gpus: [
    {
      index: 0,
      name: 'A4000',
      memory_total_mib: 16376,
      memory_used_mib: 0,
      utilization_pct: 0,
    },
  ],
  probed_at: new Date().toISOString(),
  probe_error: null,
};

function fakeSettings(overrides: Record<string, unknown> = {}) {
  return {
    id: 'abc',
    served_model_name: 'qwen3.8-27b',
    hf_repo: 'ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF',
    hf_revision: 'main',
    gpu_indices: [0],
    tensor_parallel_size: 1,
    dtype: null,
    max_model_len: 8192,
    gpu_memory_utilization: 0.9,
    trust_remote_code: false,
    extra_args: [] as string[],
    extra_env: {},
    supports_tools: null,
    supports_vision: null,
    supports_reasoning: null,
    status: 'pulled',
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
    backend: 'llamacpp',
    n_gpu_layers: null,
    mmproj_filename: null,
    ...overrides,
  };
}

function install(settings: Record<string, unknown>) {
  const patches: Record<string, unknown>[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const method = (init?.method ?? 'GET').toUpperCase();
    if (url === '/api/models/abc/settings' && method === 'GET') {
      return new Response(JSON.stringify(settings), { status: 200 });
    }
    if (url === '/api/models/abc/settings' && method === 'PATCH') {
      patches.push(JSON.parse(String(init?.body ?? '{}')));
      return new Response('{"ok":true}', { status: 200 });
    }
    if (url === '/api/system/gpus') {
      return new Response(JSON.stringify(DEFAULT_GPU_PROBE), { status: 200 });
    }
    if (url.startsWith('/api/models/abc/effective-argv')) {
      return new Response(JSON.stringify({ argv: [], changed: [] }), { status: 200 });
    }
    if (url.startsWith('/api/presets')) {
      return new Response(JSON.stringify({ presets: [] }), { status: 200 });
    }
    return new Response('{}', { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return patches;
}

async function renderPage(settings: Record<string, unknown>) {
  const patches = install(settings);
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div data-testid="suspense-fallback">loading</div>}>
        <ModelSettingsPage params={syncResolved({ id: 'abc' })} />
      </Suspense>
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByLabelText(/HF revision/i);
  return patches;
}

/** The Select primitive is a custom listbox, not a native <select>: a button
 *  trigger showing the current label, and role="option" rows that commit on
 *  mousedown. Driving it the way a person does is also the only way the test
 *  exercises the same onChange the page wires up. */
function selectValue(label: string, optionLabel: string) {
  fireEvent.click(screen.getByLabelText(label));
  fireEvent.mouseDown(
    screen.getAllByRole('option').find((o) => o.textContent === optionLabel)!,
  );
}

/** The label currently shown on a Select trigger. */
function selectedLabel(label: string): string {
  return screen.getByLabelText(label).textContent ?? '';
}

async function save() {
  const btn = screen.getByRole('button', { name: /save/i });
  await waitFor(() => expect(btn).not.toBeDisabled());
  fireEvent.click(btn);
}

describe('ModelSettingsPage — llama.cpp engine controls', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('offers flash attention and both KV cache types on a llama.cpp row', async () => {
    await renderPage(fakeSettings());
    expect(screen.getByTestId('section-llamacpp')).toBeInTheDocument();
    expect(screen.getByLabelText('Flash attention')).toBeInTheDocument();
    expect(screen.getByLabelText('KV cache type (K)')).toBeInTheDocument();
    expect(screen.getByLabelText('KV cache type (V)')).toBeInTheDocument();
  });

  it('hides the whole section on a vLLM row', async () => {
    // Same rule and same table as every other per-backend field. Rendering
    // the section header over an empty body would be a control that opens
    // onto nothing.
    await renderPage(fakeSettings({ backend: 'vllm' }));
    expect(screen.queryByTestId('section-llamacpp')).toBeNull();
    expect(screen.queryByLabelText('Flash attention')).toBeNull();
  });

  it('defaults every control to "default", so an untouched row changes no argv', async () => {
    // The launch command for an unset control has to be byte-identical to
    // today's. Writing out the engine's own default instead would move the
    // argv of every existing row -- and tests/fixtures/launch_golden.json
    // would be the first to say so.
    await renderPage(fakeSettings());
    expect(selectedLabel('Flash attention')).toContain('default');
    expect(selectedLabel('KV cache type (K)')).toContain('default');
  });

  it('reads an existing hand-typed flag INTO the control', async () => {
    // Not left in the free-text box to be duplicated by the control's own
    // write. One flag, one home.
    await renderPage(fakeSettings({ extra_args: ['--flash-attn', 'on'] }));
    expect(selectedLabel('Flash attention')).toContain('on');
  });

  it('reads the short alias too', async () => {
    // `-fa` is what an operator following upstream docs types.
    await renderPage(fakeSettings({ extra_args: ['-ctk', 'q8_0'] }));
    expect(selectedLabel('KV cache type (K)')).toContain('q8_0');
  });

  it('keeps managed flags OUT of the Extra args box', async () => {
    await renderPage(
      fakeSettings({ extra_args: ['--threads', '8', '--flash-attn', 'on'] }),
    );
    const box = screen.getByLabelText(/Extra args/i) as HTMLTextAreaElement;
    expect(box.value).toContain('--threads');
    expect(box.value).not.toContain('--flash-attn');
  });

  it('writes the control into extra_args on save', async () => {
    const patches = await renderPage(fakeSettings());
    selectValue('Flash attention', 'on');
    await save();
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]).toEqual({ extra_args: ['--flash-attn', 'on'] });
  });

  it('never sends a flash_attn key to the backend', async () => {
    // The control has no column. PATCHing one would 400 -- these three are
    // rendering keys, and the write they produce is an extra_args write.
    const patches = await renderPage(fakeSettings());
    selectValue('KV cache type (K)', 'q8_0');
    await save();
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(Object.keys(patches[0])).toEqual(['extra_args']);
  });

  it('REPLACES a hand-typed flag rather than appending a second copy', async () => {
    // The failure this whole design exists to prevent: llama-server receiving
    // `--flash-attn off --flash-attn on` and honouring the last, while the UI
    // shows one value and the operator believes another.
    const patches = await renderPage(
      fakeSettings({ extra_args: ['--flash-attn', 'off'] }),
    );
    selectValue('Flash attention', 'on');
    await save();
    await waitFor(() => expect(patches).toHaveLength(1));
    const args = patches[0].extra_args as string[];
    expect(args.filter((a) => a === '--flash-attn')).toHaveLength(1);
    expect(args).toEqual(['--flash-attn', 'on']);
  });

  it('clearing a control removes the flag entirely', async () => {
    const patches = await renderPage(
      fakeSettings({ extra_args: ['--cache-type-k', 'q8_0'] }),
    );
    selectValue('KV cache type (K)', 'default');
    await save();
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0].extra_args).toEqual([]);
  });

  it('editing Extra args does not clear a control the operator did not touch', async () => {
    // The box shows only the unmanaged half, so a naive write-back would drop
    // the managed flags it never saw.
    const patches = await renderPage(
      fakeSettings({ extra_args: ['--threads', '8', '--flash-attn', 'on'] }),
    );
    const box = screen.getByLabelText(/Extra args/i);
    fireEvent.change(box, { target: { value: '--threads\n16' } });
    await save();
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0].extra_args).toEqual(['--threads', '16', '--flash-attn', 'on']);
  });

  it('explains that split-mode and tensor-split are derived, not settable', async () => {
    // Explain rather than offer a dead control: --split-mode and --main-gpu
    // follow the GPU selection (decision D4), so a control here would be a
    // second, competing source for a fact the Compute section owns.
    await renderPage(fakeSettings());
    expect(screen.getByTestId('llamacpp-derived-note').textContent).toMatch(
      /--split-mode/,
    );
  });

  it('says Extra args holds only what the operator added', async () => {
    // The operator's question, verbatim: "why is Extra args empty but
    // Effective argv has so many arguments?"
    await renderPage(fakeSettings());
    expect(screen.getByTestId('effective-argv-subtitle').textContent).toMatch(
      /whole/i,
    );
  });
});
