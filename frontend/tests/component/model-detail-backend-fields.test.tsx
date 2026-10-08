import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import ModelDetailPage from '@/app/models/[id]/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// The page calls useRouter() to navigate after a delete; jsdom has no mounted
// app router. Nothing in this file exercises navigation.
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

// ---------------------------------------------------------------------------
// The Status block on the model detail page.
//
// Operator report against the live deployment: a model served by llama.cpp
// showed `Engine: vllm`, a `Tensor parallel size` and a `gpu_memory_utilization`
// that llama.cpp has no concept of, and a blank `n_gpu_layers`. The API half of
// that (the read path dropping `backend`) is covered by
// tests/unit/models/test_model_serialisation.py; this file covers the render
// half — given a response that DOES carry `backend`, the page must show the
// right engine and only the fields that engine actually consumes.
// ---------------------------------------------------------------------------

// See tests/component/model-settings.test.tsx for why the params promise is
// forged with a resolved shape rather than awaited.
function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = 'fulfilled';
  p.value = value;
  return p;
}

function renderPage(id: string) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div data-testid="suspense-fallback">loading</div>}>
        <ModelDetailPage params={syncResolved({ id })} />
      </Suspense>
    </SWRConfig>,
  );
}

function fakeModel(overrides: Record<string, unknown> = {}) {
  return {
    id: 'abc',
    served_model_name: 'llama3-8b',
    hf_repo: 'meta-llama/Llama-3-8B',
    hf_revision: 'main',
    gpu_indices: [1],
    tensor_parallel_size: 1,
    backend: 'vllm',
    mmproj_filename: null,
    n_gpu_layers: null,
    dtype: null,
    max_model_len: 8192,
    gpu_memory_utilization: 0.9,
    trust_remote_code: false,
    extra_args: [],
    extra_env: {},
    status: 'loaded',
    pulled_bytes: 1,
    pulled_total: 1,
    last_error: null,
    ...overrides,
  };
}

// The page mounts LogStream, which opens an SSE connection. jsdom has no
// EventSource; an inert stand-in keeps the log panel from throwing without
// making this file a test of the log stream.
class InertEventSource {
  close() {}
  addEventListener() {}
  removeEventListener() {}
}

/** Stub every endpoint the detail page and its child panels touch. */
function stubFetch(model: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url === '/api/models/abc') {
      return new Response(JSON.stringify(model), { status: 200 });
    }
    if (url === '/api/models/abc/dp-routing') {
      return new Response(
        JSON.stringify({
          model_id: 'abc',
          data_parallel_size: 7,
          affinity_enabled: true,
          spill_threshold: 64,
          spill_threshold_source: 'auto',
          since: null,
          totals: { in_flight: 0, sticky: 0, spilled: 0, client_pinned: 0, balanced: 0, unrouted: 0 },
          ranks: Array.from({ length: 7 }, (_, i) => ({
            rank: i, in_flight: 0, sticky: 0, spilled_in: 0, client_pinned: 0,
            requests_running: null, requests_waiting: null, kv_cache_usage_perc: null,
            prefix_cache_hits: null, prefix_cache_queries: null, prefix_cache_hit_rate: null,
          })),
          engine_metrics: { available: false, error: 'model not loaded', scraped_at: null },
        }),
        { status: 200 },
      );
    }
    if (url === '/api/models/abc/try-stack') {
      return new Response(JSON.stringify({ attempts: [] }), { status: 200 });
    }
    if (url === '/api/system/engine') {
      return new Response(
        JSON.stringify({
          driver: 'subprocess',
          supports_version_select: false,
          vllm_version: '0.26.0',
        }),
        { status: 200 },
      );
    }
    if (url.startsWith('/api/templates/engine-versions')) {
      return new Response(
        JSON.stringify({ channel: 'cuda-stable', family: null, versions: [], error: null }),
        { status: 200 },
      );
    }
    return new Response('{}', { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function renderWith(model: Record<string, unknown>) {
  stubFetch(model);
  renderPage('abc');
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByText(/Engine:/);
}

describe('ModelDetailPage — Status block, per-backend fields', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('names the engine a llama.cpp row is actually served by', async () => {
    await renderWith(fakeModel({ backend: 'llamacpp' }));
    expect(screen.getByTestId('model-engine')).toHaveTextContent('llama.cpp');
  });

  it('hides tensor parallel size and gpu_memory_utilization for llama.cpp', async () => {
    // Both are vLLM-only concepts. llama.cpp splits by layer and sizes its own
    // KV cache; rendering either invites the operator to reason about a knob
    // the engine has never heard of. Hide, do not disable (design spec §9.1).
    await renderWith(
      fakeModel({ backend: 'llamacpp', tensor_parallel_size: 1, gpu_memory_utilization: 0.9 }),
    );
    expect(screen.queryByText(/Tensor parallel size/i)).toBeNull();
    expect(screen.queryByText(/gpu_memory_utilization/i)).toBeNull();
  });

  it('renders n_gpu_layers with its value for llama.cpp, never blank', async () => {
    await renderWith(fakeModel({ backend: 'llamacpp', n_gpu_layers: 20 }));
    expect(screen.getByTestId('model-n-gpu-layers')).toHaveTextContent('20');
  });

  it('omits n_gpu_layers entirely when llama.cpp is left to auto-fit', async () => {
    // NULL means "omit the flag and let mainline choose", not "zero layers on
    // the GPU". A row with no number to show must show no row.
    await renderWith(fakeModel({ backend: 'llamacpp', n_gpu_layers: null }));
    expect(screen.queryByTestId('model-n-gpu-layers')).toBeNull();
  });

  it('keeps the vLLM-only fields on a vLLM row and hides the llama.cpp ones', async () => {
    await renderWith(fakeModel({ backend: 'vllm', tensor_parallel_size: 2 }));
    expect(screen.getByText(/Tensor parallel size/i)).toBeInTheDocument();
    expect(screen.getByText(/gpu_memory_utilization/i)).toBeInTheDocument();
    expect(screen.queryByTestId('model-n-gpu-layers')).toBeNull();
  });

  it('renders no half-empty row when the API predates the backend columns', async () => {
    // The exact shape the operator hit: a response with no `backend`,
    // `n_gpu_layers` or `mmproj_filename` key at all. The old guard tested
    // `!== null`, and `undefined !== null`, so it printed the label
    // "n_gpu_layers:" followed by nothing. A label with no value is worse than
    // an absent row -- it reads as "zero layers on the GPU", which is a real
    // and very different configuration.
    const {
      backend: _b,
      n_gpu_layers: _n,
      mmproj_filename: _m,
      ...legacy
    } = fakeModel();
    await renderWith(legacy);
    expect(screen.queryByText(/n_gpu_layers/i)).toBeNull();
    expect(screen.getByTestId('model-engine')).toHaveTextContent('vLLM');
  });

  it('labels the GPU list as indices, not as a count', async () => {
    // "GPUs: 1" on a model pinned to GPU index 1 reads as "one GPU". The
    // deployment this was reported from has exactly two cards and one model on
    // each, so both readings produce a plausible-but-wrong sentence.
    await renderWith(fakeModel({ gpu_indices: [1] }));
    expect(screen.getByTestId('model-gpu-indices')).toHaveTextContent('1');
    expect(screen.queryByText(/^GPUs:/)).toBeNull();
    expect(screen.getByText(/GPU index/i)).toBeInTheDocument();
  });
});

describe('ModelDetailPage — data-parallel layout (#286)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows replicas, per-replica TP, affinity and the routing card for dp > 1', async () => {
    await renderWith(
      fakeModel({
        gpu_indices: [0, 1, 2, 3, 4, 5, 6],
        data_parallel_size: 7,
        tensor_parallel_size: 1,
        dp_affinity_enabled: true,
        dp_spill_threshold: null,
      }),
    );
    expect(screen.getByTestId('model-data-parallel')).toHaveTextContent('7');
    expect(screen.getByText(/tensor parallel 1 per replica/i)).toBeInTheDocument();
    expect(screen.getByTestId('model-dp-affinity')).toHaveTextContent(/on/i);
    expect(screen.getByTestId('model-dp-affinity')).toHaveTextContent(/auto/i);
    expect(await screen.findByText('Data-parallel routing')).toBeInTheDocument();
    expect(await screen.findByTestId('dp-rank-table')).toBeInTheDocument();
  });

  it('shows an explicit spill threshold and affinity off', async () => {
    await renderWith(
      fakeModel({
        gpu_indices: [0, 1],
        data_parallel_size: 2,
        dp_affinity_enabled: false,
        dp_spill_threshold: 12,
      }),
    );
    expect(screen.getByTestId('model-dp-affinity')).toHaveTextContent(/off/i);
    expect(screen.getByTestId('model-dp-affinity')).toHaveTextContent('12');
  });

  it('shows neither the replicas line nor the card without the key (older API)', async () => {
    await renderWith(fakeModel());
    expect(screen.queryByTestId('model-data-parallel')).toBeNull();
    expect(screen.queryByTestId('model-dp-affinity')).toBeNull();
    expect(screen.queryByText('Data-parallel routing')).toBeNull();
  });

  it('shows neither for data_parallel_size = 1', async () => {
    await renderWith(fakeModel({ data_parallel_size: 1 }));
    expect(screen.queryByTestId('model-data-parallel')).toBeNull();
    expect(screen.queryByText('Data-parallel routing')).toBeNull();
  });
});
