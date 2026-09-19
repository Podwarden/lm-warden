// The model pages name their breadcrumbs through useBreadcrumb — the
// app-wide strip replaced their "← Back to models" / "← Back to <name>" links.
//   /models/[id]           → Home › Models › <served name>
//   /models/[id]/settings  → Home › Models › <served name> › Settings
// A model that fails to load (404 included) is named by its id, so the crumb
// doesn't sit on the italic "Model" placeholder forever.
import { Suspense } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';

const navState = vi.hoisted(() => ({ pathname: '/models/abc' }));
vi.mock('next/navigation', () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

import ModelDetailPage from '@/app/models/[id]/page';
import ModelSettingsPage from '@/app/models/[id]/settings/page';
import { NavStackProvider } from '@/lib/nav-stack';
import { BreadcrumbHeader } from '@/components/breadcrumb-header';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// See tests/component/model-settings.test.tsx for why the params promise is
// forged with a resolved shape rather than awaited.
function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = 'fulfilled';
  p.value = value;
  return p;
}

class InertEventSource {
  close() {}
  addEventListener() {}
  removeEventListener() {}
}

const MODEL = {
  id: 'abc',
  served_model_name: 'Qwen3-8B',
  hf_repo: 'Qwen/Qwen3-8B',
  hf_revision: 'main',
  gpu_indices: [0],
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
  supports_tools: null,
  supports_vision: null,
  supports_reasoning: null,
  status: 'pulled',
  pulled_bytes: 1,
  pulled_total: 1,
  last_error: null,
};

function stubFetch(modelStatus = 200) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      if (url === '/api/models/abc' || url === '/api/models/abc/settings') {
        return modelStatus === 200
          ? new Response(JSON.stringify(MODEL), { status: 200 })
          : new Response(JSON.stringify({ detail: 'not found' }), { status: modelStatus });
      }
      if (url === '/api/system/gpus') {
        return new Response(JSON.stringify({ gpus: [], probed_at: '', probe_error: null }), { status: 200 });
      }
      if (url.startsWith('/api/models/abc/effective-argv')) {
        return new Response(JSON.stringify({ argv: ['vllm', 'serve'] }), { status: 200 });
      }
      if (url === '/api/models/abc/try-stack') return new Response(JSON.stringify({ attempts: [] }), { status: 200 });
      return new Response('{}', { status: 200 });
    }),
  );
}

function renderAt(path: string, page: React.ReactNode) {
  navState.pathname = path;
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, shouldRetryOnError: false }}>
      <NavStackProvider>
        <BreadcrumbHeader />
        <Suspense fallback={<div>loading</div>}>{page}</Suspense>
      </NavStackProvider>
    </SWRConfig>,
  );
}

const trailText = () => screen.getByRole('navigation', { name: 'Breadcrumb' }).querySelector('ol')!.textContent;

describe('model pages — breadcrumb titles', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('/models/[id] names its crumb after the served model name, and draws no back link of its own', async () => {
    stubFetch();
    renderAt('/models/abc', <ModelDetailPage params={syncResolved({ id: 'abc' })} />);
    await screen.findByRole('heading', { level: 1, name: 'Qwen3-8B' });
    await waitFor(() => expect(trailText()).toBe('HomeModelsQwen3-8B'));
    expect(screen.queryByText(/Back to models/)).toBeNull();
  });

  it('/models/[id] on a 404 names the crumb by the id', async () => {
    stubFetch(404);
    renderAt('/models/abc', <ModelDetailPage params={syncResolved({ id: 'abc' })} />);
    await screen.findByText('Model not found');
    await waitFor(() => expect(trailText()).toBe('HomeModelsabc'));
    expect(screen.queryByText(/Back to models/)).toBeNull();
  });

  it('/models/[id]/settings names the parent crumb after the model (direct load)', async () => {
    stubFetch();
    renderAt('/models/abc/settings', <ModelSettingsPage params={syncResolved({ id: 'abc' })} />);
    await screen.findByRole('heading', { level: 1, name: 'Settings' });
    await waitFor(() => expect(trailText()).toBe('HomeModelsQwen3-8BSettings'));
    const crumb = screen.getAllByRole('link', { name: 'Qwen3-8B' })[0];
    expect(crumb).toHaveAttribute('href', '/models/abc');
    expect(screen.queryByText(/Back to/)).toBeNull();
  });
});
