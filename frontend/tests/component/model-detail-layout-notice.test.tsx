import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act, fireEvent, waitFor } from '@testing-library/react';
import { SWRConfig } from 'swr';
import ModelDetailPage from '@/app/models/[id]/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

// #286 review #8: the boot reconcile's outcome (model.layout_notice) is shown on
// the model page -- a warning when the operator must act, an info note otherwise.

function syncResolved<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & { status?: string; value?: T };
  p.status = 'fulfilled';
  p.value = value;
  return p;
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
    status: 'pulled',
    pulled_bytes: 1,
    pulled_total: 1,
    last_error: null,
    ...overrides,
  };
}

class InertEventSource {
  close() {}
  addEventListener() {}
  removeEventListener() {}
}

const deletes: string[] = [];

async function renderWith(initial: Record<string, unknown>) {
  let model = initial;
  deletes.length = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      if (url === '/api/models/abc/layout-notice' && init?.method === 'DELETE') {
        deletes.push(url);
        model = { ...model, layout_notice: null };
        return new Response(null, { status: 204 });
      }
      if (url === '/api/models/abc') return new Response(JSON.stringify(model), { status: 200 });
      if (url === '/api/models/abc/try-stack')
        return new Response(JSON.stringify({ attempts: [] }), { status: 200 });
      return new Response('{}', { status: 200 });
    }),
  );
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div>loading</div>}>
        <ModelDetailPage params={syncResolved({ id: 'abc' })} />
      </Suspense>
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByText(/Engine:/);
}

describe('ModelDetailPage — parallel layout notice', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows a warning that tells the operator what to change', async () => {
    await renderWith(
      fakeModel({
        layout_notice: { level: 'warning', message: 'Remove the flags from extra_args.' },
      }),
    );
    const el = screen.getByTestId('layout-notice');
    expect(el).toHaveAttribute('role', 'alert');
    expect(el).toHaveAttribute('data-level', 'warning');
    expect(el).toHaveTextContent('Remove the flags from extra_args.');
  });

  it('shows an informational note for a promoted row', async () => {
    await renderWith(
      fakeModel({ layout_notice: { level: 'info', message: 'Layout moved out of extra_args.' } }),
    );
    const el = screen.getByTestId('layout-notice');
    expect(el).toHaveAttribute('data-level', 'info');
    expect(el).toHaveTextContent('Layout moved out of extra_args.');
  });

  it('dismisses an info note through DELETE /layout-notice (#286 re-review)', async () => {
    await renderWith(fakeModel({ layout_notice: { level: 'info', message: 'Moved.' } }));
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    });
    await waitFor(() => expect(screen.queryByTestId('layout-notice')).toBeNull());
    expect(deletes).toEqual(['/api/models/abc/layout-notice']);
  });

  it('offers no dismiss on a warning', async () => {
    await renderWith(fakeModel({ layout_notice: { level: 'warning', message: 'Fix it.' } }));
    expect(screen.queryByRole('button', { name: /dismiss/i })).toBeNull();
  });

  it('shows nothing when there is no notice or the API predates it', async () => {
    await renderWith(fakeModel({ layout_notice: null }));
    expect(screen.queryByTestId('layout-notice')).toBeNull();
    cleanup();
    await renderWith(fakeModel());
    expect(screen.queryByTestId('layout-notice')).toBeNull();
  });
});
