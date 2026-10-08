import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import ModelDetailPage from '@/app/models/[id]/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { versionPinControlApplies } from '@/lib/backend-fields';

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
}));

// ---------------------------------------------------------------------------
// The Try-stack card offers a CUDA channel and a vLLM version. Both are vLLM's
// image catalogue (app/runtime/backends/vllm/images.py), and there is no
// llama.cpp counterpart -- so on a llama.cpp row the card offered a control
// that, if it could be operated at all, would resolve a *vLLM* image and launch
// vLLM under the operator's llama.cpp model name.
//
// The rule is already written down for the field-level case in
// frontend/src/lib/backend-fields.ts: hide what the engine has no concept of,
// disable-and-explain what this deployment merely cannot do. This file pins
// that the panel obeys the same rule, driven by version_pin_reason_code:
//
//   "driver"  -> removable by the operator: card visible, controls disabled,
//                the server's sentence rendered beside them.
//   "backend" -> not removable by anyone: card absent.
// ---------------------------------------------------------------------------

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

const DRIVER_REASON =
  'This deployment runs the in-container engine driver: the engine version ' +
  'is fixed by the warden image, so a pin is refused rather than applied. ' +
  'Version selection requires the docker engine driver (VW_ENGINE_DRIVER=docker).';

const BACKEND_REASON =
  'llama.cpp cannot be version-pinned on any driver: its binary is compiled ' +
  'into the warden image and there is no llama.cpp image catalogue to pin ' +
  'against.';

function backendsPayload() {
  return {
    default: 'vllm',
    driver: 'subprocess',
    engine_version: '0.26.0',
    backends: [
      {
        name: 'llamacpp',
        display_name: 'llama.cpp',
        version: 'b1234',
        supports_version_pin: true,
        version_pin_available: false,
        version_pin_reason: BACKEND_REASON,
        version_pin_reason_code: 'backend',
      },
      {
        name: 'vllm',
        display_name: 'vLLM',
        version: '0.26.0',
        supports_version_pin: true,
        version_pin_available: false,
        version_pin_reason: DRIVER_REASON,
        version_pin_reason_code: 'driver',
      },
    ],
  };
}

function fakeModel(overrides: Record<string, unknown> = {}) {
  return {
    id: 'abc',
    served_model_name: 'qwen3.8-27b',
    hf_repo: 'ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF',
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
    status: 'pulled',
    pulled_bytes: 1,
    pulled_total: 1,
    last_error: null,
    ...overrides,
  };
}

function stubFetch(model: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url === '/api/models/abc') {
      return new Response(JSON.stringify(model), { status: 200 });
    }
    if (url === '/api/models/abc/try-stack') {
      return new Response(JSON.stringify({ attempts: [] }), { status: 200 });
    }
    if (url === '/api/system/backends') {
      return new Response(JSON.stringify(backendsPayload()), { status: 200 });
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
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <Suspense fallback={<div data-testid="suspense-fallback">loading</div>}>
        <ModelDetailPage params={syncResolved({ id: 'abc' })} />
      </Suspense>
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
  await screen.findByText(/Engine:/);
  // A second flush: the panel's own SWR fetch resolves a tick after the page's.
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe('versionPinControlApplies', () => {
  it('is true when the pin works, so nothing is hidden on a capable deployment', () => {
    expect(versionPinControlApplies(null)).toBe(true);
  });

  it('keeps the control for a DRIVER obstacle -- the operator can remove it', () => {
    expect(versionPinControlApplies('driver')).toBe(true);
  });

  it('drops the control for a BACKEND obstacle -- no driver can remove it', () => {
    expect(versionPinControlApplies('backend')).toBe(false);
  });

  it('drops the control when the engine cannot be pinned at all', () => {
    expect(versionPinControlApplies('engine')).toBe(false);
  });

  it('keeps the control for an unrecognised code, failing towards visible', () => {
    // Same defensive default as /api/system/engine, which reports
    // supports_version_select=True for an unknown driver so it never WRONGLY
    // blocks. A code this build does not know is a newer server, not a reason
    // to silently remove a working control.
    expect(versionPinControlApplies('something-new-in-2027')).toBe(true);
  });
});

describe('ModelDetailPage — Try-stack card visibility', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', InertEventSource as unknown as typeof EventSource);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('hides the whole card for a llama.cpp row', async () => {
    // Not "disabled": there is no llama.cpp image catalogue, so a channel and a
    // version are not unavailable here -- they are meaningless everywhere. A
    // greyed-out box invites the operator to wonder what it does.
    await renderWith(fakeModel({ backend: 'llamacpp' }));
    expect(screen.queryByText('Try stack')).toBeNull();
    expect(screen.queryByTestId('try-stack-channel')).toBeNull();
    expect(screen.queryByTestId('try-stack-version')).toBeNull();
    expect(screen.queryByTestId('try-stack-submit')).toBeNull();
  });

  it('never shows a llama.cpp row vLLM copy about channels or versions', async () => {
    await renderWith(fakeModel({ backend: 'llamacpp' }));
    expect(screen.queryByText(/cuda-stable/i)).toBeNull();
    expect(screen.queryByText(/vLLM version/i)).toBeNull();
    expect(screen.queryByTestId('try-stack-driver-note')).toBeNull();
  });

  it('keeps the card for a vLLM row on the subprocess driver, disabled', async () => {
    await renderWith(fakeModel({ backend: 'vllm' }));
    expect(screen.getByText('Try stack')).toBeInTheDocument();
    expect(screen.getByTestId('try-stack-channel')).toBeDisabled();
    expect(screen.getByTestId('try-stack-submit')).toBeDisabled();
  });

  it("renders the server's sentence rather than copy invented in the client", async () => {
    // The frontend cannot write this sentence: one boolean does not say which
    // obstacle applies, and the remedy differs. Rendering it verbatim is what
    // keeps the explanation and the gate in agreement.
    await renderWith(fakeModel({ backend: 'vllm' }));
    expect(screen.getByTestId('try-stack-driver-note')).toHaveTextContent(
      /requires the docker engine driver/i,
    );
  });

  it('treats a row with no backend as vLLM, so old rows keep the card', async () => {
    const { backend: _b, ...legacy } = fakeModel();
    await renderWith(legacy);
    expect(screen.getByText('Try stack')).toBeInTheDocument();
  });
});
