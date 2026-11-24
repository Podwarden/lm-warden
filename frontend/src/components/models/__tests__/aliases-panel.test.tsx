// Component tests for the Models > Aliases tab (Task 7): the panel and the
// alias editor modal. Driven through user-visible interactions; the network
// is stubbed at the global fetch boundary (CSRF/token plumbing included via
// the auth-fetch wrapper, like every other component test).
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { AliasesPanel } from '@/components/models/aliases-panel';

// ---- fixtures / fetch stub helpers ────────────────────────────────────────

const MODELS = [
  {
    id: 'm1',
    served_model_name: 'qwen',
    hf_repo: 'acme/qwen',
    hf_revision: 'main',
    gpu_indices: [0],
    tensor_parallel_size: null,
    status: 'loaded',
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
  },
  {
    id: 'm2',
    served_model_name: 'llama',
    hf_repo: 'acme/llama',
    hf_revision: 'main',
    gpu_indices: [0],
    tensor_parallel_size: null,
    status: 'registered',
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
  },
];

function makeAlias(overrides: Record<string, unknown> = {}) {
  return {
    id: 'a1',
    name: 'lmw-default',
    targets: [
      { id: 'm1', served_model_name: 'qwen', status: 'loaded' },
      { id: 'm2', served_model_name: 'llama', status: 'registered' },
    ],
    guard: 'inherit',
    thinking_policy: 'inherit',
    params: {},
    enabled: true,
    created_at: '2026-11-24T00:00:00',
    updated_at: '2026-11-24T00:00:00',
    ...overrides,
  };
}

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

const posts: { url: string; body: unknown }[] = [];

function stubFetch(routes: Record<string, (init?: RequestInit) => Response>) {
  const f = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? 'GET').toUpperCase();
    if (method === 'POST') {
      posts.push({ url, body: init?.body ? JSON.parse(String(init.body)) : null });
    }
    const key = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((k) => {
        const [m, p] = k.includes(' ') ? k.split(' ') : ['GET', k];
        return m === method && url.includes(p);
      });
    if (!key) return new Response('{}', { status: 404 });
    return routes[key](init);
  });
  vi.stubGlobal('fetch', f);
  return f;
}

function renderPanel() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <AliasesPanel />
    </SWRConfig>,
  );
}

beforeEach(() => {
  setAccessToken('jwt');
  setCsrfToken('csrf');
  posts.length = 0;
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Models > Aliases tab', () => {
  it('renders the empty state when no aliases exist', async () => {
    stubFetch({
      '/api/aliases': () => json({ aliases: [] }),
      '/api/models': () => json({ models: MODELS }),
    });
    renderPanel();
    expect(await screen.findByTestId('aliases-empty-state')).toHaveTextContent(
      'No aliases yet. An alias is a name clients call; it sends requests to the first loaded model in its list, with optional settings.',
    );
    expect(screen.getByTestId('add-alias-button')).toBeInTheDocument();
  });

  it('lists two aliases with their targets in order', async () => {
    stubFetch({
      '/api/aliases': () =>
        json({
          aliases: [
            makeAlias(),
            makeAlias({
              id: 'a2',
              name: 'second',
              targets: [{ id: 'm2', served_model_name: 'llama', status: 'registered' }],
              enabled: false,
            }),
          ],
        }),
      '/api/models': () => json({ models: MODELS }),
    });
    renderPanel();
    const rows = await screen.findAllByTestId(/^alias-row-/);
    expect(rows).toHaveLength(2);
    // target order follows the API's ordered target list
    expect(within(rows[0]).getAllByTestId('alias-target').map((t) => t.textContent)).toEqual([
      'qwen',
      'llama',
    ]);
    expect(within(rows[1]).getAllByTestId('alias-target').map((t) => t.textContent)).toEqual([
      'llama',
    ]);
    expect(within(rows[0]).getByText('lmw-default')).toBeInTheDocument();
    expect(within(rows[1]).getByText('second')).toBeInTheDocument();
  });

  it('offers Cap only for numeric parameters', async () => {
    stubFetch({
      '/api/aliases': () => json({ aliases: [makeAlias()] }),
      '/api/models': () => json({ models: MODELS }),
    });
    renderPanel();
    fireEvent.click(await screen.findByTestId('add-alias-button'));
    const paramSelect = await screen.findByLabelText('Parameter');
    fireEvent.change(paramSelect, { target: { value: 'reasoning_effort' } });
    const modeSelect = screen.getByLabelText('Mode') as HTMLSelectElement;
    expect(Array.from(modeSelect.options).map((o) => o.textContent)).not.toContain('Cap');
    fireEvent.change(paramSelect, { target: { value: 'max_tokens' } });
    const modeSelect2 = screen.getByLabelText('Mode') as HTMLSelectElement;
    expect(Array.from(modeSelect2.options).map((o) => o.textContent)).toContain('Cap');
  });

  it('shows the priority note when the name equals a model name', async () => {
    stubFetch({
      '/api/aliases': () => json({ aliases: [makeAlias()] }),
      '/api/models': () => json({ models: MODELS }),
    });
    renderPanel();
    fireEvent.click(await screen.findByTestId('add-alias-button'));
    fireEvent.change(await screen.findByLabelText('Name'), { target: { value: 'qwen' } });
    expect(
      await screen.findByText('This alias takes priority over the model with the same name.'),
    ).toBeInTheDocument();
  });

  it('Save posts the expected JSON body', async () => {
    stubFetch({
      '/api/aliases': (init) => {
        const method = (init?.method ?? 'GET').toUpperCase();
        if (method === 'POST') return json(makeAlias({ id: 'a9' }), 201);
        return json({ aliases: [makeAlias()] });
      },
      '/api/models': () => json({ models: MODELS }),
    });
    renderPanel();
    fireEvent.click(await screen.findByTestId('add-alias-button'));
    fireEvent.change(await screen.findByLabelText('Name'), {
      target: { value: 'lmw-default' },
    });
    fireEvent.change(await screen.findByLabelText('Add a target model'), {
      target: { value: 'm2' },
    });
    fireEvent.change(screen.getByLabelText('Sensitive data'), {
      target: { value: 'sanitize' },
    });
    fireEvent.change(screen.getByLabelText('Thinking'), {
      target: { value: 'off' },
    });
    fireEvent.change(await screen.findByLabelText('Parameter'), {
      target: { value: 'max_tokens' },
    });
    fireEvent.change(screen.getByLabelText('Value'), {
      target: { value: '16384' },
    });
    fireEvent.change(screen.getByLabelText('Mode'), {
      target: { value: 'cap' },
    });
    fireEvent.click(screen.getByTestId('save-alias-button'));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0].url).toContain('/api/aliases');
    expect(posts[0].body).toEqual({
      name: 'lmw-default',
      targets: ['m2'],
      guard: 'sanitize',
      thinking_policy: 'off',
      enabled: true,
      params: { max_tokens: { value: 16384, mode: 'cap' } },
    });
  });
});
