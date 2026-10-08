import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import SettingsPage from '@/app/settings/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// ---------------------------------------------------------------------------
// Settings page tests — pin the spec-level contracts for the #154 redesign:
//
//   1. General tab renders Identity / Hugging Face / Defaults sections
//      with the right keys, and the secret-sentinel dirty rule still holds
//      (typing into nothing-else does NOT leak admin_password/hf_token
//      into the PATCH body).
//   2. Sessions & Tokens tab — PATCH echo `warden-restart` surfaces the
//      warden-restart banner.
//   3. General tab — PATCH echo `model-reload` surfaces the model-reload
//      banner (Hugging Face section).
//   4. PATCH 422 surfaces `detail` in an error banner.
//   5. Model tab with no loaded model: empty-state copy + dropdown.
//   6. Model tab with a loaded model: link to /models/<id>/settings.
//
// We deliberately do NOT pin SWR internals (mutate timing, dedupingInterval
// quirks). Each test renders inside a fresh SWR cache via `provider: () =>
// new Map()` so cached responses don't bleed across tests.
//
// Per-tab keyboard / wireframe coverage lives in `networking-tab.test.tsx`,
// `public-url-helper.test.ts`, and the membership contract test —
// settings.test.tsx is the "page wiring" level only.
// ---------------------------------------------------------------------------

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <SettingsPage />
    </SWRConfig>,
  );
}

// Default runtime snapshot — every key has a value so every tab renders
// fully wired. Secret keys carry the `***` sentinel the way the real
// backend returns them. Per-test we override via spread.
function fakeRuntime(overrides: Record<string, string | null> = {}): Record<string, string | null> {
  return {
    admin_username: 'admin',
    admin_password: '***',
    hf_token: '***',
    default_gpu_indices: '[0]',
    default_token_expiration_days: '365',
    rotation_grace_hours: '24',
    session_access_ttl_minutes: '15',
    session_refresh_ttl_days: '7',
    sse_ticket_ttl_seconds: '60',
    vllm_version: '0.9.2',
    log_retention_lines: '5000',
    landing_page_enabled: 'true',
    public_url: '',
    ...overrides,
  };
}

interface FakeModel {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number | null;
  status: 'registered' | 'pulling' | 'pulled' | 'loading' | 'loaded' | 'unloading' | 'failed';
  pulled_bytes: number;
  pulled_total: number | null;
  last_error: string | null;
}

function fakeModel(overrides: Partial<FakeModel> = {}): FakeModel {
  return {
    id: 'm1',
    served_model_name: 'llama3-8b',
    hf_repo: 'meta-llama/Llama-3-8B',
    hf_revision: 'main',
    gpu_indices: [0],
    tensor_parallel_size: 1,
    status: 'registered',
    pulled_bytes: 0,
    pulled_total: null,
    last_error: null,
    ...overrides,
  };
}

function mockFetch(
  routes: Record<string, (init?: RequestInit) => Response>,
): ReturnType<typeof vi.fn> {
  return vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    const method = (init?.method ?? 'GET').toUpperCase();
    const key = `${method} ${url}`;
    const handler = routes[key];
    if (handler) return handler(init);
    return new Response(`unmocked: ${key}`, { status: 404 });
  });
}

// Click a tabstrip button by label. The Tabs primitive renders <button>
// children for each tab; the role is implicit ("button"). We match by
// accessible name (exact) so tabs whose label is a prefix of another
// (e.g. "Sessions" vs "Sessions & Tokens") don't disambiguate wrong.
async function gotoTab(label: RegExp) {
  fireEvent.click(screen.getByRole('button', { name: label }));
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

describe('SettingsPage — General tab (Identity / HF / Defaults)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders Identity / Hugging Face / Defaults sections with the right keys', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/system/gpus': () =>
        new Response(JSON.stringify({ gpus: [] }), { status: 200 }),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Section headings come from setting-section.tsx — exact text per
    // wireframe at docs/superpowers/specs/2026-05-24-settings-redesign-design.md.
    expect(await screen.findByText(/^Identity$/)).toBeInTheDocument();
    expect(screen.getByText(/^Hugging Face$/)).toBeInTheDocument();
    expect(screen.getByText(/^Defaults for new models$/)).toBeInTheDocument();

    // Spot-check one field per section so a stray accidental key removal
    // in RUNTIME_GENERAL_KEYS shows up here, not just in the membership
    // contract test.
    expect(screen.getByText('Admin username')).toBeInTheDocument();
    expect(screen.getByText('Hugging Face token')).toBeInTheDocument();
    expect(screen.getByText('Default GPU indices')).toBeInTheDocument();

    // default_gpu_indices now renders as a GpuChecklist (gpu-set kind).
    // With gpus=[] stub and configured index 0, a ghost-row checkbox for
    // GPU 0 appears (missing-but-configured index path).
    expect(screen.getByRole('checkbox', { name: /GPU 0/i })).toBeInTheDocument();
  });

  it('renders a real (non-ghost) GPU checkbox that toggles selection when present', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime({ default_gpu_indices: '[0]' })), { status: 200 }),
      'GET /api/system/gpus': () =>
        new Response(
          JSON.stringify({
            gpus: [
              { index: 0, name: 'RTX 4090', memory_total_mib: 24564, memory_used_mib: 0, utilization_pct: 0 },
            ],
            probed_at: '2026-05-26T00:00:00Z',
            probe_error: null,
          }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // A present GPU renders a real checkbox (label includes the card name)
    // and starts checked because default_gpu_indices is [0]. No ghost-row
    // warning banner should appear since index 0 IS present.
    const gpuCheckbox = await screen.findByRole('checkbox', { name: /RTX 4090/i });
    expect(gpuCheckbox).toBeInTheDocument();
    expect(gpuCheckbox).toBeChecked();

    // Enter edit mode, then unchecking the GPU mutates the draft away from
    // the snapshot ([0] -> []), marking the form dirty and enabling Save.
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    expect(saveBtn).toBeDisabled();
    fireEvent.click(gpuCheckbox);
    expect(gpuCheckbox).not.toBeChecked();
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
  });

  it('shows model-reload banner when PATCH echoes model-reload (HF token)', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/system/gpus': () =>
        new Response(JSON.stringify({ gpus: [] }), { status: 200 }),
      'PATCH /api/settings/runtime': () =>
        new Response(
          JSON.stringify({
            ok: true,
            requires_restart: ['model-reload'],
            requires_restart_kinds: ['model-reload'],
          }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const tokenInput = await screen.findByLabelText(/Hugging Face token/i);
    fireEvent.change(tokenInput, { target: { value: 'hf_newtoken' } });
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => {
      const status = screen.getByRole('status');
      expect(status.textContent).toMatch(/unloaded \+ reloaded/i);
    });
    const status = screen.getByRole('status');
    expect(status.textContent).not.toMatch(/warden process must be restarted/i);
  });

  it('surfaces 422 detail in an error banner', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/system/gpus': () =>
        new Response(JSON.stringify({ gpus: [] }), { status: 200 }),
      'PATCH /api/settings/runtime': () =>
        new Response(
          JSON.stringify({ detail: 'hf_token: invalid token' }),
          { status: 422 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const hfTokenInput = await screen.findByLabelText(/Hugging Face token/i);
    fireEvent.change(hfTokenInput, { target: { value: 'hf_bogus' } });
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    // Use getAllByRole since the GpuChecklist also emits a role="alert"
    // warning banner for configured-but-missing GPU indices. Find the
    // save-error banner specifically by its content.
    const alerts = await screen.findAllByRole('alert');
    const errorBanner = alerts.find((el) => el.textContent?.includes('hf_token: invalid token'));
    expect(errorBanner).toBeDefined();
    expect(errorBanner!.textContent).toMatch(/hf_token: invalid token/);
  });

  it('does NOT include unchanged secret fields in PATCH body', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/system/gpus': () =>
        new Response(JSON.stringify({ gpus: [] }), { status: 200 }),
      'PATCH /api/settings/runtime': () =>
        new Response(
          JSON.stringify({ ok: true, requires_restart: [], requires_restart_kinds: [] }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Enter edit, change ONLY admin_username. Both secret fields are left
    // untouched — their inputs render empty (the sentinel `***` is never
    // displayed) and the dirty-tracker must not interpret "empty" as a
    // wipe of the existing password/token.
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    // The ghost hf_cache_dir setting was removed entirely (2026-06-15 ENOSPC
    // follow-up): editing it in the UI silently did nothing because the real
    // cache path is env-driven (VW_HF_CACHE_DIR). The field must not render.
    expect(screen.queryByLabelText(/HF cache directory/i)).toBeNull();
    const usernameInput = await screen.findByLabelText(/Admin username/i);
    fireEvent.change(usernameInput, { target: { value: 'newadmin' } });
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => {
      const patchCall = fetchMock.mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
      );
      expect(patchCall).toBeDefined();
    });
    const patchCall = fetchMock.mock.calls.find(
      (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH',
    )!;
    const body = JSON.parse((patchCall[1] as RequestInit).body as string);
    expect(body.admin_username).toBe('newadmin');
    expect(body.admin_password).toBeUndefined();
    expect(body.hf_token).toBeUndefined();
  });
});

describe('SettingsPage — admin password rule (>= 12 characters, <= 72 bytes)', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function typePassword(value: string, save = true) {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/system/gpus': () =>
        new Response(JSON.stringify({ gpus: [] }), { status: 200 }),
      'PATCH /api/settings/runtime': () =>
        new Response(
          JSON.stringify({ ok: true, requires_restart: [], requires_restart_kinds: [] }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);
    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const input = await screen.findByLabelText(/Admin password/i);
    fireEvent.change(input, { target: { value } });
    if (!save) return fetchMock;
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });
    return fetchMock;
  }

  const patches = (m: ReturnType<typeof vi.fn>) =>
    m.mock.calls.filter((c) => (c[1] as RequestInit | undefined)?.method === 'PATCH');

  it('shows the rule in the field hint', async () => {
    await typePassword('', false);
    expect(screen.getByText(/At least 12 characters \(at most 72 bytes\)/)).toBeInTheDocument();
  });

  it('refuses an 11-character password without sending it', async () => {
    const fetchMock = await typePassword('a'.repeat(11));
    const alerts = await screen.findAllByRole('alert');
    expect(alerts.some((el) => /at least 12 characters/.test(el.textContent ?? ''))).toBe(true);
    expect(patches(fetchMock)).toHaveLength(0);
  });

  it('refuses a password over 72 bytes without sending it', async () => {
    const fetchMock = await typePassword('é'.repeat(37)); // 37 chars, 74 bytes
    const alerts = await screen.findAllByRole('alert');
    expect(alerts.some((el) => /at most 72 bytes/.test(el.textContent ?? ''))).toBe(true);
    expect(patches(fetchMock)).toHaveLength(0);
  });

  it('sends a 12-character password', async () => {
    const fetchMock = await typePassword('twelve-chars');
    await waitFor(() => expect(patches(fetchMock)).toHaveLength(1));
    const body = JSON.parse((patches(fetchMock)[0][1] as RequestInit).body as string);
    expect(body.admin_password).toBe('twelve-chars');
  });
});

describe('SettingsPage — Sessions & Tokens tab', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows warden-restart banner when PATCH echoes warden-restart (session TTL)', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'PATCH /api/settings/runtime': () =>
        new Response(
          JSON.stringify({
            ok: true,
            requires_restart: ['warden-restart'],
            requires_restart_kinds: ['warden-restart'],
          }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    // Switch to the Sessions & Tokens tab — exact label match so we don't
    // collide with anything that has "session" as a substring.
    await gotoTab(/^Sessions & Tokens$/);

    // Sanity: section headings come through.
    expect(await screen.findByText(/^Browser session$/)).toBeInTheDocument();
    expect(screen.getByText(/^Token defaults$/)).toBeInTheDocument();
    expect(screen.getByText(/^Streaming$/)).toBeInTheDocument();

    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const ttlInput = await screen.findByLabelText(/Session access TTL/i);
    fireEvent.change(ttlInput, { target: { value: '30' } });
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => {
      const status = screen.getByRole('status');
      expect(status.textContent).toMatch(/warden process must be restarted/i);
    });
  });
});

describe('SettingsPage — Maintenance tab', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders vLLM runtime + Logs sections', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    await gotoTab(/^Maintenance$/);

    expect(await screen.findByText(/^vLLM runtime$/)).toBeInTheDocument();
    expect(screen.getByText(/^Logs$/)).toBeInTheDocument();
    expect(screen.getByText('vLLM version')).toBeInTheDocument();
    expect(screen.getByText('Log retention')).toBeInTheDocument();
  });
});

describe('SettingsPage — Model tab', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows empty-state copy + dropdown when no model is loaded', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/models': () =>
        new Response(
          JSON.stringify({
            models: [
              fakeModel({ id: 'm1', served_model_name: 'llama3-8b', status: 'registered' }),
              fakeModel({ id: 'm2', served_model_name: 'mixtral', status: 'pulled' }),
            ],
          }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    await gotoTab(/^Model$/);

    expect(
      await screen.findByText(/no model is currently loaded/i),
    ).toBeInTheDocument();
    const trigger = screen.getByRole('button', { name: /select a model to edit/i });
    expect(trigger).toBeInTheDocument();
    fireEvent.click(trigger);
    const listbox = await screen.findByRole('listbox');
    expect(within(listbox).getByText('llama3-8b')).toBeInTheDocument();
    expect(within(listbox).getByText('mixtral')).toBeInTheDocument();
  });

  it('shows a link to /models/<id>/settings when a model is loaded', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'GET /api/models': () =>
        new Response(
          JSON.stringify({
            models: [
              fakeModel({ id: 'm1', served_model_name: 'llama3-8b', status: 'loaded' }),
              fakeModel({ id: 'm2', served_model_name: 'mixtral', status: 'pulled' }),
            ],
          }),
          { status: 200 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderPage();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    await gotoTab(/^Model$/);

    const link = await screen.findByRole('link', { name: /edit settings/i });
    expect(link).toHaveAttribute('href', '/models/m1/settings');
  });
});
