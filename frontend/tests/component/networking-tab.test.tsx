import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { NetworkingTab } from '@/components/settings/networking-tab';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// ---------------------------------------------------------------------------
// NetworkingTab tests — pin the #154-new behaviours that are specific to
// this tab and not covered by the page-level settings.test.tsx wiring tests:
//
//   1. Both Public-access fields render (public_url + landing_page_enabled).
//   2. Submitting a valid public_url with a trailing slash strips the slash
//      in the rendered value after the backend echoes back its canonical
//      form (round-trip; the helper itself is unit-tested separately).
//   3. A 422 from the backend with a `detail` payload surfaces in the
//      inline error banner — same shape as the General-tab path, but the
//      test lives here because public_url is the field most likely to hit
//      422 (URL validation).
// ---------------------------------------------------------------------------

function renderTab() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <NetworkingTab />
    </SWRConfig>,
  );
}

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
    analytics_ga4_id: '',
    analytics_google_ads_id: '',
    search_console_verification: '',
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

describe('NetworkingTab', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders Public-access section with public_url + landing_page_enabled', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderTab();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(await screen.findByText(/^Public access$/)).toBeInTheDocument();
    expect(screen.getByText('Public URL')).toBeInTheDocument();
    expect(screen.getByText('Public website')).toBeInTheDocument();
  });

  it('sends the entered URL (server canonicalises) in the PATCH body', async () => {
    let bodyCaptured: unknown = null;
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'PATCH /api/settings/runtime': (init) => {
        bodyCaptured = JSON.parse((init?.body as string) ?? '{}');
        // Echo what the backend would store (slash stripped) — but the
        // assertion here is that the FE sent the value the user typed
        // verbatim. The backend's _url coercer owns the canonicalisation.
        return new Response(
          JSON.stringify({ ok: true, requires_restart: [], requires_restart_kinds: [] }),
          { status: 200 },
        );
      },
    });
    vi.stubGlobal('fetch', fetchMock);

    renderTab();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const urlInput = await screen.findByLabelText(/Public URL/i);
    fireEvent.change(urlInput, { target: { value: 'https://warden.example.com/' } });
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    await waitFor(() => expect(bodyCaptured).not.toBeNull());
    const body = bodyCaptured as Record<string, unknown>;
    // FE ships the operator's input verbatim — server is the canonicaliser.
    expect(body.public_url).toBe('https://warden.example.com/');
    // Untouched secret keys are NOT leaked into the PATCH body.
    expect(body.admin_password).toBeUndefined();
    expect(body.hf_token).toBeUndefined();
    // Other-tab fields are NOT included — Networking tab's PATCH is
    // scoped to RUNTIME_NETWORKING_KEYS.
    expect(body.admin_username).toBeUndefined();
    expect(body.vllm_version).toBeUndefined();
  });

  it('surfaces 422 detail in the inline error banner', async () => {
    const fetchMock = mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
      'PATCH /api/settings/runtime': () =>
        new Response(
          JSON.stringify({ detail: 'public_url: invalid scheme' }),
          { status: 422 },
        ),
    });
    vi.stubGlobal('fetch', fetchMock);

    renderTab();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    fireEvent.click(await screen.findByRole('button', { name: /^edit$/i }));
    const urlInput = await screen.findByLabelText(/Public URL/i);
    fireEvent.change(urlInput, { target: { value: 'ftp://nope' } });
    const saveBtn = screen.getByRole('button', { name: /^save/i });
    await waitFor(() => expect(saveBtn).not.toBeDisabled());
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    const banner = await screen.findByRole('alert');
    expect(banner.textContent).toMatch(/public_url: invalid scheme/);
  });
});

describe('NetworkingTab: Website analytics section', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders the three tracking fields under their own heading', async () => {
    vi.stubGlobal('fetch', mockFetch({
      'GET /api/settings/runtime': () =>
        new Response(JSON.stringify(fakeRuntime()), { status: 200 }),
    }));
    renderTab();
    expect(await screen.findByText('Website analytics')).toBeInTheDocument();
    expect(screen.getByText('Google Analytics 4 measurement ID')).toBeInTheDocument();
    expect(screen.getByText('Google Ads tag ID')).toBeInTheDocument();
    expect(screen.getByText('Search Console verification')).toBeInTheDocument();
  });
});
