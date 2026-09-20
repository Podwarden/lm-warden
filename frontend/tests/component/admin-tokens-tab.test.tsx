import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { AdminTokensTab } from '@/components/settings/admin-tokens-tab';
import SettingsPage from '@/app/settings/page';
import type { AdminAuditRow, AdminToken } from '@/lib/admin-tokens';

// Settings → Admin tokens (spec docs/superpowers/specs/2026-09-19-admin-tokens-design.md,
// "UI"): issue shows the plaintext once, then never; "Never" warns; revoke
// confirms; refresh shows the new plaintext; Activity pages; empty state.
//
// `peer_ip` (added after the brief was written): the socket peer, which the
// client cannot forge; `client_ip` honours X-Forwarded-For and can be
// forged. The activity panel shows `client_ip` and, when `peer_ip` differs,
// also surfaces it — see "shows the peer address when it differs from the
// client IP" below.

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

function token(over: Partial<AdminToken> = {}): AdminToken {
  return {
    id: 'tok-1', name: 'ci', prefix: 'vwa_k2x7', created_by: 'admin',
    created_at: '2026-09-19 10:00:00', last_used_at: null, expires_at: '2026-12-18 10:00:00',
    revoked_at: null, rotated_from: null, status: 'active', ...over,
  };
}

function auditRow(over: Partial<AdminAuditRow> = {}): AdminAuditRow {
  return {
    id: 1, ts: 1758300000.25, method: 'POST', path: '/api/models/{model_id}/load',
    status: 202, duration_ms: 12, client_ip: '198.51.100.7', peer_ip: '198.51.100.7',
    username: 'admin', ...over,
  };
}

type Handler = (init?: RequestInit) => Response;

function mockFetch(routes: Record<string, Handler>) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    const key = `${(init?.method ?? 'GET').toUpperCase()} ${url}`;
    const handler = routes[key];
    return handler ? handler(init) : new Response(`unmocked: ${key}`, { status: 404 });
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

const RUNTIME: Record<string, Handler> = { 'GET /api/settings/runtime': () => json({ public_url: '' }) };

function renderTab() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <AdminTokensTab />
    </SWRConfig>,
  );
}

const calls = (fn: ReturnType<typeof vi.fn>, key: string) =>
  fn.mock.calls.filter(([u, i]) => `${((i as RequestInit | undefined)?.method ?? 'GET').toUpperCase()} ${String(u)}` === key);

describe('Settings → Admin tokens', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('explains admin tokens and offers Issue when there are none', async () => {
    mockFetch({ ...RUNTIME, 'GET /api/admin-tokens': () => json({ items: [] }) });
    renderTab();
    expect(await screen.findByText(/No admin tokens yet/)).toBeInTheDocument();
    expect(screen.getByText(/same power as signing in/)).toBeInTheDocument();
    expect(screen.getByTestId('admin-intro-curl').textContent).toContain('/api/openapi.json');
    expect(screen.getByRole('button', { name: 'Issue token' })).toBeInTheDocument();
  });

  it('builds curl examples from the page origin, not public_url (I1)', async () => {
    // A leaked admin token can PATCH public_url (it is ordinary runtime
    // config to everyone except a session -- see the session-only backend
    // test). The curl examples next to a freshly revealed secret must not
    // follow that value: they're built from window.location.origin, the
    // host the operator is actually looking at.
    mockFetch({
      'GET /api/settings/runtime': () => json({ public_url: 'https://proxy.example' }),
      'GET /api/admin-tokens': () => json({ items: [] }),
      'POST /api/admin-tokens': () => json({ ...token(), plaintext: 'vwa_secret_value' }, 201),
    });
    renderTab();
    const introCurl = await screen.findByTestId('admin-intro-curl');
    expect(introCurl.textContent).toContain(window.location.origin);
    expect(introCurl.textContent).not.toContain('proxy.example');

    fireEvent.click(screen.getByRole('button', { name: 'Issue token' }));
    const dialog = await screen.findByRole('dialog');
    fireEvent.change(within(dialog).getByLabelText('Name'), { target: { value: 'ci' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Issue' }));
    const revealCurl = await screen.findByTestId('admin-token-curl');
    expect(revealCurl.textContent).toContain(window.location.origin);
    expect(revealCurl.textContent).not.toContain('proxy.example');
  });

  it('issues a token and shows the secret once, then never again', async () => {
    let body: unknown = null;
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [] }),
      'POST /api/admin-tokens': (init) => {
        body = JSON.parse(String(init?.body));
        return json({ ...token(), plaintext: 'vwa_secret_value' }, 201);
      },
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Issue token' }));
    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByRole('radio', { name: '90 days' })).toBeChecked();
    fireEvent.change(within(dialog).getByLabelText('Name'), { target: { value: '  ci  ' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Issue' }));

    expect(await screen.findByTestId('admin-token-plaintext')).toHaveTextContent('vwa_secret_value');
    expect(screen.getByText(/You will not see this again/)).toBeInTheDocument();
    expect(screen.getByTestId('admin-token-curl').textContent).toContain('Bearer $VW_ADMIN_TOKEN');
    expect(screen.getByTestId('admin-token-curl').textContent).not.toContain('vwa_secret_value');
    expect(body).toEqual({ name: 'ci', expires_in_days: 90 });

    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    await waitFor(() => expect(screen.queryByTestId('admin-token-plaintext')).toBeNull());
    fireEvent.click(screen.getByRole('button', { name: 'Issue token' }));
    const again = await screen.findByRole('dialog');
    expect(screen.queryByTestId('admin-token-plaintext')).toBeNull();
    expect(within(again).getByLabelText('Name')).toHaveValue('');
  });

  it('makes Never an explicit choice with a warning', async () => {
    let body: unknown = null;
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [] }),
      'POST /api/admin-tokens': (init) => {
        body = JSON.parse(String(init?.body));
        return json({ ...token({ expires_at: null }), plaintext: 'vwa_x' }, 201);
      },
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Issue token' }));
    const dialog = await screen.findByRole('dialog');
    expect(screen.queryByTestId('never-warning')).toBeNull();
    fireEvent.click(within(dialog).getByRole('radio', { name: 'Never' }));
    expect(screen.getByTestId('never-warning')).toHaveTextContent(/until someone revokes it/);
    fireEvent.click(within(dialog).getByRole('radio', { name: '30 days' }));
    expect(screen.queryByTestId('never-warning')).toBeNull();
    fireEvent.click(within(dialog).getByRole('radio', { name: 'Never' }));
    fireEvent.change(within(dialog).getByLabelText('Name'), { target: { value: 'forever' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Issue' }));
    await screen.findByTestId('admin-token-plaintext');
    expect(body).toEqual({ name: 'forever', expires_in_days: null });
  });

  it('lists tokens and dims the dead ones', async () => {
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({
        items: [
          token(),
          token({ id: 'tok-2', name: 'ci (old 1)', status: 'grace', revoked_at: '2026-09-19 11:00:00' }),
          token({ id: 'tok-3', name: 'old', status: 'revoked', revoked_at: '2026-09-10 10:00:00' }),
        ],
      }),
    });
    renderTab();
    const rows = await screen.findAllByTestId('admin-token-row');
    expect(rows.map((r) => r.getAttribute('data-dead'))).toEqual(['false', 'false', 'true']);
    expect(within(rows[0]).getByText('Active')).toBeInTheDocument();
    expect(within(rows[1]).getByText('Grace')).toBeInTheDocument();
    expect(within(rows[2]).getByText('Revoked')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh ci' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Refresh ci (old 1)' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Revoke ci (old 1)' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Revoke old' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Activity old' })).toBeEnabled();
  });

  it('revokes only after the confirm step', async () => {
    const fetchMock = mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [token()] }),
      'DELETE /api/admin-tokens/tok-1': () => new Response(null, { status: 204 }),
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Revoke ci' }));
    let dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent(/refused from now on/);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(calls(fetchMock, 'DELETE /api/admin-tokens/tok-1')).toHaveLength(0);

    fireEvent.click(screen.getByRole('button', { name: 'Revoke ci' }));
    dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Revoke' }));
    await waitFor(() => expect(calls(fetchMock, 'DELETE /api/admin-tokens/tok-1')).toHaveLength(1));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('warns in the revoke dialog while the predecessor is in its grace window (M3)', async () => {
    // Revoking a refreshed token does not end its predecessor's grace
    // window -- the two are separate rows.
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () =>
        json({
          items: [
            token({ rotated_from: 'tok-0' }),
            token({ id: 'tok-0', name: 'ci (old 1)', prefix: 'vwa_old0', status: 'grace' }),
          ],
        }),
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Revoke ci' }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent(/previous secret \(vwa_old0\) stays valid until its grace ends/);
  });

  it('does not warn when the predecessor is already dead', async () => {
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () =>
        json({
          items: [
            token({ rotated_from: 'tok-0' }),
            token({ id: 'tok-0', name: 'ci (old 1)', prefix: 'vwa_old0', status: 'revoked' }),
          ],
        }),
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Revoke ci' }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).not.toHaveTextContent(/previous secret/);
  });

  it('does not warn in the revoke dialog when the token has no predecessor', async () => {
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [token({ rotated_from: null })] }),
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Revoke ci' }));
    const dialog = await screen.findByRole('dialog');
    expect(dialog).not.toHaveTextContent(/previous secret stays valid/);
  });

  it('refresh sends the chosen grace and shows the new secret once', async () => {
    let body: unknown = null;
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [token()] }),
      'POST /api/admin-tokens/tok-1/rotate': (init) => {
        body = JSON.parse(String(init?.body));
        return json({ ...token({ id: 'tok-9', rotated_from: 'tok-1' }), plaintext: 'vwa_fresh' }, 201);
      },
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Refresh ci' }));
    const dialog = await screen.findByRole('dialog');
    const grace = within(dialog).getByLabelText('Grace period') as HTMLSelectElement;
    expect(grace.value).toBe('1');
    fireEvent.change(grace, { target: { value: '24' } });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Refresh' }));
    expect(await screen.findByTestId('admin-token-plaintext')).toHaveTextContent('vwa_fresh');
    expect(screen.getByText(/keeps working for 24 hours/)).toBeInTheDocument();
    expect(body).toEqual({ grace_hours: 24 });
  });

  it('pages the activity with Load more', async () => {
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [token()] }),
      'GET /api/admin-tokens/tok-1/audit?limit=50': () =>
        json({ items: [auditRow()], next_before: 1758300000.25 }),
      'GET /api/admin-tokens/tok-1/audit?limit=50&before=1758300000.25': () =>
        json({ items: [auditRow({ id: 2, ts: 1758290000, method: 'GET', path: '/api/models' })], next_before: null }),
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Activity ci' }));
    const dialog = await screen.findByRole('dialog');
    expect(await within(dialog).findByText('/api/models/{model_id}/load')).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Load more' }));
    expect(await within(dialog).findByText('/api/models')).toBeInTheDocument();
    expect(within(dialog).getAllByTestId('audit-row')).toHaveLength(2);
    await waitFor(() => expect(within(dialog).queryByRole('button', { name: 'Load more' })).toBeNull());
  });

  it('shows the peer address when it differs from the client IP', async () => {
    mockFetch({
      ...RUNTIME,
      'GET /api/admin-tokens': () => json({ items: [token()] }),
      'GET /api/admin-tokens/tok-1/audit?limit=50': () =>
        json({
          items: [auditRow({ client_ip: '203.0.113.5', peer_ip: '198.51.100.9' })],
          next_before: null,
        }),
    });
    renderTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Activity ci' }));
    const dialog = await screen.findByRole('dialog');
    const row = await within(dialog).findByTestId('audit-row');
    expect(within(row).getByText('203.0.113.5')).toBeInTheDocument();
    expect(within(row).getByText(/via 198\.51\.100\.9/)).toBeInTheDocument();
  });

  it('is the sixth settings tab', async () => {
    mockFetch({ ...RUNTIME, 'GET /api/admin-tokens': () => json({ items: [] }) });
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <SettingsPage />
      </SWRConfig>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Admin tokens' }));
    expect(await screen.findByRole('heading', { name: 'Admin tokens' })).toBeInTheDocument();
    expect(await screen.findByText(/No admin tokens yet/)).toBeInTheDocument();
  });
});
