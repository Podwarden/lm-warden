import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import RouterPage from '@/app/router/page';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { RULE_A, RULE_B, SETTINGS, json, stubFetch, callsTo, baseRoutes } from './router-test-utils';

vi.mock('next/navigation', () => ({
  usePathname: () => '/router',
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <RouterPage />
    </SWRConfig>,
  );
}

const body = (fn: ReturnType<typeof vi.fn>, key: string) =>
  JSON.parse((callsTo(fn, key)[0][1] as RequestInit).body as string);

async function openSettings() {
  const toggle = await screen.findByRole('button', { name: /Timeouts, breaker and upstream/ });
  expect(toggle).toHaveAttribute('aria-expanded', 'false');
  fireEvent.click(toggle);
  expect(toggle).toHaveAttribute('aria-expanded', 'true');
  return screen.getByTestId('router-settings');
}

describe('Router page', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders one h1, the purpose line, the master switch, the map slot, connect and settings', async () => {
    stubFetch(baseRoutes());
    renderPage();
    expect(await screen.findByTestId('router-rules')).toBeInTheDocument();
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Claude Code router');
    expect(screen.getByText(/The Claude models you choose run on your GPUs/)).toBeInTheDocument();
    expect(screen.getByTestId('router-enabled')).toHaveAttribute('role', 'switch');
    expect(screen.getByTestId('router-connect')).toBeInTheDocument();
    expect(screen.getByTestId('router-settings-section')).toBeInTheDocument();
    // The strip container exists from the first render (live region before content).
    expect(screen.getByTestId('router-strip')).toBeInTheDocument();
  });

  it('the old Status card is gone: no Save next to the switch, no pass-through checkbox', async () => {
    stubFetch(baseRoutes());
    renderPage();
    await screen.findByTestId('router-rules');
    expect(screen.queryByText('Pass unmatched models through to Anthropic')).not.toBeInTheDocument();
    expect(screen.queryByText('Router is off — every request is served locally by served name.')).not.toBeInTheDocument();
  });

  it('settings: collapsed summary of the current values', async () => {
    stubFetch(baseRoutes());
    renderPage();
    const toggle = await screen.findByRole('button', { name: /Timeouts, breaker and upstream/ });
    expect(toggle).toHaveTextContent(
      'first byte 30 s · non-stream 300 s · breaker 3 failures → 30 s · body 32 MB · api.anthropic.com',
    );
  });

  it('settings: Save is disabled with "No unsaved changes"; one edit PATCHes only that key', async () => {
    const fn = stubFetch(
      baseRoutes({ 'PATCH /api/router/settings': () => json({ ...SETTINGS, breaker_open_s: 120 }) }),
    );
    renderPage();
    const sec = await openSettings();
    expect(within(sec).getByText('No unsaved changes')).toBeInTheDocument();
    expect(within(sec).getByRole('button', { name: 'Save' })).toBeDisabled();
    fireEvent.change(within(sec).getByLabelText(/Stays open for \(s\)/), { target: { value: '120' } });
    expect(within(sec).getByText('1 unsaved change')).toBeInTheDocument();
    expect(within(sec).getByText('(unsaved)')).toBeInTheDocument();
    expect(sec).toHaveTextContent('Breaker stays open for 30 → 120 s');
    fireEvent.click(within(sec).getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(1));
    expect(body(fn, 'PATCH /api/router/settings')).toEqual({ breaker_open_s: 120 });
  });

  it('settings: Discard drops the draft', async () => {
    stubFetch(baseRoutes());
    renderPage();
    const sec = await openSettings();
    const field = within(sec).getByLabelText(/First-byte timeout/);
    fireEvent.change(field, { target: { value: '45' } });
    fireEvent.click(within(sec).getByRole('button', { name: 'Discard' }));
    expect(field).toHaveValue(30);
    expect(within(sec).getByText('No unsaved changes')).toBeInTheDocument();
  });

  it('settings: a 422 detail lands in the footer alert and the input is kept', async () => {
    stubFetch(
      baseRoutes({
        'PATCH /api/router/settings': () => json({ detail: 'upstream_url: must be https' }, 422),
      }),
    );
    renderPage();
    const sec = await openSettings();
    const url = within(sec).getByLabelText('Upstream URL');
    fireEvent.change(url, { target: { value: 'http://example.com' } });
    fireEvent.click(within(sec).getByRole('button', { name: 'Save' }));
    expect(await within(sec).findByRole('alert')).toHaveTextContent('upstream_url: must be https');
    expect(url).toHaveValue('http://example.com');
    await waitFor(() => expect(document.activeElement).toBe(url));
  });

  it('settings: an out-of-range value is flagged on blur and blocks Save with a reason', async () => {
    const fn = stubFetch(baseRoutes());
    renderPage();
    const sec = await openSettings();
    const field = within(sec).getByLabelText(/Failures before it opens/);
    fireEvent.change(field, { target: { value: '0' } });
    fireEvent.blur(field);
    expect(field).toHaveAttribute('aria-invalid', 'true');
    expect(within(sec).getByText('Between 1 and 100.')).toBeInTheDocument();
    expect(within(sec).getByRole('button', { name: 'Save' })).toBeDisabled();
    expect(sec).toHaveTextContent(/Fix 1 field to save/);
    expect(callsTo(fn, 'PATCH /api/router/settings')).toHaveLength(0);
  });

  it('rules or settings fetch failure: a section error with Retry', async () => {
    stubFetch(baseRoutes({ 'GET /api/router/rules': () => json({ detail: 'boom' }, 500) }));
    renderPage();
    const err = await screen.findByTestId('router-load-error');
    expect(within(err).getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('renders the rules slot with both fixtures (map contract: rules + settings + stats)', async () => {
    stubFetch(baseRoutes({ 'GET /api/router/rules': () => json({ rules: [RULE_A, RULE_B] }) }));
    renderPage();
    const map = await screen.findByTestId('router-rules');
    expect(map).toHaveTextContent('claude-haiku*');
    expect(map).toHaveTextContent('claude-sonnet-4-5');
    // The real map, not a stand-in: the pass-through rest row is part of it.
    expect(screen.getByTestId('router-rest')).toBeInTheDocument();
  });

  it("the map's Add rule opens the page-owned dialog, empty and alone", async () => {
    stubFetch(baseRoutes({ 'GET /api/router/rules': () => json({ rules: [RULE_A, RULE_B] }) }));
    renderPage();
    await screen.findByTestId('router-rules');
    fireEvent.click(screen.getByRole('button', { name: /^Add rule$/ }));
    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByLabelText('Pattern')).toHaveValue('');
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
  });
});
