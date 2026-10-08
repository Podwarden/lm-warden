// The Connect a client page (plan 2026-10-04-connect-clients §3, §5, §7):
// tool list, key and model pickers, snippets, strip states and an
// unsupported entry. Verify has its own file (connect-verify.test.tsx).

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, within } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { json, stubFetch, TOKENS_URL } from './router-test-utils';
import {
  CONNECT_URL,
  MODEL_SMALL,
  ROUTER_OFF,
  TOKENS_DEAD,
  TOKENS_EMPTY,
  TOKENS_LIVE,
  connectOut,
  tokenPage,
  token,
} from './connect-test-utils';

const nav = vi.hoisted(() => ({
  params: new URLSearchParams(),
  replace: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  usePathname: () => '/connect',
  useRouter: () => ({ replace: nav.replace, push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => nav.params,
}));

// The real dialog posts to /api/tokens; the page only cares about its props
// and the plaintext it hands back.
vi.mock('@/components/tokens/create-token-dialog', () => ({
  CreateTokenDialog: (p: {
    open: boolean;
    initialName?: string;
    initialRelay?: boolean;
    onClose: () => void;
    onCreated?: (plaintext: string, info: { anthropicRelay: boolean }) => void;
  }) =>
    p.open ? (
      <div role="dialog" aria-label="Create token">
        <span data-testid="dlg-name">{p.initialName}</span>
        <span data-testid="dlg-relay">{String(!!p.initialRelay)}</span>
        <button
          type="button"
          onClick={() => {
            p.onCreated?.('vw_newsecret_0123456789', { anthropicRelay: !!p.initialRelay });
          }}
        >
          Fake create
        </button>
        <button type="button" onClick={p.onClose}>
          Fake close
        </button>
      </div>
    ) : null,
}));

import ConnectPage from '@/app/connect/page';

type Routes = Record<string, () => Response>;

function routes(over: Routes = {}): Routes {
  return {
    [`GET ${CONNECT_URL}`]: () => json(connectOut()),
    [`GET ${TOKENS_URL}`]: () => json(TOKENS_LIVE),
    'GET /api/settings/runtime': () => json({ public_url: 'https://warden.example' }),
    ...over,
  };
}

function renderPage() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ConnectPage />
    </SWRConfig>,
  );
}

async function ready() {
  return screen.findByTestId('connect-panel');
}

const snippet = () => screen.getByTestId('connect-snippet');
const tool = (name: RegExp) => screen.getByRole('tab', { name });

describe('Connect page', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    nav.params = new URLSearchParams();
    nav.replace.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders one h1 and the tool list in catalogue order, grouped, with support levels', async () => {
    stubFetch(routes());
    renderPage();
    await ready();
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Connect a client');
    const list = screen.getByRole('tablist', { name: 'Tools' });
    const tabs = within(list).getAllByRole('tab');
    expect(tabs.map((t) => t.getAttribute('data-tool'))).toEqual(['claude-code', 'opencode', 'continue', 'codex-cli']);
    // The support level is part of each tab's name (never colour alone).
    expect(tabs[0]).toHaveAccessibleName(/Claude Code.*Full/);
    expect(tabs[1]).toHaveAccessibleName(/OpenCode.*Local models only/);
    expect(tabs[2]).toHaveAccessibleName(/Continue.*Documented, not run/);
    expect(tabs[3]).toHaveAccessibleName(/Codex CLI.*Not supported/);
    // Group labels, in order.
    const labels = within(screen.getByTestId('connect-tools')).getAllByTestId('connect-group').map((g) => g.textContent);
    expect(labels).toEqual(['Anthropic protocol', 'OpenAI protocol', 'Editors', 'Not supported']);
    // Claude Code is selected by default and its panel is labelled by its tab.
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');
    const panel = screen.getByTestId('connect-panel');
    expect(panel).toHaveAttribute('aria-labelledby', tabs[0].id);
    expect(within(panel).getByRole('heading', { level: 2 })).toHaveTextContent('Claude Code');
  });

  it('?tool=opencode selects it; arrow keys move the selection and mirror the URL with replace', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes());
    renderPage();
    await ready();
    expect(tool(/OpenCode/)).toHaveAttribute('aria-selected', 'true');
    expect(tool(/OpenCode/)).toHaveAttribute('tabindex', '0');
    expect(tool(/Claude Code/)).toHaveAttribute('tabindex', '-1');
    fireEvent.keyDown(tool(/OpenCode/), { key: 'ArrowDown' });
    expect(tool(/Continue/)).toHaveAttribute('aria-selected', 'true');
    expect(document.activeElement).toBe(tool(/Continue/));
    expect(nav.replace).toHaveBeenLastCalledWith('/connect?tool=continue', { scroll: false });
    fireEvent.keyDown(tool(/Continue/), { key: 'End' });
    expect(tool(/Codex CLI/)).toHaveAttribute('aria-selected', 'true');
    fireEvent.keyDown(tool(/Codex CLI/), { key: 'Home' });
    expect(tool(/Claude Code/)).toHaveAttribute('aria-selected', 'true');
    fireEvent.keyDown(tool(/Claude Code/), { key: 'ArrowUp' });
    expect(tool(/Codex CLI/)).toHaveAttribute('aria-selected', 'true');
  });

  it('an unknown ?tool= falls back to the first tool and corrects the URL', async () => {
    nav.params = new URLSearchParams('tool=nope');
    stubFetch(routes());
    renderPage();
    await ready();
    expect(tool(/Claude Code/)).toHaveAttribute('aria-selected', 'true');
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith('/connect?tool=claude-code&mode=router', { scroll: false }));
  });

  it('key picker: live keys by name and prefix, dead keys disabled with their status, and Create a key…', async () => {
    stubFetch(routes({ [`GET ${TOKENS_URL}`]: () => json(tokenPage([...TOKENS_LIVE.items, ...TOKENS_DEAD.items])) }));
    renderPage();
    await ready();
    const select = await screen.findByLabelText('Key');
    await waitFor(() => expect(within(select).getByRole('option', { name: /laptop/ })).toBeInTheDocument());
    expect(within(select).getByRole('option', { name: 'laptop (vw_abcd…)' })).not.toBeDisabled();
    expect(within(select).getByRole('option', { name: /old .* revoked/ })).toBeDisabled();
    expect(within(select).getByRole('option', { name: /stale .* expired/ })).toBeDisabled();
    expect(within(select).getByRole('option', { name: /held .* paused/ })).toBeDisabled();
    expect(within(select).getByRole('option', { name: 'Create a key…' })).toBeInTheDocument();
  });

  it('picking an existing key keeps the placeholder and names the key in a comment', async () => {
    nav.params = new URLSearchParams('tool=claude-code&mode=local');
    stubFetch(routes());
    renderPage();
    await ready();
    const select = await screen.findByLabelText('Key');
    await waitFor(() => expect(within(select).getByRole('option', { name: /laptop/ })).toBeInTheDocument());
    fireEvent.change(select, { target: { value: 't1' } });
    expect(snippet()).toHaveTextContent('# key: laptop (vw_abcd…)');
    expect(snippet()).toHaveTextContent('ANTHROPIC_AUTH_TOKEN=vw_YOUR_KEY');
    expect(screen.getByText(/keeps only a hash of each key/)).toBeInTheDocument();
  });

  it('Create a key… opens the dialog named for the tool, with relay only in Claude Code router mode', async () => {
    stubFetch(routes());
    renderPage();
    await ready();
    const select = await screen.findByLabelText('Key');
    fireEvent.change(select, { target: { value: '__create' } });
    expect(screen.getByTestId('dlg-name')).toHaveTextContent('claude-code');
    expect(screen.getByTestId('dlg-relay')).toHaveTextContent('true');
    fireEvent.click(screen.getByRole('button', { name: 'Fake close' }));
    // The select did not stay on the action.
    expect(select).not.toHaveValue('__create');

    fireEvent.click(tool(/OpenCode/));
    fireEvent.change(screen.getByLabelText('Key'), { target: { value: '__create' } });
    expect(screen.getByTestId('dlg-name')).toHaveTextContent('opencode');
    expect(screen.getByTestId('dlg-relay')).toHaveTextContent('false');
  });

  it('a created key fills every snippet with the "Shown once" line; Forget key restores the placeholder', async () => {
    stubFetch(routes());
    renderPage();
    await ready();
    fireEvent.change(await screen.findByLabelText('Key'), { target: { value: '__create' } });
    fireEvent.click(screen.getByRole('button', { name: 'Fake create' }));
    fireEvent.click(screen.getByRole('button', { name: 'Fake close' }));
    expect(snippet()).toHaveTextContent('X-LMWarden-Key: vw_newsecret_0123456789');
    expect(screen.getByText(/Shown once — copy it now/)).toBeInTheDocument();
    // Another tool's snippet uses it too.
    fireEvent.click(tool(/OpenCode/));
    expect(snippet()).toHaveTextContent('"apiKey":"vw_newsecret_0123456789"');
    fireEvent.click(screen.getByRole('button', { name: 'Forget key' }));
    expect(snippet()).toHaveTextContent('"apiKey":"vw_YOUR_KEY"');
    expect(screen.queryByText(/vw_newsecret/)).toBeNull();
  });

  it('after the create dialog closes, focus lands on the key picker', async () => {
    stubFetch(routes());
    renderPage();
    await ready();
    const select = await screen.findByLabelText('Key');
    fireEvent.change(select, { target: { value: '__create' } });
    const close = screen.getByRole('button', { name: 'Fake close' });
    close.focus();
    fireEvent.click(close);
    await waitFor(() => expect(document.activeElement).toBe(screen.getByLabelText('Key')));
  });

  it('model picker: loaded first, not-loaded disabled, context and tools in the label, too-small marked', async () => {
    nav.params = new URLSearchParams('tool=claude-code&mode=local');
    stubFetch(routes());
    renderPage();
    await ready();
    const select = screen.getByLabelText('Model');
    const opts = within(select).getAllByRole('option');
    expect(opts.map((o) => o.textContent)).toEqual([
      'qwen3-4b-dp4 · 65,536 tokens · tools',
      'mystery · context unknown · tools',
      'tiny-8k · 8,192 tokens · too small for Claude Code',
      'llama-70b (not loaded)',
    ]);
    expect(opts[3]).toBeDisabled();
    expect(select).toHaveValue('qwen3-4b-dp4');
    fireEvent.change(select, { target: { value: 'tiny-8k' } });
    expect(snippet()).toHaveTextContent('ANTHROPIC_MODEL=tiny-8k');
    expect(snippet()).toHaveTextContent('CLAUDE_CODE_MAX_CONTEXT_TOKENS=8192');
    const row = screen.getByTestId('req-min_context');
    expect(row).toHaveAttribute('data-state', 'warn');
    expect(row).toHaveTextContent('tiny-8k: 8,192 tokens (needs ≥ 49,152)');
    expect(row).toHaveTextContent(/below/);
  });

  it('router mode hides the model picker and lists the rules instead', async () => {
    stubFetch(routes());
    renderPage();
    await ready();
    expect(screen.queryByLabelText('Model')).toBeNull();
    const rules = screen.getByTestId('connect-rules');
    expect(rules).toHaveTextContent('claude-haiku*');
    expect(rules).toHaveTextContent('qwen3-4b-dp4');
    // Mode tabs switch to local and the picker returns.
    fireEvent.click(screen.getByRole('tab', { name: /Local only/ }));
    expect(screen.getByLabelText('Model')).toBeInTheDocument();
    expect(nav.replace).toHaveBeenLastCalledWith('/connect?tool=claude-code&mode=local', { scroll: false });
  });

  it('snippet: each file tab renders the template with the public URL and the picks; Copy writes it', async () => {
    const write = vi.fn(async () => {});
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText: write } });
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true });
    nav.params = new URLSearchParams('tool=continue');
    stubFetch(routes());
    renderPage();
    await ready();
    // qwen supports tools → the tools variant only.
    const files = screen.getByRole('tablist', { name: 'File' });
    expect(within(files).getAllByRole('tab')).toHaveLength(1);
    expect(snippet()).toHaveTextContent('capabilities: [tool_use]');
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'tiny-8k' } });
    expect(snippet()).not.toHaveTextContent('tool_use');
    expect(snippet()).toHaveTextContent('model: tiny-8k');

    fireEvent.click(tool(/OpenCode/));
    expect(snippet()).toHaveTextContent('"baseURL":"https://warden.example/v1"');
    fireEvent.click(screen.getByRole('button', { name: 'Copy opencode.json' }));
    await waitFor(() => expect(write).toHaveBeenCalledWith(snippet().textContent));
    expect(await screen.findByText('Copied')).toBeInTheDocument();
  });

  it('snippet: a failed copy says so', async () => {
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText: vi.fn(async () => { throw new Error('no'); }) } });
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true });
    document.execCommand = vi.fn(() => false) as unknown as typeof document.execCommand;
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes());
    renderPage();
    await ready();
    fireEvent.click(screen.getByRole('button', { name: 'Copy opencode.json' }));
    expect(await screen.findByText('Copy failed — select the text.')).toBeInTheDocument();
  });

  it('public_url unset: snippets use this tab and say so', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes({ 'GET /api/settings/runtime': () => json({ public_url: null }) }));
    renderPage();
    await ready();
    await waitFor(() => expect(snippet()).toHaveTextContent(`"baseURL":"${window.location.origin}/v1"`));
    expect(screen.getByText(/Using this tab's address/)).toBeInTheDocument();
  });

  it('an unsupported client shows the reason and neither snippet nor Verify', async () => {
    nav.params = new URLSearchParams('tool=codex-cli');
    stubFetch(routes());
    renderPage();
    await ready();
    const panel = screen.getByTestId('connect-panel');
    expect(within(panel).getByText(/cannot use LM Warden today/)).toBeInTheDocument();
    expect(within(panel).getByText('Not supported')).toBeInTheDocument();
    expect(screen.queryByTestId('connect-snippet')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Send a test request' })).toBeNull();
    expect(within(panel).getByRole('link', { name: /docs/i })).toHaveAttribute('href', 'https://learn.chatgpt.com/docs/config-file/config-reference');
  });

  it('the verified line and docs link', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes());
    renderPage();
    await ready();
    expect(screen.getByText('Verified 2026-10-04 with opencode 1.2.3')).toBeInTheDocument();
    nav.params = new URLSearchParams('tool=continue');
    fireEvent.click(tool(/Continue/));
    expect(screen.getByText(/Documented 2026-10-04, not run/)).toBeInTheDocument();
  });

  it('a run with a note shows the note after the verified line', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    const base = connectOut();
    const clients = base.clients.map((c) =>
      c.id === 'opencode' && c.verified ? { ...c, verified: { ...c.verified, note: 'Tool calls worked.' } } : c,
    );
    stubFetch(routes({ [`GET ${CONNECT_URL}`]: () => json(connectOut({ clients })) }));
    renderPage();
    await ready();
    expect(screen.getByText('Verified 2026-10-04 with opencode 1.2.3. Tool calls worked.')).toBeInTheDocument();
  });

  describe('strip', () => {
    it('clients fetch failed: danger alert with Retry', async () => {
      const fn = stubFetch(routes({ [`GET ${CONNECT_URL}`]: () => json({ detail: 'boom' }, 500) }));
      renderPage();
      const alert = await screen.findByText(/Couldn't load the client catalogue/);
      expect(alert.closest('[role="alert"]')).not.toBeNull();
      const before = fn.mock.calls.length;
      fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
      await waitFor(() => expect(fn.mock.calls.length).toBeGreaterThan(before));
    });

    it('no loaded model: amber alert linking to Models', async () => {
      nav.params = new URLSearchParams('tool=opencode');
      stubFetch(routes({ [`GET ${CONNECT_URL}`]: () => json(connectOut({ models: [], selected_model: null })) }));
      renderPage();
      await ready();
      const s = await screen.findByText(/No model is loaded/);
      expect(s.closest('[role="alert"]')).not.toBeNull();
      expect(screen.getByRole('link', { name: 'Models' })).toHaveAttribute('href', '/models');
      expect(screen.getByLabelText('Model')).toBeDisabled();
    });

    it('no live key: a status with Create a key', async () => {
      stubFetch(routes({ [`GET ${TOKENS_URL}`]: () => json(TOKENS_EMPTY) }));
      renderPage();
      await ready();
      const s = await screen.findByText(/You have no API key yet/);
      expect(s.closest('[role="status"]')).not.toBeNull();
      fireEvent.click(within(screen.getByTestId('connect-strip')).getByRole('button', { name: 'Create a key' }));
      expect(screen.getByRole('dialog', { name: 'Create token' })).toBeInTheDocument();
    });

    it('tokens fetch failed: a status with Retry; snippets keep the placeholder', async () => {
      stubFetch(routes({ [`GET ${TOKENS_URL}`]: () => json({ detail: 'x' }, 500) }));
      renderPage();
      await ready();
      expect(await screen.findByText(/Couldn't load your keys/)).toBeInTheDocument();
      expect(snippet()).toHaveTextContent('vw_YOUR_KEY');
    });

    it('ok: no strip content', async () => {
      stubFetch(routes());
      renderPage();
      await ready();
      await waitFor(() => expect(within(screen.getByLabelText('Key')).getByRole('option', { name: /laptop/ })).toBeInTheDocument());
      expect(screen.getByTestId('connect-strip')).toHaveTextContent('');
    });
  });

  it('requirement rows carry their state in text', async () => {
    stubFetch(routes({ [`GET ${CONNECT_URL}`]: () => json(connectOut({ router: ROUTER_OFF })) }));
    renderPage();
    await ready();
    const row = screen.getByTestId('req-router_on');
    expect(row).toHaveAttribute('data-state', 'fail');
    expect(row).toHaveTextContent(/not met/);
    expect(row).toHaveTextContent('routing is off');
  });

  it('the paused key list caps at 500 and says so', async () => {
    const items = Array.from({ length: 3 }, (_, i) => token({ id: `k${i}`, name: `k${i}` }));
    stubFetch(routes({ [`GET ${TOKENS_URL}`]: () => json(tokenPage(items, 812)) }));
    renderPage();
    await ready();
    expect(await screen.findByText(/first 500 shown/)).toBeInTheDocument();
  });

  it('a small model in OpenCode marks the tools row failed', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes());
    renderPage();
    await ready();
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: MODEL_SMALL.served_name } });
    expect(screen.getByTestId('req-tools')).toHaveAttribute('data-state', 'fail');
  });
});
