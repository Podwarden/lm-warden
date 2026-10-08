// Verify on the Connect page (plan §5.2, §5.4): browser-side, with a
// plaintext the operator holds; credentials omitted; the result line never
// echoes the key; the route comes from the router's decision log.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { buildVerifyRequest } from '@/lib/connect';
import { callsTo, json, stubFetch, TOKENS_URL, decision } from './router-test-utils';
import { CLAUDE_CODE, CONNECT_URL, OPENCODE, ROUTER_OFF, TOKENS_LIVE, connectOut } from './connect-test-utils';

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }));

vi.mock('next/navigation', () => ({
  usePathname: () => '/connect',
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => nav.params,
}));

vi.mock('@/components/tokens/create-token-dialog', () => ({
  CreateTokenDialog: (p: { open: boolean; onCreated?: (k: string, i: { anthropicRelay: boolean }) => void; onClose: () => void }) =>
    p.open ? (
      <div role="dialog" aria-label="Create token">
        <button type="button" onClick={() => { p.onCreated?.('vw_created_abcdef123456', { anthropicRelay: true }); p.onClose(); }}>
          Fake create
        </button>
      </div>
    ) : null,
}));

import ConnectPage from '@/app/connect/page';

const KEY = 'vw_pasted_secret_9876543210';
const ORIGIN = () => window.location.origin;
const MESSAGES = () => `POST ${ORIGIN()}/v1/messages`;
const CHAT = () => `POST ${ORIGIN()}/v1/chat/completions`;
const DECISIONS = 'GET /api/router/decisions?limit=1';

const ANTHROPIC_OK = { id: 'msg_1', type: 'message', model: 'qwen3-4b-dp4', content: [{ type: 'text', text: 'OK' }] };
const OPENAI_OK = { id: 'c1', model: 'qwen3-4b-dp4', choices: [{ index: 0, message: { role: 'assistant', content: 'OK' } }] };

type Routes = Record<string, (init?: RequestInit) => Response>;

function routes(over: Routes = {}): Routes {
  return {
    [`GET ${CONNECT_URL}`]: () => json(connectOut()),
    [`GET ${TOKENS_URL}`]: () => json(TOKENS_LIVE),
    'GET /api/settings/runtime': () => json({ public_url: 'https://warden.example' }),
    [DECISIONS]: () => json({ decisions: [decision({ model_in: 'claude-haiku-4-5', route: 'local' })] }),
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

const button = () => screen.getByRole('button', { name: 'Send a test request' });
const status = () => screen.getByTestId('connect-verify-status');
const alertEl = () => screen.getByTestId('connect-verify-alert');

async function paste(value = KEY) {
  const input = await screen.findByLabelText('Paste the key to test it');
  fireEvent.change(input, { target: { value } });
  return input;
}

describe('Connect page: Verify', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    nav.params = new URLSearchParams();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('is disabled without a plaintext and says why; the masked input is described as tab-only', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes());
    renderPage();
    await screen.findByTestId('connect-panel');
    expect(button()).toBeDisabled();
    expect(screen.getByText(/Paste or create a key to send a test request/)).toBeInTheDocument();
    const input = screen.getByLabelText('Paste the key to test it');
    expect(input).toHaveAttribute('type', 'password');
    expect(input).toHaveAttribute('autocomplete', 'off');
    expect(input).toHaveAccessibleDescription(/kept in this tab only/i);
    // Both result regions exist from the first render, empty.
    expect(status()).toHaveAttribute('role', 'status');
    expect(alertEl()).toHaveAttribute('role', 'alert');
    expect(status()).toBeEmptyDOMElement();
  });

  it('a pasted key enables it and sends exactly buildVerifyRequest, cookies omitted, no admin JWT', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    const fn = stubFetch(routes({ [CHAT()]: () => json(OPENAI_OK) }));
    renderPage();
    await screen.findByTestId('connect-panel');
    await paste();
    expect(button()).toBeEnabled();
    fireEvent.click(button());
    await waitFor(() => expect(callsTo(fn, CHAT())).toHaveLength(1));
    const [url, init] = callsTo(fn, CHAT())[0] as [string, RequestInit];
    const expected = buildVerifyRequest(OPENCODE.modes[0].verify!, {
      key: KEY, model: 'qwen3-4b-dp4', verify_model: 'qwen3-4b-dp4', header: 'X-LMWarden-Key', context: 65536,
    });
    expected.done();
    expect(url).toBe(expected.url);
    expect(url).not.toContain(KEY);
    expect(init.method).toBe('POST');
    expect(init.credentials).toBe('omit');
    expect(init.headers).toEqual(expected.init.headers);
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${KEY}`);
    expect(JSON.parse(init.body as string)).toEqual(JSON.parse(expected.init.body as string));
    await waitFor(() => expect(status()).toHaveTextContent(/^200 in \d+ ms · answered by qwen3-4b-dp4 · 'OK'$/));
    // Focus stays on the button.
    expect(alertEl()).toBeEmptyDOMElement();
    // The OpenAI protocol never reads the router's decision log.
    expect(callsTo(fn, DECISIONS)).toHaveLength(0);
  });

  it('Anthropic + router on: router-mode request uses the rule example model and appends the route', async () => {
    const fn = stubFetch(routes({ [MESSAGES()]: () => json(ANTHROPIC_OK) }));
    renderPage();
    await screen.findByTestId('connect-panel');
    await paste();
    fireEvent.click(button());
    await waitFor(() => expect(status()).toHaveTextContent(/answered by qwen3-4b-dp4 · 'OK' · route: local on qwen$/));
    const [, init] = callsTo(fn, MESSAGES())[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers['X-LMWarden-Key']).toBe(KEY);
    expect(headers.Authorization).toBeUndefined();
    expect(JSON.parse(init.body as string).model).toBe('claude-haiku-4-5');
    const verify = CLAUDE_CODE.modes[0].verify!;
    expect(init.credentials).toBe('omit');
    expect(verify.path).toBe('/v1/messages');
    // Only one decision fetch, through the admin API.
    expect(callsTo(fn, DECISIONS)).toHaveLength(1);
  });

  it('a fallback in the decision log is reported as such', async () => {
    stubFetch(routes({
      [MESSAGES()]: () => json(ANTHROPIC_OK),
      [DECISIONS]: () => json({ decisions: [decision({ model_in: 'claude-haiku-4-5', route: 'fallback', reason: 'status_400' })] }),
    }));
    renderPage();
    await screen.findByTestId('connect-panel');
    await paste();
    fireEvent.click(button());
    await waitFor(() => expect(status()).toHaveTextContent(/· fell back to Anthropic \(status_400\)$/));
  });

  it('router mode with routing off: disabled with the reason', async () => {
    stubFetch(routes({ [`GET ${CONNECT_URL}`]: () => json(connectOut({ router: ROUTER_OFF })) }));
    renderPage();
    await screen.findByTestId('connect-panel');
    await paste();
    expect(button()).toBeDisabled();
    expect(screen.getByText(/Routing is off, so there is nothing to test in router mode/)).toBeInTheDocument();
  });

  it('a just-created key enables Verify without pasting', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes({ [CHAT()]: () => json(OPENAI_OK) }));
    renderPage();
    await screen.findByTestId('connect-panel');
    fireEvent.change(await screen.findByLabelText('Key'), { target: { value: '__create' } });
    fireEvent.click(screen.getByRole('button', { name: 'Fake create' }));
    expect(button()).toBeEnabled();
    expect(screen.queryByLabelText('Paste the key to test it')).toBeNull();
  });

  const failures: [string, () => Response, RegExp][] = [
    ['401', () => json({ error: { message: 'bad key' } }, 401), /Key not accepted: unknown, expired or revoked\./],
    ['403 relay', () => json({ error: { type: 'relay_not_allowed', message: 'relay_not_allowed' } }, 403), /may not relay to Anthropic/],
    ['403 allow list', () => json({ detail: 'token_not_allowed' }, 403), /model allow list excludes this model/],
    ['404', () => json({ error: { message: 'nope' } }, 404), /Model qwen3-4b-dp4 is not served here/],
    ['529', () => json({ error: { message: 'engine down' } }, 529), /The local model failed: engine down/],
  ];
  for (const [name, res, text] of failures) {
    it(`failure ${name}: one sentence in the alert region, never the key`, async () => {
      nav.params = new URLSearchParams('tool=opencode');
      stubFetch(routes({ [CHAT()]: res }));
      renderPage();
      await screen.findByTestId('connect-panel');
      await paste();
      fireEvent.click(button());
      await waitFor(() => expect(alertEl()).toHaveTextContent(text));
      expect(status()).toBeEmptyDOMElement();
      expect(document.body.textContent).not.toContain(KEY);
    });
  }

  it('network error: cannot reach', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes({ [CHAT()]: () => { throw new TypeError('Failed to fetch'); } }));
    renderPage();
    await screen.findByTestId('connect-panel');
    await paste();
    fireEvent.click(button());
    await waitFor(() => expect(alertEl()).toHaveTextContent(/Couldn't reach .* from this tab/));
  });

  it('no answer in 15 s: aborts and says so', async () => {
    nav.params = new URLSearchParams('tool=opencode');
    stubFetch(routes({
      [CHAT()]: (init) =>
        new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
        }) as unknown as Response,
    }));
    renderPage();
    await screen.findByTestId('connect-panel');
    await paste();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const b = button();
    fireEvent.click(b);
    expect(b).toHaveAttribute('aria-busy', 'true');
    expect(b).toHaveTextContent('Sending…');
    await act(async () => {
      vi.advanceTimersByTime(15_001);
    });
    await waitFor(() => expect(alertEl()).toHaveTextContent('No answer in 15 s.'));
  });

  it('prints what the test covers and what it does not', async () => {
    stubFetch(routes());
    renderPage();
    await screen.findByTestId('connect-panel');
    expect(screen.getByText(/A rule-matched Claude model served locally with the key in X-LMWarden-Key\./)).toBeInTheDocument();
    expect(screen.getByText(/Pass-through to Anthropic needs your Claude login\./)).toBeInTheDocument();
  });
});
