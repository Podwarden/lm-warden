import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { ConnectClaudeCode } from '@/components/router/connect-claude-code';
import { NavBar } from '@/components/nav-bar';
import { envSnippet, settingsJsonSnippet, localOnlySnippet } from '@/lib/router';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import type { RelayInfo } from '@/lib/router';
import {
  RULE_A,
  RULE_B,
  SETTINGS,
  DECISIONS_HEALTHY,
  decision,
  json,
  stubFetch,
  baseRoutes,
} from './router-test-utils';

vi.mock('@/components/header-metrics', () => ({ HeaderMetrics: () => null }));
vi.mock('next/navigation', () => ({
  usePathname: () => '/models',
  useRouter: () => ({ replace: vi.fn() }),
}));

describe('lib/router snippets', () => {
  it('envSnippet is the exact three-line block', () => {
    expect(envSnippet('https://w.example', 'X-LMWarden-Key', 'vw_YOUR_KEY')).toBe(
      [
        'export ANTHROPIC_BASE_URL=https://w.example',
        'export ANTHROPIC_CUSTOM_HEADERS="X-LMWarden-Key: vw_YOUR_KEY"',
        'claude',
      ].join('\n'),
    );
  });
  it('settingsJsonSnippet is valid JSON with both env keys', () => {
    const j = JSON.parse(settingsJsonSnippet('https://w.example', 'X-LMWarden-Key', 'vw_K'));
    expect(j).toEqual({
      env: {
        ANTHROPIC_BASE_URL: 'https://w.example',
        ANTHROPIC_CUSTOM_HEADERS: 'X-LMWarden-Key: vw_K',
      },
    });
  });
  it('localOnlySnippet uses the documented local mode', () => {
    const s = localOnlySnippet('https://w.example', 'qwen');
    expect(s).toContain('export ANTHROPIC_BASE_URL=https://w.example');
    expect(s).toContain('export ANTHROPIC_AUTH_TOKEN=vw_YOUR_KEY');
    for (const k of ['MODEL', 'DEFAULT_OPUS_MODEL', 'DEFAULT_SONNET_MODEL', 'DEFAULT_HAIKU_MODEL']) {
      expect(s).toContain(`export ANTHROPIC_${k}=qwen`);
    }
  });
});

const NO_RELAY: RelayInfo = { relayKeys: 0, relayKnown: true, relayLoading: false };

function renderConnect(
  over: Partial<Parameters<typeof ConnectClaudeCode>[0]> = {},
) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ConnectClaudeCode
        settings={SETTINGS}
        rules={[RULE_A, RULE_B]}
        decisions={[]}
        relay={NO_RELAY}
        plaintext={null}
        defaultOpen
        onCreateRelayKey={() => {}}
        {...over}
      />
    </SWRConfig>,
  );
}

describe('ConnectClaudeCode', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    stubFetch(baseRoutes());
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders the shell block with public_url and header line', async () => {
    renderConnect();
    expect(await screen.findByText(/ANTHROPIC_BASE_URL=https:\/\/warden\.example/)).toBeInTheDocument();
    expect(screen.getByText(/X-LMWarden-Key: vw_YOUR_KEY/)).toBeInTheDocument();
    expect(screen.getByText(/May relay to Anthropic/)).toBeInTheDocument();
    expect(screen.getByText(/2\.1\.227/)).toBeInTheDocument();
  });

  it('settings.json tab shows valid JSON', async () => {
    renderConnect();
    await screen.findByText(/ANTHROPIC_BASE_URL=https:\/\/warden\.example/);
    fireEvent.click(screen.getByRole('tab', { name: 'settings.json' }));
    const pre = await screen.findByTestId('router-snippet');
    const j = JSON.parse(pre.textContent ?? '');
    expect(j.env.ANTHROPIC_BASE_URL).toBe('https://warden.example');
    expect(j.env.ANTHROPIC_CUSTOM_HEADERS).toBe('X-LMWarden-Key: vw_YOUR_KEY');
  });

  it('local-only tab names the first served model', async () => {
    renderConnect();
    await screen.findByText(/ANTHROPIC_BASE_URL=/);
    fireEvent.click(screen.getByRole('tab', { name: 'Local only (no Anthropic account)' }));
    expect((await screen.findByTestId('router-snippet')).textContent).toContain(
      'ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen',
    );
  });

  it('relay pill: "No relay key yet" (amber) vs "1 key may relay"', async () => {
    renderConnect();
    expect(await screen.findByText('No relay key yet')).toBeInTheDocument();
    cleanup();
    renderConnect({ relay: { relayKeys: 1, relayKnown: true, relayLoading: false } });
    expect(await screen.findByText('1 key may relay')).toBeInTheDocument();
  });

  it('a created key fills the snippet and is flagged as shown once', async () => {
    renderConnect({ plaintext: 'vw_fresh_key' });
    const pre = await screen.findByTestId('router-snippet');
    expect(pre.textContent).toContain('X-LMWarden-Key: vw_fresh_key');
    expect(screen.getByText(/Shown once — copy it now/)).toBeInTheDocument();
  });

  it('collapsed once traffic was seen, with a "Seen from N keys" summary', async () => {
    renderConnect({ defaultOpen: false, decisions: DECISIONS_HEALTHY.decisions });
    const toggle = screen.getByRole('button', { name: /Connect Claude Code/ });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveTextContent('Seen from 2 keys: laptop, ci-runner');
    expect(screen.queryByTestId('router-snippet')).not.toBeInTheDocument();
    fireEvent.click(toggle);
    expect(await screen.findByTestId('router-snippet')).toBeInTheDocument();
  });

  it('collapsed summary with no requests and one key', () => {
    renderConnect({ defaultOpen: false });
    expect(screen.getByRole('button', { name: /Connect Claude Code/ })).toHaveTextContent('No requests yet');
    cleanup();
    renderConnect({ defaultOpen: false, decisions: [decision()] });
    expect(screen.getByRole('button', { name: /Connect Claude Code/ })).toHaveTextContent('Seen from 1 key: laptop');
  });

  it('the overrides list and the off-state line are gone (the map and switch show them)', async () => {
    renderConnect();
    await screen.findByTestId('router-snippet');
    expect(screen.queryByTestId('router-overrides')).not.toBeInTheDocument();
    expect(screen.queryByText(/Router is off/)).not.toBeInTheDocument();
  });

  it('fine print names both env vars to avoid', async () => {
    renderConnect();
    const c = await screen.findByTestId('router-connect');
    expect(c).toHaveTextContent('ANTHROPIC_API_KEY');
    expect(c).toHaveTextContent('ANTHROPIC_AUTH_TOKEN');
  });
});

describe('NavBar', () => {
  it('has a Router entry pointing at /router', async () => {
    stubFetch({ 'GET /api/version': () => json({ version: '1' }) });
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <NavBar />
      </SWRConfig>,
    );
    fireEvent.click(screen.getByRole('button', { name: /menu/i }));
    expect(screen.getByRole('menuitem', { name: /^router$/i })).toHaveAttribute('href', '/router');
    cleanup();
  });
});
