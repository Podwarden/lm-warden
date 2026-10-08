// Links into the Connect page (plan §3 "Links to the page"): the router panel's
// one line, and the API keys page header. The nav entry and the breadcrumb are
// pinned in nav-bar.test.tsx and breadcrumbs.test.ts.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { ConnectClaudeCode } from '@/components/router/connect-claude-code';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import type { RelayInfo } from '@/lib/router';
import { RULE_A, SETTINGS, stubFetch, baseRoutes } from './router-test-utils';

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => '/tokens',
}));

import TokensPage from '@/app/tokens/page';

const RELAY: RelayInfo = { relayKeys: 1, relayKnown: true, relayLoading: false };
const LINE = /Using another tool\? Connect Codex, OpenCode, Aider…/;

function renderPanel(defaultOpen: boolean) {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <ConnectClaudeCode
        settings={SETTINGS}
        rules={[RULE_A]}
        decisions={[]}
        relay={RELAY}
        plaintext={null}
        defaultOpen={defaultOpen}
        onCreateRelayKey={() => {}}
      />
    </SWRConfig>,
  );
}

describe('router panel → Connect', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    stubFetch(baseRoutes());
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('the expanded panel links to /connect?tool=opencode', () => {
    renderPanel(true);
    const link = screen.getByRole('link', { name: LINE });
    expect(link).toHaveAttribute('href', '/connect?tool=opencode');
  });

  it('the collapsed panel keeps its summary and has no link', () => {
    renderPanel(false);
    expect(screen.queryByRole('link', { name: LINE })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /Connect Claude Code/ }));
    expect(screen.getByRole('link', { name: LINE })).toBeInTheDocument();
  });
});

describe('API keys page → Connect', () => {
  beforeEach(() => {
    setAccessToken('jwt');
    setCsrfToken('csrf');
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ items: [], total: 0, limit: 25, offset: 0, near_expiry: 0 }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('the header has "Connect a client" next to Create token', async () => {
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <TokensPage />
      </SWRConfig>,
    );
    const link = await screen.findByRole('link', { name: /Connect a client/ });
    expect(link).toHaveAttribute('href', '/connect');
    // Left of the primary action, in the same header row.
    const create = screen.getByRole('button', { name: 'Create token' });
    expect(link.compareDocumentPosition(create) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
