import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import type { EeModule } from '@/lib/ext/types';

// Editions split, Task 6: the `@ee` slot. With the community stub Settings
// shows exactly today's tabs; an ee tab appears only when its feature is
// listed by GET /api/edition; the gated core controls follow the same list.

const hoisted = vi.hoisted(() => ({
  ee: { settingsTabs: [] as unknown[] } as { settingsTabs: unknown[] },
}));
vi.mock('@/lib/extensions', () => ({
  default: new Proxy({}, { get: (_t, k) => (hoisted.ee as Record<string | symbol, unknown>)[k] ?? (k === 'settingsTabs' ? [] : undefined) }),
}));

import SettingsPage from '@/app/settings/page';
import { NetworkingTab } from '@/components/settings/networking-tab';
import { useEdition } from '@/lib/edition';
import stub from '../../src/ee-stub';

const RUNTIME = {
  admin_username: 'admin', public_url: '', home_page_mode: 'builtin', home_page_off_action: 'console',
  analytics_ga4_id: '', analytics_google_ads_id: '', search_console_verification: '',
};

function mockApi(edition: { status?: number; features?: string[] } = {}) {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url === '/api/edition') {
      if (edition.status && edition.status !== 200) return new Response('no', { status: edition.status });
      return new Response(JSON.stringify({ edition: 'enterprise', features: edition.features ?? [], load_errors: [] }), { status: 200 });
    }
    if (url === '/api/settings/runtime') return new Response(JSON.stringify(RUNTIME), { status: 200 });
    return new Response('{}', { status: 200 });
  }));
}

const wrap = (ui: React.ReactNode) =>
  render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{ui}</SWRConfig>);

beforeEach(() => {
  setAccessToken('t'); setCsrfToken('c');
  hoisted.ee.settingsTabs = [];
});
afterEach(() => cleanup());

describe('community stub', () => {
  it('is empty in every slot and carries no canary', () => {
    const m: EeModule = stub;
    expect(m.settingsTabs).toEqual([]);
    expect(m.navItems).toEqual([]);
    expect(m.views).toEqual({});
    expect(m.tokenPanels).toEqual({});
    expect(m.loginProviders).toEqual([]);
    expect(JSON.stringify(m)).not.toContain('LMW-EE-CANARY');
  });
});

describe('Settings tabs', () => {
  const TODAY = ['General', 'Networking', 'Sessions & Tokens', 'Maintenance', 'Model', 'Access Tokens', 'Sensitive data'];

  it('shows exactly today\'s tabs with no ee tabs', async () => {
    mockApi();
    wrap(<SettingsPage />);
    const general = await screen.findByRole('button', { name: 'General' });
    const tabs = Array.from(general.parentElement!.children).map((t) => t.textContent);
    expect(tabs).toEqual(TODAY);
  });

  it('skips (and warns about) an ee tab whose id duplicates a core tab', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    hoisted.ee.settingsTabs = [
      { id: 'general', label: 'Imposter', feature: 'x.feat', Component: () => <div>imposter body</div> },
    ];
    mockApi({ features: ['x.feat'] });
    wrap(<SettingsPage />);
    expect(await screen.findByRole('button', { name: 'General' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Imposter' })).toBeNull();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('"general"'));
    warn.mockRestore();
  });

  it('adds an ee tab whose feature is listed', async () => {
    hoisted.ee.settingsTabs = [{ id: 'x', label: 'Extra', feature: 'x.feat', Component: () => <div>extra body</div> }];
    mockApi({ features: ['x.feat'] });
    wrap(<SettingsPage />);
    const tab = await screen.findByRole('button', { name: 'Extra' });
    tab.click();
    expect(await screen.findByText('extra body')).toBeInTheDocument();
  });

  it('omits an ee tab whose feature is not listed, or when /api/edition is 403', async () => {
    hoisted.ee.settingsTabs = [{ id: 'x', label: 'Extra', feature: 'x.feat', Component: () => null }];
    mockApi({ features: ['other'] });
    const { unmount } = wrap(<SettingsPage />);
    await screen.findByRole('button', { name: 'General' });
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Extra' })).toBeNull());
    unmount();
    mockApi({ status: 403 });
    wrap(<SettingsPage />);
    await screen.findByRole('button', { name: 'General' });
    expect(screen.queryByRole('button', { name: 'Extra' })).toBeNull();
  });
});

describe('useEdition', () => {
  function Probe() {
    const e = useEdition();
    return <div data-testid="p">{e ? `${e.edition}:${e.features.join(',')}` : 'undefined'}</div>;
  }
  it('returns edition and features', async () => {
    mockApi({ features: ['a', 'b'] });
    wrap(<Probe />);
    await waitFor(() => expect(screen.getByTestId('p').textContent).toBe('enterprise:a,b'));
  });
  it.each([401, 403, 500])('returns undefined on %i', async (status) => {
    mockApi({ status });
    wrap(<Probe />);
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.getByTestId('p').textContent).toBe('undefined');
  });
});

describe('Networking tab gating', () => {
  it('hides the analytics fields and keeps the home control without features', async () => {
    mockApi();
    wrap(<NetworkingTab />);
    await screen.findByText('Public access');
    expect(screen.queryByText('Website analytics')).toBeNull();
    expect(screen.queryByText('Google Analytics 4 measurement ID')).toBeNull();
    expect(screen.getByLabelText('Built-in')).toBeInTheDocument();
  });
  it('shows the analytics fields when site.analytics is listed', async () => {
    mockApi({ features: ['site.analytics'] });
    wrap(<NetworkingTab />);
    expect(await screen.findByText('Website analytics')).toBeInTheDocument();
    expect(screen.getByText('Google Analytics 4 measurement ID')).toBeInTheDocument();
    expect(screen.getByText('Search Console verification')).toBeInTheDocument();
  });
  it('hides the Off/Built-in home control when home.custom is listed', async () => {
    mockApi({ features: ['home.custom'] });
    wrap(<NetworkingTab />);
    await screen.findByText('Public access');
    await waitFor(() => expect(screen.queryByLabelText('Built-in')).toBeNull());
    expect(screen.getByText('Public URL')).toBeInTheDocument();
  });
});
