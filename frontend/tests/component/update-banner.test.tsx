// Update banner (spec 2026-11-20): a red, non-dismissible line for a security
// advisory that applies to the running version; otherwise a dismissible line
// for a newer release; nothing when neither, on error, or off the console.
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

let mockPath = '/models';
vi.mock('next/navigation', () => ({ usePathname: () => mockPath }));

import { UpdateBanner } from '@/components/update-banner';

const BASE = {
  enabled: true,
  current: 'v2026.11.20.1',
  last_checked_at: '2026-11-20T10:00:00Z',
  latest: { version: 'v2026.12.01.1', released_at: '2026-12-01', url: 'https://example.test/notes' },
  update_available: true,
  dismissed: false,
  advisories: [] as unknown[],
};

function stub(status: object | null, code = 200) {
  const posts: unknown[] = [];
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : (input as Request).url;
    if (url === '/api/system/update-status/dismiss') {
      posts.push(JSON.parse(String(init?.body)));
      return new Response('{}', { status: 200 });
    }
    if (url === '/api/system/update-status') {
      if (status === null) return new Response('no', { status: code });
      return new Response(JSON.stringify(status), { status: 200 });
    }
    return new Response('{}', { status: 200 });
  }));
  return posts;
}

const view = () =>
  render(<SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}><UpdateBanner /></SWRConfig>);

beforeEach(() => { mockPath = '/models'; setAccessToken('t'); setCsrfToken('c'); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('UpdateBanner', () => {
  it('announces a newer release and dismisses it for that version', async () => {
    const posts = stub(BASE);
    view();
    expect(await screen.findByText(/LM Warden v2026\.12\.01\.1 is available/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /release notes/i })).toHaveAttribute('href', 'https://example.test/notes');
    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    await waitFor(() => expect(posts).toEqual([{ version: 'v2026.12.01.1' }]));
    await waitFor(() => expect(screen.queryByText(/is available/)).toBeNull());
  });

  it('shows an applying advisory in red with no dismiss button', async () => {
    stub({ ...BASE, update_available: false, advisories: [
      { id: 'LMW-2026-01', severity: 'high', title: 'Proxy leaks keys', fixed_in: '2026.12.01.1', url: 'https://example.test/a' },
    ] });
    view();
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('LMW-2026-01');
    expect(alert).toHaveTextContent('Proxy leaks keys');
    expect(alert).toHaveTextContent('v2026.12.01.1');
    expect(screen.queryByRole('button', { name: /dismiss/i })).toBeNull();
  });

  it('renders nothing when there is nothing to say', async () => {
    const f = stub({ ...BASE, update_available: false, latest: null });
    const { container } = view();
    await waitFor(() => expect(f).toEqual([]));
    await new Promise((r) => setTimeout(r, 20));
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing on an error', async () => {
    stub(null, 500);
    const { container } = view();
    await new Promise((r) => setTimeout(r, 20));
    expect(container).toBeEmptyDOMElement();
  });

  it('stays off the login page', async () => {
    mockPath = '/login';
    stub(BASE);
    const { container } = view();
    await new Promise((r) => setTimeout(r, 20));
    expect(container).toBeEmptyDOMElement();
  });
});
