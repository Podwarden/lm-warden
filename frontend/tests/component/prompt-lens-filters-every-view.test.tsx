// The Prompt Lens filter bar on every view (Requests, Sessions, Trends): the Key
// picker (admin only, every key the tokens list names), Model and Client, the
// same picks on every view, an inline single-key picker on Trends, and the
// Sessions list filtered by the server.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { PromptLensView } from '@/components/prompt-lens/prompt-lens-view';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { clearKeyToken, storeKeyToken } from '@/lib/sessions/key-token';

class FakeES {
  onopen?: () => void;
  constructor(public url: string) {
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {}
}

const json = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });

let urls: string[] = [];
function route(u: string): Response | Promise<Response> {
  urls.push(u);
  if (u === '/api/tokens?limit=500') return json({ items: [{ id: 'k-alpha', name: 'alpha-key' }, { id: 'k-beta', name: 'beta-key' }] });
  if (u.includes('/prompt-lens/sessions?')) return json({ sessions: [], next_cursor: null });
  if (u.includes('/prompt-lens/trends')) return json({ points: [], versions: [], models: ['qwen3'], model: 'qwen3' });
  if (u.includes('/ticket') || u === '/api/auth/sse-ticket') return json({ ticket: 't' });
  if (u.includes('/prompt-lens/kept')) return json({ kept: [] });
  if (u === '/api/me/key') return json({ id: 'self', name: 'Dana', keep_captures: false });
  return new Promise<Response>(() => {});
}

function mount(scope: 'admin' | 'key') {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <div style={{ height: 800 }}>
        <PromptLensView scope={scope} pageKey={`every-view-${scope}`} />
      </div>
    </SWRConfig>,
  );
}

const bar = () => screen.getByTestId('prompt-lens-filter-line');
const filter = (name: RegExp) => within(bar()).getByRole('button', { name });
const tab = (name: string) => fireEvent.click(screen.getByRole('tab', { name }));

beforeAll(() => {
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
});
beforeEach(() => {
  urls = [];
  window.localStorage.clear();
  setAccessToken('jwt');
  setCsrfToken('csrf');
  storeKeyToken('key-signin', 3600);
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  vi.stubGlobal('fetch', vi.fn((u: RequestInfo | URL) => Promise.resolve(route(String(u)))));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  clearKeyToken();
});

describe('Prompt Lens filter bar on every view', () => {
  it('admin: the Key picker lists every key from the tokens list, by name, on Requests, Sessions and Trends', async () => {
    mount('admin');
    for (const view of ['Requests', 'Sessions', 'Trends']) {
      tab(view);
      fireEvent.click(filter(/^Key/));
      await waitFor(() => expect(screen.getByRole('menuitemcheckbox', { name: 'alpha-key' })).toBeInTheDocument());
      expect(screen.getByRole('menuitemcheckbox', { name: 'beta-key' })).toBeInTheDocument();
      expect(screen.queryByText(/^k-alpha$/)).toBeNull();
      fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
    }
  });

  it('Sessions: Key, Model and Client are sent to the list API; the picks stay across views', async () => {
    mount('admin');
    tab('Sessions');
    fireEvent.click(filter(/^Key/));
    fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'alpha-key' }));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'beta-key' }));
    fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
    fireEvent.click(filter(/^Client/));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'OpenCode' }));
    fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
    await waitFor(() => {
      const last = urls.filter((u) => u.includes('/admin/prompt-lens/sessions?')).pop() ?? '';
      expect(last).toContain('key=k-alpha');
      expect(last).toContain('key=k-beta');
      expect(last).toContain('harness=opencode');
    });
    expect(screen.getByTestId('prompt-lens-header')).toHaveTextContent('alpha-key, beta-key · OpenCode · scored sessions');
    // The same picks on the other views.
    tab('Requests');
    expect(filter(/^Key/)).toHaveTextContent('2 picked');
    tab('Trends');
    expect(filter(/^Key/)).toHaveTextContent('2 picked');
    expect(filter(/^Client/)).toBeDisabled();
  });

  it('Trends: with no key or several, an inline picker in the message sets the one key', async () => {
    mount('admin');
    tab('Trends');
    const pick = await screen.findByRole('combobox', { name: 'Pick a key' });
    expect(screen.getByTestId('trends-pick-key')).toHaveTextContent('Pick a key');
    expect(urls.some((u) => u.includes('/prompt-lens/trends'))).toBe(false);
    await waitFor(() => expect(within(pick).getByRole('option', { name: 'beta-key' })).toBeInTheDocument());
    fireEvent.change(pick, { target: { value: 'k-beta' } });
    await waitFor(() => expect(urls.some((u) => u.includes('/admin/prompt-lens/trends') && u.includes('key=k-beta'))).toBe(true));
    expect(filter(/^Key/)).toHaveTextContent('beta-key');
    expect(screen.getByTestId('prompt-lens-header')).toHaveTextContent('beta-key · trends');
    // Two keys again: back to the picker, which says so.
    fireEvent.click(filter(/^Key/));
    fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'alpha-key' }));
    fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' });
    expect(await screen.findByTestId('trends-pick-key')).toHaveTextContent('2 keys are picked');
  });

  it('a key holder has no Key picker on any view; Model and Client work on Sessions', async () => {
    mount('key');
    for (const view of ['Requests', 'Sessions', 'Trends']) {
      tab(view);
      expect(within(bar()).queryByRole('button', { name: /^Key/ })).toBeNull();
      expect(within(bar()).getByRole('button', { name: /^Model/ })).toBeInTheDocument();
    }
    tab('Sessions');
    fireEvent.click(filter(/^Client/));
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Claude Code' }));
    await waitFor(() => {
      const last = urls.filter((u) => u.startsWith('/api/me/prompt-lens/sessions?')).pop() ?? '';
      expect(last).toContain('harness=claude_code');
      expect(last).not.toContain('key=');
    });
    tab('Trends');
    expect(screen.queryByTestId('trends-pick-key')).toBeNull();
  });
});
