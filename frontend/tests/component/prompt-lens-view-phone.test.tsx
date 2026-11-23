// The Prompt Lens page at phone width (Task 16, spec 2026-11-22 section 6,
// mockup states 7 and 12; a page since 2026-11-23, it was a dock): a 44 px
// header without tools, "Filters (n)" opening a filter sheet, the list then
// the request screen with "‹ Requests", 160 px section boxes, the parts table
// behind "Show the N parts" as cards, the pop-up full screen, and the system
// back gesture closing the top layer. The Stats Requests table keeps Status,
// (Key), Phase, tok/s and the eye on a phone.

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { SWRConfig } from 'swr';

vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { PromptLensView } from '@/components/prompt-lens/prompt-lens-view';
import { AdminRequestsTable, KeyRequestsTable } from '@/components/stats/redesign/requests-table';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { resetServerClock } from '@/lib/prompt-lens/server-clock';
import { clearKeyToken, storeKeyToken } from '@/lib/sessions/key-token';
import { PHONE_QUERY } from '@/lib/use-media-query';
import type { LiveRequestRow } from '@/lib/live-stats';
import type { MeLiveRow } from '@/lib/stats-redesign';

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.all.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {
    this.closed = true;
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

function route(url: string): Response | Promise<Response> {
  if (url === '/api/auth/sse-ticket') return json({ ticket: 'a1' });
  if (url === '/api/me/prompt-lens/ticket') return json({ ticket: 'k1' });
  if (url.startsWith('/api/me/prompt-lens/kept') || url.startsWith('/api/admin/prompt-lens/kept')) return json({ kept: [] });
  if (url === '/api/me/key') return json({ id: 'self', name: 'Dana', keep_captures: false });
  return new Promise<Response>(() => {});
}

const flush = (ms = 0) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
    vi.advanceTimersToNextFrame();
  });

const es = () => FakeES.all[FakeES.all.length - 1];
function emit(ev: unknown) {
  act(() => {
    es().onmessage?.({ data: JSON.stringify(ev) } as MessageEvent);
    vi.advanceTimersToNextFrame();
  });
}

let T = 1_900_000_000;
function start(reqId: string, o: Record<string, unknown> = {}) {
  T += 1;
  return {
    type: 'request_start',
    req_id: reqId,
    ts: T,
    token_id: 'tok-a',
    request_id: `rq-${reqId}`,
    model: 'qwen3.8-27b',
    served_name: 'qwen3.8-27b',
    stream: true,
    prompt: `prompt of ${reqId}`,
    session_id: 'sess0001aaaa',
    turn: 1,
    ...o,
  };
}

const PARTS = [
  { p: 'sys', from: -1, to: -1, chars: 1200, roles: { system: 1 }, exact8: 'aaaa1111', loose8: 'bbbb2222', state: 'exact' },
  { p: 'tools', from: -1, to: -1, chars: 3600, roles: {}, exact8: 'cccc3333', loose8: 'dddd4444', state: 'exact' },
  { p: 'nodes', from: 0, to: 0, chars: 900, roles: { user: 1 }, exact8: 'eeee5555', loose8: 'ffff6666', state: 'new' },
];

function kvsig(reqId: string) {
  return {
    type: 'kvsig',
    req_id: reqId,
    ts: T + 1,
    token_id: 'tok-a',
    session_id: 'sess0001aaaa',
    turn: 1,
    available: true,
    profile: 'loose/1',
    conv8: 'abcd1234',
    parts: PARTS,
  };
}

function Harness({ scope, presetModel = null }: { scope: 'key' | 'admin'; presetModel?: string | null }) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <div style={{ height: 812 }}>
        <PromptLensView scope={scope} pageKey={`test-${scope}`} presetModel={presetModel} />
      </div>
    </SWRConfig>
  );
}

const bar = () => screen.getByTestId('prompt-lens-header');
const dock = () => screen.getByTestId('prompt-lens-view');
const filterLine = () => screen.getByTestId('prompt-lens-filter-line');
const filtersBtn = () => within(filterLine()).getByRole('button', { name: /^Filters/ });

/** The system back gesture: the browser pops one entry, so the state handed
 *  to popstate is the entry below the current one. */
function back(state: unknown) {
  act(() => {
    window.dispatchEvent(new PopStateEvent('popstate', { state }));
  });
}

function setPhone(phone: boolean) {
  vi.stubGlobal(
    'matchMedia',
    (q: string) =>
      ({
        matches: q === PHONE_QUERY ? phone : false,
        media: q,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {},
      }) as unknown as MediaQueryList,
  );
}

async function openDock(scope: 'key' | 'admin' = 'key', presetModel: string | null = null) {
  render(<Harness scope={scope} presetModel={presetModel} />);
  await flush();
}

beforeAll(() => {
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
});

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date', 'requestAnimationFrame', 'cancelAnimationFrame'] });
  vi.setSystemTime(new Date((T + 5) * 1000));
  FakeES.all = [];
  resetServerClock();
  window.localStorage.clear();
  setAccessToken('jwt');
  setCsrfToken('csrf');
  storeKeyToken('key-signin', 3600);
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  vi.stubGlobal('fetch', vi.fn((u: RequestInfo | URL) => Promise.resolve(route(String(u)))));
  setPhone(true);
});
afterEach(async () => {
  cleanup();
  vi.useRealTimers();
  // Let a history.go() from a layer closed by the unmount land before the
  // next test pushes its own entries.
  await new Promise((r) => setTimeout(r, 20));
  vi.unstubAllGlobals();
  clearKeyToken();
});

// ---------------------------------------------------------------------------

describe('Prompt Lens on a phone', () => {
  it('the header is 44 px and keeps only what fits: no Follow newest, no Pause', async () => {
    await openDock();
    expect(bar()).toHaveClass('h-11');
    expect(dock()).toHaveClass('h-full');
    expect(screen.queryByRole('button', { name: 'Follow newest' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Pause' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Collapse' })).toBeNull();
  });

  it('Filters (n) opens the filter sheet and shows the active count', async () => {
    await openDock('admin', 'qwen3.8-27b');
    emit(start('r1'));
    const line = filterLine();
    // The line keeps the search and one button; the menus live in the sheet.
    expect(within(line).getByRole('searchbox')).toBeInTheDocument();
    expect(within(line).queryByRole('button', { name: /^Session/ })).toBeNull();
    expect(within(line).queryByRole('button', { name: /^Model/ })).toBeNull();
    expect(filtersBtn()).toHaveTextContent('Filters (1)');
    expect(filtersBtn()).toHaveClass('min-h-11');

    fireEvent.click(filtersBtn());
    const sheet = screen.getByRole('dialog', { name: 'Filters' });
    for (const g of ['Key', 'Session', 'Model', 'Client IP', 'Phase', 'Finish', 'Has']) {
      expect(within(sheet).getByRole('group', { name: g })).toBeInTheDocument();
    }
    fireEvent.click(within(within(sheet).getByRole('group', { name: 'Phase' })).getByRole('button', { name: 'Live' }));
    fireEvent.click(within(within(sheet).getByRole('group', { name: 'Finish' })).getByRole('checkbox', { name: 'error' }));
    expect(filtersBtn()).toHaveTextContent('Filters (3)');
    // The search is in the line, not counted by the button.
    fireEvent.change(within(line).getByRole('searchbox'), { target: { value: 'hello' } });
    expect(filtersBtn()).toHaveTextContent('Filters (3)');

    fireEvent.click(within(sheet).getByRole('button', { name: 'Done' }));
    expect(screen.queryByRole('dialog', { name: 'Filters' })).toBeNull();
    expect(dock()).toBeInTheDocument();
  });

  it('the key holder filter sheet has no Key and no Client IP', async () => {
    await openDock('key');
    fireEvent.click(filtersBtn());
    const sheet = screen.getByRole('dialog', { name: 'Filters' });
    expect(within(sheet).queryByRole('group', { name: 'Key' })).toBeNull();
    expect(within(sheet).queryByRole('group', { name: 'Client IP' })).toBeNull();
    for (const g of ['Session', 'Model', 'Phase', 'Finish', 'Has']) {
      expect(within(sheet).getByRole('group', { name: g })).toBeInTheDocument();
    }
  });

  it("tapping a row shows the request; '‹ Requests' goes back", async () => {
    await openDock();
    emit(start('r1'));
    // The list fills the sheet; no request screen beside it.
    expect(screen.getByTestId('prompt-lens-list').style.width).toBe('');
    expect(screen.queryByTestId('prompt-lens-screen')).toBeNull();
    fireEvent.click(screen.getByTestId('prompt-lens-row-r1'));
    expect(screen.queryByTestId('prompt-lens-list')).toBeNull();
    expect(screen.getByTestId('request-screen')).toBeInTheDocument();
    const backBtn = screen.getByRole('button', { name: '‹ Requests' });
    expect(backBtn).toHaveClass('min-h-11');
    fireEvent.click(backBtn);
    expect(screen.getByTestId('prompt-lens-list')).toBeInTheDocument();
    expect(screen.queryByTestId('request-screen')).toBeNull();
  });

  it('a live request arriving with Follow newest does not leave the list', async () => {
    await openDock();
    emit(start('r1'));
    emit(start('r2'));
    expect(screen.getByTestId('prompt-lens-list')).toBeInTheDocument();
    expect(screen.queryByTestId('request-screen')).toBeNull();
  });

  it('section boxes are 160 px; parts are cards without id columns', async () => {
    await openDock();
    emit(start('r1'));
    emit(kvsig('r1'));
    fireEvent.click(screen.getByTestId('prompt-lens-row-r1'));
    const box = screen.getByTestId('section-box-thinking');
    expect(box.style.maxHeight || box.style.height).toBe('160px');
    // Input and Output stack.
    expect(screen.getByTestId('rs-cols')).toHaveClass('flex-col');
    expect(screen.queryByRole('columnheader')).toBeNull();
    const show = screen.getByRole('button', { name: /^Show the 3 parts/ });
    expect(show).toHaveClass('min-h-11');
    fireEvent.click(show);
    expect(screen.getAllByTestId('parts-card')).toHaveLength(3);
    const text = screen.getByTestId('request-screen').textContent ?? '';
    expect(text).not.toMatch(/exact id|loose id/);
    expect(text).not.toContain('aaaa');
    expect(text).not.toMatch(/cobs2/i);
    // Section header buttons are touch-sized.
    expect(screen.getByRole('button', { name: 'Open Thinking in a large view' })).toHaveClass('min-h-11', 'min-w-11');
  });

  it('back gesture closes the pop-up, then the request (the page stays)', async () => {
    await openDock();
    const sheetState = window.history.state;
    emit(start('r1'));
    fireEvent.click(screen.getByTestId('prompt-lens-row-r1'));
    const requestState = window.history.state;
    expect(requestState).not.toEqual(sheetState);
    fireEvent.click(screen.getByRole('button', { name: 'Open Thinking in a large view' }));
    const popup = screen.getByRole('dialog');
    expect(popup).toHaveAttribute('data-phone', 'true');
    expect(popup).toHaveClass('inset-0');

    back(requestState);
    expect(screen.queryByTestId('section-popup')).toBeNull();
    expect(screen.getByTestId('request-screen')).toBeInTheDocument();

    back(sheetState);
    expect(screen.queryByTestId('request-screen')).toBeNull();
    expect(screen.getByTestId('prompt-lens-list')).toBeInTheDocument();
    expect(dock()).toBeInTheDocument();
  });

  it('back gesture closes the filter sheet only', async () => {
    await openDock();
    const sheetState = window.history.state;
    fireEvent.click(filtersBtn());
    expect(screen.getByRole('dialog', { name: 'Filters' })).toBeInTheDocument();
    back(sheetState);
    expect(screen.queryByRole('dialog', { name: 'Filters' })).toBeNull();
    expect(dock()).toBeInTheDocument();
  });

  it('Esc closes the filter sheet and leaves the page as it was', async () => {
    await openDock();
    fireEvent.click(filtersBtn());
    expect(screen.getByRole('dialog', { name: 'Filters' })).toBeInTheDocument();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Filters' })).toBeNull();
    expect(dock()).toBeInTheDocument();
  });

  it('Clear all turns every sheet filter off and keeps the search', async () => {
    await openDock('admin', 'qwen3.8-27b');
    emit(start('r1'));
    fireEvent.change(within(filterLine()).getByRole('searchbox'), { target: { value: 'hello' } });
    fireEvent.click(filtersBtn());
    const sheet = screen.getByRole('dialog', { name: 'Filters' });
    fireEvent.click(within(within(sheet).getByRole('group', { name: 'Phase' })).getByRole('button', { name: 'Live' }));
    expect(filtersBtn()).toHaveTextContent('Filters (2)');
    const clear = within(sheet).getByRole('button', { name: 'Clear all' });
    expect(clear).toBeEnabled();
    fireEvent.click(clear);
    expect(filtersBtn()).toHaveTextContent('Filters (0)');
    expect(within(sheet).getByRole('button', { name: 'Clear all' })).toBeDisabled();
    expect(within(within(sheet).getByRole('group', { name: 'Phase' })).getByRole('button', { name: 'All' })).toHaveAttribute('aria-pressed', 'true');
    expect(within(filterLine()).getByRole('searchbox')).toHaveValue('hello');
  });

  it('kvsig map segments are 44 px tall on a phone', async () => {
    await openDock();
    emit(start('r1'));
    emit(kvsig('r1'));
    fireEvent.click(screen.getByTestId('prompt-lens-row-r1'));
    expect(screen.getByTestId('kvsig-bar')).toHaveClass('h-11');
    for (const s of screen.getAllByTestId('kvsig-seg')) expect(s).toHaveClass('h-full');
  });

  it('✕ closes the pop-up and keeps the request screen', async () => {
    await openDock();
    emit(start('r1'));
    fireEvent.click(screen.getByTestId('prompt-lens-row-r1'));
    fireEvent.click(screen.getByRole('button', { name: 'Open Reply in a large view' }));
    expect(screen.getByTestId('section-popup')).toBeInTheDocument();
    // ✕ closes it too, and leaves the request screen.
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByTestId('section-popup')).toBeNull();
    expect(screen.getByTestId('request-screen')).toBeInTheDocument();
  });

  it('a desktop page pushes no history entries and has a 40 px header with its tools', async () => {
    setPhone(false);
    const len = window.history.length;
    await openDock();
    emit(start('r1'));
    fireEvent.click(screen.getByTestId('prompt-lens-row-r1'));
    expect(window.history.length).toBe(len);
    expect(bar()).toHaveClass('h-10');
    expect(screen.getByRole('button', { name: 'Follow newest' })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------

function adminRow(id: string): LiveRequestRow {
  return {
    id,
    token_id: 't1',
    token_name: 'dev',
    client_ip: '10.0.0.7',
    model: 'qwen3.8-27b',
    path: '/v1/chat/completions',
    prompt_tokens: 31_000,
    completion_tokens: 200,
    context_tokens: 31_200,
    max_model_len: 131_072,
    context_pct: 0.238,
    elapsed_s: 12.3,
    phase: 'answering',
    decode_tps_5s: 40,
  } as LiveRequestRow;
}

/** Columns a phone keeps: their header is not hidden below md. */
function phoneHeads(): string[] {
  return screen
    .getAllByRole('columnheader')
    .filter((h) => !h.className.split(/\s+/).includes('hidden'))
    .map((h) => h.textContent ?? '');
}

describe("the page's Requests table on a phone", () => {
  it('admin keeps Status, Key, Phase, tok/s and the eye', () => {
    render(
      <AdminRequestsTable
        rows={[adminRow('a1')]}
        scopeModel={null}
        multiModel={false}
        windowGroups={null}
        onGroupChange={() => {}}
        lastFinishedAt={null}
        range="1h"
        onEye={() => {}}
      />,
    );
    expect(phoneHeads()).toEqual(['', 'Key', 'Phase', 'tok/s', '']);
    expect(screen.getByRole('button', { name: 'Open in Prompt Lens' })).toHaveClass('max-md:h-11', 'max-md:w-11');
  });

  it('key holder keeps Status, Phase, tok/s and the eye', () => {
    const row = { ...adminRow('k1'), expected_start_s: null } as unknown as MeLiveRow;
    render(<KeyRequestsTable rows={[row]} multiModel={false} lastFinishedAt={null} onEye={() => {}} />);
    expect(phoneHeads()).toEqual(['', 'Phase', 'tok/s', '']);
  });

  it('without Prompt Lens the phone keeps Elapsed as before', () => {
    render(
      <AdminRequestsTable
        rows={[adminRow('a1')]}
        scopeModel={null}
        multiModel={false}
        windowGroups={null}
        onGroupChange={() => {}}
        lastFinishedAt={null}
        range="1h"
      />,
    );
    expect(phoneHeads()).toEqual(['', 'Key', 'Phase', 'tok/s', 'Elapsed']);
  });
});
