import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';

import type { SectionResponse } from '@/lib/prompt-lens/events';
import type { RequestView, Section, ToolCallView } from '@/lib/prompt-lens/store';

import { PROMPT, part, reqWith } from '../unit/prompt-lens-kvsig-fixtures';

const h = vi.hoisted(() => ({
  admin: vi.fn(),
  key: vi.fn(),
  copy: vi.fn(),
}));

vi.mock('@/lib/auth-fetch', async (orig) => ({
  ...(await orig<typeof import('@/lib/auth-fetch')>()),
  authFetchJSON: (u: string) => h.admin(u),
}));
vi.mock('@/lib/sessions/client', async (orig) => ({
  ...(await orig<typeof import('@/lib/sessions/client')>()),
  sessionsFetcher: () => (u: string) => h.key(u),
}));
vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: (t: string) => h.copy(t) }));

import { SectionPopup } from '@/components/prompt-lens/dock/section-popup';
import { SessionsHttpError } from '@/lib/sessions/client';

// ---------------------------------------------------------------------------

function resp(o: Partial<SectionResponse> = {}): SectionResponse {
  return {
    kind: 'thinking',
    seq: 10,
    live: true,
    dropped_chars: 0,
    offset: 0,
    text: '',
    dropped_calls: 0,
    kept: false,
    ...o,
  } as SectionResponse;
}

function sec(text: string, clipped = 0): Section {
  return { text, chars: clipped + text.length, clipped, frames: 1, firstTs: 1, lastTs: 1 };
}

function req(o: Partial<RequestView> = {}): RequestView {
  return { ...reqWith(), ...o };
}

function httpError(status: number) {
  return Object.assign(new Error(`HTTP ${status}`), { status });
}

/** Resolve the next fetch(es) of a scope with `body` (or reject with `err`). */
function answer(fn: typeof h.admin, body: SectionResponse | Error) {
  fn.mockImplementation(async () => {
    if (body instanceof Error) throw body;
    return body;
  });
}

const body = () => screen.getByTestId('popup-body');
const count = () => screen.getByTestId('popup-count').textContent;
const search = () => screen.getByRole('searchbox', { name: /search this section/i });

beforeEach(() => {
  h.admin.mockReset();
  h.key.mockReset();
  h.copy.mockReset();
  h.copy.mockImplementation(async () => {});
});

afterEach(() => {
  cleanup();
});

async function settle() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

// ---------------------------------------------------------------------------

describe('SectionPopup: whole section, then live deltas after the offset (G-F3)', () => {
  it('shows the whole section from the server, then appends live deltas after seq', async () => {
    // The dock holds only the tail ("gamma "); the server has it all.
    answer(h.admin, resp({ text: 'alpha beta gamma ', offset: 0, seq: 10 }));
    const r = req({ thinking: sec('gamma ', 11) });
    const { rerender } = render(<SectionPopup scope="admin" kind="thinking" request={r} onClose={() => {}} />);
    await settle();
    expect(body().textContent).toContain('alpha beta gamma ');

    // A live delta lands in the store: appended once, after the server's end.
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma delta ', 11) })} onClose={() => {}} />);
    expect(body().textContent).toContain('alpha beta gamma delta ');
    expect(body().textContent!.match(/gamma/g)).toHaveLength(1);
  });

  it('does not repeat text the server already had when the store catches up', async () => {
    answer(h.admin, resp({ text: 'alpha beta gamma delta ', offset: 0 }));
    const { rerender } = render(
      <SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('alpha beta gamma ') })} onClose={() => {}} />,
    );
    await settle();
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('alpha beta gamma delta ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('alpha beta gamma delta ');
    expect(body().textContent!.match(/delta/g)).toHaveLength(1);
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('alpha beta gamma delta eps ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('alpha beta gamma delta eps ');
  });

  it('aligns a store that joined mid-stream (its counts start late) by content', async () => {
    // The store saw only "gamma " and counts it from 0; the server's offset is absolute.
    answer(h.key, resp({ text: 'alpha beta gamma ', offset: 0, seq: null }));
    const { rerender } = render(<SectionPopup scope="key" kind="thinking" request={req({ thinking: sec('gamma ') })} onClose={() => {}} />);
    await settle();
    rerender(<SectionPopup scope="key" kind="thinking" request={req({ thinking: sec('gamma delta ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('alpha beta gamma delta ');
    expect(body().textContent!.match(/gamma/g)).toHaveLength(1);
  });

  // Owner bug 2026-11-23: "if i expand this panel, streaming stops". A dock
  // that joined mid-request counts from a later start, and by the time the
  // whole section arrives the store has already moved past the server's end:
  // its last chars are not in the fetched text, so the old alignment fell back
  // to pinning the store's end to the fetched end and appended nothing, on
  // every later delta too.
  it('keeps streaming when a mid-stream store is already past the fetched end', async () => {
    answer(h.admin, resp({ text: 'alpha beta gamma ', offset: 0 }));
    const { rerender } = render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma delta ') })} onClose={() => {}} />);
    await settle();
    expect(body().textContent).toContain('alpha beta gamma delta ');
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma delta epsilon ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('alpha beta gamma delta epsilon ');
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma delta epsilon zeta ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('alpha beta gamma delta epsilon zeta ');
    expect(body().textContent!.match(/gamma/g)).toHaveLength(1);
  });

  it('keeps streaming after the store lost a delta (a lagged stream)', async () => {
    answer(h.key, resp({ text: 'one two three four ', offset: 0, seq: null }));
    const { rerender } = render(<SectionPopup scope="key" kind="reply" request={req({ reply: sec('three four ') })} onClose={() => {}} />);
    await settle();
    rerender(<SectionPopup scope="key" kind="reply" request={req({ reply: sec('three four five ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('one two three four five ');
    // "six " never reached the store; what comes after it still shows.
    rerender(<SectionPopup scope="key" kind="reply" request={req({ reply: sec('three four five seven ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('seven ');
    rerender(<SectionPopup scope="key" kind="reply" request={req({ reply: sec('three four five seven eight ') })} onClose={() => {}} />);
    expect(body().textContent).toContain('seven eight ');
  });

  it('refetches with ?from=<end> when the store dropped text past the end (resume)', async () => {
    answer(h.admin, resp({ text: 'alpha beta gamma ', offset: 0 })); // end = 17
    const { rerender } = render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma ', 11) })} onClose={() => {}} />);
    await settle();
    // The store moved on by more than it keeps: its head (40) is past our end (17).
    answer(h.admin, resp({ text: 'delta epsilon zeta eta theta iota kappa ', offset: 17 }));
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('kappa ', 51) })} onClose={() => {}} />);
    await settle();
    expect(h.admin).toHaveBeenLastCalledWith('/api/admin/prompt-lens/requests/r1/section/thinking?from=17');
    expect(body().textContent).toContain('alpha beta gamma delta epsilon zeta eta theta iota kappa ');
  });

  it('marks what the server dropped from the head', async () => {
    answer(h.admin, resp({ text: 'tail end', offset: 5000, dropped_chars: 5000, live: false }));
    render(<SectionPopup scope="admin" kind="reply" request={req({ reply: sec('tail end', 5000), endedTs: 3 })} onClose={() => {}} />);
    await settle();
    expect(screen.getByTestId('popup-marks').textContent).toMatch(/first 5,000 chars dropped on the server/);
    // the whole size counts the dropped head
    expect(screen.getByTestId('popup-size').textContent).toMatch(/5,008 chars/);
  });
});

describe('SectionPopup: search (Q13 plain, case-insensitive)', () => {
  async function open(text: string, live = false) {
    answer(h.admin, resp({ text, live }));
    const r = req({ thinking: sec(text), endedTs: live ? null : 3 });
    const utils = render(<SectionPopup scope="admin" kind="thinking" request={r} onClose={() => {}} />);
    await settle();
    return utils;
  }

  it("Enter / Shift+Enter step through hits and show '2 of 3'", async () => {
    await open('Dock one, dock two, DOCK three');
    fireEvent.change(search(), { target: { value: 'dock' } });
    expect(count()).toBe('1 of 3');
    const cur = () => body().querySelector('mark[data-current="true"]');
    expect(cur()?.textContent).toBe('Dock');
    expect(body().querySelectorAll('mark')).toHaveLength(3);
    fireEvent.keyDown(search(), { key: 'Enter' });
    expect(count()).toBe('2 of 3');
    expect(cur()?.textContent).toBe('dock');
    fireEvent.keyDown(search(), { key: 'Enter' });
    expect(count()).toBe('3 of 3');
    expect(cur()?.textContent).toBe('DOCK');
    fireEvent.keyDown(search(), { key: 'Enter' });
    expect(count()).toBe('1 of 3');
    fireEvent.keyDown(search(), { key: 'Enter', shiftKey: true });
    expect(count()).toBe('3 of 3');
    // the arrow buttons do the same
    fireEvent.click(screen.getByRole('button', { name: /previous match/i }));
    expect(count()).toBe('2 of 3');
    fireEvent.click(screen.getByRole('button', { name: /next match/i }));
    expect(count()).toBe('3 of 3');
  });

  it('says "no matches" and treats regex characters as text', async () => {
    await open('a+b and a.b');
    fireEvent.change(search(), { target: { value: 'a.b' } });
    expect(count()).toBe('1 of 1');
    fireEvent.change(search(), { target: { value: 'zzz' } });
    expect(count()).toBe('no matches');
  });

  it('a current hit stops following; Jump to latest clears the search', async () => {
    await open('one dock two', true);
    expect(screen.getByRole('button', { name: /^following$/i })).toBeInTheDocument();
    fireEvent.change(search(), { target: { value: 'dock' } });
    expect(screen.queryByRole('button', { name: /^following$/i })).toBeNull();
    const jump = screen.getByRole('button', { name: /jump to latest/i });
    fireEvent.click(jump);
    expect((search() as HTMLInputElement).value).toBe('');
    expect(count()).toBe('');
    expect(screen.getByRole('button', { name: /^following$/i })).toBeInTheDocument();
  });

  it('marks only visible blocks, plus the block of the current hit', async () => {
    // Three blocks of > 4096 chars each, a hit in every block; only the first is on screen.
    const blk = (w: string) => `${w} ` + 'x'.repeat(4200) + '\n';
    const text = blk('aa needle') + blk('bb needle') + blk('cc needle');
    const observed: Element[] = [];
    let cb: IntersectionObserverCallback = () => {};
    class IO {
      constructor(c: IntersectionObserverCallback) {
        cb = c;
      }
      observe(el: Element) {
        observed.push(el);
      }
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal('IntersectionObserver', IO);
    await open(text);
    fireEvent.change(search(), { target: { value: 'needle' } });
    expect(count()).toBe('1 of 3');
    act(() => {
      cb(
        observed.map((el, i) => ({ target: el, isIntersecting: i === 0 }) as unknown as IntersectionObserverEntry),
        {} as IntersectionObserver,
      );
    });
    expect(body().querySelectorAll('mark')).toHaveLength(1);
    // The current hit's block is marked even off screen.
    fireEvent.keyDown(search(), { key: 'Enter' });
    fireEvent.keyDown(search(), { key: 'Enter' });
    expect(count()).toBe('3 of 3');
    const marks = body().querySelectorAll('mark');
    expect(marks).toHaveLength(2);
    expect(body().querySelector('mark[data-current="true"]')?.getAttribute('data-hit')).toBe('2');
  });
});

describe('SectionPopup: dialog', () => {
  it('Esc and the backdrop close it', async () => {
    answer(h.admin, resp({ text: 'x' }));
    const onClose = vi.fn();
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('x') })} onClose={onClose} />);
    await settle();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('dialog'));
    expect(onClose).toHaveBeenCalledTimes(1); // a click inside does not close
    fireEvent.click(screen.getByTestId('popup-backdrop'));
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole('button', { name: /^close$/i }));
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it('is an aria-modal dialog that traps focus and returns it to the expand button', async () => {
    answer(h.admin, resp({ text: 'x' }));
    function Host() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open Thinking in a large view
          </button>
          {open && <SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('x') })} onClose={() => setOpen(false)} />}
        </>
      );
    }
    render(<Host />);
    const expand = screen.getByRole('button', { name: /large view/i });
    expand.focus();
    fireEvent.click(expand);
    await settle();
    const dlg = screen.getByRole('dialog');
    expect(dlg).toHaveAttribute('aria-modal', 'true');
    expect(dlg).toHaveAccessibleName(/thinking/i);
    expect(document.activeElement).toBe(search());
    // Tab from the last focusable wraps to the first, Shift+Tab from the first to the last.
    const focusables = Array.from(dlg.querySelectorAll<HTMLElement>('button:not([disabled]), input, [tabindex="0"]'));
    focusables[focusables.length - 1].focus();
    fireEvent.keyDown(document.activeElement!, { key: 'Tab' });
    expect(document.activeElement).toBe(focusables[0]);
    fireEvent.keyDown(document.activeElement!, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(focusables[focusables.length - 1]);
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.activeElement).toBe(expand);
  });

  it('header: name, whole size, context and Live; the key scope shows no key', async () => {
    answer(h.key, resp({ text: 'abc', seq: null }));
    render(
      <SectionPopup
        scope="key"
        kind="thinking"
        request={req({ thinking: sec('abc'), tokenLabel: 'dana', tokenId: 'tok1', sessionId: '3f9a0c21deadbeef', turn: 17, servedName: 'qwen3' })}
        onClose={() => {}}
      />,
    );
    await settle();
    const head = screen.getByTestId('popup-header');
    expect(head.textContent).toMatch(/Thinking/);
    expect(head.textContent).toMatch(/3 chars/);
    expect(head.textContent).toMatch(/3f9a0c21/);
    expect(head.textContent).toMatch(/turn 17/);
    expect(head.textContent).toMatch(/qwen3/);
    expect(head.textContent).toMatch(/Live/);
    expect(head.textContent).not.toMatch(/dana|tok1/);
    expect(document.body.textContent).not.toMatch(/cobs2|god[ -]?mode/i);
  });

  it('header: the admin sees the key', async () => {
    answer(h.admin, resp({ text: 'abc' }));
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('abc'), tokenLabel: 'dana' })} onClose={() => {}} />);
    await settle();
    expect(screen.getByTestId('popup-header').textContent).toMatch(/dana/);
  });

  it('Wrap toggles; Copy copies the plain text', async () => {
    answer(h.admin, resp({ text: 'plain <b>text</b>' }));
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('plain <b>text</b>') })} onClose={() => {}} />);
    await settle();
    const wrap = screen.getByRole('button', { name: /wrap/i });
    expect(wrap).toHaveAttribute('aria-pressed', 'true');
    expect(body()).toHaveAttribute('data-wrap', 'true');
    fireEvent.click(wrap);
    expect(wrap).toHaveAttribute('aria-pressed', 'false');
    expect(body()).toHaveAttribute('data-wrap', 'false');
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    });
    expect(h.copy).toHaveBeenCalledWith('plain <b>text</b>');
    expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument();
  });

  it('renders HTML and markdown model output literally, never as HTML', async () => {
    const evil = '<img src=x onerror="alert(1)"><script>alert(2)</script> **bold** <b>b</b>';
    answer(h.admin, resp({ kind: 'reply', text: evil }));
    render(<SectionPopup scope="admin" kind="reply" request={req({ reply: sec(evil) })} onClose={() => {}} />);
    await settle();
    expect(body().querySelector('img, script, b, strong')).toBeNull();
    expect(body().textContent).toContain(evil);
    // and with search marks on
    fireEvent.change(search(), { target: { value: 'bold' } });
    expect(body().querySelector('img, script, b, strong')).toBeNull();
    expect(body().textContent).toContain(evil);
  });
});

describe('SectionPopup: tool calls', () => {
  const calls = [
    { index: 0, call_id: 'call_1', name: 'bash', args: '{"cmd":"ls","n":2}' },
    { index: 1, call_id: 'call_2', name: 'edit', args: '{"path": "a.t' },
  ];

  it('tool calls render one block per call with pretty JSON; invalid JSON stays raw', async () => {
    answer(h.admin, resp({ kind: 'tool_calls', calls, live: true }));
    render(<SectionPopup scope="admin" kind="tool_calls" request={req()} onClose={() => {}} />);
    await settle();
    expect(screen.getAllByTestId(/^tool-call-\d+$/)).toHaveLength(2);
    const a0 = screen.getByTestId('tool-call-args-0');
    expect(a0).toHaveAttribute('data-json', 'true');
    expect(a0.textContent).toBe('{\n  "cmd": "ls",\n  "n": 2\n}');
    expect(a0.querySelector('.text-vw-info-fg')?.textContent).toBe('"cmd"');
    const a1 = screen.getByTestId('tool-call-args-1');
    expect(a1).toHaveAttribute('data-json', 'false');
    expect(a1.textContent).toBe('{"path": "a.t');
    // the live call streams; no call is "done" before request_end (G-F4)
    expect(screen.getByTestId('tool-call-1').textContent).toMatch(/arguments streaming/);
    expect(screen.getByTestId('tool-call-0').textContent).not.toMatch(/arguments streaming|arguments done/);
    expect(screen.getByTestId('tool-call-0').textContent).toMatch(/call_1/);
  });

  it('appends live argument deltas from the store and parses once complete', async () => {
    answer(h.admin, resp({ kind: 'tool_calls', calls: [calls[1]], live: true }));
    const tc = (args: string): ToolCallView => ({ index: 1, callId: 'call_2', name: 'edit', args, done: false, elided: false });
    const { rerender } = render(
      <SectionPopup scope="admin" kind="tool_calls" request={req({ toolCalls: [tc('{"path": "a')] })} onClose={() => {}} />,
    );
    await settle();
    rerender(<SectionPopup scope="admin" kind="tool_calls" request={req({ toolCalls: [tc('{"path": "a.txt"}')] })} onClose={() => {}} />);
    const a = screen.getByTestId('tool-call-args-1');
    expect(a).toHaveAttribute('data-json', 'true');
    expect(a.textContent).toBe('{\n  "path": "a.txt"\n}');
  });

  it('shows trimmed marks and what the server dropped; done only at request_end', async () => {
    answer(
      h.admin,
      resp({ kind: 'tool_calls', live: false, calls: [{ ...calls[0], elided: true }], dropped_chars: 1200, dropped_calls: 2 }),
    );
    render(<SectionPopup scope="admin" kind="tool_calls" request={req({ endedTs: 3 })} onClose={() => {}} />);
    await settle();
    expect(screen.getByTestId('tool-call-0').textContent).toMatch(/trimmed/);
    expect(screen.getByTestId('popup-marks').textContent).toMatch(/1,200 chars and 2 calls dropped on the server/);
  });

  it('search runs across calls; Expand all / Collapse all; Copy is a JSON array', async () => {
    answer(h.admin, resp({ kind: 'tool_calls', calls: [calls[0], { ...calls[1], args: '{"path":"ls.txt"}' }], live: false }));
    render(<SectionPopup scope="admin" kind="tool_calls" request={req({ endedTs: 3 })} onClose={() => {}} />);
    await settle();
    fireEvent.change(search(), { target: { value: 'ls' } });
    expect(count()).toBe('1 of 2');
    fireEvent.keyDown(search(), { key: 'Enter' });
    expect(screen.getByTestId('tool-call-1').querySelector('mark[data-current="true"]')).not.toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /collapse all/i }));
    expect(screen.getByTestId('tool-call-0')).not.toHaveAttribute('open');
    expect(screen.getByTestId('tool-call-1')).not.toHaveAttribute('open');
    fireEvent.click(screen.getByRole('button', { name: /expand all/i }));
    expect(screen.getByTestId('tool-call-0')).toHaveAttribute('open');

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    });
    expect(JSON.parse(h.copy.mock.calls[0][0])).toEqual([
      { name: 'bash', arguments: '{"cmd":"ls","n":2}' },
      { name: 'edit', arguments: '{"path":"ls.txt"}' },
    ]);
  });
});

describe('SectionPopup: tool results', () => {
  it('one block per result: tool name, call id, size, message index, new or cached', async () => {
    const tool = PROMPT.messages[1]; // i 15, read_file, call_8f2a
    answer(h.admin, resp({ kind: 'tool_results', live: false, messages: [tool, { ...tool, i: 16, tool_call_id: 'call_9', text: 'fresh' }] as never }));
    const r = reqWith({
      prompt: PROMPT,
      end: {},
      parts: [part('sys', -1, -1, 'exact'), part('nodes', 0, 15, 'exact'), part('nodes', 16, 16, 'new')],
    });
    render(<SectionPopup scope="admin" kind="tool_results" request={r} onClose={() => {}} />);
    await settle();
    const a = screen.getByTestId('result-block-15');
    expect(a.textContent).toMatch(/read_file/);
    expect(a.textContent).toMatch(/call_8f2a/);
    expect(a.textContent).toMatch(/900 chars/);
    // Numbered from 1, like the request screen and the map (body index 15).
    expect(a.textContent).toMatch(/msg 16\b/);
    expect(a.textContent).toMatch(/cached/);
    expect(screen.getByTestId('result-block-16').textContent).toMatch(/· new/);
    expect(screen.getByRole('button', { name: /collapse all/i })).toBeInTheDocument();
  });
});

describe('SectionPopup: scope and errors', () => {
  it('key scope fetches /api/me/prompt-lens/... and never the admin path', async () => {
    answer(h.key, resp({ text: 'mine', seq: null }));
    render(<SectionPopup scope="key" kind="thinking" request={req({ thinking: sec('mine') })} onClose={() => {}} />);
    await settle();
    expect(h.key).toHaveBeenCalledWith('/api/me/prompt-lens/requests/r1/section/thinking');
    expect(h.admin).not.toHaveBeenCalled();
    for (const [u] of h.key.mock.calls) expect(u).not.toMatch(/admin/);
  });

  it('admin scope fetches the admin path', async () => {
    answer(h.admin, resp({ text: 'x' }));
    render(<SectionPopup scope="admin" kind="reply" request={req({ reply: sec('x') })} onClose={() => {}} />);
    await settle();
    expect(h.admin).toHaveBeenCalledWith('/api/admin/prompt-lens/requests/r1/section/reply');
    expect(h.key).not.toHaveBeenCalled();
  });

  it('a 404 says the request left server memory', async () => {
    answer(h.key, new SessionsHttpError(404, 'no capture of yours with this id'));
    render(<SectionPopup scope="key" kind="thinking" request={req({ thinking: sec('the dock copy', 30) })} onClose={() => {}} />);
    await settle();
    expect(screen.getByRole('status').textContent).toMatch(/This request left server memory/);
    // the client copy stays, with its trimmed head said
    expect(body().textContent).toContain('the dock copy');
    expect(body().textContent).toMatch(/30 earlier chars/);
  });

  it('a 404 with nothing in the dock says it is not available', async () => {
    answer(h.admin, httpError(404));
    render(<SectionPopup scope="admin" kind="reply" request={req()} onClose={() => {}} />);
    await settle();
    expect(screen.getByRole('status').textContent).toMatch(/left server memory.*not available/i);
  });

  it('a 429 on a kept read says to retry, and Retry fetches again', async () => {
    answer(h.key, new SessionsHttpError(429, 'too many kept capture reads for this key; try again shortly'));
    render(<SectionPopup scope="key" kind="reply" request={req({ kept: true, endedTs: 3 })} onClose={() => {}} />);
    await settle();
    expect(screen.getByRole('status').textContent).toMatch(/too many reads.*try again in a minute/i);
    answer(h.key, resp({ kind: 'reply', text: 'kept reply', kept: true, live: false, seq: null }));
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    await settle();
    expect(h.key).toHaveBeenCalledTimes(2);
    expect(body().textContent).toContain('kept reply');
    expect(screen.getByTestId('popup-header').textContent).toMatch(/kept/);
  });

  it('a kept capture without a store copy loads its prompt from section/prompt', async () => {
    answer(
      h.key,
      resp({
        kind: 'prompt',
        live: false,
        seq: null,
        kept: true,
        messages: [
          { role: 'sys', chars: 12, text: 'You are kind', elided: false },
          { role: 'tools', count: 2, chars: 40, names: ['bash', 'edit'] },
          { i: 0, role: 'user', chars: 5, text: 'hello', elided: false },
        ] as never,
        older: { older: { from: 0, roles: '', chars: [] }, collapsed: 0, omitted: 0, total_chars: 57 } as never,
      }),
    );
    render(<SectionPopup scope="key" kind="prompt" request={req({ kept: true, endedTs: 3, prompt: null })} onClose={() => {}} />);
    await settle();
    expect(h.key).toHaveBeenCalledWith('/api/me/prompt-lens/requests/r1/section/prompt');
    expect(body().textContent).toMatch(/You are kind/);
    expect(body().textContent).toMatch(/bash, edit/);
    expect(body().textContent).toMatch(/hello/);
  });
});

describe('SectionPopup: waits', () => {
  it('shows the dock copy while the whole section loads', async () => {
    let release: (v: SectionResponse) => void = () => {};
    h.admin.mockImplementation(() => new Promise<SectionResponse>((r) => (release = r)));
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('partial') })} onClose={() => {}} />);
    expect(body().textContent).toContain('partial');
    expect(screen.getByRole('status').textContent).toMatch(/loading the whole section/i);
    await act(async () => release(resp({ text: 'whole partial' })));
    await waitFor(() => expect(body().textContent).toContain('whole partial'));
  });
});

describe('SectionPopup: review fixes', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('refetches live tool calls whose store copy does not continue the server copy, at most every 2 s', async () => {
    vi.useFakeTimers();
    answer(h.admin, resp({ kind: 'tool_calls', live: true, calls: [{ index: 0, call_id: 'c0', name: 'bash', args: '{"cmd":"l' }] }));
    // The store joined mid-call: its args are longer but do not continue the server's.
    const tc: ToolCallView = { index: 0, callId: 'c0', name: 'bash', args: 'md":"ls -la /tmp"', done: false, elided: false };
    render(<SectionPopup scope="admin" kind="tool_calls" request={req({ toolCalls: [tc] })} onClose={() => {}} />);
    await settle();
    expect(h.admin).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1900);
    });
    expect(h.admin).toHaveBeenCalledTimes(1);
    answer(h.admin, resp({ kind: 'tool_calls', live: true, calls: [{ index: 0, call_id: 'c0', name: 'bash', args: '{"cmd":"ls -la /tmp"' }] }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(h.admin).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId('tool-call-args-0').textContent).toBe('{"cmd":"ls -la /tmp"');
  });

  it('always refetches tool calls once at request_end', async () => {
    answer(h.admin, resp({ kind: 'tool_calls', live: true, calls: [{ index: 0, call_id: 'c0', name: 'bash', args: '{"cmd":"ls"' }] }));
    const { rerender } = render(<SectionPopup scope="admin" kind="tool_calls" request={req()} onClose={() => {}} />);
    await settle();
    expect(h.admin).toHaveBeenCalledTimes(1);
    answer(h.admin, resp({ kind: 'tool_calls', live: false, calls: [{ index: 0, call_id: 'c0', name: 'bash', args: '{"cmd":"ls"}' }] }));
    rerender(<SectionPopup scope="admin" kind="tool_calls" request={req({ endedTs: 3 })} onClose={() => {}} />);
    await settle();
    await settle();
    expect(h.admin).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId('tool-call-args-0')).toHaveAttribute('data-json', 'true');
    rerender(<SectionPopup scope="admin" kind="tool_calls" request={req({ endedTs: 3 })} onClose={() => {}} />);
    await settle();
    expect(h.admin).toHaveBeenCalledTimes(2);
  });

  it('a failed resume fetch retries with backoff', async () => {
    vi.useFakeTimers();
    answer(h.admin, resp({ text: 'alpha beta gamma ', offset: 0 }));
    const { rerender } = render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma ', 11) })} onClose={() => {}} />);
    await settle();
    answer(h.admin, httpError(503));
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('kappa ', 51) })} onClose={() => {}} />);
    await settle();
    expect(h.admin).toHaveBeenCalledTimes(2);
    answer(h.admin, resp({ text: 'delta epsilon zeta eta theta iota kappa ', offset: 17 }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(h.admin).toHaveBeenCalledTimes(3);
    expect(h.admin).toHaveBeenLastCalledWith('/api/admin/prompt-lens/requests/r1/section/thinking?from=17');
    expect(body().textContent).toContain('alpha beta gamma delta epsilon zeta eta theta iota kappa ');
  });

  it('gives up resuming after 5 retries', async () => {
    vi.useFakeTimers();
    answer(h.admin, resp({ text: 'alpha beta gamma ', offset: 0 }));
    const { rerender } = render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('gamma ', 11) })} onClose={() => {}} />);
    await settle();
    answer(h.admin, httpError(503));
    rerender(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('kappa ', 51) })} onClose={() => {}} />);
    for (let k = 0; k < 12; k++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
    }
    expect(h.admin).toHaveBeenCalledTimes(1 + 1 + 5);
  });

  it('Copy is disabled, and says why, until the whole section is loaded', async () => {
    let release: (v: SectionResponse) => void = () => {};
    h.admin.mockImplementation(() => new Promise<SectionResponse>((r) => (release = r)));
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('partial') })} onClose={() => {}} />);
    const copy = screen.getByRole('button', { name: /^copy$/i });
    expect(copy).toBeDisabled();
    expect(copy.getAttribute('title')).toMatch(/loading/i);
    await act(async () => release(resp({ text: 'whole partial' })));
    await waitFor(() => expect(screen.getByRole('button', { name: /^copy$/i })).not.toBeDisabled());
  });

  it('Copy stays disabled when the section could not be loaded', async () => {
    answer(h.admin, httpError(404));
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('dock copy') })} onClose={() => {}} />);
    await settle();
    const copy = screen.getByRole('button', { name: /^copy$/i });
    expect(copy).toBeDisabled();
    expect(copy.getAttribute('title')).toMatch(/not loaded/i);
  });

  it('Esc reaches no later listener, capture or bubble', async () => {
    answer(h.admin, resp({ text: 'x' }));
    const onClose = vi.fn();
    render(<SectionPopup scope="admin" kind="thinking" request={req({ thinking: sec('x') })} onClose={onClose} />);
    await settle();
    const laterCapture = vi.fn();
    const bubble = vi.fn();
    window.addEventListener('keydown', laterCapture, true);
    document.addEventListener('keydown', bubble);
    try {
      fireEvent.keyDown(search(), { key: 'Escape' });
      expect(onClose).toHaveBeenCalledTimes(1);
      expect(laterCapture).not.toHaveBeenCalled();
      expect(bubble).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('keydown', laterCapture, true);
      document.removeEventListener('keydown', bubble);
    }
  });
});

describe('SectionPopup: phone toolbar (Task 17)', () => {
  const calls = [{ index: 0, call_id: 'call_1', name: 'bash', args: '{"cmd":"ls"}' }];

  it('the second toolbar row stays one row: short labels, sideways scroll as a last resort', async () => {
    answer(h.admin, resp({ kind: 'tool_calls', calls, live: true }));
    render(<SectionPopup scope="admin" kind="tool_calls" request={req()} onClose={() => {}} phone />);
    await settle();
    const tools = screen.getByTestId('popup-tools');
    expect(tools).toHaveClass('flex-nowrap', 'overflow-x-auto', 'w-full');
    expect(tools).not.toHaveClass('flex-wrap');
    // Short visible labels on a phone; the accessible names stay whole.
    expect(screen.getByRole('button', { name: 'Expand all' })).toHaveTextContent(/^Expand$/);
    expect(screen.getByRole('button', { name: 'Collapse all' })).toHaveTextContent(/^Collapse$/);
  });

  it('desktop keeps the wrapping row and the long labels', async () => {
    answer(h.admin, resp({ kind: 'tool_calls', calls, live: true }));
    render(<SectionPopup scope="admin" kind="tool_calls" request={req()} onClose={() => {}} />);
    await settle();
    expect(screen.getByTestId('popup-tools')).toHaveClass('flex-wrap');
    expect(screen.getByRole('button', { name: 'Expand all' })).toHaveTextContent(/^Expand all$/);
  });
});
