import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';

import type { RequestView, Section, ToolCallView } from '@/lib/prompt-lens/store';
import { K } from '@/lib/key-area/strings';

import { PROMPT, part, reqWith } from '../unit/prompt-lens-kvsig-fixtures';

const h = vi.hoisted(() => ({ admin: vi.fn(), key: vi.fn() }));

vi.mock('@/lib/auth-fetch', async (orig) => ({
  ...(await orig<typeof import('@/lib/auth-fetch')>()),
  authFetchJSON: (u: string) => h.admin(u),
  authFetch: vi.fn(async () => new Response(new Blob(['x']))),
}));
vi.mock('@/lib/sessions/client', async (orig) => ({
  ...(await orig<typeof import('@/lib/sessions/client')>()),
  sessionsFetcher: () => (u: string) => h.key(u),
}));
vi.mock('@podwarden/chat-ui', () => ({ copyToClipboard: vi.fn() }));

import { RequestScreen } from '@/components/prompt-lens/dock/request-screen';
import { useCollapseMemory } from '@/components/prompt-lens/dock/section-box';
import { resetServerClock } from '@/lib/prompt-lens/server-clock';

beforeEach(() => {
  h.admin.mockReset();
  h.key.mockReset();
  h.admin.mockImplementation(() => new Promise(() => {}));
  h.key.mockImplementation(() => new Promise(() => {}));
  resetServerClock();
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------

function sec(text: string, o: Partial<Section> = {}): Section {
  return { text, chars: (o.clipped ?? 0) + text.length, clipped: 0, frames: 0, firstTs: null, lastTs: null, ...o };
}

const PARTS = [
  part('sys', -1, -1, 'exact', { chars: 1200 }),
  part('tools', -1, -1, 'exact', { chars: 3600 }),
  part('nodes', 0, 13, 'exact', { chars: 9800, roles: { user: 5, assistant: 5, tool: 4 } }),
  part('nodes', 14, 14, 'loose', { chars: 700, roles: { assistant: 1 } }),
  part('nodes', 15, 15, 'new', { chars: 900, roles: { tool: 1 } }),
  part('nodes', 16, 16, 'new', { chars: 160, roles: { user: 1 } }),
];

/** Turn 17, finished, with an admin-only label, IP and cached figure on it. */
function finished(o: Partial<RequestView> = {}): RequestView {
  const base = reqWith({
    parts: PARTS,
    prompt: PROMPT,
    end: {
      prompt_tokens: 43920,
      completion_tokens: 1024,
      reasoning_tokens: 410,
      cached_tokens: 8740,
      own_reused_tokens: 6120,
      queued_s: 0.5,
      ttft_s: 1.25,
      duration_s: 14.6,
      decode_tps: 57,
    },
  });
  return {
    ...base,
    tokenLabel: 'Dana',
    clientIp: '192.0.2.54',
    sessionId: '3f9a0c21deadbeef',
    turn: 17,
    servedName: 'qwen3.8-27b-fp8',
    finish: 'stop',
    reply: sec('The answer.', { frames: 3, firstTs: 5, lastTs: 6 }),
    ...o,
  };
}

/** Turn 17, live, answering (the reply has started after the thinking). */
function live(o: Partial<RequestView> = {}): RequestView {
  return finished({
    end: null,
    endedTs: null,
    finish: null,
    phase: 'answering',
    thinking: sec('Let me think.', { frames: 62, firstTs: 100, lastTs: 106.2 }),
    reply: sec('Here is', { frames: 20, firstTs: 106.2, lastTs: 107.2 }),
    frames: { reasoning: 62, content: 20, tool_call: 0 },
    startedTs: 99,
    ...o,
  });
}

function Screen({
  r,
  scope,
  phone = false,
  keepsCaptures,
}: {
  r: RequestView;
  scope: 'key' | 'admin';
  phone?: boolean;
  keepsCaptures?: boolean;
}) {
  const collapse = useCollapseMemory();
  return <RequestScreen request={r} scope={scope} phone={phone} collapse={collapse} keepsCaptures={keepsCaptures} />;
}

/** A screen whose request can be swapped from the test. */
function Swappable({ initial }: { initial: RequestView }) {
  const [r, setR] = useState(initial);
  (Swappable as unknown as { set: (r: RequestView) => void }).set = setR;
  return <Screen r={r} scope="admin" />;
}

// ---------------------------------------------------------------------------

describe('RequestScreen: scope', () => {
  it('key scope shows no IP, no key label, no cached figure, and the privacy line', () => {
    const { container } = render(<Screen r={finished()} scope="key" />);
    const text = container.textContent ?? '';
    expect(text).not.toContain('192.0.2.54');
    expect(text).not.toContain('Dana');
    expect(text).not.toContain('8,740');
    expect(text).not.toMatch(/engine cached/i);
    expect(text).not.toMatch(/another key/i);
    expect(text).toContain(K.stats.capture);
    // The key holder's own figure, by its name.
    const usage = screen.getByTestId('usage');
    expect(usage.textContent).toMatch(/reused from your earlier requests\s*6,120/);
    // The title has no key label.
    expect(screen.getByTestId('rs-title').textContent).toBe('session 3f9a0c21 · turn 17');
  });

  it('admin scope shows the client IP and the engine cached figure', () => {
    render(<Screen r={finished()} scope="admin" />);
    expect(screen.getByTestId('rs-facts').textContent).toContain('192.0.2.54');
    expect(screen.getByTestId('rs-title').textContent).toBe('Dana · session 3f9a0c21 · turn 17');
    expect(screen.getByTestId('usage').textContent).toMatch(/engine cached\s*8,740/);
    expect(screen.queryByText(K.stats.capture)).toBeNull();
  });
});

describe('RequestScreen: header and facts', () => {
  it('a finished request shows its finish chip, model, total wait, duration and decode', () => {
    render(<Screen r={finished()} scope="admin" />);
    expect(screen.getByTestId('rs-chip').textContent).toBe('stop');
    expect(screen.getByText('qwen3.8-27b-fp8')).toBeTruthy();
    const facts = screen.getByTestId('rs-facts').textContent ?? '';
    expect(facts).toContain('queued0.5 s');
    // total wait = ttft_s + queued_s
    expect(facts).toContain('first token1.75 s');
    expect(facts).toContain('duration14.6 s');
    expect(facts).toContain('prompt43,920');
    expect(facts).toContain('out1,024');
    expect(facts).toContain('decode57 tok/s');
  });

  it('finished comes from endedTs, not phase (a kept row reads queued)', () => {
    render(<Screen r={finished({ phase: 'queued', kept: true })} scope="key" />);
    expect(screen.getByTestId('rs-chip').textContent).toBe('stop');
    expect(screen.queryByText(/after admission/)).toBeNull();
    expect(screen.getByText('kept')).toBeTruthy();
  });

  it('a live request shows its phase, elapsed time and an approximate live rate', () => {
    vi.useFakeTimers();
    vi.setSystemTime(109_000);
    render(<Screen r={live()} scope="admin" />);
    expect(screen.getByTestId('rs-chip').textContent).toBe('answering');
    const facts = screen.getByTestId('rs-facts').textContent ?? '';
    expect(facts).toContain('elapsed10 s');
    expect(facts).toMatch(/now≈ \d+ tok\/s/);
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByTestId('rs-facts').textContent).toContain('elapsed12 s');
  });

  it('shows the raw tool_call phase as words', () => {
    render(<Screen r={live({ phase: 'tool_call' })} scope="admin" />);
    expect(screen.getByTestId('rs-chip').textContent).toBe('tool call');
  });

  it('calls onBack from the back button', () => {
    const back = vi.fn();
    function WithBack() {
      const collapse = useCollapseMemory();
      return <RequestScreen request={finished()} scope="admin" phone={false} collapse={collapse} onBack={back} />;
    }
    render(<WithBack />);
    fireEvent.click(screen.getByRole('button', { name: /requests/i }));
    expect(back).toHaveBeenCalled();
  });
});

describe('RequestScreen: non-streamed', () => {
  it('non-streamed request shows the one-piece banner and no live rate', () => {
    const r = finished({ stream: false, endedTs: 1_700_000_000 });
    const { container } = render(<Screen r={r} scope="admin" />);
    const d = new Date(1_700_000_000 * 1000);
    const p = (n: number) => String(n).padStart(2, '0');
    const hms = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
    expect(screen.getByTestId('rs-banner').textContent).toContain(`arrived in one piece at ${hms}`);
    expect(container.textContent).not.toMatch(/≈ [\d,]+ tok\/s/);
    expect(screen.getByTestId('rs-facts').textContent).toContain('first tokenn/a');
    expect(screen.getByTestId('rs-facts').textContent).toContain('avg57 tok/s');
    expect(screen.getByTestId('usage').textContent).toMatch(/decode\s*57 tok\/s average/);
    expect(screen.getByText('non-streamed')).toBeTruthy();
  });

  it('a non-streamed runaway (an end and no reply deltas) says why the reply is empty', () => {
    const r = finished({ stream: false, finish: 'runaway', reply: sec('') });
    render(<Screen r={r} scope="admin" />);
    expect(screen.getByTestId('section-reply').textContent).toMatch(/stopped as a runaway/);
  });
});

describe('RequestScreen: output sections', () => {
  it("thinking shows 'done in' once the reply starts", () => {
    render(<Screen r={live()} scope="admin" />);
    expect(screen.getByTestId('section-thinking').textContent).toContain('done in 6.2 s');
    // The reply streams with an approximate rate and a caret.
    const reply = screen.getByTestId('section-reply');
    expect(reply.textContent).toMatch(/≈ 20 tok\/s/);
    expect(within(reply).getByTestId('caret-reply')).toBeTruthy();
  });

  it('thinking streams with a caret while the reply has not started', () => {
    render(<Screen r={live({ reply: sec(''), phase: 'thinking' })} scope="admin" />);
    const th = screen.getByTestId('section-thinking');
    expect(within(th).getByTestId('caret-thinking')).toBeTruthy();
    expect(th.textContent).not.toContain('done in');
    expect(screen.getByTestId('section-reply').textContent).toContain('The reply starts when thinking ends.');
  });

  it('renders HTML and markdown model output literally, never as HTML', () => {
    const evil = '<img src=x onerror="alert(1)"><b>bold</b> **md**';
    const { container } = render(
      <Screen r={finished({ thinking: sec(evil), reply: sec(evil) })} scope="admin" />,
    );
    expect(container.querySelector('img[src="x"]')).toBeNull();
    expect([...container.querySelectorAll('b')].some((b) => b.textContent === 'bold')).toBe(false);
    expect(screen.getByTestId('section-reply').textContent).toContain(evil);
    expect(screen.getByTestId('section-thinking').textContent).toContain(evil);
  });

  it('keys the streaming text by request, so a look-alike tail never keeps the old head', () => {
    const tail = 'z'.repeat(80);
    const a = finished({ reqId: 'rA', reply: sec('A'.repeat(50) + tail) });
    const b = finished({ reqId: 'rB', reply: sec('B'.repeat(50) + tail) });
    render(<Swappable initial={a} />);
    expect(screen.getByTestId('section-reply').textContent).toContain('AAAA');
    act(() => (Swappable as unknown as { set: (r: RequestView) => void }).set(b));
    const t = screen.getByTestId('section-reply').textContent ?? '';
    expect(t).toContain('BBBB');
    expect(t).not.toContain('AAAA');
  });

  it('tool calls show streaming arguments live and "arguments done" at the end', () => {
    const calls: ToolCallView[] = [
      { index: 0, callId: 'call_9b04', name: 'read_file', args: '{"path":"a.ts"}', done: false, elided: false },
      { index: 1, callId: 'call_9b05', name: 'edit', args: '{"path":"b', done: false, elided: false },
    ];
    const { rerender } = render(<Screen r={live({ toolCalls: calls, phase: 'tool_call' })} scope="admin" />);
    const box = screen.getByTestId('section-tool_calls');
    expect(box.textContent).toContain('arguments streaming');
    expect(box.textContent).not.toContain('arguments done');
    rerender(<Screen r={finished({ toolCalls: calls.map((c) => ({ ...c, done: true, args: '{"path":"b.ts"}' })) })} scope="admin" />);
    expect(screen.getByTestId('section-tool_calls').textContent).toContain('arguments done');
    expect(screen.getByTestId('section-tool_calls').textContent).toContain('2 calls');
  });

  it('usage says it fills in at the end, then shows the figures and the dropped counters', () => {
    const { rerender } = render(<Screen r={live()} scope="admin" />);
    expect(screen.getByTestId('usage').textContent).toContain('Filled in when the stream ends');
    const r = finished();
    rerender(
      <Screen
        r={{ ...r, end: { ...r.end!, tool_calls_dropped: 2, tool_call_chars_dropped: 5000 } }}
        scope="admin"
      />,
    );
    const u = screen.getByTestId('usage').textContent ?? '';
    expect(u).toMatch(/prompt\s*43,920/);
    expect(u).toMatch(/completion\s*1,024/);
    expect(u).toMatch(/of which thinking\s*410/);
    expect(u).toMatch(/first token\s*1.75 s/);
    expect(u).toMatch(/duration\s*14.6 s/);
    expect(u).toMatch(/decode\s*57 tok\/s/);
    expect(u).toMatch(/finish\s*stop/);
    expect(u).toMatch(/tool calls dropped\s*2/);
    expect(u).toMatch(/tool-call chars dropped\s*5,000/);
  });

  it('marks a section the dock trimmed', () => {
    render(<Screen r={finished({ reply: sec('end', { clipped: 1200 }) })} scope="admin" />);
    expect(screen.getByTestId('section-marks-reply').textContent).toContain('first 1,200 chars trimmed');
    expect(screen.getByTestId('rs-trimmed')).toBeTruthy();
  });
});

describe('RequestScreen: input sections', () => {
  it('lists tool results newest first, 8 lines each, new or cached', () => {
    const lines = Array.from({ length: 20 }, (_, k) => `line ${k + 1}`).join('\n');
    const prompt = {
      ...PROMPT,
      messages: [
        { i: 13, role: 'tool', chars: 300, text: lines, elided: false, tool_call_id: 'call_old', tool_name: 'grep' },
        ...PROMPT.messages,
      ],
    };
    render(<Screen r={finished({ prompt })} scope="admin" />);
    const box = screen.getByTestId('section-tool_results');
    const cards = within(box).getAllByTestId('tool-result');
    expect(cards.map((c) => c.getAttribute('data-msg'))).toEqual(['15', '13']);
    expect(cards[1].textContent).toContain('line 8');
    expect(cards[1].textContent).not.toContain('line 9');
    expect(cards[1].textContent).toContain('12 more lines');
    expect(cards[0].textContent).toContain('new');
    expect(cards[1].textContent).toContain('cached');
  });

  it('queued request says the prompt map comes after admission', () => {
    const r = { ...reqWith({ kvsig: null, phase: 'queued' }), prompt: PROMPT };
    render(<Screen r={r} scope="admin" />);
    expect(screen.getByTestId('kvsig-empty').textContent).toMatch(/prompt map after admission/);
  });

  it('says why there is no prompt map', () => {
    render(<Screen r={reqWith({ kvsig: { available: false, reason: 'not_chat' }, end: {} })} scope="admin" />);
    expect(screen.getByTestId('kvsig-empty').textContent).toMatch(/not a chat request/);
  });

  it('prompt by role lists sys and tools first, collapsed messages as "msg i · role · N chars"', () => {
    render(<Screen r={finished()} scope="admin" />);
    // Prompt by role starts collapsed.
    fireEvent.click(screen.getByRole('button', { name: 'Show Prompt by role' }));
    const rows = within(screen.getByTestId('section-prompt')).getAllByTestId('prompt-row');
    expect(rows[0].getAttribute('data-msg')).toBe('sys');
    expect(rows[1].getAttribute('data-msg')).toBe('tools');
    expect(rows[1].textContent).toContain('24 tools');
    expect(rows[2].textContent).toBe('msg 1 · user · 700 chars');
    expect(rows[3].textContent).toBe('msg 2 · assistant · 700 chars');
    expect(rows.at(-1)!.textContent).toContain('now wire the dock');
  });

  it('clicking a segment opens Prompt by role at that message', () => {
    render(<Screen r={finished()} scope="admin" />);
    expect(screen.queryByTestId('section-box-prompt')).toBeNull(); // collapsed by default
    const segs = screen.getAllByTestId('kvsig-seg');
    fireEvent.click(segs[4]); // node 16: the read_file tool result (body index 15)
    const box = screen.getByTestId('section-box-prompt');
    const active = within(box).getAllByTestId('prompt-row').filter((r) => r.getAttribute('aria-current') === 'true');
    expect(active).toHaveLength(1);
    expect(active[0].getAttribute('data-msg')).toBe('15');
    expect(segs[4].className).toContain('outline-chat-fg');

    // A run's first message: nodes 1-14 start at the collapsed message 1.
    fireEvent.click(segs[2]);
    const now = within(box).getAllByTestId('prompt-row').filter((r) => r.getAttribute('aria-current') === 'true');
    expect(now[0].getAttribute('data-msg')).toBe('0');
    expect(now[0].textContent).toBe('msg 1 · user · 700 chars');
  });

  it('puts a parts table of more than 8 rows inside a section box', () => {
    const many = [
      part('sys', -1, -1, 'exact'),
      part('tools', -1, -1, 'exact'),
      ...Array.from({ length: 8 }, (_, k) => part('nodes', k, k, k % 2 ? 'new' : 'exact')),
    ];
    render(<Screen r={finished({ kvsig: { ...finished().kvsig!, parts: many } })} scope="admin" />);
    expect(screen.getByTestId('section-parts')).toBeTruthy();
    expect(screen.getByTestId('section-box-parts').querySelector('table')).toBeTruthy();
  });

  it('shows an image placeholder to a key holder (no key media route)', () => {
    render(<Screen r={finished({ media: [{ kind: 'image', media_id: 'm1', chars: 100 }] })} scope="key" />);
    expect(screen.getByText('image · 1')).toBeTruthy();
    expect(screen.queryByTestId('prompt-lens-media-strip')).toBeNull();
  });

  it('shows the media strip to the admin', () => {
    render(<Screen r={finished({ media: [{ kind: 'image', url: 'https://example.com/a.png' }] })} scope="admin" />);
    expect(screen.getByTestId('prompt-lens-media-strip')).toBeTruthy();
  });
});

describe('RequestScreen: pop-up and naming', () => {
  it('⤢ opens the section pop-up and closes it again', () => {
    render(<Screen r={finished()} scope="admin" />);
    fireEvent.click(screen.getByRole('button', { name: 'Open Reply in a large view' }));
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(h.admin).toHaveBeenCalledWith('/api/admin/prompt-lens/requests/r1/section/reply');
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('never renders the word cobs2', () => {
    for (const scope of ['key', 'admin'] as const) {
      for (const r of [finished(), live(), finished({ stream: false })]) {
        const { container, unmount } = render(<Screen r={r} scope={scope} />);
        fireEvent.click(screen.getByRole('button', { name: 'Show Prompt by role' }));
        expect(container.textContent).not.toMatch(/cobs2/i);
        unmount();
      }
    }
  });

  it('stacks the columns on a phone', () => {
    render(<Screen r={finished()} scope="admin" phone />);
    expect(screen.getByTestId('rs-cols').className).not.toContain('grid-cols-2');
  });
});

describe('RequestScreen: review round 1 (G-R14 and minors)', () => {
  it('a key that keeps captures gets the on-disk privacy line', () => {
    const { rerender } = render(<Screen r={finished()} scope="key" keepsCaptures />);
    expect(screen.getByText(K.stats.captureKept)).toBeTruthy();
    expect(screen.queryByText(K.stats.capture)).toBeNull();
    expect(K.stats.captureKept).toContain('the last 50 captures are kept on disk for this key');
    expect(K.stats.captureKept).not.toContain('not on disk');
    // A kept capture is on disk whatever the switch says now.
    rerender(<Screen r={finished({ kept: true })} scope="key" keepsCaptures={false} />);
    expect(screen.getByText(K.stats.captureKept)).toBeTruthy();
    expect(screen.getByText('kept').getAttribute('title')).toMatch(/on disk for this key/);
  });

  it('shows the admission wait from the kvsig event once admitted', () => {
    const r = live();
    render(<Screen r={{ ...r, kvsig: { ...r.kvsig!, queued_s: 0.42 } }} scope="admin" />);
    expect(screen.getByTestId('rs-facts').textContent).toContain('queued0.42 s');
  });

  it('the live rate counts thinking and reply frames only, not tool-call frames', () => {
    vi.useFakeTimers();
    vi.setSystemTime(109_000);
    render(<Screen r={live({ frames: { reasoning: 62, content: 20, tool_call: 500 } })} scope="admin" />);
    // 82 frames over 100 .. 107.2 s
    expect(screen.getByTestId('rs-facts').textContent).toContain('now≈ 11 tok/s');
  });

  it('key scope renders a stray other part as new, never as another key', () => {
    const parts = [
      part('sys', -1, -1, 'other', { chars: 1200 }),
      part('tools', -1, -1, 'exact', { chars: 3600 }),
      part('nodes', 0, 13, 'other', { chars: 9800 }),
      part('nodes', 14, 16, 'new', { chars: 1760 }),
    ];
    const r = finished();
    const { container } = render(
      <Screen r={{ ...r, kvsig: { ...r.kvsig!, parts, reusable_any: 40000 } }} scope="key" />,
    );
    const states = screen.getAllByTestId('kvsig-seg').map((s) => s.getAttribute('data-state'));
    expect(states).not.toContain('other');
    expect(states[0]).toBe('new');
    expect(container.textContent).not.toMatch(/another key|any-key|40,000/i);
  });

  it('scroll regions take the keyboard, the title has its full text, the chip is announced', () => {
    render(<Screen r={finished()} scope="admin" />);
    expect(screen.getByTestId('rs-facts').getAttribute('tabindex')).toBe('0');
    expect(screen.getByTestId('usage-box').getAttribute('tabindex')).toBe('0');
    expect(screen.getByTestId('rs-title').getAttribute('title')).toBe('Dana · session 3f9a0c21 · turn 17');
    expect(screen.getByTestId('rs-chip').getAttribute('aria-live')).toBe('polite');
  });

  it('elapsed follows the server clock when the client clock is off', () => {
    vi.useFakeTimers();
    // The client clock is 1,000 s ahead of the server's.
    vi.setSystemTime(1_109_000);
    const r = live();
    const { rerender } = render(<Screen r={r} scope="admin" />);
    // A new event arrives (server ts 108) at client 1,109 s.
    rerender(<Screen r={{ ...r, reply: { ...r.reply, lastTs: 108, frames: 21 } }} scope="admin" />);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByTestId('rs-facts').textContent).toContain('elapsed10 s');
  });

  it('after the end, thinking shows ≈ tokens and its time, as mockup state 4', () => {
    render(
      <Screen
        r={finished({ thinking: sec('Let me think.', { frames: 62, firstTs: 100, lastTs: 106 }), reply: sec('A', { firstTs: 106.2, lastTs: 107 }) })}
        scope="admin"
      />,
    );
    const th = screen.getByTestId('section-thinking').textContent ?? '';
    expect(th).toContain('≈ 410 tok · 6.2 s');
    expect(th).not.toContain('done in');
  });

  it('first token on a tool-call turn with no thinking comes from the first tool-call delta', () => {
    const calls: ToolCallView[] = [{ index: 0, callId: 'c', name: 'bash', args: '{', done: false, elided: false }];
    render(
      <Screen
        r={live({ thinking: sec(''), reply: sec(''), toolCalls: calls, toolCallFirstTs: 101.5, phase: 'tool_call' })}
        scope="admin"
      />,
    );
    expect(screen.getByTestId('rs-facts').textContent).toContain('first token2.5 s');
  });
});
