// Component tests for the god-mode live viewer.
//
// Reuses the log-stream test harness: a FakeES stub drives onopen/onmessage,
// and useEventSource mints a ticket via POST /api/auth/sse-ticket which we
// stub. Unlike the global react-virtuoso shim in tests/setup.ts (which drops
// followOutput/atBottomStateChange), this file installs a LOCAL Virtuoso mock
// that CAPTURES those props — that's what lets us assert the fast-load fix
// (followOutput stays "auto" through a transient not-at-bottom) at the
// component boundary, i.e. the regression test for the model-loading
// fast-load-enters-explore bug the user reported.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup, fireEvent, within } from '@testing-library/react';

// Local Virtuoso mock — renders all rows AND records the last props so tests
// can drive atBottomStateChange and read followOutput. `vi.hoisted` makes the
// capture object available inside the hoisted vi.mock factory.
const vh = vi.hoisted(() => ({ lastProps: null as any, scrollTo: null as any, scroller: null as any, scrollHeight: 0 }));
vi.mock('react-virtuoso', () => {
  const React = require('react');
  function Virtuoso(props: any) {
    vh.lastProps = props;
    // React 19 passes `ref` as a plain prop; expose a spy-able handle.
    React.useImperativeHandle(props.ref, () => ({ scrollToIndex: () => {}, scrollTo: (o: unknown) => vh.scrollTo?.(o) }), []);
    React.useEffect(() => { props.scrollerRef?.(vh.scroller); }, [props.scrollerRef]);
    const { data = [], itemContent, computeItemKey, components, style } = props;
    const children = data.map((item: unknown, idx: number) => {
      const key = computeItemKey ? computeItemKey(idx, item) : idx;
      return React.createElement(
        'div',
        { key, 'data-virtuoso-row-index': idx },
        itemContent ? itemContent(idx, item) : null,
      );
    });
    const ListComp = components?.List;
    if (ListComp) {
      return React.createElement('div', { style }, React.createElement(ListComp, {}, children));
    }
    return React.createElement('div', { style }, children);
  }
  return { Virtuoso };
});

import { createRef, type ComponentProps } from 'react';
import { GodModeViewer, clearFinished, MAX_EVENTS, type GodModeViewerHandle } from '@/components/godmode/godmode-viewer';
import { reqColor } from '@/components/godmode/req-color';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

class FakeES {
  static last: FakeES;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) {
    FakeES.last = this;
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {
    this.closed = true;
  }
}

function push(ev: Record<string, unknown>) {
  act(() => {
    FakeES.last.onmessage?.(new MessageEvent('message', { data: JSON.stringify(ev) }));
  });
}

async function mountConnected(props: Partial<ComponentProps<typeof GodModeViewer>> = {}) {
  vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}')));
  render(<GodModeViewer tokenIds={['tok_abc']} {...props} />);
  // Flush the ticket-mint promise + the FakeES onopen setTimeout.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

describe('GodModeViewer', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vh.lastProps = null;
    vh.scrollTo = vi.fn();
    // A scroller 400 px tall showing the bottom of 1000 px of content.
    vh.scroller = document.createElement('div');
    vh.scrollHeight = 1000;
    Object.defineProperty(vh.scroller, 'scrollHeight', { configurable: true, get: () => vh.scrollHeight });
    Object.defineProperty(vh.scroller, 'clientHeight', { configurable: true, get: () => 400 });
    vh.scroller.scrollTop = 600;
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('groups deltas under a request block header with token label, model and client', async () => {
    await mountConnected();

    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'harness-key', token_id: 'tok_abc', model: 'llama-3', served_name: 'llama', client_ip: '10.0.0.5', stream: true, prompt: 'hello there' });
    push({ seq: 2, type: 'delta', req_id: 'r1', ts: 1, channel: 'content', text: 'Hi! ' });
    push({ seq: 3, type: 'delta', req_id: 'r1', ts: 2, channel: 'content', text: 'How can I help?' });

    expect(screen.getByText('harness-key')).toBeInTheDocument();
    expect(screen.getByText('llama-3')).toBeInTheDocument();
    expect(screen.getByText('10.0.0.5')).toBeInTheDocument();
    expect(screen.getByText(/hello there/)).toBeInTheDocument();
    expect(screen.getByText('Hi!', { exact: false })).toBeInTheDocument();
    expect(screen.getByRole('log')).toBeInTheDocument();

    // finish_reason surfaces once the request ends.
    push({ seq: 4, type: 'request_end', req_id: 'r1', ts: 3, finish_reason: 'stop', prompt_tokens: 3, completion_tokens: 5 });
    expect(screen.getByTestId('godmode-finish')).toHaveTextContent('stop');
  });

  it('renders reasoning distinct from content but sharing the request color', async () => {
    await mountConnected();

    push({ seq: 1, type: 'request_start', req_id: 'r7', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    push({ seq: 2, type: 'delta', req_id: 'r7', ts: 1, channel: 'reasoning', text: 'let me think' });
    push({ seq: 3, type: 'delta', req_id: 'r7', ts: 2, channel: 'content', text: 'the answer is 42' });

    const reasoning = document.querySelector('[data-channel="reasoning"]') as HTMLElement;
    const content = document.querySelector('[data-channel="content"]') as HTMLElement;
    expect(reasoning).toBeTruthy();
    expect(content).toBeTruthy();
    // Visually distinct — reasoning is italicized/dimmed.
    expect(reasoning.className).toMatch(/italic/);
    expect(content.className).not.toMatch(/italic/);

    // Both live inside the same block, which carries the request's color on
    // its left-edge accent bar — so the two channels share the request color.
    const block = screen.getByTestId('godmode-block');
    expect(block).toContainElement(reasoning);
    expect(block).toContainElement(content);
    // First request in the session → index 0 → golden-angle hue. jsdom
    // normalizes inline hsl() to rgb(), so compare through a probe element
    // that goes through the same normalization rather than the raw string.
    const probe = document.createElement('div');
    probe.style.borderLeftColor = reqColor('r7', 0).accent;
    expect(block.style.borderLeftColor).toBe(probe.style.borderLeftColor);
    expect(block.style.borderLeftColor).not.toBe('');
  });

  it('gives two interleaved requests distinguishable block colors', async () => {
    await mountConnected();

    push({ seq: 1, type: 'request_start', req_id: 'a', ts: 0, token_label: 'ka', token_id: 'ta', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'pa' });
    push({ seq: 2, type: 'request_start', req_id: 'b', ts: 0, token_label: 'kb', token_id: 'tb', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'pb' });

    const blocks = screen.getAllByTestId('godmode-block');
    expect(blocks).toHaveLength(2);
    expect(blocks[0].style.borderLeftColor).not.toBe(blocks[1].style.borderLeftColor);
  });

  it('live-follow: mounts in stick mode with instant follow', async () => {
    await mountConnected();
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });

    // followOutput is the always-on "auto" from the shared hook; no
    // "Jump to latest" button while stuck to the tail.
    expect(vh.lastProps.followOutput).toBe('auto');
    expect(screen.queryByRole('button', { name: /jump to latest/i })).not.toBeInTheDocument();
  });

  it('scrolling up latches free and shows "Jump to latest"; a fast burst keeps follow pinned (regression)', async () => {
    await mountConnected();

    // A fast append burst — the model-loading scenario that used to unstick
    // the tail. Interleave a transient not-at-bottom (what Virtuoso emits
    // mid-burst when a fresh row lands below the fold) among the appends.
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    for (let i = 0; i < 40; i++) {
      push({ seq: 2 + i, type: 'delta', req_id: 'r1', ts: i, channel: 'content', text: `tok${i} ` });
      if (i === 20) {
        act(() => vh.lastProps.atBottomStateChange(false));
      }
    }

    // REGRESSION: followOutput must remain "auto" (never false) through the
    // transient not-at-bottom. Under the OLD hook (followOutput gated on the
    // stick/free latch) this would have flipped to false and frozen the tail.
    expect(vh.lastProps.followOutput).toBe('auto');

    // The latch still flips to free so the button can appear — the only path
    // back to stick is the button.
    const jump = screen.getByRole('button', { name: /jump to latest/i });
    expect(jump).toBeInTheDocument();

    // Clicking it re-sticks and the button disappears.
    fireEvent.click(jump);
    expect(screen.queryByRole('button', { name: /jump to latest/i })).not.toBeInTheDocument();
    // Still auto after re-stick.
    expect(vh.lastProps.followOutput).toBe('auto');
  });

  it('renders the disabled placeholder when the backend returns 409', async () => {
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    // Ticket mint returns 409 → useEventSource classifies it terminal; the
    // viewer maps 404/409 to the "god mode is disabled" placeholder.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('disabled', { status: 409 })));

    render(<GodModeViewer tokenIds={['tok_abc']} />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByText(/god mode is disabled/i)).toBeInTheDocument();
    expect(screen.getByText('VW_GODMODE_ENABLED')).toBeInTheDocument();
    expect(screen.queryByRole('log')).not.toBeInTheDocument();
  });

  it('shows a waiting placeholder while connected with no traffic', async () => {
    await mountConnected();
    expect(screen.getByText(/waiting for requests/i)).toBeInTheDocument();
    expect(screen.queryByRole('log')).not.toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // Repeated-system-prompt collapse
  // -------------------------------------------------------------------------

  // A realistic lengthy system prompt (>400 chars) carrying a sentinel we can
  // assert is hidden while collapsed.
  const SYS = [
    'SYSTEM_PROMPT_SENTINEL: You are a helpful assistant with a very long set of instructions.',
    'Rule 1: Always be concise and accurate in your responses to the user queries you receive.',
    'Rule 2: Never reveal internal reasoning unless explicitly asked to show your work step by step.',
    'Rule 3: Prefer structured output and cite tools when you use them during a multi-step task here.',
    'Rule 4: Maintain a professional and neutral tone across the entire length of the conversation.',
  ].join('\n');

  function mkStart(seq: number, req_id: string, token_label: string, prompt: string) {
    return {
      seq,
      type: 'request_start',
      req_id,
      ts: 0,
      token_label,
      token_id: token_label,
      model: 'm',
      served_name: 's',
      client_ip: 'ip',
      stream: true,
      prompt,
    };
  }

  it('collapses a repeated system-prompt head, showing only the delta until expanded', async () => {
    await mountConnected();

    push(mkStart(1, 'r1', 'agent', `${SYS}\nUser: FIRST_QUESTION`));
    push(mkStart(2, 'r2', 'agent', `${SYS}\nUser: SECOND_QUESTION_DELTA`));

    const r2 = document.querySelector('[data-req-id="r2"]') as HTMLElement;
    const toggle = within(r2).getByTestId('godmode-prompt-toggle');
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveTextContent(/^system prompt · [\d,]+ chars · unchanged from previous$/);

    // The divergent NEW turn is visible; the shared head is hidden.
    expect(within(r2).getByTestId('godmode-prompt-remainder')).toHaveTextContent(
      'SECOND_QUESTION_DELTA',
    );
    expect(within(r2).queryByTestId('godmode-prompt-collapsed')).toBeNull();
    expect(within(r2).queryByText(/SYSTEM_PROMPT_SENTINEL/)).toBeNull();

    // Expanding reveals the collapsed head.
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(within(r2).getByTestId('godmode-prompt-collapsed')).toHaveTextContent(
      'SYSTEM_PROMPT_SENTINEL',
    );
  });

  it('shows the full prompt for the first request from a token (no collapse)', async () => {
    await mountConnected();

    push(mkStart(1, 'r1', 'solo', `${SYS}\nUser: only question`));

    expect(screen.queryByTestId('godmode-prompt-toggle')).toBeNull();
    expect(screen.getByText(/SYSTEM_PROMPT_SENTINEL/)).toBeInTheDocument();
  });

  it('does not diff one token prompt against a different token', async () => {
    await mountConnected();

    // Identical prompts but from DIFFERENT tokens → never collapsed.
    push(mkStart(1, 'ra', 'agent-a', `${SYS}\nUser: same body`));
    push(mkStart(2, 'rb', 'agent-b', `${SYS}\nUser: same body`));

    expect(screen.queryByTestId('godmode-prompt-toggle')).toBeNull();
    // Both blocks render the full prompt (sentinel appears twice).
    expect(screen.getAllByText(/SYSTEM_PROMPT_SENTINEL/)).toHaveLength(2);
  });

  it('degrades gracefully when the capture was elided — never claims the tail matched', async () => {
    await mountConnected();

    const marker = '\n\n…[5000 chars elided]…\n\n';
    push(mkStart(1, 'r1', 'agent', `${SYS}${marker}TAIL_OLD_CONTENT`));
    push(mkStart(2, 'r2', 'agent', `${SYS}${marker}TAIL_NEW_CONTENT`));

    const r2 = document.querySelector('[data-req-id="r2"]') as HTMLElement;
    const toggle = within(r2).getByTestId('godmode-prompt-toggle');
    const remainder = within(r2).getByTestId('godmode-prompt-remainder');

    // The tail is presented as NEW content, and the elision marker sits in the
    // visible remainder — proof we did not diff across the gap.
    expect(remainder).toHaveTextContent('TAIL_NEW_CONTENT');
    expect(remainder).toHaveTextContent('chars elided');

    // The collapsed region, once revealed, is ONLY the shared head — the tail
    // is never folded into the "unchanged" region.
    expect(within(r2).queryByTestId('godmode-prompt-collapsed')).toBeNull();
    fireEvent.click(toggle);
    const collapsed = within(r2).getByTestId('godmode-prompt-collapsed');
    expect(collapsed).toHaveTextContent('SYSTEM_PROMPT_SENTINEL');
    expect(collapsed).not.toHaveTextContent('TAIL_NEW_CONTENT');
  });

  // -------------------------------------------------------------------------
  // Inline media (spec 2026-08-03)
  // -------------------------------------------------------------------------

  it('renders the media strip when request_start carries a media array, and omits it otherwise', async () => {
    await mountConnected();

    // Remote-url entry — renders as a plain <img>, no authFetch involved, so
    // this test doesn't need to mock @/lib/auth-fetch (the file already uses
    // the real module for setAccessToken/setCsrfToken).
    push({
      ...mkStart(1, 'r1', 'agent', 'describe this image'),
      media: [{ kind: 'image', url: 'https://example.com/cat.jpg' }],
    });
    push(mkStart(2, 'r2', 'agent2', 'no image here'));

    const r1 = document.querySelector('[data-req-id="r1"]') as HTMLElement;
    const r2 = document.querySelector('[data-req-id="r2"]') as HTMLElement;
    expect(within(r1).getByTestId('godmode-media-strip')).toBeInTheDocument();
    expect(within(r2).queryByTestId('godmode-media-strip')).toBeNull();
  });

  it('streams only the given tokens: ?token_ids= is on the EventSource URL', async () => {
    await mountConnected({ tokenIds: ['tok_a', 'tok_b'] });
    expect(FakeES.last.url).toBe('/api/admin/godmode/stream?token_ids=tok_a%2Ctok_b&ticket=t1');
  });

  // #251: no tokens means no stream at all — not an unfiltered one, and not
  // a ticket for an empty `token_ids=` (the server 422s that).
  it('opens no stream (and mints no ticket) for an empty token list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}'));
    vi.stubGlobal('fetch', fetchMock);
    const made: string[] = [];
    vi.stubGlobal('EventSource', class { constructor(u: string) { made.push(u); } close() {} } as unknown as typeof EventSource);
    render(<GodModeViewer tokenIds={[]} />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(made).toHaveLength(0);
  });

  // #251: the banner sits over the scrolled list, so it needs the dock's own
  // background (and to stack above the rows), or the text under it shows through.
  it('draws the "older events elided" banner on the dock background, above the rows', async () => {
    await mountConnected();
    act(() => {
      for (let i = 0; i <= MAX_EVENTS; i++) {
        FakeES.last.onmessage?.(new MessageEvent('message', {
          data: JSON.stringify(i === 0
            ? { seq: 0, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 'tok_abc', model: 'm', served_name: 'm', client_ip: '10.0.0.5', stream: true, prompt: 'p' }
            : { seq: i, type: 'delta', req_id: 'r1', ts: i, channel: 'content', text: 'x' }),
        }));
      }
    });
    const banner = screen.getByText(/older event elided/);
    expect(banner).toHaveTextContent('… 1 older event elided');
    expect(banner).toHaveClass('bg-vw-dock', 'z-10');
  });

  it('has no model selector any more', async () => {
    await mountConnected();
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    // The ModelSelector rendered checkboxes (inline chips) or a summary button.
    expect(screen.queryByRole('checkbox')).toBeNull();
  });

  it('marks a running request "streaming" with a caret, then shows the finish reason', async () => {
    await mountConnected();
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    expect(screen.getByTestId('godmode-running')).toHaveTextContent('streaming');
    expect(screen.getByTestId('godmode-caret')).toBeInTheDocument();
    push({ seq: 2, type: 'request_end', req_id: 'r1', ts: 1, finish_reason: 'stop', prompt_tokens: 1, completion_tokens: 1 });
    expect(screen.queryByTestId('godmode-running')).toBeNull();
    expect(screen.queryByTestId('godmode-caret')).toBeNull();
    expect(screen.getByTestId('godmode-finish')).toHaveTextContent('stop');
  });

  it('reports counts, and clear() drops finished requests but keeps running ones', async () => {
    const onCounts = vi.fn();
    const ref = createRef<GodModeViewerHandle>();
    await mountConnected({ onCounts, ref });
    push({ seq: 1, type: 'request_start', req_id: 'done', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p1' });
    push({ seq: 2, type: 'request_end', req_id: 'done', ts: 1, finish_reason: 'stop', prompt_tokens: 1, completion_tokens: 1 });
    push({ seq: 3, type: 'request_start', req_id: 'live', ts: 2, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p2' });
    expect(onCounts).toHaveBeenLastCalledWith({ requests: 2, running: 1 });

    act(() => ref.current!.clear());
    expect(document.querySelector('[data-req-id="done"]')).toBeNull();
    expect(document.querySelector('[data-req-id="live"]')).not.toBeNull();
    expect(onCounts).toHaveBeenLastCalledWith({ requests: 1, running: 1 });
  });

  it('setFollowing(false) stops auto-scroll; setFollowing(true) resumes it', async () => {
    const onFollowingChange = vi.fn();
    const ref = createRef<GodModeViewerHandle>();
    await mountConnected({ onFollowingChange, ref });
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    expect(onFollowingChange).toHaveBeenLastCalledWith(true);

    act(() => ref.current!.setFollowing(false));
    expect(vh.lastProps.followOutput).toBe(false);
    expect(onFollowingChange).toHaveBeenLastCalledWith(false);

    act(() => ref.current!.setFollowing(true));
    expect(vh.lastProps.followOutput).toBe('auto');
    expect(onFollowingChange).toHaveBeenLastCalledWith(true);
  });

  it('lands on the newest request, following, once the replay burst settles', async () => {
    const onFollowingChange = vi.fn();
    await mountConnected({ onFollowingChange });
    for (let i = 0; i < 3; i++) {
      push({ seq: i + 1, type: 'request_start', req_id: `r${i}`, ts: i, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    }
    // The burst pushes the tail below the fold; Virtuoso reports not-at-bottom.
    act(() => vh.lastProps.atBottomStateChange(false));
    expect(screen.getByRole('button', { name: /jump to latest/i })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(60); });
    expect(screen.queryByRole('button', { name: /jump to latest/i })).not.toBeInTheDocument();
    expect(onFollowingChange).toHaveBeenLastCalledWith(true);
  });

  it('lands and stays following even under a continuous supply of deltas faster than the 50ms landing timer (regression)', async () => {
    // useStickyBottom returns a fresh object every render. Before the fix, the
    // landing effect depended on that whole object, so a delta-driven
    // re-render (deltas can arrive well under 50ms apart while streaming)
    // tore down and rescheduled the timer before it ever got to fire — the
    // dock never landed/followed until the stream went quiet.
    const onFollowingChange = vi.fn();
    await mountConnected({ onFollowingChange });
    for (let i = 0; i < 3; i++) {
      push({ seq: i + 1, type: 'request_start', req_id: `r${i}`, ts: i, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    }
    act(() => vh.lastProps.atBottomStateChange(false));
    expect(screen.getByRole('button', { name: /jump to latest/i })).toBeInTheDocument();

    let seq = 4;
    for (let i = 0; i < 10; i++) {
      push({ seq: seq++, type: 'delta', req_id: 'r1', ts: i, channel: 'content', text: 'x' });
      await act(async () => { await vi.advanceTimersByTimeAsync(20); });
    }

    expect(screen.queryByRole('button', { name: /jump to latest/i })).not.toBeInTheDocument();
    expect(onFollowingChange).toHaveBeenLastCalledWith(true);
  });

  it('keeps the growing last block pinned to the bottom while following, like the mockup\'s scrollEnd() per delta', async () => {
    const ref = createRef<GodModeViewerHandle>();
    await mountConnected({ ref });
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    await act(async () => { await vi.advanceTimersByTimeAsync(60); });   // landed, following
    act(() => vh.lastProps.totalListHeightChanged(950));
    vh.scrollTo.mockClear();

    // A delta grows the last block by 50 px while the view is at the bottom.
    act(() => vh.lastProps.totalListHeightChanged(1000));
    expect(vh.scrollTo).toHaveBeenCalledWith({ top: Number.MAX_SAFE_INTEGER });

    // The operator scrolled up 300 px to read: growth leaves them there.
    vh.scroller.scrollTop = 300;
    fireEvent.scroll(vh.scroller);
    vh.scrollTo.mockClear();
    act(() => vh.lastProps.totalListHeightChanged(1030));
    expect(vh.scrollTo).not.toHaveBeenCalled();

    // "Following" off: growth at the bottom no longer scrolls either.
    vh.scroller.scrollTop = 600;
    fireEvent.scroll(vh.scroller);
    act(() => ref.current!.setFollowing(false));
    vh.scrollTo.mockClear();
    act(() => vh.lastProps.totalListHeightChanged(1060));
    expect(vh.scrollTo).not.toHaveBeenCalled();
  });

  it('a reader 120 px above the bottom is not pulled down when a large block arrives', async () => {
    await mountConnected();
    push({ seq: 1, type: 'request_start', req_id: 'r1', ts: 0, token_label: 'k', token_id: 't', model: 'm', served_name: 's', client_ip: 'ip', stream: true, prompt: 'p' });
    await act(async () => { await vi.advanceTimersByTimeAsync(60); });   // landed, following
    act(() => vh.lastProps.totalListHeightChanged(950));
    // Scroll 120 px up from the bottom of 1000 px of content (400 px viewport).
    vh.scroller.scrollTop = 1000 - 400 - 120;
    fireEvent.scroll(vh.scroller);
    vh.scrollTo.mockClear();

    // A 300 px block arrives; Virtuoso reports it before the DOM grows…
    act(() => vh.lastProps.totalListHeightChanged(1250));
    expect(vh.scrollTo).not.toHaveBeenCalled();
    // …and again once the scroller already includes it.
    vh.scrollHeight = 1600;
    act(() => vh.lastProps.totalListHeightChanged(1550));
    expect(vh.scrollTo).not.toHaveBeenCalled();
    expect(vh.scroller.scrollTop).toBe(480);
  });

  it('reports the stream status for the dock\'s Live indicator', async () => {
    const onStatusChange = vi.fn();
    await mountConnected({ onStatusChange });
    expect(onStatusChange).toHaveBeenLastCalledWith('connected');
  });

  it('clearFinished is a no-op when nothing has finished', () => {
    const s = { events: [], elided: 0, reqIndex: {}, nextIndex: 0 };
    expect(clearFinished(s)).toBe(s);
  });
});
