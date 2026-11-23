import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';

import { SectionBox, useCollapseMemory, type SectionKind } from '@/components/prompt-lens/dock/section-box';
import { PartsTable } from '@/components/prompt-lens/dock/parts-table';
import { TailText } from '@/components/prompt-lens/dock/tail-text';
import { mapParts } from '@/lib/prompt-lens/kvsig';
import { sectionMarks } from '@/lib/prompt-lens/section-marks';

import { PROMPT, part, reqWith } from '../unit/prompt-lens-kvsig-fixtures';
import { FOLLOW_SLACK_PX } from '@/components/prompt-lens/dock/use-follow-tail';

// jsdom has no layout. These tests give every section box a fake one: the
// box is CLIENT px high (or its content height when that is smaller, like a
// shrink-to-fit box), its content is 20 px per 100 chars of text, and
// scrollTop clamps to [0, scrollHeight - clientHeight] as in a browser.
// Setting scrollTop never fires `scroll` (as in a browser, the event comes
// later); the tests fire it by hand to stand for that later event.
const CLIENT = 200;
const LINE = 20;
const PER_LINE = 100;

const tops = new WeakMap<Element, number>();
const extra = new WeakMap<Element, number>();
const isBox = (el: Element) => (el.getAttribute('data-testid') ?? '').startsWith('section-box-');
const contentHeight = (el: Element) =>
  Math.ceil((el.textContent ?? '').length / PER_LINE) * LINE + (extra.get(el) ?? 0);
const clientOf = (el: Element) => Math.min(CLIENT, contentHeight(el));
const scrollHeightOf = (el: Element) => Math.max(clientOf(el), contentHeight(el));
const maxTop = (el: Element) => scrollHeightOf(el) - clientOf(el);

const saved: Record<string, PropertyDescriptor | undefined> = {};
const PROPS = ['scrollTop', 'scrollHeight', 'clientHeight'] as const;

beforeEach(() => {
  for (const p of PROPS) saved[p] = Object.getOwnPropertyDescriptor(Element.prototype, p);
  Object.defineProperty(Element.prototype, 'scrollHeight', {
    configurable: true,
    get(this: Element) {
      return isBox(this) ? scrollHeightOf(this) : 0;
    },
  });
  Object.defineProperty(Element.prototype, 'clientHeight', {
    configurable: true,
    get(this: Element) {
      return isBox(this) ? clientOf(this) : 0;
    },
  });
  Object.defineProperty(Element.prototype, 'scrollTop', {
    configurable: true,
    get(this: Element) {
      return tops.get(this) ?? 0;
    },
    set(this: Element, v: number) {
      tops.set(this, Math.max(0, Math.min(v, isBox(this) ? maxTop(this) : 0)));
    },
  });
});

afterEach(() => {
  cleanup();
  for (const p of PROPS) {
    const d = saved[p];
    if (d) Object.defineProperty(Element.prototype, p, d);
    else delete (Element.prototype as unknown as Record<string, unknown>)[p];
  }
});

/** The reader drags the box to `top` (the browser's later scroll event). */
function scrollTo(el: HTMLElement, top: number) {
  el.scrollTop = top;
  fireEvent.scroll(el);
}

const text = (n: number) => 'x'.repeat(n);

interface BoxProps {
  chars: number;
  live?: boolean;
  resetKey?: string;
  phone?: boolean;
  tok?: number;
  kind?: SectionKind;
  collapsed?: boolean;
  marks?: { trimmed?: number; elided?: boolean; stripped?: boolean };
}

function Box({ chars, live = true, resetKey = 'r1', phone = false, tok, kind = 'thinking', collapsed = false, marks }: BoxProps) {
  return (
    <SectionBox
      kind={kind}
      title="Thinking"
      chars={chars}
      tok={tok ?? Math.round(chars / 4)}
      live={live}
      rate={live ? 41.6 : null}
      status={live ? 'streaming' : 'done'}
      collapsed={collapsed}
      onToggle={() => {}}
      onExpand={() => {}}
      phone={phone}
      resetKey={resetKey}
      marks={marks}
    >
      {text(chars)}
    </SectionBox>
  );
}

const box = (kind: SectionKind = 'thinking') => screen.getByTestId(`section-box-${kind}`);
const jumpButton = () => screen.queryByRole('button', { name: /Jump to latest/ });

describe('useFollowTail through SectionBox', () => {
  it('slack is 24 px', () => {
    expect(FOLLOW_SLACK_PX).toBe(24);
  });

  it('pins to the bottom while following', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    expect(el.scrollTop).toBe(maxTop(el));
    expect(el.scrollTop).toBe(400);
    rerender(<Box chars={4200} />);
    expect(el.scrollTop).toBe(el.scrollHeight - el.clientHeight);
    expect(el.scrollTop).toBe(640);
    expect(screen.getByText('● following')).toBeVisible();
    expect(jumpButton()).toBeNull();
  });

  it('stops following when the reader scrolls up more than 24 px and keeps position', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    scrollTo(el, 0);
    rerender(<Box chars={4200} />);
    expect(el.scrollTop).toBe(0);
    expect(screen.getByRole('button', { name: /Jump to latest · 1,200 new chars/ })).toBeVisible();
    expect(screen.getByText('● not following')).toBeVisible();
    rerender(<Box chars={4300} />);
    expect(el.scrollTop).toBe(0);
    expect(screen.getByRole('button', { name: /Jump to latest · 1,300 new chars/ })).toBeVisible();
  });

  it('a scroll that stays within 24 px of the bottom keeps following', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    scrollTo(el, 400 - FOLLOW_SLACK_PX);
    rerender(<Box chars={3200} />);
    expect(el.scrollTop).toBe(440);
    expect(jumpButton()).toBeNull();
  });

  it('a programmatic scroll never counts as scrolling up', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    rerender(<Box chars={4200} />);
    expect(el.scrollTop).toBe(640);
    // More text lands before the browser delivers the scroll event of that
    // pin, so the event finds the box 200 px short of the bottom.
    extra.set(el, 200);
    fireEvent.scroll(el);
    expect(jumpButton()).toBeNull();
    expect(screen.getByText('● following')).toBeVisible();
    extra.delete(el);
    rerender(<Box chars={4400} />);
    expect(el.scrollTop).toBe(maxTop(el));
  });

  it('a content-growth race without a pin still does not count as scrolling up', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    // The pin's own event already arrived; then content grows and the
    // browser clamps nothing, so scrollTop did not move: not a reader.
    fireEvent.scroll(el);
    extra.set(el, 300);
    fireEvent.scroll(el);
    expect(jumpButton()).toBeNull();
    extra.delete(el);
    rerender(<Box chars={3100} />);
    expect(el.scrollTop).toBe(maxTop(el));
  });

  it('a wheel up stops following at once, before any scroll event', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    fireEvent.wheel(el, { deltaY: -60 });
    rerender(<Box chars={3500} />);
    expect(el.scrollTop).toBe(400);
    expect(screen.getByRole('button', { name: /Jump to latest · 500 new chars/ })).toBeVisible();
  });

  it('a wheel down does not stop following', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    fireEvent.wheel(el, { deltaY: 60 });
    rerender(<Box chars={3500} />);
    expect(el.scrollTop).toBe(maxTop(el));
  });

  it('PageUp and a finger pulling the text down stop following', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    fireEvent.keyDown(el, { key: 'PageUp' });
    rerender(<Box chars={3100} />);
    expect(jumpButton()).not.toBeNull();
    fireEvent.click(jumpButton()!);
    fireEvent.touchStart(el, { touches: [{ clientY: 100 }] });
    fireEvent.touchMove(el, { touches: [{ clientY: 140 }] });
    rerender(<Box chars={3300} />);
    expect(screen.getByRole('button', { name: /Jump to latest · 200 new chars/ })).toBeVisible();
  });

  it('scrolling back to the bottom by hand resumes', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    scrollTo(el, 0);
    rerender(<Box chars={4200} />);
    expect(jumpButton()).not.toBeNull();
    scrollTo(el, maxTop(el) - 10);
    expect(jumpButton()).toBeNull();
    expect(screen.getByText('● following')).toBeVisible();
    rerender(<Box chars={4400} />);
    expect(el.scrollTop).toBe(maxTop(el));
  });

  it('jump resumes and clears the count', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    scrollTo(el, 0);
    rerender(<Box chars={4200} />);
    fireEvent.click(screen.getByRole('button', { name: /Jump to latest · 1,200 new chars/ }));
    expect(el.scrollTop).toBe(maxTop(el));
    expect(jumpButton()).toBeNull();
    rerender(<Box chars={4300} />);
    expect(el.scrollTop).toBe(maxTop(el));
    // The jump's own scroll event is not the reader scrolling up.
    fireEvent.scroll(el);
    scrollTo(el, 0);
    rerender(<Box chars={4400} />);
    expect(screen.getByRole('button', { name: /Jump to latest · 100 new chars/ })).toBeVisible();
  });

  it('a new resetKey (another request) starts following at the bottom', () => {
    const { rerender } = render(<Box chars={3000} resetKey="a" />);
    const el = box();
    scrollTo(el, 0);
    rerender(<Box chars={4200} resetKey="a" />);
    expect(jumpButton()).not.toBeNull();
    rerender(<Box chars={2500} resetKey="b" />);
    expect(jumpButton()).toBeNull();
    expect(el.scrollTop).toBe(maxTop(el));
    rerender(<Box chars={2600} resetKey="b" />);
    expect(el.scrollTop).toBe(maxTop(el));
  });

  it('a parent re-render with the same content keeps the reader where they are', () => {
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    scrollTo(el, 120);
    rerender(<Box chars={3000} />);
    rerender(<Box chars={3000} />);
    expect(el.scrollTop).toBe(120);
  });

  it('a finished section shows no follow tag and no jump control', () => {
    const { rerender } = render(<Box chars={3000} live={false} />);
    const el = box();
    expect(el.scrollTop).toBe(maxTop(el));
    scrollTo(el, 0);
    rerender(<Box chars={3000} live={false} />);
    expect(screen.queryByText(/following/)).toBeNull();
    expect(jumpButton()).toBeNull();
    expect(el.scrollTop).toBe(0);
  });

  it('a width change re-pins while following and keeps the reading place otherwise', () => {
    const observers: { cb: ResizeObserverCallback; els: Element[] }[] = [];
    vi.stubGlobal(
      'ResizeObserver',
      class {
        cb: ResizeObserverCallback;
        els: Element[] = [];
        constructor(cb: ResizeObserverCallback) {
          this.cb = cb;
          observers.push(this);
        }
        observe(el: Element) {
          this.els.push(el);
        }
        unobserve() {}
        disconnect() {}
      },
    );
    let width = 600;
    const rect = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(
      () => ({ width, height: CLIENT, top: 0, left: 0, right: width, bottom: CLIENT, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect,
    );
    const { rerender } = render(<Box chars={3000} />);
    const el = box();
    const ro = observers.find((o) => o.els.includes(el));
    expect(ro).toBeDefined();
    const fire = () =>
      act(() => ro!.cb([{ target: el, contentRect: { width } } as unknown as ResizeObserverEntry], ro as unknown as ResizeObserver));

    // Narrower: the text wraps onto more lines (scrollHeight 600 -> 1200).
    width = 300;
    extra.set(el, 600);
    fire();
    expect(el.scrollTop).toBe(maxTop(el));

    // Not following: the reader at a third of the text stays at a third.
    scrollTo(el, 200); // 1200 px of content: the reader's top sits at 1/6
    width = 600;
    extra.delete(el); // back to 600 px of content
    fire();
    expect(el.scrollTop).toBe(100);
    expect(jumpButton()).not.toBeNull();
    rerender(<Box chars={3000} />);
    expect(el.scrollTop).toBe(100);
    rect.mockRestore();
  });
  it('a head trim (the 200k client bound) keeps the reader on the same text', () => {
    const T = ({ t, offset }: { t: string; offset: number }) => (
      <SectionBox kind="thinking" title="Thinking" chars={offset + t.length} tok={1} live collapsed={false} onToggle={() => {}} onExpand={() => {}} phone={false} resetKey="r">
        <TailText text={t} offset={offset} />
      </SectionBox>
    );
    const a = 'a'.repeat(1000) + 'b'.repeat(2000);
    const { rerender } = render(<T t={a} offset={0} />);
    const el = box();
    scrollTo(el, 300); // reading the b's
    // 1,000 chars arrive and the dock drops the oldest 1,000 (200 px).
    rerender(<T t={'b'.repeat(2000) + 'c'.repeat(1000)} offset={1000} />);
    expect(el.textContent).toBe('b'.repeat(2000) + 'c'.repeat(1000));
    expect(el.scrollTop).toBe(100);
    fireEvent.scroll(el); // the nudge's own event
    expect(screen.getByRole('button', { name: /Jump to latest · 1,000 new chars/ })).toBeVisible();
    expect(el.scrollTop).toBe(100);
  });
});

describe('SectionBox', () => {
  it('box is 200 px, 160 px on phones, shrink-to-fit under 500 chars', () => {
    const { rerender } = render(<Box chars={3000} />);
    expect(box().style.height).toBe('200px');
    expect(box().className).toMatch(/overscroll-contain/);
    expect(box().className).toMatch(/overflow-y-auto/);
    rerender(<Box chars={3000} phone />);
    expect(box().style.height).toBe('160px');
    rerender(<Box chars={499} />);
    expect(box().style.height).toBe('');
    expect(box().style.maxHeight).toBe('200px');
    rerender(<Box chars={499} phone />);
    expect(box().style.maxHeight).toBe('160px');
    rerender(<Box chars={500} />);
    expect(box().style.height).toBe('200px');
  });

  it('long single lines wrap inside the box', () => {
    render(<Box chars={3000} />);
    expect(box().className).toMatch(/whitespace-pre-wrap/);
    expect(box().className).toMatch(/\[overflow-wrap:anywhere\]/);
  });

  it('header shows the whole size, status, rate and the actions', () => {
    const onToggle = vi.fn();
    const onExpand = vi.fn();
    render(
      <SectionBox kind="reply" title="Reply" chars={12_345} tok={3_011} live rate={41.6} status="streaming" collapsed={false} onToggle={onToggle} onExpand={onExpand} phone={false} resetKey="r">
        hello
      </SectionBox>,
    );
    const root = screen.getByTestId('section-reply');
    expect(root).toHaveTextContent('Reply');
    expect(screen.getByTestId('section-size-reply')).toHaveTextContent('12,345 chars · ≈ 3,011 tok');
    expect(root).toHaveTextContent('streaming');
    expect(root).toHaveTextContent('≈ 42 tok/s');
    fireEvent.click(screen.getByRole('button', { name: 'Collapse Reply' }));
    expect(onToggle).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Open Reply in a large view' }));
    expect(onExpand).toHaveBeenCalledTimes(1);
  });

  it('size turns amber over 2,000 tok', () => {
    const { rerender } = render(<Box chars={8000} tok={2000} />);
    const size = () => screen.getByTestId('section-size-thinking');
    expect(size()).toHaveAttribute('data-big', 'false');
    expect(size().className).not.toMatch(/amber/);
    rerender(<Box chars={8004} tok={2001} />);
    expect(size()).toHaveAttribute('data-big', 'true');
    expect(size().className).toMatch(/text-vw-amber-fg/);
  });

  it('collapsed folds to the header and keeps the size visible', () => {
    render(<Box chars={3000} collapsed />);
    expect(screen.queryByTestId('section-box-thinking')).toBeNull();
    expect(screen.getByTestId('section-size-thinking')).toHaveTextContent('3,000 chars');
    const b = screen.getByRole('button', { name: 'Show Thinking' });
    expect(b).toHaveAttribute('aria-expanded', 'false');
    expect(b).toHaveTextContent('▸');
    expect(screen.getByText(/collapsed · remembered while this page is open/)).toBeVisible();
  });

  it('expanding again starts following at the bottom', () => {
    const { rerender } = render(<Box chars={3000} />);
    scrollTo(box(), 0);
    rerender(<Box chars={3200} />);
    rerender(<Box chars={3400} collapsed />);
    rerender(<Box chars={3600} />);
    expect(box().scrollTop).toBe(maxTop(box()));
    expect(jumpButton()).toBeNull();
  });

  it('an empty section says so, with its own text', () => {
    render(
      <SectionBox kind="tool_calls" title="Tool calls" chars={0} tok={0} live={false} collapsed={false} onToggle={() => {}} onExpand={() => {}} phone={false} resetKey="r" empty="No tool calls" />,
    );
    expect(screen.getByTestId('section-size-tool_calls')).toHaveTextContent('empty');
    expect(screen.getByText('No tool calls')).toBeVisible();
  });

  it('trimmed, elided and stripped marks are visible', () => {
    const { rerender } = render(<Box chars={250_000} marks={{ trimmed: 50_000 }} live={false} />);
    expect(screen.getByTestId('section-marks-thinking')).toHaveTextContent('first 50,000 chars trimmed');
    expect(screen.getByTestId('section-trim-note-thinking')).toHaveTextContent('50,000 earlier chars are not kept on this page');
    rerender(<Box chars={3000} marks={{ elided: true }} live={false} />);
    expect(screen.getByTestId('section-marks-thinking')).toHaveTextContent('cut on the server');
    rerender(<Box chars={3000} marks={{ stripped: true }} live={false} />);
    expect(screen.getByTestId('section-marks-thinking')).toHaveTextContent('text dropped to save memory');
    rerender(<Box chars={3000} live={false} />);
    expect(screen.queryByTestId('section-marks-thinking')).toBeNull();
  });

  it('a top fade marks the box while not following', () => {
    const { rerender } = render(<Box chars={3000} />);
    expect(screen.queryByTestId('section-fade-thinking')).toBeNull();
    scrollTo(box(), 0);
    rerender(<Box chars={3100} />);
    expect(screen.getByTestId('section-fade-thinking')).toBeInTheDocument();
  });

  it('never says Prompt Lens or cobs2', () => {
    render(<Box chars={3000} marks={{ trimmed: 10, elided: true, stripped: true }} />);
    const html = document.body.innerHTML.toLowerCase();
    expect(html).not.toMatch(/god ?mode|cobs2/);
  });
});

describe('useCollapseMemory', () => {
  function Dock({ reqId, toolResultsTok }: { reqId: string; toolResultsTok: number }) {
    const [isCollapsed, setCollapsed] = useCollapseMemory();
    const kinds: SectionKind[] = ['thinking', 'reply', 'tool_calls', 'tool_results', 'prompt', 'parts'];
    return (
      <div data-req={reqId}>
        {kinds.map((k) => {
          const tok = k === 'tool_results' ? toolResultsTok : 10;
          const c = isCollapsed(k, tok);
          return (
            <button key={k} type="button" data-testid={`c-${k}`} data-collapsed={String(c)} onClick={() => setCollapsed(k, !c)}>
              {k}
            </button>
          );
        })}
      </div>
    );
  }
  const state = (k: SectionKind) => screen.getByTestId(`c-${k}`).getAttribute('data-collapsed');

  it('defaults: prompt collapsed, tool results collapsed over 4,000 tok, the rest open', () => {
    const { rerender } = render(<Dock reqId="a" toolResultsTok={4000} />);
    expect(state('prompt')).toBe('true');
    expect(state('tool_results')).toBe('false');
    for (const k of ['thinking', 'reply', 'tool_calls', 'parts'] as SectionKind[]) expect(state(k)).toBe('false');
    rerender(<Dock reqId="a" toolResultsTok={4001} />);
    expect(state('tool_results')).toBe('true');
  });

  it('collapse memory survives switching requests but starts fresh with a new dock', () => {
    const { rerender, unmount } = render(<Dock reqId="a" toolResultsTok={10} />);
    fireEvent.click(screen.getByTestId('c-thinking'));
    fireEvent.click(screen.getByTestId('c-prompt'));
    expect(state('thinking')).toBe('true');
    expect(state('prompt')).toBe('false');
    rerender(<Dock reqId="b" toolResultsTok={9000} />);
    expect(state('thinking')).toBe('true');
    expect(state('prompt')).toBe('false');
    // Not chosen by hand, so the default still applies per request.
    expect(state('tool_results')).toBe('true');
    unmount();
    render(<Dock reqId="b" toolResultsTok={10} />);
    expect(state('thinking')).toBe('false');
    expect(state('prompt')).toBe('true');
  });

  it('never touches localStorage', () => {
    const set = vi.spyOn(Storage.prototype, 'setItem');
    render(<Dock reqId="a" toolResultsTok={10} />);
    fireEvent.click(screen.getByTestId('c-thinking'));
    expect(set).not.toHaveBeenCalled();
  });

  it('is usable from a dock that owns it and boxes that read it', () => {
    function Screen() {
      const [isCollapsed, setCollapsed] = useCollapseMemory();
      const [req, setReq] = useState('a');
      return (
        <div>
          <button type="button" onClick={() => setReq(req === 'a' ? 'b' : 'a')}>switch</button>
          <SectionBox kind="thinking" title="Thinking" chars={100} tok={25} live={false} collapsed={isCollapsed('thinking')} onToggle={() => setCollapsed('thinking', !isCollapsed('thinking'))} onExpand={() => {}} phone={false} resetKey={req}>
            {req}
          </SectionBox>
        </div>
      );
    }
    render(<Screen />);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse Thinking' }));
    fireEvent.click(screen.getByRole('button', { name: 'switch' }));
    expect(screen.getByRole('button', { name: 'Show Thinking' })).toBeInTheDocument();
    expect(screen.queryByTestId('section-box-thinking')).toBeNull();
  });
});

describe('hosting the parts table and the marks of a request', () => {
  it('a parts table over 8 rows sits in a SectionBox instead of its own box', () => {
    const r = reqWith({ parts: Array.from({ length: 12 }, (_, i) => part('nodes', i, i, 'new')), end: { prompt_tokens: 12000 } });
    const v = mapParts(r, 'admin');
    render(
      <SectionBox kind="parts" title="Parts" chars={12000} tok={12000 / 4} live={false} collapsed={false} onToggle={() => {}} onExpand={() => {}} phone={false} resetKey="r1">
        <PartsTable parts={v.parts} phone={false} bare />
      </SectionBox>,
    );
    const b = screen.getByTestId('section-box-parts');
    expect(b.querySelector('table')).not.toBeNull();
    expect(screen.queryByTestId('parts-box')).toBeNull();
    expect(b.querySelectorAll('tbody tr')).toHaveLength(12);
  });

  it('without bare the parts table keeps its own box (Task 10 behaviour)', () => {
    const r = reqWith({ parts: Array.from({ length: 12 }, (_, i) => part('nodes', i, i, 'new')), end: { prompt_tokens: 12000 } });
    render(<PartsTable parts={mapParts(r, 'admin').parts} phone={false} />);
    expect(screen.getByTestId('parts-box')).toBeInTheDocument();
  });

  it('sectionMarks reads each section kind from the store', () => {
    const r = reqWith();
    const base = { ...r, thinking: { ...r.thinking, clipped: 1200 }, reply: { ...r.reply, clipped: 30 }, promptClipped: 400, promptElided: true, toolCallsClipped: 50 };
    expect(sectionMarks(base, 'thinking')).toEqual({ trimmed: 1200, elided: false, stripped: false });
    expect(sectionMarks(base, 'reply')).toEqual({ trimmed: 30, elided: false, stripped: false });
    expect(sectionMarks(base, 'prompt')).toEqual({ trimmed: 400, elided: true, stripped: false });
    expect(sectionMarks(base, 'tool_calls')).toEqual({ trimmed: 50, elided: false, stripped: false });
    const ended = { ...base, end: { ...(reqWith({ end: {} }).end!), tool_call_chars_dropped: 7, tool_calls_dropped: 2 } };
    expect(sectionMarks(ended, 'tool_calls')).toEqual({ trimmed: 57, elided: true, stripped: false });
    const elidedCall = { ...r, toolCalls: [{ index: 0, callId: 'c', name: 'bash', args: '{}', done: true, elided: true }] };
    expect(sectionMarks(elidedCall, 'tool_calls').elided).toBe(true);
    expect(sectionMarks({ ...r, stripped: true }, 'reply').stripped).toBe(true);
    expect(sectionMarks({ ...r, stripped: true }, 'parts')).toEqual({ trimmed: 0, elided: false, stripped: false });
    const withToolMsg = { ...r, prompt: { ...PROMPT, messages: [{ i: 3, role: 'tool', chars: 10, text: 'x', elided: true }] } };
    expect(sectionMarks(withToolMsg, 'tool_results').elided).toBe(true);
  });
});
