/**
 * Regression: a drag-selection in the live log can span more lines than fit
 * on screen, and auto-follow never yanks the view mid-selection.
 *
 * Symptom (operator report): on /ui/models/<id>, starting a text selection
 * in "Live logs" and dragging past the bottom edge auto-scrolled the panel,
 * but the lines that scrolled out of view dropped out of the selection.
 *
 * Root cause: the panel was a react-virtuoso list, which mounts only the
 * visible window of rows. Rows that scroll away are unmounted, and a DOM
 * selection cannot survive its nodes being removed from the document.
 *
 * Fix contract pinned here: every buffered line (the FIFO is capped at
 * MAX_LINES) is a real DOM node inside one native scroll container, so the
 * browser's own selection + drag-autoscroll work. Tail-following is done by
 * the component and is paused while a mouse button is held in the log or a
 * selection lives inside it.
 *
 * The file-local Virtuoso mock below renders only the LAST few rows, like a
 * real virtualized window. It overrides the render-everything shim in
 * tests/setup.ts, so the "all lines are in the DOM" assertion fails if the
 * component goes back to virtualizing.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { render, screen, act, cleanup, fireEvent } from '@testing-library/react';
import { LogStream } from '@/components/models/log-stream';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

vi.mock('react-virtuoso', () => {
  const WINDOW = 5;
  function Virtuoso(props: {
    data?: ReadonlyArray<unknown>;
    itemContent?: (index: number, item: unknown) => React.ReactNode;
    style?: React.CSSProperties;
    components?: { List?: React.ComponentType<React.HTMLAttributes<HTMLDivElement>> };
  }) {
    const { data = [], itemContent, style, components } = props;
    const start = Math.max(0, data.length - WINDOW);
    const rows = data.slice(start).map((item, i) =>
      React.createElement('div', { key: start + i }, itemContent?.(start + i, item)),
    );
    const List = components?.List ?? 'div';
    return React.createElement('div', { style }, React.createElement(List, {}, rows));
  }
  return { Virtuoso };
});

class FakeES {
  static last: FakeES;
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  constructor(public url: string) {
    FakeES.last = this;
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {}
}

const ROW_PX = 20;
const VIEWPORT_PX = 100;

function push(line: string) {
  act(() => {
    FakeES.last.onmessage?.(new MessageEvent('message', { data: JSON.stringify({ line }) }));
  });
}

async function mountWithLines(n: number) {
  render(<LogStream modelId="abc" />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
  push('line-0');
  const log = screen.getByRole('log');
  // jsdom has no layout: give the scroll container a fake geometry of
  // ROW_PX per rendered line and a writable scrollTop.
  let top = 0;
  Object.defineProperty(log, 'clientHeight', { configurable: true, get: () => VIEWPORT_PX });
  Object.defineProperty(log, 'scrollHeight', {
    configurable: true,
    get: () => log.querySelectorAll('[data-log-line]').length * ROW_PX,
  });
  Object.defineProperty(log, 'scrollTop', {
    configurable: true,
    get: () => top,
    set: (v: number) => {
      top = Math.max(0, Math.min(v, log.scrollHeight - VIEWPORT_PX));
    },
  });
  for (let i = 1; i < n; i++) push(`line-${i}`);
  return log;
}

const bottomOf = (el: HTMLElement) => el.scrollHeight - el.clientHeight;

describe('LogStream drag-selection', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}')));
  });
  afterEach(() => {
    document.getSelection()?.removeAllRanges();
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('keeps every buffered line in the DOM so a selection can span off-screen lines', async () => {
    const log = await mountWithLines(50);

    // All 50 lines are mounted, not just a viewport-sized window.
    expect(log.querySelectorAll('[data-log-line]')).toHaveLength(50);
    expect(screen.getByText('line-0')).toBeInTheDocument();

    // A selection from the first line to the last holds the whole range.
    const range = document.createRange();
    range.setStartBefore(screen.getByText('line-0'));
    range.setEndAfter(screen.getByText('line-49'));
    const sel = document.getSelection()!;
    sel.removeAllRanges();
    sel.addRange(range);
    const text = sel.toString();
    expect(text).toContain('line-0');
    expect(text).toContain('line-25');
    expect(text).toContain('line-49');
  });

  it('follows the tail while the operator is at the bottom', async () => {
    const log = await mountWithLines(20);
    expect(log.scrollTop).toBe(bottomOf(log));
    push('line-20');
    expect(log.scrollTop).toBe(bottomOf(log));
    expect(screen.queryByRole('button', { name: /jump to latest/i })).not.toBeInTheDocument();
  });

  it('does not scroll while the mouse button is held in the log (drag-select in progress)', async () => {
    const log = await mountWithLines(20);
    const before = log.scrollTop;

    fireEvent.mouseDown(log, { button: 0 });
    // 5 rows = 100px, past the 64px at-bottom tolerance.
    for (let i = 20; i < 25; i++) push(`line-${i}`);
    expect(log.scrollTop).toBe(before);
    // New lines landed below the viewport: the operator is told so.
    expect(screen.getByRole('button', { name: /jump to latest/i })).toBeInTheDocument();

    fireEvent.mouseUp(window);
    fireEvent.click(screen.getByRole('button', { name: /jump to latest/i }));
    expect(log.scrollTop).toBe(bottomOf(log));
    push('line-25');
    expect(log.scrollTop).toBe(bottomOf(log));
  });

  it('does not scroll while a selection lives inside the log, and resumes once it is cleared', async () => {
    const log = await mountWithLines(20);
    const before = log.scrollTop;

    const range = document.createRange();
    range.selectNodeContents(screen.getByText('line-18'));
    document.getSelection()!.addRange(range);

    for (let i = 20; i < 25; i++) push(`line-${i}`);
    expect(log.scrollTop).toBe(before);

    // Clearing the selection + Jump to latest resumes tailing.
    document.getSelection()!.removeAllRanges();
    fireEvent.click(screen.getByRole('button', { name: /jump to latest/i }));
    push('line-25');
    expect(log.scrollTop).toBe(bottomOf(log));
  });

  it('leaves the view alone once the operator has scrolled up', async () => {
    const log = await mountWithLines(20);
    log.scrollTop = 40;
    fireEvent.scroll(log);

    push('line-20');
    expect(log.scrollTop).toBe(40);
    expect(screen.getByRole('button', { name: /jump to latest/i })).toBeInTheDocument();
  });
});
