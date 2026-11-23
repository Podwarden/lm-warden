import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';

import { TAIL_CHUNK_CHARS, TailText } from '@/components/prompt-lens/dock/tail-text';
import { CLIENT_SECTION_CHARS } from '@/lib/prompt-lens/store';

afterEach(cleanup);

/** The store's rule for a streaming section: keep the last 200k, count the cut. */
function grow(prev: { text: string; offset: number }, delta: string) {
  const all = prev.text + delta;
  const cut = Math.max(0, all.length - CLIENT_SECTION_CHARS);
  return { text: cut ? all.slice(cut) : all, offset: prev.offset + cut };
}

const root = (c: HTMLElement) => c.querySelector('[data-testid="tail-text"]') as HTMLElement;

function records(el: Node) {
  const mo = new MutationObserver(() => {});
  mo.observe(el, { childList: true, characterData: true, subtree: true });
  return () => {
    const n = mo.takeRecords().length;
    mo.disconnect();
    return n;
  };
}

describe('TailText', () => {
  it('renders the text', () => {
    const { container } = render(<TailText text={'hello\nworld'} offset={0} />);
    expect(root(container).textContent).toBe('hello\nworld');
  });

  it('appends a delta without replacing the nodes already shown', () => {
    const { container, rerender } = render(<TailText text={'abc'} offset={0} />);
    const first = root(container).firstChild!;
    const firstText = first.firstChild!;
    const done = records(root(container));
    rerender(<TailText text={'abcdef'} offset={0} />);
    expect(root(container).textContent).toBe('abcdef');
    expect(root(container).firstChild).toBe(first);
    expect(first.firstChild).toBe(firstText);
    expect(done()).toBeLessThanOrEqual(1);
  });

  it('a parent re-render with the same text touches nothing', () => {
    const { container, rerender } = render(<TailText text={'abc\n'.repeat(10)} offset={0} />);
    const done = records(root(container));
    rerender(<TailText text={'abc\n'.repeat(10)} offset={0} />);
    expect(done()).toBe(0);
  });

  it('splits into blocks at a newline once a block is big, so a new block holds new lines', () => {
    const line = 'y'.repeat(99) + '\n';
    let s = { text: '', offset: 0 };
    const { container, rerender } = render(<TailText text={s.text} offset={s.offset} />);
    for (let i = 0; i < 200; i++) {
      s = grow(s, line);
      rerender(<TailText text={s.text} offset={s.offset} />);
    }
    const el = root(container);
    expect(el.textContent).toBe(line.repeat(200));
    expect(el.childNodes.length).toBeGreaterThan(1);
    const blocks = Array.from(el.childNodes) as HTMLElement[];
    for (const b of blocks.slice(0, -1)) {
      expect(b.textContent!.endsWith('\n')).toBe(true);
      expect(b.textContent!.length).toBeGreaterThanOrEqual(TAIL_CHUNK_CHARS);
    }
  });

  it('a long single line stays in one block (it wraps, it is never split into lines)', () => {
    const { container, rerender } = render(<TailText text={''} offset={0} />);
    let t = '';
    for (let i = 0; i < 100; i++) {
      t += 'z'.repeat(500);
      rerender(<TailText text={t} offset={0} />);
    }
    expect(root(container).childNodes.length).toBe(1);
    expect(root(container).textContent).toBe(t);
  });

  it('drops the head as the store clips at 200k', () => {
    let s = { text: 'q'.repeat(CLIENT_SECTION_CHARS - 10) + '\n', offset: 0 };
    const { container, rerender } = render(<TailText text={s.text} offset={s.offset} />);
    for (let i = 0; i < 30; i++) {
      s = grow(s, `line ${i}\n`);
      rerender(<TailText text={s.text} offset={s.offset} />);
    }
    expect(s.offset).toBeGreaterThan(0);
    expect(root(container).textContent).toBe(s.text);
  });

  it('rebuilds when the text is not a continuation (a section fill replaced it)', () => {
    const { container, rerender } = render(<TailText text={'abc\ndef'} offset={0} />);
    rerender(<TailText text={'xyz\nuvw and more'} offset={0} />);
    expect(root(container).textContent).toBe('xyz\nuvw and more');
    rerender(<TailText text={'restored from the server'} offset={5000} />);
    expect(root(container).textContent).toBe('restored from the server');
    rerender(<TailText text={''} offset={0} />);
    expect(root(container).textContent).toBe('');
  });

  it('wraps long lines', () => {
    const { container } = render(<TailText text={'a'} offset={0} />);
    expect(root(container).className).toMatch(/whitespace-pre-wrap/);
    expect(root(container).className).toMatch(/\[overflow-wrap:anywhere\]/);
  });

  // Performance (jsdom: DOM work only, no layout; the real-browser layout
  // check is Task 17's). A 200k-char section streaming at ~50 deltas/s must
  // cost O(delta) DOM work per delta, never a re-render of the whole text.
  it('streams 300k chars in 50-char deltas with O(1) DOM mutations per delta', () => {
    const DELTAS = 6000;
    let s = { text: '', offset: 0 };
    const { container, rerender } = render(<TailText text={s.text} offset={s.offset} />);
    const el = root(container);
    let maxRecords = 0;
    let rebuilt = 0;
    const t0 = performance.now();
    for (let i = 0; i < DELTAS; i++) {
      const delta = i % 3 === 0 ? `step ${i} looks fine.\n`.padStart(50, ' ') : 'w'.repeat(50);
      s = grow(s, delta);
      const firstBefore = el.firstChild;
      const done = records(el);
      rerender(<TailText text={s.text} offset={s.offset} />);
      const n = done();
      maxRecords = Math.max(maxRecords, n);
      if (firstBefore && !el.contains(firstBefore) && s.offset === 0) rebuilt++;
    }
    const ms = performance.now() - t0;
    expect(el.textContent).toBe(s.text);
    expect(s.offset).toBeGreaterThan(0);
    expect(rebuilt).toBe(0);
    // append (1) + a new block (1) + head delete or block removals (≤ 2)
    expect(maxRecords).toBeLessThanOrEqual(4);
    // 6,000 deltas = two minutes of streaming at 50/s; allow 2 ms each.
    expect(ms / DELTAS).toBeLessThan(2);
    console.info(`tail-text: ${DELTAS} deltas, ${(ms / DELTAS).toFixed(3)} ms/delta (jsdom), max ${maxRecords} mutation records`);
  });
});
