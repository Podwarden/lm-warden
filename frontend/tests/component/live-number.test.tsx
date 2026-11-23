// <LiveNumber>: a live counter whose width never shrinks or jitters, so text
// after it never moves (owner request 2026-11-23; shared convention
// ui.live-numbers). jsdom has no layout: these tests pin the contract the
// browser check (scripts/prompt-lens-check.mjs, "counters") measures.

import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';

import { LiveNumber, liveWidth } from '@/components/ui/live-number';

afterEach(() => cleanup());

const el = () => screen.getByTestId('n');

describe('LiveNumber', () => {
  it('renders tabular figures, right-aligned, in a reserved slot', () => {
    render(<LiveNumber data-testid="n" reserve={7}>1,234</LiveNumber>);
    expect(el()).toHaveTextContent('1,234');
    expect(el()).toHaveClass('num-live');
    expect(el().style.minWidth).toBe('7ch');
  });

  it('a value wider than the reserve widens the slot, and it never narrows again', () => {
    const { rerender } = render(<LiveNumber data-testid="n" reserve={3}>9</LiveNumber>);
    expect(el().style.minWidth).toBe('3ch');
    rerender(<LiveNumber data-testid="n" reserve={3}>12,345</LiveNumber>);
    expect(el().style.minWidth).toBe('6ch');
    rerender(<LiveNumber data-testid="n" reserve={3}>7</LiveNumber>);
    expect(el().style.minWidth).toBe('6ch');
  });

  it('a reserve that drops (the counter stopped being live) keeps the slot', () => {
    const { rerender } = render(<LiveNumber data-testid="n" reserve={7}>12</LiveNumber>);
    rerender(<LiveNumber data-testid="n" reserve={0}>12</LiveNumber>);
    expect(el().style.minWidth).toBe('7ch');
  });

  it('a new resetKey (another request) starts the slot over', () => {
    const { rerender } = render(<LiveNumber data-testid="n" resetKey="a">123,456</LiveNumber>);
    expect(el().style.minWidth).toBe('7ch');
    rerender(<LiveNumber data-testid="n" resetKey="b">12</LiveNumber>);
    expect(el().style.minWidth).toBe('2ch');
  });

  it('numbers are formatted with the given formatter', () => {
    render(<LiveNumber data-testid="n" value={1234567} />);
    expect(el()).toHaveTextContent('1,234,567');
  });

  it('liveWidth reserves the width of the widest value a format can reach', () => {
    expect(liveWidth('200,000')).toBe(7);
    expect(liveWidth('')).toBe(0);
  });
});
