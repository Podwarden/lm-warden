import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { UsageSection, summaryCaption, type UsageSectionProps } from '@/components/tokens/detail/usage-section';
import { stripBuckets, stripLabel } from '@/components/tokens/detail/history-strip';
import { SERIES } from '@/components/tokens/detail/styles';
import { toInputValue } from '@/lib/token-format';
import { NOW, seriesFixture } from './token-fixtures';

// The strip measures itself; give every element an 800 px width. jsdom has no
// PointerEvent — MouseEvent carries clientX, which is all the brush reads.
// (jsdom defines clientWidth on Element.prototype; the override shadows it on
// HTMLElement.prototype and is deleted again afterwards.)
beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 800 });
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
});
afterAll(() => { delete (HTMLElement.prototype as unknown as { clientWidth?: number }).clientWidth; });

const START = NOW - 13 * 86400;             // strip span: 13 days = 18720 min → 23.4 min per px
const WIN = { from: NOW - 7 * 86400, to: NOW };

function setup(p: Partial<UsageSectionProps> = {}) {
  const props: UsageSectionProps = {
    sel: { kind: 'preset', preset: '7d' }, window: WIN, bounds: { from: START, to: NOW },
    showChainToggle: true, chain: true, onChainChange: vi.fn(), onPreset: vi.fn(), onCustom: vi.fn(),
    onReset: vi.fn(), resetDisabled: true,
    onWindowChange: vi.fn(), onApply: vi.fn(), rangeError: null,
    strip: { chain: null, own: null, rotations: [NOW - 12 * 86400] },
    series: seriesFixture(), ...p,
  };
  const r = render(<UsageSection {...props} />);
  return { props, ...r };
}

describe('UsageSection', () => {
  afterEach(() => cleanup());

  it('renders the head, strip copy, range text and summary', () => {
    setup();
    expect(screen.getByRole('heading', { name: 'Usage' })).toBeInTheDocument();
    expect(screen.getByLabelText('Include earlier keys')).toBeChecked();
    expect(screen.getByRole('button', { name: '7d' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Whole history of this key. Drag the window or its edges, or choose Custom to type exact times.')).toBeInTheDocument();
    expect(screen.getByText('Sep 11 13:21 to Sep 18 13:21')).toBeInTheDocument();
    expect(screen.getByText('4.2k')).toBeInTheDocument();
    expect(screen.getByText('324M')).toBeInTheDocument();
    expect(screen.getByText('5.0M')).toBeInTheDocument();
    expect(screen.getByText('prefill tokens')).toBeInTheDocument();
    expect(screen.getByText('generation tokens')).toBeInTheDocument();
    expect(screen.queryByText('prompt tokens')).toBeNull();
    expect(screen.getByText('last 7 days · 30 min bins · 337 points')).toBeInTheDocument();
    expect(screen.getByText('rotated')).toBeInTheDocument();
    expect(screen.queryByLabelText('From')).toBeNull();            // no custom row outside Custom
  });

  it('draws the mockup strip: floor(width / 4) bars, 3 px wide with 1 px gaps, whatever the server bin width', () => {
    // 60-min server bins over a 13-day strip (the server's ladder rung for
    // max_bins=360) → still 200 bars on an 800 px strip, like the mockup.
    const w = 60;
    const start = Math.floor(START / 60 / w) * w;
    const bins = Array.from({ length: 13 * 24 }, (_, i) => ({
      ...seriesFixture().bins[0], minute: start + i * w, prompt_per_min: 1000 + i, completion_per_min: 0,
    }));
    const own = seriesFixture({ bin_minutes: w, bins: bins.map((b) => ({ ...b, prompt_per_min: b.prompt_per_min / 2 })) });
    const chain = seriesFixture({ bin_minutes: w, bins });
    const { container } = setup({ strip: { chain, own, rotations: [] } });
    const rects = [...container.querySelectorAll('svg rect')];
    const chainBars = rects.filter((r) => r.getAttribute('fill') === SERIES.stripChain);
    const ownBars = rects.filter((r) => r.getAttribute('fill') === SERIES.stripOwn);
    expect(chainBars.length).toBe(200);
    expect(ownBars.length).toBe(200);
    expect(Number(chainBars[0].getAttribute('width'))).toBeCloseTo(3);
    expect(Number(chainBars[1].getAttribute('x'))).toBeCloseTo(4);
  });

  it('stripBuckets spreads each server bin over the buckets it overlaps', () => {
    const b = (minute: number, perMin: number) => ({ ...seriesFixture().bins[0], minute, prompt_per_min: perMin, completion_per_min: 0 });
    // strip [0, 100) in 4 buckets of 25 min; one 60-min bin at 0 (10/min), one at 60 (1/min)
    expect(stripBuckets([b(0, 10), b(60, 1)], 60, 0, 100, 4)).toEqual([250, 250, 100 + 15, 25]);
    // bins outside the strip are ignored; an empty list gives zeros
    expect(stripBuckets([b(-60, 5), b(100, 5)], 60, 0, 100, 4)).toEqual([0, 0, 0, 0]);
  });

  it('names a mid-minute preset window by the minute it runs to (same minute at both ends, like the mockup)', () => {
    // resolveWindow for 24h at 13:21:25: from = 13:22 yesterday, to = now.
    const now = NOW + 25;
    setup({ window: { from: NOW + 60 - 86_400, to: now } });
    expect(screen.getByText('Sep 17 13:22 to Sep 18 13:22')).toBeInTheDocument();
    expect(screen.getByRole('slider')).toHaveAttribute('aria-valuetext', 'Sep 17 13:22 to Sep 18 13:22');
  });

  it('custom caption names the period', () => {
    const s = seriesFixture({ from_minute: Date.UTC(2026, 8, 12, 9, 0) / 60000, to_minute: Date.UTC(2026, 8, 14, 18, 30) / 60000, bin_minutes: 10 });
    expect(summaryCaption({ kind: 'custom', from: 0, to: 1 }, s)).toBe('Sep 12 09:00 to Sep 14 18:30 · 10 min bins · 345 points');
  });

  it('hides the toggle when there are no earlier keys', () => {
    setup({ showChainToggle: false });
    expect(screen.queryByLabelText('Include earlier keys')).toBeNull();
  });

  it('presets and Custom report up', () => {
    const { props } = setup();
    fireEvent.click(screen.getByRole('button', { name: '24h' }));
    expect(props.onPreset).toHaveBeenCalledWith('24h');
    fireEvent.click(screen.getByRole('button', { name: 'Custom' }));
    expect(props.onCustom).toHaveBeenCalled();
  });

  it('arrow keys move the window; Shift+arrows resize it', () => {
    const { props } = setup();
    const brush = screen.getByRole('slider');
    fireEvent.keyDown(brush, { key: 'ArrowLeft' });
    // len = 10080 min → step = 1008 min
    expect(props.onWindowChange).toHaveBeenLastCalledWith({ from: WIN.from - 1008 * 60, to: WIN.to - 1008 * 60 });
    fireEvent.keyDown(brush, { key: 'ArrowLeft', shiftKey: true });
    expect(props.onWindowChange).toHaveBeenLastCalledWith({ from: WIN.from + 1008 * 60, to: WIN.to });
  });

  it('dragging the window moves it, clamped to the strip', () => {
    const { props } = setup();
    const brush = screen.getByRole('slider');
    fireEvent.pointerDown(brush, { clientX: 400, pointerId: 1 });
    fireEvent.pointerMove(brush, { clientX: 300, pointerId: 1 });   // −100 px ≈ −2340 min
    const last = (props.onWindowChange as ReturnType<typeof vi.fn>).mock.lastCall![0];
    expect(last.to - last.from).toBe(WIN.to - WIN.from);
    expect(last.from).toBe(WIN.from - 2340 * 60);
    fireEvent.pointerMove(brush, { clientX: 1400, pointerId: 1 });  // far right → clamped at the end
    expect((props.onWindowChange as ReturnType<typeof vi.fn>).mock.lastCall![0].to).toBe(NOW);
    fireEvent.pointerUp(brush, { pointerId: 1 });
  });

  it('custom fields follow the window, and typing + Apply/Enter moves the window', () => {
    const custom = { kind: 'custom', from: WIN.from, to: WIN.to } as const;
    const { props, rerender } = setup({ sel: custom });
    const from = screen.getByLabelText('From') as HTMLInputElement;
    const to = screen.getByLabelText('To') as HTMLInputElement;
    expect(from.value).toBe(toInputValue(WIN.from));
    expect(to.value).toBe(toInputValue(WIN.to));
    expect(screen.getByText('Your local time')).toBeInTheDocument();
    // Mockup: `button, input { font: inherit; color: inherit; }` — the typed
    // value inherits the label's --muted colour, not --fg.
    expect(from.className).toContain('text-chat-muted');
    expect(from.className).not.toContain('text-chat-fg');

    // window → fields
    const moved = { from: WIN.from - 3600, to: WIN.to - 3600 };
    rerender(<UsageSection {...props} sel={custom} window={moved} />);
    expect(from.value).toBe(toInputValue(moved.from));

    // fields → window (Enter submits the form)
    fireEvent.change(from, { target: { value: '2026-09-12T09:00' } });
    fireEvent.change(to, { target: { value: '2026-09-14T18:30' } });
    fireEvent.submit(from.closest('form')!);
    expect(props.onWindowChange).not.toHaveBeenCalled();
    expect(props.onApply).toHaveBeenLastCalledWith({
      from: Date.UTC(2026, 8, 12, 9, 0) / 1000, to: Date.UTC(2026, 8, 14, 18, 30) / 1000,
    });
  });

  it('validates and clamps the typed period', () => {
    const custom = { kind: 'custom', from: WIN.from, to: WIN.to } as const;
    const { props } = setup({ sel: custom });
    const from = screen.getByLabelText('From');
    const to = screen.getByLabelText('To');
    const apply = screen.getByRole('button', { name: 'Apply' });

    fireEvent.change(from, { target: { value: '' } });
    fireEvent.click(apply);
    expect(screen.getByRole('alert')).toHaveTextContent('Enter both a start and an end.');

    fireEvent.change(from, { target: { value: '2026-09-14T18:30' } });
    fireEvent.change(to, { target: { value: '2026-09-12T09:00' } });
    fireEvent.click(apply);
    expect(screen.getByRole('alert')).toHaveTextContent('Start must be before end.');

    fireEvent.change(from, { target: { value: '2020-01-01T00:00' } });
    fireEvent.change(to, { target: { value: '2030-01-01T00:00' } });
    fireEvent.click(apply);
    expect(props.onApply).toHaveBeenLastCalledWith({ from: START, to: NOW });
    expect(screen.getByRole('alert').textContent).toBe('');
  });

  it('Apply with the current times still reports up (the page refetches), with no note', () => {
    const custom = { kind: 'custom', from: WIN.from, to: WIN.to } as const;
    const { props } = setup({ sel: custom });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(props.onApply).toHaveBeenCalledTimes(1);
    expect(props.onApply).toHaveBeenLastCalledWith({ from: WIN.from, to: WIN.to });
    expect(screen.queryByText("Adjusted to this key's lifetime.")).toBeNull();
  });

  it('clamped times are written back into the fields with a note, cleared by the next edit', () => {
    const custom = { kind: 'custom', from: WIN.from, to: WIN.to } as const;
    const { props } = setup({ sel: custom });
    const from = screen.getByLabelText('From') as HTMLInputElement;
    const to = screen.getByLabelText('To') as HTMLInputElement;

    // Clamps to exactly the current window: the window never changes, so the
    // fields would otherwise keep showing what was typed (reproduced bug).
    fireEvent.change(to, { target: { value: '2030-01-01T00:00' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(props.onApply).toHaveBeenLastCalledWith({ from: WIN.from, to: NOW });
    expect(to.value).toBe(toInputValue(NOW));
    expect(from.value).toBe(toInputValue(WIN.from));
    const note = screen.getByText("Adjusted to this key's lifetime.");
    expect(note.className).toContain('text-[12px]');
    expect(note.className).toContain('text-chat-dim');
    // Same row as "Your local time".
    expect(note.parentElement).toBe(screen.getByText('Your local time').parentElement);
    expect(screen.getByRole('alert').textContent).toBe('');

    // Both ends clamped.
    fireEvent.change(from, { target: { value: '2020-01-01T00:00' } });
    expect(screen.queryByText("Adjusted to this key's lifetime.")).toBeNull(); // the edit cleared it
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(props.onApply).toHaveBeenLastCalledWith({ from: START, to: NOW });
    expect(from.value).toBe(toInputValue(START));
    expect(screen.getByText("Adjusted to this key's lifetime.")).toBeInTheDocument();

    // An in-bounds Apply shows no note.
    fireEvent.change(from, { target: { value: '2026-09-12T09:00' } });
    fireEvent.change(to, { target: { value: '2026-09-14T18:30' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(screen.queryByText("Adjusted to this key's lifetime.")).toBeNull();
    expect(from.value).toBe('2026-09-12T09:00');
  });

  it('moving the strip clears the clamping note', () => {
    const custom = { kind: 'custom', from: WIN.from, to: WIN.to } as const;
    const { props } = setup({ sel: custom });
    fireEvent.change(screen.getByLabelText('To'), { target: { value: '2030-01-01T00:00' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(screen.getByText("Adjusted to this key's lifetime.")).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('slider'), { key: 'ArrowLeft' });
    expect(props.onWindowChange).toHaveBeenCalled();
    expect(screen.queryByText("Adjusted to this key's lifetime.")).toBeNull();
  });

  it('labels a young key\'s strip with times, not five identical dates', () => {
    const { container } = setup({ bounds: { from: NOW - 6 * 3600, to: NOW }, window: { from: NOW - 3600, to: NOW } });
    const axis = container.querySelector('.justify-between.text-\\[11px\\]')!;
    expect([...axis.children].map((c) => c.textContent)).toEqual(['07:21', '08:51', '10:21', '11:51', '13:21']);
  });

  it('strip labels: day + time up to 2 days, plain dates beyond', () => {
    expect(stripLabel(NOW, 1440)).toBe('13:21');
    expect(stripLabel(NOW, 2 * 1440)).toBe('Sep 18 13:21');
    expect(stripLabel(NOW, 2 * 1440 + 1)).toBe('Sep 18');
    expect(stripLabel(NOW, 13 * 1440)).toBe('Sep 18');
  });

  // #251: a preset window's `to` is the page's unfloored "now", but a young
  // key's strip ends at the server's `to_minute` — up to a minute earlier.
  // The brush must stop at the strip's right edge rather than hang past it.
  it("clamps the brush's right edge to the strip end on a young key (no overhang)", () => {
    const bounds = { from: NOW - 10 * 60, to: NOW };     // a 10-minute-old key: 80 px per minute
    setup({ bounds, window: { from: NOW - 5 * 60, to: NOW + 45 } });
    const brush = screen.getByRole('slider');
    const left = parseFloat(brush.style.left);
    const width = parseFloat(brush.style.width);
    expect(left).toBeCloseTo(400);
    expect(left + width).toBeCloseTo(800);               // not 860

    // A window entirely past the strip's end still stays inside it.
    cleanup();
    setup({ bounds, window: { from: NOW + 30, to: NOW + 90 } });
    const b2 = screen.getByRole('slider');
    expect(parseFloat(b2.style.left) + parseFloat(b2.style.width)).toBeLessThanOrEqual(800);
    expect(parseFloat(b2.style.left)).toBeGreaterThanOrEqual(0);
  });

  it('Reset is disabled at the default period and reports up otherwise', () => {
    setup();
    expect(screen.getByRole('button', { name: 'Reset' })).toBeDisabled();
    cleanup();
    const { props } = setup({ resetDisabled: false });
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    expect(props.onReset).toHaveBeenCalledTimes(1);
  });

  it('zoomed: the strip draws the view, says so, and typed times still clamp to the lifetime', () => {
    const view = { from: NOW - 2 * 86400, to: NOW - 86400 };
    const { props } = setup({ sel: { kind: 'custom', ...view }, window: view, view, zoomed: true });
    const slider = screen.getByRole('slider');
    expect(slider).toHaveAttribute('aria-valuemin', String(view.from));
    expect(slider).toHaveAttribute('aria-valuemax', String(view.to));
    expect(screen.getByText(/^Zoomed to the applied period/)).toBeInTheDocument();
    // Earlier than the zoomed view but inside the key's life: not clamped.
    fireEvent.change(screen.getByLabelText('From'), { target: { value: toInputValue(NOW - 10 * 86400) } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(props.onApply).toHaveBeenLastCalledWith({ from: NOW - 10 * 86400, to: view.to });
    expect(screen.queryByText(/Adjusted to this key/)).not.toBeInTheDocument();
  });

  it('shows the server 422 text in the custom row', () => {
    setup({ sel: { kind: 'custom', from: WIN.from, to: WIN.to }, rangeError: 'range longer than 366 days' });
    expect(screen.getByRole('alert')).toHaveTextContent('range longer than 366 days');
  });
});
