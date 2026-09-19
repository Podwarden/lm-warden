import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWindowSelection } from '@/components/tokens/detail/use-window-selection';
import type { RangeSel } from '@/lib/token-series';

const SEL: RangeSel = { kind: 'preset', preset: '7d' };
const COMMITTED = { from: 1000, to: 2000 };

describe('useWindowSelection', () => {
  afterEach(() => { vi.useRealTimers(); });

  it('debounces several onWindowChange calls into exactly one URL write after 250 ms, using the LAST window', () => {
    vi.useFakeTimers();
    const replaceQuery = vi.fn();
    const { result } = renderHook(() => useWindowSelection(SEL, COMMITTED, replaceQuery));

    act(() => result.current.onWindowChange({ from: 1100, to: 1900 }));
    act(() => { vi.advanceTimersByTime(100); });
    act(() => result.current.onWindowChange({ from: 1150, to: 1950 }));
    act(() => { vi.advanceTimersByTime(100); });
    act(() => result.current.onWindowChange({ from: 1200, to: 2000 }));

    // Neither of the first two calls' timers ever reached 250 ms — each new
    // call cancels the previous one's pending write.
    expect(replaceQuery).not.toHaveBeenCalled();
    act(() => { vi.advanceTimersByTime(249); });
    expect(replaceQuery).not.toHaveBeenCalled();
    act(() => { vi.advanceTimersByTime(1); });
    expect(replaceQuery).toHaveBeenCalledTimes(1);
    expect(replaceQuery).toHaveBeenCalledWith('from=1200&to=2000');
  });

  it('shows the drag immediately via `pending`, then clears it once `sel` (the URL) catches up', () => {
    vi.useFakeTimers();
    const replaceQuery = vi.fn();
    // `committed` tracks `sel` the way the page's `resolveWindow(sel, nowSec)`
    // does for a custom selection: it becomes exactly {from, to} once the URL
    // reflects the drag.
    const { result, rerender } = renderHook<ReturnType<typeof useWindowSelection>, { sel: RangeSel; committed: typeof COMMITTED }>(
      ({ sel, committed }) => useWindowSelection(sel, committed, replaceQuery),
      { initialProps: { sel: SEL, committed: COMMITTED } },
    );

    expect(result.current.pending).toBeNull();
    expect(result.current.shownWindow).toEqual(COMMITTED);

    act(() => result.current.onWindowChange({ from: 1100, to: 1900 }));
    expect(result.current.pending).toEqual({ from: 1100, to: 1900 });
    expect(result.current.shownWindow).toEqual({ from: 1100, to: 1900 });
    expect(result.current.shownSel).toEqual({ kind: 'custom', from: 1100, to: 1900 });

    act(() => { vi.advanceTimersByTime(250); });
    expect(replaceQuery).toHaveBeenCalledTimes(1);
    expect(replaceQuery).toHaveBeenCalledWith('from=1100&to=1900');
    // The write fired, but nothing has told this hook the URL changed yet
    // (that only happens once the page re-renders with the new `sel`) — the
    // drag's own values still drive the display.
    expect(result.current.pending).toEqual({ from: 1100, to: 1900 });

    // Simulate the URL catching up: the page re-derives `sel` (and so
    // `committed`) from `useSearchParams` once `router.replace`'s query
    // lands, and passes the new values back in as props.
    rerender({ sel: { kind: 'custom', from: 1100, to: 1900 }, committed: { from: 1100, to: 1900 } });
    expect(result.current.pending).toBeNull();
    expect(result.current.shownWindow).toEqual({ from: 1100, to: 1900 });
    expect(result.current.shownSel).toEqual({ kind: 'custom', from: 1100, to: 1900 });
  });

  it('a preset click clears any pending drag and writes immediately, with no debounce', () => {
    vi.useFakeTimers();
    const replaceQuery = vi.fn();
    const { result } = renderHook(() => useWindowSelection(SEL, COMMITTED, replaceQuery));

    act(() => result.current.onWindowChange({ from: 1100, to: 1900 }));
    expect(result.current.pending).not.toBeNull();

    act(() => result.current.onPreset('24h'));
    expect(replaceQuery).toHaveBeenCalledTimes(1);
    expect(replaceQuery).toHaveBeenCalledWith('range=24h');
    expect(result.current.pending).toBeNull();

    // The cancelled drag's debounce timer must not still fire later.
    act(() => { vi.advanceTimersByTime(250); });
    expect(replaceQuery).toHaveBeenCalledTimes(1);
  });

  it('onApply with a new window writes the URL at once and cancels a pending drag', () => {
    vi.useFakeTimers();
    const replaceQuery = vi.fn();
    const custom: RangeSel = { kind: 'custom', from: 1000, to: 2000 };
    const { result } = renderHook(() => useWindowSelection(custom, COMMITTED, replaceQuery));

    act(() => result.current.onWindowChange({ from: 1100, to: 1900 }));
    let changed: boolean | undefined;
    act(() => { changed = result.current.onApply({ from: 1200, to: 1800 }); });
    expect(changed).toBe(true);
    expect(replaceQuery).toHaveBeenCalledTimes(1);
    expect(replaceQuery).toHaveBeenCalledWith('from=1200&to=1800');
    expect(result.current.shownWindow).toEqual({ from: 1200, to: 1800 });

    act(() => { vi.advanceTimersByTime(250); });
    expect(replaceQuery).toHaveBeenCalledTimes(1); // the drag's write never lands
  });

  it('onApply with the window already in the URL reports "unchanged" and writes nothing', () => {
    vi.useFakeTimers();
    const replaceQuery = vi.fn();
    const custom: RangeSel = { kind: 'custom', from: 1000, to: 2000 };
    const { result } = renderHook(() => useWindowSelection(custom, COMMITTED, replaceQuery));

    // Even with a drag waiting on its debounce, Applying the URL's own window
    // drops the drag and snaps the display back.
    act(() => result.current.onWindowChange({ from: 1100, to: 1900 }));
    let changed: boolean | undefined;
    act(() => { changed = result.current.onApply({ from: 1000, to: 2000 }); });
    expect(changed).toBe(false);
    expect(result.current.pending).toBeNull();
    expect(result.current.shownWindow).toEqual(COMMITTED);
    act(() => { vi.advanceTimersByTime(250); });
    expect(replaceQuery).not.toHaveBeenCalled();
  });
});
