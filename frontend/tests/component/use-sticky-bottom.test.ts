/**
 * Regression for the live-log "won't stick to bottom" bug.
 *
 * Symptom (operator report): during a live vLLM startup burst the log tail
 * stops following new lines and the operator has to click "Jump to latest"
 * over and over.
 *
 * Root cause: `useStickyBottom` gated react-virtuoso's `followOutput` on its
 * own stick/free latch, and the latch flips to "free" on ANY
 * `atBottomStateChange(false)` with no path back except the button. Under a
 * ~200 line/s append burst Virtuoso emits a *transient* `atBottom=false`
 * (the just-appended row sits below the viewport for a frame before the
 * follow-scroll lands — and "smooth" can't animate fast enough to keep up).
 * That transient false permanently switched `followOutput` to `false`,
 * disabling auto-follow until the operator clicked "Jump to latest".
 *
 * Fix contract pinned here: `followOutput` is ALWAYS the instant "auto"
 * behavior, decoupled from the latch. react-virtuoso only auto-scrolls when
 * the list is already at the bottom, so an always-on follow keeps the tail
 * pinned through bursts yet never yanks a user who has scrolled up to read.
 * The `mode` latch survives, but only to drive "Jump to latest" visibility.
 */

import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useStickyBottom } from '@/components/shared/use-sticky-bottom';

describe('useStickyBottom', () => {
  it('keeps auto-follow enabled after a transient not-at-bottom during a streaming burst', () => {
    const { result } = renderHook(() => useStickyBottom('stick'));

    // Virtuoso reports not-at-bottom for a frame while a fresh row lands
    // below the viewport mid-burst. This MUST NOT disable auto-follow —
    // otherwise the tail stops and the operator is forced to click
    // "Jump to latest" (the reported bug).
    act(() => result.current.onAtBottomStateChange(false));

    expect(result.current.followOutput).not.toBe(false);
  });

  it('uses the instant "auto" follow behavior so smooth-scroll lag cannot unstick the tail', () => {
    const { result } = renderHook(() => useStickyBottom('stick'));

    // "smooth" animates each scroll and falls behind a 200-line/s burst,
    // leaving the viewport perpetually shy of the bottom. "auto" jumps
    // instantly and keeps pace.
    expect(result.current.followOutput).toBe('auto');
  });

  it('still flips mode to "free" on scroll-up so the "Jump to latest" button can appear', () => {
    const { result } = renderHook(() => useStickyBottom('stick'));

    act(() => result.current.onAtBottomStateChange(false));

    expect(result.current.mode).toBe('free');
  });

  it('jumpToLatest restores stick mode', () => {
    const { result } = renderHook(() => useStickyBottom('stick'));

    act(() => result.current.onAtBottomStateChange(false));
    expect(result.current.mode).toBe('free');

    act(() => result.current.jumpToLatest());
    expect(result.current.mode).toBe('stick');
  });
});
