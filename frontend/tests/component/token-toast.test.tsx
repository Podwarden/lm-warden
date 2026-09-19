import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, renderHook, act } from '@testing-library/react';
import { Toast, useToast, type ToastApi } from '@/components/tokens/detail/toast';

const api = (visible: boolean): ToastApi => ({ message: 'Token paused', visible, show: () => {} });

describe('Toast', () => {
  afterEach(() => cleanup());

  it('uses the mockup transition: opacity + transform, 180 ms, default `ease`', () => {
    render(<Toast api={api(false)} />);
    const el = screen.getByRole('status');
    expect(el.className).toContain('transition-[opacity,transform]');
    expect(el.className).toContain('duration-[180ms]');
    expect(el.className).toContain('ease-[ease]');
    expect(el.className).toContain('translate-y-5');
    expect(el.className).toContain('opacity-0');
  });

  it('slides in when visible', () => {
    render(<Toast api={api(true)} bottomPx={64} />);
    const el = screen.getByRole('status');
    expect(el).toHaveTextContent('Token paused');
    expect(el.className).toContain('translate-y-0');
    expect(el.className).toContain('opacity-100');
    expect(el.style.bottom).toBe('64px');
  });
});

describe('useToast', () => {
  afterEach(() => { vi.useRealTimers(); });

  it('shows a message, hides it after 2200 ms, and a re-show restarts the clock', () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useToast());
    expect(result.current.visible).toBe(false);

    act(() => result.current.show('Name saved'));
    expect(result.current).toMatchObject({ message: 'Name saved', visible: true });
    act(() => { vi.advanceTimersByTime(2199); });
    expect(result.current.visible).toBe(true);
    act(() => { vi.advanceTimersByTime(1); });
    expect(result.current.visible).toBe(false);
    expect(result.current.message).toBe('Name saved'); // kept while it fades out

    act(() => result.current.show('Token paused'));
    act(() => { vi.advanceTimersByTime(2000); });
    act(() => result.current.show('Token resumed')); // re-show 200 ms before the first would hide
    expect(result.current).toMatchObject({ message: 'Token resumed', visible: true });
    act(() => { vi.advanceTimersByTime(2199); });
    expect(result.current.visible).toBe(true);        // the old timer did not fire
    act(() => { vi.advanceTimersByTime(1); });
    expect(result.current.visible).toBe(false);
  });

  it('keeps `show` stable across renders and clears its timer on unmount', () => {
    vi.useFakeTimers();
    const { result, rerender, unmount } = renderHook(() => useToast());
    const show = result.current.show;
    rerender();
    expect(result.current.show).toBe(show);
    act(() => result.current.show('x'));
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
