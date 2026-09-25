import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from 'vitest';
import { useState, type ReactNode } from 'react';
import { SWRConfig, useSWRConfig } from 'swr';
import { render, screen, cleanup, fireEvent, act, renderHook, waitFor } from '@testing-library/react';
import { GodModeDock, useDockHeight, useGodModeEnabled, clampDockHeight, DOCK_HEIGHT_KEY } from '@/components/tokens/detail/godmode-dock';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

class FakeES {
  static all: FakeES[] = [];
  onopen?: () => void;
  onmessage?: (e: MessageEvent) => void;
  onerror?: () => void;
  closed = false;
  constructor(public url: string) { FakeES.all.push(this); setTimeout(() => this.onopen?.(), 0); }
  close() { this.closed = true; }
}

function Harness({ includesEarlier = true }: { includesEarlier?: boolean }) {
  const [open, setOpen] = useState(false);
  const [h, setH] = useState(300);
  return (
    <GodModeDock tokenName="opencode-laptop" tokenIds={['old1', 'self']} includesEarlier={includesEarlier}
      open={open} onOpenChange={setOpen} height={h} onHeightChange={(px) => setH(px)} />
  );
}

// Wires the real useDockHeight hook (rather than the Harness's plain
// useState) so a window resize's re-clamp is exercised end to end, through
// to the rendered body element's inline height.
function ResizingHarness() {
  const [open, setOpen] = useState(true);
  const [h, setH] = useDockHeight();
  return (
    <GodModeDock tokenName="opencode-laptop" tokenIds={['self']} includesEarlier={false}
      open={open} onOpenChange={setOpen} height={h} onHeightChange={setH} />
  );
}

// jsdom has no PointerEvent; MouseEvent carries clientY, which is all the grip reads.
beforeAll(() => {
  if (!('PointerEvent' in window)) (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = MouseEvent;
});

const bar = () => screen.getByRole('button', { name: /God mode/ });
const flush = () => act(async () => { await vi.advanceTimersByTimeAsync(0); });

describe('GodModeDock', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    vi.useFakeTimers();
    FakeES.all = [];
    setAccessToken('jwt');
    setCsrfToken('csrf');
    vi.stubGlobal('EventSource', FakeES as unknown as typeof EventSource);
    fetchMock = vi.fn().mockResolvedValue(new Response('{"ticket":"t1"}'));
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

  // #251: without a label the bar's accessible name was all of its text
  // (title, hint, counts, button labels) — and changed as it streamed.
  it('names the dock bar "God mode", open or closed', async () => {
    render(<Harness />);
    expect(screen.getByRole('button', { name: 'God mode' })).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(bar());
    await flush();
    expect(screen.getByRole('button', { name: 'God mode' })).toHaveAttribute('aria-expanded', 'true');
  });

  it('starts closed and opens NO connection while closed', async () => {
    render(<Harness />);
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(bar()).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByText("Open to watch this key's prompts and responses as they happen")).toBeInTheDocument();
    expect(FakeES.all).toHaveLength(0);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('opening starts exactly one stream for the token set; closing ends it', async () => {
    render(<Harness />);
    fireEvent.click(bar());
    await flush();
    expect(bar()).toHaveAttribute('aria-expanded', 'true');
    expect(FakeES.all).toHaveLength(1);
    expect(FakeES.all[0].url).toBe('/api/admin/godmode/stream?token_ids=old1%2Cself&ticket=t1');
    expect(screen.getByText('opencode-laptop and its earlier keys · replayed the last requests, now streaming')).toBeInTheDocument();
    expect(screen.getByText('Live')).toBeInTheDocument();
    expect(screen.getByText('0 requests · 0 running')).toBeInTheDocument();

    fireEvent.click(bar());
    await flush();
    expect(FakeES.all[0].closed).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(FakeES.all).toHaveLength(1);           // no background reconnect
    expect(screen.queryByText('Live')).toBeNull();
  });

  it('turns the chevron with the mockup transition (transform .15s, default ease)', async () => {
    render(<Harness />);
    const chevron = () => [...bar().querySelectorAll('svg')].at(-1)!;
    const chev = chevron();
    expect(chev).toHaveClass('transition-transform', 'duration-150', 'ease-[ease]');
    expect(chev).not.toHaveClass('rotate-180');
    fireEvent.click(bar());
    await flush();
    expect(chevron()).toHaveClass('rotate-180');
  });

  it('Enter and Space toggle; the tool buttons do not', async () => {
    render(<Harness includesEarlier={false} />);
    fireEvent.keyDown(bar(), { key: 'Enter' });
    await flush();
    expect(bar()).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('opencode-laptop · replayed the last requests, now streaming')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    fireEvent.click(screen.getByRole('button', { name: 'Following' }));
    expect(bar()).toHaveAttribute('aria-expanded', 'true');
    fireEvent.keyDown(bar(), { key: ' ' });
    await flush();
    expect(bar()).toHaveAttribute('aria-expanded', 'false');
  });

  it('dragging the grip resizes the dock', async () => {
    render(<Harness />);
    fireEvent.click(bar());
    await flush();
    const grip = screen.getByTitle('Drag to resize');
    fireEvent.pointerDown(grip, { clientY: 500, pointerId: 1 });
    fireEvent.pointerMove(grip, { clientY: 400, pointerId: 1 });
    fireEvent.pointerUp(grip, { clientY: 400, pointerId: 1 });
    expect(document.getElementById(bar().getAttribute('aria-controls')!)).toHaveStyle({ height: '400px' });
  });

  it('a cancelled drag stops resizing — a later pointermove leaves the height unchanged', async () => {
    render(<Harness />);
    fireEvent.click(bar());
    await flush();
    const grip = screen.getByTitle('Drag to resize');
    const body = document.getElementById(bar().getAttribute('aria-controls')!)!;
    expect(body).toHaveStyle({ height: '300px' });
    fireEvent.pointerDown(grip, { clientY: 500, pointerId: 1 });
    fireEvent.pointerCancel(grip, { clientY: 500, pointerId: 1 });
    fireEvent.pointerMove(grip, { clientY: 100, pointerId: 1 });
    expect(body).toHaveStyle({ height: '300px' });
  });

  it('re-clamps the dock height when the window shrinks, without persisting it', async () => {
    window.localStorage.clear();
    vi.stubGlobal('innerHeight', 1000);
    render(<ResizingHarness />);
    await flush();
    const grip = screen.getByTitle('Drag to resize');
    const body = document.getElementById(bar().getAttribute('aria-controls')!)!;
    // Drag up to 850px — exactly 85% of the 1000px viewport, persisted.
    fireEvent.pointerDown(grip, { clientY: 500, pointerId: 1 });
    fireEvent.pointerMove(grip, { clientY: 100, pointerId: 1 });
    fireEvent.pointerUp(grip, { clientY: 100, pointerId: 1 });
    expect(body).toHaveStyle({ height: '850px' });
    expect(window.localStorage.getItem(DOCK_HEIGHT_KEY)).toBe('850');

    // Shrink the viewport to 600px (85% = 510px) and fire a resize.
    vi.stubGlobal('innerHeight', 600);
    act(() => { window.dispatchEvent(new Event('resize')); });
    expect(body).toHaveStyle({ height: '510px' });
    // Re-clamp never persists — the saved preference still reflects the drag.
    expect(window.localStorage.getItem(DOCK_HEIGHT_KEY)).toBe('850');
    window.localStorage.clear();
  });
});

describe('dock height', () => {
  afterEach(() => { vi.restoreAllMocks(); window.localStorage.clear(); });

  it('clamps to 160 px … 85% of the viewport', () => {
    expect(clampDockHeight(50, 1000)).toBe(160);
    expect(clampDockHeight(9999, 1000)).toBe(850);
    expect(clampDockHeight(400.4, 1000)).toBe(400);
  });

  it('defaults to 45% of the viewport and restores a saved height', () => {
    const { result: fresh } = renderHook(() => useDockHeight());
    expect(fresh.current[0]).toBe(Math.round(window.innerHeight * 0.45));
    window.localStorage.setItem(DOCK_HEIGHT_KEY, '300');
    const { result } = renderHook(() => useDockHeight());
    expect(result.current[0]).toBe(300);
    act(() => result.current[1](420, true));
    expect(window.localStorage.getItem(DOCK_HEIGHT_KEY)).toBe('420');
  });

  it('survives a throwing localStorage', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('denied'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('denied'); });
    const { result } = renderHook(() => useDockHeight());
    expect(result.current[0]).toBe(Math.round(window.innerHeight * 0.45));
    act(() => result.current[1](300, true));
    expect(result.current[0]).toBe(300);
  });
});

// Spec §4.5/§5: the dock renders only when the status endpoint says enabled;
// loading or any error hides it — including an error on a LATER refetch,
// which SWR reports alongside the last good `data` (#251).
describe('useGodModeEnabled', () => {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>{children}</SWRConfig>
  );
  const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
  function useProbe() {
    return { enabled: useGodModeEnabled(), mutate: useSWRConfig().mutate };
  }

  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('is false while loading', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})));
    const { result } = renderHook(useProbe, { wrapper });
    expect(result.current.enabled).toBe(false);
  });

  it('is false when the status fetch fails', async () => {
    const f = vi.fn().mockResolvedValue(json({ detail: 'boom' }, 500));
    vi.stubGlobal('fetch', f);
    const { result } = renderHook(useProbe, { wrapper });
    await waitFor(() => expect(f).toHaveBeenCalled());
    await act(async () => { await new Promise((r) => setTimeout(r, 10)); });
    expect(result.current.enabled).toBe(false);
  });

  it('is true for {enabled: true} and false for {enabled: false}', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ enabled: true })));
    const on = renderHook(useProbe, { wrapper });
    await waitFor(() => expect(on.result.current.enabled).toBe(true));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ enabled: false })));
    const off = renderHook(useProbe, { wrapper });
    await act(async () => { await new Promise((r) => setTimeout(r, 10)); });
    expect(off.result.current.enabled).toBe(false);
  });

  it('turns false when a refetch fails, although SWR still holds the last {enabled: true}', async () => {
    const f = vi.fn().mockResolvedValue(json({ enabled: true }));
    vi.stubGlobal('fetch', f);
    const { result } = renderHook(useProbe, { wrapper });
    await waitFor(() => expect(result.current.enabled).toBe(true));

    f.mockResolvedValue(json({ detail: 'boom' }, 500));
    await act(async () => { await result.current.mutate('/api/admin/godmode/status').catch(() => {}); });
    expect(result.current.enabled).toBe(false);
  });
});
