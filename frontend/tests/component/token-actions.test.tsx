import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act, renderHook, fireEvent, waitFor } from '@testing-library/react';
import { TokenRow, type TokenItem } from '@/components/tokens/token-row';
import { RotateTokenDialog } from '@/components/tokens/rotate-token-dialog';
import { useTokenActions, errorDetail } from '@/components/tokens/use-token-actions';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

function item(o: Partial<TokenItem> = {}): TokenItem {
  return {
    id: 't1', name: 'ci-bot', prefix: 'vw_abcde', preview: 'vw_abcde', created_at: '2026-01-01 00:00:00',
    last_used_at: null, expires_at: null, rotated_at: null, rotated_from: null, successor_id: null,
    successor_deleted: false, is_expired: false, is_near_expiry: false, revoked_at: null, is_revoked: false,
    paused_at: null, is_paused: false, priority: 5,
    usage_24h: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }, ...o,
  };
}
const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

describe('useTokenActions', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it('setPaused sends one PATCH {paused} and calls onChange', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ id: 't1', is_paused: true }));
    vi.stubGlobal('fetch', fetchMock);
    const onChange = vi.fn();
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'ci-bot' }, onChange));
    let ok = false;
    await act(async () => { ok = await result.current.setPaused(true); });
    expect(ok).toBe(true);
    const patches = fetchMock.mock.calls.filter(([, init]) => (init as RequestInit)?.method === 'PATCH');
    expect(patches).toHaveLength(1);
    expect(patches[0][0]).toBe('/api/tokens/t1');
    expect(JSON.parse(String((patches[0][1] as RequestInit).body))).toEqual({ paused: true });
    expect(onChange).toHaveBeenCalled();
  });

  it('a 409 on pause surfaces the server wording', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json({ detail: 'token is expired' }, 409)));
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'x' }, () => {}));
    await act(async () => { await result.current.setPaused(true); });
    expect(result.current.pauseError).toBe('token is expired');
  });

  it('runTest reports paused keys and the model count', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(json({ ok: true, revoked: false, expired: false, paused: true, proxy_reachable: true, allowed_models: ['m'] }))
      .mockResolvedValueOnce(json({ ok: true, revoked: false, expired: false, paused: false, proxy_reachable: true, allowed_models: ['m', 'n'] })));
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'x' }, () => {}));
    let r1, r2;
    await act(async () => { r1 = await result.current.runTest(); });
    expect(r1).toMatchObject({ ok: false, paused: true, detail: 'token is paused' });
    await act(async () => { r2 = await result.current.runTest(); });
    expect(r2).toMatchObject({ ok: true, models: 2, detail: '2 model(s) reachable' });
  });
});

describe('useTokenActions re-entrancy (#251)', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it('runTest ignores a second call while one is running, then runs again once it is done', async () => {
    let release!: (r: Response) => void;
    const body = { ok: true, revoked: false, expired: false, paused: false, proxy_reachable: true, allowed_models: ['m'] };
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => new Promise<Response>((r) => { release = r; }))
      .mockResolvedValue(json(body));
    vi.stubGlobal('fetch', fetchMock);
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'x' }, () => {}));

    let first!: Promise<unknown>;
    let second: unknown = 'unset';
    await act(async () => {
      first = result.current.runTest();
      second = await result.current.runTest(); // same frame: `testing` state is still false
    });
    expect(second).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => { release(json(body)); await first; });
    expect(await first).toMatchObject({ ok: true, models: 1 });

    let third: unknown;
    await act(async () => { third = await result.current.runTest(); });
    expect(third).toMatchObject({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('remove resolves null (nothing sent) when the confirm is cancelled', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'x' }, () => {}));
    let r: unknown = 'unset';
    await act(async () => { r = await result.current.remove(); });
    expect(r).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('TokenRow delete (#251)', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  const renderRow = (onChange: () => void) =>
    render(<table><tbody><TokenRow item={item()} onChange={onChange} /></tbody></table>);

  it('does not refresh the list when the confirm is cancelled', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    const onChange = vi.fn();
    renderRow(onChange);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Delete' })); });
    expect(window.confirm).toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('refreshes after a delete, and after a failed one (the row comes back)', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(json({ detail: 'boom' }, 500)));
    const onChange = vi.fn();
    renderRow(onChange);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Delete' })); });
    expect(onChange).toHaveBeenCalledTimes(1);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Delete' })); });
    expect(onChange).toHaveBeenCalledTimes(2);
    expect(screen.getByText('boom')).toBeInTheDocument();
  });
});

describe('errorDetail', () => {
  // Controller ruling: PATCH 422s from Pydantic return `detail` as an ARRAY
  // of { msg } objects, not a string (409s stay plain strings). errorDetail
  // must reuse token-series.ts's detailOf rather than `String(body.detail)`,
  // which would render "[object Object]" for the array shape.
  it('surfaces the msg text from a Pydantic validation-error array, not "[object Object]"', async () => {
    const r = json({ detail: [{ msg: 'Input should be greater than or equal to 1' }] }, 422);
    const msg = await errorDetail(r, 'fallback');
    expect(msg).toBe('Input should be greater than or equal to 1');
    expect(msg).not.toMatch(/\[object Object\]/);
  });
});

describe('TokenRow additions', () => {
  afterEach(() => cleanup());

  it('links the name to the details page', () => {
    render(<table><tbody><TokenRow item={item()} onChange={() => {}} /></tbody></table>);
    expect(screen.getByRole('link', { name: 'ci-bot' })).toHaveAttribute('href', '/tokens/t1');
  });

  it('shows Paused ahead of every other status', () => {
    render(<table><tbody><TokenRow item={item({ is_paused: true, paused_at: '2026-09-18 13:21:00', is_near_expiry: true })} onChange={() => {}} /></tbody></table>);
    expect(screen.getByText('Paused')).toBeInTheDocument();
    expect(screen.queryByText('Expiring soon')).toBeNull();
  });
});

describe('RotateTokenDialog onRotated', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('hands the successor id over when the success modal is closed', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(json(
      { plaintext: 'vw_new', rotated_from: 'old', id: 'new-id', name: 'ci-bot', renamed_to: 'ci-bot (old 1)', grace_hours: 24 }, 201)));
    const onRotated = vi.fn();
    const onClose = vi.fn();
    render(<RotateTokenDialog open tokenId="old" onClose={onClose} onRotated={onRotated} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_new')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /done/i }));
    expect(onClose).toHaveBeenCalled();
    expect(onRotated).toHaveBeenCalledWith('new-id');
  });
});
