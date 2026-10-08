import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { LimitsCard } from '@/components/tokens/detail/limits-card';
import { tokenDetail } from './token-fixtures';

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

describe('create dialog relay checkbox', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); });

  async function submit(tick: boolean) {
    const mock = vi.fn(async (_i: RequestInfo | URL, init?: RequestInit) =>
      json({ id: 'x', name: 'n', plaintext: 'vw_x', prefix: 'vw_', preview: 'vw_', expires_at: null }, init ? 201 : 200));
    vi.stubGlobal('fetch', mock);
    render(<CreateTokenDialog open onClose={() => {}} />);
    const box = screen.getByTestId('token-anthropic-relay') as HTMLInputElement;
    if (tick) fireEvent.click(box);
    fireEvent.change(screen.getByPlaceholderText('ci-bot'), { target: { value: 'n' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toBeInTheDocument());
    const post = mock.mock.calls.find(([, init]) => (init as RequestInit)?.method === 'POST')!;
    return JSON.parse(String((post[1] as RequestInit).body));
  }

  it('is off by default and sends anthropic_relay: false', async () => {
    vi.stubGlobal('fetch', vi.fn());
    render(<CreateTokenDialog open onClose={() => {}} />);
    expect(screen.getByTestId('token-anthropic-relay')).not.toBeChecked();
    expect(screen.getByLabelText(/May relay to Anthropic \(Claude Code router\)/)).toBeInTheDocument();
    cleanup();
    expect((await submit(false)).anthropic_relay).toBe(false);
  });

  it('ticked sends anthropic_relay: true', async () => {
    expect((await submit(true)).anthropic_relay).toBe(true);
  });
});

describe('limits card relay toggle', () => {
  afterEach(() => { cleanup(); });

  it('shows the state and a warning, and toggling calls onRelay with the new value', async () => {
    const onRelay = vi.fn().mockResolvedValue(true);
    render(<LimitsCard token={tokenDetail({ anthropic_relay: false })} onSave={vi.fn()} onRelay={onRelay} />);
    expect(screen.getByText(/Relay to Anthropic/)).toHaveTextContent('off');
    expect(screen.getByText(/own Anthropic credential/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', { name: /relay to anthropic/i }));
    await waitFor(() => expect(onRelay).toHaveBeenCalledWith(true));
  });

  it('reads on and turns it off', async () => {
    const onRelay = vi.fn().mockResolvedValue(true);
    render(<LimitsCard token={tokenDetail({ anthropic_relay: true })} onSave={vi.fn()} onRelay={onRelay} />);
    expect(screen.getByText(/Relay to Anthropic/)).toHaveTextContent('on');
    fireEvent.click(screen.getByRole('switch', { name: /relay to anthropic/i }));
    await waitFor(() => expect(onRelay).toHaveBeenCalledWith(false));
  });

  it('shows the error when the PATCH fails', async () => {
    render(<LimitsCard token={tokenDetail()} onSave={vi.fn()} onRelay={vi.fn()} relayError="nope" />);
    expect(screen.getByRole('alert')).toHaveTextContent('nope');
  });
});

describe('useTokenActions.setAnthropicRelay', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); });

  it('PATCHes {anthropic_relay} and calls onChange', async () => {
    const { renderHook, act } = await import('@testing-library/react');
    const { useTokenActions } = await import('@/components/tokens/use-token-actions');
    const fetchMock = vi.fn().mockResolvedValue(json({ id: 't1' }));
    vi.stubGlobal('fetch', fetchMock);
    const onChange = vi.fn();
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'x' }, onChange));
    let ok = false;
    await act(async () => { ok = await result.current.setAnthropicRelay(true); });
    expect(ok).toBe(true);
    const p = fetchMock.mock.calls.find(([, i]) => (i as RequestInit)?.method === 'PATCH')!;
    expect(p[0]).toBe('/api/tokens/t1');
    expect(JSON.parse(String((p[1] as RequestInit).body))).toEqual({ anthropic_relay: true });
    expect(onChange).toHaveBeenCalled();
  });
});

describe('tokens list relay badge', () => {
  afterEach(() => { cleanup(); });

  async function row(anthropic_relay: boolean) {
    const { TokenRow } = await import('@/components/tokens/token-row');
    const item = {
      id: 't1', name: 'laptop', prefix: 'vw_abcde', preview: 'vw_abcde',
      created_at: '2026-01-01 00:00:00', last_used_at: null, expires_at: null,
      rotated_at: null, rotated_from: null, successor_id: null, successor_deleted: false,
      is_expired: false, is_near_expiry: false, revoked_at: null, is_revoked: false,
      paused_at: null, is_paused: false, priority: 5, anthropic_relay,
      usage_24h: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
    render(<table><tbody><TokenRow item={item as never} onChange={() => {}} /></tbody></table>);
  }

  it('shows a Relay badge on a key that may relay', async () => {
    await row(true);
    expect(screen.getByTestId('token-relay-badge')).toHaveTextContent('Relay');
    expect(screen.getByTestId('token-relay-badge')).toHaveAttribute('title', expect.stringMatching(/Anthropic/));
  });

  it('shows none otherwise', async () => {
    await row(false);
    expect(screen.queryByTestId('token-relay-badge')).toBeNull();
  });
});
