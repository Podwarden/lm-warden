// Thinking policy controls: token detail switch, create dialog radios, token
// list chip, model settings section, in-flight chip.
import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, waitFor, act, renderHook } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { LimitsCard } from '@/components/tokens/detail/limits-card';
import { useTokenActions } from '@/components/tokens/use-token-actions';
import { ModelSettingsPage } from '@/components/models/model-settings-page';
import { LiveRequestsPanel } from '@/components/stats/live-panels';
import type { LiveRequestRow } from '@/lib/live-stats';
import { tokenDetail } from './token-fixtures';

const json = (b: unknown, status = 200) =>
  new Response(JSON.stringify(b), { status, headers: { 'Content-Type': 'application/json' } });

describe('limits card thinking switch', () => {
  afterEach(cleanup);

  it('client decides: shows the value and turning it off calls onThinking("off")', async () => {
    const onThinking = vi.fn().mockResolvedValue(true);
    render(<LimitsCard token={tokenDetail({ thinking_policy: 'client' })} onSave={vi.fn()} onThinking={onThinking} />);
    expect(screen.getByText(/^Thinking:/)).toHaveTextContent('client decides');
    expect(screen.getByText(/model.s own policy is off, it stays off/i)).toBeInTheDocument();
    const sw = screen.getByRole('switch', { name: /force thinking off/i });
    expect(sw).toHaveAttribute('aria-checked', 'false');
    fireEvent.click(sw);
    await waitFor(() => expect(onThinking).toHaveBeenCalledWith('off'));
  });

  it('forced off: shows it and the button hands the decision back', async () => {
    const onThinking = vi.fn().mockResolvedValue(true);
    render(<LimitsCard token={tokenDetail({ thinking_policy: 'off' })} onSave={vi.fn()} onThinking={onThinking} />);
    expect(screen.getByText(/^Thinking:/)).toHaveTextContent('forced off');
    const sw = screen.getByRole('switch', { name: /force thinking off/i });
    expect(sw).toHaveAttribute('aria-checked', 'true');
    expect(sw).toHaveTextContent('Let the client decide');
    fireEvent.click(sw);
    await waitFor(() => expect(onThinking).toHaveBeenCalledWith('client'));
  });

  it('shows the error', () => {
    render(<LimitsCard token={tokenDetail()} onSave={vi.fn()} onThinking={vi.fn()} thinkingError="nope" />);
    expect(screen.getByRole('alert')).toHaveTextContent('nope');
  });
});

describe('useTokenActions.setThinkingPolicy', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(cleanup);

  it('PATCHes {thinking_policy} off and back, calling onChange', async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ id: 't1' }));
    vi.stubGlobal('fetch', fetchMock);
    const onChange = vi.fn();
    const { result } = renderHook(() => useTokenActions({ id: 't1', name: 'x' }, onChange));
    for (const p of ['off', 'client'] as const) {
      let ok = false;
      await act(async () => { ok = await result.current.setThinkingPolicy(p); });
      expect(ok).toBe(true);
    }
    const bodies = fetchMock.mock.calls
      .filter(([, i]) => (i as RequestInit)?.method === 'PATCH')
      .map(([, i]) => JSON.parse(String((i as RequestInit).body)));
    expect(bodies).toEqual([{ thinking_policy: 'off' }, { thinking_policy: 'client' }]);
    expect(onChange).toHaveBeenCalledTimes(2);
  });
});

describe('create dialog thinking radios', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(cleanup);

  async function submit(pickOff: boolean) {
    const mock = vi.fn(async (_i: RequestInfo | URL, init?: RequestInit) =>
      json({ id: 'x', name: 'n', plaintext: 'vw_x', prefix: 'vw_', preview: 'vw_', expires_at: null }, init ? 201 : 200));
    vi.stubGlobal('fetch', mock);
    render(<CreateTokenDialog open onClose={() => {}} />);
    if (pickOff) fireEvent.click(screen.getByRole('radio', { name: /always off for this key/i }));
    fireEvent.change(screen.getByPlaceholderText('ci-bot'), { target: { value: 'n' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toBeInTheDocument());
    const post = mock.mock.calls.find(([, init]) => (init as RequestInit)?.method === 'POST')!;
    return JSON.parse(String((post[1] as RequestInit).body));
  }

  it('defaults to "Let the client decide" and sends thinking_policy: client', async () => {
    vi.stubGlobal('fetch', vi.fn());
    render(<CreateTokenDialog open onClose={() => {}} />);
    expect(screen.getByRole('radio', { name: /let the client decide/i })).toBeChecked();
    expect(screen.getByRole('radio', { name: /always off for this key/i })).not.toBeChecked();
    cleanup();
    expect((await submit(false)).thinking_policy).toBe('client');
  });

  it('"Always off for this key" sends thinking_policy: off', async () => {
    expect((await submit(true)).thinking_policy).toBe('off');
  });
});

describe('tokens list thinking chip', () => {
  afterEach(cleanup);

  async function row(thinking_policy: 'client' | 'off') {
    const { TokenRow } = await import('@/components/tokens/token-row');
    const item = {
      id: 't1', name: 'laptop', prefix: 'vw_abcde', preview: 'vw_abcde',
      created_at: '2026-01-01 00:00:00', last_used_at: null, expires_at: null,
      rotated_at: null, rotated_from: null, successor_id: null, successor_deleted: false,
      is_expired: false, is_near_expiry: false, revoked_at: null, is_revoked: false,
      paused_at: null, is_paused: false, priority: 5, anthropic_relay: false, thinking_policy,
      usage_24h: { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
    render(<table><tbody><TokenRow item={item as never} onChange={() => {}} /></tbody></table>);
  }

  it('shows a chip when the key is always off', async () => {
    await row('off');
    expect(screen.getByTestId('token-thinking-badge')).toHaveTextContent(/thinking off/i);
  });

  it('shows none when the client decides', async () => {
    await row('client');
    expect(screen.queryByTestId('token-thinking-badge')).toBeNull();
  });
});

describe('model settings Thinking section', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  const GPUS = {
    gpus: [{ index: 0, name: 'A4000', memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 }],
    probed_at: new Date().toISOString(), probe_error: null,
  };
  const settings = (o: Record<string, unknown> = {}) => ({
    id: 'abc', served_model_name: 'm', hf_repo: 'o/m', hf_revision: 'main', gpu_indices: [0],
    tensor_parallel_size: 1, dtype: null, max_model_len: null, gpu_memory_utilization: 0.9,
    trust_remote_code: false, extra_args: [], extra_env: {}, supports_tools: null,
    supports_vision: null, supports_reasoning: null, status: 'loaded', pulled_bytes: 0,
    pulled_total: null, last_error: null, backend: 'vllm', n_gpu_layers: null, mmproj_filename: null,
    thinking_policy: 'client', thinking_policy_supported: true as boolean | null, ...o,
  });

  async function open(s: ReturnType<typeof settings>) {
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') return new Response(JSON.stringify(s));
      if (url === '/api/models/abc/settings' && method === 'PATCH') return new Response('{"ok":true}');
      if (url === '/api/system/gpus') return new Response(JSON.stringify(GPUS));
      throw new Error(`unexpected fetch: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <Suspense fallback={null}><ModelSettingsPage id="abc" /></Suspense>
      </SWRConfig>,
    );
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    await screen.findByLabelText(/HF revision/i);
    return fetchMock;
  }

  it('is a live-band section after Replica routing; PATCHes {thinking_policy} on a loaded model', async () => {
    const fetchMock = await open(settings());
    const section = screen.getByTestId('section-thinking');
    expect(screen.getByTestId('band-live')).toContainElement(section);
    const group = screen.getByRole('radiogroup', { name: /thinking policy for this model/i });
    const clientR = screen.getByRole('radio', { name: /let the client decide/i });
    const offR = screen.getByRole('radio', { name: /off for every key/i });
    expect(group).toBeInTheDocument();
    expect(clientR).toBeChecked();
    expect(offR).not.toBeDisabled();
    fireEvent.click(offR);
    const save = screen.getByRole('button', { name: /save/i });
    await waitFor(() => expect(save).not.toBeDisabled());
    fireEvent.click(save);
    await waitFor(() =>
      expect(fetchMock.mock.calls.some((c) => (c[1] as RequestInit | undefined)?.method === 'PATCH')).toBe(true));
    const patch = fetchMock.mock.calls.find((c) => (c[1] as RequestInit | undefined)?.method === 'PATCH')!;
    expect(JSON.parse((patch[1] as RequestInit).body as string)).toEqual({ thinking_policy: 'off' });
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes('unload'))).toBe(false);
  });

  it('says the template has no thinking switch when supported is false', async () => {
    await open(settings({ thinking_policy_supported: false }));
    expect(screen.getByTestId('thinking-unsupported-note')).toHaveTextContent(
      "This model's template has no thinking switch, so this setting has no effect on it.");
  });

  it('suggests no fault when supported is null', async () => {
    await open(settings({ thinking_policy_supported: null }));
    expect(screen.queryByTestId('thinking-unsupported-note')).toBeNull();
    expect(screen.queryByText(/no thinking switch/i)).toBeNull();
    expect(screen.getByTestId('thinking-unknown-note')).toBeInTheDocument();
  });

  it('shows no note when supported is true', async () => {
    await open(settings({ thinking_policy_supported: true }));
    expect(screen.queryByTestId('thinking-unsupported-note')).toBeNull();
    expect(screen.queryByTestId('thinking-unknown-note')).toBeNull();
  });
});

describe('in-flight thinking chip', () => {
  afterEach(cleanup);
  const row = (over: Partial<LiveRequestRow> = {}): LiveRequestRow => ({
    id: 'r1', token_id: 't1', token_name: 'dev', client_ip: '10.0.0.1', model: 'model-a',
    path: '/v1/chat/completions', prompt_tokens: 1000, completion_tokens: 0, context_tokens: 1000,
    max_model_len: 200_000, context_pct: 0.005, elapsed_s: 1.2, phase: 'answering', orphan: false,
    session_id: null, cache_est_tokens: null, cache_est_pct: null, prefill_slow: false, queued_s: null,
    ...over,
  });
  const show = (rows: LiveRequestRow[]) =>
    render(<LiveRequestsPanel rows={rows} error={null} isLoading={false} />);

  it('shows the chip only when thinking_forced_off is true', () => {
    show([row({ thinking_forced_off: true })]);
    expect(screen.getAllByTestId('badge-thinking-off').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/thinking off · policy/i).length).toBeGreaterThan(0);
  });

  it('shows no chip when false or absent', () => {
    show([row({ id: 'a', thinking_forced_off: false }), row({ id: 'b' })]);
    expect(screen.queryByTestId('badge-thinking-off')).toBeNull();
  });
});
