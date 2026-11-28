// Reasoning budget (#348): the key's limits card, the model's Thinking section, the Stats pill.
import { Suspense } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';
import { SWRConfig } from 'swr';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { LimitsCard } from '@/components/tokens/detail/limits-card';
import { ModelSettingsPage } from '@/components/models/model-settings-page';
import { finishClassOf } from '@/lib/request-history';
import { tokenDetail } from './token-fixtures';

describe('limits card reasoning budget', () => {
  afterEach(cleanup);

  it('shows no cap, and saving a budget with a mode sends both', async () => {
    const onReasoning = vi.fn().mockResolvedValue(null);
    render(<LimitsCard token={tokenDetail()} onSave={vi.fn()} onReasoning={onReasoning} />);
    const box = screen.getByTestId('reasoning-budget');
    expect(box).toHaveTextContent('no cap');
    const save = screen.getByRole('button', { name: /save budget/i });
    expect(save).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/reasoning budget \(tokens\)/i), { target: { value: '2048' } });
    fireEvent.change(screen.getByLabelText(/when the budget is spent/i), { target: { value: 'stop' } });
    expect(save).not.toBeDisabled();
    fireEvent.click(save);
    await waitFor(() =>
      expect(onReasoning).toHaveBeenCalledWith({ reasoning_budget: 2048, reasoning_budget_mode: 'stop' }));
  });

  it('shows the saved cap and clearing the box sends null', async () => {
    const onReasoning = vi.fn().mockResolvedValue(null);
    render(
      <LimitsCard
        token={tokenDetail({ reasoning_budget: 512, reasoning_budget_mode: 'force_answer' })}
        onSave={vi.fn()}
        onReasoning={onReasoning}
      />,
    );
    expect(screen.getByTestId('reasoning-budget')).toHaveTextContent('512 tokens');
    fireEvent.change(screen.getByLabelText(/reasoning budget \(tokens\)/i), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /save budget/i }));
    await waitFor(() =>
      expect(onReasoning).toHaveBeenCalledWith({ reasoning_budget: null, reasoning_budget_mode: 'force_answer' }));
  });

  it('refuses a budget below 1 and shows the server error', async () => {
    const onReasoning = vi.fn().mockResolvedValue('reasoning_budget: too small');
    render(<LimitsCard token={tokenDetail()} onSave={vi.fn()} onReasoning={onReasoning} />);
    const input = screen.getByLabelText(/reasoning budget \(tokens\)/i);
    fireEvent.change(input, { target: { value: '0' } });
    expect(screen.getByRole('button', { name: /save budget/i })).toBeDisabled();
    fireEvent.change(input, { target: { value: '5' } });
    fireEvent.click(screen.getByRole('button', { name: /save budget/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent('too small');
  });
});

describe('request history outcome', () => {
  it('draws a budget-ended request like a length-cut one', () => {
    expect(finishClassOf('reasoning_budget')).toBe('length');
    expect(finishClassOf('stop')).toBe('stop');
    expect(finishClassOf('runaway_think')).toBe('other');
  });
});

describe('model settings reasoning budget', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  const GPUS = {
    gpus: [{ index: 0, name: 'A4000', memory_total_mib: 16376, memory_used_mib: 0, utilization_pct: 0 }],
    probed_at: new Date().toISOString(), probe_error: null,
  };
  const settings = {
    id: 'abc', served_model_name: 'm', hf_repo: 'o/m', hf_revision: 'main', gpu_indices: [0],
    tensor_parallel_size: 1, dtype: null, max_model_len: null, gpu_memory_utilization: 0.9,
    trust_remote_code: false, extra_args: [], extra_env: {}, supports_tools: null,
    supports_vision: null, supports_reasoning: null, status: 'loaded', pulled_bytes: 0,
    pulled_total: null, last_error: null, backend: 'vllm', n_gpu_layers: null, mmproj_filename: null,
    thinking_policy: 'client', thinking_policy_supported: true, reasoning_budget: null,
    reasoning_budget_mode: 'force_answer',
  };

  it('saves a budget and a mode live, without unloading the model', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : (input as Request).url;
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url === '/api/models/abc/settings' && method === 'GET') return new Response(JSON.stringify(settings));
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
    const section = screen.getByTestId('section-thinking');
    expect(screen.getByTestId('band-live')).toContainElement(section);
    fireEvent.change(screen.getByLabelText(/reasoning budget/i), { target: { value: '4096' } });
    fireEvent.click(screen.getByRole('button', { name: /when the budget is spent/i }));
    fireEvent.mouseDown(screen.getByRole('option', { name: 'stop' }));
    const save = screen.getByRole('button', { name: /save/i });
    await waitFor(() => expect(save).not.toBeDisabled());
    fireEvent.click(save);
    await waitFor(() =>
      expect(fetchMock.mock.calls.some((c) => (c[1] as RequestInit | undefined)?.method === 'PATCH')).toBe(true));
    const patch = fetchMock.mock.calls.find((c) => (c[1] as RequestInit | undefined)?.method === 'PATCH')!;
    expect(JSON.parse((patch[1] as RequestInit).body as string)).toEqual({
      reasoning_budget: 4096,
      reasoning_budget_mode: 'stop',
    });
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes('unload'))).toBe(false);
  });
});
