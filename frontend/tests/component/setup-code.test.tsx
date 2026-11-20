import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react';

// The first-run setup code (app/setup/bootstrap.py): a fresh install asks for
// it before the wizard does anything, and every setup POST carries it.

import {
  SetupCodeGate,
  clearSetupCode,
  getSetupCode,
  setSetupCode,
  setupPost,
} from '@/app/setup/setup-code';

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('SetupCodeGate', () => {
  beforeEach(() => clearSetupCode());
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows the step when no code is required (an installed box, the Mac app)', () => {
    render(<SetupCodeGate required={false} source={null}><p>step</p></SetupCodeGate>);
    expect(screen.getByText('step')).toBeInTheDocument();
  });

  it('asks for the code first, and says where the installer put it', () => {
    render(<SetupCodeGate required source="env"><p>step</p></SetupCodeGate>);
    expect(screen.queryByText('step')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /enter the setup code/i })).toBeInTheDocument();
    expect(screen.getByText(/LM_SETUP_CODE in the \.env file/)).toBeInTheDocument();
  });

  it('points at the log and the data directory for a generated code', () => {
    render(<SetupCodeGate required source="file"><p>step</p></SetupCodeGate>);
    expect(screen.getByText(/docker compose exec api cat \/data\/setup-code/)).toBeInTheDocument();
  });

  it('keeps the code for this tab and shows the step', () => {
    render(<SetupCodeGate required source="file"><p>step</p></SetupCodeGate>);
    fireEvent.change(screen.getByLabelText(/setup code/i), { target: { value: ' ABCD-EFGH ' } });
    fireEvent.click(screen.getByRole('button', { name: /continue/i }));
    expect(screen.getByText('step')).toBeInTheDocument();
    expect(getSetupCode()).toBe('ABCD-EFGH');
  });

  it('asks again, with a reason, when the server refuses the code', async () => {
    setSetupCode('wrong-code');
    const fetchMock = vi.fn(async () => json({ detail: 'setup code invalid' }, 403));
    vi.stubGlobal('fetch', fetchMock);
    render(<SetupCodeGate required source="file"><p>step</p></SetupCodeGate>);
    expect(screen.getByText('step')).toBeInTheDocument();
    await act(async () => {
      await setupPost('/api/setup/welcome');
    });
    expect(getSetupCode()).toBeNull();
    expect(screen.getByRole('alert')).toHaveTextContent(/not accepted/);
    expect(screen.queryByText('step')).not.toBeInTheDocument();
  });
});

describe('setupPost', () => {
  beforeEach(() => clearSetupCode());
  afterEach(() => vi.unstubAllGlobals());

  it('sends the code as X-Setup-Code with a JSON body', async () => {
    setSetupCode('K7QM-3XRT-9WPD-HN2F');
    const fetchMock = vi.fn(async () => json({ step: 'hf_token' }));
    vi.stubGlobal('fetch', fetchMock);
    await setupPost('/api/setup/gpus', { allowed_gpu_indices: [0] });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/setup/gpus');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['X-Setup-Code']).toBe('K7QM-3XRT-9WPD-HN2F');
    expect(JSON.parse(init.body as string)).toEqual({ allowed_gpu_indices: [0] });
  });

  it('sends no code header when the tab has none (the Mac app needs none)', async () => {
    const fetchMock = vi.fn(async () => json({ step: 'gpus' }));
    vi.stubGlobal('fetch', fetchMock);
    await setupPost('/api/setup/welcome');
    const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1];
    expect((init.headers as Record<string, string>)['X-Setup-Code']).toBeUndefined();
  });

  it('keeps the code on an unrelated 403', async () => {
    setSetupCode('keep-me-please');
    vi.stubGlobal('fetch', vi.fn(async () => json({ detail: 'origin mismatch' }, 403)));
    await setupPost('/api/setup/welcome');
    expect(getSetupCode()).toBe('keep-me-please');
  });
});
