import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';
import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';

// Router redesign T2: the router page opens this dialog inline with
// presets (name "claude-code", relay on) and receives the plaintext via
// onCreated so the Connect snippet can show the real key once.

const CREATED = {
  id: 'new', name: 'claude-code', plaintext: 'vw_relaysecret', prefix: 'vw_relay',
  preview: 'vw_relay', expires_at: null,
};

function stubCreate(status = 201, body: unknown = CREATED) {
  const f = vi.fn().mockImplementation(async () =>
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', f);
  return f;
}

function postBody(f: ReturnType<typeof vi.fn>) {
  const call = f.mock.calls.find(([, init]) => (init as RequestInit)?.method === 'POST')!;
  return JSON.parse(String((call[1] as RequestInit).body));
}

describe('CreateTokenDialog presets', () => {
  beforeEach(() => { setAccessToken('jwt'); setCsrfToken('csrf'); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('prefills the name and ticks relay from the presets', () => {
    stubCreate();
    render(<CreateTokenDialog open onClose={() => {}} initialName="claude-code" initialRelay />);
    expect(screen.getByLabelText(/name/i)).toHaveValue('claude-code');
    expect(screen.getByTestId('token-anthropic-relay')).toBeChecked();
  });

  it('keeps the defaults when no presets are given', () => {
    stubCreate();
    render(<CreateTokenDialog open onClose={() => {}} />);
    expect(screen.getByLabelText(/name/i)).toHaveValue('');
    expect(screen.getByTestId('token-anthropic-relay')).not.toBeChecked();
  });

  it('POSTs the preset name with anthropic_relay:true without any edit', async () => {
    const f = stubCreate();
    render(<CreateTokenDialog open onClose={() => {}} initialName="claude-code" initialRelay />);
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toHaveTextContent('vw_relaysecret'));
    expect(postBody(f)).toEqual({
      name: 'claude-code', expires_in_days: 365, priority: 5, anthropic_relay: true,
    });
  });

  it('presets stay editable: unticking relay sends false', async () => {
    const f = stubCreate();
    render(<CreateTokenDialog open onClose={() => {}} initialName="claude-code" initialRelay />);
    fireEvent.click(screen.getByTestId('token-anthropic-relay'));
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'laptop' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toBeInTheDocument());
    expect(postBody(f)).toMatchObject({ name: 'laptop', anthropic_relay: false });
  });

  it('calls onCreated with the plaintext exactly once, and still shows it once in the dialog', async () => {
    stubCreate();
    const onCreated = vi.fn();
    render(
      <CreateTokenDialog open onClose={() => {}} initialName="claude-code" initialRelay onCreated={onCreated} />,
    );
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByTestId('new-token')).toHaveTextContent('vw_relaysecret'));
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(onCreated).toHaveBeenCalledWith('vw_relaysecret', { anthropicRelay: true });
    // Copy and re-render do not re-fire it.
    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /cop(ied|y)/i })).toBeInTheDocument());
    expect(onCreated).toHaveBeenCalledTimes(1);
  });

  it('does not call onCreated when the POST fails', async () => {
    stubCreate(422, { detail: 'name taken' });
    const onCreated = vi.fn();
    render(<CreateTokenDialog open onClose={() => {}} initialName="claude-code" initialRelay onCreated={onCreated} />);
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('name taken')).toBeInTheDocument());
    expect(onCreated).not.toHaveBeenCalled();
  });

  it('closing clears the plaintext and reopening shows the preset form again', async () => {
    stubCreate();
    const onCreated = vi.fn();
    const onClose = vi.fn();
    const { rerender } = render(
      <CreateTokenDialog open onClose={onClose} initialName="claude-code" initialRelay onCreated={onCreated} />,
    );
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_relaysecret')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /done/i }));
    expect(onClose).toHaveBeenCalled();
    expect(screen.queryByText('vw_relaysecret')).not.toBeInTheDocument();

    rerender(<CreateTokenDialog open={false} onClose={onClose} initialName="claude-code" initialRelay onCreated={onCreated} />);
    rerender(<CreateTokenDialog open onClose={onClose} initialName="claude-code" initialRelay onCreated={onCreated} />);
    expect(screen.queryByText('vw_relaysecret')).not.toBeInTheDocument();
    expect(screen.getByLabelText(/name/i)).toHaveValue('claude-code');
    expect(screen.getByTestId('token-anthropic-relay')).toBeChecked();
    expect(onCreated).toHaveBeenCalledTimes(1);
  });
});
