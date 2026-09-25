import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup, act, waitFor } from '@testing-library/react';

// The setup wizard's admin step checks the server's new-password rule
// (app/auth/passwords.py: >= 12 characters, <= 72 bytes) before it posts,
// and shows the rule next to the field.

const push = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push }),
}));

import AdminPage from '@/app/setup/admin/page';
import { newPasswordProblem } from '@/lib/password-policy';

function fill(username: string, password: string, confirm = password) {
  const [pw, cf] = screen.getAllByLabelText(/password/i);
  fireEvent.change(screen.getByLabelText(/username/i), { target: { value: username } });
  fireEvent.change(pw, { target: { value: password } });
  fireEvent.change(cf, { target: { value: confirm } });
}

async function submit() {
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: /create admin/i }));
  });
}

describe('setup admin step — password rule', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    push.mockReset();
    fetchMock = vi.fn(async () => new Response(JSON.stringify({ step: 'done' }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    render(<AdminPage />);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('shows the rule next to the field', () => {
    expect(screen.getByText('At least 12 characters (at most 72 bytes).')).toBeInTheDocument();
  });

  it('refuses 11 characters without posting', async () => {
    fill('admin', 'a'.repeat(11));
    await submit();
    expect(screen.getByText('Password must be at least 12 characters')).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses more than 72 bytes without posting', async () => {
    fill('admin', 'é'.repeat(37));
    await submit();
    expect(screen.getByText('Password must be at most 72 bytes')).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('posts a 12-character password', async () => {
    fill('admin', 'twelve-chars');
    await submit();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ username: 'admin', password: 'twelve-chars' });
    await waitFor(() => expect(push).toHaveBeenCalledWith('/setup/done'));
  });
});

describe('newPasswordProblem', () => {
  it('counts characters, not UTF-16 units, and bytes for the ceiling', () => {
    expect(newPasswordProblem('😀'.repeat(11))).toMatch(/at least 12/); // 22 UTF-16 units
    expect(newPasswordProblem('😀'.repeat(12))).toBeNull(); // 48 bytes
    expect(newPasswordProblem('😀'.repeat(19))).toMatch(/at most 72 bytes/); // 76 bytes
    expect(newPasswordProblem('é'.repeat(36))).toBeNull(); // exactly 72 bytes
  });
});
