import { describe, it, expect, vi, afterEach } from 'vitest';
process.env.TZ = 'UTC';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';
import { TokenHeader, deriveDetailStatus, type TokenHeaderProps } from '@/components/tokens/detail/token-header';
import { NOW, tokenDetail } from './token-fixtures';

function renderHeader(p: Partial<TokenHeaderProps> = {}) {
  const props: TokenHeaderProps = {
    token: tokenDetail(), nowSec: NOW, pausing: false, testing: false, deleting: false, pauseError: null,
    onRename: vi.fn().mockResolvedValue(true), onPauseToggle: vi.fn(), onTest: vi.fn(), onRotate: vi.fn(), onDelete: vi.fn(),
    ...p,
  };
  render(<TokenHeader {...props} />);
  return props;
}

describe('TokenHeader', () => {
  afterEach(() => cleanup());

  it('renders the mockup header copy', () => {
    renderHeader();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('opencode-laptop');
    expect(screen.getByText('vw_6gqfa…')).toBeInTheDocument();
    expect(screen.getByText('Created Sep 6, 13:59')).toBeInTheDocument();
    expect(screen.getByText('Never expires')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
    for (const name of ['Pause', 'Test', 'Rotate', 'Delete']) {
      expect(screen.getByRole('button', { name })).toBeEnabled();
    }
  });

  // The mockup's "API tokens / <name>" row is the app-wide breadcrumb strip
  // now (components/breadcrumb-header.tsx; the page names the crumb through
  // useBreadcrumb). The header must not draw a second trail, and nothing may
  // sit above the title row — the strip took the removed row's height, so
  // the title stays put.
  it('draws no breadcrumb of its own; the title row is the first thing in it', () => {
    const { container } = render(
      <TokenHeader
        token={tokenDetail()} nowSec={NOW} pausing={false} testing={false} deleting={false} pauseError={null}
        onRename={vi.fn()} onPauseToggle={vi.fn()} onTest={vi.fn()} onRotate={vi.fn()} onDelete={vi.fn()}
      />,
    );
    expect(screen.queryByRole('navigation', { name: 'Breadcrumb' })).toBeNull();
    expect(screen.queryByRole('link', { name: 'API tokens' })).toBeNull();
    const first = container.firstElementChild!;
    expect(first.className).toBe('flex flex-wrap items-start justify-between gap-4');
    expect(first.querySelector('h1')).toHaveTextContent('opencode-laptop');
  });

  it('lays the pencil icon out inline on the baseline, like the mockup (preflight makes svg block + middle)', () => {
    renderHeader();
    const svg = screen.getByRole('button', { name: 'Rename token' }).querySelector('svg')!;
    expect(svg).toHaveClass('inline', 'align-baseline');
  });

  it('rename: Enter saves the trimmed name, Esc cancels', async () => {
    const p = renderHeader();
    fireEvent.click(screen.getByRole('button', { name: 'Rename token' }));
    const input = screen.getByRole('textbox', { name: 'Token name' });
    expect(input).toHaveValue('opencode-laptop');
    expect(input).toHaveFocus();
    expect(screen.queryByRole('heading', { level: 1 })).toBeNull();

    fireEvent.keyDown(input, { key: 'Escape' });
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
    expect(p.onRename).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Rename token' }));
    const again = screen.getByRole('textbox', { name: 'Token name' });
    fireEvent.change(again, { target: { value: '  laptop key  ' } });
    await act(async () => { fireEvent.submit(again.closest('form')!); });
    expect(p.onRename).toHaveBeenCalledWith('laptop key');
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('rename: a failed save (onRename resolves false) keeps the form open with the draft', async () => {
    const p = renderHeader({ onRename: vi.fn().mockResolvedValue(false) });
    fireEvent.click(screen.getByRole('button', { name: 'Rename token' }));
    const input = screen.getByRole('textbox', { name: 'Token name' });
    fireEvent.change(input, { target: { value: 'taken name' } });
    await act(async () => { fireEvent.submit(input.closest('form')!); });
    expect(p.onRename).toHaveBeenCalledWith('taken name');
    expect(screen.getByRole('textbox', { name: 'Token name' })).toHaveValue('taken name');
    expect(screen.getByRole('button', { name: 'Save name' })).toBeEnabled(); // not stuck saving
    expect(screen.queryByRole('heading', { level: 1 })).toBeNull();
  });

  it('rename: Cancel button and blank names do nothing', async () => {
    const p = renderHeader();
    fireEvent.click(screen.getByRole('button', { name: 'Rename token' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Token name' }), { target: { value: '   ' } });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Save name' })); });
    expect(p.onRename).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('badge order is Paused → Revoked → Expired → Grace → Active', () => {
    const base = { is_paused: false, is_revoked: false, is_expired: false, rotated_at: null };
    expect(deriveDetailStatus({ ...base, is_paused: true, is_revoked: true, is_expired: true, rotated_at: 'x' }).label).toBe('Paused');
    expect(deriveDetailStatus({ ...base, is_revoked: true, is_expired: true, rotated_at: 'x' }).label).toBe('Revoked');
    expect(deriveDetailStatus({ ...base, is_expired: true, rotated_at: 'x' }).label).toBe('Expired');
    expect(deriveDetailStatus({ ...base, rotated_at: 'x' }).label).toBe('Grace');
    expect(deriveDetailStatus(base).label).toBe('Active');
  });

  it('paused: Resume button, Paused badge and the amber band', () => {
    renderHeader({ token: tokenDetail({ is_paused: true, paused_at: '2026-09-18 13:21:00' }) });
    expect(screen.getByRole('button', { name: 'Resume' })).toBeInTheDocument();
    expect(screen.getByText('Paused')).toBeInTheDocument();
    const band = screen.getByRole('status');
    expect(band).toHaveTextContent(
      'Paused since 13:21. New requests with this key get 403 token paused. Requests that were already running will finish. Resume to let it back in immediately.',
    );
  });

  it('keeps Resume enabled on a key that is paused AND expired (or revoked)', () => {
    renderHeader({ token: tokenDetail({ is_paused: true, paused_at: '2026-09-18 13:21:00', is_expired: true, expires_at: '2026-09-18 12:00:00' }) });
    expect(screen.getByRole('button', { name: 'Resume' })).toBeEnabled();
    cleanup();
    renderHeader({ token: tokenDetail({ is_paused: true, paused_at: '2026-09-18 13:21:00', is_revoked: true }) });
    expect(screen.getByRole('button', { name: 'Resume' })).toBeEnabled();
    cleanup();
    renderHeader({ token: tokenDetail({ is_paused: true, paused_at: '2026-09-18 13:21:00', is_expired: true }), pausing: true });
    expect(screen.getByRole('button', { name: 'Resume' })).toBeDisabled(); // only while the PATCH runs
  });

  it('disables Pause on a dead key and Rotate on a rotated one; shows the pause error', () => {
    renderHeader({
      token: tokenDetail({ is_revoked: true, rotated_at: '2026-09-06 13:59:00' }),
      pauseError: 'token is revoked',
    });
    expect(screen.getByRole('button', { name: 'Pause' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Rotate' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('token is revoked');
  });
});
