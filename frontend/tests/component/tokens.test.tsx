import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { RotateTokenDialog } from '@/components/tokens/rotate-token-dialog';
import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { TokenRow, type TokenItem } from '@/components/tokens/token-row';
import { ExpiryBanner } from '@/components/tokens/expiry-banner';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// The Modal uses portals (document.body) and the `inert` attribute on
// #app-root; testing-library's default cleanup pulls the portal back out
// between renders only when globals: true, but vitest.config sets
// globals: false. Each test must therefore `cleanup()` itself.

function fakeItem(overrides: Partial<TokenItem> = {}): TokenItem {
  return {
    id: 't1',
    name: 'ci-bot',
    prefix: 'vw_abcde',
    preview: 'vw_abcde',
    created_at: '2026-01-01 00:00:00',
    last_used_at: null,
    expires_at: null,
    rotated_at: null,
    rotated_from: null,
    successor_id: null,
    successor_deleted: false,
    is_expired: false,
    is_near_expiry: false,
    revoked_at: null,
    // #185 — server-computed `revoked_at <= now`; false for a token whose
    // grace window is still open as well as for one that was never revoked.
    is_revoked: false,
    paused_at: null,
    is_paused: false,
    // S5 (#104) defaults — matches the backend's "mid-priority / no usage"
    // defaults for a freshly minted token.
    priority: 5,
    usage_24h: {
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
    ...overrides,
  };
}

describe('RotateTokenDialog', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('shows new plaintext once and a copy button', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_newnewnew","rotated_from":"old","id":"new","name":"ci-bot","renamed_to":"ci-bot (old 1)"}', { status: 201 })));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_newnewnew')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('surfaces the new active name and the predecessor\'s renamed_to (#150)', async () => {
    // Pin both names so a future change that drops the "renamed_to" field
    // from the response (or the dialog stops reading it) surfaces in CI
    // instead of as a quiet UI regression — operators rely on the modal
    // to confirm WHICH row is now the active one.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_renametest","rotated_from":"old","id":"new","name":"prod-bot","renamed_to":"prod-bot (old 2)"}',
      { status: 201 },
    )));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_renametest')).toBeInTheDocument());
    // Active token name in the success footer.
    expect(screen.getByText('prod-bot')).toBeInTheDocument();
    // Predecessor's new name after the rename.
    expect(screen.getByText('prod-bot (old 2)')).toBeInTheDocument();
  });

  it('clears plaintext from DOM after close', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_secret_value","rotated_from":"old","id":"new"}',
      { status: 201 },
    )));
    const onClose = vi.fn();
    render(<RotateTokenDialog open tokenId="old" onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_secret_value')).toBeInTheDocument());

    // Click Done — this should call reset() (wiping the plaintext from
    // local state) and then onClose. The parent in this test deliberately
    // does NOT flip `open`, so the dialog stays mounted; that lets us
    // assert the wipe happened by checking the plaintext is no longer
    // rendered. If a future change removes reset() from handleClose,
    // this test will fail because the post-success view would still be
    // visible.
    fireEvent.click(screen.getByRole('button', { name: /done/i }));
    expect(onClose).toHaveBeenCalled();
    expect(screen.queryByText('vw_secret_value')).not.toBeInTheDocument();
  });

  it('uses execCommand fallback on non-secure context and reports Copied (#149)', async () => {
    // Mirrors a production setup: navigator.clipboard is undefined
    // because the page is served over plain HTTP. Without the textarea
    // fallback added in #149 this test would land on the "select
    // manually" failure path — pinning the success path here prevents
    // a regression that drops the fallback.
    Object.defineProperty(window, 'isSecureContext', {
      value: false, configurable: true, writable: true,
    });
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    // jsdom drops document.execCommand (deprecated WHATWG API); attach
    // an own property so vi.spyOn has something to wrap.
    (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand = () => true;
    const execSpy = vi.spyOn(document, 'execCommand').mockReturnValue(true);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_fallback","rotated_from":"old","id":"new"}', { status: 201 })));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_fallback')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument());
    expect(execSpy).toHaveBeenCalledWith('copy');
    expect(screen.queryByText(/select and copy the token manually/i)).not.toBeInTheDocument();
  });

  // ---- #185: grace vs immediate revoke -------------------------------------

  function rotateOk(extra = '') {
    return vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_rotated","rotated_from":"old","id":"new","name":"ci-bot",' +
      `"renamed_to":"ci-bot (old 1)"${extra}}`,
      { status: 201 },
    ));
  }

  function bodyOf(fetchMock: ReturnType<typeof vi.fn>) {
    return JSON.parse(String(fetchMock.mock.calls[0][1].body));
  }

  it('defaults to the grace mode and submits grace_hours: 24 (#185 AC2)', async () => {
    const f = rotateOk();
    vi.stubGlobal('fetch', f);
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    expect((screen.getByTestId('rotate-mode-grace') as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: /^rotate$/i }));
    await waitFor(() => expect(f).toHaveBeenCalled());
    expect(bodyOf(f).grace_hours).toBe(24);
  });

  it('submits grace_hours: 0 when "revoke immediately" is chosen (#185 AC1)', async () => {
    const f = rotateOk();
    vi.stubGlobal('fetch', f);
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByTestId('rotate-mode-immediate'));
    // The hours input is gone in immediate mode — the mode IS the value.
    expect(screen.queryByLabelText(/grace period \(hours\)/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^rotate$/i }));
    await waitFor(() => expect(f).toHaveBeenCalled());
    expect(bodyOf(f).grace_hours).toBe(0);
  });

  it('warns that in-flight requests are not cancelled (#185)', () => {
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    expect(screen.getByText(/finish or hit the read timeout/i)).toBeInTheDocument();
  });

  it('rejects a blank grace field instead of silently hard-revoking (#185)', async () => {
    // Regression guard: `Number("") === 0`, so before #185 clearing the box —
    // the natural "never mind, leave the default" gesture — passed validation
    // and POSTed grace_hours: 0, hard-revoking a token nobody meant to cut.
    const f = rotateOk();
    vi.stubGlobal('fetch', f);
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/grace period \(hours\)/i), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /^rotate$/i }));
    await waitFor(() =>
      expect(screen.getByText(/enter a grace period in hours/i)).toBeInTheDocument(),
    );
    expect(f).not.toHaveBeenCalled();
  });

  it('resets to grace mode after close so the next token is not hard-cut (#185)', async () => {
    const f = rotateOk();
    vi.stubGlobal('fetch', f);
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByTestId('rotate-mode-immediate'));
    expect((screen.getByTestId('rotate-mode-immediate') as HTMLInputElement).checked).toBe(true);

    // Cancel calls handleClose() → reset(). The parent in this test keeps the
    // dialog mounted, which is exactly how we can observe the reset.
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect((screen.getByTestId('rotate-mode-grace') as HTMLInputElement).checked).toBe(true);
    expect((screen.getByTestId('rotate-mode-immediate') as HTMLInputElement).checked).toBe(false);
  });

  it('narrates the hard cut from the grace_hours the server echoed (#185)', async () => {
    vi.stubGlobal('fetch', rotateOk(',"grace_hours":0'));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByTestId('rotate-mode-immediate'));
    fireEvent.click(screen.getByRole('button', { name: /^rotate$/i }));
    await waitFor(() => expect(screen.getByText('vw_rotated')).toBeInTheDocument());
    const outcome = screen.getByTestId('rotate-outcome').textContent ?? '';
    expect(outcome).toMatch(/revoked now/i);
    expect(outcome).not.toMatch(/keep working through the grace period/i);
  });

  it('narrates the grace window when the server echoed a non-zero grace (#185)', async () => {
    vi.stubGlobal('fetch', rotateOk(',"grace_hours":24'));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /^rotate$/i }));
    await waitFor(() => expect(screen.getByText('vw_rotated')).toBeInTheDocument());
    const outcome = screen.getByTestId('rotate-outcome').textContent ?? '';
    expect(outcome).toMatch(/keep working through the grace period/i);
    expect(outcome).not.toMatch(/revoked now/i);
  });

  it('shows "select manually" only when BOTH copy paths fail (#149)', async () => {
    Object.defineProperty(window, 'isSecureContext', {
      value: false, configurable: true, writable: true,
    });
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand = () => false;
    vi.spyOn(document, 'execCommand').mockReturnValue(false);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_doomed","rotated_from":"old","id":"new"}', { status: 201 })));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_doomed')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() =>
      expect(screen.getByText(/select and copy the token manually/i)).toBeInTheDocument(),
    );
  });
});

describe('CreateTokenDialog', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('shows plaintext once after successful create', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_freshtoken","prefix":"vw_fresh","preview":"vw_fresh","expires_at":null}',
      { status: 201 },
    )));
    render(<CreateTokenDialog open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_freshtoken')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('has no rate limit field and never sends rate_limit_tps', async () => {
    const f = vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_freshtoken","prefix":"vw_fresh","preview":"vw_fresh","expires_at":null}',
      { status: 201 },
    ));
    vi.stubGlobal('fetch', f);
    render(<CreateTokenDialog open onClose={() => {}} />);
    expect(screen.queryByLabelText(/rate limit/i)).toBeNull();
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_freshtoken')).toBeInTheDocument());
    const body = JSON.parse(String((f.mock.calls[0][1] as RequestInit).body));
    expect(body).toEqual({ name: 'ci-bot', expires_in_days: 365, priority: 5 });
  });

  it('clears plaintext from DOM after close', async () => {
    // Mirror of the RotateTokenDialog wipe test. Pins the §11.6
    // "plaintext must NOT remain accessible — clear it from state on
    // close" invariant for the create dialog too. Without this, removing
    // `reset()` from handleClose() would slip past CI.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_createdsecret","prefix":"vw_creat","preview":"vw_creat","expires_at":null}',
      { status: 201 },
    )));
    const onClose = vi.fn();
    render(<CreateTokenDialog open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_createdsecret')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /done/i }));
    expect(onClose).toHaveBeenCalled();
    expect(screen.queryByText('vw_createdsecret')).not.toBeInTheDocument();
  });

  it('uses execCommand fallback on non-secure context and reports Copied (#149)', async () => {
    Object.defineProperty(window, 'isSecureContext', {
      value: false, configurable: true, writable: true,
    });
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand = () => true;
    const execSpy = vi.spyOn(document, 'execCommand').mockReturnValue(true);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_create_fallback","prefix":"vw_create_fb","preview":"vw_create_fb","expires_at":null}',
      { status: 201 },
    )));
    render(<CreateTokenDialog open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_create_fallback')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument());
    expect(execSpy).toHaveBeenCalledWith('copy');
    expect(screen.queryByText(/select and copy the token manually/i)).not.toBeInTheDocument();
  });
});

describe('TokenRow', () => {
  afterEach(() => { cleanup(); });

  it('renders all spec columns and surfaces created_at', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem({ created_at: '2026-02-03 10:11:12' })} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.getByText('ci-bot')).toBeInTheDocument();
    expect(screen.getByText(/vw_abcde/)).toBeInTheDocument();
    // The created_at column renders something derived from the timestamp.
    // We don't pin on the exact locale string — just that it's there.
    const created = screen.getByTestId('token-created');
    expect(created.textContent).toBeTruthy();
  });

  it('renders no Rate cell (per-token rate limits were removed)', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem()} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.queryByTestId('token-rate')).toBeNull();
    expect(screen.queryByText('unlimited')).toBeNull();
  });

  it('shows "Expiring soon" status when is_near_expiry', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem({ is_near_expiry: true })} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.getByText(/expiring soon/i)).toBeInTheDocument();
  });

  it('shows "Expired" status when is_expired', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem({ is_expired: true, is_near_expiry: false })} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.getByText(/^expired$/i)).toBeInTheDocument();
  });

  it('shows "Rotated (orphan)" when rotated and successor deleted', () => {
    render(
      <table><tbody>
        <TokenRow
          item={fakeItem({ rotated_at: '2026-02-01 00:00:00', successor_deleted: true })}
          onChange={() => {}}
        />
      </tbody></table>,
    );
    expect(screen.getByText(/orphan/i)).toBeInTheDocument();
  });

  it('shows "Rotated (grace)" while the predecessor still authenticates (#185)', () => {
    render(
      <table><tbody>
        <TokenRow
          item={fakeItem({
            rotated_at: '2026-02-01 00:00:00',
            revoked_at: '2026-02-02 00:00:00',
            is_revoked: false,
          })}
          onChange={() => {}}
        />
      </tbody></table>,
    );
    expect(screen.getByText(/rotated \(grace\)/i)).toBeInTheDocument();
  });

  it('shows "Rotated (revoked)" once the predecessor is cut off (#185)', () => {
    // Before #185 this row read amber "Rotated (grace)" forever, because the
    // only revoked branch was gated on `rotated_at == null`. An operator who
    // had just hard-revoked a leaked key was told a grace window was open.
    render(
      <table><tbody>
        <TokenRow
          item={fakeItem({
            rotated_at: '2026-02-01 00:00:00',
            revoked_at: '2026-02-01 00:00:00',
            is_revoked: true,
          })}
          onChange={() => {}}
        />
      </tbody></table>,
    );
    expect(screen.getByText(/rotated \(revoked\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/rotated \(grace\)/i)).not.toBeInTheDocument();
  });

  it('surfaces delete error to the operator', async () => {
    // Without the r.ok branch in token-row.tsx::onDelete, a 500 from
    // DELETE /api/tokens/:id was silently swallowed: the row would
    // disappear from the optimistic refresh, then reappear on the next
    // 10s poll with no explanation. This test pins that the failure
    // mode surfaces inline.
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('confirm', () => true);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"detail":"token in use"}', { status: 500 },
    )));
    render(
      <table><tbody>
        <TokenRow item={fakeItem()} onChange={() => {}} />
      </tbody></table>,
    );
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByRole('alert').textContent).toMatch(/token in use/i);
    vi.unstubAllGlobals();
  });
});

describe('ExpiryBanner', () => {
  afterEach(() => { cleanup(); });

  it('hides when nothing is near expiry', () => {
    const { container } = render(<ExpiryBanner count={0} />);
    expect(container.textContent).toBe('');
  });

  it('renders the server-side count, not the page in hand', () => {
    // `near_expiry` counts the whole list; the page may hold none of them.
    render(<ExpiryBanner count={1234} />);
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toMatch(/expiring soon \(1,234\)/i);
    expect(alert.textContent).toMatch(/1,234 tokens expire within 30 days/);
    expect(screen.queryByRole('button', { name: /show them/i })).not.toBeInTheDocument();
  });

  it('offers to show just those tokens', () => {
    const onShow = vi.fn();
    render(<ExpiryBanner count={1} onShow={onShow} />);
    expect(screen.getByRole('alert').textContent).toMatch(/1 token expires within 30 days/);
    fireEvent.click(screen.getByRole('button', { name: /show them/i }));
    expect(onShow).toHaveBeenCalledOnce();
  });
});
