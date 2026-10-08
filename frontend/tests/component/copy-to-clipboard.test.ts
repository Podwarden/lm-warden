// Unit tests for the shared copyToClipboard helper (lib/utils.ts).
//
// Issue #149: on a real deployment the warden UI is served over plain
// HTTP via a VPN or the LAN (http://192.0.2.10:8080), where
// `navigator.clipboard` is undefined. The "Copy" buttons on the
// freshly-minted token modal therefore always landed on the
// "select manually" fallback. The helper now tries the async API
// first, then falls back to a hidden-`<textarea>` +
// `document.execCommand("copy")`, and only rejects when BOTH paths
// fail — so the "manual" hint is reserved for actual failure.
//
// These tests pin the three branches the AC calls out:
//   1. navigator.clipboard available  → uses writeText, resolves
//   2. navigator.clipboard undefined + execCommand returns true
//      → uses textarea fallback, resolves
//   3. navigator.clipboard undefined + execCommand returns false
//      → both paths failed, rejects so caller can render fallback hint

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { copyToClipboard } from '@/lib/utils';

describe('copyToClipboard', () => {
  // Pin a secure-context default so the navigator.clipboard branch is
  // even reachable when stubbed in. jsdom leaves window.isSecureContext
  // unset (=> falsy) which would otherwise force every test through the
  // textarea fallback regardless of what we mock onto navigator.
  beforeEach(() => {
    Object.defineProperty(window, 'isSecureContext', {
      value: true,
      configurable: true,
      writable: true,
    });
    // jsdom does not implement document.execCommand — it's removed from
    // the WHATWG spec and only kept by browser engines for legacy compat.
    // vi.spyOn requires the property to already exist as own/own-proto,
    // so we attach a stub default first; individual tests then mock its
    // return value via vi.spyOn or vi.fn.
    if (typeof (document as { execCommand?: unknown }).execCommand !== 'function') {
      (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand =
        () => false;
    }
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    // Restore non-secure default so accidental cross-test leakage is
    // visible immediately.
    Object.defineProperty(window, 'isSecureContext', {
      value: false,
      configurable: true,
      writable: true,
    });
  });

  it('uses navigator.clipboard.writeText when available and resolves', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('navigator', {
      ...navigator,
      clipboard: { writeText },
    } as unknown as Navigator);

    await expect(copyToClipboard('vw_secret')).resolves.toBeUndefined();
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText).toHaveBeenCalledWith('vw_secret');
  });

  it('falls back to execCommand when navigator.clipboard is undefined and resolves on success', async () => {
    // Drop the clipboard API entirely — this is the non-secure
    // production case we are actually fixing.
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    const execSpy = vi.spyOn(document, 'execCommand').mockReturnValue(true);

    await expect(copyToClipboard('vw_secret')).resolves.toBeUndefined();
    expect(execSpy).toHaveBeenCalledWith('copy');

    // The hidden textarea must be cleaned up — otherwise we'd leak DOM
    // nodes containing the plaintext on every copy click.
    const leaked = document.querySelectorAll('textarea');
    expect(leaked).toHaveLength(0);
  });

  it('rejects when both navigator.clipboard is undefined AND execCommand returns false', async () => {
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    const execSpy = vi.spyOn(document, 'execCommand').mockReturnValue(false);

    await expect(copyToClipboard('vw_secret')).rejects.toThrow(/clipboard/i);
    expect(execSpy).toHaveBeenCalledWith('copy');

    // Cleanup still happens even on failure — see finally block in
    // lib/utils.ts. Without this assertion a regression that removed
    // the finally would silently leak the textarea.
    const leaked = document.querySelectorAll('textarea');
    expect(leaked).toHaveLength(0);
  });

  it('rejects when execCommand synchronously throws', async () => {
    // Some hardened browsers throw NotAllowedError from execCommand
    // instead of returning false. The helper catches that and treats
    // it as a failure rather than leaking the throw up to the caller.
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    vi.spyOn(document, 'execCommand').mockImplementation(() => {
      throw new Error('NotAllowedError');
    });

    await expect(copyToClipboard('vw_secret')).rejects.toThrow(/clipboard/i);
    expect(document.querySelectorAll('textarea')).toHaveLength(0);
  });
});
