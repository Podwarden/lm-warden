import { describe, it, expect } from 'vitest';
import { _resolvePublicBaseUrl } from '@/lib/public-url';

// ---------------------------------------------------------------------------
// _resolvePublicBaseUrl unit tests — pin the contract for the helper that
// powers user-facing snippets (curl examples, OpenAI client configs).
//
// The function is exported under the `_` prefix specifically so we can
// unit-test it without setting up a fake `window.location.origin`.
//
// Behaviours pinned:
//   * settings.public_url branch: when set + http/https + parses, the
//     trailing slash is stripped, scheme/host preserved verbatim.
//   * fallback branch: empty / null / undefined / whitespace public_url
//     falls back to origin, with the same trailing-slash strip applied.
//   * defence in depth: a misconfigured row (`ftp://x`) that snuck past
//     the backend coercer must NOT propagate to snippets — fall back to
//     origin instead.
// ---------------------------------------------------------------------------

describe('_resolvePublicBaseUrl — settings.public_url branch', () => {
  it('strips a trailing slash from a valid https URL', () => {
    expect(
      _resolvePublicBaseUrl('https://warden.example.com/', 'http://localhost:8000'),
    ).toBe('https://warden.example.com');
  });

  it('strips multiple trailing slashes', () => {
    expect(
      _resolvePublicBaseUrl('https://warden.example.com///', 'http://localhost:8000'),
    ).toBe('https://warden.example.com');
  });

  it('preserves a URL without a trailing slash verbatim', () => {
    expect(
      _resolvePublicBaseUrl('https://warden.example.com', 'http://localhost:8000'),
    ).toBe('https://warden.example.com');
  });

  it('preserves a non-default port', () => {
    expect(
      _resolvePublicBaseUrl('https://warden.example.com:8443/', 'http://localhost:8000'),
    ).toBe('https://warden.example.com:8443');
  });

  it('accepts http (plaintext) — the backend coercer allows both schemes', () => {
    expect(
      _resolvePublicBaseUrl('http://warden.lan/', 'http://localhost:8000'),
    ).toBe('http://warden.lan');
  });
});

describe('_resolvePublicBaseUrl — fallback branch', () => {
  it('falls back to origin (slash-stripped) when public_url is empty', () => {
    expect(_resolvePublicBaseUrl('', 'http://localhost:8000/')).toBe(
      'http://localhost:8000',
    );
  });

  it('falls back to origin when public_url is null', () => {
    expect(_resolvePublicBaseUrl(null, 'http://localhost:8000')).toBe(
      'http://localhost:8000',
    );
  });

  it('falls back to origin when public_url is undefined', () => {
    expect(_resolvePublicBaseUrl(undefined, 'http://localhost:8000')).toBe(
      'http://localhost:8000',
    );
  });

  it('falls back to origin when public_url is whitespace-only', () => {
    expect(_resolvePublicBaseUrl('   ', 'http://localhost:8000')).toBe(
      'http://localhost:8000',
    );
  });

  it('strips a trailing slash from the origin too', () => {
    expect(_resolvePublicBaseUrl(null, 'http://localhost:8000/')).toBe(
      'http://localhost:8000',
    );
  });
});

describe('_resolvePublicBaseUrl — defence in depth', () => {
  it('rejects ftp:// and falls back to origin (mismatched scheme)', () => {
    expect(
      _resolvePublicBaseUrl('ftp://warden.example.com', 'http://localhost:8000'),
    ).toBe('http://localhost:8000');
  });

  it('rejects javascript: and falls back to origin', () => {
    expect(
      _resolvePublicBaseUrl('javascript:alert(1)', 'http://localhost:8000'),
    ).toBe('http://localhost:8000');
  });

  it('rejects unparseable garbage and falls back to origin', () => {
    expect(
      _resolvePublicBaseUrl('not a url at all', 'http://localhost:8000'),
    ).toBe('http://localhost:8000');
  });
});
