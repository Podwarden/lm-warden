// Every constant `app/chat2/limits.py` defines falls into exactly one of
// three buckets, and this file pins all three:
//   1. Mirrored numeric/collection constants — hand-maintained a second time
//      in `LIMITS` from `@podwarden/chat-ui` because the backend exposes no
//      schema for these. This is the only thing standing between a package
//      upgrade and a silent disagreement with THIS Warden's backend about how
//      big an image may be or how long a signed URL lives.
//   2. Backend-only constants — deliberately NOT mirrored (decompression
//      guards, GC/TTL housekeeping) because the frontend never needs them.
//      The classification test below fails if a new backend constant shows
//      up in neither this list nor the mirrored list, so an unclassified
//      addition can't slip through silently.
//   3. UI-only constants — `LIMITS` entries with no backend counterpart at
//      all (`limits.py` never defines them); pinned to their spec values so a
//      package upgrade can't drift these either.
// If a mirrored assertion ever fails, `LIMITS` and `app/chat2/limits.py` have
// drifted; fix whichever is wrong rather than relaxing the assertion.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { LIMITS as L } from '@podwarden/chat-ui';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIMITS_PY = path.resolve(here, '../../../../app/chat2/limits.py');

describe('limits mirror', () => {
  const py = readFileSync(LIMITS_PY, 'utf8');

  // Python int literals in limits.py use only syntax that is also valid JS
  // (`10 * 1024**2`, `40_000_000`, `6 * 3600`). Evaluate the RHS directly;
  // if a constant ever gains Python-only syntax, adjust this parser rather
  // than the mirrored value.
  const num = (name: string): number => {
    const m = new RegExp(`^${name}\\s*=\\s*(.+?)\\s*(?:#.*)?$`, 'm').exec(py);
    if (!m) throw new Error(`${name} not found in app/chat2/limits.py`);
    return Function(`return (${m[1]})`)() as number;
  };

  it('matches app/chat2/limits.py', () => {
    expect(L.MAX_IMAGE_BYTES).toBe(num('MAX_IMAGE_BYTES'));
    expect(L.MAX_IMAGES_PER_MESSAGE).toBe(num('MAX_IMAGES_PER_MESSAGE'));
    expect(L.RESERVE_CAP_TOKENS).toBe(num('RESERVE_CAP_TOKENS'));
    expect(L.SIGNED_URL_TTL_S).toBe(num('SIGNED_URL_TTL_S'));
    expect(L.TITLE_MAX_CHARS).toBe(num('TITLE_MAX_CHARS'));
  });

  it('mirrors the allowed image MIME set', () => {
    expect(py).toContain('"image/png"');
    expect(py).toContain('"image/jpeg"');
    expect(py).toContain('"image/webp"');
    expect([...L.ALLOWED_IMAGE_MIMES]).toEqual(['image/png', 'image/jpeg', 'image/webp']);
  });

  it('keeps the UI-only constants at their spec values', () => {
    expect(L.DATA_URL_MAX_BYTES).toBe(2 * 1024 ** 2);
    expect(L.TOOL_BLOCK_CHAR_CAP).toBe(10_000);
    expect(L.CODE_FOLD_PX).toBe(400);
    expect(L.CONTEXT_AMBER).toBe(0.85);
    expect(L.TICKET_REFRESH_MS).toBe(8 * 60_000);
  });

  // Every constant mirrored above (numeric or not) accounts for one bucket;
  // everything else defined in limits.py must be an explicitly-classified
  // backend-only constant. If a new constant is added to limits.py without
  // updating this list, this test fails — that is the point.
  it('classifies every remaining app/chat2/limits.py constant as backend-only', () => {
    const mirrored = [
      'MAX_IMAGE_BYTES',
      'MAX_IMAGES_PER_MESSAGE',
      'RESERVE_CAP_TOKENS',
      'SIGNED_URL_TTL_S',
      'TITLE_MAX_CHARS',
      'ALLOWED_IMAGE_MIMES',
    ];
    const backendOnly = [
      'MAX_IMAGE_PIXELS',
      'MULTIPART_OVERHEAD_BYTES',
      'DRAFT_TTL_DAYS',
      'ATTACHMENT_TTL_DAYS',
      'GC_INTERVAL_S',
    ];
    const allNames = [...py.matchAll(/^([A-Z][A-Z0-9_]*)\s*(?::[^=\n]+)?=/gm)].map((m) => m[1]);
    const actualBackendOnly = allNames.filter((n) => !mirrored.includes(n));
    expect(new Set(backendOnly)).toEqual(new Set(actualBackendOnly));
  });
});
