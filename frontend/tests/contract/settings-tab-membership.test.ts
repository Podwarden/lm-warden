import { describe, it, expect } from 'vitest';
import {
  RUNTIME_HINTS,
  RUNTIME_GENERAL_KEYS,
  RUNTIME_NETWORKING_KEYS,
  RUNTIME_ANALYTICS_KEYS,
  RUNTIME_SESSIONS_KEYS,
  RUNTIME_MAINTENANCE_KEYS,
} from '@/lib/settings-hints';

// ---------------------------------------------------------------------------
// Settings tab membership — the four per-tab arrays must together
// partition Object.keys(RUNTIME_HINTS): exactly once, no orphans, no
// duplicates. Adding a new RUNTIME_HINTS entry without placing it in
// exactly one of these arrays fails CI here — that's the point.
//
// Why a contract test, not a runtime assertion: the per-tab arrays drive
// rendering, but a missing key would silently disappear from the UI
// (rendered nowhere) rather than crash. Without this guard the only signal
// would be a manual stage-click — easy to miss. The membership test is
// the cheap automated proxy.
//
// Also pins `public_url` restart='none' — a regression that flipped it to
// `model-reload` or `warden-restart` would mislead the operator into
// expecting a reload/restart for what is purely a cosmetic snippet input.
// ---------------------------------------------------------------------------

describe('settings tab membership — RUNTIME_*_KEYS partition RUNTIME_HINTS', () => {
  // Widen the union element type to `string` for set-based comparison; the
  // per-tab arrays are `readonly RuntimeKey[]` and would not accept a
  // plain `string` predicate without this widening cast.
  const allTabKeys: readonly string[] = [
    ...RUNTIME_GENERAL_KEYS,
    ...RUNTIME_NETWORKING_KEYS,
    ...RUNTIME_ANALYTICS_KEYS,
    ...RUNTIME_SESSIONS_KEYS,
    ...RUNTIME_MAINTENANCE_KEYS,
  ];

  it('every RUNTIME_HINTS key appears in exactly one tab array', () => {
    for (const key of Object.keys(RUNTIME_HINTS)) {
      const occurrences = allTabKeys.filter((k) => k === key).length;
      expect(occurrences, `key "${key}" appears ${occurrences} times across tabs`).toBe(1);
    }
  });

  it('union of all tab arrays equals the RUNTIME_HINTS key set', () => {
    const hintsKeys = new Set(Object.keys(RUNTIME_HINTS));
    const tabsKeys = new Set(allTabKeys);

    // Symmetric difference is empty when the two sets are equal.
    const onlyInHints = [...hintsKeys].filter((k) => !tabsKeys.has(k));
    const onlyInTabs = [...tabsKeys].filter((k) => !hintsKeys.has(k));

    expect(
      onlyInHints,
      `keys defined in RUNTIME_HINTS but not placed in any tab: ${JSON.stringify(onlyInHints)}`,
    ).toEqual([]);
    expect(
      onlyInTabs,
      `keys placed in a tab but missing from RUNTIME_HINTS: ${JSON.stringify(onlyInTabs)}`,
    ).toEqual([]);
  });

  it('no key appears in more than one tab array (defensive duplicate check)', () => {
    const seen = new Set<string>();
    const dupes: string[] = [];
    for (const k of allTabKeys) {
      if (seen.has(k)) dupes.push(k);
      seen.add(k);
    }
    expect(dupes, `duplicated keys: ${JSON.stringify(dupes)}`).toEqual([]);
  });
});

describe('settings tab membership — restart-kind contracts', () => {
  it('public_url restart kind is "none" — cosmetic snippet input, no reload required', () => {
    expect(RUNTIME_HINTS.public_url).toBeDefined();
    expect(RUNTIME_HINTS.public_url.restart).toBe('none');
  });

  it('landing_page_enabled restart kind is "none"', () => {
    expect(RUNTIME_HINTS.landing_page_enabled.restart).toBe('none');
  });

  it('session_access_ttl_minutes restart kind is "warden-restart"', () => {
    expect(RUNTIME_HINTS.session_access_ttl_minutes.restart).toBe('warden-restart');
  });

  it('hf_token restart kind is "model-reload"', () => {
    expect(RUNTIME_HINTS.hf_token.restart).toBe('model-reload');
  });

  it('vllm_version restart kind is "warden-restart"', () => {
    expect(RUNTIME_HINTS.vllm_version.restart).toBe('warden-restart');
  });
});
