// SSR-safe localStorage-backed range selector.
//
// Used by /stats to remember the operator's chosen range (1h/6h/24h/7d)
// across reloads. The user brief locked this behaviour:
//
//   "selected range for stats charts should be stored in browser storage
//    so no need to select again after page reload"
//
// The hook is generic enough to be reused for any future page that
// wants the same persistence (e.g. a /cache filter), so the storage key
// is a required parameter rather than hard-coded.
//
// SSR contract: on the server (and during the first client render before
// hydration completes) the hook returns `fallback`. The localStorage value
// — if any — is read inside a `useEffect`, then a `setState` flushes the
// real value on the next paint. This avoids a hydration mismatch between
// the server-rendered HTML ("1h") and the client value ("24h") that
// React 19 would otherwise mark as a warning.
//
// Robustness:
//   - Wrapped in try/catch — localStorage can throw under privacy modes
//     and quota-exceeded; we'd rather silently fall back than crash the
//     page.
//   - Invalid stored values (e.g. someone edited DevTools to "lol") are
//     ignored and the fallback is returned. Once the user picks a valid
//     range we overwrite the bad value.

"use client";

import { useCallback, useEffect, useState } from "react";

/** Selector ranges shown across the stats UI. Mirrors the type in
 *  `@/lib/stats` so old + new code agrees on the valid set. */
export type StatsRange = "1h" | "6h" | "24h" | "7d";

const VALID_RANGES: ReadonlySet<StatsRange> = new Set<StatsRange>([
  "1h",
  "6h",
  "24h",
  "7d",
]);

function isStatsRange(value: unknown): value is StatsRange {
  return typeof value === "string" && VALID_RANGES.has(value as StatsRange);
}

/**
 * Read-and-persist a `StatsRange` value under `storageKey`.
 *
 * Returns `[value, setValue]` mirroring `useState`. On the first render
 * (and on the server), `value` is the `fallback`; after hydration the
 * stored value (if valid) replaces it. Calling `setValue` updates state
 * and writes to localStorage — write failures are silent.
 */
export function usePersistedRange(
  storageKey: string,
  fallback: StatsRange = "1h",
): [StatsRange, (next: StatsRange) => void] {
  const [value, setValue] = useState<StatsRange>(fallback);

  // Hydrate from localStorage on mount. We intentionally do not gate
  // this on `typeof window !== 'undefined'` outside the effect — the
  // effect itself is client-only, so the gate is moot. The try/catch
  // covers privacy modes / disabled storage.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw !== null && isStatsRange(raw)) {
        setValue(raw);
      }
    } catch {
      /* localStorage unavailable — keep the fallback */
    }
  }, [storageKey]);

  const set = useCallback(
    (next: StatsRange) => {
      setValue(next);
      try {
        window.localStorage.setItem(storageKey, next);
      } catch {
        /* persist failed — state still updates for this session */
      }
    },
    [storageKey],
  );

  return [value, set];
}
