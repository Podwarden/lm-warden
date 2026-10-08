// R19 — vitest test-isolation contract.
//
// The suite has leaked global stubs and module singletons between tests
// (a `vi.stubGlobal('fetch')` or an `Object.defineProperty(window,
// 'location', ...)` in one test visible to the next one), and those leaks
// are what made `vitest:frontend` the flakiest CI job: an unstubbed
// navigation attempt ("Not implemented: navigation") or a stale token/timer
// landed in whichever test happened to be running when the async residue
// settled. The two config-level switches below are the backstop for the
// fetch/ResizeObserver/EventSource stubs that live in test bodies; without
// them a single forgotten `vi.unstubAllGlobals()` re-opens the leak.
//
// This file pins the switches so they cannot be dropped by a future
// "tidy the config" edit.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const CONFIG = readFileSync(path.resolve(__dirname, '../../vitest.config.ts'), 'utf8');

describe('vitest.config.ts test isolation', () => {
  it('reverts every vi.stubGlobal after each test (unstubGlobals)', () => {
    expect(/unstubGlobals:\s*true/.test(CONFIG)).toBe(true);
  });

  it('restores every vi.spyOn after each test (restoreMocks)', () => {
    expect(/restoreMocks:\s*true/.test(CONFIG)).toBe(true);
  });
});
