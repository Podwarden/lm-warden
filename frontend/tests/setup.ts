// Vitest setup — extends Vitest's `expect` with @testing-library/jest-dom
// matchers (`toBeInTheDocument`, `toHaveAttribute`, etc.). Importing the
// dedicated `/vitest` entry point auto-extends the matcher registry, so a
// bare `import` (no symbols) is sufficient.
import '@testing-library/jest-dom/vitest';
import React from 'react';
import { JSDOM } from 'jsdom';
import { afterEach, beforeEach, vi } from 'vitest';
import { __resetLoginRedirectInFlightForTests } from '@/lib/auth-fetch';

// v2026.08.23 (chat2 Task 12) — Node 24+ ships its own global `localStorage`
// (Web Storage API), on by default with no backing file, so every method
// throws `<method> is not a function`. Under vitest's jsdom environment
// `window` IS `globalThis`, so that broken built-in shadows jsdom's own
// `window.localStorage` too — hence 16 pre-existing failures across
// `stats-page.test.tsx` / `use-persisted-range.test.tsx` (documented in
// chat2 Task 5's report as a known local Node-version issue, not touched
// there because other task agents were live in this file). Repoint the
// global at a real, working `Storage` from a throwaway jsdom instance.
//
// Assigned directly via `Object.defineProperty`, NOT `vi.stubGlobal`: a
// dozen existing test files call `vi.unstubAllGlobals()` in their own
// afterEach (to undo their own `fetch`/`ResizeObserver` stubs), and that
// call reverts every stub ever registered process-wide, not just the
// caller's own — it would silently undo this fix after their first test.
Object.defineProperty(globalThis, 'localStorage', {
  value: new JSDOM('', { url: 'http://localhost/' }).window.localStorage,
  configurable: true,
  writable: true,
});

// v17.11 Bundle 2 — `react-virtuoso` uses ResizeObserver + getBoundingClientRect
// for window measurement, neither of which work in jsdom. The library
// silently renders zero items in this environment, breaking every test
// that asserts on a row rendered through Virtuoso.
//
// We replace Virtuoso with a plain "render all items in order" shim that
// passes through the props tests actually care about (style, data,
// itemContent, computeItemKey, components.List).
//
// Still load-bearing after /chat2 moved to @podwarden/chat-ui (v2026.08.25):
// `models/log-stream.tsx` and `godmode/godmode-viewer.tsx` both render their
// rows through <Virtuoso>, and their suites assert on row TEXT
// (`log-stream.test.tsx` → "hello world", "first output"). Without this shim
// those rows never mount. `godmode-viewer.test.tsx` installs its own richer
// local mock, which overrides this one for that file only.
//
// The chat2-era extras are gone with the feature that needed them: the
// `__virtuosoTestHandle` escape hatch (published `atBottomStateChange` /
// `followOutput` / a spy-able `scrollToIndex` so <Thread>'s "Jump to latest"
// debounce could be driven from a test) and the `components.Footer` /
// `EmptyPlaceholder` overrides. Nothing outside chat2 used either — the two
// remaining callers pass `components={{ List }}` and nothing else.
vi.mock('react-virtuoso', () => {
  type Row = { kind?: string; row?: unknown; event?: unknown } | unknown;
  function Virtuoso(props: {
    data?: ReadonlyArray<Row>;
    itemContent?: (index: number, item: Row) => React.ReactNode;
    computeItemKey?: (index: number, item: Row) => React.Key;
    style?: React.CSSProperties;
    ref?: React.Ref<{ scrollToIndex: (o: unknown) => void }>;
    components?: {
      List?: React.ComponentType<React.HTMLAttributes<HTMLDivElement>>;
    };
  }) {
    const { data = [], itemContent, computeItemKey, style, components } = props;
    // React 19 passes `ref` as a plain prop to function components. The
    // callers use it only for `scrollToIndex`, which has nothing to scroll
    // here — but the handle must exist so `ref.current?.scrollToIndex(...)`
    // is a real call rather than an optional-chain no-op.
    React.useImperativeHandle(props.ref, () => ({ scrollToIndex: () => {} }), []);
    const rows = data.map((item, idx) => {
      const key = computeItemKey ? computeItemKey(idx, item) : idx;
      return React.createElement(
        'div',
        { key, 'data-virtuoso-row-index': idx },
        itemContent ? itemContent(idx, item) : null,
      );
    });
    const ListComp = components?.List;
    if (ListComp) {
      return React.createElement('div', { style }, React.createElement(ListComp, {}, rows));
    }
    return React.createElement('div', { style }, rows);
  }
  return { Virtuoso };
});

// v2026.05.15.3 — `auth-fetch.ts` carries a module-level
// `loginRedirectInFlight` flag that de-dupes parallel 401 redirects in a
// real page. Module state persists across test files in vitest's worker,
// so a test that exhausts a 401 refresh leaves the flag stuck `true`
// and subsequent tests (e.g. sse-terminal-error checking 404/502/429)
// see the SSE preflight short-circuit to 401. Reset between every test
// so test isolation is preserved. Production code must never call this
// — the flag's job is to NOT reset until a real navigation reloads the
// module.
beforeEach(() => {
  __resetLoginRedirectInFlightForTests();
});

// R19 — several suites replace `window.location` with a plain object
// (`Object.defineProperty(window, 'location', ...)`) because jsdom 25 marks
// `Location.replace` non-configurable, so a bare `vi.spyOn` on it throws.
// The replacement used to leak: jsdom's `window.location` is an
// accessor property the test overwrites with a data property, and nothing
// ever restored it, so every later test in the file ran against the
// previous test's fake location (no `pathname`, no `origin`), and the first
// real `redirectToLogin()` in the file hit the real jsdom location and
// emitted "Not implemented: navigation" into whichever test was running.
// Capture this file's original location and put it back after every test.
// (`vi.unstubGlobals` cannot do this: the tests define the property
// directly, not through `vi.stubGlobal`.)
const originalLocation = window.location;
afterEach(() => {
  if (window.location !== originalLocation) {
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      writable: true,
      configurable: true,
    });
  }
});
