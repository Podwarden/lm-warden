import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: false,
    setupFiles: ['./tests/setup.ts'],
    // R19 — per-test isolation backstop. Test bodies install globals with
    // `vi.stubGlobal` (fetch, EventSource, ResizeObserver) and spies with
    // `vi.spyOn`; a forgotten `vi.unstubAllGlobals()` / `vi.restoreAllMocks()`
    // in one test leaked the stub into the next test, and that leakage is
    // what made `vitest:frontend` the flakiest CI job (a stale stub or a
    // real `window.location` navigation landing in whichever test happened
    // to be running when the async residue settled). Revert them after
    // EVERY test so no test can inherit another's globals or spies.
    // `tests/contract/vitest-config-isolation.test.ts` pins these switches.
    unstubGlobals: true,
    restoreMocks: true,
    // `tests/conformance/**` needs a live backend and node's fetch; it runs
    // only via `npm run test:conformance` (vitest.conformance.config.ts),
    // which `make conformance` drives after booting the server.
    exclude: ['node_modules/**', 'tests/e2e/**', 'tests/conformance/**'],
    server: {
      deps: {
        // `@podwarden/chat-ui` imports `katex/dist/katex.min.css` for math
        // rendering. Left externalized, that import is handed to Node's ESM
        // loader, which throws ERR_UNKNOWN_FILE_EXTENSION on `.css` before a
        // single test collects. Inlining routes the package through Vite,
        // which understands CSS imports (and, with `css` off by default,
        // resolves them to nothing). A `vi.mock` cannot fix this: mocks only
        // intercept Vite-transformed imports, and the offending one lives
        // inside the externalized package.
        inline: ['@podwarden/chat-ui'],
      },
    },
  },
  esbuild: { jsx: 'automatic' },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
});
