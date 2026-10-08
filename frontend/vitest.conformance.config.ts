import { defineConfig } from 'vitest/config';

// The conformance kit drives a REAL server over HTTP, so it wants node's
// fetch (jsdom's would apply CORS and drop the Cookie header) and none of
// the jsdom setup file. `server.deps.inline` is carried over from
// vitest.config.ts for the same reason it exists there: `@podwarden/chat-ui`
// has a CSS side-effect import that Node's ESM loader refuses.
export default defineConfig({
  test: {
    environment: 'node',
    globals: false,
    include: ['tests/conformance/**/*.test.ts'],
    testTimeout: 15000,
    server: { deps: { inline: ['@podwarden/chat-ui'] } },
  },
});
