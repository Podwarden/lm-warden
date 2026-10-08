import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/e2e',
  // forest.spec.ts brings its own stack: run it with playwright.forest.config.ts
  testIgnore: /forest\.spec\.ts$/,
  use: { baseURL: 'http://localhost:3000' },
  webServer: { command: 'echo "expects docker compose up running"', port: 3000, reuseExistingServer: true },
});
