import os from 'node:os';
import path from 'node:path';
import { defineConfig } from '@playwright/test';

// The session forest e2e (tests/e2e/forest.spec.ts) against a throwaway local stack that global setup brings up and
// tears down (tests/e2e/fixtures/forest-stack.ts): uvicorn + the UI + a one-origin proxy, seeded request_history.
// Separate from playwright.config.ts, which expects a docker compose stack on :3000.
//
//   npx playwright test -c playwright.forest.config.ts                 # both projects
//   npx playwright test -c playwright.forest.config.ts --project=gpu   # motion only, headed system Chrome
//
// Projects:
//   - headless: every test, headless. Bundled Chromium by default; FOREST_E2E_CHANNEL=chrome uses the system Chrome
//     (no browser download). Wherever WebGL is software (SwiftShader, llvmpipe) the frame-time bound is test.fixme;
//     the camera, girth and flight bounds still hold.
//   - gpu: the motion tests only (@motion), headed, in the system Chrome (channel "chrome"), so WebGL runs on this
//     machine's GPU and the frame-time bound is meaningful. Keep the window in front while it samples.
export default defineConfig({
  testDir: './tests/e2e',
  testMatch: /forest\.spec\.ts$/,
  globalSetup: './tests/e2e/fixtures/forest-stack.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: [['list']],
  outputDir: path.join(os.tmpdir(), 'forest-e2e-results'),
  use: {
    viewport: { width: 1440, height: 900 },
    // FOREST_E2E_DPR: force a device pixel ratio (diagnostics: the scene renders at min(1.5, devicePixelRatio))
    ...(process.env.FOREST_E2E_DPR ? { deviceScaleFactor: Number(process.env.FOREST_E2E_DPR) } : {}),
    trace: 'off', // tracing a failed WebGL page made context close hang for the whole test timeout
  },
  projects: [
    {
      name: 'headless',
      use: { browserName: 'chromium', headless: true, ...(process.env.FOREST_E2E_CHANNEL ? { channel: process.env.FOREST_E2E_CHANNEL } : {}) },
    },
    {
      name: 'gpu',
      grep: /@motion/,
      use: { browserName: 'chromium', channel: 'chrome', headless: false },
    },
  ],
});
