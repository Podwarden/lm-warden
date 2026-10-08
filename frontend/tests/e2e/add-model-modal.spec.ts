// #86 add-model modal happy-path (Playwright, route-mocked).
//
// The existing `happy-path.spec.ts` drives the modal against the real
// backend; this spec instead pins the modal's 4-state flow with mocked
// routes so the modal contract is exercised even when the API isn't
// reachable (CI environments without an attached vLLM cluster). The two
// specs complement each other — keep both green.

import { test, expect, type Route } from "@playwright/test";

const GIB = 1024 * 1024 * 1024;

const DISCOVERY_FIXTURE = {
  files: [
    {
      filename: "model.safetensors",
      size: 8 * GIB,
      kind: "safetensors_single",
      quant: "fp16",
      params: 7_000_000_000,
    },
    {
      filename: "config.json",
      size: 1024,
      kind: "config",
      quant: null,
      params: null,
    },
  ],
  config: { architectures: ["LlamaForCausalLM"] },
  repo: { id: "meta-llama/Llama-3-8B" },
  errors: [],
};

const FIT_FIXTURE = {
  verdict: "green",
  breakdown: {
    total_vram: 24 * GIB,
    weights_budget: 20 * GIB,
    kv_reserve: 2 * GIB,
    file_size: 8 * GIB,
    ratio: 0.4,
    dtype_bytes: 2,
    max_model_len_used: 4096,
  },
  recommended_max_model_len: null,
  warnings: [],
};

const GPUS_FIXTURE = [
  {
    index: 0,
    name: "RTX 4090",
    memory_total_mib: 24576,
    memory_used_mib: 0,
    utilization_pct: 0,
  },
];

function jsonRoute(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

test.describe("Add Model modal — #86 4-state machine", () => {
  test.beforeEach(async ({ page }) => {
    // Stub auth so we don't bounce to /login. The modal's authFetch wrapper
    // expects an access token in module state; we seed it via the
    // /api/auth/login redirect dance Playwright would normally drive. For a
    // pure UI test, just fake the cookies + token in-page.
    await page.addInitScript(() => {
      // localStorage isn't where auth-fetch.ts reads from, but having the
      // module's setAccessToken/setCsrfToken called via the live login flow
      // would require running the real backend. Instead we route the auth
      // endpoints below — the modal driver will treat itself as logged in
      // once /api/csrf returns 200.
    });

    // Route mocks. Order matters — match the more specific first.
    await page.route("**/api/csrf", (route) => jsonRoute(route, { csrf: "test-csrf" }));
    await page.route("**/api/auth/me", (route) =>
      jsonRoute(route, { username: "admin", role: "admin" }),
    );
    await page.route("**/api/auth/refresh", (route) =>
      jsonRoute(route, { access_token: "test-token" }),
    );
    await page.route("**/api/system/gpus", (route) =>
      jsonRoute(route, {
        probed_at: "2026-05-19T12:00:00Z",
        probe_error: null,
        gpus: GPUS_FIXTURE,
        // null = no allowlist recorded = every GPU selectable. Not [].
        allowed_indices: null,
      }),
    );
    // The dialog's engine-version control follows the running driver
    // (supports_version_select). Mirrors `bonus` as deployed: the subprocess
    // driver cannot swap the engine image, so the wizard hides the selector
    // and says the engine version is fixed by this build.
    await page.route("**/api/system/backends", (route) =>
      jsonRoute(route, {
        default: "vllm",
        driver: "subprocess",
        engine_version: "0.26.0",
        backends: [
          {
            name: "llamacpp",
            display_name: "llama.cpp",
            version: "b10731",
            supports_version_pin: true,
            version_pin_available: false,
            version_pin_reason:
              "This deployment runs the in-container engine driver: the engine version is fixed by the warden image, so a pin is refused rather than applied. Version selection requires the docker engine driver (VW_ENGINE_DRIVER=docker).",
          },
          {
            name: "vllm",
            display_name: "vLLM",
            version: "0.26.0",
            supports_version_pin: true,
            version_pin_available: false,
            version_pin_reason:
              "This deployment runs the in-container engine driver: the engine version is fixed by the warden image, so a pin is refused rather than applied. Version selection requires the docker engine driver (VW_ENGINE_DRIVER=docker).",
          },
        ],
      }),
    );
    await page.route("**/api/models/discover**", (route) =>
      jsonRoute(route, DISCOVERY_FIXTURE),
    );
    await page.route("**/api/models/fit-preview", (route) =>
      jsonRoute(route, FIT_FIXTURE),
    );
    await page.route("**/api/models", (route) => {
      if (route.request().method() === "POST") {
        return jsonRoute(
          route,
          { id: "new-model", served_model_name: "llama-3-8b" },
          201,
        );
      }
      return jsonRoute(route, { models: [] });
    });
    await page.route("**/api/models/*/pull", (route) =>
      route.fulfill({ status: 202, body: "" }),
    );
  });

  test("discover → select-file → submit lands on 201", async ({ page }) => {
    await page.goto("/models");
    // The page button might be blocked by the login redirect when the
    // unmocked /api/auth/me hasn't resolved yet — wait for the Add Model
    // trigger explicitly before clicking it.
    const addBtn = page.getByRole("button", { name: /add model/i });
    await expect(addBtn).toBeVisible({ timeout: 10_000 });
    await addBtn.click();

    // Stage 1: enter-repo. Fill HF repo and click Discover.
    await page.fill('input[name=hf_repo]', "meta-llama/Llama-3-8B");
    await page.locator('[role=dialog]').getByRole("button", { name: /discover/i }).click();

    // Stage 3: select-file. The file table is now visible.
    await expect(page.getByTestId("file-table")).toBeVisible({ timeout: 5_000 });
    // The fit badge appears once /api/models/fit-preview returns.
    const badge = page.getByTestId("fit-badge-model.safetensors");
    await expect(badge).toBeVisible({ timeout: 5_000 });
    await expect(badge).toHaveAttribute("data-verdict", "green");

    // Click Add to submit.
    const postPromise = page.waitForResponse(
      (r) =>
        r.url().endsWith("/api/models") &&
        r.request().method() === "POST" &&
        r.status() === 201,
      { timeout: 10_000 },
    );
    await page.locator('[role=dialog]').getByRole("button", { name: /^Add$/ }).click();
    await postPromise;

    // Modal closes on success.
    await expect(page.locator("[role=dialog]")).toBeHidden({ timeout: 5_000 });
  });
});
