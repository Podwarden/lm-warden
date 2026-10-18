# LM Warden E2E Tests

End-to-end tests using Playwright. **Not run in CI** — exercised manually against a live `docker compose` stack to validate the operator's happy-path flow before each release.

## Running

```bash
# 1. Bring the stack up
cd /path/to/lm-warden
docker compose up -d

# 2. Wait for the UI to be reachable
curl -fsS http://localhost:3000/api/health  # should return {"ok": true}

# 3. Run Playwright (inside the frontend dir, in Docker)
cd frontend
docker run --rm --network host -v "$PWD:/work" -w /work \
  mcr.microsoft.com/playwright:v1.49.0-jammy \
  npx playwright test
```

## Required env

- `E2E_ADMIN_PW` — the admin password set during `/setup`. Defaults to `change-me` in the spec.

## Test scope

The single happy-path spec exercises: login → add tiny model (facebook/opt-125m) → pull → load → mint API token → /v1/completions call → rotate token → unload → delete model. Anything beyond this is unit/component test territory.

`chat-playground.spec.ts` (`/chat`, issue #117) and `chat2.spec.ts` (`/chat2`,
issue #232) share their login + model-bootstrap helpers (`loginViaUi`,
`ensureOptLoaded`) from `helpers.ts`.

`chat2.spec.ts` covers: create chat → pick model → send → stream → regenerate
→ fork-here → delete; and image attach → signed-URL refresh on reload →
context-window lock (via an oversized `max_tokens`) → fork out of the lock →
two browser tabs racing the same chat (`turn_in_flight`). It has **not been
executed** — this worktree has no compose stack with a model loaded to run
it against; it was written and verified only with `npx playwright test
--list` (syntax/selector sanity), against the real component markup. Run it
once against a live stack before this lands, per the workflow above. The
`budget_blocked` banner (design doc §7) is intentionally not covered here —
the warden's `BudgetPolicy` is `AlwaysAllow`, so there is no real budget to
trip; it is exercised with a stubbed budget by the `Banners` component test
instead.
