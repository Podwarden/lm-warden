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

## Session forest (`forest.spec.ts`)

Unlike the specs above, the forest e2e brings up its own throwaway stack — no
docker, GPU or model: `fixtures/forest-stack.ts` (global setup of
`playwright.forest.config.ts`) migrates a temp SQLite DB, seeds an admin and
two inference keys plus `request_history` rows (`fixtures/forest-seed.ts`),
starts uvicorn, `next build` + the standalone server, and a one-origin proxy
(`/ui/*` → next, the rest → FastAPI), and tears it all down afterwards.

```bash
cd frontend
npx playwright test -c playwright.forest.config.ts                  # headless project + headed "gpu" motion project
FOREST_E2E_CHANNEL=chrome npx playwright test -c playwright.forest.config.ts   # headless in the system Chrome
npx playwright test -c playwright.forest.config.ts --project=gpu    # motion only, headed system Chrome
```

Knobs: `VW_VENV` (Python venv; default `../.venv` or the main checkout's),
`FOREST_E2E_SKIP_BUILD=1` (reuse `.next`), `FOREST_E2E_NEXT=dev`,
`FOREST_E2E_KEEP=1` (keep the temp data dir and logs), `FOREST_E2E_DPR=2`
(device pixel ratio for the motion run), `FOREST_E2E_PROFILE=1` (CPU profile of
the motion sample). The frame-time bound is `test.fixme` wherever WebGL is a
software renderer (SwiftShader, llvmpipe); the other motion bounds hold anywhere.

The frame-time check (p99 ≤ 25 ms) is a **manual release check**: it is skipped
unless `FOREST_PERF=1`, and must be run on an unloaded target machine (RTX A4000
class). Every motion run still records p99, the camera/girth steps, the pixel
ratio and the load average in a `motion` test annotation.

The `stats card` cases cover the forest card on `/ui/stats` (1440×900, so the
card starts below the fold):

- **No three.js before the scroll.** Every script response from the login page
  on is recorded and its body checked for `WebGLRenderer`: none before the card
  scrolls into view, at least one after.
- **In view, then offscreen.** After the scroll a canvas renders and
  `/api/stats/forest?range=6h` is polled; after scrolling away, 5 s pass with no
  forest poll and the debug hooks report `stats().paused === true`.
  While still paused, `__forest.snapshot()` must return a PNG data URL over
  5 KB, and the card must stay paused.
- **Hand-off.** "Open full window" lands on `/ui/forest?debug=1` (the card keeps
  `?debug=1`). The hand-off is front-relative (x − X(shown time)): the card's
  6 h layout and the full window's 48 h one put the forest at different world x.
  When the full window's first data lands, its front-relative view
  (`__forest.exportView()`) must be within 0.5 of the card's at the click (read
  from the card's hooks in a capture-phase click listener). The camera may move
  at most 0.1 per frame over the next 30 frames, and 10 s later it must be within
  2 units of its follow goal (`__forest.followGoal()`). The `handoff` annotation
  records the distances, the steps, the frame times and the settle time.
- **Phone** (390×844, touch, mobile): the card shows `<img alt="Session forest">`,
  a PNG data URL over 5 KB; no canvas and no scene remain, and polling stops.
- **Nav.** The admin menu's "Forest" item links to `/ui/forest`.
