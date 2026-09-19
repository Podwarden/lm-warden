# tests/live -- live priority + queueing proof

Talks to a REAL vllm-warden deployment over HTTP: it logs in as an admin,
creates 11 short-lived tokens, **pauses every other live token on the
deployment for the duration of the run**, drives 5-10 minutes of saturating
load through `/v1/chat/completions`, and asserts strict-priority admission +
FIFO-within-tier queueing from the server's own `request_history` rows.

It is opt-in only: the `live` pytest marker, gated by `-m "not live"` in
`pyproject.toml`'s `addopts`. A bare `pytest`, `pytest tests/unit`, or CI's
`pytest tests/unit` / `pytest tests/conformance` never collects it.

## Env contract

Nothing is hardcoded -- everything comes from `VW_LIVE_*` env vars, sourced
from a private env file OUTSIDE the repo. Missing any required var (or
`VW_LIVE_ACK_PAUSE_ALL` not exactly `1`) skips the whole directory with the
list of what's missing.

Required:

| Var | Meaning |
|---|---|
| `VW_LIVE_BASE_URL` | e.g. `https://host:port`, no trailing slash |
| `VW_LIVE_ADMIN_USER` / `VW_LIVE_ADMIN_PASSWORD` | admin login; never logged |
| `VW_LIVE_MODEL` | served model name as used in `/v1/chat/completions` |
| `VW_LIVE_ACK_PAUSE_ALL=1` | safety interlock -- the run pauses EVERY other live token |

Optional (defaults in parentheses) -- see `tests/live/_env.py` for the
complete, authoritative list: `VW_LIVE_MAX_INFLIGHT` (16),
`VW_LIVE_PRIORITIES` (`0,0,2,3,4,4,6,7,9,9`), `VW_LIVE_FILLER_PRIORITY` (1),
`VW_LIVE_FILLER_PROMPT_TOKENS` (3000), `VW_LIVE_PROBE_PROMPT_TOKENS` (2000),
`VW_LIVE_HOLD_S` / `VW_LIVE_STEP_S` (8 / 0.75), `VW_LIVE_FILLER_MAX_TOKENS`
(512), `VW_LIVE_PROBE_MAX_TOKENS` (64), `VW_LIVE_BURST_GAP_MS` (150),
`VW_LIVE_MIN_ROUNDS` / `VW_LIVE_MIN_LOAD_S` / `VW_LIVE_MAX_LOAD_S` (4 / 300 /
540), `VW_LIVE_MIN_CLEAN_ROUNDS` (3), `VW_LIVE_TIMEOUT_S` (900),
`VW_LIVE_DRAIN_TIMEOUT_S` (180), `VW_LIVE_SATURATION_TIMEOUT_S` (60),
`VW_LIVE_ROUND_TIMEOUT_S` (180), `VW_LIVE_MIN_QUEUE_SPREAD_S` (1.0),
`VW_LIVE_ASSERT_TTFT` (0), `VW_LIVE_STATE_DIR` / `VW_LIVE_REPORT_DIR`
(`~/.local/state/vw-live` -- deliberately NOT a tempdir: it must survive a
reboot between a crash and an operator running `restore.py`),
`VW_LIVE_TOKEN_PREFIX` (`vwlive`),
`VW_LIVE_SWEEP_LEFTOVERS` (0), `VW_LIVE_VERIFY_TLS` / `VW_LIVE_CA_BUNDLE` (1
/ unset), `VW_LIVE_DRY_RUN` (0 -- see "Rehearsal" below).

## Running it

```
set -a; . /path/outside/repo/vw-live.env; set +a
pytest -m live -s tests/live/test_priority_queueing.py
```

A run takes 5-15 minutes. Run it detached (`nohup ... &`) if your shell
might disconnect -- see the operator report for the exact production
invocation.

## Isolation and crash recovery

Before any load, the test pauses every OTHER live token on the deployment
(never one that's already paused, expired, or revoked), records exactly
which ones it paused in a crash-safe state file
(`${VW_LIVE_STATE_DIR}/<run id>.json`, written before the first PATCH and
after every change), and restores them all in teardown -- restore always
runs BEFORE our own test tokens are deleted.

A plain `kill <pid>` sends SIGTERM, which this suite maps to
KeyboardInterrupt (conftest.py, module import time) so it triggers the same
teardown a Ctrl-C would. A `run_state` guard fixture, whose own setup does
no network I/O and no `await` before it hands the state object to every
other fixture, is what makes that teardown reliable even when the
interrupt lands *inside* another fixture's own suspended `await` (pytest
still finalizes every fixture that already reached its `yield`, which
`run_state` always has by the time isolation could possibly be
interrupted). No OTHER fixture in the chain has a finalizer of its own
(`isolation` used to; that step moved into `run_state`'s, after signals
are ignored) -- a second SIGINT/SIGTERM cannot land in an unprotected
window anywhere in teardown. When restore starts, it prints "restoring
paused tokens -- do not kill" and ignores further signals for its
duration (a few seconds; restore itself is capped at ~10 minutes, after
which it prints the `restore.py` command and exits non-zero). Rehearsed:
SIGTERM mid-drain, SIGTERM mid-load, a second SIGTERM during what used to
be an unprotected teardown window, and SIGKILL (which nothing in-process
can catch) all restore correctly -- SIGKILL via the standalone
`restore.py` afterward.

If the test process is killed (SIGKILL, power loss -- anything that defeats
even that), the state file survives and tells you what's still paused:

```
set -a; . /path/outside/repo/vw-live.env; set +a
python -m tests.live.restore --list
python -m tests.live.restore --state <path from --list>
```

Restore only unpauses a token whose `paused_at` still equals what our own
PATCH returned -- if an operator touched it mid-run, restore leaves it alone
and reports it (`--force` skips that check). A token an operator already
UNPAUSED by hand is treated as already restored, not a failure. A `pending`
entry (our PATCH response was lost, but the pause may have landed
server-side) is reconciled via a fresh GET: ours only if the token's
current `paused_at` is at/after this run's start. `--sweep` also deletes
leftover test tokens, matched by the exact shape
`<prefix>-<8 hex run id>-<role>` (never a bare prefix match).

**A narrow race can still make the tool touch a key it shouldn't.** If an
operator pauses a key in the few milliseconds between our pre-PATCH
ownership check and our PATCH landing, or if that check itself fails
(network blip) and we fall through to the PATCH anyway, that key can end
up recorded as ours and unpaused by our restore. To spot this: the state
file (`${VW_LIVE_STATE_DIR}/<run id>.json`) lists every key the run
touched, by id, under `paused` -- check it against what you expect to have
been live before the run.

## Round verdicts: hard failure vs. excluded

A round can be excluded from the A-D assertions two different ways, and
only one of them is silent:

- **Foreign traffic** (another operator's request landed on the target
  engine during our round) is excluded and reported -- not our bug, not a
  failure.
- **`setup_failure`** (could not saturate, one of OUR OWN probes jumped the
  burst, the round timed out, or server rows never showed up) is ALWAYS a
  hard failure. It is never silently dropped. Two consecutive
  `setup_failure` rounds abort the load loop early rather than keep
  everyone paused for the full `VW_LIVE_MAX_LOAD_S` chasing a result that
  will not change.

## Rehearsal (required before a production run)

Two layers, both required before running against production:

**1. `VW_LIVE_DRY_RUN=1`** -- isolation + token lifecycle only, no traffic.
With it set, the test logs in, fetches CSRF, creates the 11 test tokens,
pauses every other live token and drains their in-flight requests, then
prints the restore command and exits -- before calibration, before any
load. A local dev stack with no inference engine is enough for this.

**2. The LOAD path** -- saturate/burst/drain against the REAL
`PriorityScheduler`, which needs something OpenAI-compatible on the other
end. `tests/fakes/fake_vllm.py` doesn't fit here (it answers fixed probe
shapes near-instantly, no `max_tokens`/`ignore_eos` duration control, so
nothing ever queues) -- a small standalone fake engine that honors
`max_tokens` with a controllable per-token delay was used instead,
kept OUTSIDE the repo (it's rehearsal-only, not a reusable fixture) and
stitched into a real `app.main.app` instance via
`Supervisor._ports`/`_hosts` (the same in-memory routing table the real
loader populates) so every request genuinely goes through the real proxy
route, the real admission gate, and the real `request_history` store.
Verified: saturation to `VW_PROXY_MAX_INFLIGHT`, strict-priority admission
order, FIFO within a tier, and all of A1-D4 passing on a clean round.

Combine either layer with killing the test process mid-run and running
`python -m tests.live.restore --state <file>` to rehearse the crash path.
Rehearsed here: SIGTERM mid-drain (interrupt landing inside a fixture's own
`await`), SIGTERM mid-load, a simulated lost-PATCH-response (a hand-built
`pending` state entry, reconciled via `restore.py` against the real
server), and SIGKILL (recovered via `restore.py --state ... --sweep`) --
all four left the token set byte-identical to its pre-run baseline.

## What the report never contains

Only token ids and our own generated names (`<prefix>-<run id>-<role>`)
ever appear in a report or the state file -- never a plaintext key, never
the admin password or JWT, never another token's name (counts only), never
a hostname (`base_url` is redacted to `scheme://<redacted>` before it
reaches a report).
