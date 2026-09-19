"""tests/live: env contract + skip, `live` marker, fixtures.

Everything under tests/live is tagged with the `live` marker and skipped
with an explicit reason when the VW_LIVE_* env contract is incomplete, so a
bare `pytest` / `pytest tests/unit` never talks to a real deployment --
pyproject.toml's addopts carries `-m "not live"`, and this file is the other
half of that guarantee (it is what makes `-m live` actually select these
tests, and what tags them so the exclusion has something to filter on).
"""

from __future__ import annotations

import asyncio
import secrets
import signal
import sys
import time

import pytest
import pytest_asyncio

from tests.live._admin import AdminSession
from tests.live._env import LiveConfig, load_config, missing_vars
from tests.live._isolation import (
    StateFile,
    cancel_pending_tasks,
    delete_created_tokens,
    is_our_token_name,
    pause_foreign_tokens,
    restore_paused_tokens,
    select_foreign_tokens,
)


# C2: a plain `kill <pid>` sends SIGTERM, which Python does NOT map to
# KeyboardInterrupt by default (only SIGINT gets that treatment). Without
# this, `kill` (as opposed to Ctrl-C) skips every try/finally and fixture
# finalizer between here and the process exiting -- exactly the crash this
# whole module's state file + restore.py machinery exists to survive.
# Installed at import time (before any fixture runs) so it is armed for the
# entire session. Guarded because signal.signal() only works on the main
# thread -- harmless to skip under a runner that starts pytest off-thread.
#
# Minor (round 5): ONE-SHOT. `run_state`'s finalizer installs SIG_IGN too
# (see below), but only once it actually reaches that line -- there is a
# sub-millisecond gap between the FIRST signal being delivered (raising
# KeyboardInterrupt somewhere arbitrary) and Python unwinding to the point
# of running that line. A second signal in that gap would still hit
# `default_int_handler` and raise a SECOND KeyboardInterrupt before any
# protection exists. Closing it needs the disarm to happen INSIDE the
# handler itself, atomically with the first signal's delivery -- so this
# handler's first action, before it even raises, is to arm SIG_IGN for
# itself. (After the finalizer's own signal.signal(..., SIG_IGN) /
# restore-previous dance runs, "previous" will read back as SIG_IGN rather
# than this handler -- harmless: nothing meaningful happens later in the
# same process that would need SIGTERM mapped to KeyboardInterrupt again.)
def _one_shot_int_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)
    raise KeyboardInterrupt()


try:
    signal.signal(signal.SIGTERM, _one_shot_int_handler)
except ValueError:
    pass


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = missing_vars()
    skip_marker = None
    if missing:
        skip_marker = pytest.mark.skip(reason=f"tests/live: missing env: {', '.join(missing)}")
    for item in items:
        path = str(item.path).replace("\\", "/")
        if "/tests/live/" not in f"/{path}":
            continue
        item.add_marker(pytest.mark.live)
        if skip_marker is not None:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def live_cfg() -> LiveConfig:
    missing = missing_vars()
    if missing:
        pytest.skip(f"tests/live: missing env: {', '.join(missing)}")
    return load_config()


#: All async fixtures below are `scope="session"` AND explicitly
#: `loop_scope="session"`. Without the explicit loop_scope, pytest-asyncio
#: 0.25 (see the "asyncio_default_fixture_loop_scope" deprecation warning)
#: falls back to the fixture's pytest `scope`, so these already ran on a
#: session-scoped event loop -- but the TEST FUNCTION defaults to its own
#: function-scoped loop unless told otherwise. `admin`'s httpx connections
#: (and the asyncio.Event/anyio primitives httpcore binds to whatever loop
#: is running on first use) got bound to the session loop during fixture
#: setup; the first time the test body reused that same AdminSession from
#: its own (different) loop -- inside run_round's `_quiet` poll -- httpcore
#: raised "is bound to a different event loop". The matching half of this
#: fix is `@pytest.mark.asyncio(loop_scope="session")` on the test function
#: itself (test_priority_queueing.py), so fixtures and test share one loop.
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def admin(live_cfg: LiveConfig):
    session = AdminSession(
        live_cfg.base_url, live_cfg.admin_user, live_cfg.admin_password, verify=live_cfg.httpx_verify
    )
    await session.login()
    yield session
    await session.aclose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def run_state(live_cfg: LiveConfig, admin: AdminSession):
    """C3: the crash-safety guard. This fixture's SETUP does no network I/O
    and no `await` before its `yield` -- it only generates the run id and
    writes the initial (empty) state file -- so it completes, and pytest
    registers its finalizer, before `isolation` ever starts pausing anyone.

    Why this matters: a KeyboardInterrupt/SIGTERM delivered while a LATER
    fixture (`isolation`) is suspended in a long `await` (the up-to-180s
    pause+drain) is typically raised at the asyncio event loop's own
    select/poll boundary, not inside that fixture's coroutine frame -- so a
    try/except *inside* `isolation` does not reliably catch it; the task is
    simply abandoned. Pytest's fixture teardown-on-setup-failure, however,
    still calls the finalizer of every fixture that already reached its
    `yield` -- which this one always has, by design, by the time isolation
    could possibly be interrupted. Its teardown restores from THIS SAME
    StateFile object, which `pause_foreign_tokens` mutates and persists
    incrementally, so whatever got paused before the interrupt is exactly
    what gets restored.

    I3: restore runs BEFORE delete (a paused real user's token matters more
    than a leftover test token), and restore always runs even if delete
    raises for some reason.

    CR1: an interrupt does not CANCEL the task it interrupts -- it is
    raised at the event loop's own poll boundary and escapes
    `run_until_complete()` without touching the task that was suspended
    inside it. That task stays registered on this SAME session loop,
    still pending, mid-execution. Reproduced: SIGTERM during
    `select_foreign_tokens`/`pause_foreign_tokens` left the abandoned task
    pending; the next time the loop ran (right here, for this very
    finalizer) it got a chance to RESUME and raced this teardown --
    observed as "restore finds nothing to do, then the abandoned task
    pauses everyone anyway." Fixed by cancelling every OTHER task on the
    loop and waiting for that to actually land before restoring anything,
    plus ignoring a second SIGINT/SIGTERM for the duration so it can't
    abort the restore loop partway through.

    I1 (round 4): a SECOND signal delivered while `isolation`'s own
    finalizer was still running (its up-to-30s "drain our own in-flight
    requests" wait, which used to live there) escaped entirely --
    `SetupState.teardown_exact` only catches the test's own outcome, not
    an arbitrary second KeyboardInterrupt, so it propagated straight out
    and THIS finalizer never ran at all: restore skipped completely,
    pytest exiting 130. Reproduced (scratchpad/r3repro): "isolation
    teardown start" logged, a second SIGTERM during its sleep loop, and
    "RESTORE RAN" never printed. `isolation` no longer has any finalizer
    of its own -- that drain step now lives HERE, after signals are
    already ignored (see below), so there is no fixture finalizer left
    anywhere in the chain that runs before signals are ignored.
    """
    run_id = secrets.token_hex(4)
    state = StateFile.new(run_id=run_id, base_url=live_cfg.base_url, model=live_cfg.model)
    state.write(live_cfg.state_dir)
    state_path = state.path(live_cfg.state_dir)
    # I4: printed BEFORE the first pause (not after isolation completes), so
    # a kill in the first second still leaves the operator the exact command.
    print(f"[tests/live] state file: {state_path}")
    print(f"[tests/live] if this run is interrupted, restore with: python -m tests.live.restore --state {state_path}")

    yield state

    # A second signal must not interrupt restore itself -- ignore SIGINT/
    # SIGTERM for the duration of this finalizer and put the previous
    # handlers back afterward (there is no "afterward" that matters if the
    # process is about to exit anyway, but this also runs on ordinary,
    # non-crash teardown too). This MUST be the very first thing that runs
    # after `yield` -- nothing before it may `await` anything, or a signal
    # could still land in that gap (I1).
    prev_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(
        "[tests/live] restoring paused tokens -- do not kill; if you must, run: "
        f"python -m tests.live.restore --state {state_path}",
        file=sys.stderr,
    )
    restore_failures: list[str] = []
    try:
        # CR1: cancel every other task on this loop and let those
        # cancellations actually complete before touching the server. A
        # task cancelled mid-PATCH may already have sent the request even
        # though our await never saw the reply -- the same shape as I1's
        # lost-response case, and reconciled the same way (a `pending`
        # state-file entry restore_paused_tokens's outer loop re-checks
        # against server truth).
        #
        # Minor (round 5): cancel_pending_tasks(timeout=15.0) bounds its own
        # wait so a task that refuses to respond to cancellation promptly
        # doesn't silently eat the whole 10-minute budget before restore
        # even starts. CAUTION: called DIRECTLY, never wrapped in
        # `asyncio.wait_for(cancel_pending_tasks(), ...)` at the call site
        # -- that would create a new Task around it, making
        # `asyncio.current_task()` INSIDE cancel_pending_tasks() resolve to
        # that wrapper instead of this finalizer's own task, which would
        # then itself look like "some other pending task" and get
        # cancelled as collateral damage (this was tried and broke exactly
        # that way -- see cancel_pending_tasks()'s docstring).
        await cancel_pending_tasks(timeout=15.0)
        await asyncio.sleep(2.0)  # let any fire-and-forget PATCH land server-side

        try:
            # Minor: cap restore_paused_tokens at ~10 minutes. Its own
            # outer-pass/backoff loops are already bounded per-entry, but
            # with many entries the product can run long; past this cap,
            # stop waiting and tell the operator exactly what to run
            # instead. Safe to wrap in wait_for (unlike cancel_pending_tasks
            # above) -- this function never inspects the task tree, so
            # wrapping it in a new Task has no self-cancellation risk.
            restore_failures = await asyncio.wait_for(
                restore_paused_tokens(admin, state=state, state_dir=live_cfg.state_dir),
                timeout=600.0,
            )
        except TimeoutError:
            # Minor (round 5): the STATE FILE is accurate (every change is
            # written atomically as it happens) -- it is the RESTORE that
            # is incomplete, not the file describing it. Said precisely so
            # an operator trusts the file's "restored: false" entries.
            restore_failures = [
                "restore did not finish within 10 minutes -- the state file is "
                "accurate and lists exactly which tokens are still unrestored"
            ]
            print(
                f"\033[31mRestore exceeded 10 minutes. Run: "
                f"python -m tests.live.restore --state {state_path}\033[0m",
                file=sys.stderr,
            )
        for line in restore_failures:
            print(f"\033[31mRESTORE FAILED: {line}\033[0m", file=sys.stderr)
        if restore_failures:
            # Minor (round 5): --force now genuinely reopens exactly the
            # entries it can help (ones left alone only because paused_at
            # didn't match ours) -- see restore_paused_tokens. It does
            # nothing for e.g. a still-deferred pending entry or a bare
            # timeout; a plain re-run (no --force) picks those back up.
            print(
                f"\033[31mRun: python -m tests.live.restore --state {state_path}\033[0m",
                file=sys.stderr,
            )
            print(
                "\033[31m(add --force only if a line above says "
                '"paused_at changed" -- it reopens exactly that kind of entry, '
                "nothing else)\033[0m",
                file=sys.stderr,
            )

        # Best-effort: give our own in-flight requests a chance to drain
        # before deleting our tokens. Moved here from `isolation`'s old
        # finalizer (I1, round 4) -- signals are ignored by this point, so
        # a second interrupt during this wait can no longer skip restore
        # (which already ran, above) or leave this finalizer half-run.
        own_ids = {t["id"] for t in state.created_tokens}
        try:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                snap = await admin.live_requests()
                ours = [r for r in snap["requests"] if r["token_id"] in own_ids]
                if not ours:
                    break
                await asyncio.sleep(1.0)
        except Exception:  # noqa: BLE001 -- best-effort; restore above is what matters
            pass

        delete_failures = await delete_created_tokens(admin, state=state)
        for line in delete_failures:
            print(f"WARNING: could not delete test token during teardown: {line}", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

    if restore_failures:
        pytest.fail(f"{len(restore_failures)} token(s) could not be restored; see stderr above")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def model_info(live_cfg: LiveConfig, admin: AdminSession) -> dict:
    """GET /api/models, resolve VW_LIVE_MODEL, sanity-check headroom against
    max_model_len. Skipped in VW_LIVE_DRY_RUN=1 -- the local rehearsal stack
    has no inference engine, so there is nothing to resolve; the rehearsal
    only exercises isolation + token lifecycle."""
    if live_cfg.dry_run:
        return {
            "id": None,
            "served_model_name": live_cfg.model,
            "status": "dry-run",
            "backend": None,
            "max_model_len": None,
        }
    models = await admin.list_models()
    match = next((m for m in models if m.get("served_model_name") == live_cfg.model), None)
    if match is None:
        pytest.fail(f"model {live_cfg.model!r} not found in GET /api/models")
    if match.get("status") != "loaded":
        pytest.fail(f"model {live_cfg.model!r} status={match.get('status')!r}, expected 'loaded'")
    finished_probe = await admin.finished(model_row_id=match["id"], limit=1)
    if not finished_probe.get("available", True):
        pytest.fail("history store not present (GET /api/stats/live/finished available=false)")
    max_len = match.get("max_model_len") or 0
    if max_len:
        for tag, tokens in (
            ("filler", live_cfg.filler_prompt_tokens + live_cfg.filler_max_tokens),
            ("probe", live_cfg.probe_prompt_tokens + live_cfg.probe_max_tokens),
        ):
            if tokens >= max_len - 256:
                pytest.fail(
                    f"{tag} prompt+max_tokens ({tokens}) too close to max_model_len "
                    f"({max_len}); shrink VW_LIVE_*_PROMPT_TOKENS or *_MAX_TOKENS"
                )
    return match


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_tokens(live_cfg: LiveConfig, admin: AdminSession, run_state: StateFile):
    """Create the 11 test tokens (1 filler + up to 10 probes). Each creation
    is appended to `run_state.created_tokens` and written to disk
    IMMEDIATELY (not batched at the end), so a failure partway through a
    create leaves `--sweep` something to find. No finalizer of its own --
    cleanup is consolidated in `run_state`'s teardown (see its docstring for
    why that is the one place it can safely live).
    """
    run_id = run_state.run_id
    prefix = f"{live_cfg.token_prefix}-{run_id}"

    if live_cfg.sweep_leftovers:
        existing = await admin.list_tokens()
        for t in existing:
            if is_our_token_name(t["name"], live_cfg.token_prefix):
                await admin.delete_token(t["id"])

    created: dict[str, dict] = {}
    labels = ["filler", *live_cfg.probe_labels]
    priorities = [live_cfg.filler_priority, *live_cfg.priorities]
    for label, prio in zip(labels, priorities, strict=False):
        body = await admin.create_token(name=f"{prefix}-{label}", priority=prio, expires_in_days=1)
        created[label] = body
        run_state.created_tokens.append({"id": body["id"], "name": body["name"]})
        run_state.write(live_cfg.state_dir)

    return {"run_id": run_id, "prefix": prefix, "created": created}


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def isolation(
    live_cfg: LiveConfig, admin: AdminSession, test_tokens: dict, run_state: StateFile
):
    """Pause every OTHER live token (the safety interlock) and drain their
    in-flight requests on the target engine.

    I1 (round 4): this fixture has NO finalizer of its own -- not even the
    "drain our own in-flight requests" step it used to run after its
    `yield`. That step now lives in `run_state`'s finalizer, AFTER signals
    are ignored there. Reason: fixtures tear down in the reverse of their
    setup order, so a finalizer here would run BEFORE `run_state`'s (which
    is where SIGINT/SIGTERM get ignored) -- a second signal landing during
    THIS fixture's old up-to-30s wait was not caught by anything (pytest's
    fixture teardown machinery only catches the test's own outcome, not an
    arbitrary second KeyboardInterrupt) and escaped straight out, skipping
    `run_state`'s restore entirely. A `return` instead of a `yield` here
    guarantees there is no finalizer at all for this fixture to worry
    about -- the only fixture between `run_state` and the test itself with
    nothing left to interrupt.
    """
    if not live_cfg.ack_pause_all:
        pytest.skip("VW_LIVE_ACK_PAUSE_ALL != 1: refusing to pause other tokens")

    state = run_state
    own_ids = {v["id"] for v in test_tokens["created"].values()}

    t0 = time.monotonic()
    candidates = await select_foreign_tokens(admin, own_ids=own_ids)
    await pause_foreign_tokens(admin, state=state, state_dir=live_cfg.state_dir, candidates=candidates)

    foreign_inflight_at_pause = 0
    try:
        snap = await admin.live_requests()
        foreign_inflight_at_pause = sum(
            1
            for r in snap["requests"]
            if r["token_id"] not in own_ids and r.get("model") == live_cfg.model
        )
    except Exception:  # noqa: BLE001 -- best-effort telemetry only
        pass

    # Drain: wait for already-admitted FOREIGN in-flight requests on OUR
    # target engine to finish (a busy, unrelated engine on the same
    # deployment must never make this loop wait). A single transient poll
    # failure (a 502 mid-deploy, say) must not abort the run -- only the
    # overall deadline does that.
    deadline = time.monotonic() + live_cfg.drain_timeout_s
    while True:
        try:
            snap = await admin.live_requests()
        except Exception:  # noqa: BLE001 -- transient; retry until the deadline
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"drain timeout: admin API still unreachable after "
                    f"{live_cfg.drain_timeout_s}s"
                ) from None
            await asyncio.sleep(1.0)
            continue
        foreign = [
            r
            for r in snap["requests"]
            if r["token_id"] not in own_ids and r.get("model") == live_cfg.model
        ]
        if not foreign:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"drain timeout: {len(foreign)} foreign request(s) still in flight "
                f"after {live_cfg.drain_timeout_s}s"
            )
        await asyncio.sleep(1.0)

    drain_seconds = time.monotonic() - t0
    print(
        f"[tests/live] isolation ready: paused {len(state.paused)} token(s), "
        f"drain {drain_seconds:.1f}s, foreign in-flight at pause "
        f"{foreign_inflight_at_pause}."
    )

    return {
        "paused_count": len(state.paused),
        "drain_seconds": drain_seconds,
        "foreign_inflight_at_pause": foreign_inflight_at_pause,
    }
