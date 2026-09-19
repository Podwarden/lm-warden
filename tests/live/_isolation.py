"""Isolation: pause every other live token, crash-safe state file, restore.

Shared by the `isolation` fixture (conftest.py) and the standalone
`python -m tests.live.restore` so the two recovery paths cannot drift (plan
section 5). No credentials ever go in the state file -- only token ids and
the `paused_at` value our own PATCH returned (the proof that WE paused a
given token, so restore never touches one an operator paused or unpaused
independently mid-run). Foreign token NAMES are deliberately kept OUT of
every failure/log line (ids only) -- only OUR OWN generated names
(`created_tokens`, never sensitive: `<prefix>-<run id>-<role>`) ever appear.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from tests.live._admin import AdminSession

SCHEMA = 1

#: Same format TokenRepo's sqlite_utc_now() writes (app/db/repos/tokens.py
#: _SQLITE_UTC_FMT) -- comparable lexicographically AND as wall-clock order,
#: which the pending-reconciliation check relies on.
_SQLITE_UTC_FMT = "%Y-%m-%d %H:%M:%S"

#: I-a: pending-entry reconciliation margin. Client/server clock skew plus
#: second-truncation makes an exact `>=` boundary comparison unsafe (see
#: CR2's history: it used to drop entries outright on a skew, which is worse
#: than this). Safe to be generous here because a `pending` entry only ever
#: exists for a token `select_foreign_tokens` observed UNPAUSED moments
#: before -- nothing plausible pauses it minutes earlier for an unrelated
#: reason, so a wide margin costs nothing but closes the skew gap.
_PENDING_MARGIN = timedelta(minutes=10)


def _sqlite_utc_now() -> str:
    return datetime.now(UTC).strftime(_SQLITE_UTC_FMT)


def _sqlite_minus(ts: str, delta: timedelta) -> str:
    return (datetime.strptime(ts, _SQLITE_UTC_FMT).replace(tzinfo=UTC) - delta).strftime(
        _SQLITE_UTC_FMT
    )


def is_our_token_name(name: str, prefix: str) -> bool:
    """True iff `name` is one of ours: `<prefix>-<8 hex run id>-<role>`.

    Used by both the sweep-leftovers path and `restore.py --sweep` so a
    misconfigured (or empty) VW_LIVE_TOKEN_PREFIX can never make either one
    match tokens it doesn't own. An empty prefix is rejected outright --
    `f"{''}-"` would otherwise match almost anything with a hyphen in it.
    """
    if not prefix:
        return False
    pattern = re.escape(prefix) + r"-[0-9a-f]{8}-(filler|p\d[ab]?)$"
    return re.match(pattern, name) is not None


@dataclass
class StateFile:
    schema: int
    run_id: str
    base_url: str
    model: str
    created_utc: str
    #: Same-format wall clock as `paused_at` (see _SQLITE_UTC_FMT above) --
    #: the pending-reconciliation check compares against this (minus
    #: _PENDING_MARGIN), not `created_utc` (ISO8601/Z, a different format).
    run_started_sqlite: str
    phase: str  # "pausing" | "running" | "restoring" | "restored"
    created_tokens: list[dict[str, str]] = field(default_factory=list)
    paused: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def new(cls, *, run_id: str, base_url: str, model: str) -> StateFile:
        return cls(
            schema=SCHEMA,
            run_id=run_id,
            base_url=base_url,
            model=model,
            created_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            run_started_sqlite=_sqlite_utc_now(),
            phase="pausing",
            created_tokens=[],
            paused=[],
        )

    def path(self, state_dir: str) -> str:
        return os.path.join(state_dir, f"{self.run_id}.json")

    def write(self, state_dir: str) -> None:
        os.makedirs(state_dir, exist_ok=True)
        try:
            os.chmod(state_dir, stat.S_IRWXU)  # 0700 -- the dir holds run ids, nothing worse
        except OSError:
            pass
        target = self.path(state_dir)
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)  # atomic -- readable state at every instant

    @classmethod
    def load(cls, path: str) -> StateFile:
        with open(path) as f:
            data = json.load(f)
        data.setdefault("run_started_sqlite", data.get("created_utc", "")[:19].replace("T", " "))
        return cls(**data)


async def select_foreign_tokens(admin: AdminSession, *, own_ids: set[str]) -> list[dict]:
    """Live tokens (not expired/revoked/already-paused) that are not ours.

    Predecessors still inside a rotation grace window appear live and CAN
    still make requests -- they are included, and pausing them is allowed.
    Already-paused, expired, or revoked tokens are never touched.
    """
    items = await admin.list_tokens()
    return [
        t
        for t in items
        if not t["is_expired"]
        and not t["is_revoked"]
        and not t["is_paused"]
        and t["id"] not in own_ids
    ]


async def pause_foreign_tokens(
    admin: AdminSession, *, state: StateFile, state_dir: str, candidates: list[dict]
) -> None:
    """Pause every candidate, writing the state file BEFORE each PATCH (a
    `{paused_at: None, pending: True}` placeholder -- I1: a lost PATCH
    *response* must not lose the pause from our records, since the PATCH may
    have taken effect server-side even though we never saw the reply) and
    again once we know the outcome.

    CR2: a 200 PATCH response is ALWAYS recorded as ours, never dropped
    based on a clock comparison. An earlier version compared the server's
    returned `paused_at` (second-truncated, server clock) against a
    client-clock timestamp to detect an operator racing us for the same
    token -- but client/server clock skew (or nothing more than
    second-truncation) could make our OWN pause look "earlier" than our own
    request time, silently dropping the ONE record standing between us and
    an unrecoverable orphaned pause. That is a strictly worse failure mode
    than the race it guarded against. The race is narrowed instead, not
    eliminated: a fresh GET immediately before the PATCH skips a token an
    operator already paused since the `select_foreign_tokens` listing. The
    residual window (between that GET and our PATCH) is now microseconds,
    and if we still race someone there, the cost is a restorable "we
    unpause a token someone else also wanted paused" -- not a leak.
    """
    state.phase = "pausing"
    state.write(state_dir)
    for t in candidates:
        entry: dict[str, Any] = {
            "id": t["id"],
            "paused_at": None,
            "pending": True,
            "restored": False,
        }
        state.paused.append(entry)
        state.write(state_dir)  # visible to a crash/kill BEFORE the PATCH lands

        try:
            current = await admin.get_token(t["id"])
        except Exception:  # noqa: BLE001 -- can't narrow the race; fall through to the PATCH
            current = None
        if current is not None and current.get("is_paused"):
            # An operator paused it since the listing -- not ours.
            state.paused.remove(entry)
            state.write(state_dir)
            continue

        try:
            result = await admin.set_paused(t["id"], True)
        except Exception:  # noqa: BLE001 -- response lost; PATCH may have landed
            # Leave the `pending` entry in place. restore() reconciles it via
            # a follow-up GET (it does not trust a stale in-memory result).
            state.write(state_dir)
            continue

        if result is None or result.get("conflict"):
            # 404 (gone) or 409 (went dead meanwhile) -- nothing was paused
            # by us. The PATCH definitely did not take effect, so it is safe
            # to drop the speculative entry outright.
            state.paused.remove(entry)
            state.write(state_dir)
            continue

        # CR2: any successful PATCH is ours. Full stop.
        entry["paused_at"] = result.get("paused_at")
        entry["pending"] = False
        state.write(state_dir)
    state.phase = "running"
    state.write(state_dir)


#: _restore_one's three possible outcomes.
#: "done"  -- fully handled; caller marks the entry restored.
#: "retry" -- transient; caller's backoff loop tries again.
#: "defer" -- I2: a PENDING entry seen unpaused needs a SECOND look, at
#:            least `_PENDING_CONFIRM_S` after the first, before we trust
#:            it -- see _restore_one's docstring.
_DONE, _RETRY, _DEFER = "done", "retry", "defer"

#: I2 (round 5): the pending-unpaused confirmation window. A PATCH whose
#: client-side wait we cancelled (cancel_pending_tasks()) does not stop
#: an already-sent request from still landing on the server -- for as
#: long as the admin client's own httpx timeout would have kept us
#: waiting (_admin.py: httpx.Timeout(30)). 35s is deliberately LONGER
#: than that 30s, so "confirmed never paused" can never be reached while
#: the PATCH could still plausibly be in flight. Round 4's version used a
#: PASS NUMBER instead of wall-clock time, which broke the moment
#: restore.py re-runs restore_paused_tokens in a FRESH process (its own
#: pass counter restarts at 0, so an entry first seen on pass 4 in the
#: original run would defer forever in the new one -- an absolute
#: timestamp is valid across process restarts on the same machine, a
#: pass index is not).
_PENDING_CONFIRM_S = 35.0


async def _restore_one(
    admin: AdminSession, entry: dict[str, Any], *, run_started_sqlite: str, force: bool
) -> tuple[str, str | None]:
    """Returns (status, failure_line) -- status is one of _DONE/_RETRY/_DEFER.

    I2: a PENDING entry (I1 -- we never confirmed our own PATCH landed)
    seen UNPAUSED is NOT proof it was never ours. A PATCH we sent can still
    be in flight server-side even after `cancel_pending_tasks()` cancelled
    our client-side wait for it (cancellation stops US waiting for the
    reply; it does not un-send an already-sent request) -- it can land any
    time up to the admin client's own request timeout. A single "seen
    unpaused" GET at the wrong moment would OTHERWISE mark this entry
    `restored` on the spot, and the token would end up paused forever with
    nothing left tracking it. So: the FIRST time a pending entry is seen
    unpaused, record the wall-clock time and defer. Only a SECOND
    observation, at least `_PENDING_CONFIRM_S` after the first, confirms
    "genuinely never paused, not ours" and marks it done. The caller keeps
    polling (its own outer-pass loop) until that window elapses or its own
    overall cap is hit.

    A pending entry seen PAUSED, by contrast, needs no such caution: it
    either IS ours (paused_at at/after our run's start, within the I-a
    margin) -- unpause it and verify the PATCH succeeds below -- or it
    was paused by someone else well before our run, which the margin
    check also settles in one look (an already-paused token cannot
    become "more paused" later, so there is nothing timing-sensitive to
    reconfirm).
    """
    token_id = entry["id"]
    current = await admin.get_token(token_id)
    if current is None:
        return _DONE, None  # token deleted -- nothing to restore

    if not current.get("is_paused"):
        if entry.get("pending"):
            first_seen = entry.get("_pending_unpaused_since_ts")
            now = time.time()
            if first_seen is None:
                entry["_pending_unpaused_since_ts"] = now
                return _DEFER, None
            if now - first_seen >= _PENDING_CONFIRM_S:
                return _DONE, None  # confirmed unpaused across the full window -- not ours
            return _DEFER, None
        # Minor: an operator unpaused it by hand mid-run (a NON-pending
        # entry -- we already know we own this one, no ambiguity). That IS
        # the end state we want -- not a failure, nothing left to do.
        return _DONE, None

    if entry.get("pending"):
        # I1 reconciliation, I-a margin: we never confirmed OUR PATCH
        # landed (the response was lost). It's ours to restore if the
        # token's current paused_at is at/after this run's start, with a
        # generous margin subtracted for clock skew -- safe because a
        # `pending` entry only exists for a token the listing showed
        # unpaused moments before.
        current_paused_at = current.get("paused_at")
        margin = _sqlite_minus(run_started_sqlite, _PENDING_MARGIN)
        if current_paused_at is None or current_paused_at < margin:
            return _DONE, None  # paused, but clearly by someone else -- leave it
    elif not force and current.get("paused_at") != entry.get("paused_at"):
        # Minor (round 5): this entry is marked `restored` below (nothing
        # MORE for a normal pass to do), which means a bare re-run cannot
        # retry it -- only `--force` reopens exactly this kind of entry
        # (see restore_paused_tokens). Flagged so the caller knows to.
        entry["_left_mismatch"] = True
        return _DONE, (
            f"token {token_id}: paused_at changed "
            f"({entry.get('paused_at')!r} -> {current.get('paused_at')!r}); "
            "left paused -- an operator touched it during the run. This "
            "entry will not be retried automatically; re-run with --force "
            "to attempt unpausing it anyway."
        )

    result = await admin.set_paused(token_id, False)
    if result is not None and not result.get("conflict"):
        return _DONE, None
    return _RETRY, None  # 409/None -- transient, worth a retry


#: I3: exponential backoff totalling ~3-5 minutes per token, not ~1.5s --
#: this runs in the finalizer that's the last line of defence against
#: leaving a real user's token paused, so it is worth waiting out a
#: transient network blip or a deployment mid-restart.
_RETRY_DELAYS_S = (1, 2, 4, 8, 16, 32, 60, 60, 60)  # sum ≈ 243s

async def cancel_pending_tasks(*, timeout: float | None = None) -> None:  # noqa: ASYNC109
    """CR1: cancel every OTHER task on the CURRENT event loop and wait for
    those cancellations to actually land before touching the server.

    An interrupt (SIGTERM/Ctrl-C/KeyboardInterrupt) does not cancel the
    task it interrupts -- it is raised at the event loop's own poll
    boundary and escapes `run_until_complete()` without touching whatever
    task was suspended inside it. That task stays registered on the SAME
    loop, still pending, mid-execution -- and the next time the loop runs
    (e.g. for a finalizer's own `await`s), it can RESUME and race whatever
    that finalizer does. Reproduced: an interrupted `select_foreign_tokens`
    / `pause_foreign_tokens` resumed during the restore finalizer and paused
    tokens AFTER restore had already run and found nothing to do. Call this
    before `restore_paused_tokens` for exactly that reason -- see
    conftest.py's `run_state` fixture, the real caller.

    CAUTION (round 5): never wrap a CALL to this function in
    `asyncio.wait_for(cancel_pending_tasks(), ...)` -- `wait_for` on a bare
    coroutine calls `ensure_future()`, which wraps it in a NEW Task. That
    new task becomes `asyncio.current_task()` from INSIDE this function,
    not the real caller, so the real caller (now merely "some other
    pending task" from this function's point of view) ends up in `others`
    and gets cancelled as collateral damage -- self-inflicted, and exactly
    what happened the first time this was tried (surfaced as an unhandled
    CancelledError escaping test teardown). `timeout`, if given, bounds
    the wait CORRECTLY instead: `me`/`others` are captured before any
    wrapping happens, and only the already-constructed `gather()` FUTURE
    (not a bare coroutine needing its own new task) is what `wait_for`
    wraps -- `ensure_future` on an existing Future is a no-op, so no new
    task is created and this function's own caller is never at risk.
    """
    me = asyncio.current_task()
    others = [t for t in asyncio.all_tasks() if t is not me]
    for t in others:
        t.cancel()
    if not others:
        return
    if timeout is None:
        await asyncio.gather(*others, return_exceptions=True)
        return
    # asyncio.wait, not wait_for(gather(...)): on 3.11 wait_for still waits
    # for every child after its timeout, so a task that swallows its
    # cancellation would hold teardown forever. wait() returns at the cap.
    await asyncio.wait(others, timeout=timeout)  # some task may ignore cancellation -- proceed anyway


#: I2: outer passes over the WHOLE set, spaced _OUTER_PASS_DELAY_S apart.
#: This is what gives a `_DEFER`red pending-unpaused entry its required
#: SECOND look, `_PENDING_CONFIRM_S` after the first (_restore_one refuses
#: to confirm "not ours" on the very first observation -- see its
#: docstring for why one look is not enough). It also gives any entry
#: that failed with a transient exception more chances across a longer
#: span than one pass's own per-entry backoff alone.
#:
#: _MAX_OUTER_PASSES is sized so the loop can ALWAYS reach
#: _PENDING_CONFIRM_S for a defer-on-the-very-first-pass entry (with
#: generous headroom for other entries' own retries in between), while
#: staying well inside the ~10-minute wall-clock cap the caller
#: (conftest.py's run_state finalizer, restore.py's CLI) wraps this call
#: in -- THAT is the authoritative bound; this is defense in depth against
#: this function ever being called without one.
_OUTER_PASS_DELAY_S = 3.0
_MAX_OUTER_PASSES = 150  # 150 * 3s = 450s, comfortably >> _PENDING_CONFIRM_S


async def restore_paused_tokens(
    admin: AdminSession, *, state: StateFile, state_dir: str, force: bool = False
) -> list[str]:
    """Restore every token WE paused (see `_restore_one` for the ownership
    and pending-reconciliation rules). Loops over the whole set until no
    unrestored entry remains or `_MAX_OUTER_PASSES` is exhausted (CR1, I2).
    Returns failure lines (empty = every token this run paused is now
    restored, force-restored, or was never really ours).

    `force=True` also REOPENS (retries) any entry previously left alone
    ONLY because its `paused_at` didn't match ours (`_left_mismatch`) --
    without this, an entry already marked `restored` from an earlier call
    (in this process or a prior one, loaded from the state file) would be
    skipped outright regardless of `force`, making `--force` a no-op for
    exactly the case it exists to handle.
    """
    state.phase = "restoring"
    state.write(state_dir)
    # One report line per token, kept across outer passes: a "paused_at
    # changed" line from pass 1 must survive a later pass that only
    # re-checks some OTHER, deferred entry.
    reports: dict[str, str] = {}
    for outer_pass in range(_MAX_OUTER_PASSES):
        for entry in state.paused:
            if entry.get("restored"):
                if not (force and entry.get("_left_mismatch")):
                    continue
                entry["restored"] = False  # --force reopens this one entry kind
            token_id = entry["id"]
            status = _RETRY
            reported: str | None = None
            last_exc: Exception | None = None
            for delay in (*_RETRY_DELAYS_S, None):
                try:
                    status, reported = await _restore_one(
                        admin, entry, run_started_sqlite=state.run_started_sqlite, force=force
                    )
                    last_exc = None
                    if status in (_DONE, _DEFER):
                        break
                except Exception as exc:  # noqa: BLE001 -- retry loop; last exc reported below
                    last_exc = exc
                if delay is not None:
                    await asyncio.sleep(delay)
            if reported:
                reports[token_id] = reported
            elif status == _DONE:
                reports.pop(token_id, None)  # an earlier pass's transient error is resolved
            if status == _DONE:
                entry["restored"] = True
            elif status == _RETRY and last_exc is not None:
                reports[token_id] = f"token {token_id}: {type(last_exc).__name__}"
            # _DEFER: not restored yet, no failure either -- a later pass,
            # at least _PENDING_CONFIRM_S after the first observation,
            # re-checks it.
            state.write(state_dir)
        remaining = [e for e in state.paused if not e.get("restored")]
        if not remaining:
            break
        if outer_pass < _MAX_OUTER_PASSES - 1:
            await asyncio.sleep(_OUTER_PASS_DELAY_S)
    # Anything still unrestored after every pass (including a `defer` that
    # never got its confirming second look) must be reported, not silently
    # dropped -- the caller's pytest.fail depends on `failures` being
    # non-empty whenever a real user's token might still be paused.
    still_unrestored = [e for e in state.paused if not e.get("restored")]
    for e in still_unrestored:
        reports.setdefault(
            e["id"], f"token {e['id']}: not confirmed restored after {_MAX_OUTER_PASSES} passes"
        )
    failures = list(reports.values())
    if not failures:
        state.phase = "restored"
        state.write(state_dir)
    return failures


async def delete_created_tokens(admin: AdminSession, *, state: StateFile) -> list[str]:
    """Best-effort delete of every token this run created, with the same
    ~3-5 minute exponential backoff as restore (leftovers are cheap to sweep
    later; still worth not giving up in 1.5s)."""
    failures: list[str] = []
    for t in state.created_tokens:
        ok = False
        last_exc: Exception | None = None
        for delay in (*_RETRY_DELAYS_S, None):
            try:
                await admin.delete_token(t["id"])
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            if delay is not None:
                await asyncio.sleep(delay)
        if not ok and last_exc is not None:
            # Our own generated name (vwlive-<run>-<role>) -- never sensitive.
            failures.append(f"token {t['id']} ({t.get('name')}): {type(last_exc).__name__}")
    return failures
