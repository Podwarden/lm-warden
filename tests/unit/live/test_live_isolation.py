"""Pure-logic unit tests for tests/live/_isolation.py: state-file handling,
the "only unpause if paused_at still equals ours" invariant, the I1
lost-PATCH-response reconciliation, the CR2 ownership-race guard, and the
CR1 abandoned-task-resumes-during-restore race -- all against a fake
AdminSession, no network.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from tests.live._isolation import (
    StateFile,
    cancel_pending_tasks,
    delete_created_tokens,
    is_our_token_name,
    pause_foreign_tokens,
    restore_paused_tokens,
    select_foreign_tokens,
)


def _now_sqlite() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _sqlite_offset(minutes: int) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


class FakeAdmin:
    """Minimal stand-in for AdminSession's token-CRUD surface.

    `set_paused(True)` mirrors the real server's `COALESCE(paused_at, ?)`:
    an already-paused token KEEPS its existing paused_at rather than being
    overwritten. `lose_response_for` simulates I1: the PATCH mutates state
    but the caller never sees the reply. `block` maps a key (a token id, or
    `"__list__"` for `list_tokens`) to a delay in seconds the matching call
    sleeps before doing anything -- a REAL, self-completing await (like an
    actual in-flight network call), not a wait that hangs forever, so the
    CR1 reproduction below models a task that WOULD resume on its own if
    nothing cancelled it first -- pinning down exactly where an "abandoned"
    task is suspended when the interrupt lands. `late_land` (I2) models the
    CR1 residual case precisely: the underlying server-side effect happens
    on its own delay, DETACHED from whether the caller's own wait for the
    HTTP response gets cancelled -- cancelling `set_paused`'s caller stops
    OUR wait, not the request already in flight. Scheduled via
    `loop.call_later()`, a `TimerHandle`, deliberately NOT `asyncio.create_task()`
    -- a Task is exactly what `cancel_pending_tasks()` enumerates and cancels
    (`asyncio.all_tasks()`), so a "detached" task would be cancelled right
    along with everything else and the PATCH would never land at all,
    making the whole scenario vacuous. A TimerHandle is invoked directly by
    the event loop's own callback machinery and is invisible to
    `asyncio.all_tasks()`.
    """

    def __init__(
        self,
        tokens: dict[str, dict],
        *,
        lose_response_for: set[str] | None = None,
        block: dict[str, float] | None = None,
        late_land: dict[str, float] | None = None,
    ):
        self.tokens = tokens  # id -> token dict, mutated in place like the real API
        self.deleted: list[str] = []
        self.pause_calls: list[tuple[str, bool]] = []
        self.get_calls: list[str] = []
        self._lose_response_for = lose_response_for or set()
        self._block = block or {}
        self._late_land = late_land or {}

    async def list_tokens(self) -> list[dict]:
        if "__list__" in self._block:
            await asyncio.sleep(self._block["__list__"])
        return list(self.tokens.values())

    async def set_paused(self, token_id: str, paused: bool) -> dict | None:
        if paused and token_id in self._late_land:
            delay = self._late_land[token_id]

            def _land() -> None:
                t = self.tokens.get(token_id)
                if t is not None:
                    if t.get("paused_at") is None:
                        t["paused_at"] = _now_sqlite()
                    t["is_paused"] = True

            # TimerHandle, NOT a Task -- see the class docstring. Survives
            # cancel_pending_tasks() because asyncio.all_tasks() never sees it.
            asyncio.get_running_loop().call_later(delay, _land)
            # Simulate "still waiting for the HTTP response" -- in the real
            # world this is what cancel_pending_tasks() cancels; here the
            # caller task itself gets cancelled by the test before this
            # ever returns.
            await asyncio.sleep(3600)
        if paused and token_id in self._block:
            await asyncio.sleep(self._block[token_id])
        self.pause_calls.append((token_id, paused))
        t = self.tokens.get(token_id)
        if t is None:
            return None
        if t.get("_dead"):
            return {"conflict": True}
        if paused:
            if t.get("paused_at") is None:
                t["paused_at"] = _now_sqlite()
            t["is_paused"] = True
        else:
            t["is_paused"] = False
            t["paused_at"] = None
        if token_id in self._lose_response_for:
            raise ConnectionError("simulated lost response")
        return dict(t)

    async def get_token(self, token_id: str) -> dict | None:
        self.get_calls.append(token_id)
        t = self.tokens.get(token_id)
        return dict(t) if t is not None else None

    async def delete_token(self, token_id: str) -> bool:
        self.deleted.append(token_id)
        self.tokens.pop(token_id, None)
        return True


def _token(id_, name, *, is_expired=False, is_revoked=False, is_paused=False, paused_at=None):
    return {
        "id": id_,
        "name": name,
        "is_expired": is_expired,
        "is_revoked": is_revoked,
        "is_paused": is_paused,
        "paused_at": paused_at if paused_at is not None else (_now_sqlite() if is_paused else None),
    }


@pytest.mark.asyncio
async def test_select_foreign_tokens_excludes_own_dead_and_already_paused():
    admin = FakeAdmin(
        {
            "own1": _token("own1", "vwlive-abc-filler"),
            "other-live": _token("other-live", "ci-bot"),
            "other-expired": _token("other-expired", "old-key", is_expired=True),
            "other-revoked": _token("other-revoked", "dead-key", is_revoked=True),
            "other-paused": _token("other-paused", "already-paused", is_paused=True),
        }
    )
    candidates = await select_foreign_tokens(admin, own_ids={"own1"})
    assert [c["id"] for c in candidates] == ["other-live"]


@pytest.mark.asyncio
async def test_pause_foreign_tokens_writes_state_before_first_patch(tmp_path):
    admin = FakeAdmin({"t1": _token("t1", "ci-bot"), "t2": _token("t2", "legacy-key")})
    state = StateFile.new(run_id="abc12345", base_url="https://example.invalid", model="m")
    state_dir = str(tmp_path)

    await pause_foreign_tokens(
        admin, state=state, state_dir=state_dir, candidates=[admin.tokens["t1"], admin.tokens["t2"]]
    )

    # Pre-PATCH ownership-narrowing GET (CR2), then the PATCH itself.
    assert admin.get_calls == ["t1", "t2"]
    assert admin.pause_calls == [("t1", True), ("t2", True)]
    assert {p["id"] for p in state.paused} == {"t1", "t2"}
    assert all(p["restored"] is False for p in state.paused)
    assert all(p["pending"] is False for p in state.paused)  # confirmed by the response
    assert state.phase == "running"

    # The state file on disk reflects the final state (written after every change).
    on_disk = StateFile.load(state.path(state_dir))
    assert {p["id"] for p in on_disk.paused} == {"t1", "t2"}


@pytest.mark.asyncio
async def test_pause_foreign_tokens_skips_conflict_and_gone(tmp_path):
    admin = FakeAdmin({"t1": {**_token("t1", "dies-mid-run"), "_dead": True}})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    await pause_foreign_tokens(
        admin, state=state, state_dir=str(tmp_path), candidates=[admin.tokens["t1"]]
    )
    assert state.paused == []  # 409 conflict -- nothing recorded as paused by us


@pytest.mark.asyncio
async def test_pause_foreign_tokens_i1_lost_response_leaves_pending_entry(tmp_path):
    """I1: the PATCH mutated server state, but we never saw the response --
    the state file must still carry a record of it (pending=True), not
    silently drop it."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot")}, lose_response_for={"t1"})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")

    await pause_foreign_tokens(
        admin, state=state, state_dir=str(tmp_path), candidates=[admin.tokens["t1"]]
    )

    assert len(state.paused) == 1
    assert state.paused[0]["id"] == "t1"
    assert state.paused[0]["pending"] is True
    assert state.paused[0]["paused_at"] is None
    # The server-side pause DID happen, even though we never saw the reply.
    assert admin.tokens["t1"]["is_paused"] is True


@pytest.mark.asyncio
async def test_restore_reconciles_a_pending_entry_when_it_is_ours(tmp_path):
    """I1 follow-through: restore's re-GET sees the token paused with a
    paused_at at/after this run's start, so a `pending` entry (lost PATCH
    response) is still ours to restore."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    # Simulate what pause_foreign_tokens leaves behind on a lost response,
    # then have the server-side pause actually be in effect (as it would be
    # if the PATCH really landed).
    admin.tokens["t1"]["is_paused"] = True
    admin.tokens["t1"]["paused_at"] = _now_sqlite()
    state.paused = [{"id": "t1", "paused_at": None, "pending": True, "restored": False}]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []
    assert admin.tokens["t1"]["is_paused"] is False
    assert state.paused[0]["restored"] is True


@pytest.mark.asyncio
async def test_restore_leaves_a_pending_entry_alone_when_not_ours(tmp_path):
    """A `pending` entry whose token's paused_at predates this run's start
    (by more than the I-a margin) was never actually paused by us -- restore
    must not touch it."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot", is_paused=True, paused_at="2020-01-01 00:00:00")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.paused = [{"id": "t1", "paused_at": None, "pending": True, "restored": False}]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []  # handled, not a failure -- just not ours
    assert ("t1", False) not in admin.pause_calls
    assert admin.tokens["t1"]["is_paused"] is True  # left exactly as found


@pytest.mark.asyncio
async def test_restore_pending_entry_within_ia_margin_is_still_ours(tmp_path):
    """I-a: the pending-reconciliation comparison uses run_started_sqlite
    MINUS a 10-minute margin, to absorb client/server clock skew -- a
    paused_at a few minutes "before" the recorded run start is still
    treated as ours (a `pending` entry only exists for a token the listing
    showed unpaused moments earlier, so this is safe)."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    # 3 minutes "before" run_started_sqlite -- within the 10-minute margin.
    skewed_paused_at = _sqlite_offset(-3)
    admin.tokens["t1"]["is_paused"] = True
    admin.tokens["t1"]["paused_at"] = skewed_paused_at
    state.paused = [{"id": "t1", "paused_at": None, "pending": True, "restored": False}]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []
    assert admin.tokens["t1"]["is_paused"] is False  # restored -- treated as ours
    assert state.paused[0]["restored"] is True


@pytest.mark.asyncio
async def test_restore_pending_entry_outside_ia_margin_is_not_ours(tmp_path):
    """The same shape, 20 minutes "before" run start -- outside the margin,
    correctly left alone."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    admin.tokens["t1"]["is_paused"] = True
    admin.tokens["t1"]["paused_at"] = _sqlite_offset(-20)
    state.paused = [{"id": "t1", "paused_at": None, "pending": True, "restored": False}]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []
    assert admin.tokens["t1"]["is_paused"] is True  # left alone -- not ours


@pytest.mark.asyncio
async def test_pause_foreign_tokens_skips_a_token_paused_since_the_listing(tmp_path):
    """CR2 (narrowed race): a fresh GET runs immediately before the PATCH.
    If an operator paused the token since `select_foreign_tokens`'s
    listing, we see that in the GET and skip it -- never PATCH it, never
    record it."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot")})
    # Simulate the operator's pause landing between listing and our PATCH.
    admin.tokens["t1"]["is_paused"] = True
    admin.tokens["t1"]["paused_at"] = _now_sqlite()

    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    await pause_foreign_tokens(
        admin, state=state, state_dir=str(tmp_path), candidates=[{"id": "t1", "name": "ci-bot"}]
    )

    assert state.paused == []  # never recorded -- not ours to restore later
    assert admin.pause_calls == []  # PATCH was never even sent


@pytest.mark.asyncio
async def test_pause_foreign_tokens_cr2_never_drops_a_successful_patch(tmp_path):
    """CR2 regression: a PATCH that returns 200 is recorded as ours no
    matter what `paused_at` value comes back -- there is no client/server
    clock comparison left that could discard it. (The old mechanism dropped
    exactly this shape of response when client and server clocks disagreed,
    permanently orphaning the pause -- worse than the race it guarded
    against.)"""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")

    await pause_foreign_tokens(
        admin, state=state, state_dir=str(tmp_path), candidates=[{"id": "t1", "name": "ci-bot"}]
    )

    assert len(state.paused) == 1
    assert state.paused[0]["id"] == "t1"
    assert state.paused[0]["pending"] is False
    assert state.paused[0]["paused_at"] == admin.tokens["t1"]["paused_at"]


@pytest.mark.asyncio
async def test_restore_only_unpauses_when_paused_at_still_matches(tmp_path):
    """The load-bearing invariant: restore must NOT touch a token whose
    paused_at changed after we paused it (an operator re-touched it)."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot", is_paused=True)})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.paused = [
        {"id": "t1", "paused_at": "2026-01-01 00:00:00", "pending": False, "restored": False}
    ]

    # Operator re-paused it at a different time after our run paused it.
    admin.tokens["t1"]["paused_at"] = "2026-01-01 09:99:99-DIFFERENT"

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert len(failures) == 1
    assert "paused_at changed" in failures[0]
    assert "ci-bot" not in failures[0]  # minor: no foreign token name in the failure line
    # set_paused(False) must never have been called for this token.
    assert ("t1", False) not in admin.pause_calls
    assert admin.tokens["t1"]["is_paused"] is True


@pytest.mark.asyncio
async def test_restore_treats_already_unpaused_as_restored_not_a_failure(tmp_path):
    """Minor: an operator unpaused the token by hand mid-run. That IS the
    desired end state -- not a failure, nothing left to do."""
    admin = FakeAdmin({"t1": _token("t1", "ci-bot", is_paused=False, paused_at=None)})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.paused = [
        {"id": "t1", "paused_at": "2026-01-01 00:00:00", "pending": False, "restored": False}
    ]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []
    assert ("t1", False) not in admin.pause_calls  # nothing to PATCH -- already unpaused
    assert state.paused[0]["restored"] is True


@pytest.mark.asyncio
async def test_restore_unpauses_when_paused_at_matches(tmp_path):
    admin = FakeAdmin({"t1": _token("t1", "ci-bot", is_paused=True, paused_at="2026-01-01 00:00:00")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.paused = [
        {"id": "t1", "paused_at": "2026-01-01 00:00:00", "pending": False, "restored": False}
    ]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []
    assert admin.tokens["t1"]["is_paused"] is False
    assert state.paused[0]["restored"] is True
    assert state.phase == "restored"


@pytest.mark.asyncio
async def test_restore_force_skips_equality_check(tmp_path):
    admin = FakeAdmin({"t1": _token("t1", "ci-bot", is_paused=True, paused_at="SOMETHING-ELSE")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.paused = [
        {"id": "t1", "paused_at": "2026-01-01 00:00:00", "pending": False, "restored": False}
    ]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path), force=True)
    assert failures == []
    assert admin.tokens["t1"]["is_paused"] is False


@pytest.mark.asyncio
async def test_restore_tolerates_already_deleted_token(tmp_path):
    admin = FakeAdmin({})  # token already gone (404)
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.paused = [{"id": "gone", "paused_at": "y", "pending": False, "restored": False}]

    failures = await restore_paused_tokens(admin, state=state, state_dir=str(tmp_path))
    assert failures == []
    assert state.paused[0]["restored"] is True


@pytest.mark.asyncio
async def test_delete_created_tokens_best_effort(tmp_path):
    admin = FakeAdmin({"a": _token("a", "vwlive-x-filler")})
    state = StateFile.new(run_id="run1", base_url="https://example.invalid", model="m")
    state.created_tokens = [{"id": "a", "name": "vwlive-x-filler"}, {"id": "missing", "name": "y"}]

    failures = await delete_created_tokens(admin, state=state)
    assert failures == []  # delete_token tolerates a token that's already gone
    assert "a" in admin.deleted


def test_state_file_roundtrip_atomic_write(tmp_path):
    state = StateFile.new(run_id="deadbeef", base_url="https://example.invalid", model="m")
    state.created_tokens = [{"id": "1", "name": "vwlive-deadbeef-filler"}]
    state.paused = [{"id": "2", "paused_at": "t", "pending": False, "restored": False}]
    state.write(str(tmp_path))

    path = state.path(str(tmp_path))
    loaded = StateFile.load(path)
    assert loaded.run_id == "deadbeef"
    assert loaded.created_tokens == state.created_tokens
    assert loaded.paused == state.paused
    assert loaded.run_started_sqlite  # I1: present so pending-reconciliation has something to compare

    # No credentials of any kind land in the file.
    raw = json.loads(open(path).read())
    dumped = json.dumps(raw)
    for forbidden in ("password", "access_token", "csrf", "bearer", "vw_"):
        assert forbidden not in dumped.lower()


def test_state_dir_is_chmod_0700(tmp_path):
    state = StateFile.new(run_id="deadbeef", base_url="https://example.invalid", model="m")
    state.write(str(tmp_path))
    import stat as stat_module

    mode = stat_module.S_IMODE(tmp_path.stat().st_mode)
    assert mode == 0o700


def test_state_file_path_scoped_to_run_id(tmp_path):
    state = StateFile.new(run_id="abc123", base_url="https://example.invalid", model="m")
    assert state.path(str(tmp_path)).endswith("abc123.json")


def test_state_file_load_backfills_run_started_sqlite_for_old_files(tmp_path):
    """A state file written before run_started_sqlite existed must still
    load (forward-compatible with an in-flight run across a code upgrade)."""
    state = StateFile.new(run_id="old1", base_url="https://example.invalid", model="m")
    state.write(str(tmp_path))
    path = state.path(str(tmp_path))
    raw = json.loads(open(path).read())
    del raw["run_started_sqlite"]
    with open(path, "w") as f:
        json.dump(raw, f)

    loaded = StateFile.load(path)
    assert loaded.run_started_sqlite  # backfilled, not a KeyError


class TestIsOurTokenName:
    def test_matches_expected_shape(self):
        assert is_our_token_name("vwlive-deadbeef-filler", "vwlive")
        assert is_our_token_name("vwlive-deadbeef-p9a", "vwlive")
        assert is_our_token_name("vwlive-deadbeef-p2", "vwlive")

    def test_rejects_unrelated_names(self):
        assert not is_our_token_name("ci-bot", "vwlive")
        assert not is_our_token_name("vwlive-deadbeef", "vwlive")  # missing role
        assert not is_our_token_name("vwlive-notenoughhex-filler", "vwlive")
        assert not is_our_token_name("other-deadbeef-filler", "vwlive")

    def test_rejects_empty_prefix(self):
        # An empty prefix must never match -- "-..." would otherwise catch
        # almost anything with a hyphen in its name.
        assert not is_our_token_name("anything-at-all", "")
        assert not is_our_token_name("-deadbeef-filler", "")


# ---------------------------------------------------------------------------
# CR1: an interrupted select_foreign_tokens/pause_foreign_tokens task must
# never be able to resume and race a later restore_paused_tokens call.
# ---------------------------------------------------------------------------


async def _run_cr1_scenario(
    tmp_path, monkeypatch, *, block_key: str, n_candidates: int = 5
) -> None:
    """Model the reviewer's reproduction: a task running
    select_foreign_tokens -> pause_foreign_tokens is suspended at
    `block_key`, mid a REAL (self-completing) delay -- an "interrupt" (the
    real cause is a signal delivered at the event loop's poll boundary; the
    OBSERVABLE effect is identical: the task is abandoned, still pending,
    mid-`await`, and WOULD resume on its own once that delay elapses,
    exactly like an in-flight network call completing after the signal).

    Applies the real fix's code path -- run_state's finalizer sequence:
    cancel every other task, wait for it, then restore -- and asserts that
    leaves NOTHING paused and the state file at phase "restored", regardless
    of where the abandoned task was suspended.

    The cancelled candidate is left as a `pending, unpaused` entry (I2's
    DEFER path, since nothing actually landed before cancellation) -- shrink
    `_PENDING_CONFIRM_S`/`_OUTER_PASS_DELAY_S` so confirming that is fast;
    CR1's own property (nothing resumes and pauses anything after the fact)
    is orthogonal to I2's confirmation WINDOW length.
    """
    import tests.live._isolation as iso_mod

    monkeypatch.setattr(iso_mod, "_OUTER_PASS_DELAY_S", 0.02)
    monkeypatch.setattr(iso_mod, "_PENDING_CONFIRM_S", 0.1)

    tokens = {f"t{i}": _token(f"t{i}", f"name{i}") for i in range(n_candidates)}
    # 30ms: long enough that the attacker is reliably still asleep when we
    # get around to cancelling it a couple of scheduler ticks later.
    admin = FakeAdmin(tokens, block={block_key: 0.03})
    state = StateFile.new(run_id="cr1", base_url="https://example.invalid", model="m")
    state_dir = str(tmp_path)

    async def attacker() -> None:
        candidates = await select_foreign_tokens(admin, own_ids=set())
        await pause_foreign_tokens(admin, state=state, state_dir=state_dir, candidates=candidates)

    task = asyncio.create_task(attacker())
    await asyncio.sleep(0.01)  # let the attacker start and reach block_key's sleep

    # The fix: run_state's finalizer sequence, invoked the same way
    # conftest.py's run_state fixture does.
    await cancel_pending_tasks()

    failures = await restore_paused_tokens(admin, state=state, state_dir=state_dir)

    assert failures == [], f"block_key={block_key}: unexpected failures {failures}"
    assert all(not t["is_paused"] for t in tokens.values()), (
        f"block_key={block_key}: a token is still paused after restore: {tokens}"
    )
    assert state.phase == "restored", f"block_key={block_key}: phase is {state.phase!r}"
    assert task.done()  # well and truly finished, not quietly still pending


@pytest.mark.asyncio
async def test_cr1_interrupt_during_listing(tmp_path, monkeypatch):
    """Interrupt point 1: suspended inside select_foreign_tokens's
    list_tokens() call, before any candidate is even known."""
    await _run_cr1_scenario(tmp_path, monkeypatch, block_key="__list__")


@pytest.mark.asyncio
async def test_cr1_interrupt_during_early_pausing(tmp_path, monkeypatch):
    """Interrupt point 2: listing completed; t0 was successfully paused;
    suspended pausing t1 (early in the candidate loop)."""
    await _run_cr1_scenario(tmp_path, monkeypatch, block_key="t1")


@pytest.mark.asyncio
async def test_cr1_interrupt_during_late_pausing(tmp_path, monkeypatch):
    """Interrupt point 3: t0-t2 were successfully paused; suspended pausing
    t3 (late in a 5-candidate loop, most of the damage already done)."""
    await _run_cr1_scenario(tmp_path, monkeypatch, block_key="t3")


# ---------------------------------------------------------------------------
# I2: a pending entry seen UNPAUSED must not be trusted on a single look --
# a PATCH cancelled client-side (by cancel_pending_tasks()) can still land
# server-side after the grace period run_state's finalizer gives it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_i2_patch_landing_well_after_a_naive_pass_count_would_confirm_is_still_restored(
    tmp_path, monkeypatch
):
    """FakeAdmin's `late_land` (a loop.call_later TimerHandle, NOT a Task --
    see its docstring for why a Task would make this vacuous) models the
    CR1 residual case precisely: our client-side wait for the PATCH
    response is cancelled (cancel_pending_tasks()), but the request already
    reached the server and takes effect on its own some time later.

    This is deliberately timed so a PASS-COUNT-based confirmation (round 4's
    design -- "seen unpaused twice, on different passes, done") would ALREADY
    have wrongly confirmed "not ours" after just 2 fast passes, well before
    the PATCH lands: with `_OUTER_PASS_DELAY_S` shrunk to 0.05s, 2 passes is
    ~0.1s -- the PATCH lands at 0.5s. Only a WALL-CLOCK confirmation window
    (`_PENDING_CONFIRM_S`, shrunk here to 1.0s so the test stays fast) can
    still be "waiting" at 0.5s and catch it.
    """
    import tests.live._isolation as iso_mod

    monkeypatch.setattr(iso_mod, "_OUTER_PASS_DELAY_S", 0.05)
    monkeypatch.setattr(iso_mod, "_PENDING_CONFIRM_S", 1.0)

    admin = FakeAdmin({"t1": _token("t1", "ci-bot")}, late_land={"t1": 0.5})
    state = StateFile.new(run_id="i2", base_url="https://example.invalid", model="m")
    state_dir = str(tmp_path)

    async def attacker() -> None:
        await pause_foreign_tokens(
            admin, state=state, state_dir=state_dir, candidates=[{"id": "t1", "name": "ci-bot"}]
        )

    task = asyncio.create_task(attacker())
    await asyncio.sleep(0.05)  # let it send the PATCH and start "waiting" for the reply

    # The fix: cancel our own wait (the PATCH is already "in flight" server-side).
    await cancel_pending_tasks()
    assert task.done()

    # Mirrors run_state's 2s grace period, shortened for the test -- still
    # well BEFORE the late-landing PATCH (0.5s after cancel).
    await asyncio.sleep(0.1)

    failures = await restore_paused_tokens(admin, state=state, state_dir=state_dir)
    # Past the late landing (0.5s after cancel): without this the assert below
    # runs before the PATCH lands and passes even on the one-look logic.
    await asyncio.sleep(0.6)

    assert failures == []
    assert admin.tokens["t1"]["is_paused"] is False  # correctly unpaused despite landing late
    assert state.phase == "restored"
    assert len(state.paused) == 1 and state.paused[0]["restored"] is True
