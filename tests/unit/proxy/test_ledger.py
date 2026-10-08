import asyncio
import sqlite3
from types import SimpleNamespace

import aiosqlite
import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.proxy import bookkeeping
from app.proxy.ledger import Ledger
from app.runtime.variants import variant_of
from tests.conftest import wait_until_async


@pytest.fixture
async def db_path(tmp_data_dir):
    path = tmp_data_dir / "vllm-warden.db"
    async with open_db(path) as db:
        await apply_migrations(db)
    # counters and model_samples carry foreign keys to models and api_tokens.
    with sqlite3.connect(path) as raw:
        raw.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices) "
            "VALUES ('m', 'm', 'org/repo', '[0]')"
        )
        raw.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('t', 'test', 'vw_x', 'h', 'inference')"
        )
        raw.commit()
    return path


async def _q(db_path, sql, *args):
    async with open_db(db_path) as db:
        cur = await db.execute(sql, args)
        return await cur.fetchall()


async def test_aggregates_and_writes_once(db_path):
    led = Ledger(db_path, clock=lambda: 60 * 1000 + 5)
    for _ in range(3):
        led.record_request(
            model_id="m",
            token_id="t",
            prompt_tokens=10,
            completion_tokens=2,
            cached_tokens=4,
            variant_id=None,
        )
    led.record_request(
        model_id="m",
        token_id=None,
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=None,
        variant_id=None,
    )
    assert await led.flush() > 0
    assert await _q(
        db_path,
        "SELECT token_id, requests, prompt_tokens, completion_tokens FROM counters ORDER BY token_id",
    ) == [(None, 1, 1, 1), ("t", 3, 30, 6)]
    assert await _q(
        db_path,
        "SELECT requests, prompt_tokens, cached_tokens, cached_measured_requests FROM model_samples",
    ) == [(4, 31, 12, 3)]
    assert await _q(db_path, "SELECT minute, prompt_tokens FROM token_usage_minute") == [(1000, 30)]
    assert led.pending() is False
    assert await led.flush() == 0


async def test_null_token_flushes_land_in_one_row(db_path):
    # #298: the second flush must UPDATE the existing (m, NULL) row, not INSERT
    # beside it -- 0047's partial unique index would refuse that INSERT.
    led = Ledger(db_path)
    for prompt in (5, 7):
        led.record_request(
            model_id="m",
            token_id=None,
            prompt_tokens=prompt,
            completion_tokens=1,
            cached_tokens=None,
            variant_id=None,
        )
        assert await led.flush() > 0
    assert await _q(
        db_path, "SELECT token_id, requests, prompt_tokens, completion_tokens FROM counters"
    ) == [(None, 2, 12, 2)]


async def test_records_during_flush_go_to_next_batch(db_path, monkeypatch):
    led = Ledger(db_path)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=1,
        completion_tokens=0,
        cached_tokens=None,
        variant_id=None,
    )
    real = Ledger._write

    async def slow_write(self, batch):
        self.record_request(
            model_id="m",
            token_id="t",
            prompt_tokens=100,
            completion_tokens=0,
            cached_tokens=None,
            variant_id=None,
        )
        await real(self, batch)

    monkeypatch.setattr(Ledger, "_write", slow_write)
    await led.flush()
    monkeypatch.setattr(Ledger, "_write", real)
    await led.flush()
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(2, 101)]


async def test_failed_flush_merges_back_and_retries(db_path, monkeypatch):
    led = Ledger(db_path)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=5,
        completion_tokens=0,
        cached_tokens=None,
        variant_id=None,
    )
    before = bookkeeping.dropped_total()

    async def locked(self, batch):
        raise sqlite3.OperationalError("database is locked")

    real = Ledger._write
    monkeypatch.setattr(Ledger, "_write", locked)
    assert await led.flush() == 0
    # A lock retry loses nothing, so it is not a drop (#279 final review I1).
    assert bookkeeping.dropped_total() == before
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=7,
        completion_tokens=0,
        cached_tokens=None,
        variant_id=None,
    )
    monkeypatch.setattr(Ledger, "_write", real)
    await led.flush()
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(2, 12)]


async def test_touch_updates_last_used_once_per_token(db_path):
    led = Ledger(db_path)
    for _ in range(3):
        led.touch("t")
    assert led.pending() is True
    assert await led.flush() == 1  # three touches of one key: one UPDATE
    [(stamp,)] = await _q(db_path, "SELECT last_used_at FROM api_tokens WHERE id = 't'")
    assert stamp is not None

    # Inside the 60 s granularity the stamp is left alone (TokenRepo.touch_last_used).
    old = await _q(db_path, "SELECT datetime('now', '-10 seconds')")
    with sqlite3.connect(db_path) as raw:
        raw.execute("UPDATE api_tokens SET last_used_at = ? WHERE id = 't'", (old[0][0],))
        raw.commit()
    led.touch("t")
    await led.flush()
    assert await _q(db_path, "SELECT last_used_at FROM api_tokens WHERE id = 't'") == [(old[0][0],)]


async def test_variant_usage_is_written_and_descriptor_ensured(db_path):
    variant = variant_of(SimpleNamespace(id="m", served_model_name="m", dtype="auto"))
    led = Ledger(db_path, clock=lambda: 60 * 2000 + 1)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=10,
        completion_tokens=2,
        cached_tokens=None,
        variant_id=variant.id,
        variant=variant,
    )
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=5,
        completion_tokens=3,
        cached_tokens=None,
        variant_id=variant.id,
        variant=variant,
    )
    assert await led.flush() > 0
    assert await _q(
        db_path,
        "SELECT token_id, variant_id, model_id, minute, requests, prompt_tokens, completion_tokens "
        "FROM token_model_usage_minute",
    ) == [("t", variant.id, "m", 2000, 2, 15, 5)]
    assert await _q(db_path, "SELECT id, model_id FROM model_variants") == [(variant.id, "m")]


def _rec(led, model, token="t", prompt=1):
    led.record_request(
        model_id=model,
        token_id=token,
        prompt_tokens=prompt,
        completion_tokens=0,
        cached_tokens=None,
        variant_id=None,
    )


def _raw(db_path, sql, *args):
    with sqlite3.connect(db_path) as raw:
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(sql, args)
        raw.commit()


async def test_deleted_model_drops_only_its_key(db_path):
    _raw(
        db_path,
        "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices) "
        "VALUES ('a', 'a', 'o/r', '[0]')",
    )
    led = Ledger(db_path)
    _rec(led, "a")
    _raw(db_path, "DELETE FROM models WHERE id = 'a'")
    _rec(led, "m", prompt=5)
    _rec(led, "m", prompt=6)
    before = bookkeeping.dropped_total().get("ledger_key_dropped", 0)
    assert await led.flush() > 0
    assert await _q(db_path, "SELECT model_id, requests, prompt_tokens FROM counters") == [
        ("m", 2, 11)
    ]
    assert await _q(db_path, "SELECT model_id, requests FROM model_samples") == [("m", 2)]
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM token_usage_minute") == [(3, 12)]
    # Two keys (the counter and the sample) refused in one flush: one drop.
    assert bookkeeping.dropped_total().get("ledger_key_dropped", 0) == before + 1
    assert led.pending() is False
    assert await led.flush() == 0


async def test_deleted_token_drops_only_its_key(db_path):
    _raw(
        db_path,
        "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
        "VALUES ('gone', 'g', 'vw_g', 'h2', 'inference')",
    )
    led = Ledger(db_path)
    _rec(led, "m", token="gone", prompt=9)
    _raw(db_path, "DELETE FROM api_tokens WHERE id = 'gone'")
    _rec(led, "m", token="t", prompt=5)
    before = bookkeeping.dropped_total().get("ledger_key_dropped", 0)
    assert await led.flush() > 0
    assert await _q(db_path, "SELECT token_id, requests, prompt_tokens FROM counters") == [
        ("t", 1, 5)
    ]
    assert bookkeeping.dropped_total().get("ledger_key_dropped", 0) == before + 1
    assert led.pending() is False


async def test_cancelled_flush_loses_nothing(db_path, monkeypatch):
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    gate, started = asyncio.Event(), asyncio.Event()
    real_execute = aiosqlite.Connection.execute

    def slow_execute(self, sql, *args, **kwargs):
        if sql.startswith("INSERT INTO model_samples"):
            started.set()

            async def _wait():
                await gate.wait()
                return await real_execute(self, sql, *args, **kwargs)

            return _wait()
        return real_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "execute", slow_execute)
    task = asyncio.create_task(led.flush())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(aiosqlite.Connection, "execute", real_execute)
    assert led.pending() is True
    _rec(led, "m", prompt=2)
    await led.flush()
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(2, 7)]


async def test_record_request_is_all_or_nothing(db_path):
    led = Ledger(db_path)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=None,
        completion_tokens=1,
        cached_tokens=None,
        variant_id=None,
    )
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=-1,
        completion_tokens=1,
        cached_tokens=None,
        variant_id=None,
    )
    assert led.pending() is False


async def test_late_failure_is_all_or_nothing_then_exact(db_path, monkeypatch):
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    led.touch("t")
    real_commit = aiosqlite.Connection.commit

    async def failing_commit(self):
        raise RuntimeError("boom")  # every statement already ran

    monkeypatch.setattr(aiosqlite.Connection, "commit", failing_commit)
    before = bookkeeping.dropped_total()
    assert await led.flush() == 0
    # Retried, not lost: no drop is counted until the batch is given up on.
    assert bookkeeping.dropped_total() == before
    monkeypatch.setattr(aiosqlite.Connection, "commit", real_commit)
    assert await _q(db_path, "SELECT * FROM counters") == []
    assert await _q(db_path, "SELECT * FROM model_samples") == []
    assert await _q(db_path, "SELECT last_used_at FROM api_tokens WHERE id = 't'") == [(None,)]
    _rec(led, "m", prompt=2)
    assert await led.flush() > 0
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(2, 7)]


async def test_touch_rewrites_a_stale_stamp(db_path):
    _raw(
        db_path,
        "UPDATE api_tokens SET last_used_at = datetime('now', '-300 seconds') WHERE id = 't'",
    )
    [(old,)] = await _q(db_path, "SELECT last_used_at FROM api_tokens WHERE id = 't'")
    led = Ledger(db_path)
    led.touch("t")
    await led.flush()
    [(new,)] = await _q(db_path, "SELECT last_used_at FROM api_tokens WHERE id = 't'")
    assert new > old


async def test_cancel_during_commit_never_double_counts(db_path, monkeypatch):
    import threading

    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    release, in_commit = threading.Event(), threading.Event()
    real_commit = aiosqlite.Connection.commit

    async def blocked_commit(self):
        def fn():
            in_commit.set()
            release.wait(5)
            self._conn.commit()

        await self._execute(fn)

    monkeypatch.setattr(aiosqlite.Connection, "commit", blocked_commit)
    task = asyncio.create_task(led.flush())
    assert await asyncio.to_thread(in_commit.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(aiosqlite.Connection, "commit", real_commit)
    assert led.pending() is False
    await led.flush()
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(1, 5)]


async def test_bogus_variant_drops_only_that_key(db_path):
    led = Ledger(db_path)
    led.record_request(
        model_id="m",
        token_id="t",
        prompt_tokens=4,
        completion_tokens=0,
        cached_tokens=None,
        variant_id="v1",
        variant=object(),
    )
    before = bookkeeping.dropped_total().get("ledger_key_dropped", 0)
    assert await led.flush() > 0
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(1, 4)]
    assert await _q(db_path, "SELECT requests FROM token_usage_minute") == [(1,)]
    assert bookkeeping.dropped_total().get("ledger_key_dropped", 0) == before + 1
    assert led.pending() is False


async def test_repeated_escaping_failures_drop_the_batch(db_path, monkeypatch):
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)

    async def boom(self, batch):
        raise RuntimeError("poison")

    monkeypatch.setattr(Ledger, "_write", boom)
    before = bookkeeping.dropped_total()
    await led.flush()
    await led.flush()
    assert led.pending() is True
    assert bookkeeping.dropped_total() == before  # retries lose nothing
    await led.flush()
    assert led.pending() is False
    after = bookkeeping.dropped_total()
    assert after.get("ledger_batch_dropped", 0) == before.get("ledger_batch_dropped", 0) + 1
    assert after.get("ledger_flush", 0) == before.get("ledger_flush", 0)


def _lost(site: str) -> int:
    return bookkeeping.snapshot()["sites"].get(site, {}).get("requests_lost") or 0


async def test_batch_drop_exports_requests_lost(db_path, monkeypatch):
    """#294: the requests a dropped batch carried are counted, not just logged."""
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    _rec(led, "m", prompt=6)

    async def boom(self, batch):
        raise RuntimeError("poison")

    monkeypatch.setattr(Ledger, "_write", boom)
    before = _lost("ledger_batch_dropped")
    for _ in range(3):
        await led.flush()
    assert _lost("ledger_batch_dropped") == before + 2


async def test_key_drop_exports_requests_lost(db_path):
    """#294: a refused token key's requests are counted (once per request)."""
    _raw(
        db_path,
        "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
        "VALUES ('gone', 'g', 'vw_g', 'h2', 'inference')",
    )
    led = Ledger(db_path)
    _rec(led, "m", token="gone", prompt=9)
    _rec(led, "m", token="gone", prompt=9)
    _raw(db_path, "DELETE FROM api_tokens WHERE id = 'gone'")
    before = _lost("ledger_key_dropped")
    await led.flush()
    assert _lost("ledger_key_dropped") == before + 2


async def test_non_lock_operational_errors_are_bounded_but_locks_retry_forever(
    db_path, monkeypatch
):
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)

    async def io_error(self, batch):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(Ledger, "_write", io_error)
    await led.flush()
    await led.flush()
    assert led.pending() is True
    await led.flush()
    assert led.pending() is False

    _rec(led, "m", prompt=5)

    async def locked(self, batch):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Ledger, "_write", locked)
    for _ in range(6):
        await led.flush()
    assert led.pending() is True


# --- #279 final review I1: the drop counters mean "data lost" -------------


async def test_lock_retry_counts_no_drop_and_logs_once_a_minute(db_path, monkeypatch, caplog):
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)

    async def locked(self, batch):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Ledger, "_write", locked)
    before = bookkeeping.dropped_total()
    with caplog.at_level("WARNING", logger="app.proxy"):
        for _ in range(3):
            assert await led.flush() == 0
    assert bookkeeping.dropped_total() == before
    retries = [r for r in caplog.records if "flush deferred, database busy; retrying" in r.message]
    assert len(retries) == 1
    assert [r for r in caplog.records if r.levelname == "WARNING"] == retries
    assert led.pending() is True


async def test_key_drops_count_once_per_flush_with_keys_and_requests(db_path, caplog):
    _raw(
        db_path,
        "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices) "
        "VALUES ('a', 'a', 'o/r', '[0]')",
    )
    _raw(
        db_path,
        "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
        "VALUES ('gone', 'g', 'vw_g', 'h2', 'inference')",
    )
    led = Ledger(db_path)
    _rec(led, "a")  # counter (a, t) and sample (a, minute) will be refused
    _rec(led, "m", token="gone")  # counter (m, gone); token_usage_minute has no FK
    _rec(led, "m", token="gone")
    _rec(led, "m", prompt=5)  # survives
    _raw(db_path, "DELETE FROM models WHERE id = 'a'")
    _raw(db_path, "DELETE FROM api_tokens WHERE id = 'gone'")
    before = bookkeeping.dropped_total()
    with caplog.at_level("WARNING", logger="app.proxy"):
        assert await led.flush() > 0
    after = bookkeeping.dropped_total()
    assert after.get("ledger_key_dropped", 0) == before.get("ledger_key_dropped", 0) + 1
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1, [r.message for r in warns]
    # 3 keys refused; 3 requests lost (1 on model a, 2 on the deleted token).
    assert "3 key(s)" in warns[0].message and "3 request(s)" in warns[0].message
    assert await _q(db_path, "SELECT model_id, token_id, requests FROM counters") == [("m", "t", 1)]


async def test_a_usage_only_key_drop_counts_its_requests(db_path, monkeypatch, caplog):
    led = Ledger(db_path)
    for _ in range(4):
        _rec(led, "m", prompt=5)
    # The usage row is refused but its counter row lands: those 4 requests
    # are still lost to the per-key usage numbers and must be counted.
    real_execute = aiosqlite.Connection.execute

    def refuse_usage(self, sql, *args, **kwargs):
        if sql.startswith("INSERT INTO token_usage_minute"):

            async def _fail():
                raise sqlite3.IntegrityError("refused")

            return _fail()
        return real_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "execute", refuse_usage)
    before = bookkeeping.dropped_total().get("ledger_key_dropped", 0)
    with caplog.at_level("WARNING", logger="app.proxy"):
        assert await led.flush() > 0
    monkeypatch.setattr(aiosqlite.Connection, "execute", real_execute)
    assert await _q(db_path, "SELECT requests FROM counters") == [(4,)]
    assert bookkeeping.dropped_total().get("ledger_key_dropped", 0) == before + 1
    [warn] = [r for r in caplog.records if r.levelname == "WARNING"]
    assert "1 key(s)" in warn.message and "4 request(s)" in warn.message


async def test_batch_drop_warning_names_the_requests_lost(db_path, monkeypatch, caplog):
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    _rec(led, "m", token=None, prompt=5)

    async def boom(self, batch):
        raise RuntimeError("poison")

    monkeypatch.setattr(Ledger, "_write", boom)
    with caplog.at_level("WARNING", logger="app.proxy"):
        for _ in range(3):
            await led.flush()
    [drop] = [r for r in caplog.records if "dropping a batch" in r.message]
    assert "2 request(s)" in drop.message


# --- #279 final review M1: a cancel during a failing commit is not swallowed --


async def test_cancel_during_a_failing_commit_ends_the_loop_cancelled(db_path, monkeypatch):
    """The reviewer's probe: the commit the cancel waits on then raises. The
    cancel must still end the task (shutdown would otherwise hang), and the
    uncommitted batch is kept."""
    led = Ledger(db_path)
    _rec(led, "m", prompt=3)
    started = asyncio.Event()

    async def bad_commit(self):
        started.set()
        await asyncio.sleep(0.2)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(aiosqlite.Connection, "commit", bad_commit)
    task = asyncio.create_task(led.run_forever(interval=0.01))
    await started.wait()
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=1.0)
    if not done:  # pragma: no cover -- the bug: stop the runaway loop
        monkeypatch.undo()
        task.cancel()
        with pytest.raises(BaseException):  # noqa: B017,PT011
            await task
    assert done, "the cancel was swallowed: run_forever kept running"
    assert task.cancelled()
    monkeypatch.undo()
    assert led.pending() is True  # never committed, so kept
    await led.flush()
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(1, 3)]


async def test_a_second_cancel_during_the_commit_cannot_double_count(db_path, monkeypatch):
    import threading

    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    release, in_commit = threading.Event(), threading.Event()
    real_commit = aiosqlite.Connection.commit

    async def blocked_commit(self):
        def fn():
            in_commit.set()
            release.wait(5)
            self._conn.commit()

        await self._execute(fn)

    monkeypatch.setattr(aiosqlite.Connection, "commit", blocked_commit)
    task = asyncio.create_task(led.flush())
    assert await asyncio.to_thread(in_commit.wait, 5)
    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()  # a second cancel while the commit is still running
    await asyncio.sleep(0.02)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(aiosqlite.Connection, "commit", real_commit)
    assert led.pending() is False
    await led.flush()
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(1, 5)]


async def test_run_forever_cancel_does_a_last_flush_and_ends_cancelled(db_path):
    led = Ledger(db_path)
    task = asyncio.create_task(led.run_forever(interval=3600))
    await asyncio.sleep(0)  # into its sleep
    _rec(led, "m", prompt=7)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert task.cancelled()
    assert led.pending() is False
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(1, 7)]


# --- #295: a wedged commit cannot hold a cancelled flush (and shutdown) forever --


async def test_a_wedged_commit_is_given_up_on_without_merging_back(db_path, monkeypatch, caplog):
    """The cancel branch waits for the commit, but only so long. Past the cap
    the outcome is unknown: the batch is NOT merged back, because a commit
    that did land would then be counted twice."""
    import threading

    from app.proxy import ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "LEDGER_COMMIT_WAIT_S", 0.2)
    led = Ledger(db_path)
    _rec(led, "m", prompt=5)
    release, in_commit = threading.Event(), threading.Event()
    real_commit = aiosqlite.Connection.commit

    async def wedged_commit(self):
        def fn():
            in_commit.set()
            release.wait(10)
            self._conn.commit()

        await self._execute(fn)

    monkeypatch.setattr(aiosqlite.Connection, "commit", wedged_commit)
    task = asyncio.create_task(led.flush())
    assert await asyncio.to_thread(in_commit.wait, 5)
    task.cancel()
    try:
        # Given up on well before the commit returns (it is still blocked).
        await wait_until_async(
            lambda: "commit outcome unknown" in caplog.text,
            what="the capped commit wait to expire",
            timeout_s=3,
        )
        assert not release.is_set()
        assert led.pending() is False  # not merged back
    finally:
        release.set()  # the commit lands after all; the close is acknowledged
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    monkeypatch.setattr(aiosqlite.Connection, "commit", real_commit)
    assert led.pending() is False
    await led.flush()
    # Exactly once: the landed commit, never a merged-back second copy.
    assert await _q(db_path, "SELECT requests, prompt_tokens FROM counters") == [(1, 5)]
