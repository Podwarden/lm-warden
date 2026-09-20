"""The admin-audit writer (#258): a bounded in-process queue in front of the
``admin_audit`` table, drained in batches by one background flusher.

The middleware used to open its own connection and commit one row per request,
on the response path. This covers what replaced it: rows land on a flush, a
burst lands whole, one transaction carries a whole batch, a full queue drops
and COUNTS (with one warning per flush) rather than growing or going quiet,
and cancelling the flusher -- what the lifespan does at shutdown -- writes
what is still queued.
"""

import asyncio
import logging
import shutil
import sqlite3
from pathlib import Path

import pytest

from app.auth.admin_audit import AdminAuditWriter
from app.db.repos.admin_audit import AdminAuditRepo


@pytest.fixture
def db_path(tmp_data_dir: Path, migrated_db_template: Path) -> Path:
    path = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, path)
    return path


def _paths(path: Path) -> list[str]:
    with sqlite3.connect(path) as db:
        return [r[0] for r in db.execute("SELECT path FROM admin_audit ORDER BY id")]


def _row(path: Path) -> tuple:
    with sqlite3.connect(path) as db:
        return db.execute(
            "SELECT ts, token_id, username, method, path, status, duration_ms, "
            "client_ip, peer_ip FROM admin_audit"
        ).fetchall()[0]


def _record(writer: AdminAuditWriter, n: int = 1, *, ts: float = 1.0) -> None:
    for i in range(n):
        writer.record(
            ts=ts + i,
            token_id="t1",
            username="admin",
            method="GET",
            path=f"/api/p{i}",
            status=200,
            duration_ms=i,
            client_ip="198.51.100.7",
            peer_ip="127.0.0.1",
        )


async def test_a_flush_writes_every_column_it_was_given(db_path: Path) -> None:
    writer = AdminAuditWriter(db_path)
    _record(writer, ts=1700.5)
    assert writer.pending() == 1
    assert await writer.flush() == 1
    assert _row(db_path) == (
        1700.5, "t1", "admin", "GET", "/api/p0", 200, 0, "198.51.100.7", "127.0.0.1",
    )
    assert (writer.pending(), writer.written, writer.dropped) == (0, 1, 0)


async def test_a_burst_larger_than_one_batch_lands_whole(db_path: Path) -> None:
    writer = AdminAuditWriter(db_path, batch_max=10)
    _record(writer, 25)
    assert await writer.flush() == 25
    assert len(_paths(db_path)) == 25


async def test_a_batch_is_one_transaction(db_path: Path, monkeypatch) -> None:
    """``batch_max`` rows per ``executemany``, not one INSERT per row."""
    sizes: list[int] = []
    real = AdminAuditRepo.record_many

    async def spy(self: AdminAuditRepo, rows) -> None:
        rows = list(rows)
        sizes.append(len(rows))
        await real(self, rows)

    monkeypatch.setattr(AdminAuditRepo, "record_many", spy)
    writer = AdminAuditWriter(db_path, batch_max=10)
    _record(writer, 25)
    assert await writer.flush() == 25
    assert sizes == [10, 10, 5]


async def test_an_empty_flush_touches_nothing(db_path: Path, monkeypatch) -> None:
    async def boom(*_: object, **__: object) -> None:
        raise AssertionError("an empty flush must not open the database")

    monkeypatch.setattr("app.auth.admin_audit.open_db", boom)
    assert await AdminAuditWriter(db_path).flush() == 0


async def test_a_full_queue_drops_and_counts(db_path: Path) -> None:
    writer = AdminAuditWriter(db_path, queue_max=2)
    _record(writer, 5)
    # Never unbounded: the queue holds its cap, the rest are counted.
    assert (writer.pending(), writer.dropped) == (2, 3)
    assert await writer.flush() == 2
    assert _paths(db_path) == ["/api/p0", "/api/p1"]


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


async def test_drops_are_reported_once_per_window_from_both_sides(
    db_path: Path, caplog
) -> None:
    """Drops are never silent, and never logged per row.

    Reported from the ENQUEUE side as well as from ``flush``: the flush side
    alone meant that a dead flusher -- the one state in which the queue
    reliably fills -- reported nothing at all, because ``_log_drops`` was only
    reachable from ``flush`` (I3).
    """
    writer = AdminAuditWriter(db_path, queue_max=2)
    with caplog.at_level(logging.WARNING, logger="app.auth.admin_audit"):
        _record(writer, 5)
        # One warning for the burst of three drops, not three.
        assert len(_warnings(caplog)) == 1, _warnings(caplog)
        caplog.clear()
        # The flush reports what the throttle held back, and the running total.
        await writer.flush()
        rest = _warnings(caplog)
        assert len(rest) == 1 and "3" in rest[0], rest
        # Nothing new was dropped, so the next flush is silent.
        caplog.clear()
        _record(writer, 1)
        await writer.flush()
        assert _warnings(caplog) == []


async def test_drops_are_reported_even_if_nothing_ever_flushes(
    db_path: Path, caplog
) -> None:
    """I3: the queue filling up while the flusher is dead must show in the log.

    ``drop_log_interval_s`` is retuned to 0 so every drop is reportable; the
    point is that the reports come from ``record`` with no ``flush`` in sight.
    """
    writer = AdminAuditWriter(db_path, queue_max=2)
    writer.drop_log_interval_s = 0.0
    with caplog.at_level(logging.WARNING, logger="app.auth.admin_audit"):
        _record(writer, 5)
    said = _warnings(caplog)
    assert said, "a full queue must be reported without waiting for a flush"
    assert "3" in said[-1], said


async def test_record_never_raises(db_path: Path) -> None:
    """The response path must never see the audit fail, whatever it is handed."""
    writer = AdminAuditWriter(db_path, queue_max=1)
    for _ in range(3):
        writer.record(
            ts=float("nan"), token_id="t1", username="admin", method="GET",
            path="/api/x", status=200, duration_ms=0, client_ip=None, peer_ip=None,
        )


async def test_a_failed_write_is_swallowed_and_the_writer_keeps_going(
    db_path: Path, monkeypatch, caplog
) -> None:
    calls: list[int] = []
    real = AdminAuditRepo.record_many

    async def flaky(self: AdminAuditRepo, rows) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("disk full")
        await real(self, rows)

    monkeypatch.setattr(AdminAuditRepo, "record_many", flaky)
    writer = AdminAuditWriter(db_path)
    _record(writer, 1)
    with caplog.at_level(logging.ERROR, logger="app.auth.admin_audit"):
        assert await writer.flush() == 0
        # M2: a write failure is a gap in the trail like any other, so it is
        # counted and logged rather than only logged. Two drop causes, one
        # counter -- otherwise `dropped` understates a real gap.
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, [r.getMessage() for r in errors]
    assert writer.dropped == 1
    assert _paths(db_path) == []
    _record(writer, 1)
    assert await writer.flush() == 1


async def test_the_flusher_writes_on_its_interval(db_path: Path) -> None:
    writer = AdminAuditWriter(db_path, flush_interval_s=0.02)
    task = asyncio.create_task(writer.run_forever())
    try:
        _record(writer, 3)
        for _ in range(200):
            if writer.written == 3:
                break
            await asyncio.sleep(0.02)
        assert len(_paths(db_path)) == 3
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_full_batch_does_not_wait_out_the_interval(db_path: Path) -> None:
    writer = AdminAuditWriter(db_path, batch_max=5, flush_interval_s=30.0)
    task = asyncio.create_task(writer.run_forever())
    try:
        _record(writer, 5)
        for _ in range(200):
            if writer.written == 5:
                break
            await asyncio.sleep(0.02)
        assert len(_paths(db_path)) == 5, "a full batch must not wait for the interval"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancelling_the_flusher_writes_what_is_queued(db_path: Path) -> None:
    """What the lifespan does at shutdown: nothing queued is lost."""
    writer = AdminAuditWriter(db_path, flush_interval_s=3600.0)
    task = asyncio.create_task(writer.run_forever())
    await asyncio.sleep(0)  # let it reach its wait
    _record(writer, 4)
    assert _paths(db_path) == []
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert len(_paths(db_path)) == 4


async def test_a_batch_in_flight_when_the_flusher_is_cancelled_is_not_lost(
    db_path: Path, monkeypatch
) -> None:
    """M1: "a clean shutdown loses nothing" has to hold for the batch already
    taken out of the queue, not only for what is still in it.

    ``flush`` pops up to ``batch_max`` rows and THEN awaits the write, so a
    shutdown cancel landing inside that write used to drop them: the final
    flush only drained what was still queued.
    """
    started = asyncio.Event()
    hang = asyncio.Event()  # never set -- the cancel lands in the write
    real = AdminAuditRepo.record_many

    async def stuck(self: AdminAuditRepo, rows) -> None:
        started.set()
        await hang.wait()

    monkeypatch.setattr(AdminAuditRepo, "record_many", stuck)
    writer = AdminAuditWriter(db_path, flush_interval_s=0.01)
    task = asyncio.create_task(writer.run_forever())
    try:
        _record(writer, 3)
        await asyncio.wait_for(started.wait(), 5)
        assert _paths(db_path) == []
        # The volume recovers; the shutdown flush must still have the rows.
        monkeypatch.setattr(AdminAuditRepo, "record_many", real)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert len(_paths(db_path)) == 3
    finally:
        hang.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_the_flusher_survives_an_unexpected_exception(
    db_path: Path, caplog
) -> None:
    """I3: an exception escaping the loop body used to end the task, and
    nothing observed it -- auditing stopped silently and, because
    ``_log_drops`` was only reachable from ``flush``, the queue then filled to
    the cap without a single log line.
    """
    writer = AdminAuditWriter(db_path, flush_interval_s=0.01)
    real = writer.flush
    calls: list[int] = []

    async def flaky() -> int:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("something nobody predicted")
        return await real()

    writer.flush = flaky  # type: ignore[method-assign]
    task = asyncio.create_task(writer.run_forever())
    try:
        with caplog.at_level(logging.ERROR, logger="app.auth.admin_audit"):
            _record(writer, 2)
            for _ in range(500):
                if writer.written == 2:
                    break
                await asyncio.sleep(0.01)
        assert len(_paths(db_path)) == 2, "the flusher died and took the audit with it"
        assert not task.done()
        assert [r for r in caplog.records if r.levelno >= logging.ERROR], (
            "a flusher that swallows the failure silently is no better"
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_persistently_broken_flush_does_not_spin_hot(db_path: Path) -> None:
    """I3: surviving the exception must not turn the flusher into a busy loop.

    With rows permanently queued and every flush raising, the loop must pace
    itself at roughly ``flush_interval_s`` rather than retrying as fast as the
    event loop will let it.
    """
    writer = AdminAuditWriter(db_path, flush_interval_s=0.05)
    attempts: list[int] = []

    async def always_fails() -> int:
        attempts.append(1)
        raise RuntimeError("the volume is gone")

    writer.flush = always_fails  # type: ignore[method-assign]
    task = asyncio.create_task(writer.run_forever())
    try:
        _record(writer, 1)
        await asyncio.sleep(0.3)
        # ~6 intervals of headroom; a hot loop runs to the hundreds.
        assert 1 <= len(attempts) <= 20, len(attempts)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
