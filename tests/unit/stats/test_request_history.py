"""app/stats/request_history -- the store, its queries, its pruning and its
statistics, against a real SQLite file.

Fixture names are neutral on purpose (model-a, key-a, 10.0.0.x): the repo was
swept for vendor and checkpoint names in test data.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.config import Settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats import request_history as rh
from app.stats.request_history import RequestHistoryStore
from tests.conftest import wait_until_async


def _record(i: int, **over):
    base = {
        "id": f"req-{i}",
        "model": "model-a",
        "model_id": "id-model-a",
        "token_name": "key-a",
        "client_ip": "10.0.0.1",
        "prompt_tokens": 100 + i,
        "completion_tokens": 50,
        "duration_s": 2.0 + i,
        "ttft_s": 0.5,
        "finish_reason": "stop",
        "orphan": False,
        "started_iso": "2026-09-05T18:59:00Z",
    }
    base.update(over)
    return base


async def _migrated(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


def _count(db_path) -> int:
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT COUNT(*) FROM request_history").fetchone()[0]


# ---- write side -----------------------------------------------------------


async def test_record_then_flush_lands_a_row(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    assert store.record(_record(1), finished_at=1000.0) is True
    assert store.pending() == 1
    assert await store.flush() == 1
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT id, finished_at, model_id, model, token_name, prompt_tokens, "
            "duration_s, ttft_s, finish_reason, orphan FROM request_history"
        ).fetchone()
    assert row == ("req-1", 1000.0, "id-model-a", "model-a", "key-a", 101, 3.0, 0.5, "stop", 0)
    assert store.written == 1


async def test_cache_columns_round_trip(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, cached_tokens=640, cache_est_tokens=700), finished_at=1000.0)
    store.record(_record(2), finished_at=1001.0)  # not measured, not estimated
    assert await store.flush() == 2
    async with open_db(db_path) as db:
        rows = {r["id"]: r for r in await rh.query_last(db, n=10, model_ids=None)}
    assert (rows["req-1"]["cached_tokens"], rows["req-1"]["cache_est_tokens"]) == (640, 700)
    assert (rows["req-2"]["cached_tokens"], rows["req-2"]["cache_est_tokens"]) == (None, None)


async def test_session_source_columns_round_trip(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(
        _record(1, session_source="x_session_id", parent_session_id="ses_parentparent01"),
        finished_at=1000.0,
    )
    store.record(_record(2), finished_at=1001.0)
    assert await store.flush() == 2
    async with open_db(db_path) as db:
        rows = {r["id"]: r for r in await rh.query_last(db, n=10, model_ids=None)}
    assert rows["req-1"]["session_source"] == "x_session_id"
    assert rows["req-1"]["parent_session_id"] == "ses_parentparent01"
    assert rows["req-2"]["session_source"] is None


async def test_forest_columns_round_trip(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(
        _record(
            1,
            session_key="a1b2c3d4e5f60718",
            parent_session_key=None,
            turn_index=3,
            batch_id="farm-A27",
            tools_out=[["read", "0a1b2c3d"]],
            tools_in='[["shell",120,false]]',
        )
    )
    await store.flush()
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT session_key, turn_index, batch_id, tools_out, tools_in FROM request_history"
        ).fetchone()
    assert row == (
        "a1b2c3d4e5f60718",
        3,
        "farm-A27",
        '[["read","0a1b2c3d"]]',
        '[["shell",120,false]]',
    )


async def test_forest_columns_without_values_round_trip_as_nulls(tmp_path):
    """Records without forest keys are written with all six forest columns NULL."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=1000.0)
    await store.flush()
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT session_key, parent_session_key, turn_index, batch_id, tools_out, tools_in FROM request_history"
        ).fetchone()
    assert row == (None, None, None, None, None, None)


async def test_forest_bad_values_cost_one_row_not_the_batch(tmp_path):
    """Bad forest column values (list, dict, set) are coerced to NULL,
    allowing good records in the same batch to succeed."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    # Bad values: session_key is a list, batch_id is a dict, turn_index is unparseable,
    # tools_out is a set (not JSON-serializable)
    store.record(
        _record(
            1,
            session_key=["x"],
            batch_id={"a": 1},
            turn_index="nope",
            tools_out={1, 2},
        ),
        finished_at=1000.0,
    )
    store.record(_record(2), finished_at=1001.0)
    assert await store.flush() == 2
    with sqlite3.connect(db_path) as db:
        rows = {
            r["id"]: r
            for r in [
                dict(
                    zip(
                        ["id", "session_key", "batch_id", "turn_index", "tools_out"],
                        row,
                        strict=True,
                    )
                )
                for row in db.execute(
                    "SELECT id, session_key, batch_id, turn_index, tools_out FROM request_history ORDER BY id"
                ).fetchall()
            ]
        }
    # Bad values become NULL; good record is unaffected
    assert rows["req-1"]["session_key"] is None
    assert rows["req-1"]["batch_id"] is None
    assert rows["req-1"]["turn_index"] is None
    assert rows["req-1"]["tools_out"] is None
    assert rows["req-2"]["session_key"] is None


async def test_record_never_raises_and_never_blocks(tmp_path):
    """The proxy calls this in the streaming `finally`, before the slot is
    released. A full queue drops and counts; garbage is refused quietly."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, queue_max=2)
    assert store.record(_record(1)) is True
    assert store.record(_record(2)) is True
    assert store.record(_record(3)) is False  # full: dropped, not blocked
    assert store.dropped == 1
    assert store.record({"no": "id"}) is False  # unusable: refused
    assert store.pending() == 2


async def test_a_bad_field_costs_one_value_not_the_batch(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, prompt_tokens="not a number", ttft_s="?"), finished_at=1.0)
    store.record(_record(2), finished_at=2.0)
    assert await store.flush() == 2
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT id, prompt_tokens, ttft_s FROM request_history ORDER BY id"
        ).fetchall()
    assert rows == [("req-1", 0, None), ("req-2", 102, 0.5)]


async def test_a_duplicate_id_is_ignored_not_an_error(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=1.0)
    store.record(_record(1), finished_at=2.0)
    assert await store.flush() == 2
    assert _count(db_path) == 1


async def test_a_failed_write_drops_the_batch_and_keeps_going(tmp_path):
    """No table (an unmigrated file) is the simplest broken volume."""
    db_path = tmp_path / "empty.db"
    store = RequestHistoryStore(db_path)
    store.record(_record(1))
    assert await store.flush() == 0
    assert store.pending() == 0
    # Still usable afterwards.
    assert store.record(_record(2)) is True


async def test_run_forever_batches_and_flushes_on_cancel(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, flush_interval_s=0.05)
    task = asyncio.create_task(store.run_forever())
    for i in range(5):
        store.record(_record(i), finished_at=float(i))
    await wait_until_async(
        lambda: _count(db_path) == 5, what="run_forever to flush the first batch"
    )
    assert _count(db_path) == 5
    # Queued after the last batch, written by the cancellation flush.
    store.record(_record(99), finished_at=99.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _count(db_path) == 6


# ---- #295: shutdown is bounded under a held write lock --------------------


def _hold_write_lock(db_path) -> sqlite3.Connection:
    lock = sqlite3.connect(db_path, isolation_level=None)
    lock.execute("BEGIN IMMEDIATE")
    return lock


async def test_cancel_during_the_batching_sleep_still_writes_the_first_record(tmp_path):
    """The record ``run_forever`` took off the queue before its batching sleep
    is no longer queued; a cancel there must not lose it."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, flush_interval_s=3600)
    task = asyncio.create_task(store.run_forever())
    store.record(_record(1), finished_at=1.0)
    await wait_until_async(lambda: store.pending() == 0, what="the writer to take the record")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _count(db_path) == 1


async def test_final_flush_under_a_held_write_lock_is_bounded(tmp_path, caplog):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, flush_interval_s=3600)
    task = asyncio.create_task(store.run_forever())
    for i in range(3):
        store.record(_record(i), finished_at=float(i))
    lock = _hold_write_lock(db_path)
    try:
        await asyncio.sleep(0.05)
        t0 = asyncio.get_running_loop().time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 15)
        elapsed = asyncio.get_running_loop().time() - t0
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    assert elapsed < rh.FINAL_FLUSH_TIMEOUT_S + 1.5, elapsed
    # Not silent: the rows the shutdown could not write are named.
    assert "3 record(s)" in caplog.text
    assert _count(db_path) == 0


async def test_a_write_cancelled_while_waiting_on_the_lock_is_bounded(tmp_path, monkeypatch):
    """The worst case: the writer is INSIDE a write that waits on the lock
    when shutdown cancels it. The connection's close waits for that statement,
    so its busy wait must be short, not open_db's 30 s."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, flush_interval_s=0.01)
    task = asyncio.create_task(store.run_forever())
    lock = _hold_write_lock(db_path)
    try:
        store.record(_record(1), finished_at=1.0)
        await asyncio.sleep(0.3)  # into the write, waiting on the lock
        store.record(_record(2), finished_at=2.0)
        t0 = asyncio.get_running_loop().time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 20)
        elapsed = asyncio.get_running_loop().time() - t0
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    budget = rh.HISTORY_BUSY_TIMEOUT_MS / 1000 + rh.FINAL_FLUSH_TIMEOUT_S
    assert elapsed < budget + 1.5, elapsed


async def test_lock_contention_requeues_rather_than_drops(tmp_path, monkeypatch):
    """With a short busy wait a held lock fails a write sooner, so a busy
    database is a retry (nothing lost), like the ledger's, not a drop."""
    monkeypatch.setattr(rh, "HISTORY_BUSY_TIMEOUT_MS", 50)
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path, flush_interval_s=0.01)
    task = asyncio.create_task(store.run_forever())
    lock = _hold_write_lock(db_path)
    try:
        store.record(_record(1), finished_at=1.0)
        await asyncio.sleep(0.5)  # several attempts, each refused
    finally:
        lock.execute("ROLLBACK")
        lock.close()
    try:
        await wait_until_async(lambda: _count(db_path) == 1, what="the retried row to land")
        assert store.dropped == 0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---- reads ----------------------------------------------------------------


async def _seed(db_path, n: int, *, model_id="id-model-a", start=1000.0, step=1.0):
    store = RequestHistoryStore(db_path)
    for i in range(n):
        store.record(
            _record(i, id=f"{model_id}-{i}", model_id=model_id, model=model_id.removeprefix("id-")),
            finished_at=start + i * step,
        )
    assert await store.flush() == n


async def test_query_window_is_newest_first_and_bounded_by_since(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 10)  # finished_at 1000..1009
    async with open_db(db_path) as db:
        out = await rh.query_window(db, since=1005.0, model_ids=None, limit=100)
    assert out.total == 5
    assert out.stride == 1
    assert [r["finished_at"] for r in out.rows] == [1009.0, 1008.0, 1007.0, 1006.0, 1005.0]
    assert out.rows[0]["orphan"] is False


async def test_query_window_strides_rather_than_truncating(tmp_path):
    """A chart with a time axis must keep its span covered. The newest N
    would leave the left of the chart empty under a heading claiming the
    whole window."""
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 100)  # 1000..1099
    async with open_db(db_path) as db:
        out = await rh.query_window(db, since=0.0, model_ids=None, limit=10)
    assert out.total == 100
    assert out.stride == 10
    assert len(out.rows) == 10
    at = [r["finished_at"] for r in out.rows]
    assert at[0] == 1099.0  # newest kept
    assert at[-1] == 1009.0  # reaches the far end of the window
    assert all(a - b == 10.0 for a, b in zip(at, at[1:], strict=False))


async def test_query_window_scopes_by_model_row_id(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 3, model_id="id-model-a")
    await _seed(db_path, 2, model_id="id-model-b", start=2000.0)
    async with open_db(db_path) as db:
        a = await rh.query_window(db, since=0.0, model_ids=["id-model-a"], limit=10)
        none = await rh.query_window(db, since=0.0, model_ids=[], limit=10)
        both = await rh.query_window(db, since=0.0, model_ids=None, limit=10)
    assert a.total == 3 and {r["model"] for r in a.rows} == {"model-a"}
    # An empty selection is NOT everything -- the widening every stats route
    # guards against.
    assert none.total == 0 and none.rows == []
    assert both.total == 5


async def test_query_last_ignores_time(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 10)
    async with open_db(db_path) as db:
        rows = await rh.query_last(db, n=3, model_ids=None)
    assert [r["finished_at"] for r in rows] == [1009.0, 1008.0, 1007.0]


async def test_v2_readers_skip_the_forest_columns(tmp_path):
    """The forest's JSON columns are written but not read back by the v2 readers."""
    assert rh._COLUMNS == rh._READ_COLUMNS + rh._FOREST_COLUMNS
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, tools_in=[["shell", 1, False]], session_key="k"), finished_at=1000.0)
    assert await store.flush() == 1
    async with open_db(db_path) as db:
        (last,) = await rh.query_last(db, n=1, model_ids=None)
        (win,) = (await rh.query_window(db, since=0, limit=10, model_ids=None)).rows
    for row in (last, win):
        assert row["id"] == "req-1"
        assert not set(rh._FOREST_COLUMNS) & set(row)


async def test_earliest_is_store_wide_and_none_when_empty(tmp_path):
    db_path = await _migrated(tmp_path)
    async with open_db(db_path) as db:
        assert await rh.earliest_finished_at(db) is None
    await _seed(db_path, 3, start=500.0)
    async with open_db(db_path) as db:
        assert await rh.earliest_finished_at(db) == 500.0


# ---- pruning --------------------------------------------------------------


async def test_prune_by_age_then_by_count(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 20)  # 1000..1019
    async with open_db(db_path) as db:
        out = await rh.prune(db, cutoff=1005.0, max_rows=5)
        await db.commit()
    assert out == {"by_age": 5, "by_count": 10, "newest_deleted": 1014.0}
    with sqlite3.connect(db_path) as db:
        left = [
            r[0] for r in db.execute("SELECT finished_at FROM request_history ORDER BY finished_at")
        ]
    # The five NEWEST survive the cap.
    assert left == [1015.0, 1016.0, 1017.0, 1018.0, 1019.0]


async def test_prune_with_no_cap_only_ages(tmp_path):
    db_path = await _migrated(tmp_path)
    await _seed(db_path, 5)
    async with open_db(db_path) as db:
        out = await rh.prune(db, cutoff=1002.0, max_rows=0)
        await db.commit()
    assert out == {"by_age": 2, "by_count": 0, "newest_deleted": 1001.0}
    assert _count(db_path) == 3


async def test_the_stats_pruner_prunes_history_with_the_configured_retention(tmp_path):
    from app.runtime.stats_pruner import prune_once

    db_path = await _migrated(tmp_path)
    import time

    now = time.time()
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=now - 40 * 86400)  # older than 30d
    store.record(_record(2), finished_at=now - 10)
    await store.flush()

    class S:
        pass

    s = S()
    s.db_path = db_path
    s.request_history_retention_days = 30
    s.request_history_max_rows = Settings.request_history_max_rows
    out = await prune_once(s)
    assert out["request_history"] == 1
    assert _count(db_path) == 1


# ---- statistics (pure) ----------------------------------------------------


def test_quantile_is_exact_and_interpolated():
    assert rh.quantile([], 0.5) is None
    assert rh.quantile([3.0], 0.99) == 3.0
    assert rh.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert rh.quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert rh.quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_histogram_is_cumulative_with_a_null_inf_bucket():
    h = rh.histogram([0.05, 0.5, 5.0, 1000.0], edges=(0.1, 1.0, 10.0))
    assert h["le"] == [0.1, 1.0, 10.0, None]
    assert h["counts"] == [1, 2, 3, 4]
    assert h["count"] == 4
    assert h["sum"] == 1005.55


def test_itl_mean_is_per_request_and_refuses_what_it_cannot_compute():
    ok = {"ttft_s": 1.0, "duration_s": 3.0, "completion_tokens": 5}
    assert rh.itl_mean_of(ok) == 0.5  # 2s of decode over 4 gaps
    assert rh.itl_mean_of({**ok, "ttft_s": None}) is None  # no first token
    assert rh.itl_mean_of({**ok, "completion_tokens": 1}) is None  # no gaps
    assert rh.itl_mean_of({**ok, "duration_s": 1.0}) is None  # no decode phase


def test_latency_summary_shapes_the_three_distributions():
    rows = [
        {"finished_at": 10.0, "ttft_s": 0.2, "duration_s": 2.2, "completion_tokens": 3},
        {"finished_at": 20.0, "ttft_s": None, "duration_s": 5.0, "completion_tokens": 0},
        {"finished_at": 30.0, "ttft_s": 0.4, "duration_s": 0.4, "completion_tokens": 10},
    ]
    s = rh.latency_summary(rows)
    assert s["count"] == 3
    assert s["span_s"] == 20.0
    assert s["oldest_epoch"] == 10.0 and s["newest_epoch"] == 30.0
    # TTFT: only the two that saw a first token.
    assert s["ttft"]["count"] == 2
    assert s["ttft"]["p50"] == pytest.approx(0.3)
    # Duration: every request.
    assert s["duration"]["count"] == 3
    assert s["duration"]["p99"] == pytest.approx(4.944)
    # ITL: only the first has a decode phase with gaps in it.
    assert s["itl"]["count"] == 1
    assert s["itl"]["mean"] == pytest.approx(1.0)
    assert s["itl"]["buckets"]["le"][-1] is None


def test_latency_summary_of_nothing_is_empty_not_zero():
    s = rh.latency_summary([])
    assert s["count"] == 0
    assert s["span_s"] is None
    for k in ("ttft", "itl", "duration"):
        assert s[k]["count"] == 0
        assert s[k]["p50"] is None
        assert s[k]["mean"] is None


# ---- token-rate statistics (prefill / generation tok-per-second) ------------


def test_prefill_tps_is_prompt_tokens_over_ttft():
    ok = {"prompt_tokens": 100, "ttft_s": 0.5}
    assert rh.prefill_tps_of(ok) == 200.0
    assert rh.prefill_tps_of({**ok, "ttft_s": None}) is None  # never saw a first token
    assert rh.prefill_tps_of({**ok, "ttft_s": 0.0}) is None  # no measurable prefill
    assert rh.prefill_tps_of({**ok, "prompt_tokens": 0}) is None  # nothing to prefill


def test_generation_tps_is_the_reciprocal_of_the_per_request_mean_itl():
    ok = {"ttft_s": 1.0, "duration_s": 3.0, "completion_tokens": 5}
    assert rh.gen_tps_of(ok) == 2.0  # 4 gaps in 2s of decode
    # Refuses exactly what itl_mean_of refuses, for the same reasons.
    assert rh.gen_tps_of({**ok, "ttft_s": None}) is None
    assert rh.gen_tps_of({**ok, "completion_tokens": 1}) is None
    assert rh.gen_tps_of({**ok, "duration_s": 1.0}) is None


def test_mode_reports_the_cluster_not_the_mean_and_not_the_outlier():
    # Four readings within 5% of each other, one far outlier. The mean (121)
    # describes no observed request; the mode has to land on the cluster.
    m = rh.mode_relative([100.0, 101.0, 102.0, 103.0, 200.0])
    assert m is not None
    assert 100.0 <= m <= 103.0


def test_mode_breaks_a_tie_towards_the_slower_bin():
    # Two equally populated clusters. Reporting the faster one would flatter
    # the deployment; the conservative read is the lower bin.
    m = rh.mode_relative([100.0, 101.0, 102.0, 200.0, 201.0, 202.0])
    assert m is not None
    assert m < 150.0


def test_mode_is_none_when_no_value_actually_repeats():
    # A flat spread has no modal value, and inventing one from a bin of size
    # 1 would dress up noise as a typical reading.
    assert rh.mode_relative([10.0, 20.0, 30.0, 40.0, 50.0]) is None


def test_mode_is_none_below_the_minimum_sample_count():
    assert rh.mode_relative([100.0, 101.0, 102.0, 103.0]) is None


def test_mode_counts_idle_as_its_own_bin_rather_than_dropping_it():
    # Wall-clock basis: a deployment that is idle most minutes HAS a modal
    # throughput, and it is zero. Dropping the zeros would report the rate of
    # the busy minutes while claiming to describe every minute.
    assert rh.mode_relative([0.0, 0.0, 0.0, 50.0, 60.0, 70.0]) == 0.0


def test_rate_summary_carries_average_max_mode_and_the_sample_count():
    s = rh.rate_summary([100.0, 101.0, 102.0, 103.0, 200.0])
    assert s["count"] == 5
    assert s["avg"] == pytest.approx(121.2)
    assert s["max"] == 200.0
    assert 100.0 <= s["mode"] <= 103.0


def test_rate_summary_of_nothing_is_empty_not_zero():
    s = rh.rate_summary([])
    assert s == {"count": 0, "avg": None, "max": None, "mode": None}


async def test_record_writes_the_token_id(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    assert store.record(_record(1, token_id="tok-a"), finished_at=1000.0) is True
    assert store.record(_record(2), finished_at=1001.0) is True  # no token_id key
    assert await store.flush() == 2
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT id, token_name, token_id FROM request_history ORDER BY id"
        ).fetchall()
    assert rows == [("req-1", "key-a", "tok-a"), ("req-2", "key-a", None)]


# ---- per-token timings and the sampling guard (token details page) -----------


def _token_rows(db_path, n: int, *, token_id="tok-a", start=6000.0, span=3600.0):
    """``n`` rows for ``token_id`` spread evenly over ``[start, start + span)``."""
    step = span / n
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "INSERT INTO request_history(id, finished_at, model_id, model, token_id, "
            "queued_s, ttft_s, duration_s, started_iso) "
            "VALUES (?, ?, 'id-model-a', 'model-a', ?, 0.1, 0.5, 2.0, 'x')",
            [(f"{token_id}-{i:06d}", start + i * step, token_id) for i in range(n)],
        )
        db.commit()


async def test_token_timings_filter_by_token_and_window(tmp_path):
    db_path = await _migrated(tmp_path)
    _token_rows(db_path, 10, token_id="tok-a", start=6000.0, span=600.0)
    _token_rows(db_path, 10, token_id="tok-b", start=6000.0, span=600.0)
    async with open_db(db_path) as db:
        got = await rh.query_token_timings(db, token_ids=["tok-a"], since=6000.0, until=6300.0)
    assert (got.total, got.stride, len(got.rows)) == (5, 1, 5)
    assert all(6000.0 <= r[0] < 6300.0 for r in got.rows)
    assert got.rows[0][1:] == (0.1, 0.5, 2.0)


async def test_exactly_at_the_cap_reads_every_row(tmp_path):
    db_path = await _migrated(tmp_path)
    _token_rows(db_path, rh.TIMING_ROW_CAP)
    async with open_db(db_path) as db:
        got = await rh.query_token_timings(db, token_ids=["tok-a"], since=0.0, until=1e12)
    assert (got.total, got.stride, len(got.rows)) == (rh.TIMING_ROW_CAP, 1, rh.TIMING_ROW_CAP)


async def test_one_past_the_cap_samples_every_second_row_across_the_whole_period(tmp_path):
    db_path = await _migrated(tmp_path)
    n = rh.TIMING_ROW_CAP + 1
    _token_rows(db_path, n, start=6000.0, span=3600.0)  # minutes 100..159
    async with open_db(db_path) as db:
        got = await rh.query_token_timings(db, token_ids=["tok-a"], since=0.0, until=1e12)
    assert got.total == n
    assert got.stride == 2
    # Every second row of 50_001, oldest first: rows 0, 2, ..., 50_000.
    assert len(got.rows) == 25_001
    minutes = {int(r[0] // 60) for r in got.rows}
    # Every k-th row, oldest first: the first AND the last minute survive,
    # which a "newest N" cut would not give.
    assert min(minutes) == 100 and max(minutes) == 159


async def test_the_cap_can_be_lowered(tmp_path):
    db_path = await _migrated(tmp_path)
    _token_rows(db_path, 25)
    async with open_db(db_path) as db:
        got = await rh.query_token_timings(
            db,
            token_ids=["tok-a"],
            since=0.0,
            until=1e12,
            cap=10,
        )
    assert (got.total, got.stride, len(got.rows)) == (25, 3, 9)  # ceil(25/10) = 3


async def test_cache_obs_columns_round_trip(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    obs = dict(
        reusable_tokens=9000,
        reusable_tokens_fleet=9500,
        reusable_tokens_own=4000,
        cache_outcome="partial",
        cache_outcome_own="hit",
        diverged_at=4,
        diverged_at_own=None,
        cached_source="engine",
        cached_ttft_est_tokens=None,
    )
    store.record(_record(1, **obs), finished_at=1000.0)
    store.record(_record(2), finished_at=1001.0)
    assert await store.flush() == 2
    async with open_db(db_path) as db:
        rows = {r["id"]: r for r in await rh.query_last(db, n=10, model_ids=None)}
    for k, v in obs.items():
        assert rows["req-1"][k] == v
    assert all(rows["req-2"][k] is None for k in obs)
