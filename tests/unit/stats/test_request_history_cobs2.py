"""request_history persists the cobs2 L1 ids (plan 3, task 5); the v2 readers
do not return them (no /api change)."""

from __future__ import annotations

import sqlite3

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats import request_history as rh
from app.stats.request_history import RequestHistoryStore

_COBS2 = (
    "cobs2_head",
    "cobs2_node",
    "cobs2_parent_node",
    "cobs2_edge",
    "cobs2_reply_msg",
    "cobs2_compaction_request",
)
HEAD = "01" + "a" * 32
NODE = "01" + "b" * 32
PARENT = "01" + "c" * 32
REPLY = "01" + "d" * 32


def _record(i: int, **over):
    base = {
        "id": f"req-{i}",
        "model": "model-a",
        "model_id": "id-model-a",
        "token_name": "key-a",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "duration_s": 1.0,
        "finish_reason": "stop",
        "orphan": False,
        "started_iso": "2026-10-25T10:00:00Z",
    }
    base.update(over)
    return base


async def _migrated(tmp_path):
    db_path = tmp_path / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    return db_path


def _rows(db_path):
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT id, " + ", ".join(_COBS2) + " FROM request_history ORDER BY id"
        ).fetchall()


async def test_cobs2_columns_round_trip(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(
        _record(
            1,
            cobs2_head=HEAD,
            cobs2_node=NODE,
            cobs2_parent_node=PARENT,
            cobs2_edge="compacted_from",
            cobs2_reply_msg=REPLY,
            cobs2_reply_loose_msg="01" + "e" * 32,
            cobs2_compaction_request=True,
        ),
        finished_at=1000.0,
    )
    store.record(
        _record(
            2, cobs2_head=HEAD, cobs2_node=NODE, cobs2_edge="new", cobs2_compaction_request=False
        ),
        finished_at=1001.0,
    )
    assert await store.flush() == 2
    assert _rows(db_path) == [
        ("req-1", HEAD, NODE, PARENT, "compacted_from", REPLY, 1),
        ("req-2", HEAD, NODE, None, "new", None, 0),
    ]


async def test_not_computed_is_null(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1), finished_at=1000.0)
    assert await store.flush() == 1
    assert _rows(db_path) == [("req-1", None, None, None, None, None, None)]


async def test_bad_values_cost_nothing(tmp_path):
    """Non-string ids and an unknown edge never reach the table as garbage."""
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(
        _record(1, cobs2_head=123, cobs2_edge=["x"], cobs2_node=b"raw"), finished_at=1000.0
    )
    assert await store.flush() == 1
    assert _rows(db_path) == [("req-1", None, None, None, None, None, None)]


async def test_v2_readers_do_not_return_cobs2_columns(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(_record(1, cobs2_head=HEAD, cobs2_edge="new"), finished_at=1000.0)
    assert await store.flush() == 1
    assert not set(_COBS2) & set(rh._READ_COLUMNS)
    async with open_db(db_path) as db:
        (last,) = await rh.query_last(db, n=1, model_ids=None)
        (win,) = (await rh.query_window(db, since=0, limit=10, model_ids=None)).rows
    for row in (last, win):
        assert row["id"] == "req-1"
        assert not set(_COBS2) & set(row)


def test_the_loose_reply_id_is_not_a_column():
    assert "cobs2_reply_loose_msg" not in rh._COLUMNS


async def test_divergence_cause_columns_round_trip_and_reject_junk(tmp_path):
    db_path = await _migrated(tmp_path)
    store = RequestHistoryStore(db_path)
    store.record(
        _record(
            1,
            cobs2_divergence_cause="harness_rewrite",
            cobs2_divergence_cause_own="compacted",
        ),
        finished_at=1000.0,
    )
    store.record(
        _record(2, cobs2_divergence_cause="evil'); --", cobs2_divergence_cause_own=3),
        finished_at=1001.0,
    )
    assert await store.flush() == 2
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT id, cobs2_divergence_cause, cobs2_divergence_cause_own "
            "FROM request_history ORDER BY id"
        ).fetchall()
    assert rows == [("req-1", "harness_rewrite", "compacted"), ("req-2", None, None)]
