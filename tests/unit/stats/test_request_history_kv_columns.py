"""request_history stores the per-key KV isolation fields (migration 0060) without shifting
the v2 readers' column slice.

``_READ_COLUMNS`` is ``_COLUMNS`` minus its tail tuples. Two columns appended without a tail
tuple of their own would move the slice by two and every v2 reader would select the forest's
``session_key`` pair instead of what it reads today.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from app.db import migrations
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.stats import request_history as rh
from app.stats.request_history import RequestHistoryStore, query_last

KV = ("kv_isolated", "kv_iso_cost_tokens")


def _record(i: int, **over):
    base = {
        "id": f"req-{i}",
        "model": "model-a",
        "model_id": "id-model-a",
        "token_name": "key-a",
        "token_id": "tok1",
        "prompt_tokens": 100 + i,
        "completion_tokens": 10 + i,
        "duration_s": 1.5,
        "ttft_s": 0.25,
        "finish_reason": "stop",
        "orphan": False,
        "started_iso": "2026-11-21T10:00:00Z",
        "queued_s": 0.01,
        "variant_id": "v1",
        "cached_tokens": 64,
        "cache_outcome": "hit",
        "cached_source": "engine",
        "thinking_forced_off": True,
        "session_key": "sk",
        "prompt_diet": '{"x":1}',
    }
    base.update(over)
    return base


def _q(db_path: Path, sql: str, args: tuple = ()) -> list[tuple]:
    with sqlite3.connect(db_path) as c:
        return c.execute(sql, args).fetchall()


def test_kv_columns_are_their_own_tail_and_the_read_slice_did_not_move() -> None:
    assert rh._KV_ISO_COLUMNS == KV
    # 0063's lens tail follows it.
    assert rh._COLUMNS[: -len(rh._LENS_COLUMNS)][-len(KV) :] == KV
    # The v2 readers read exactly the columns before the forest block, as before 0060.
    assert rh._READ_COLUMNS == rh._COLUMNS[: rh._COLUMNS.index("session_key")]
    assert rh._READ_COLUMNS[-1] == "thinking_forced_off"
    assert not set(KV) & set(rh._READ_COLUMNS)


async def test_old_row_and_new_row_read_back_through_the_v2_reader(
    tmp_data_dir, tmp_path, monkeypatch
) -> None:
    # A row written before the upgrade, on the 0058 schema.
    old_sql = tmp_path / "sql"
    old_sql.mkdir()
    for p in migrations.SQL_DIR.glob("*.sql"):
        if p.name < "0060":
            shutil.copy(p, old_sql / p.name)
    db_path = tmp_data_dir / "vllm-warden.db"
    real_sql = migrations.SQL_DIR
    monkeypatch.setattr(migrations, "SQL_DIR", old_sql)
    old_cols = rh._COLUMNS[: -len(KV) - len(rh._LENS_COLUMNS)]
    old_params = rh._row_params(_record(1), 1000.0)[: len(old_cols)]
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await db.execute(
            f"INSERT INTO request_history({', '.join(old_cols)}) "
            f"VALUES ({', '.join('?' for _ in old_cols)})",
            old_params,
        )
        await db.commit()
    monkeypatch.setattr(migrations, "SQL_DIR", real_sql)
    async with open_db(db_path) as db:
        await apply_migrations(db)

    # A row written after it, carrying the two new fields.
    store = RequestHistoryStore(db_path)
    store.record(_record(2, kv_isolated=True, kv_iso_cost_tokens=4096), finished_at=1001.0)
    store.record(_record(3, kv_isolated=False, kv_iso_cost_tokens="bad"), finished_at=1002.0)
    assert await store.flush() == 2

    async with open_db(db_path) as db:
        rows = {r["id"]: r for r in await query_last(db, n=10, model_ids=None)}
    assert set(rows) == {"req-1", "req-2", "req-3"}
    for i, finished in ((1, 1000.0), (2, 1001.0), (3, 1002.0)):
        r = rows[f"req-{i}"]
        assert tuple(r) == rh._READ_COLUMNS
        assert r["finished_at"] == finished
        assert (r["prompt_tokens"], r["completion_tokens"]) == (100 + i, 10 + i)
        assert (r["duration_s"], r["ttft_s"], r["queued_s"]) == (1.5, 0.25, 0.01)
        assert (r["variant_id"], r["cached_tokens"], r["cached_source"]) == ("v1", 64, "engine")
        assert r["cache_outcome"] == "hit"
        assert r["thinking_forced_off"] == 1
        assert r["orphan"] is False
    assert _q(
        db_path,
        "SELECT id, kv_isolated, kv_iso_cost_tokens, session_key, prompt_diet "
        "FROM request_history ORDER BY id",
    ) == [
        ("req-1", None, None, "sk", '{"x":1}'),
        ("req-2", 1, 4096, "sk", '{"x":1}'),
        ("req-3", 0, None, "sk", '{"x":1}'),
    ]
