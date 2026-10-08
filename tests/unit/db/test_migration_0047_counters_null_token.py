"""Migration 0047: one ``counters`` row per (model, NULL token) (#298).

SQLite's composite PRIMARY KEY treats NULLs as distinct, so nothing stopped a
second (model, NULL) row. pw-prod had two. Covers:
  * 0047 merges existing duplicates into one row, summing all three counts
  * the partial unique index then refuses a second (model, NULL) row
  * deleting a key folds its rows into the model's NULL row (the FK's
    ON DELETE SET NULL would otherwise collide with the index, or before it,
    create the duplicate)
  * CountersRepo.increment upserts against the index
"""

import shutil
import sqlite3

import pytest

from app.db import migrations
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.counters import CountersRepo
from app.db.repos.models import ModelRepo, ModelRow
from app.db.repos.tokens import TokenRepo

INDEX = "idx_counters_null_token_model"


def _model(model_id: str) -> ModelRow:
    return ModelRow(
        id=model_id,
        served_model_name=model_id,
        hf_repo="o/r",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        extra_env={},
        status="registered",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
    )


async def _seed(db) -> None:
    await ModelRepo(db).insert(_model("m1"))
    await ModelRepo(db).insert(_model("m2"))
    await TokenRepo(db).create("tok-a", "a", "vw_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    await TokenRepo(db).create("tok-b", "b", "vw_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")


def _rows(db_path):
    with sqlite3.connect(db_path) as c:
        return c.execute(
            "SELECT model_id, token_id, requests, prompt_tokens, completion_tokens "
            "FROM counters ORDER BY model_id, token_id"
        ).fetchall()


async def test_0047_merges_duplicate_null_token_rows(tmp_data_dir, tmp_path, monkeypatch):
    # A database at 0046, as pw-prod is before the deploy.
    old_sql = tmp_path / "sql"
    old_sql.mkdir()
    for p in migrations.SQL_DIR.glob("*.sql"):
        if p.name < "0047":
            shutil.copy(p, old_sql / p.name)
    db_path = tmp_data_dir / "vllm-warden.db"
    real_sql = migrations.SQL_DIR
    monkeypatch.setattr(migrations, "SQL_DIR", old_sql)
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed(db)
        await db.executemany(
            "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("m1", None, 4, 40, 4),
                ("m1", None, 32, 320, 32),
                ("m1", None, 1, 10, 1),
                ("m1", "tok-a", 7, 70, 7),
                ("m2", None, 5, 50, 5),
            ],
        )
        await db.commit()

    monkeypatch.setattr(migrations, "SQL_DIR", real_sql)
    async with open_db(db_path) as db:
        await apply_migrations(db)

    assert _rows(db_path) == [
        ("m1", None, 37, 370, 37),
        ("m1", "tok-a", 7, 70, 7),
        ("m2", None, 5, 50, 5),
    ]


async def test_0047_index_refuses_a_second_null_token_row(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed(db)
    with sqlite3.connect(db_path) as c:
        (sql,) = c.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (INDEX,)
        ).fetchone()
        assert "UNIQUE" in sql and "WHERE token_id IS NULL" in sql
        c.execute("INSERT INTO counters(model_id, token_id, requests) VALUES ('m1', NULL, 1)")
        # Another model's NULL row and keyed rows are untouched by the index.
        c.execute("INSERT INTO counters(model_id, token_id, requests) VALUES ('m2', NULL, 1)")
        c.execute("INSERT INTO counters(model_id, token_id, requests) VALUES ('m1', 'tok-a', 1)")
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO counters(model_id, token_id, requests) VALUES ('m1', NULL, 1)")


async def test_deleting_a_key_folds_its_counters_into_the_null_row(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed(db)
        await db.executemany(
            "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("m1", None, 32, 320, 32),  # m1 already has a NULL row: fold into it
                ("m1", "tok-a", 4, 40, 4),
                ("m2", "tok-a", 2, 20, 2),  # m2 has none: the row becomes it
                ("m2", "tok-b", 3, 30, 3),
            ],
        )
        await db.commit()
        assert await TokenRepo(db).delete("tok-a") is True

    assert _rows(db_path) == [
        ("m1", None, 36, 360, 36),
        ("m2", None, 2, 20, 2),
        ("m2", "tok-b", 3, 30, 3),
    ]


async def test_deleting_a_second_key_folds_into_the_row_the_first_created(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed(db)
        await db.executemany(
            "INSERT INTO counters(model_id, token_id, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?)",
            [("m1", "tok-a", 4, 40, 4), ("m1", "tok-b", 3, 30, 3)],
        )
        await db.commit()
        assert await TokenRepo(db).delete("tok-a") is True
        assert await TokenRepo(db).delete("tok-b") is True

    assert _rows(db_path) == [("m1", None, 7, 70, 7)]


async def test_increment_upserts_one_row_per_model_and_key(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await _seed(db)
        repo = CountersRepo(db)
        await repo.increment("m1", None, 10, 1)
        await repo.increment("m1", None, 20, 2)
        await repo.increment("m1", "tok-a", 5, 1)
        await repo.increment("m1", "tok-a", 6, 2)

    assert _rows(db_path) == [("m1", None, 2, 30, 3), ("m1", "tok-a", 2, 11, 3)]
