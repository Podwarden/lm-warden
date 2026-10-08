"""Verify migration 0014 adds the per-file / parallelism columns to ``models``.

0014 extends the ``models`` table with three columns the #85 wizard writes:

* ``filename``               TEXT, nullable — pinned weights file for per-file
                              pulls; ``NULL`` means whole-repo (legacy).
* ``parallelism_strategy``   TEXT NOT NULL DEFAULT 'auto' — one of
                              ``tp`` | ``pp`` | ``auto``.
* ``max_batch_size``         INTEGER NOT NULL DEFAULT 1.

The defaults exist so legacy rows surviving the migration decode to the
pre-#85 wizard's implicit shape. If any of these constraints drift, the
ModelRepo decoder will start raising on pre-existing rows.
"""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations


@pytest.mark.asyncio
async def test_0014_adds_filename_parallelism_columns(tmp_path):
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)

        cur = await db.execute("PRAGMA table_info(models)")
        cols = {row[1]: row for row in await cur.fetchall()}

    # cid, name, type, notnull, dflt_value, pk
    assert "filename" in cols
    assert cols["filename"][2].upper() == "TEXT"
    assert cols["filename"][3] == 0, "filename must be nullable"

    assert "parallelism_strategy" in cols
    assert cols["parallelism_strategy"][2].upper() == "TEXT"
    assert cols["parallelism_strategy"][3] == 1, "parallelism_strategy must be NOT NULL"
    # SQLite reports default literals with quotes preserved.
    assert cols["parallelism_strategy"][4] in ("'auto'", "auto")

    assert "max_batch_size" in cols
    assert cols["max_batch_size"][2].upper() == "INTEGER"
    assert cols["max_batch_size"][3] == 1, "max_batch_size must be NOT NULL"
    assert str(cols["max_batch_size"][4]) == "1"


@pytest.mark.asyncio
async def test_0014_defaults_applied_on_minimal_insert(tmp_path):
    """A row inserted without the new columns should pick up the defaults
    (mirrors what happens to rows that pre-date 0014)."""
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)

        await db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, extra_env, "
            "created_at, updated_at) "
            "VALUES ('m1', 'srv', 'org/repo', 'main', '[0]', 1, 0.9, 0, '[]', "
            "'registered', 0, '{}', datetime('now'), datetime('now'))"
        )
        await db.commit()

        cur = await db.execute(
            "SELECT filename, parallelism_strategy, max_batch_size " "FROM models WHERE id='m1'"
        )
        row = await cur.fetchone()

    assert row == (None, "auto", 1)
