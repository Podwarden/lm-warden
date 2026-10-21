"""Migration 0050: ``models.backend`` stops being implicitly 'vllm'.

0027 added the column as NULLABLE ON PURPOSE and deliberately did not
backfill (decision D6): every pre-existing row sits at ``NULL`` and
``app/db/repos/models.py::_decode_row`` decodes NULL to ``'vllm'`` through
``app/runtime/backends/registry.get(None)``. The deployment-profile split
(Metal design s10) changes the *default*: on ``apple`` the profile default is
``llamacpp``, so a NULL row decoded there would silently become a non-vllm
model. The first step of that split is this migration, which makes the
historical default explicit in the data. On Linux it is a semantic no-op:
NULL and ``'vllm'`` decode identically, so no existing row changes behaviour.

This file pins every arm of the ``WHERE`` clause and the idempotence
guarantee:

* ``NULL``  -> ``'vllm'``      (the historical default)
* ``''``    -> ``'vllm'``      (the other historical shape)
* ``'llamacpp'`` -> unchanged  (a row with a real, different backend)
* ``'vllm'``     -> unchanged  (a row already explicit)
* ``0050`` records exactly one row in ``schema_migrations``
* re-running ``apply_migrations`` changes nothing (idempotent)
"""

import aiosqlite
import pytest

from app.db.database import open_db
from app.db.migrations import SQL_DIR, apply_migrations

MIGRATION = "0050_models_backend_explicit.sql"


def _migration_file() -> str:
    """Imported here (not at module level) so the base tree -- where 0050 does
    not exist yet -- still collects this file: the FileNotFoundError fires
    inside the test, not at import time, which is the red-for-the-right-reason
    shape the worker rules require."""
    path = SQL_DIR / MIGRATION
    assert path.exists(), f"{MIGRATION} is missing from app/db/sql/"
    return path.read_text(encoding="utf-8")


async def _apply_through(db_path, last: str) -> None:
    """Migrate a fresh file through ``last`` only, the way a pre-upgrade
    deployment's database looks the moment the new image boots. Mirrors the
    0031 test's helper: replay each file's statements inside its own
    transaction and record the filename, skipping everything after ``last``."""
    import app.db.migrations as m

    files = sorted(p.name for p in m.SQL_DIR.glob("*.sql"))
    assert last in files, f"{last} is not a known migration file"
    keep = files[: files.index(last) + 1]
    async with open_db(db_path) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  filename TEXT PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        for name in keep:
            statements = m._split_statements((m.SQL_DIR / name).read_text(encoding="utf-8"))
            await db.execute("BEGIN IMMEDIATE")
            for stmt in statements:
                await db.execute(stmt)
            await db.execute("INSERT INTO schema_migrations(filename) VALUES (?)", (name,))
            await db.commit()


def _backends(db_path) -> dict[str, str | None]:
    """Read ``models.backend`` for every id, id-sorted, from a fresh read-only
    connection so the assertion does not depend on the test connection's
    journal state."""
    import sqlite3

    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT id, backend FROM models ORDER BY id").fetchall()
    return dict(rows)


@pytest.mark.asyncio
async def test_0050_null_and_empty_backend_become_explicit_vllm(tmp_data_dir):
    """The red-first leg: a pre-0050 database with a NULL backend row must
    come out of the upgrade with backend == 'vllm', and the same for the ''
    arm. Today (no 0050) the row stays NULL and this fails."""
    db_path = tmp_data_dir / "vllm-warden.db"
    # Bring the schema to 0049 so the `backend` column exists but the UPDATE
    # has not run yet -- the shape of a live pre-upgrade volume.
    await _apply_through(db_path, "0049_model_load_history.sql")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "backend) VALUES "
            "('m-null', 'served-null', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'loaded', NULL), "
            "('m-empty', 'served-empty', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'registered', '')"
        )
        await db.commit()

    async with open_db(db_path) as db:
        await apply_migrations(db)

    got = _backends(db_path)
    assert got["m-null"] == "vllm", "NULL backend must be made explicit 'vllm'"
    assert got["m-empty"] == "vllm", "'' backend must be made explicit 'vllm'"


@pytest.mark.asyncio
async def test_0050_leaves_real_backends_unchanged(tmp_data_dir):
    """A row that already carries a real backend must not be touched: the
    WHERE clause only fires on NULL and ''."""
    db_path = tmp_data_dir / "vllm-warden.db"
    await _apply_through(db_path, "0049_model_load_history.sql")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "backend) VALUES "
            "('m-llama', 'served-llama', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'registered', 'llamacpp'), "
            "('m-vllm', 'served-vllm', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'registered', 'vllm')"
        )
        await db.commit()

    async with open_db(db_path) as db:
        await apply_migrations(db)

    got = _backends(db_path)
    assert got["m-llama"] == "llamacpp", "an explicit 'llamacpp' row must stay 'llamacpp'"
    assert got["m-vllm"] == "vllm", "an explicit 'vllm' row must stay 'vllm'"


@pytest.mark.asyncio
async def test_0050_records_once_and_reapply_is_a_noop(tmp_data_dir):
    """0050 lands in schema_migrations exactly once, and a second
    apply_migrations() pass changes no data and adds no row -- the same
    idempotence every other migration in this directory guarantees."""
    db_path = tmp_data_dir / "vllm-warden.db"
    await _apply_through(db_path, "0049_model_load_history.sql")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "backend) VALUES "
            "('m-null', 'served-null', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'loaded', NULL), "
            "('m-llama', 'served-llama', 'o/r', 'main', '[0]', 1, NULL, NULL, "
            "0.9, 0, '[]', 'registered', 'llamacpp')"
        )
        await db.commit()

    async with open_db(db_path) as db:
        await apply_migrations(db)
        mid = _backends(db_path)
        await apply_migrations(db)  # must be a no-op

    import sqlite3

    with sqlite3.connect(db_path) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE filename = ?", (MIGRATION,)
        ).fetchone()[0]
        after = dict(db.execute("SELECT id, backend FROM models ORDER BY id").fetchall())
    assert count == 1, f"{MIGRATION} must be recorded exactly once, found {count}"
    assert after == mid, "a second apply_migrations() must not change any row"
    assert after["m-null"] == "vllm"
    assert after["m-llama"] == "llamacpp"


def test_0050_file_is_transaction_safe():
    """The file must not carry any keyword the migration runner's
    _check_transaction_safe guard rejects: the guard runs on the leading
    keyword of every statement, so a PRAGMA / BEGIN / COMMIT anywhere as a
    statement would make apply_migrations raise."""
    import app.db.migrations as m

    sql = _migration_file()
    statements = m._split_statements(sql)
    for stmt in statements:
        keyword = m._leading_keyword(stmt)
        assert keyword not in m._TRANSACTION_HOSTILE, (
            f"{MIGRATION} opens with a {keyword.upper()} statement, which the "
            f"migration runner rejects (it must run inside the runner's own "
            f"per-file transaction)."
        )
    # The file must contain at least one executable statement -- an empty or
    # comment-only 0050 would pass the guard vacuously and silently do nothing.
    assert any(
        m._leading_keyword(s) for s in statements
    ), f"{MIGRATION} contains no executable statement"


def test_0050_is_the_immediately_next_migration():
    """0050 must sort right after 0049 in the SQL_DIR glob: the runner applies
    files in lexicographic order, so a 0050 that sorts earlier (or later,
    with a gap) would run at the wrong point in the upgrade."""
    import app.db.migrations as m

    files = sorted(p.name for p in m.SQL_DIR.glob("*.sql"))
    idx = files.index(MIGRATION)
    assert files[idx - 1] == "0049_model_load_history.sql", (
        f"{MIGRATION} must sort immediately after 0049, but the file before it "
        f"is {files[idx - 1]!r}"
    )
