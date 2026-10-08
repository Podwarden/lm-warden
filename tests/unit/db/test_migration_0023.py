"""Migration 0023: remove the ghost `hf_cache_dir` setting.

`hf_cache_dir` was seeded by 0010 and exposed as an editable runtime key, but
its DB value was NEVER read — the engine + pull task derive the cache path from
`VW_HF_CACHE_DIR` (env → `settings.hf_cache_dir`), not from this KV row. Editing
it in the UI silently did nothing. 0023 deletes the stray row from existing DBs
(0010's seed line is removed in the same change so fresh DBs never get it).

What this test pins:
  * The migration runner records 0023 as applied.
  * No `hf_cache_dir` row remains after a full migration run.
  * Re-applying migrations is a no-op (delete stays deleted, recorded once).
"""

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_migration_0023_recorded_as_applied(tmp_data_dir):
    """The migration runner must register 0023 in schema_migrations so it
    isn't re-applied on every startup."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT filename FROM schema_migrations "
            "WHERE filename = '0023_drop_hf_cache_dir_setting.sql'"
        )
        row = await cur.fetchone()
    assert row is not None, "0023 should be recorded as applied"


async def test_migration_0023_removes_hf_cache_dir_row(tmp_data_dir):
    """After the full migration set, the ghost `hf_cache_dir` row is gone."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT key FROM settings WHERE key = 'hf_cache_dir'")
        row = await cur.fetchone()
    assert row is None, "hf_cache_dir must be deleted by migration 0023"


async def test_migration_0023_idempotent(tmp_data_dir):
    """Re-applying migrations must be a no-op — the row stays deleted and 0023
    is recorded exactly once."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second pass — no-op
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM settings WHERE key = 'hf_cache_dir'")
        (n,) = await cur.fetchone()
        assert n == 0
        cur = await db.execute(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE filename = '0023_drop_hf_cache_dir_setting.sql'"
        )
        (m,) = await cur.fetchone()
        assert m == 1
