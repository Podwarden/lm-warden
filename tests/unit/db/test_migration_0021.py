"""Migration 0021: documents the `public_url` runtime key.

Spec: docs/superpowers/specs/2026-05-24-settings-redesign-design.md (#154).

0021 is intentionally a no-op at the row level — the FE falls back to
`window.location.origin` when the key is absent, so seeding a placeholder
would either be wrong (an arbitrary URL) or noisy (a sentinel the FE then
has to recognise as "treat-as-unset"). What this test pins:

  * The migration runner records 0021 as applied (so future migrations
    don't re-run it).
  * No `public_url` row appears in the settings table (no accidental seed).
  * The settings-table seed from 0010 is untouched.
  * Re-applying migrations is a no-op.
"""

import aiosqlite

from app.db.database import open_db
from app.db.migrations import apply_migrations


async def test_migration_0021_recorded_as_applied(tmp_data_dir):
    """The migration runner must register 0021 in schema_migrations so it
    isn't re-applied on every startup."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT filename FROM schema_migrations "
            "WHERE filename = '0021_public_url_setting.sql'"
        )
        row = await cur.fetchone()
    assert row is not None, "0021 should be recorded as applied"


async def test_migration_0021_does_not_seed_public_url(tmp_data_dir):
    """The conservative default is `absent row` — the FE treats absence as
    'use window.location.origin'. Any seed here would mask that fallback."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT key FROM settings WHERE key = 'public_url'")
        row = await cur.fetchone()
    assert row is None, "public_url must NOT be seeded by migration 0021"


async def test_migration_0021_preserves_prior_seed(tmp_data_dir):
    """The 0010 / 0020 seeds must remain untouched after 0021 runs — we're
    only documenting a new key, not rewriting existing rows."""
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT key, value FROM settings")
        rows = dict(await cur.fetchall())
    # Spot-check a handful of seeded keys from 0010 / 0020 — exhaustive
    # coverage lives in test_migration_0010.py / test_landing_setting.py.
    assert rows.get("session_access_ttl_minutes") == "15"
    assert rows.get("vllm_version") == "0.9.2"
    assert rows.get("landing_page_enabled") == "false"


async def test_migration_0021_idempotent(tmp_data_dir):
    """Re-applying migrations must be a no-op — neither double-seeds nor
    errors out. Mirrors the pattern in test_migration_0019_power_samples.py.
    """
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await apply_migrations(db)  # second pass — no-op
    async with aiosqlite.connect(db_path) as db:
        # Still no public_url row after two passes.
        cur = await db.execute("SELECT COUNT(*) FROM settings WHERE key = 'public_url'")
        (n,) = await cur.fetchone()
        assert n == 0
        # 0021 still recorded exactly once.
        cur = await db.execute(
            "SELECT COUNT(*) FROM schema_migrations "
            "WHERE filename = '0021_public_url_setting.sql'"
        )
        (m,) = await cur.fetchone()
        assert m == 1
