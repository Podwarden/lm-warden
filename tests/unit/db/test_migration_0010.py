"""Verify migration 0010 creates the settings table and seeds 8 defaults.

The ghost `hf_cache_dir` key 0010 used to seed is removed from the seed (and
deleted from existing DBs by 0023) — its DB value was never read; the cache
path is env-driven via `VW_HF_CACHE_DIR` → `settings.hf_cache_dir`.

Spec: docs/superpowers/specs/2026-05-11-vllm-warden-ui-redesign-design.md (Phase 3).

The settings table did not exist in any prior migration — 0010 is the first
to create it. The seed must be idempotent so re-running migrations on an
already-initialized DB is a no-op (INSERT OR IGNORE).
"""

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations

EXPECTED_DEFAULTS = {
    "session_access_ttl_minutes": "15",
    "session_refresh_ttl_days": "7",
    "sse_ticket_ttl_seconds": "60",
    "default_token_expiration_days": "365",
    "rotation_grace_hours": "24",
    "log_retention_lines": "5000",
    "vllm_version": "0.9.2",
    "default_gpu_indices": "[0]",
    # Seeded by 0020 for the #155 unified-port landing page; off on a new
    # install since 2026-10-07 (the public website is opt-in).
    "landing_page_enabled": "false",
}


@pytest.mark.asyncio
async def test_settings_table_created_and_seeded(tmp_path):
    async with open_db(tmp_path / "vw.db") as db:
        await apply_migrations(db)

        # Table must exist.
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
        )
        assert await cur.fetchone() is not None, "settings table should be created"

        cur = await db.execute("SELECT key, value FROM settings")
        rows = dict(await cur.fetchall())

    for key, value in EXPECTED_DEFAULTS.items():
        assert key in rows, f"missing default for {key}"
        assert rows[key] == value, f"{key} default mismatch: got {rows[key]!r}"


@pytest.mark.asyncio
async def test_migration_is_idempotent(tmp_path):
    """Re-applying migrations on an existing DB must not duplicate or change rows."""
    db_path = tmp_path / "vw.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
        # Mutate one row so we can prove INSERT OR IGNORE preserves it.
        await db.execute(
            "UPDATE settings SET value = ? WHERE key = ?",
            ("42", "rotation_grace_hours"),
        )
        await db.commit()

    # Re-open and re-apply — schema_migrations should short-circuit the file,
    # but even if a future migration re-runs the seed the IGNORE clause must
    # leave the mutated value alone.
    async with open_db(db_path) as db:
        await apply_migrations(db)
        cur = await db.execute("SELECT value FROM settings WHERE key = 'rotation_grace_hours'")
        (val,) = await cur.fetchone()
        assert val == "42", "INSERT OR IGNORE must not overwrite an existing key"

        # Total row count unchanged.
        cur = await db.execute("SELECT COUNT(*) FROM settings")
        (n,) = await cur.fetchone()
        assert n == len(EXPECTED_DEFAULTS)
