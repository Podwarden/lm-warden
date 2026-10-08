from app.db.database import open_db


async def test_open_db_sets_busy_timeout(tmp_data_dir):
    """open_db must set an explicit, generous busy_timeout so background
    writers (stats sampler, pull poller) wait out a contention window during
    a heavy model download instead of dropping a write with 'database is
    locked'. aiosqlite's connect() default is only 5000ms; we set it
    explicitly and higher so the intent survives that default changing."""
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        cur = await conn.execute("PRAGMA busy_timeout")
        row = await cur.fetchone()
    assert row[0] == 30000


async def test_open_db_keeps_wal_journal_mode(tmp_data_dir):
    """Regression guard: busy_timeout must not displace WAL mode."""
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        cur = await conn.execute("PRAGMA journal_mode")
        row = await cur.fetchone()
    assert row[0].lower() == "wal"
