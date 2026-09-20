import time

import pytest

from app.config import load_settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.stats_pruner import RETENTION_MINUTES, prune_once


@pytest.fixture
async def settings_with_db(tmp_data_dir):
    settings = load_settings()
    async with open_db(settings.db_path) as conn:
        await apply_migrations(conn)
        await ModelRepo(conn).insert(ModelRow(
            id="m1", served_model_name="m1", hf_repo="o/r", hf_revision="main",
            gpu_indices=[0], tensor_parallel_size=1, dtype=None,
            max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False,
            extra_args=[], extra_env={}, status="registered", pulled_bytes=0,
            pulled_total=None, last_error=None,
        ))
    return settings


async def test_prune_removes_rows_older_than_retention(settings_with_db):
    now_min = int(time.time() // 60)
    fresh = now_min
    stale = now_min - RETENTION_MINUTES - 1
    async with open_db(settings_with_db.db_path) as db:
        await db.execute(
            "INSERT INTO model_samples(model_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES ('m1', ?, 1, 1, 1), ('m1', ?, 1, 1, 1)",
            (fresh, stale),
        )
        await db.execute(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, memory_used_mib, memory_total_mib) "
            "VALUES (0, ?, 1, 1, 1), (0, ?, 1, 1, 1)",
            (fresh, stale),
        )
        await db.commit()

    deleted = await prune_once(settings_with_db)
    assert deleted["model_samples"] == 1
    assert deleted["gpu_samples"] == 1

    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM model_samples")
        assert (await cur.fetchone())[0] == 1
        cur = await db.execute("SELECT COUNT(*) FROM gpu_samples")
        assert (await cur.fetchone())[0] == 1


async def test_prune_keeps_admin_audit_for_ninety_days(settings_with_db):
    from app.db.repos.admin_audit import RETENTION_DAYS, AdminAuditRepo

    now = time.time()
    async with open_db(settings_with_db.db_path) as db:
        repo = AdminAuditRepo(db)
        for ts in (now - (RETENTION_DAYS + 1) * 86400, now - 60):
            await repo.record(ts=ts, token_id="a1", username="admin", method="GET",
                              path="/api/models", status=200, duration_ms=1, client_ip=None)
    deleted = await prune_once(settings_with_db)
    assert deleted["admin_audit"] == 1
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM admin_audit")
        assert (await cur.fetchone())[0] == 1


async def test_prune_caps_admin_audit_rows(settings_with_db, monkeypatch):
    from app.db.repos import admin_audit

    monkeypatch.setattr(admin_audit, "MAX_ROWS", 2)
    # Isolate the global cap from the per-token cap (#257): a single token
    # writing 5 rows would otherwise also trip a small PER_TOKEN_MAX_ROWS.
    monkeypatch.setattr(admin_audit, "PER_TOKEN_MAX_ROWS", 20_000)
    now = time.time()
    async with open_db(settings_with_db.db_path) as db:
        repo = admin_audit.AdminAuditRepo(db)
        for i in range(5):
            await repo.record(ts=now - i, token_id="a1", username="admin", method="GET",
                              path="/api/models", status=200, duration_ms=1, client_ip=None)
    deleted = await prune_once(settings_with_db)
    assert deleted["admin_audit"] == 3
    # The cap keeps the NEWEST rows (ts DESC), not whichever three survive a
    # flipped ORDER BY -- pins the direction, not just the count.
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT ts FROM admin_audit ORDER BY ts DESC")
        assert [r[0] for r in await cur.fetchall()] == [now, now - 1]


async def test_prune_caps_admin_audit_rows_per_token(settings_with_db, monkeypatch):
    """Issue #257: prune_once must wire admin_audit.PER_TOKEN_MAX_ROWS through
    to the repo, so a single noisy token evicts only its own rows and a
    quiet token's trail survives even though the noisy token alone would
    have blown the global cap."""
    from app.db.repos import admin_audit

    monkeypatch.setattr(admin_audit, "PER_TOKEN_MAX_ROWS", 2)
    monkeypatch.setattr(admin_audit, "MAX_ROWS", 200_000)
    now = time.time()
    async with open_db(settings_with_db.db_path) as db:
        repo = admin_audit.AdminAuditRepo(db)
        for i in range(5):
            await repo.record(ts=now - i, token_id="noisy", username="admin", method="GET",
                              path="/api/models", status=200, duration_ms=1, client_ip=None)
        await repo.record(ts=now - 100, token_id="quiet", username="admin", method="GET",
                          path="/api/models", status=200, duration_ms=1, client_ip=None)
    deleted = await prune_once(settings_with_db)
    assert deleted["admin_audit"] == 3  # noisy's 3 oldest, over its cap of 2
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute(
            "SELECT token_id, ts FROM admin_audit ORDER BY ts DESC"
        )
        rows = await cur.fetchall()
    assert [r[0] for r in rows] == ["noisy", "noisy", "quiet"]
    assert [r[1] for r in rows[:2]] == [now, now - 1]  # noisy's newest 2
    assert rows[2][1] == now - 100  # quiet's row, untouched
