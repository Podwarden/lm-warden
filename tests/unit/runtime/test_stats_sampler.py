from unittest.mock import AsyncMock, patch

import pytest

from app.config import load_settings
from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.runtime.stats_sampler import sample_gpus_once
from app.system.gpu import GpuInfo


@pytest.fixture
async def settings_with_db(tmp_data_dir):
    settings = load_settings()
    async with open_db(settings.db_path) as conn:
        await apply_migrations(conn)
    return settings


async def test_sample_gpus_once_inserts_one_row_per_gpu(settings_with_db):
    fake = AsyncMock(
        return_value=[
            GpuInfo(
                index=0, name="A", memory_total_mib=24000, memory_used_mib=1000, utilization_pct=42
            ),
            GpuInfo(
                index=1, name="B", memory_total_mib=24000, memory_used_mib=2000, utilization_pct=55
            ),
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        n = await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
    assert n == 2
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute(
            "SELECT gpu_index, minute, utilization_pct, memory_used_mib FROM gpu_samples ORDER BY gpu_index"
        )
        rows = await cur.fetchall()
    assert rows == [(0, 1_700_000_000 // 60, 42, 1000), (1, 1_700_000_000 // 60, 55, 2000)]


async def test_sample_gpus_once_upserts_same_minute(settings_with_db):
    fake = AsyncMock(
        side_effect=[
            [
                GpuInfo(
                    index=0,
                    name="A",
                    memory_total_mib=24000,
                    memory_used_mib=100,
                    utilization_pct=10,
                )
            ],
            [
                GpuInfo(
                    index=0,
                    name="A",
                    memory_total_mib=24000,
                    memory_used_mib=200,
                    utilization_pct=90,
                )
            ],
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
        await sample_gpus_once(settings_with_db, now=1_700_000_030.0)  # same minute
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT utilization_pct, memory_used_mib FROM gpu_samples")
        rows = await cur.fetchall()
    assert rows == [(90, 200)]


async def test_sample_gpus_once_swallows_probe_failure(settings_with_db):
    fake = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        n = await sample_gpus_once(settings_with_db)
    assert n == 0


# ----------------------------------------------------------------------
# S7 (#124) — power sampling. The sampler MUST:
#   * write one power_samples row per GPU with non-None power_w
#   * accumulate within the same minute bucket (watts_sum + samples++)
#   * skip the power write when power_w is None (no zero-row corruption)
#   * use a SINGLE query_gpus() call per tick (CTO decision #6)
# ----------------------------------------------------------------------


async def test_sample_gpus_once_writes_power_when_available(settings_with_db):
    """One non-None power_w GPU → one power_samples row with samples=1."""
    fake = AsyncMock(
        return_value=[
            GpuInfo(
                index=0,
                name="A",
                memory_total_mib=24000,
                memory_used_mib=1000,
                utilization_pct=42,
                power_w=180.5,
            ),
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT gpu_idx, minute, watts_sum, samples FROM power_samples")
        rows = await cur.fetchall()
    assert rows == [(0, 1_700_000_000 // 60, 180.5, 1)]


async def test_sample_gpus_once_accumulates_power_within_minute(settings_with_db):
    """Two 5s ticks in the same minute → samples=2, watts_sum=sum, avg recoverable."""
    fake = AsyncMock(
        side_effect=[
            [
                GpuInfo(
                    index=0,
                    name="A",
                    memory_total_mib=24000,
                    memory_used_mib=1000,
                    utilization_pct=42,
                    power_w=100.0,
                )
            ],
            [
                GpuInfo(
                    index=0,
                    name="A",
                    memory_total_mib=24000,
                    memory_used_mib=1000,
                    utilization_pct=50,
                    power_w=200.0,
                )
            ],
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
        await sample_gpus_once(settings_with_db, now=1_700_000_005.0)  # +5s, same minute
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute(
            "SELECT gpu_idx, watts_sum, samples, "
            "       watts_sum / NULLIF(samples, 0) AS avg_w "
            "FROM power_samples"
        )
        rows = await cur.fetchall()
    assert rows == [(0, 300.0, 2, 150.0)]


async def test_sample_gpus_once_skips_power_when_none(settings_with_db):
    """power_w=None (driver reported [Not Supported]) → no power_samples row,
    but gpu_samples still gets util/mem. The chart shouldn't see a fake 0W."""
    fake = AsyncMock(
        return_value=[
            GpuInfo(
                index=0,
                name="A",
                memory_total_mib=24000,
                memory_used_mib=1000,
                utilization_pct=42,
                power_w=None,
            ),
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT COUNT(*) FROM power_samples")
        (power_n,) = await cur.fetchone()
        cur = await db.execute("SELECT COUNT(*) FROM gpu_samples")
        (gpu_n,) = await cur.fetchone()
    assert power_n == 0
    assert gpu_n == 1


async def test_sample_gpus_once_single_query_gpus_per_tick(settings_with_db):
    """CTO decision #6 — one nvidia-smi acquisition per tick gives util + mem +
    power together. The sampler must NOT make a second query_gpus() call to
    fetch power separately."""
    fake = AsyncMock(
        return_value=[
            GpuInfo(
                index=0,
                name="A",
                memory_total_mib=24000,
                memory_used_mib=1000,
                utilization_pct=42,
                power_w=180.5,
            ),
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake) as q:
        await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
    assert q.call_count == 1


async def test_sample_gpus_once_mixed_power_some_none(settings_with_db):
    """Mixed fleet — one GPU reports power, one doesn't. Only the reporting
    card lands in power_samples; both land in gpu_samples."""
    fake = AsyncMock(
        return_value=[
            GpuInfo(
                index=0,
                name="A",
                memory_total_mib=24000,
                memory_used_mib=1000,
                utilization_pct=42,
                power_w=180.5,
            ),
            GpuInfo(
                index=1,
                name="B",
                memory_total_mib=24000,
                memory_used_mib=2000,
                utilization_pct=55,
                power_w=None,
            ),
        ]
    )
    with patch("app.runtime.stats_sampler.query_gpus", fake):
        await sample_gpus_once(settings_with_db, now=1_700_000_000.0)
    async with open_db(settings_with_db.db_path) as db:
        cur = await db.execute("SELECT gpu_idx FROM power_samples ORDER BY gpu_idx")
        power_rows = await cur.fetchall()
        cur = await db.execute("SELECT gpu_index FROM gpu_samples ORDER BY gpu_index")
        gpu_rows = await cur.fetchall()
    assert power_rows == [(0,)]
    assert gpu_rows == [(0,), (1,)]
