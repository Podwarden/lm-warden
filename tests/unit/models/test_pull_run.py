import sqlite3
from contextlib import asynccontextmanager

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


async def _seed_model(db_path, model_id="m1", repo="o/r"):
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id=model_id,
                served_model_name=model_id,
                hf_repo=repo,
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
        )


async def test_run_pull_success_marks_pulled(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_disk_check(*args, **kwargs):
        return None

    async def fake_download(*args, **kwargs):
        return str(settings.hf_cache_dir / "fake")

    async def fake_estimate(*args, **kwargs):
        return 1024

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute("SELECT status, last_error FROM models WHERE id = 'm1'").fetchone()
    assert status == "pulled"
    assert err is None


async def test_run_pull_disk_shortage_marks_failed(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_disk_check(*args, **kwargs):
        raise pt.DiskShortage("not enough space")

    async def fake_estimate(*args, **kwargs):
        return None

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)

    await pt.run_pull("m1", settings)
    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute("SELECT status, last_error FROM models WHERE id = 'm1'").fetchone()
    assert status == "failed"
    assert "not enough" in err.lower()


async def test_run_pull_marks_pulling_during(tmp_data_dir, monkeypatch):
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path)

    seen_statuses = []

    async def fake_disk_check(*args, **kwargs):
        return None

    async def fake_download(model_id, settings_, repo, revision, token, allow_patterns=None):
        with sqlite3.connect(settings_.db_path) as db:
            (s,) = db.execute("SELECT status FROM models WHERE id = ?", (model_id,)).fetchone()
        seen_statuses.append(s)
        return "/tmp/fake"

    async def fake_estimate(*args, **kwargs):
        return None

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)
    assert "pulling" in seen_statuses


async def test_run_pull_short_lived_db_connections(tmp_data_dir, monkeypatch):
    """Pull must use short-lived DB connections, never one held across the
    whole pull (the old design did, which blocked progress writes and SSE
    readers). This test pins ≥2 open_db calls for any successful or failed
    pull (Phase 1 status flip + at least one further write) and exactly 1
    for a missing-row early-return path.
    """
    import app.models.pull_task as pt_mod
    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()

    def make_counting_open_db(counter: list):
        real_open_db = open_db

        @asynccontextmanager
        async def counting_open_db(path):
            counter.append(1)
            async with real_open_db(path) as db:
                yield db

        return counting_open_db

    async def fake_estimate(*a, **kw):
        return None

    monkeypatch.setattr(pt_mod, "estimate_repo_bytes", fake_estimate)

    # --- successful pull ---
    await _seed_model(settings.db_path)
    calls_success: list = []
    monkeypatch.setattr(pt_mod, "open_db", make_counting_open_db(calls_success))

    async def fake_disk_check(*a, **kw):
        return None

    async def fake_download(*a, **kw):
        return "/tmp/fake"

    monkeypatch.setattr(pt_mod, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt_mod, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)
    assert (
        len(calls_success) >= 2
    ), f"expected ≥2 open_db calls for successful pull, got {len(calls_success)}"

    # --- DiskShortage ---
    calls_shortage: list = []
    monkeypatch.setattr(pt_mod, "open_db", make_counting_open_db(calls_shortage))

    async def fake_disk_short(*a, **kw):
        raise pt.DiskShortage("no space")

    monkeypatch.setattr(pt_mod, "insufficient_disk", fake_disk_short)

    await pt.run_pull("m1", settings)
    assert (
        len(calls_shortage) >= 2
    ), f"expected ≥2 open_db calls for DiskShortage, got {len(calls_shortage)}"

    # --- missing model (early return after Phase 1) ---
    calls_missing: list = []
    monkeypatch.setattr(pt_mod, "open_db", make_counting_open_db(calls_missing))

    await pt.run_pull("nonexistent", settings)
    assert calls_missing == [
        1
    ], f"expected 1 open_db call for missing model, got {len(calls_missing)}"
