import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        yield conn


async def test_insert_and_get_model(db):
    repo = ModelRepo(db)
    await repo.insert(
        ModelRow(
            id="m1",
            served_model_name="qwen3.5-9b",
            hf_repo="Qwen/Qwen3.5-9B",
            hf_revision="main",
            gpu_indices=[1, 2],
            tensor_parallel_size=2,
            dtype="auto",
            max_model_len=4096,
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
    m = await repo.get("m1")
    assert m.served_model_name == "qwen3.5-9b"
    assert m.gpu_indices == [1, 2]
    assert m.extra_args == []


async def test_update_status(db):
    repo = ModelRepo(db)
    await repo.insert(_make_row("m1"))
    await repo.update_status("m1", "loaded")
    m = await repo.get("m1")
    assert m.status == "loaded"


async def test_note_progress_writes_last_error_only(db):
    """#282 — a soft-deadline progress note must not disturb ``status`` or
    ``prior_status``. Writing it through ``update_status`` instead would
    silently clear ``prior_status`` on a row still mid-restore (see
    ``note_progress``'s own docstring)."""
    repo = ModelRepo(db)
    await repo.insert(_make_row("m1", status="loading"))
    await repo.set_prior_status("m1", "loaded")

    await repo.note_progress("m1", "still loading after 600s (engine process alive)")

    m = await repo.get("m1")
    assert m.status == "loading"
    assert m.last_error == "still loading after 600s (engine process alive)"
    row = await db.execute("SELECT prior_status FROM models WHERE id = 'm1'")
    assert (await row.fetchone())[0] == "loaded"


async def test_set_status_failed_on_startup_wipes_loaded(db):
    """At app startup, any 'loaded'/'loading'/'unloading'/'pulling' rows
    must be marked failed."""
    repo = ModelRepo(db)
    await repo.insert(_make_row("m1", status="loaded"))
    await repo.insert(_make_row("m2", status="loading"))
    await repo.insert(_make_row("m3", status="registered"))
    n = await repo.mark_runtime_dead_on_startup()
    assert n == 2
    assert (await repo.get("m1")).status == "failed"
    assert (await repo.get("m2")).status == "failed"
    assert (await repo.get("m3")).status == "registered"


async def test_mark_runtime_dead_on_startup_includes_pulling_and_zeros_progress(db):
    """#11 — restart while a row is mid-pull. The row must transition to
    'failed' AND its pull progress counters must be zeroed so the UI does
    not show stale progress for a pull that no longer has a backing task.
    """
    repo = ModelRepo(db)
    await repo.insert(
        _make_row(
            "puller",
            status="pulling",
            pulled_bytes=12345,
            pulled_total=99999,
        )
    )
    # A bystander row whose progress is non-zero but whose status is not
    # in the wipe set MUST be untouched (regression guard).
    await repo.insert(
        _make_row(
            "bystander",
            status="pulled",
            pulled_bytes=77,
            pulled_total=88,
        )
    )
    n = await repo.mark_runtime_dead_on_startup()
    assert n == 1
    puller = await repo.get("puller")
    assert puller.status == "failed"
    assert puller.pulled_bytes == 0
    assert puller.pulled_total == 0
    bystander = await repo.get("bystander")
    assert bystander.status == "pulled"
    assert bystander.pulled_bytes == 77
    assert bystander.pulled_total == 88


def _make_row(
    model_id: str,
    status: str = "registered",
    *,
    pulled_bytes: int = 0,
    pulled_total: int | None = None,
) -> ModelRow:
    return ModelRow(
        id=model_id,
        served_model_name=f"name-{model_id}",
        hf_repo="org/repo",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        extra_env={},
        status=status,
        pulled_bytes=pulled_bytes,
        pulled_total=pulled_total,
        last_error=None,
    )
