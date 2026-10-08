"""#286 re-review N4b -- clean parallel flags out of saved user templates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.runtime.boot_reconcile import reconcile_legacy_template_parallel_flags
from app.templates import store
from app.templates.registry import ModelTemplate


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(db_path=tmp_path / "vllm-warden.db")


async def _seed(tmp_path: Path, tid: str, *, tp: int = 1, dp: int = 1, extra_args: list[str]):
    async with open_db(tmp_path / "vllm-warden.db") as db:
        await apply_migrations(db)
        await store.save_user_template(
            db,
            ModelTemplate(
                id=tid,
                label=tid,
                hf_repo="org/model",
                hf_revision="main",
                dtype="auto",
                max_model_len=4096,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=extra_args,
                source="user",
                data_parallel_size=dp,
            ),
        )


async def _get(tmp_path: Path, tid: str) -> ModelTemplate:
    async with open_db(tmp_path / "vllm-warden.db") as db:
        t = await store.get_template(db, tid)
    assert t is not None
    return t


async def test_dp_flag_is_promoted_into_the_template_field(tmp_path):
    await _seed(
        tmp_path, "t1", extra_args=["--max-num-seqs", "8", "-dp", "7", "--tensor_parallel_size=1"]
    )
    out = await reconcile_legacy_template_parallel_flags(_settings(tmp_path))
    assert out == [("t1", "promoted tp=1 dp=7")]
    t = await _get(tmp_path, "t1")
    assert (t.tensor_parallel_size, t.data_parallel_size) == (1, 7)
    assert t.extra_args == ["--max-num-seqs", "8"]


async def test_redundant_flags_are_stripped(tmp_path):
    await _seed(tmp_path, "t2", tp=2, extra_args=["--tensor-parallel-size", "2", "--keep"])
    out = await reconcile_legacy_template_parallel_flags(_settings(tmp_path))
    assert out[0][1].startswith("cleaned")
    t = await _get(tmp_path, "t2")
    assert (t.tensor_parallel_size, t.data_parallel_size, t.extra_args) == (2, 1, ["--keep"])


async def test_unfixable_template_is_left_and_logged(tmp_path, caplog):
    await _seed(tmp_path, "t3", extra_args=["--data-parallel-size", "many"])
    await _seed(tmp_path, "t4", extra_args=["-pp", "2"])
    with caplog.at_level("WARNING"):
        out = await reconcile_legacy_template_parallel_flags(_settings(tmp_path))
    assert sorted(i for i, _ in out) == ["t3", "t4"] and all(n.startswith("left") for _, n in out)
    assert "template 't3'" in caplog.text and "template 't4'" in caplog.text
    assert (await _get(tmp_path, "t3")).extra_args == ["--data-parallel-size", "many"]


async def test_clean_templates_and_second_run_are_noops(tmp_path):
    await _seed(tmp_path, "ok", extra_args=["--keep"])
    await _seed(tmp_path, "t1", extra_args=["-dp", "4"])
    assert [i for i, _ in await reconcile_legacy_template_parallel_flags(_settings(tmp_path))] == [
        "t1"
    ]
    assert await reconcile_legacy_template_parallel_flags(_settings(tmp_path)) == []
