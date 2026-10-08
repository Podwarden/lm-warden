"""#286 -- promote a legacy ``--data-parallel-size`` flag in extra_args into
the ``data_parallel_size`` / ``tensor_parallel_size`` columns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.runtime.boot_reconcile import reconcile_legacy_data_parallel_rows


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(db_path=tmp_path / "vllm-warden.db")


async def _seed(tmp_path: Path, model_id: str, *, tp: int, extra_args: list[str], n_gpus=7) -> None:
    async with open_db(tmp_path / "vllm-warden.db") as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id=model_id,
                served_model_name=model_id,
                hf_repo="org/model",
                hf_revision="main",
                gpu_indices=list(range(n_gpus)),
                tensor_parallel_size=tp,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=extra_args,
                extra_env={},
                status="pulled",
                pulled_bytes=0,
                pulled_total=None,
                last_error=None,
            )
        )


async def _row(tmp_path: Path, model_id: str) -> ModelRow:
    async with open_db(tmp_path / "vllm-warden.db") as db:
        row = await ModelRepo(db).get(model_id)
    assert row is not None
    return row


async def test_hack_row_is_promoted(tmp_path):
    await _seed(
        tmp_path,
        "m1",
        tp=7,
        extra_args=[
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "7",
            "--max-num-seqs",
            "32",
        ],  # fmt: skip
    )
    out = await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    assert out == [("m1", "promoted tp=1 dp=7")]
    row = await _row(tmp_path, "m1")
    assert (row.tensor_parallel_size, row.data_parallel_size) == (1, 7)
    assert row.extra_args == ["--max-num-seqs", "32"]


async def test_dp_flag_without_tp_flag_derives_tp(tmp_path):
    await _seed(tmp_path, "m1", tp=7, extra_args=["--data-parallel-size=7"])
    await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    row = await _row(tmp_path, "m1")
    assert (row.tensor_parallel_size, row.data_parallel_size) == (1, 7)
    assert row.extra_args == []


async def test_mismatched_product_is_left_and_reported(tmp_path):
    args = ["--tensor-parallel-size", "2", "--data-parallel-size", "7"]
    await _seed(tmp_path, "m1", tp=7, extra_args=list(args))
    out = await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    assert len(out) == 1 and out[0][0] == "m1" and out[0][1].startswith("left:")
    row = await _row(tmp_path, "m1")
    assert (row.tensor_parallel_size, row.data_parallel_size) == (7, 1)
    assert row.extra_args == args


async def test_row_without_flags_is_untouched(tmp_path):
    await _seed(tmp_path, "m1", tp=7, extra_args=["--max-num-seqs", "8"])
    assert await reconcile_legacy_data_parallel_rows(_settings(tmp_path)) == []
    row = await _row(tmp_path, "m1")
    assert (row.tensor_parallel_size, row.data_parallel_size) == (7, 1)


async def test_second_run_is_a_noop(tmp_path):
    await _seed(tmp_path, "m1", tp=7, extra_args=["--data-parallel-size", "7"])
    assert len(await reconcile_legacy_data_parallel_rows(_settings(tmp_path))) == 1
    assert await reconcile_legacy_data_parallel_rows(_settings(tmp_path)) == []


def test_main_calls_the_pass_after_the_poisoned_fields_pass():
    src = (Path(__file__).parents[3] / "app" / "main.py").read_text()
    a = src.index("await reconcile_poisoned_extra_fields(settings)")
    b = src.index("await reconcile_legacy_data_parallel_rows(settings)")
    assert a < b


# --- #286 review #2/#3/#8 -------------------------------------------------


async def test_redundant_flags_matching_the_columns_are_stripped(tmp_path):
    await _seed(
        tmp_path,
        "m1",
        tp=2,
        n_gpus=2,
        extra_args=["--tensor-parallel-size", "2", "-dp", "1", "--max-num-seqs", "8"],
    )
    out = await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    assert len(out) == 1 and out[0][0] == "m1" and out[0][1].startswith("cleaned")
    row = await _row(tmp_path, "m1")
    assert row.extra_args == ["--max-num-seqs", "8"]
    assert (row.tensor_parallel_size, row.data_parallel_size) == (2, 1)
    assert await reconcile_legacy_data_parallel_rows(_settings(tmp_path)) == []


async def test_unfixable_flags_are_left_warned_and_logged(tmp_path, caplog):
    import logging

    # tp column 2 but the flag says 1: cannot be fixed without guessing
    args = ["--tensor-parallel-size", "1"]
    await _seed(tmp_path, "m1", tp=2, n_gpus=2, extra_args=list(args))
    with caplog.at_level(logging.WARNING):
        out = await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    assert out and out[0][1].startswith("left:")
    assert "m1" in caplog.text
    row = await _row(tmp_path, "m1")
    assert row.extra_args == args


async def test_short_alias_hack_row_is_promoted(tmp_path):
    await _seed(tmp_path, "m1", tp=7, extra_args=["-tp", "1", "-dp", "7"])
    out = await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    assert out == [("m1", "promoted tp=1 dp=7")]
    row = await _row(tmp_path, "m1")
    assert (row.tensor_parallel_size, row.data_parallel_size, row.extra_args) == (1, 7, [])


async def test_outcomes_are_stored_as_a_layout_notice(tmp_path):
    await _seed(tmp_path, "promo", tp=7, extra_args=["--data-parallel-size", "7"])
    await _seed(tmp_path, "bad", tp=7, extra_args=["--tensor-parallel-size", "2", "-dp", "7"])
    await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    promo = await _row(tmp_path, "promo")
    assert promo.layout_notice_level == "info"
    assert "data_parallel_size=7" in (promo.layout_notice or "")
    bad = await _row(tmp_path, "bad")
    assert bad.layout_notice_level == "warning"
    assert "extra_args" in (bad.layout_notice or "")


async def test_warning_clears_once_the_row_is_clean(tmp_path):
    await _seed(tmp_path, "bad", tp=7, extra_args=["--tensor-parallel-size", "2", "-dp", "7"])
    await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    async with open_db(tmp_path / "vllm-warden.db") as db:
        await ModelRepo(db).update_fields("bad", {"extra_args": []})
    await reconcile_legacy_data_parallel_rows(_settings(tmp_path))
    row = await _row(tmp_path, "bad")
    assert row.layout_notice is None and row.layout_notice_level is None
