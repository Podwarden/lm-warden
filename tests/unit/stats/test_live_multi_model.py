"""``GET /api/stats/live`` with more than one engine loaded.

``_loaded_model`` ended in ``LIMIT 1``, so on a box running two engines the live
dashboard reported one of them — and which one was whatever SQLite returned
first, so it was not even a stable answer, let alone a stated one.

Two properties are pinned here and they pull in opposite directions, which is
why both are tested:

  * the frame carries EVERY loaded model, so a client can choose;
  * ``build_frame``'s per-model block is BYTE-IDENTICAL to what it always
    produced, because eleven dashboard panels read it field by field and its
    shape is committed as a golden.
"""

import sqlite3

from app.stats.live_engine import _loaded_models, _null_frame, envelope
from tests.conftest import seed_admin_user


def _insert(db, mid, name, backend, ctx=8192):
    db.execute(
        "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
        "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
        "gpu_memory_utilization, trust_remote_code, extra_args, status, backend) "
        f"VALUES ('{mid}', '{name}', 'o/r', 'main', '[0]', 1, "
        f"NULL, {ctx}, 0.9, 0, '[]', 'loaded', {backend})"
    )
    db.execute(f"INSERT INTO model_runtime(model_id, pid, port) VALUES ('{mid}', 1, 10000)")


async def test_every_loaded_model_is_returned(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    with sqlite3.connect(db_path) as db:
        _insert(db, "m-qwen", "qwen3.8-27b", "'llamacpp'")
        _insert(db, "m-llama", "llama-3.1-8b", "'vllm'")
        db.commit()

    rows = await _loaded_models(client.app.state.settings.db_path)
    assert len(rows) == 2
    # Ordered by served name: a view that repaints every two seconds cannot
    # have its panels swap places between ticks.
    assert [r[1] for r in rows] == ["llama-3.1-8b", "qwen3.8-27b"]
    assert [r[3] for r in rows] == ["vllm", "llamacpp"]


async def test_no_loaded_model_is_an_empty_list(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    assert await _loaded_models(client.app.state.settings.db_path) == []


async def test_a_loaded_row_without_a_runtime_sibling_is_not_reported(tmp_data_dir, client):
    """Same strictness as before: a 'loaded' row the supervisor does not own is
    stale state, and scraping a port it never opened would surface as an error
    panel for a model that is not running."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
            "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
            "gpu_memory_utilization, trust_remote_code, extra_args, status) "
            "VALUES ('m-orphan', 'orphan', 'o/r', 'main', '[0]', 1, "
            "NULL, NULL, 0.9, 0, '[]', 'loaded')"
        )
        db.commit()
    assert await _loaded_models(client.app.state.settings.db_path) == []


# ---------------------------------------------------------------------------
# The envelope.
# ---------------------------------------------------------------------------


def _block(name, model_id):
    return _null_frame(name, model_id, 8192, "engine not running", "vllm")


def test_envelope_lists_every_block():
    out = envelope([_block("a", "m-a"), _block("b", "m-b")])
    assert [b["model"] for b in out["models"]] == ["a", "b"]


def test_envelope_does_not_touch_a_block():
    """The wrapper must not edit what it wraps.

    build_frame's output is pinned by a committed golden because the dashboard
    reads it field by field; multi-model support is a list around that shape,
    never a change to it.
    """
    block = _block("a", "m-a")
    before = dict(block)
    out = envelope([block])
    assert out["models"][0] == before


def test_envelope_mirrors_the_leading_block_at_the_top_level():
    """ui and api ship as separate images and skew in both directions, so a
    bundle that predates `models` must keep reading the frame it knows rather
    than rendering nothing while an engine serves."""
    out = envelope([_block("a", "m-a"), _block("b", "m-b")])
    assert out["model"] == "a"
    assert out["model_id"] == "m-a"
    assert out["backend"] == "vllm"


def test_envelope_with_no_models_still_says_so_both_ways():
    out = envelope([])
    assert out["models"] == []
    assert out["model"] is None
    assert out["scrape_error"] == "no model loaded"
