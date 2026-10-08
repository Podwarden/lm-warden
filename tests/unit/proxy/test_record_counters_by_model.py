"""``_record_counters`` writes the per-(key, model variant) minute rollup (0036).

The token page's "Usage by model" card and per-model tokens chart read
``token_model_usage_minute``; a proxy path that stopped writing it would leave
both empty with nothing failing. Pinned directly on ``_record_counters``:
  * a keyed request lands in BOTH rollups, on the same minute integer
  * two models under one key stay apart; repeats in a minute accumulate
  * two variants of one model stay apart, each under its model id
  * the variant is recorded in model_variants (INSERT OR IGNORE: first_seen
    stays the first sighting)
  * a request without a key writes no per-model row (no NULL-key orphans)
"""

import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.proxy.ledger import Ledger
from app.proxy.routes import _record_counters
from app.runtime.variants import variant_of

MINUTE = 29_000_000


async def _request(tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    # counters and model_samples carry foreign keys to both.
    with sqlite3.connect(db_path) as db:
        for mid, name in (("id-qwen", "qwen"), ("id-llama", "llama")):
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices) "
                "VALUES (?, ?, 'org/repo', '[0]')",
                (mid, name),
            )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) "
            "VALUES ('tok1', 'test', 'vw_x', 'h', 'inference')"
        )
        db.commit()
    settings = SimpleNamespace(db_path=db_path)
    # #279 stage 3: _record_counters accumulates into the ledger; tests flush it
    # (inside the same time patch) before reading. The clock looks time.time up
    # at call time so the patches below still steer the minute bucket.
    ledger = Ledger(db_path, clock=lambda: time.time())
    state = SimpleNamespace(settings=settings, ledger=ledger)
    return db_path, SimpleNamespace(app=SimpleNamespace(state=state))


def _rows(db_path, table):
    with sqlite3.connect(db_path) as db:
        return db.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3").fetchall()


async def test_a_keyed_request_is_split_by_model_and_variant(tmp_data_dir):
    db_path, req = await _request(tmp_data_dir)
    qwen = SimpleNamespace(id="id-qwen", served_model_name="qwen", dtype="auto")
    qwen_fp16 = SimpleNamespace(id="id-qwen", served_model_name="qwen", dtype="float16")
    llama = SimpleNamespace(id="id-llama", served_model_name="llama")
    v_q, v_q16, v_l = variant_of(qwen), variant_of(qwen_fp16), variant_of(llama)
    with patch("app.proxy.routes.time.time", return_value=MINUTE * 60 + 30):
        await _record_counters(req, qwen, "tok1", 100, 10, v_q)
        await _record_counters(req, qwen, "tok1", 50, 5, v_q)
        await _record_counters(req, qwen, "tok1", 1, 1, v_q16)
        await _record_counters(req, llama, "tok1", 7, 3, v_l)
        await req.app.state.ledger.flush()

    with sqlite3.connect(db_path) as db:
        usage = db.execute(
            "SELECT variant_id, model_id, minute, requests, prompt_tokens, completion_tokens "
            "FROM token_model_usage_minute WHERE token_id = 'tok1' ORDER BY 2, 4"
        ).fetchall()
        variants = dict(db.execute("SELECT id, model_id FROM model_variants").fetchall())
    assert sorted(usage) == sorted(
        [
            (v_l.id, "id-llama", MINUTE, 1, 7, 3),
            (v_q16.id, "id-qwen", MINUTE, 1, 1, 1),
            (v_q.id, "id-qwen", MINUTE, 2, 150, 15),
        ]
    )
    assert variants == {v_q.id: "id-qwen", v_q16.id: "id-qwen", v_l.id: "id-llama"}
    # Same minute integer as the per-key rollup, whose totals it splits.
    assert _rows(db_path, "token_usage_minute") == [("tok1", MINUTE, 4, 158, 19)]


async def test_first_seen_is_the_first_sighting(tmp_data_dir):
    db_path, req = await _request(tmp_data_dir)
    model = SimpleNamespace(id="id-qwen", served_model_name="qwen")
    with patch("app.runtime.variants.time.time", return_value=1000.0):
        await _record_counters(req, model, "tok1", 1, 1, variant_of(model))
        await req.app.state.ledger.flush()
    with patch("app.runtime.variants.time.time", return_value=2000.0):
        await _record_counters(req, model, "tok1", 1, 1, variant_of(model))
        await req.app.state.ledger.flush()
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT first_seen FROM model_variants").fetchall() == [(1000.0,)]


async def test_without_a_variant_it_resolves_the_running_one(tmp_data_dir):
    db_path, req = await _request(tmp_data_dir)
    model = SimpleNamespace(id="id-qwen", served_model_name="qwen")
    await _record_counters(req, model, "tok1", 1, 1)
    await req.app.state.ledger.flush()
    with sqlite3.connect(db_path) as db:
        (vid,) = db.execute("SELECT variant_id FROM token_model_usage_minute").fetchone()
    assert vid == variant_of(model).id


async def test_a_request_without_a_key_writes_no_per_model_row(tmp_data_dir):
    db_path, req = await _request(tmp_data_dir)
    model = SimpleNamespace(id="id-qwen", served_model_name="qwen")
    await _record_counters(req, model, None, 100, 10)
    await req.app.state.ledger.flush()
    assert _rows(db_path, "token_model_usage_minute") == []


async def test_cached_tokens_pass_through_and_none_is_unmeasured(tmp_data_dir):
    db_path, req = await _request(tmp_data_dir)
    model = SimpleNamespace(id="id-qwen", served_model_name="qwen")
    with patch("app.proxy.routes.time.time", return_value=MINUTE * 60 + 30):
        await _record_counters(req, model, "tok1", 100, 5, variant_of(model), 80)
        await _record_counters(req, model, "tok1", 100, 5, variant_of(model), None)
        await _record_counters(req, model, "tok1", 100, 5, variant_of(model), 0)
        await req.app.state.ledger.flush()
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT requests, prompt_tokens, cached_tokens, cached_measured_requests "
            "FROM model_samples"
        ).fetchone()
    assert row == (3, 300, 80, 2)
