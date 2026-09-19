"""The proxy writes the key's row id onto every request_history row.

Pinned end to end through ``_forward`` rather than only on
``finished_record``: the series endpoint (GET /api/tokens/{id}/series) filters
timings by this column, so a proxy path that dropped it would leave every
token page's latency charts empty with nothing failing.
"""

import asyncio
import sqlite3
import time

from app.db.database import open_db
from app.db.repos.models import ModelRepo
from tests.unit.proxy.test_queue_wait_recorded import _fake_tokenizer, _post, _seed_loaded


def test_a_proxied_request_records_its_token_id(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)  # token row id "tok1", name "test"
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()

    assert _post(client, plaintext).status_code == 200

    # The store writes from a background task; wait for it.
    deadline = time.time() + 30
    rows: list = []
    while time.time() < deadline:
        with sqlite3.connect(db_path) as db:
            rows = db.execute("SELECT token_name, token_id FROM request_history").fetchall()
        if rows:
            break
        time.sleep(0.05)
    assert rows == [("test", "tok1")]


def test_a_proxied_request_records_its_variant_in_history_and_usage(tmp_data_dir, client):
    # 0036: request_history and the per-model rollup name the SAME variant --
    # the running engine's. This engine was not launched by the supervisor
    # (the test wires its port by hand), so the variant is computed from the
    # row.
    from app.runtime.variants import variant_of

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _fake_tokenizer()

    assert _post(client, plaintext).status_code == 200

    deadline = time.time() + 30
    rows: list = []
    while time.time() < deadline:
        with sqlite3.connect(db_path) as db:
            rows = db.execute("SELECT variant_id FROM request_history").fetchall()
        if rows:
            break
        time.sleep(0.05)
    cached = variant_of(_model_row(db_path))
    assert rows == [(cached.id,)]
    with sqlite3.connect(db_path) as db:
        usage = db.execute(
            "SELECT variant_id, model_id FROM token_model_usage_minute WHERE token_id = 'tok1'"
        ).fetchall()
        named = db.execute("SELECT id FROM model_variants").fetchall()
    assert usage == [(cached.id, "qwen")]
    assert named == [(cached.id,)]


def _model_row(db_path):
    async def _get():
        async with open_db(db_path) as db:
            return await ModelRepo(db).get("qwen")

    return asyncio.run(_get())
