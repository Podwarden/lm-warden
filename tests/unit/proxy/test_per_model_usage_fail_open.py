"""The per-model rollup (0036) is bookkeeping: it must never fail a response.

``_record_counters`` runs AFTER the upstream replied -- in the non-stream path
before the response is returned, in the stream generator's ``finally``. A DB
error in the variant/per-model write used to 500 a successful non-stream
response and raise out of the stream's ``finally``. Pinned for both paths:
the client still gets the upstream reply, the per-key rollup
(token_usage_minute) is still recorded, and nothing per-model is half-written.
"""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.repos.tokens import TokenModelUsageRepo
from tests.unit.proxy.test_proxy_accounting import _make_fake_tokenizer, _seed_loaded

AUTH = "Bearer vw_validtoken1234567890abcdef12345"


async def _boom(*_a, **_kw):
    raise sqlite3.OperationalError("disk I/O error")


def _ready(tmp_data_dir, client, monkeypatch):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 3 if text else 0)
    monkeypatch.setattr(TokenModelUsageRepo, "add", _boom)
    return db_path


def _tables(db_path):
    with sqlite3.connect(db_path) as db:
        per_key = db.execute(
            "SELECT token_id, requests FROM token_usage_minute"
        ).fetchall()
        per_model = db.execute("SELECT COUNT(*) FROM token_model_usage_minute").fetchone()[0]
        variants = db.execute("SELECT COUNT(*) FROM model_variants").fetchone()[0]
    return per_key, per_model, variants


def test_non_stream_response_survives_a_failing_per_model_write(
    tmp_data_dir, client, monkeypatch
):
    db_path = _ready(tmp_data_dir, client, monkeypatch)
    body = {
        "id": "x", "model": "qwen",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "application/json"}
    fake.aread = AsyncMock(return_value=json.dumps(body).encode())
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)):
        r = client.post(
            "/v1/chat/completions", headers={"Authorization": AUTH},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    # The per-key row is kept; the failed statement's transaction-mates (the
    # variant INSERT) are committed with it, which is harmless and correct.
    per_key, per_model, _ = _tables(db_path)
    assert per_key == [("tok1", 1)]
    assert per_model == 0


@pytest.mark.parametrize("chunks", [[
    b'data: {"choices":[{"delta":{"content":"hello"}}],"model":"qwen"}\n\n',
    b'data: [DONE]\n\n',
]])
def test_stream_survives_a_failing_per_model_write(tmp_data_dir, client, monkeypatch, chunks):
    db_path = _ready(tmp_data_dir, client, monkeypatch)

    async def aiter():
        for c in chunks:
            yield c

    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "text/event-stream"}
    fake.aiter_bytes = aiter
    fake.aclose = AsyncMock()
    got = b""
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)):
        with client.stream(
            "POST", "/v1/chat/completions", headers={"Authorization": AUTH},
            json={"model": "qwen", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            for b in r.iter_bytes():
                got += b
            assert r.status_code == 200
    assert b"hello" in got and b"[DONE]" in got
    per_key, per_model, _ = _tables(db_path)
    assert per_key == [("tok1", 1)]
    assert per_model == 0


def test_a_failing_variant_insert_is_retried_on_the_next_request(tmp_data_dir, client, monkeypatch):
    # The in-process "already recorded" set is only updated after a commit, so
    # a failed first attempt does not stop the next request from naming it.
    from app.runtime import variants

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 3 if text else 0)
    real_ensure = variants.ModelVariantRepo.ensure
    monkeypatch.setattr(variants.ModelVariantRepo, "ensure", _boom)

    body = {"id": "x", "model": "qwen", "choices": [{"message": {"content": "ok"}}]}

    def post():
        fake = MagicMock()
        fake.status_code = 200
        fake.headers = {"content-type": "application/json"}
        fake.aread = AsyncMock(return_value=json.dumps(body).encode())
        fake.aclose = AsyncMock()
        with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)):
            return client.post(
                "/v1/chat/completions", headers={"Authorization": AUTH},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert post().status_code == 200
    assert _tables(db_path)[2] == 0
    monkeypatch.setattr(variants.ModelVariantRepo, "ensure", real_ensure)
    assert post().status_code == 200
    per_key, per_model, named = _tables(db_path)
    assert per_key == [("tok1", 2)]
    assert (per_model, named) == (1, 1)
