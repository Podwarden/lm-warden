"""max_tokens clamp on the plain local /v1/messages path (router off, or a served name)."""

import json
import sqlite3

from app.proxy import model_cache
from tests.unit.router.harness import (
    CLAUDE_CODE,
    chat_completion,
    enable_router,
    fake_engine,
    ready,
)


def _engine(_req):
    async def h(_r):
        return chat_completion()

    return h


def _body(chars=3000, max_tokens=32000):
    return {
        "model": "qwen",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "x" * chars}],
    }


def _ctx(tmp_data_dir, value):
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET max_model_len=? WHERE id='qwen'", (value,))
        db.commit()
    # A hand edit bypasses ModelRepo, so drop the /v1 model cache (#293) the
    # way every ModelRepo write does.
    model_cache.invalidate_all()


def _sent(client, body):
    with fake_engine(client, _engine(None)) as calls:
        r = client.post("/v1/messages", headers=CLAUDE_CODE, json=body)
    assert r.status_code == 200, r.text
    return json.loads(calls[0].content)["max_tokens"]


def test_router_off_clamps_to_remaining_context(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    _ctx(tmp_data_dir, 4096)
    assert _sent(client, _body()) < 4096 - 3000


def test_served_name_with_router_on_clamps(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    enable_router(client, tmp_data_dir, [{"pattern": "claude-haiku*"}])
    _ctx(tmp_data_dir, 4096)
    assert _sent(client, _body()) < 4096 - 3000


def test_small_request_and_unknown_window_untouched(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    _ctx(tmp_data_dir, 4096)
    assert _sent(client, _body(chars=10, max_tokens=500)) == 500
    _ctx(tmp_data_dir, None)
    assert _sent(client, _body()) == 32000


def test_prompt_over_context_is_not_clamped(tmp_data_dir, client):
    ready(client, tmp_data_dir)
    _ctx(tmp_data_dir, 4096)
    with fake_engine(client, _engine(None)) as calls:
        client.post("/v1/messages", headers=CLAUDE_CODE, json=_body(chars=5000))
    assert json.loads(calls[0].content)["max_tokens"] == 32000
