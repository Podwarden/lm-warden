"""``_forward``'s keyword-only ``affinity_source`` / ``session_id`` overrides.

A route that swaps the request body for a translated one (``/v1/responses``)
derives the replica key and the session from the ORIGINAL body and hands them
in. Without the overrides ``_forward`` behaves exactly as before.
"""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import Request

from app.db.repos.tokens import TokenRow, hash_token
from app.proxy.dp_affinity import RANK_HEADER
from app.proxy.routes import _forward

PLAINTEXT = "vw_validtoken1234567890abcdef12345"
CHUNKS = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'] * 2
BODY = {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hello there"}]}


def _ready(client, tmp_data_dir, dp=4):
    client.get("/healthz")
    gpus = list(range(dp))
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": gpus}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, data_parallel_size, dp_affinity_enabled, dtype, "
            "max_model_len, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,?,1,'auto',4096,0.9,0,'[]',"
            "'loaded',0,NULL,NULL)",
            (json.dumps(gpus), dp),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", PLAINTEXT[:8], hash_token(PLAINTEXT), "inference"),
        )
        db.commit()
    client.app.state.supervisor._ports["qwen"] = 19099
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text))
    client.app.state.tokenizers = cache


def _run(client, headers=None, **overrides):
    """Open one streamed ``_forward``; returns (live row, upstream request)."""
    raw = json.dumps(BODY).encode()
    app = client.app
    seen: list = []

    async def _send(request, *, stream=False, **kw):
        seen.append(request)
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/event-stream"}

        async def aiter():
            for c in CHUNKS:
                yield c

        resp.aiter_bytes = aiter
        resp.aclose = AsyncMock()
        return resp

    async def go():
        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")]
            + [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "app": app,
            "query_string": b"",
            "client": ("127.0.0.1", 1),
        }
        request = Request(scope, receive)
        from app.db.database import open_db
        from app.db.repos.models import ModelRepo

        async with open_db(app.state.settings.db_path) as db:
            model = await ModelRepo(db).get_by_served_name("qwen")
        token = TokenRow(
            id="tok1",
            name="test",
            prefix="vw_valid",
            scope="inference",
            allowed_models=None,
            rate_limit_rpm=None,
            rate_limit_tpm=None,
            revoked_at=None,
            last_used_at=None,
            created_at="2026-01-01",
            expires_at=None,
            rotated_at=None,
            rotated_from=None,
        )
        with patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=_send)):
            resp = await _forward(
                request, model, "127.0.0.1", 19099, "/v1/chat/completions", token, **overrides
            )
            it = resp.body_iterator
            await it.__anext__()
            rows = app.state.request_registry.snapshot()
            await it.aclose()
        return rows[0], seen[0]

    return client.portal.call(go)


def test_overrides_set_the_session_and_the_replica_key(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    row, up = _run(
        client,
        affinity_key="thread-0001-abcd",
        affinity_source="prompt_cache_key",
        session_id="thread-0001-abcd",
    )
    assert row.session_id == "thread-0001-abcd"
    assert row.session_source == "prompt_cache_key"
    snap = client.app.state.dp_routing.snapshot("qwen", 4)
    assert snap["totals"]["placed"] == 1
    assert RANK_HEADER in up.headers


def test_prompt_hash_override_shows_no_session_but_the_source(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    row, _ = _run(client, affinity_key="tok1\x00hi", affinity_source="prompt_hash")
    assert row.session_id is None
    assert row.session_source == "prompt_hash"


def test_parent_session_still_comes_from_the_headers(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    row, _ = _run(
        client,
        headers={"x-codex-parent-thread-id": "thread-parent-01"},
        affinity_key="thread-0001-abcd",
        affinity_source="session_id_header",
        session_id="thread-0001-abcd",
    )
    assert row.parent_session_id == "thread-parent-01"


def test_without_overrides_the_translated_body_is_read_as_before(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    row, _ = _run(client, headers={"X-Session-Id": "sess-0000009"})
    assert (row.session_id, row.session_source) == ("sess-0000009", "x_session_id")
