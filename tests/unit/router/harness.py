"""Shared fakes for the router data-plane tests (#287 task 3).

Two upstreams, two fakes, asserted separately:

* the local engine: ``httpx.AsyncClient.send`` is patched on the class exactly
  like ``tests/unit/proxy/test_messages.py`` does, but the patch hands the
  router's own client straight to the real ``send`` so the two never collide;
* Anthropic: ``app.state.router_http`` is replaced by a client over
  ``httpx.MockTransport`` that records every request it receives.
"""

import json
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.db.repos.tokens import hash_token

PLAINTEXT = "vw_validtoken1234567890abcdef12345"
ANTHROPIC_TOKEN = "sk-ant-oat01-FAKE"
CLAUDE_CODE = {
    "X-LMWarden-Key": PLAINTEXT,
    "Authorization": f"Bearer {ANTHROPIC_TOKEN}",
    "anthropic-version": "2023-06-01",
}
USER_TEXT = "TOP-SECRET-USER-PROMPT-TEXT"


def seed(db_path, *, relay: bool = True, allowed_models: str | None = None) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]',"
            "'loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, allowed_models, "
            "anthropic_relay) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "tok1",
                "laptop",
                PLAINTEXT[:8],
                hash_token(PLAINTEXT),
                "inference",
                allowed_models,
                1 if relay else 0,
            ),
        )
        db.commit()


def ready(client, tmp_data_dir, **kw) -> None:
    client.get("/healthz")
    seed(tmp_data_dir / "vllm-warden.db", **kw)
    client.app.state.supervisor._ports["qwen"] = 19099
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text))
    client.app.state.tokenizers = cache


def enable_router(client, tmp_data_dir, rules: list[dict[str, Any]] | None = None, **settings):
    """Write the router kv keys + rule rows, then invalidate the cached rule set."""
    kv = {"router_enabled": "true", "router_upstream_url": "https://anthropic.example"}
    kv.update({f"router_{k}": str(v).lower() for k, v in settings.items()})
    ids = []
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        for k, v in kv.items():
            db.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (k, v),
            )
        for i, rule in enumerate(rules or []):
            rid = uuid.uuid4().hex
            ids.append(rid)
            db.execute(
                "INSERT INTO router_rules(id, position, pattern, target_model_id, enabled, "
                "fallback, strip_thinking, min_max_tokens) VALUES (?,?,?,?,?,?,?,?)",
                (
                    rid,
                    i,
                    rule["pattern"],
                    rule.get("target", "qwen"),
                    int(rule.get("enabled", True)),
                    int(rule.get("fallback", True)),
                    int(rule.get("strip_thinking", True)),
                    int(rule.get("min_max_tokens", 0)),
                ),
            )
        db.commit()
    client.app.state.router.invalidate()
    return ids


def _as_stream(resp: httpx.Response) -> httpx.Response:
    """A real network response is a stream; ``Response(content=bytes)`` is
    already consumed. Re-wrap it so the relay reads it the way it would in
    production."""
    if not resp.is_stream_consumed:
        return resp
    data = resp.content

    async def gen():
        yield data

    return httpx.Response(resp.status_code, headers=resp.headers, content=gen())


class Anthropic:
    """Recorder over the Anthropic fake."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    @property
    def called(self) -> bool:
        return bool(self.requests)


def fake_anthropic(
    client,
    handler: Callable[[httpx.Request], Awaitable[httpx.Response] | httpx.Response] | None = None,
) -> Anthropic:
    rec = Anthropic()

    async def _handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        rec.requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query.decode(),
                "host": request.url.host,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": body,
            }
        )
        if handler is None:
            out: Any = httpx.Response(
                200,
                json={"id": "msg_up", "type": "message", "usage": {"input_tokens": 5}},
                headers={"request-id": "req_up"},
            )
        else:
            out = handler(request)
            if not isinstance(out, httpx.Response):
                out = await out
        return _as_stream(out)

    client.app.state.router_http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), follow_redirects=False, trust_env=False
    )
    return rec


@contextmanager
def fake_engine(
    client, handler: Callable[[httpx.Request], Awaitable[Any]] | None = None
) -> Iterator[list[httpx.Request]]:
    """Fake the local engine; the router's own client is passed to the real send."""
    calls: list[httpx.Request] = []
    real_send = httpx.AsyncClient.send
    router_http = client.app.state.router_http

    async def send(self, request, **kw):
        if self is router_http or self is client.app.state.router_http:
            return await real_send(self, request, **kw)
        calls.append(request)
        if handler is None:
            return chat_completion()
        return await handler(request)

    with patch.object(httpx.AsyncClient, "send", send):
        yield calls


def json_upstream(body: dict[str, Any], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def chat_completion(text: str = "Hello there") -> MagicMock:
    return json_upstream(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }
    )


def chunk(delta=None, finish=None, usage=None) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "qwen",
        "choices": [],
    }
    if delta is not None or finish is not None:
        ev["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        ev["usage"] = usage
    return ev


def sse_upstream(events: list[dict[str, Any]], *, then: BaseException | None = None) -> MagicMock:
    frames = [b"data: " + json.dumps(ev).encode() + b"\n\n" for ev in events]
    if then is None:
        frames.append(b"data: [DONE]\n\n")

    async def aiter():
        for f in frames:
            yield f
        if then is not None:
            raise then

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def good_stream() -> MagicMock:
    usage = {"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9}
    return sse_upstream(
        [
            chunk({"role": "assistant", "content": ""}, usage=usage),
            chunk({"content": "Hel"}, usage={**usage, "completion_tokens": 1}),
            chunk({"content": "lo"}, usage={**usage, "completion_tokens": 2}),
            chunk({}, finish="stop", usage={**usage, "completion_tokens": 2}),
        ]
    )


def haiku_body(**extra: Any) -> dict[str, Any]:
    body = {
        "model": "claude-haiku-4-5",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": USER_TEXT}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    body.update(extra)
    return body


def decisions(client) -> list[dict[str, Any]]:
    return client.app.state.router.decisions(200)


def sse_events(raw: bytes) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for frame in raw.decode().split("\n\n"):
        if not frame.strip():
            continue
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out
