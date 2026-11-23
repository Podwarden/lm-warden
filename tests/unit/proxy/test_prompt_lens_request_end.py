"""G5 + G-F1: ``request_end`` carries usage, timings and cache, once per request, on every exit.

The exit matrix below drives every way ``_forward`` can end after ``request_start`` is
published (normal finish, client abort while streaming and while queued, engine error,
runaway stop, queue refusal, a raise before the live row is registered, ...). Each exit
runs twice, in two fresh apps: Prompt Lens on (a real hub) and Prompt Lens off (a spy hub).

  * on: exactly one ``request_start`` and exactly one ``request_end``, with the exit's
    finish reason;
  * off: not one call on the hub;
  * both: the same client-visible outcome (status and bytes), the same request-history
    row, the same stats-ledger call and an empty live registry. Prompt Lens only observes.
"""

import asyncio
import contextlib
import dataclasses
import json
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.cobs2.keys import NetworkKey
from app.db.repos.tokens import TokenRow, hash_token
from app.proxy import routes
from app.proxy.request_id import REQUEST_ID_HEADER
from app.proxy.request_registry import LiveRequest, finished_record

_PLAIN = "vw_validtoken1234567890abcdef12345"
_AUTH = {"Authorization": f"Bearer {_PLAIN}"}
_MSGS = [{"role": "user", "content": "hi there"}]
_STREAM = {"model": "qwen", "stream": True, "messages": _MSGS}
_NONSTREAM = {"model": "qwen", "messages": _MSGS}

#: Every key the enriched request_end carries (G5).
_END_KEYS = {
    "type",
    "req_id",
    "token_id",
    "ts",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "queued_s",
    "ttft_s",
    "duration_s",
    "decode_tps",
    "own_reused_tokens",
    "cache",
}


# --------------------------------------------------------------------------
# Harness: a fresh app per run, seeded the same way every time
# --------------------------------------------------------------------------
def _seed(db_path: Path) -> None:
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
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
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "my-token", _PLAIN[:8], hash_token(_PLAIN), "inference"),
        )
        db.commit()


@pytest.fixture
def fresh_app(
    tmp_data_dir: Path, migrated_db_template: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., contextlib.AbstractContextManager[TestClient]]:
    n = [0]

    @contextlib.contextmanager
    def make(*, gm: bool, settings: dict[str, Any] | None = None) -> Iterator[TestClient]:
        n[0] += 1
        d = tmp_data_dir / f"app{n[0]}"
        d.mkdir()
        monkeypatch.setenv("VW_DATA_DIR", str(d))
        monkeypatch.setenv("VW_HF_CACHE_DIR", str(d / "hf-cache"))
        shutil.copyfile(migrated_db_template, d / "vllm-warden.db")
        from app.main import build_app

        with TestClient(build_app()) as client:
            client.get("/healthz")
            _seed(d / "vllm-warden.db")
            client.app.state.supervisor._ports["qwen"] = 19099
            tok = MagicMock()
            tok.count = AsyncMock(
                side_effect=lambda repo, text, *, fallback_repo=None: (
                    len(text.split()) if text else 0
                )
            )
            client.app.state.tokenizers = tok
            # One cobs2 key for every app, so the two runs' ids (and history rows) compare.
            client.app.state.cobs2_key = NetworkKey(epoch=1, key=b"k" * 32)
            client.app.state.settings = dataclasses.replace(
                client.app.state.settings, prompt_lens_enabled=gm, **(settings or {})
            )
            if not gm:
                client.app.state.prompt_lens_hub = MagicMock()
            yield client

    return make


_VOLATILE_ROW = {"id", "started_iso", "duration_s", "ttft_s", "queued_s", "gap_s"}


@dataclasses.dataclass
class Run:
    outcome: Any
    events: list[dict[str, Any]] | None
    spy: MagicMock | None
    rows: list[dict[str, Any]]
    ledger: list[dict[str, Any]]
    live_left: int


def _run(fresh_app: Any, scenario: "Scenario", *, gm: bool) -> Run:
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    real_record = routes.record_finished

    def record(history: Any, req: LiveRequest, kv: Any, *, now: float, **kw: Any) -> None:
        # Pass every keyword through (iso_star_frac, Plan 2): a shim with a
        # narrower signature raised inside the bookkeeping guard, so no row
        # was ever seen.
        real_record(history, req, kv, now=now, **kw)
        row = finished_record(req, now=now)
        rows.append({k: v for k, v in row.items() if k not in _VOLATILE_ROW})

    with fresh_app(gm=gm, settings=scenario.settings) as client:
        led = client.app.state.ledger
        real_rr = led.record_request

        def record_request(**kw: Any) -> Any:
            ledger.append({k: v for k, v in kw.items() if k not in {"produced_s", "variant"}})
            return real_rr(**kw)

        with (
            patch.object(routes, "record_finished", record),
            patch.object(led, "record_request", record_request),
        ):
            outcome = scenario.run(client)
        hub = client.app.state.prompt_lens_hub
        live_left = len(client.app.state.request_registry.snapshot())
        if gm:
            return Run(outcome, hub.events(), None, rows, ledger, live_left)
        return Run(outcome, None, hub, rows, ledger, live_left)


# --------------------------------------------------------------------------
# Fake upstreams
# --------------------------------------------------------------------------
def _ev(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> bytes:
    ev: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1000,
        "model": "qwen",
        "choices": [],
    }
    if delta is not None or finish is not None:
        ev["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        ev["usage"] = usage
    return b"data: " + json.dumps(ev).encode() + b"\n\n"


_USAGE = {
    "prompt_tokens": 2,
    "completion_tokens": 3,
    "prompt_tokens_details": {"cached_tokens": 1},
    "completion_tokens_details": {"reasoning_tokens": 1},
}
_OK_CHUNKS = [
    _ev({"role": "assistant", "reasoning_content": "hm"}),
    _ev({"content": "hello"}),
    _ev({"content": " world"}, finish="stop"),
    _ev(usage=_USAGE),
    b"data: [DONE]\n\n",
]
_RUNAWAY_CHUNKS = [_ev({"role": "assistant", "content": "x"})] + [
    _ev({"content": "x"}) for _ in range(20)
]
_NONSTREAM_OK = {
    "id": "c1",
    "object": "chat.completion",
    "model": "qwen",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello world", "reasoning_content": "hm"},
            "finish_reason": "stop",
        }
    ],
    "usage": _USAGE,
}


def _sse_resp(chunks: list[bytes], *, status: int = 200, then: Any = None) -> MagicMock:
    async def aiter() -> Any:
        for c in chunks:
            yield c
        if then == "hang":
            await asyncio.sleep(5)
        elif then is not None:
            raise then

    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "text/event-stream"}
    resp.aiter_bytes = aiter
    resp.aclose = AsyncMock()
    return resp


def _json_resp(body: Any, *, status: int = 200, aread: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.aread = aread or AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def _send(result: Any) -> AsyncMock:
    if isinstance(result, list):  # one outcome per send (the keep-alive retry sends twice)
        return AsyncMock(side_effect=result)
    if isinstance(result, BaseException) or (
        isinstance(result, type) and issubclass(result, BaseException)
    ):
        return AsyncMock(side_effect=result)
    return AsyncMock(return_value=result)


# --------------------------------------------------------------------------
# Ways to drive one request
# --------------------------------------------------------------------------
def _headers(h: Any) -> dict[str, str]:
    """Response headers as the client sees them, minus the per-request id."""
    return {k.lower(): v for k, v in dict(h).items() if k.lower() != REQUEST_ID_HEADER.lower()}


def _http_stream(resp: Any, body: dict | None = None) -> Callable[[TestClient], Any]:
    def run(client: TestClient) -> Any:
        with patch("httpx.AsyncClient.send", new=_send(resp)):
            with client.stream(
                "POST", "/v1/chat/completions", headers=_AUTH, json=body or _STREAM
            ) as r:
                return r.status_code, _headers(r.headers), b"".join(r.iter_bytes())

    return run


def _http_post(resp: Any, body: dict | None = None) -> Callable[[TestClient], Any]:
    def run(client: TestClient) -> Any:
        with patch("httpx.AsyncClient.send", new=_send(resp)):
            r = client.post("/v1/chat/completions", headers=_AUTH, json=body or _NONSTREAM)
        return r.status_code, _headers(r.headers), r.content

    return run


def _token() -> TokenRow:
    return TokenRow(
        id="tok1",
        name="my-token",
        prefix=_PLAIN[:8],
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


async def _call_forward(client: TestClient, body: dict) -> Any:
    raw = json.dumps(body).encode()

    async def receive() -> dict:
        return {"type": "http.request", "body": raw, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "app": client.app,
        "query_string": b"",
        "client": ("127.0.0.1", 1),
    }
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    async with open_db(client.app.state.settings.db_path) as db:
        model = await ModelRepo(db).get_by_served_name("qwen")
    return await routes._forward(
        Request(scope, receive), model, "127.0.0.1", 19099, "/v1/chat/completions", _token()
    )


def _direct(
    resp: Any, body: dict, consume: Callable[[Any], Any] | None = None
) -> Callable[[TestClient], Any]:
    """``_forward`` called on the app's loop (exits a TestClient cannot reach: a raise or a
    cancellation inside the handler). ``consume`` drives the returned response."""

    def run(client: TestClient) -> Any:
        async def go() -> Any:
            with patch("httpx.AsyncClient.send", new=_send(resp)):
                try:
                    out = await _call_forward(client, body)
                    got = await consume(out) if consume is not None else bytes(out.body)
                    return out.status_code, _headers(out.headers), got
                except asyncio.CancelledError:
                    return "cancelled"
                except HTTPException as e:
                    return ("http", e.status_code, e.detail)
                except Exception as e:  # noqa: BLE001 -- the raise is the outcome
                    return ("raised", type(e).__name__, str(e))

        return client.portal.call(go)

    return run


async def _read_one_then_close(resp: Any) -> Any:
    it = resp.body_iterator
    first = await it.__anext__()
    await it.aclose()
    return first


async def _read_all(resp: Any) -> Any:
    it = resp.body_iterator
    got = []
    async for c in it:
        got.append(c)
    return b"".join(got)


async def _never_read(resp: Any) -> Any:
    await resp.background()
    return b""


def _suspending_tokenizer(client: TestClient) -> None:
    """A tokenizer whose count suspends, as the real one does (it runs off the loop)."""

    async def count(repo: str, text: str, *, fallback_repo: Any = None) -> int:
        await asyncio.sleep(0.02)
        return len(text.split()) if text else 0

    tok = MagicMock()
    tok.count = AsyncMock(side_effect=count)
    client.app.state.tokenizers = tok


def _asgi_disconnect(client: TestClient) -> Any:
    """A real client disconnect: the tab closes mid-stream and Starlette cancels the body
    task through its listen-for-disconnect scope (spec 2.3, as uvicorn runs it)."""
    from tests.asgi_stream import open_stream

    _suspending_tokenizer(client)
    hub = client.app.state.prompt_lens_hub
    reg = client.app.state.request_registry
    resp = _sse_resp(_OK_CHUNKS[:3], then="hang")

    async def go() -> Any:
        with patch("httpx.AsyncClient.send", new=_send(resp)):
            async with open_stream(
                client.app,
                "/v1/chat/completions",
                _AUTH,
                method="POST",
                body=json.dumps(_STREAM).encode(),
            ) as stream:
                await asyncio.sleep(0.2)
            # The settle task outlives the cancelled body task: wait for it to finish.
            real_hub = not isinstance(hub, MagicMock)
            for _ in range(500):
                ended = not real_hub or any(e["type"] == "request_end" for e in hub.events())
                if ended and not reg.snapshot():
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
        start = next(m for m in stream.messages if m["type"] == "http.response.start")
        heads = {k.decode(): v.decode() for k, v in start["headers"]}
        return stream.status, _headers(heads), stream.body

    return client.portal.call(go)


def _settle_raises(client: TestClient, stack: contextlib.ExitStack) -> None:
    stack.enter_context(
        patch.object(routes, "_record_counters", AsyncMock(side_effect=RuntimeError("ledger")))
    )


# --------------------------------------------------------------------------
# Patches that build a scenario's exit
# --------------------------------------------------------------------------
def _with(run: Callable[[TestClient], Any], setup: Callable[[TestClient], Any]) -> Any:
    def wrapped(client: TestClient) -> Any:
        with contextlib.ExitStack() as stack:
            setup(client, stack)
            return run(client)

    return wrapped


def _refuse(client: TestClient, stack: contextlib.ExitStack) -> None:
    class Refused:
        async def __aenter__(self) -> None:
            raise HTTPException(status_code=503, detail="queue full")

        async def __aexit__(self, *a: Any) -> bool:
            return False

    stack.enter_context(patch.object(client.app.state.scheduler, "acquire", lambda **kw: Refused()))


def _queued_cancel(client: TestClient) -> Any:
    class Blocked:
        async def __aenter__(self) -> None:
            await asyncio.Event().wait()

        async def __aexit__(self, *a: Any) -> bool:
            return False

    reg = client.app.state.request_registry

    async def go() -> Any:
        with patch.object(client.app.state.scheduler, "acquire", lambda **kw: Blocked()):
            task = asyncio.ensure_future(_call_forward(client, _STREAM))
            for _ in range(500):
                if any(r.phase == "queued" for r in reg.snapshot()):
                    break
                await asyncio.sleep(0.01)
            assert any(r.phase == "queued" for r in reg.snapshot())
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return "cancelled"
        return "not cancelled"

    return client.portal.call(go)


def _register_raises(*effects: Any) -> Callable[[TestClient, contextlib.ExitStack], None]:
    def setup(client: TestClient, stack: contextlib.ExitStack) -> None:
        reg = client.app.state.request_registry
        stack.enter_context(patch.object(reg, "register", AsyncMock(side_effect=list(effects))))

    return setup


def _cobs2_cancelled(client: TestClient, stack: contextlib.ExitStack) -> None:
    stack.enter_context(
        patch.object(routes, "_cobs2_compute", MagicMock(side_effect=asyncio.CancelledError()))
    )


def _disconnected(client: TestClient, stack: contextlib.ExitStack) -> None:
    stack.enter_context(
        patch("starlette.requests.Request.is_disconnected", new=AsyncMock(return_value=True))
    )


def _guard_blocks(client: TestClient, stack: contextlib.ExitStack) -> None:
    from app.guard import proxy as guard_proxy

    stack.enter_context(patch.object(routes.guard_proxy, "guard_for", lambda *a: object()))
    stack.enter_context(
        patch.object(
            routes,
            "_guard_inspect",
            AsyncMock(return_value=guard_proxy.Outcome(action="block", body={})),
        )
    )
    stack.enter_context(patch.object(routes, "_record_blocked", AsyncMock()))


async def _slow_read() -> bytes:
    await asyncio.sleep(5)
    return b"{}"


@dataclasses.dataclass
class Scenario:
    name: str
    run: Callable[[TestClient], Any]
    finish: str | None  # the request_end finish_reason; None with ends=0
    ends: int = 1
    settings: dict[str, Any] | None = None
    #: Outcome normaliser (a guard block reply carries a fresh id and time).
    norm: Callable[[Any], Any] | None = None

    def normal(self, outcome: Any) -> Any:
        return self.norm(outcome) if self.norm is not None else outcome


_ENFORCE = {
    "runaway_mode": "enforce",
    "runaway_hard_max": 3,
    "runaway_think_budget": 10**9,
    "runaway_repeat_max": 10**9,
}
_LOG = {**_ENFORCE, "runaway_mode": "log", "runaway_hard_max": 10**9}

SCENARIOS = [
    # Streaming client.
    Scenario("stream_ok", _http_stream(_sse_resp(_OK_CHUNKS)), "stop"),
    Scenario(
        "stream_upstream_500",
        _http_stream(_sse_resp([b'{"error": {"message": "boom"}}'], status=500)),
        "error",
    ),
    Scenario(
        "stream_error_frame",
        _http_stream(_sse_resp([b'data: {"error": {"message": "ctx exceeded"}}\n\n'])),
        "error",
    ),
    Scenario(
        "stream_client_gone", _with(_http_stream(_sse_resp(_OK_CHUNKS)), _disconnected), "abort"
    ),
    Scenario(
        "stream_wall_reaper",
        _http_stream(_sse_resp(_OK_CHUNKS)),
        "error",
        settings={"request_max_wall_s": 1e-9},
    ),
    Scenario(
        "stream_idle_limit",
        _http_stream(_sse_resp(_OK_CHUNKS[:2], then="hang")),
        "error",
        settings={"upstream_idle_s": 0.05},
    ),
    Scenario(
        "stream_runaway_trip",
        _http_stream(_sse_resp(_RUNAWAY_CHUNKS)),
        "runaway",
        settings=_ENFORCE,
    ),
    Scenario("stream_runaway_log_mode", _http_stream(_sse_resp(_OK_CHUNKS)), "stop", settings=_LOG),
    Scenario(
        "stream_cancelled_mid_stream",
        _direct(_sse_resp(_OK_CHUNKS[:2], then=asyncio.CancelledError()), _STREAM, _read_all),
        "abort",
    ),
    Scenario(
        "stream_upstream_raises_mid_stream",
        _direct(_sse_resp(_OK_CHUNKS[:2], then=RuntimeError("read failed")), _STREAM, _read_all),
        "error",
    ),
    Scenario(
        "stream_client_closes_body",
        _direct(_sse_resp(_OK_CHUNKS), _STREAM, _read_one_then_close),
        "abort",
    ),
    Scenario("stream_never_read", _direct(_sse_resp(_OK_CHUNKS), _STREAM, _never_read), "abort"),
    Scenario(
        "stream_never_read_upstream_500",
        _direct(_sse_resp([b'{"error": {"message": "boom"}}'], status=500), _STREAM, _never_read),
        "error",
    ),
    Scenario("stream_asgi_disconnect", _asgi_disconnect, "abort"),
    Scenario(
        "stream_settle_raises",
        _with(_direct(_sse_resp(_OK_CHUNKS), _STREAM, _read_all), _settle_raises),
        "error",
    ),
    # Non-streaming client.
    Scenario("nonstream_ok", _http_post(_json_resp(_NONSTREAM_OK)), "stop"),
    Scenario(
        "nonstream_upstream_500",
        _http_post(_json_resp({"error": {"message": "boom"}}, status=500)),
        "error",
    ),
    Scenario(
        "nonstream_wall_timeout",
        _http_post(_json_resp(None, aread=AsyncMock(side_effect=_slow_read))),
        "error",
        settings={"request_max_wall_s": 0.05},
    ),
    Scenario(
        "nonstream_cancelled_reading",
        _direct(
            _json_resp(None, aread=AsyncMock(side_effect=asyncio.CancelledError())), _NONSTREAM
        ),
        "abort",
    ),
    Scenario("runaway_nonstream_ok", _http_post(_sse_resp(_OK_CHUNKS)), "stop", settings=_LOG),
    Scenario(
        "runaway_nonstream_trip",
        _http_post(_sse_resp(_RUNAWAY_CHUNKS)),
        "runaway",
        settings=_ENFORCE,
    ),
    Scenario(
        "runaway_nonstream_upstream_raises",
        _direct(_sse_resp(_OK_CHUNKS[:2], then=RuntimeError("read failed")), _NONSTREAM),
        "error",
        settings=_LOG,
    ),
    Scenario(
        "runaway_nonstream_cancelled",
        _direct(_sse_resp(_OK_CHUNKS[:2], then=asyncio.CancelledError()), _NONSTREAM),
        "abort",
        settings=_LOG,
    ),
    # The engine never answers.
    Scenario("send_connect_error", _http_post(httpx.ConnectError("refused")), "error"),
    Scenario("send_stalled", _http_post(TimeoutError()), "error"),
    Scenario("send_cancelled", _direct(asyncio.CancelledError(), _STREAM), "abort"),
    Scenario("send_raises", _direct(RuntimeError("send failed"), _STREAM), "error"),
    Scenario(
        "send_stale_keepalive_retry_fails",
        _direct([httpx.RemoteProtocolError("stale"), httpx.ReadError("again")], _STREAM),
        "error",
    ),
    Scenario(
        "send_stale_keepalive_retry_ok",
        _http_stream([httpx.RemoteProtocolError("stale"), _sse_resp(_OK_CHUNKS)]),
        "stop",
    ),
    # Before the engine: admission and registration.
    Scenario("queue_refused", _with(_http_post(_json_resp(_NONSTREAM_OK)), _refuse), "error"),
    Scenario("cancelled_while_queued", _queued_cancel, "abort"),
    Scenario(
        "cancelled_registering",
        _with(_direct(_sse_resp(_OK_CHUNKS), _STREAM), _register_raises(asyncio.CancelledError())),
        "abort",
    ),
    Scenario(
        "cancelled_registering_after_admission",
        _with(
            _direct(_sse_resp(_OK_CHUNKS), _STREAM),
            _register_raises(RuntimeError("registry down"), asyncio.CancelledError()),
        ),
        "abort",
    ),
    Scenario(
        "cancelled_hashing_ids",
        _with(_direct(_sse_resp(_OK_CHUNKS), _STREAM), _cobs2_cancelled),
        "abort",
    ),
    # The guard blocks before request_start: neither event, on or off.
    Scenario(
        "guard_block",
        _with(_http_post(_json_resp(_NONSTREAM_OK)), _guard_blocks),
        None,
        ends=0,
        norm=lambda o: o[:2],  # the block reply carries a fresh id and time
    ),
]


def _diff(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> Any:
    if len(a) != len(b):
        return len(a), len(b)
    return [
        {k: (x.get(k), y.get(k)) for k in x.keys() | y.keys() if x.get(k) != y.get(k)}
        for x, y in zip(a, b, strict=True)
    ]


def _of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("type") == kind]


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_exit_publishes_exactly_one_request_end(fresh_app: Any, sc: Scenario) -> None:
    on = _run(fresh_app, sc, gm=True)
    assert on.events is not None
    starts, ends = _of(on.events, "request_start"), _of(on.events, "request_end")
    assert len(starts) == sc.ends and len(ends) == sc.ends, (on.outcome, ends)
    if sc.ends:
        end = ends[0]
        assert end["finish_reason"] == sc.finish, end
        assert end["req_id"] == starts[0]["req_id"] and end["token_id"] == "tok1"
        assert _END_KEYS <= end.keys()
        assert end["seq"] > starts[0]["seq"]
        assert isinstance(end["duration_s"], float) and end["duration_s"] >= 0


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_exit_is_unchanged_by_prompt_lens(fresh_app: Any, sc: Scenario) -> None:
    off = _run(fresh_app, sc, gm=False)
    on = _run(fresh_app, sc, gm=True)
    # Off: zero Prompt Lens work reaches the hub, on every exit.
    assert off.spy is not None and off.spy.method_calls == []
    # Client-visible behaviour, history row, stats ledger and the live registry: identical.
    assert sc.normal(on.outcome) == sc.normal(off.outcome)
    assert on.rows == off.rows, _diff(on.rows, off.rows)
    assert on.ledger == off.ledger
    assert on.live_left == off.live_left == 0


# --------------------------------------------------------------------------
# What the end carries
# --------------------------------------------------------------------------
def _one_end(fresh_app: Any, sc: Scenario) -> dict[str, Any]:
    on = _run(fresh_app, sc, gm=True)
    assert on.events is not None
    ends = _of(on.events, "request_end")
    assert len(ends) == 1
    return ends[0]


def test_end_carries_timings_and_cache(fresh_app: Any) -> None:
    end = _one_end(fresh_app, SCENARIOS[0])
    assert {
        "cached_tokens",
        "ttft_s",
        "queued_s",
        "duration_s",
        "decode_tps",
        "own_reused_tokens",
        "cache",
    } <= end.keys()
    assert (end["prompt_tokens"], end["completion_tokens"]) == (2, 3)
    assert (end["cached_tokens"], end["reasoning_tokens"]) == (1, 1)
    assert isinstance(end["cache"], dict)
    assert end["own_reused_tokens"] == min(
        end["cached_tokens"], end["cache"]["reusable_tokens_own"]
    )
    assert end["queued_s"] >= 0 and end["ttft_s"] >= 0
    assert end["duration_s"] >= end["ttft_s"]


def test_nonstream_end_carries_usage(fresh_app: Any) -> None:
    end = _one_end(fresh_app, next(s for s in SCENARIOS if s.name == "nonstream_ok"))
    assert (end["prompt_tokens"], end["completion_tokens"]) == (2, 3)
    assert (end["cached_tokens"], end["reasoning_tokens"]) == (1, 1)
    assert end["ttft_s"] is None  # a non-stream reply has no first token to time
    assert isinstance(end["cache"], dict)


def test_real_disconnect_end_carries_the_settled_figures(fresh_app: Any) -> None:
    """I1: on a real ASGI disconnect the end waits for _settle (cache block, final count)."""
    sc = next(s for s in SCENARIOS if s.name == "stream_asgi_disconnect")
    on = _run(fresh_app, sc, gm=True)
    assert on.events is not None
    ends = _of(on.events, "request_end")
    assert [e["finish_reason"] for e in ends] == ["abort"]
    assert len(on.rows) == 1
    assert ends[0]["cache"] is not None
    assert ends[0]["completion_tokens"] == on.rows[0]["completion_tokens"]
    assert ends[0]["completion_tokens"] > 0


def test_runaway_trip_publishes_runaway_end(fresh_app: Any) -> None:
    sc = next(s for s in SCENARIOS if s.name == "stream_runaway_trip")
    on = _run(fresh_app, sc, gm=True)
    status, _heads, body = on.outcome
    assert status == 200 and b"runaway_hard" in body  # the client still sees the clean end
    assert on.events is not None
    ends = _of(on.events, "request_end")
    assert [e["finish_reason"] for e in ends] == ["runaway"]
    assert ends[0]["completion_tokens"] > 0


def test_client_disconnect_publishes_one_abort_end(fresh_app: Any) -> None:
    end = _one_end(fresh_app, next(s for s in SCENARIOS if s.name == "stream_cancelled_mid_stream"))
    assert end["finish_reason"] == "abort"


def test_upstream_500_publishes_one_error_end(fresh_app: Any) -> None:
    for name in ("stream_upstream_500", "nonstream_upstream_500"):
        end = _one_end(fresh_app, next(s for s in SCENARIOS if s.name == name))
        assert end["finish_reason"] == "error", name


def test_queue_refusal_ends_queued_with_the_wait(fresh_app: Any) -> None:
    end = _one_end(fresh_app, next(s for s in SCENARIOS if s.name == "queue_refused"))
    assert end["finish_reason"] == "error"
    # Never admitted: the whole life was the wait; no engine figures.
    assert end["queued_s"] == end["duration_s"]
    assert (end["ttft_s"], end["cache"], end["completion_tokens"]) == (None, None, 0)
    assert end["prompt_tokens"] == 2  # the warden's own count ("hi there")


def test_end_keeps_tool_call_drop_counters(fresh_app: Any) -> None:
    calls = [
        _ev(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "x" * 300},
                    }
                ]
            }
        ),
        _ev({}, finish="tool_calls"),
        b"data: [DONE]\n\n",
    ]
    sc = Scenario(
        "tc",
        _http_stream(_sse_resp(calls)),
        "tool_calls",
        settings={"prompt_lens_section_prompt_chars": 50},
    )
    end = _one_end(fresh_app, sc)
    assert end["finish_reason"] == "tool_calls"
    assert end["tool_call_chars_dropped"] > 0
    # Only counted drops are sent.
    ok = _one_end(fresh_app, SCENARIOS[0])
    assert "tool_call_chars_dropped" not in ok and "tool_calls_dropped" not in ok


# --------------------------------------------------------------------------
# _gm_end on its own
# --------------------------------------------------------------------------
def _live(**kw: Any) -> LiveRequest:
    base: dict[str, Any] = {
        "id": "r",
        "token_id": "tok1",
        "token_name": "t",
        "client_ip": "127.0.0.1",
        "model": "qwen",
        "model_row_id": "qwen",
        "variant_id": "v",
        "path": "/v1/chat/completions",
        "prompt_tokens": 100,
        "max_model_len": 4096,
        "started_monotonic": 0.0,
        "started_iso": "2026-11-22T00:00:00Z",
        "phase": "decode",
    }
    base.update(kw)
    return LiveRequest(**base)


def _published(**kw: Any) -> dict[str, Any]:
    hub = MagicMock()
    routes._gm_end(hub, req_id="g", token_id="tok1", **kw)
    assert hub.publish.call_count == 1
    return hub.publish.call_args.args[0]


def test_gm_end_timings_and_decode_rate_match_the_stats_page() -> None:
    from app.stats import speed

    lr = _live(queued_s=0.5, first_token_monotonic=2.0)
    lr.cache_obs = {"reusable_tokens_own": 40, "cache_outcome": "hit"}
    with patch.object(routes, "_monotonic", return_value=12.0):
        ev = _published(
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=201,
            live_req=lr,
            cached_tokens=64,
            reasoning_tokens=7,
        )
    assert (ev["queued_s"], ev["ttft_s"], ev["duration_s"]) == (0.5, 2.0, 12.0)
    want = speed.row_metrics(
        {
            "queued_s": 0.5,
            "ttft_s": 2.0,
            "duration_s": 12.0,
            "prompt_tokens": 100,
            "completion_tokens": 201,
        }
    )["decode"]
    assert ev["decode_tps"] == want == 20.0
    assert (ev["cached_tokens"], ev["reasoning_tokens"], ev["own_reused_tokens"]) == (64, 7, 40)
    assert ev["cache"] == lr.cache_obs and ev["cache"] is not lr.cache_obs


def test_gm_end_unknowns_are_none() -> None:
    ev = _published(finish_reason="abort", prompt_tokens=5, completion_tokens=0, live_req=None)
    for k in (
        "reasoning_tokens",
        "cached_tokens",
        "queued_s",
        "ttft_s",
        "decode_tps",
        "own_reused_tokens",
        "cache",
    ):
        assert ev[k] is None, k
    assert set(ev) == _END_KEYS


def test_gm_end_own_reuse_needs_both_figures() -> None:
    lr = _live()
    lr.cache_obs = {"reusable_tokens_own": None}
    ev = _published(
        finish_reason="stop", prompt_tokens=5, completion_tokens=1, live_req=lr, cached_tokens=3
    )
    assert ev["own_reused_tokens"] is None
    lr.cache_obs = {"reusable_tokens_own": 9}
    ev = _published(
        finish_reason="stop", prompt_tokens=5, completion_tokens=1, live_req=lr, cached_tokens=None
    )
    assert ev["own_reused_tokens"] is None


def test_gm_end_never_raises_into_forward() -> None:
    hub = MagicMock()
    hub.publish.side_effect = RuntimeError("hub broke")
    routes._gm_end(
        hub,
        req_id="g",
        token_id="t",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        live_req=_live(),
    )
    lr = _live()
    lr.cache_obs = {"reusable_tokens_own": "garbage"}
    routes._gm_end(
        MagicMock(),
        req_id="g",
        token_id="t",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        live_req=lr,
        cached_tokens=3,
    )
