import asyncio
import dataclasses
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest

from app.db.repos.tokens import hash_token


def _seed(db_path, *, model_status="loaded"):
    pw = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)", ("admin", pw))
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0, 1, 2, 3]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)",
            (json.dumps([0]), model_status),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def test_proxy_404_when_model_not_loaded(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db", model_status="pulled")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "qwen", "messages": []},
    )
    assert r.status_code == 404


def test_proxy_404_when_model_unknown(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"model": "ghost", "messages": []},
    )
    assert r.status_code == 404


def test_proxy_400_when_model_field_missing(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {plaintext}"},
        json={"messages": []},
    )
    assert r.status_code == 400


def test_proxy_forwards_to_subprocess_port(tmp_data_dir, client):
    """When model is loaded, requests route to its 127.0.0.1:<port>."""
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    # Register the supervisor's port for 'qwen' so _resolve_target finds it.
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(
        return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}'
    )
    fake_resp.aclose = AsyncMock()

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    sent_url = str(send.call_args.args[0].url)
    assert "127.0.0.1:19099" in sent_url
    assert "/v1/chat/completions" in sent_url


def test_proxy_closes_upstream_when_nonstream_read_fails(tmp_data_dir, client):
    """#184 — if the client disconnects while we await the non-streamed body,
    ``resp.aread()`` raises. The proxy must still close the upstream response
    and the httpx client (in ``finally``) so vLLM sees the socket close and
    aborts the generation, freeing KV blocks immediately instead of at GC time.
    Regression guard: the two ``aclose()`` calls used to sit after ``aread()``
    inside the ``try`` and were skipped on that error path."""
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    # Simulate the client-disconnect: the body read blows up mid-flight.
    fake_resp.aread = AsyncMock(side_effect=RuntimeError("client disconnected"))
    fake_resp.aclose = AsyncMock()

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)),
        patch("httpx.AsyncClient.aclose", new=AsyncMock()) as client_aclose,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        with pytest.raises(RuntimeError):
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
    # The upstream response is closed despite the read blowing up; the pooled
    # client is shared across requests (#279) and must stay open.
    fake_resp.aclose.assert_awaited_once()
    client_aclose.assert_not_awaited()


class _FakeStreamResp:
    """Upstream SSE response that keeps emitting content deltas.

    ``aiter_bytes`` yields ``n_chunks`` frames with a small inter-chunk
    sleep. Used to model a runaway/abandoned generation that never reaches
    ``[DONE]`` within the request wall-clock budget."""

    def __init__(self, n_chunks: int, delay: float):
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self._n = n_chunks
        self._delay = delay
        self.aclose = AsyncMock()

    async def aiter_bytes(self):
        for _ in range(self._n):
            await asyncio.sleep(self._delay)
            payload = {"choices": [{"delta": {"content": "x"}}], "model": "qwen"}
            yield ("data: " + json.dumps(payload) + "\n\n").encode()


def test_proxy_streaming_wallclock_reaper_tears_down_runaway(tmp_data_dir, client):
    """Reaper — a streamed generation that outlives ``request_max_wall_s`` is
    cut off: the proxy stops draining upstream, closes the upstream response
    and httpx client (so vLLM aborts + frees KV), and releases the scheduler
    slot. Without the backstop the body iterator drains the full upstream
    stream (all ``n_chunks``) because the abandoned-but-transport-alive client
    never produces an ``http.disconnect`` for Starlette to cancel on."""
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    # Enable the wall-clock backstop for this request (frozen dataclass -> replace).
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, request_max_wall_s=0.3
    )
    # Avoid loading a real tokenizer on the hot path / at teardown.
    client.app.state.tokenizers = SimpleNamespace(count=AsyncMock(return_value=1))

    fake_resp = _FakeStreamResp(n_chunks=100, delay=0.02)  # would run ~2s unclipped

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)),
        patch("httpx.AsyncClient.aclose", new=AsyncMock()) as client_aclose,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        events = r.text.count("data:")

    assert r.status_code == 200
    # Cut off well before the upstream's 100 chunks (cap 0.3s / 0.02s ≈ 15).
    assert events < 60, f"reaper did not clip the stream: {events} events drained"
    # Upstream response torn down so vLLM aborts and frees KV; the pooled
    # client (#279) is shared and stays open.
    fake_resp.aclose.assert_awaited()
    client_aclose.assert_not_awaited()


class _HangingResp:
    """Upstream non-stream response whose ``aread`` never returns within the
    request wall-clock budget — models a hung/transport-alive upstream."""

    def __init__(self, delay: float):
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self._delay = delay
        self.aclose = AsyncMock()

    async def aread(self):
        await asyncio.sleep(self._delay)
        return b'{"choices": [], "usage": {"completion_tokens": 0}}'


def test_proxy_nonstream_wallclock_reaper_returns_504(tmp_data_dir, client):
    """Reaper (non-stream) — a ``resp.aread()`` that outlives
    ``request_max_wall_s`` is bounded by ``asyncio.wait_for``: the proxy tears
    down the upstream response + httpx client (vLLM aborts + frees KV), releases
    the slot, and returns a clean 504 instead of pinning the slot forever.

    This models real httpx semantics ONLY if the upstream is sent with
    ``stream=True``: with ``stream=False`` httpx reads the whole body *inside*
    ``send()`` (which runs under ``timeout=None``), and ``resp.aread()`` then
    returns the buffered body instantly — so ``asyncio.wait_for`` would bound
    nothing and the reaper would be dead. The explicit ``stream=True`` assertion
    below is the regression guard for that (a runaway non-stream generation
    pinned the slot for 94 min in prod when this had regressed to
    ``stream=is_stream``)."""
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings, request_max_wall_s=0.3
    )
    client.app.state.tokenizers = SimpleNamespace(count=AsyncMock(return_value=1))

    fake_resp = _HangingResp(delay=5.0)  # would hang 5s without the backstop

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send_mock,
        patch("httpx.AsyncClient.aclose", new=AsyncMock()) as client_aclose,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert r.status_code == 504
    fake_resp.aclose.assert_awaited()
    client_aclose.assert_not_awaited()  # pooled client (#279) stays open
    # The reaper is only real if the body read is deferred to aread(); that
    # requires the upstream to be sent with stream=True even for a non-stream
    # client. Lock it in so a revert to stream=is_stream fails here, not in prod.
    assert send_mock.await_args.kwargs.get("stream") is True


def _set_priority(db_path, priority):
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET priority=? WHERE id='tok1'", (priority,))
        db.commit()


def test_proxy_injects_vllm_priority_into_forwarded_body(tmp_data_dir, client):
    """#173 part B — a non-zero warden priority must be pushed into the engine
    as ``priority = -warden_priority`` (vLLM orders ASCENDING, so negative puts
    high-priority traffic ahead of default-0 traffic)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 7)
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(
        return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}'
    )
    fake_resp.aclose = AsyncMock()

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert forwarded["priority"] == -7


def test_proxy_does_not_inject_priority_when_zero(tmp_data_dir, client):
    """Warden priority 0 maps to vLLM's default (inert). The forwarded body must
    stay free of a ``priority`` field so an unprioritised request is byte-clean."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 0)
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(
        return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}'
    )
    fake_resp.aclose = AsyncMock()

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert "priority" not in forwarded


def test_proxy_does_not_override_client_supplied_priority(tmp_data_dir, client):
    """If the client already set ``priority`` in its request body, the proxy
    must leave it untouched — the explicit client value wins over the token map."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 7)
    client.app.state.supervisor._ports["qwen"] = 19099

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(
        return_value=b'{"id":"x","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}'
    )
    fake_resp.aclose = AsyncMock()

    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}], "priority": 3},
        )
    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert forwarded["priority"] == 3


# ---------------------------------------------------------------------------
# Sub-project C: the engine-side priority hint is capability-gated
# ---------------------------------------------------------------------------


def _set_backend(db_path, backend):
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE models SET backend=? WHERE id='qwen'", (backend,))
        db.commit()


def _forward_and_capture(client, plaintext):
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(
        return_value=(
            b'{"id":"x","model":"qwen","choices":[],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":0}}'
        )
    )
    fake_resp.aclose = AsyncMock()
    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)) as send,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    return json.loads(send.call_args.args[0].content)


def test_priority_is_not_injected_for_a_backend_without_a_scheduler(tmp_data_dir, client):
    """llama-server has no priority scheduler. Its request parser ignores
    unknown fields rather than rejecting them, so injecting `priority` would be
    harmless -- but it would also mean re-serialising the body for no effect and
    advertising a capability the engine does not have. The warden's OWN
    admission scheduler still orders by priority; only the engine-side hint is
    dropped."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 5)
    _set_backend(db_path, "llamacpp")
    client.app.state.supervisor._ports["qwen"] = 19099

    assert "priority" not in _forward_and_capture(client, plaintext)


def test_priority_is_still_injected_for_vllm(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 5)
    _set_backend(db_path, "vllm")
    client.app.state.supervisor._ports["qwen"] = 19099

    assert _forward_and_capture(client, plaintext)["priority"] == -5


def test_a_null_backend_still_gets_the_vllm_hint(tmp_data_dir, client):
    """D6: NULL decodes to vLLM, so every pre-C row keeps the #173 behaviour."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed(db_path)
    _set_priority(db_path, 5)
    _set_backend(db_path, None)
    client.app.state.supervisor._ports["qwen"] = 19099

    assert _forward_and_capture(client, plaintext)["priority"] == -5
