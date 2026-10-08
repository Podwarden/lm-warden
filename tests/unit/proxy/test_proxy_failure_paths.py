"""R33 + R34 — the proxy's failure paths, against a real engine hop.

The sibling proxy tests patch ``httpx.AsyncClient.send`` with a MagicMock
response; that never exercises the real warden→engine hop, which is exactly
where the failure paths live (send, pass-through, accounting, slot release).
These tests instead boot the in-process ``FakeEngine`` on a real loopback port
and drive the real app over ASGI via the ``live_app`` fixture, so the proxy's
``httpx`` client performs a genuine TCP round trip to the fake.

R33 (sub-case 1) — an upstream 5xx in the default (non-runaway) proxy mode,
with ``mode="error"`` answering every request with a bare ``{"error": {...}}``
envelope, pinned for both a streaming and a non-streaming client:

* the engine's status code and error envelope pass through to the client;
* the non-stream branch additionally attaches the model's ``last_error`` as
  an ``error.hint`` (``enrich_5xx_from_db``); the stream branch passes the
  body through byte-for-byte (enrichment exists only in the non-stream
  branch — a streaming client's SDK inspects the status code before parsing
  SSE, so the envelope, not the hint, is what it acts on);
* the failed request is still counted: one counters row with the prompt
  tokens and zero completion tokens (a 5xx bills the prompt it consumed and
  nothing else).

R34 (sub-cases 2–6) — the rest of the CI-audit-A27 matrix:

* a REFUSED engine connection (the port is dead) is a 502 "upstream engine
  unreachable", never a 500 — SDK retry loops act on 502, and a 500 claims
  the warden itself broke — with no counters row and the admission slot
  released, proven by a second request being admitted;
* a client that CLOSES THE TAB MID-STREAM: Starlette's StreamingResponse
  cancels the body task on http.disconnect and the proxy's finally must still
  deregister, account the PARTIAL completion, and release the slot (the
  engine-side KV blocks are freed only if the upstream socket is closed
  here);
* the WALL-CLOCK REAPER (``request_max_wall_s``) cuts a stream that would run
  for minutes: the app returns by itself, the record is marked orphan, the
  partial completion is billed, and the slot is released;
* a non-stream 200 WITHOUT a usage block is still billed for its completion
  via the tokenizer fallback (the two streaming paths already had it);
* a non-stream 200 whose `usage` key is present but `null` (R36) is billed
  the same way — the unguarded `get("usage", {}).get(...)` raised
  AttributeError on the `None` and 500'd the served completion;
* TWO LOADED MODELS route to their own ports — the fake's unknown-model trap
  (200 + empty content on a mismatched name) makes cross-routing visible.

The real-subprocess streaming case is R35.
"""

import asyncio
import dataclasses
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.db.repos.tokens import hash_token
from app.proxy.scheduler import PriorityScheduler
from tests.asgi_stream import open_stream
from tests.conftest import wait_until_async
from tests.fakes.fake_vllm import FakeEngine, Knobs


def _seed_loaded(db_path: Path) -> str:
    """Setup done, one 'loaded' model, one inference token.

    The admin user itself is already seeded by the ``live_app`` fixture
    (``seed_admin_user``), so it must NOT be inserted here.
    """
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('qwen','qwen','Qwen/Qwen3.5-9B','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        plaintext = "vw_validtoken1234567890abcdef12345"
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.commit()
        return plaintext


def _set_last_error(db_path: Path, last_error: str) -> None:
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("UPDATE models SET last_error=? WHERE id='qwen'", (last_error,))
        db.execute("PRAGMA wal_checkpoint(FULL)")


def _read_counters(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "SELECT model_id, token_id, requests, prompt_tokens, completion_tokens FROM counters"
        )
        return [dict(zip([d[0] for d in cur.description], r, strict=False)) for r in cur.fetchall()]


def _make_fake_tokenizer(token_count_for):
    """A TokenizerCache double whose .count(repo, text) uses the dict lookup."""
    cache = MagicMock()
    cache.count = AsyncMock(
        side_effect=lambda repo, text, *, fallback_repo=None: token_count_for(text)
    )
    return cache


def _seed_llama(db_path: Path) -> None:
    """A second loaded model, so two-model routing can be asserted."""
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES ('llama','llama','Qwen/Qwen3.5-4B','main',?,1,'auto',4096,0.9,0,'[]','loaded',0,NULL,NULL)",
            (json.dumps([0]),),
        )
        db.commit()


def _read_history(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "SELECT model_id, prompt_tokens, completion_tokens, orphan, finish_reason "
            "FROM request_history ORDER BY finished_at"
        )
        return [dict(zip([d[0] for d in cur.description], r, strict=False)) for r in cur.fetchall()]


def _token_count_for(text: str) -> int:
    """The fake tokenizer: 'hi' is 5 tokens, everything else len//4 — the same
    ratio the fake engine reports, so both sides of the bill agree."""
    return 5 if text == "hi" else max(0, len(text) // 4)


@pytest.fixture
async def error_503_engine():
    """The fake engine answering every request with a 503 error envelope."""
    engine = FakeEngine(served_model_name="qwen", knobs=Knobs(mode="error", error_status=503))
    await engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


async def test_upstream_5xx_non_stream_passes_status_and_body_and_counts(
    live_app, tmp_data_dir: Path, error_503_engine
):
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    _set_last_error(db_path, "vLLM warmup probe failed")
    live_app.state.supervisor._ports["qwen"] = error_503_engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if "hi" in text else 0)

    transport = httpx.ASGITransport(app=live_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert r.status_code == 503
    body = r.json()
    # The engine's envelope passes through...
    assert body["error"]["code"] == 503
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"] == "internal failure"
    # ...and the 5xx enrichment attaches the model's last_error as a hint.
    assert body["error"]["hint"] == {"model_id": "qwen", "last_error": "vLLM warmup probe failed"}

    # The failed request is still counted: the prompt it consumed, nothing else.
    await live_app.state.ledger.flush()
    rows = _read_counters(db_path)
    assert len(rows) == 1
    assert rows[0]["model_id"] == "qwen"
    assert rows[0]["token_id"] == "tok1"
    assert rows[0]["requests"] == 1
    assert rows[0]["prompt_tokens"] == 5
    assert rows[0]["completion_tokens"] == 0


async def test_upstream_5xx_stream_passes_status_and_body_and_counts(
    live_app, tmp_data_dir: Path, error_503_engine
):
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    _set_last_error(db_path, "vLLM warmup probe failed")
    live_app.state.supervisor._ports["qwen"] = error_503_engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if "hi" in text else 0)

    transport = httpx.ASGITransport(app=live_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            assert r.status_code == 503
            chunks = [c async for c in r.aiter_bytes()]

    # The stream branch passes the engine's JSON envelope through untouched
    # (no enrichment on this path) — byte-for-byte what the engine sent.
    body = json.loads(b"".join(chunks))
    assert body["error"]["code"] == 503
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"] == "internal failure"
    assert "hint" not in body["error"]

    await live_app.state.ledger.flush()
    rows = _read_counters(db_path)
    assert len(rows) == 1
    assert rows[0]["model_id"] == "qwen"
    assert rows[0]["token_id"] == "tok1"
    assert rows[0]["requests"] == 1
    assert rows[0]["prompt_tokens"] == 5
    assert rows[0]["completion_tokens"] == 0


async def test_refused_connection_is_502_and_releases_slot(live_app, tmp_data_dir: Path):
    """R34 — the engine's port refuses the connection (crashed, not yet
    restarted). That is an upstream availability fault, not a warden bug: the
    client must get a 502, not a 500, nothing is billed, and the admission
    slot is released so the next request for the same engine is admitted."""
    dead = FakeEngine(served_model_name="qwen")
    await dead.start()
    dead_port = dead.port
    await dead.crash()  # listener gone; the port now refuses connections

    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = dead_port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    live_app.state.scheduler = PriorityScheduler(max_inflight=1)

    transport = httpx.ASGITransport(app=live_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r1 = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    # 502 is what every OpenAI-compatible gateway returns for "upstream
    # unreachable" and the status SDK retry loops act on; a 500 claims the
    # warden broke and tells clients to stop.
    assert r1.status_code == 502
    assert "detail" in r1.json()

    # Nothing was served, so nothing is billed, and nothing is in flight.
    await live_app.state.ledger.flush()
    assert _read_counters(db_path) == []
    assert live_app.state.request_registry.count() == 0
    assert live_app.state.scheduler._inflight_for_test("qwen") == 0
    assert live_app.state.scheduler._queue_size_for_test("qwen") == 0

    # The slot is actually free: a second request is admitted — and would hang
    # forever here (caught by the timeout) if the failed one kept its slot at
    # max_inflight=1.
    live = FakeEngine(served_model_name="qwen")
    await live.start()
    try:
        live_app.state.supervisor._ports["qwen"] = live.port
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r2 = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {plaintext}"},
                    json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                ),
                timeout=5.0,
            )
        assert r2.status_code == 200
    finally:
        await live.stop()


async def test_client_disconnect_mid_stream_tears_down_and_counts(live_app, tmp_data_dir: Path):
    """R34 — the client closes the tab mid-stream. Starlette's
    StreamingResponse cancels the body task on http.disconnect; the proxy's
    finally must still deregister, account the PARTIAL completion, and release
    the slot — the engine-side KV blocks are freed only if the upstream socket
    is closed here."""
    slow = FakeEngine(served_model_name="qwen", knobs=Knobs(chunk_delay_s=0.2))
    await slow.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = slow.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    live_app.state.scheduler = PriorityScheduler(max_inflight=1)
    try:
        # 50-char prompt -> "echo: ..." is 56 chars = 7 frames of 8 at 0.2s
        # each: the stream outlives the client, which leaves after ~1s of ~5
        # frames (40 chars = 10 fake tokens, full answer = 14).
        prompt = "hello world, this is a test of the disconnect path"
        body = json.dumps(
            {
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        async with open_stream(
            live_app,
            "/v1/chat/completions",
            {"Authorization": f"Bearer {plaintext}"},
            method="POST",
            body=body,
        ) as stream:
            assert stream.status == 200
            assert stream.body.startswith(b"data: ")
            await asyncio.sleep(1.0)

        # The store writes history rows from a background flusher (the full
        # lifespan is live under live_app), and the flusher pulls a record out
        # of the queue the moment it is enqueued — an inline flush() races it
        # and sees an empty queue. Poll for the row instead.
        rows = await wait_until_async(
            lambda: _read_history(db_path) or None,
            what="the request_history row for the disconnected stream",
            timeout_s=5.0,
        )
        assert rows[0]["model_id"] == "qwen"
        assert rows[0]["prompt_tokens"] == len(prompt) // 4
        # Some tokens arrived before the tab closed; the answer did not finish.
        assert 4 <= rows[0]["completion_tokens"] < len("echo: " + prompt) // 4
        # The client left on purpose and the warden followed — this is not an
        # orphan (the reaper's mark is for streams cut on the wall).
        assert rows[0]["orphan"] == 0

        # The PARTIAL completion must be billed too.
        counters = await wait_until_async(
            lambda: _read_counters(db_path) or None,
            what="the counters row for the disconnected stream",
            timeout_s=5.0,
        )
        assert counters[0]["model_id"] == "qwen"
        assert 4 <= counters[0]["completion_tokens"] < len("echo: " + prompt) // 4

        # Deregistered, slot released, nothing queued.
        await wait_until_async(
            lambda: (
                live_app.state.request_registry.count() == 0
                and live_app.state.scheduler._inflight_for_test("qwen") == 0
                and live_app.state.scheduler._queue_size_for_test("qwen") == 0
            ),
            what="the slot release after the client disconnect",
            timeout_s=5.0,
        )
    finally:
        await slow.stop()


async def test_wall_clock_reaper_cuts_slow_stream_and_records_orphan(live_app, tmp_data_dir: Path):
    """R34 — request_max_wall_s arms the server-side backstop: a stream that
    would run for a long time is cut at the wall, the app returns by itself
    (no hang), the record is marked orphan, the partial completion is billed,
    and the slot is released."""
    slow = FakeEngine(served_model_name="qwen", knobs=Knobs(chunk_delay_s=0.2))
    await slow.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = slow.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    live_app.state.scheduler = PriorityScheduler(max_inflight=1)
    # Settings is frozen: swap in an identical instance with the reaper armed.
    live_app.state.settings = dataclasses.replace(live_app.state.settings, request_max_wall_s=0.6)
    try:
        prompt = "hello world, this is a test of the reaper path"
        body = json.dumps(
            {
                "model": "qwen",
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        async with open_stream(
            live_app,
            "/v1/chat/completions",
            {"Authorization": f"Bearer {plaintext}"},
            method="POST",
            body=body,
        ) as stream:
            assert stream.status == 200
            # The answer would stream for ~1.4s at 0.2s/frame; the 0.6s wall
            # must end it first, with the client still connected.
            assert await stream.ended(within_s=3.0)

        # The store writes history rows from a background flusher (the full
        # lifespan is live under live_app), and the flusher pulls a record out
        # of the queue the moment it is enqueued — an inline flush() races it
        # and sees an empty queue. Poll for the row instead.
        rows = await wait_until_async(
            lambda: _read_history(db_path) or None,
            what="the request_history row for the reaped stream",
            timeout_s=5.0,
        )
        assert rows[0]["model_id"] == "qwen"
        assert rows[0]["completion_tokens"] > 0
        # The warden cut the stream under a connected client: an orphan.
        assert rows[0]["orphan"] == 1

        # The reaper tears down synchronously (no cancellation), so the
        # counters row is already written by the time the stream ends.
        await live_app.state.ledger.flush()
        counters = _read_counters(db_path)
        assert len(counters) == 1
        assert counters[0]["completion_tokens"] > 0

        assert live_app.state.request_registry.count() == 0
        assert live_app.state.scheduler._inflight_for_test("qwen") == 0
        assert live_app.state.scheduler._queue_size_for_test("qwen") == 0
    finally:
        await slow.stop()


async def test_non_stream_without_usage_counts_completion_via_tokenizer(
    live_app, tmp_data_dir: Path
):
    """R34 — some engines answer 200 without a usage block. The completion
    must still be billed via the tokenizer — the same fallback the two
    streaming paths already use — never zero for a served completion."""
    engine = FakeEngine(served_model_name="qwen", knobs=Knobs(non_stream_usage=False))
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert r.status_code == 200
        body = r.json()
        # The engine sent no usage block; the proxy passes the body through.
        assert "usage" not in body
        assert body["choices"][0]["message"]["content"] == "echo: hi"

        await live_app.state.ledger.flush()
        rows = _read_counters(db_path)
        assert len(rows) == 1
        assert rows[0]["model_id"] == "qwen"
        assert rows[0]["prompt_tokens"] == 5
        # "echo: hi" is 8 chars = 2 fake tokens. Zero here means a served
        # completion billed nothing.
        assert rows[0]["completion_tokens"] == 2
    finally:
        await engine.stop()


async def test_non_stream_null_usage_bills_via_tokenizer(live_app, tmp_data_dir: Path):
    """R36 — some OpenAI-compatible engines answer a non-stream 200 whose
    `usage` key is PRESENT BUT `null`. The proxy must treat that exactly like
    an absent usage block — pass the 200 through and bill the served
    completion via the tokenizer fallback — not 500 a completion the engine
    actually served (pre-R36 the unguarded `get("usage", {}).get(...)` raised
    AttributeError on the `None`, a bare 500 with nothing billed)."""
    engine = FakeEngine(served_model_name="qwen", knobs=Knobs(non_stream_usage_null=True))
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )

        # The engine answered 200; the client gets its completion, not a 500
        # from the proxy's own usage read.
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "echo: hi"
        # The proxy passes the engine's null usage through, uninterpreted.
        assert body["usage"] is None

        await live_app.state.ledger.flush()
        rows = _read_counters(db_path)
        assert len(rows) == 1
        assert rows[0]["model_id"] == "qwen"
        assert rows[0]["prompt_tokens"] == 5
        # "echo: hi" is 8 chars = 2 fake tokens — the tokenizer fallback.
        # Zero here means the crash ate the bill.
        assert rows[0]["completion_tokens"] == 2
    finally:
        await engine.stop()


async def test_two_loaded_models_route_to_their_own_ports(live_app, tmp_data_dir: Path):
    """R34 — each model id resolves to its own engine port. The fake's
    unknown-model trap (200 + empty content when the requested name does not
    match the served one) makes any cross-routing visible as a wrong model
    echo and an empty completion."""
    qwen_eng = FakeEngine(served_model_name="qwen")
    llama_eng = FakeEngine(served_model_name="llama")
    await qwen_eng.start()
    await llama_eng.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    _seed_llama(db_path)
    live_app.state.supervisor._ports["qwen"] = qwen_eng.port
    live_app.state.supervisor._ports["llama"] = llama_eng.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            rq = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
            rl = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "llama", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert rq.status_code == 200
        assert rl.status_code == 200
        # The right engine served each: the response names the model that
        # actually answered, and the trap's empty content is not present.
        assert rq.json()["model"] == "qwen"
        assert rl.json()["model"] == "llama"
        assert rq.json()["choices"][0]["message"]["content"] == "echo: hi"
        assert rl.json()["choices"][0]["message"]["content"] == "echo: hi"
        # Each engine saw exactly one request, for its own name.
        assert [s["model"] for s in qwen_eng.knobs.seen] == ["qwen"]
        assert [s["model"] for s in llama_eng.knobs.seen] == ["llama"]

        await live_app.state.ledger.flush()
        rows = _read_counters(db_path)
        by_model = {r["model_id"]: r for r in rows}
        assert set(by_model) == {"qwen", "llama"}
        assert by_model["qwen"]["requests"] == 1
        assert by_model["llama"]["requests"] == 1
    finally:
        await qwen_eng.stop()
        await llama_eng.stop()
