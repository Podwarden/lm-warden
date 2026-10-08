"""R43 (issue #279) -- a stats-bookkeeping failure must never fail a paid /v1
request.

Production shape (7x RTX 5090, DP7 engine, 224 concurrency): ~2% of requests
failed with ``sqlite3.OperationalError: database is locked`` inside the
proxy's OWN bookkeeping (the last_used_at touch in auth, the counters write
after the answer). Stage 1 caught and counted those writes; stage 3 (#279) took
them off the request path altogether: ``_record_counters`` and auth's touch only
accumulate in the in-memory ``Ledger`` (app/proxy/ledger.py), which writes one
transaction a second.

These tests drive the real app over ASGI (``live_app``) with a real
``FakeEngine`` loopback hop. The ledger is swapped for a fresh one with no
background loop so the test decides when it flushes. They pin:

* a request never touches the ledger tables, so a locked DB cannot fail it, and
  nothing is counted as a drop at request time;
* a locked DB at flush time is retried, not dropped: no drop counter moves
  and the data survives to the next flush, which writes it exactly;
* a locked token READ still fails the request (authentication is not
  bookkeeping);
* a bug in the accounting call itself still cannot fail the request;
* the error the tests simulate is the class a genuinely locked DB raises
  (proven against a real held write transaction, with a short busy_timeout).
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest

from app.db.repos.tokens import TokenRepo, hash_token
from app.proxy import bookkeeping
from app.proxy.ledger import Ledger
from app.proxy.routes import _record_counters
from app.proxy.scheduler import PriorityScheduler
from tests.fakes.fake_vllm import FakeEngine


def _seed_loaded(db_path: Path) -> str:
    """Setup done, one 'loaded' model, one inference token (id ``tok1``)."""
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


def _make_fake_tokenizer(token_count_for):
    cache = MagicMock()
    cache.count = AsyncMock(
        side_effect=lambda repo, text, *, fallback_repo=None: token_count_for(text)
    )
    return cache


def _token_count_for(text: str) -> int:
    """'hi' is 5 tokens, everything else len//4 — the fake engine's ratio."""
    return 5 if text == "hi" else max(0, len(text) // 4)


def _drop_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """What ``after`` added on top of ``before``, keyed by call site."""
    return {
        site: after[site] - before.get(site, 0)
        for site in after
        if site not in before or after[site] != before.get(site, 0)
    }


def _locked() -> sqlite3.OperationalError:
    """The production error, verbatim."""
    return sqlite3.OperationalError("database is locked")


def _manual_ledger(live_app, db_path: Path) -> Ledger:
    """A ledger with no background loop: the test flushes it when it chooses."""
    live_app.state.ledger = Ledger(db_path)
    return live_app.state.ledger


async def _flush_while_locked(ledger: Ledger, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Flush with the DB 'locked'; returns the drops it counted (none: a lock
    retry loses nothing). The pending
    data is kept (the real ``_write`` is restored afterwards)."""
    real = Ledger._write

    async def locked(self, batch):
        raise _locked()

    before = bookkeeping.dropped_total()
    monkeypatch.setattr(Ledger, "_write", locked)
    try:
        assert await ledger.flush() == 0
    finally:
        monkeypatch.setattr(Ledger, "_write", real)
    return _drop_delta(before, bookkeeping.dropped_total())


def _counter_rows(db_path: Path) -> list[tuple]:
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT model_id, token_id, requests FROM counters").fetchall()


def _last_used(db_path: Path):
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT last_used_at FROM api_tokens WHERE id='tok1'").fetchone()[0]


async def test_genuinely_locked_db_raises_the_operational_error_the_fix_catches(
    live_app, tmp_data_dir: Path
):
    """Shape proof: hold an exclusive write transaction on the same file from
    a second connection. The bookkeeping write then fails with
    ``sqlite3.OperationalError: database is locked`` — the exact error class
    the HTTP tests below monkeypatch, only here it comes from a real lock.
    (busy_timeout is 200 ms on this connection, not the production 30 s, so
    the test does not wait out the issue's own retry budget.)"""
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_loaded(db_path)
    lock = sqlite3.connect(db_path, isolation_level=None)
    lock.execute("BEGIN EXCLUSIVE")
    lock.execute("UPDATE api_tokens SET last_used_at = datetime('now') WHERE id='tok1'")
    try:
        db = await aiosqlite.connect(db_path)
        await db.execute("PRAGMA busy_timeout = 200")
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                await TokenRepo(db).touch_last_used("tok1")
        finally:
            await db.close()
    finally:
        lock.execute("ROLLBACK")
        lock.close()


async def test_locked_touch_last_used_does_not_fail_the_request(
    live_app, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The 8/18 case: auth no longer writes last_used_at on the request path,
    so even a repo that would raise "locked" is never reached, the request
    completes, and nothing is dropped at request time. At flush a locked DB
    is not a drop (nothing is lost) and the touch survives to the next flush."""
    engine = FakeEngine(served_model_name="qwen")
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    ledger = _manual_ledger(live_app, db_path)
    # Pins "never called": were auth to touch the DB itself, this would raise.
    touch = AsyncMock(side_effect=_locked())
    monkeypatch.setattr(TokenRepo, "touch_last_used", touch)
    before = bookkeeping.dropped_total()
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "echo: hi"
        touch.assert_not_called()
        assert bookkeeping.dropped_total() == before
        assert _last_used(db_path) is None

        assert await _flush_while_locked(ledger, monkeypatch) == {}
        assert _last_used(db_path) is None
        await ledger.flush()
        assert _last_used(db_path) is not None
    finally:
        await engine.stop()


async def test_locked_counters_do_not_fail_a_non_stream_request(
    live_app, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The 10/18 case, non-stream: the engine answered, the client gets its
    200 and body. Nothing is written (or dropped) on the request path; the
    locked flush counts no drop, and the next flush writes the request
    exactly once."""
    engine = FakeEngine(served_model_name="qwen")
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    ledger = _manual_ledger(live_app, db_path)
    before = bookkeeping.dropped_total()
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "echo: hi"
        assert bookkeeping.dropped_total() == before
        assert _counter_rows(db_path) == []

        assert await _flush_while_locked(ledger, monkeypatch) == {}
        assert _counter_rows(db_path) == []
        await ledger.flush()
        assert _counter_rows(db_path) == [("qwen", "tok1", 1)]
    finally:
        await engine.stop()


async def test_locked_counters_do_not_cut_a_stream(
    live_app, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The 10/18 case, streaming: the client has the FULL answer, and the
    end-of-stream settle only accumulates. The stream ends cleanly (with its
    [DONE]), the admission slot is released, nothing is dropped at request
    time, and a locked flush later keeps the data for the next one."""
    engine = FakeEngine(served_model_name="qwen")
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    live_app.state.scheduler = PriorityScheduler(max_inflight=1)
    ledger = _manual_ledger(live_app, db_path)
    before = bookkeeping.dropped_total()
    try:
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
                assert r.status_code == 200
                chunks = [c async for c in r.aiter_bytes()]
        text = b"".join(chunks).decode()
        assert text.rstrip().endswith("data: [DONE]")
        assert "echo: hi" in text
        assert bookkeeping.dropped_total() == before
        assert live_app.state.scheduler._inflight_for_test("qwen") == 0

        assert await _flush_while_locked(ledger, monkeypatch) == {}
        await ledger.flush()
        assert _counter_rows(db_path) == [("qwen", "tok1", 1)]
    finally:
        await engine.stop()


async def test_a_locked_token_read_still_fails_the_request(
    live_app, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Control: the read that AUTHENTICATES the request is not bookkeeping.
    A DB too locked even to read the token still fails the request (the
    client retries) — the fix must not be widened over it, and nothing is
    counted as a drop. Under the ASGI test transport the unhandled error
    surfaces at the client as the same exception (under a real server the
    connection is dropped) — either way the request fails and no answer is
    delivered."""
    engine = FakeEngine(served_model_name="qwen")
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    monkeypatch.setattr(TokenRepo, "find_by_plaintext", AsyncMock(side_effect=_locked()))
    before = bookkeeping.dropped_total()
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {plaintext}"},
                    json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                )
        assert bookkeeping.dropped_total() == before
    finally:
        await engine.stop()


async def test_an_error_in_the_accounting_call_cannot_fail_the_request(
    live_app, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Unit-level on ``_record_counters`` itself: it only accumulates in memory
    and swallows ANY error from that (a bug in our own code too -- accounting
    must never fail a paid request). It is not a locked DB, so it is not
    counted as a drop either."""
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_loaded(db_path)
    request = SimpleNamespace(app=SimpleNamespace(state=live_app.state))
    model = MagicMock(id="qwen")

    before = bookkeeping.dropped_total()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(live_app.state.ledger, "record_request", boom)
    await _record_counters(request, model, "tok1", 5, 2)
    assert bookkeeping.dropped_total() == before


async def test_a_healthy_db_records_as_before_and_drops_nothing(live_app, tmp_data_dir: Path):
    """Control: with no lock at all the ledger writes land exactly as before
    (once flushed) and nothing is counted as a drop."""
    engine = FakeEngine(served_model_name="qwen")
    await engine.start()
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    live_app.state.supervisor._ports["qwen"] = engine.port
    live_app.state.tokenizers = _make_fake_tokenizer(_token_count_for)
    before = bookkeeping.dropped_total()
    try:
        transport = httpx.ASGITransport(app=live_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {plaintext}"},
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "echo: hi"
        await live_app.state.ledger.flush()
        with sqlite3.connect(db_path) as db:
            cur = db.execute(
                "SELECT model_id, token_id, requests, prompt_tokens, completion_tokens "
                "FROM counters"
            )
            rows = [
                dict(zip([d[0] for d in cur.description], row, strict=False))
                for row in cur.fetchall()
            ]
        assert rows == [
            {
                "model_id": "qwen",
                "token_id": "tok1",
                "requests": 1,
                # the fake engine reports usage.prompt_tokens=1, which wins over
                # the warden's own count
                "prompt_tokens": 1,
                "completion_tokens": 2,
            }
        ]
        assert bookkeeping.dropped_total() == before
    finally:
        await engine.stop()
