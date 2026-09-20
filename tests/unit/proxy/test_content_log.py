"""Tests for the config-gated, per-token-scoped content logger (content_log.py).

Pure-function tests cover the gate and the truncation. Integration tests drive
a real proxied request through the FastAPI TestClient (mirroring
test_proxy_accounting.py: seed a loaded model + token, patch httpx.send) and
assert on the JSONL file the logger writes — for both a non-streaming and a
streaming completion, plus the disabled / not-allowlisted no-op paths and the
max-chars cap.
"""

import asyncio
import dataclasses
import inspect
import json
import os
import sqlite3
import stat
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest

from app.config import Settings
from app.db.repos.tokens import hash_token
from app.proxy import content_log

# ---------------------------------------------------------------------------
# Pure-function tests: should_log gate + _cap truncation
# ---------------------------------------------------------------------------

def _settings(**over) -> Settings:
    base = dict(
        data_dir=Path("/data"),
        hf_cache_dir=Path("/hf"),
        cookie_secret="x" * 32,
        container_gpu_count=0,
    )
    base.update(over)
    return Settings(**base)


def test_should_log_disabled_returns_false():
    s = _settings(content_log_enabled=False, content_log_tokens=frozenset({"tok1"}))
    assert content_log.should_log(s, "tok1") is False


def test_should_log_enabled_but_token_not_allowlisted_returns_false():
    s = _settings(content_log_enabled=True, content_log_tokens=frozenset({"other"}))
    assert content_log.should_log(s, "tok1") is False


def test_should_log_enabled_and_allowlisted_returns_true():
    s = _settings(content_log_enabled=True, content_log_tokens=frozenset({"tok1"}))
    assert content_log.should_log(s, "tok1") is True


def test_should_log_none_token_returns_false():
    s = _settings(content_log_enabled=True, content_log_tokens=frozenset({"tok1"}))
    assert content_log.should_log(s, None) is False


def test_cap_under_limit_unchanged():
    assert content_log._cap("hello", 40000) == "hello"


def test_cap_none_passthrough():
    assert content_log._cap(None, 10) is None


def test_cap_over_limit_truncates_with_marker():
    out = content_log._cap("abcdefghij", 4)
    assert out == "abcd…[truncated 6 chars]"


# ---------------------------------------------------------------------------
# Integration tests through the proxy forward path
# ---------------------------------------------------------------------------

def _seed_loaded(db_path):
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
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: token_count_for(text))
    return cache


def _configure(client, *, path, enabled, tokens, max_chars=40000):
    """Replace app.state.settings (frozen) with a content-log-configured copy."""
    client.app.state.settings = dataclasses.replace(
        client.app.state.settings,
        content_log_enabled=enabled,
        content_log_tokens=frozenset(tokens),
        content_log_path=path,
        content_log_max_chars=max_chars,
    )


def _fake_nonstream_resp(body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    resp.aclose = AsyncMock()
    return resp


def _read_lines(path: Path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_disabled_writes_nothing(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    _configure(client, path=log_path, enabled=False, tokens={"tok1"})

    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert _read_lines(log_path) == []


def test_enabled_but_token_not_allowlisted_writes_nothing(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    # Enabled, but the allowlist does NOT contain tok1.
    _configure(client, path=log_path, enabled=True, tokens={"someone-else"})

    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert _read_lines(log_path) == []


def test_nonstream_allowlisted_writes_one_line(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if "hi" in text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    _configure(client, path=log_path, enabled=True, tokens={"tok1"})

    body = {
        "choices": [{"message": {"content": "the answer is 42"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200

    lines = _read_lines(log_path)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["token_id"] == "tok1"
    assert rec["model_id"] == "qwen"
    assert rec["served_name"] == "qwen"
    assert rec["stream"] is False
    assert rec["max_tokens"] == 128
    assert rec["prompt_tokens"] == 5
    assert rec["completion_tokens"] == 4
    assert rec["finish_reason"] == "stop"
    assert rec["prompt"] == "hi"
    assert rec["completion"] == "the answer is 42"
    assert isinstance(rec["ts"], int | float)


def test_stream_allowlisted_writes_one_line(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(
        lambda text: len(text.split()) if text else 0
    )
    log_path = tmp_data_dir / "content.jsonl"
    _configure(client, path=log_path, enabled=True, tokens={"tok1"})

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def aiter():
        for c in sse_chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/event-stream"}
    fake_resp.aiter_bytes = aiter
    fake_resp.aclose = AsyncMock()

    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={
                "model": "qwen",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            for _ in r.iter_bytes():
                pass
            assert r.status_code == 200

    lines = _read_lines(log_path)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["token_id"] == "tok1"
    assert rec["stream"] is True
    assert rec["max_tokens"] == 64
    assert rec["prompt_tokens"] == 1  # "hi" -> 1 word
    assert rec["completion_tokens"] == 2  # "hello world" -> 2 words
    assert rec["finish_reason"] == "stop"
    assert rec["prompt"] == "hi"
    assert rec["completion"] == "hello world"


def test_max_chars_truncation_in_written_line(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_loaded(tmp_data_dir / "vllm-warden.db")
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda text: 5 if text else 0)
    log_path = tmp_data_dir / "content.jsonl"
    # Cap at 5 chars so the 10-char completion gets truncated.
    _configure(client, path=log_path, enabled=True, tokens={"tok1"}, max_chars=5)

    body = {
        "choices": [{"message": {"content": "abcdefghij"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_fake_nonstream_resp(body))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200

    lines = _read_lines(log_path)
    assert len(lines) == 1
    assert lines[0]["completion"] == "abcde…[truncated 5 chars]"


# ---------------------------------------------------------------------------
# On-disk permissions
#
# The records hold users' prompts and completions verbatim, on the same
# persistent volume as the SQLite database. Created at the process umask
# (typically 0o022 -> 0o644) every prompt is readable by anything else in the
# container or on the host that can reach the volume. Both modes are asserted
# under a wide-open umask, so these fail if the modes are ever left to the
# environment again.
# ---------------------------------------------------------------------------

def _write_one(path: Path, prompt: str = "hi", completion: str = "there", **over) -> None:
    """Drive one ``write_entry`` to completion from synchronous test code.

    ``write_entry`` is a coroutine since the write moved off the event loop
    (``anyio.to_thread.run_sync``), so a plain call would only build a
    coroutine object and write nothing — which is precisely what these tests
    would then fail to notice. ``asyncio.run`` gives it a loop and, crucially,
    does not return until the worker thread has finished the write.
    """
    asyncio.run(
        content_log.write_entry(
            _settings(content_log_path=path, **over),
            token_id="tok1",
            model_id="qwen",
            served_name="qwen",
            stream=False,
            max_tokens=None,
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
            prompt=prompt,
            completion=completion,
        )
    )


def _under_umask(value, fn):
    """Run ``fn`` with the process umask forced to ``value``, then restore it."""
    previous = os.umask(value)
    try:
        return fn()
    finally:
        os.umask(previous)


def test_write_entry_creates_the_log_directory_mode_0700(tmp_path):
    path = tmp_path / "logs" / "content.jsonl"
    _under_umask(0o000, lambda: _write_one(path))
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_write_entry_creates_the_log_file_mode_0600(tmp_path):
    path = tmp_path / "logs" / "content.jsonl"
    _under_umask(0o000, lambda: _write_one(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_entry_still_appends_once_the_mode_is_applied(tmp_path):
    """The hardening must not cost the module its one job: appending records."""
    path = tmp_path / "logs" / "content.jsonl"
    _under_umask(0o000, lambda: _write_one(path, prompt="first"))
    _under_umask(0o000, lambda: _write_one(path, prompt="second"))
    assert [ln["prompt"] for ln in _read_lines(path)] == ["first", "second"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_entry_leaves_an_existing_files_mode_alone(tmp_path):
    """Documented choice: create tight, never re-chmod what is already there.

    An operator may have widened the file deliberately (a log shipper reading
    it as another user), and this runs on every logged request — it must not
    fight that choice once per request. The cost is that a file created by a
    build from before this change keeps its old mode; the changelog tells
    operators to chmod that one file once.
    """
    path = tmp_path / "logs" / "content.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("")
    os.chmod(path, 0o644)
    _under_umask(0o000, lambda: _write_one(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert len(_read_lines(path)) == 1


# ---------------------------------------------------------------------------
# _cap: there is no value that means "no limit"
#
# ``max_chars >= 0 and len(text) > max_chars`` made VW_CONTENT_LOG_MAX_CHARS=-1
# an undocumented escape hatch that wrote every record in full — and -1 is
# exactly what an operator reaches for, since VW_REQUEST_MAX_WALL_S=0 means
# "no cap" elsewhere in the same .env.example. A character count has no
# negative reading, so a negative one is clamped to 0: the record keeps its
# shape and its metadata, and captures no content.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [-1, -40000])
def test_cap_negative_limit_captures_nothing(limit):
    assert content_log._cap("abcdefghij", limit) == "…[truncated 10 chars]"


def test_cap_zero_limit_captures_nothing():
    assert content_log._cap("abcdefghij", 0) == "…[truncated 10 chars]"


def test_cap_negative_limit_still_passes_none_through():
    assert content_log._cap(None, -1) is None


# ---------------------------------------------------------------------------
# The write is off the event loop, and still strictly ordered
#
# write_entry did open/write/close with no await, called straight from the
# async forward path. The image runs a SINGLE uvicorn worker, so there is no
# worker parallelism to absorb it: every other in-flight request — SSE chunk
# pumping to other clients included — stalled for the duration of an ~80 KB
# synchronous disk write, once per logged request.
#
# Offloading to a thread pool buys the loop back but introduces the risk the
# single worker had ruled out: two threads interleaving inside one file. The
# record's position is therefore fixed ON the loop, before the offload, and
# the writer drains that queue under a lock.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_content_log_state():
    """Module-level writer state must not leak between tests."""
    content_log._reset_state_for_tests()
    yield
    content_log._reset_state_for_tests()


def test_write_entry_is_a_coroutine_function():
    assert inspect.iscoroutinefunction(content_log.write_entry)


def test_write_entry_does_not_block_the_event_loop(tmp_path):
    """A slow write must not stop the loop servicing everything else.

    The tap is _open_append, held for 0.3 s. A ticker task runs alongside at
    1 ms granularity; with a synchronous write it gets zero or one tick.
    """
    path = tmp_path / "content.jsonl"
    real_open = content_log._open_append

    def slow_open(p):
        time.sleep(0.3)
        return real_open(p)

    async def scenario():
        ticks = 0
        stop = False

        async def ticker():
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.001)

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        await content_log.write_entry(
            _settings(content_log_path=path),
            token_id="tok1", model_id="qwen", served_name="qwen", stream=False,
            max_tokens=None, prompt_tokens=1, completion_tokens=1,
            finish_reason="stop", prompt="hi", completion="there",
        )
        stop = True
        await t
        return ticks

    with patch.object(content_log, "_open_append", slow_open):
        ticks = asyncio.run(scenario())

    assert ticks >= 20, f"event loop stalled during the write ({ticks} ticks)"
    assert len(_read_lines(path)) == 1


def test_concurrent_writes_are_complete_intact_and_in_submission_order(tmp_path):
    """50 concurrent callers: every record present, parseable, and in order.

    Each write is slowed inside the worker thread so a naive one-thread-per-
    record offload would interleave partial lines and shuffle the file.
    """
    path = tmp_path / "content.jsonl"
    real_open = content_log._open_append

    def slow_open(p):
        time.sleep(0.005)
        return real_open(p)

    n = 50

    async def scenario():
        s = _settings(content_log_path=path)
        await asyncio.gather(*[
            content_log.write_entry(
                s,
                token_id="tok1", model_id="qwen", served_name="qwen",
                stream=False, max_tokens=None, prompt_tokens=1,
                completion_tokens=1, finish_reason="stop",
                prompt=f"prompt-{i:03d}", completion="x" * 2000,
            )
            for i in range(n)
        ])

    with patch.object(content_log, "_open_append", slow_open):
        asyncio.run(scenario())

    raw = path.read_text().splitlines()
    assert len(raw) == n, f"expected {n} lines, found {len(raw)}"
    recs = [json.loads(ln) for ln in raw]  # raises on an interleaved line
    assert [r["prompt"] for r in recs] == [f"prompt-{i:03d}" for i in range(n)]
    assert all(r["completion"] == "x" * 2000 for r in recs)


# ---------------------------------------------------------------------------
# Failure amplification when the disk fills
#
# The bare ``log.warning(..., exc_info=True)`` emitted a full traceback per
# logged request. The failure this file causes is a full volume, so the
# handler added write pressure to a disk already out of space, once per
# request, for as long as the operator left the logger armed.
# ---------------------------------------------------------------------------


def _failing_open(_path):
    raise OSError(28, "No space left on device")


def test_repeated_identical_write_failures_warn_once(tmp_path, caplog):
    path = tmp_path / "content.jsonl"
    with caplog.at_level("WARNING", logger="vllm_warden.content_log"):
        with patch.object(content_log, "_open_append", _failing_open):
            for _ in range(5):
                _write_one(path)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert str(path) in warnings[0].getMessage()
    assert warnings[0].exc_info is not None


def test_a_successful_write_resets_the_suppression(tmp_path, caplog):
    """A transient failure must not permanently silence the warning."""
    path = tmp_path / "content.jsonl"
    with caplog.at_level("WARNING", logger="vllm_warden.content_log"):
        with patch.object(content_log, "_open_append", _failing_open):
            _write_one(path)
            _write_one(path)
        _write_one(path)  # succeeds — clears the suppression
        with patch.object(content_log, "_open_append", _failing_open):
            _write_one(path)

    detailed = [r for r in caplog.records if r.exc_info is not None]
    assert len(detailed) == 2, [r.getMessage() for r in caplog.records]
    assert len(_read_lines(path)) == 1


def test_the_suppressed_count_is_reported_on_recovery(tmp_path, caplog):
    path = tmp_path / "content.jsonl"
    with caplog.at_level("WARNING", logger="vllm_warden.content_log"):
        with patch.object(content_log, "_open_append", _failing_open):
            for _ in range(4):
                _write_one(path)
        _write_one(path)
    recovery = [
        r for r in caplog.records if "3 further identical" in r.getMessage() and r.exc_info is None
    ]
    assert recovery, [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# VW_CONTENT_LOG_MAX_BYTES — the file-level cap, fail-safe
#
# Bounded at the sink, which is the one choke point every record passes
# through. Hitting the ceiling STOPS the writing and warns once. It does not
# rotate and does not delete: the hazard is an outage (the SQLite database is
# on this volume), the right response is human attention, and single-
# generation rotation would both double the real budget and destroy the
# incidents the operator turned the logger on to capture.
# ---------------------------------------------------------------------------


def test_writes_stop_once_the_file_reaches_the_ceiling(tmp_path):
    path = tmp_path / "content.jsonl"
    _write_one(path, prompt="first", content_log_max_bytes=0)
    at_ceiling = path.stat().st_size
    for i in range(5):
        _write_one(path, prompt=f"blocked-{i}", content_log_max_bytes=at_ceiling)
    assert path.stat().st_size == at_ceiling
    assert [ln["prompt"] for ln in _read_lines(path)] == ["first"]


def test_the_ceiling_neither_rotates_nor_deletes(tmp_path):
    path = tmp_path / "content.jsonl"
    _write_one(path, prompt="keepme", content_log_max_bytes=0)
    size = path.stat().st_size
    _write_one(path, prompt="dropped", content_log_max_bytes=size)
    assert path.exists()
    assert path.stat().st_size == size
    assert list(path.parent.iterdir()) == [path], "the sink must not rotate"


def test_the_ceiling_warns_once_naming_the_file_and_the_limit(tmp_path, caplog):
    path = tmp_path / "content.jsonl"
    _write_one(path, prompt="first", content_log_max_bytes=0)
    size = path.stat().st_size
    with caplog.at_level("WARNING", logger="vllm_warden.content_log"):
        for _ in range(4):
            _write_one(path, prompt="blocked", content_log_max_bytes=size)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    msg = warnings[0].getMessage()
    assert str(path) in msg
    assert str(size) in msg


def test_a_max_bytes_of_zero_leaves_the_sink_unbounded(tmp_path):
    path = tmp_path / "content.jsonl"
    for i in range(4):
        _write_one(path, prompt=f"r{i}", content_log_max_bytes=0)
    assert [ln["prompt"] for ln in _read_lines(path)] == ["r0", "r1", "r2", "r3"]


def test_records_below_the_ceiling_still_write(tmp_path):
    path = tmp_path / "content.jsonl"
    for i in range(3):
        _write_one(path, prompt=f"r{i}", content_log_max_bytes=10 * 1024 * 1024)
    assert [ln["prompt"] for ln in _read_lines(path)] == ["r0", "r1", "r2"]
