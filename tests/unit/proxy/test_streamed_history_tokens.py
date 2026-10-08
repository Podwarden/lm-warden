"""R23: the per-request history must record the FINAL completion count.

The streaming ``gen()`` used to deregister the live request — which writes the
``request_history`` row from ``live_req.completion_tokens`` — BEFORE it
computed the final count. So the history row kept the coarse live
``delta_count`` estimate (content deltas only) while the per-key counters got
the true number: a streamed request that generated 20 tokens of reasoning plus
one content token was shown with ``completion_tokens: 1`` on the stats page.

These tests pin the ordering: the history row must agree with the per-key
counter for a stream whose channels differ from the delta count, and the live
registry must still receive the running estimate while the stream is in flight.
"""

import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt

from app.db.repos.tokens import hash_token
from app.proxy.request_registry import LiveRequest
from tests.unit.proxy.ledger_helpers import flush_ledger


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


def _read_counters(db_path):
    with sqlite3.connect(db_path) as db:
        cur = db.execute(
            "SELECT model_id, token_id, requests, prompt_tokens, completion_tokens FROM counters"
        )
        return [dict(zip([d[0] for d in cur.description], r, strict=False)) for r in cur.fetchall()]


def _history_completion_tokens(db_path, n):
    """Wait for ``n`` request_history rows (background writer) and return their
    completion_tokens, oldest first. Same generous deadline as
    test_queue_wait_recorded: this waits on the store's background writer."""
    deadline = time.time() + 30
    rows = []
    while time.time() < deadline:
        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT completion_tokens FROM request_history ORDER BY finished_at ASC"
            ).fetchall()
        if len(rows) >= n:
            return [r[0] for r in rows]
        time.sleep(0.05)
    raise AssertionError(f"only {len(rows)} of {n} request_history rows were written")


def _make_fake_tokenizer(token_count_for):
    cache = MagicMock()
    cache.count = AsyncMock(
        side_effect=lambda repo, text, *, fallback_repo=None: token_count_for(text)
    )
    return cache


def _run_stream(client, plaintext, chunks):
    """Drive the streaming forward path; returns the bytes the client received."""

    async def aiter():
        for c in chunks:
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
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as r:
            body = b"".join(r.iter_bytes())
            assert r.status_code == 200
    return body


def test_streamed_history_record_carries_the_final_count(tmp_data_dir, client):
    """The history row must equal the per-key counter, not the delta count.

    The stream has ONE content delta ("Sunny") but lots of reasoning and
    tool-call text: the live delta_count estimate is 1, the final count is the
    whole 48 characters (fake tokenizer = char count).
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"step one"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"step two"}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"Paris\\"}"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"Sunny"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    _run_stream(client, plaintext, chunks)

    flush_ledger(client)
    rows = _read_counters(db_path)
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 48
    (hist,) = _history_completion_tokens(db_path, 1)
    assert hist == rows[0]["completion_tokens"]
    assert hist == 48


def test_streamed_history_record_carries_the_usage_frame_count(tmp_data_dir, client):
    """When the stream carries usage.completion_tokens, the history row must
    carry that exact number (the delta count here is 1)."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"step one"}}],"usage":null}\n\n',
        b'data: {"choices":[{"delta":{"content":"Sunny"}}],"usage":null}\n\n',
        b'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":777,"total_tokens":788}}\n\n',
        b"data: [DONE]\n\n",
    ]
    _run_stream(client, plaintext, chunks)

    flush_ledger(client)
    rows = _read_counters(db_path)
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 777
    (hist,) = _history_completion_tokens(db_path, 1)
    assert hist == rows[0]["completion_tokens"]
    assert hist == 777


def test_non_stream_history_record_is_unchanged(tmp_data_dir, client):
    """Control: the non-stream path already wrote the body's usage count onto
    the history row before deregistering; that must stay exactly as it was."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    body = {
        "id": "x",
        "model": "qwen",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.aread = AsyncMock(return_value=json.dumps(body).encode())
    fake_resp.aclose = AsyncMock()
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake_resp)):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200

    flush_ledger(client)
    rows = _read_counters(db_path)
    assert len(rows) == 1
    assert rows[0]["completion_tokens"] == 2
    (hist,) = _history_completion_tokens(db_path, 1)
    assert hist == 2


class _RecordingLiveRequest(LiveRequest):
    """Records every write to ``completion_tokens``.

    Lets the control below distinguish the coarse mid-stream estimate (written
    by the streaming loop as it counts content deltas) from the final count
    (written once at the end).
    """

    instances: list["_RecordingLiveRequest"] = []

    def __init__(self, *args, **kwargs):
        _RecordingLiveRequest.instances.append(self)
        self._ct_writes: list[int] = []
        self._ct_value: int = 0
        super().__init__(*args, **kwargs)

    @property
    def completion_tokens(self) -> int:
        return self._ct_value

    @completion_tokens.setter
    def completion_tokens(self, value: int) -> None:
        self._ct_value = value
        self._ct_writes.append(value)


def test_live_registry_still_shows_the_running_estimate_mid_stream(tmp_data_dir, client):
    """Control: while the stream is in flight the live registry must still
    receive the running delta_count estimate — the end-of-stream write must
    not have displaced the mid-stream updates.

    The update interval is patched to 0 so the estimate is pushed on the very
    first content delta; the stream then carries reasoning text that the
    estimate does not (and must not) include.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    client.app.state.supervisor._ports["qwen"] = 19099
    client.app.state.tokenizers = _make_fake_tokenizer(lambda t: len(t) if t else 0)

    _RecordingLiveRequest.instances = []
    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ab"}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    with (
        patch("app.proxy.routes.LiveRequest", _RecordingLiveRequest),
        patch("app.proxy.routes._LIVE_UPDATE_INTERVAL_S", 0.0),
    ):
        _run_stream(client, plaintext, chunks)

    assert len(_RecordingLiveRequest.instances) == 1
    writes = _RecordingLiveRequest.instances[0]._ct_writes
    # The mid-stream estimate: one content delta seen so far.
    assert 1 in writes
    # ...and it must not have absorbed the reasoning text. The final write is
    # the full count ("thinking" + "ab" = 10 chars with the char-counting
    # fake), and it lands AFTER the estimate.
    assert writes[-1] == 10
    assert writes.index(1) < writes.index(10)

    flush_ledger(client)
    rows = _read_counters(db_path)
    assert rows[0]["completion_tokens"] == 10
    (hist,) = _history_completion_tokens(db_path, 1)
    assert hist == 10
