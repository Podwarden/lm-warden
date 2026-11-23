"""Debug capture: a key's opt-in, time-limited recording of exact request bodies and
raw responses, for replay. Admin-only; size-capped on the data volume."""

from __future__ import annotations

import os
import time

from app.proxy.capture import CaptureStore, capturing


def _rec(i: int, size: int = 100) -> dict:
    return {"id": f"req{i:04d}", "client_body": "x" * size, "response": "r"}


def test_put_then_get_round_trips_and_unknown_is_none(tmp_path):
    st = CaptureStore(tmp_path, max_bytes=10**9)
    st.put_sync(_rec(1))
    got = st.get("req0001")
    assert got is not None and got["client_body"] == "x" * 100 and got["response"] == "r"
    assert st.get("nope") is None
    assert st.get("../../etc/passwd") is None


def test_files_are_owner_only(tmp_path):
    st = CaptureStore(tmp_path, max_bytes=10**9)
    st.put_sync(_rec(1))
    (f,) = list(tmp_path.rglob("*.json.gz"))
    assert (f.stat().st_mode & 0o777) == 0o600


def test_over_the_cap_the_oldest_captures_go_first(tmp_path):
    st = CaptureStore(tmp_path, max_bytes=10**9)
    for i in range(5):
        st.put_sync(_rec(i, size=50_000))
    files = sorted(tmp_path.rglob("*.json.gz"))
    for k, f in enumerate(files):  # make age order explicit
        os.utime(f, (time.time() - 1000 + k, time.time() - 1000 + k))
    st.max_bytes = sum(f.stat().st_size for f in files[2:])
    st.enforce()
    left = sorted(p.name for p in tmp_path.rglob("*.json.gz"))
    assert left == ["req0002.json.gz", "req0003.json.gz", "req0004.json.gz"]


def test_capturing_only_inside_the_keys_window():
    now = 1000.0
    assert capturing(None, now) is False
    assert capturing(999.0, now) is False
    assert capturing(1001.0, now) is True


# -- forward and the admin read route ---------------------------------------------

import json  # noqa: E402
import sqlite3  # noqa: E402

from tests.conftest import csrf_header, seed_admin_user, wait_until  # noqa: E402
from tests.unit.proxy.test_messages import _ready, _sse_upstream  # noqa: E402
from tests.unit.proxy.test_thinking_policy import _go  # noqa: E402

CHAT = {
    "model": "qwen",
    "tools": [{"type": "function", "function": {"name": "todowrite", "parameters": {}}}],
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "a", "reasoning_content": "thought"},
        {"role": "user", "content": "again"},
    ],
}


def _key(db_path, **cols):
    with sqlite3.connect(db_path) as db:
        for k, v in cols.items():
            db.execute(f"UPDATE api_tokens SET {k} = ?", (v,))
        db.commit()


def _rid(db_path):
    def _row():
        with sqlite3.connect(db_path) as db:
            return db.execute("SELECT id FROM request_history").fetchone()

    return wait_until(_row, what="history row")[0]


def _admin(client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    return {"Authorization": f"Bearer {r.json()['access_token']}", **csrf_header(client)}


def test_a_captured_key_records_bodies_and_response_readable_by_admin(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _key(
        db_path,
        capture_until=time.time() + 600,
        prompt_diet=json.dumps({"strip_tools": ["todowrite"], "strip_reasoning": True}),
    )
    r, _ = _go(client, CHAT)
    assert r.status_code == 200
    rid = _rid(db_path)
    store = client.app.state.capture_store
    rec = wait_until(lambda: store.get(rid), what="capture file")
    assert json.loads(rec["request_body"])["tools"][0]["function"]["name"] == "todowrite"
    eng = json.loads(rec["engine_body"])
    assert "tools" not in eng and "reasoning_content" not in eng["messages"][1]
    assert json.loads(rec["prompt_diet"])["tools"] == 1 and rec["response"]
    hdrs = _admin(client, tmp_data_dir)
    got = client.get(f"/api/capture/{rid}", headers=hdrs)
    assert got.status_code == 200 and got.json()["engine_body"] == rec["engine_body"]
    assert client.get(f"/api/capture/{rid}").status_code == 401
    assert client.get("/api/capture/unknownid", headers=hdrs).status_code == 404


def test_a_stream_is_captured_as_raw_sse(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _key(db_path, capture_until=time.time() + 600)
    up = _sse_upstream(
        [
            {"choices": [{"index": 0, "delta": {"content": "he"}}]},
            {"choices": [{"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"}]},
        ]
    )
    r, _ = _go(client, {**CHAT, "stream": True}, upstream=up)
    assert r.status_code == 200
    rid = _rid(db_path)
    rec = wait_until(lambda: client.app.state.capture_store.get(rid), what="capture file")
    assert '"he"' in rec["response"] and '"llo"' in rec["response"]
    # the engine body differs only by the proxy's own priority field here
    assert json.loads(rec["engine_body"]) == {**json.loads(rec["request_body"]), "priority": -5}


def test_no_window_no_capture(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _key(db_path, capture_until=time.time() - 1)
    _go(client, CHAT)
    rid = _rid(db_path)
    time.sleep(0.3)
    assert client.app.state.capture_store.get(rid) is None


def test_patch_capture_hours_opens_and_closes_the_window(tmp_data_dir, client):
    hdrs = _admin(client, tmp_data_dir)
    tid = client.post("/api/tokens", json={"name": "k"}, headers=hdrs).json()["id"]
    r = client.patch(f"/api/tokens/{tid}", json={"capture_hours": 2}, headers=hdrs)
    assert r.status_code == 200 and abs(r.json()["capture_until"] - (time.time() + 7200)) < 60
    r = client.patch(f"/api/tokens/{tid}", json={"capture_hours": 0}, headers=hdrs)
    assert r.json()["capture_until"] is None
    assert (
        client.patch(f"/api/tokens/{tid}", json={"capture_hours": -1}, headers=hdrs).status_code
        == 422
    )


# -- the record holds what the client called, not only what the engine got (#338) ----

from unittest.mock import AsyncMock, patch  # noqa: E402

from tests.unit.proxy.test_messages import AUTH, _json_upstream  # noqa: E402
from tests.unit.proxy.test_thinking_policy import OK  # noqa: E402

# Odd spacing and key order on purpose: the record must hold these exact bytes.
_ANTHROPIC = (
    b'{"model":"qwen",  "max_tokens":20,"messages":[{"role":"user",'
    b'"content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}]}'
)
_RESPONSES = b'{"model": "qwen","input":[{"role":"user","content":"hi"}],"max_output_tokens":20}'
_CHAT_RAW = b'{"model":"qwen",   "messages":[{"role":"user","content":"hi"}]}'


def _captured(client, tmp_data_dir, path, raw):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _key(db_path, capture_until=time.time() + 600)
    send = AsyncMock(return_value=_json_upstream(OK))
    with patch("httpx.AsyncClient.send", new=send):
        r = client.post(path, headers={**AUTH, "content-type": "application/json"}, content=raw)
    assert r.status_code == 200, r.text
    rid = _rid(db_path)
    return wait_until(lambda: client.app.state.capture_store.get(rid), what="capture file"), rid


def test_chat_record_is_the_wire_request_and_has_no_translation(tmp_data_dir, client):
    rec, _ = _captured(client, tmp_data_dir, "/v1/chat/completions", _CHAT_RAW)
    assert rec["path"] == "/v1/chat/completions"
    assert rec["request_body"].encode() == _CHAT_RAW
    assert rec["translated_body"] is None


def test_messages_record_keeps_the_anthropic_body_and_the_translation(tmp_data_dir, client):
    rec, rid = _captured(client, tmp_data_dir, "/v1/messages", _ANTHROPIC)
    assert rec["path"] == "/v1/messages"
    assert rec["request_body"].encode() == _ANTHROPIC
    chat = json.loads(rec["translated_body"])
    assert chat["messages"][-1]["role"] == "user" and "cache_control" not in rec["translated_body"]
    assert "cache_control" in rec["request_body"]
    got = client.get(f"/api/capture/{rid}", headers=_admin(client, tmp_data_dir))
    assert got.status_code == 200 and got.json()["path"] == "/v1/messages"
    assert got.json()["translated_body"] == rec["translated_body"]


def test_responses_record_keeps_the_responses_body_and_the_translation(tmp_data_dir, client):
    rec, _ = _captured(client, tmp_data_dir, "/v1/responses", _RESPONSES)
    assert rec["path"] == "/v1/responses"
    assert rec["request_body"].encode() == _RESPONSES
    chat = json.loads(rec["translated_body"])
    assert "messages" in chat and "input" not in chat


def test_a_record_from_before_the_split_still_reads(tmp_data_dir, client):
    old = {
        "id": "oldrec0001",
        "ts": time.time(),
        "path": "/v1/chat/completions",
        "request_body": "{}",
        "engine_body": None,
        "response": "r",
    }
    client.get("/healthz")
    client.app.state.capture_store.put_sync(old)
    got = client.get("/api/capture/oldrec0001", headers=_admin(client, tmp_data_dir))
    assert got.status_code == 200 and got.json() == old
