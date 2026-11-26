"""Relayed (router passthrough) requests are recorded for the sessions pages,
off the stream path (sessions pages, task 5)."""

from __future__ import annotations

import gzip
import json
import threading
import zlib

import httpx

from app.cobs2.canonical import canonicalize
from app.cobs2.ids import ids_for
from app.proxy.forest_fields import hkey
from app.proxy.messages_schemas import MessagesRequest
from app.proxy.messages_translate import to_openai_request
from app.router import relay_record
from app.router.relay_record import assemble_reply
from tests.conftest import wait_until
from tests.unit.proxy.test_cache_estimate_forward import _spy_history
from tests.unit.router.harness import CLAUDE_CODE, fake_engine
from tests.unit.router.test_data_plane import _engine_returning, _setup

SESSION_ID = "0b1c2d3e-1111-4222-8333-444455556666"
SESSION = {"X-Claude-Code-Session-Id": SESSION_ID}


def _ev(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def anthropic_sse_bytes(
    *, text="hi", tool=None, usage_in=10, cache_read=900, cache_create=40, out=7
):
    frames = [
        _ev(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [],
                    "usage": {
                        "input_tokens": usage_in,
                        "cache_read_input_tokens": cache_read,
                        "cache_creation_input_tokens": cache_create,
                        "output_tokens": 1,
                    },
                },
            },
        ),
        _ev(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _ev(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        _ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]
    stop = "end_turn"
    if tool:
        frames += [
            _ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": tool,
                        "input": {},
                    },
                },
            ),
            _ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a":'},
                },
            ),
            _ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": "1}"},
                },
            ),
            _ev("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ]
        stop = "tool_use"
    frames += [
        _ev(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop},
                "usage": {"output_tokens": out},
            },
        ),
        _ev("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(frames)


def _sse_handler(raw: bytes):
    def h(_req):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw)

    return h


def sonnet_body(messages=None, **extra):
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": "be brief",
        "messages": messages or [{"role": "user", "content": "hello"}],
    }
    body.update(extra)
    return body


def _post(client, body, headers=None):
    return client.post(
        "/v1/messages", headers={**CLAUDE_CODE, **SESSION, **(headers or {})}, json=body
    )


def _secret(client):
    return client.app.state.settings.cookie_secret


def test_passthrough_records_a_relayed_row_with_upstream_usage(tmp_data_dir, client):
    raw = anthropic_sse_bytes(text="hi", tool="Task")
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    r = _post(client, sonnet_body(stream=True))
    assert r.status_code == 200 and r.content == raw  # bytes unchanged
    rec = wait_until(lambda: recs[-1] if recs else None, what="relayed row", timeout_s=5)
    secret = _secret(client)
    assert rec["relayed"] is True and rec["upstream"] == "anthropic.example"
    assert (rec["prompt_tokens"], rec["cached_tokens"], rec["completion_tokens"]) == (950, 900, 7)
    assert rec["finish_reason"] == "tool_calls"
    assert rec["tools_out"] == [["agent", hkey(secret, "Task", 8)]]
    assert rec["cobs2_session"] == hkey(secret, SESSION_ID)
    assert rec["cobs2_conv"] and rec["cobs2_reply_msg"] and rec["cobs2_sess_ref"]
    assert rec["model"] == "claude-sonnet-4-5"
    assert rec["cache_outcome"] is None
    assert r.headers["X-LMW-Request-Id"] == rec["id"]
    assert rec["ttft_s"] is not None and rec["duration_s"] >= 0


def _turn_pair(client, tmp_data_dir):
    raw = anthropic_sse_bytes(text="answer one")
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    m1 = [{"role": "user", "content": "hello"}]
    _post(client, sonnet_body(m1, stream=True))
    t1 = wait_until(lambda: recs[0] if recs else None, what="turn 1", timeout_s=5)
    m2 = [
        *m1,
        {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
        {"role": "user", "content": "more"},
    ]
    _post(client, sonnet_body(m2, stream=True))
    t2 = wait_until(lambda: recs[1] if len(recs) > 1 else None, what="turn 2", timeout_s=5)
    return t1, t2, m2


def test_relayed_turns_continue_each_other(tmp_data_dir, client):
    t1, t2, _ = _turn_pair(client, tmp_data_dir)
    assert t1["cobs2_edge"] == "new"
    assert t2["cobs2_edge"] == "continue" and t2["cobs2_parent_req"] == t1["id"]
    assert t1["cobs2_conv"] == t2["cobs2_conv"]


def test_reply_id_matches_the_next_request_message(tmp_data_dir, client):
    t1, _, m2 = _turn_pair(client, tmp_data_dir)
    oai = to_openai_request(MessagesRequest.model_validate(sonnet_body(m2)))
    ids = ids_for(canonicalize(oai), client.app.state.cobs2_key, b"")
    assert ids.msgs[1] == t1["cobs2_reply_msg"]


def test_relay_stream_is_not_delayed_by_recording(tmp_data_dir, client, monkeypatch):
    gate = threading.Event()
    real = relay_record._compute

    def blocked(*a, **k):
        gate.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(relay_record, "_compute", blocked)
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    r = _post(client, sonnet_body(stream=True))
    assert r.status_code == 200 and r.content == raw  # delivered while hashing is blocked
    assert recs == []
    gate.set()
    wait_until(lambda: recs, what="row after unblock", timeout_s=5)


def test_recording_failure_fails_open(tmp_data_dir, client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(relay_record, "_compute", boom)
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    r = _post(client, sonnet_body(stream=True))
    assert r.status_code == 200 and r.content == raw
    rec = wait_until(lambda: recs[-1] if recs else None, what="row", timeout_s=5)
    assert rec["relayed"] is True and rec["cobs2_node"] is None
    assert rec["cobs2_session"] == hkey(_secret(client), SESSION_ID)


def test_gzip_json_reply_is_decoded_and_brotli_is_skipped():
    body = {
        "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "t", "name": "Read", "input": {}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    raw = json.dumps(body).encode()
    for enc, data in (
        ("gzip", gzip.compress(raw)),
        ("deflate", zlib.compress(raw)),
        ("identity", raw),
    ):
        rep = assemble_reply(data, is_sse=False, encoding=enc)
        assert rep is not None and rep.complete and rep.tool_names == ["Read"], enc
        assert rep.usage["input_tokens"] == 3 and rep.stop_reason == "tool_use"
    assert assemble_reply(raw, is_sse=False, encoding="br") is None


def test_sse_reply_without_message_stop_is_incomplete():
    raw = anthropic_sse_bytes()
    cut = raw[: raw.index(b"event: message_stop")]
    rep = assemble_reply(cut, is_sse=True, encoding="identity")
    assert rep is not None and not rep.complete


def test_error_status_relay_is_recorded_as_error(tmp_data_dir, client):
    def h(_req):
        return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})

    _setup(client, tmp_data_dir, anthropic=h)
    recs = _spy_history(client)
    r = _post(client, sonnet_body())
    assert r.status_code == 529
    rec = wait_until(lambda: recs[-1] if recs else None, what="row", timeout_s=5)
    assert (
        rec["finish_reason"] == "error"
        and rec["cobs2_reply_msg"] is None
        and rec["relayed"] is True
    )


def test_fallback_relay_records_too(tmp_data_dir, client):
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)

    def bad(_req):
        return httpx.Response(502, json={"error": "x"})

    body = sonnet_body(model="claude-haiku-4-5", stream=True)
    with fake_engine(client, _engine_returning(bad)):
        r = _post(client, body)
    assert r.status_code == 200
    rec = wait_until(
        lambda: next((x for x in recs if x.get("relayed")), None), what="relayed row", timeout_s=5
    )
    assert rec["upstream"] == "anthropic.example"


def test_capture_over_cap_keeps_timings_drops_reply(tmp_data_dir, client, monkeypatch):
    monkeypatch.setattr(relay_record, "CAPTURE_MAX_BYTES", 10)
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    r = _post(client, sonnet_body(stream=True))
    assert r.content == raw
    rec = wait_until(lambda: recs[-1] if recs else None, what="row", timeout_s=5)
    assert rec["cobs2_reply_msg"] is None and rec["ttft_s"] is not None


def test_count_tokens_is_not_recorded(tmp_data_dir, client):
    def h(_req):
        return httpx.Response(200, json={"input_tokens": 5})

    _setup(client, tmp_data_dir, anthropic=h)
    recs = _spy_history(client)
    r = client.post(
        "/v1/messages/count_tokens", headers={**CLAUDE_CODE, **SESSION}, json=sonnet_body()
    )
    assert r.status_code == 200
    import time

    time.sleep(0.3)
    assert recs == []


def test_finished_at_is_the_close_time_not_the_recording_time(tmp_data_dir, client, monkeypatch):
    """The row is written by a background task that may run late (here: hashing blocked); its
    finished_at must still be when the relayed response ended, or lanes and inferred parents shift."""
    import time

    gate = threading.Event()
    real = relay_record._compute

    def slow(*a, **k):
        gate.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(relay_record, "_compute", slow)
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    got: list[tuple[dict, float | None]] = []
    client.app.state.request_history = type(
        "H",
        (),
        {"record": lambda self, rec, finished_at=None: got.append((rec, finished_at)) or True},
    )()
    r = _post(client, sonnet_body(stream=True))
    assert r.status_code == 200
    closed = time.time()
    time.sleep(0.4)
    gate.set()
    rec, fin = wait_until(lambda: got[-1] if got else None, what="row", timeout_s=5)
    assert fin is not None and fin <= closed + 0.05


def test_recorded_relay_asks_upstream_only_for_decodable_encodings(tmp_data_dir, client):
    """Claude Code accepts br/zstd; the image cannot decode them, so a classifier's non-stream JSON
    came back unreadable (0 tokens, no finish). A recorded relay narrows Accept-Encoding to what
    both the client and the recorder handle; the client still gets an encoding it asked for."""
    seen: dict = {}
    body = {
        "content": [{"type": "text", "text": "yes"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "cache_read_input_tokens": 1000, "output_tokens": 2},
    }
    raw = gzip.compress(json.dumps(body).encode())

    def h(req):
        seen["ae"] = req.headers.get("accept-encoding")

        async def gen():
            yield raw  # a stream, so httpx keeps the gzip bytes raw (as on the wire)

        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
            content=gen(),
        )

    _setup(client, tmp_data_dir, anthropic=h)
    recs = _spy_history(client)
    r = _post(client, sonnet_body(), headers={"Accept-Encoding": "gzip, deflate, br, zstd"})
    assert r.status_code == 200
    assert seen["ae"] == "gzip, deflate"
    rec = wait_until(lambda: recs[-1] if recs else None, what="row", timeout_s=5)
    assert (rec["prompt_tokens"], rec["cached_tokens"], rec["completion_tokens"]) == (1007, 1000, 2)
    assert rec["finish_reason"] == "stop"


def test_client_without_gzip_gets_identity_from_upstream(tmp_data_dir, client):
    seen: dict = {}

    def h(req):
        seen["ae"] = req.headers.get("accept-encoding")
        return httpx.Response(200, json={"content": [], "usage": {"input_tokens": 1}})

    _setup(client, tmp_data_dir, anthropic=h)
    _spy_history(client)
    _post(client, sonnet_body(), headers={"Accept-Encoding": "br"})
    assert seen["ae"] == "identity"


def test_relayed_row_records_dead_weight_and_session_source(tmp_data_dir, client):
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    msgs = [
        {"role": "user", "content": "<system-reminder>r</system-reminder>hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm" * 20, "signature": "s"},
                {"type": "text", "text": "hi"},
            ],
        },
        {"role": "user", "content": "more"},
    ]
    _post(client, sonnet_body(msgs, stream=True))
    rec = wait_until(lambda: recs[-1] if recs else None, what="row", timeout_s=5)
    assert rec["session_source"] == "claude_code_header"
    dw = json.loads(rec["cobs2_shape"])["dw"]
    assert dw["boil"] == len("<system-reminder>r</system-reminder>") and dw["reason"] == 0  # R3


# ---- Prompt Lens (analysis plan, Task 1.3): cache writes, marks, layer 1 -----------


def _marked_body(messages=None, **extra):
    """Two ``cache_control`` marks: one on the system block, one on a message block."""
    return sonnet_body(
        messages
        or [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
        system=[{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
        **extra,
    )


def test_relay_records_cache_writes_and_marks(tmp_data_dir, client):
    raw = anthropic_sse_bytes(usage_in=3, cache_read=0, cache_create=28434)
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    r = _post(client, _marked_body(stream=True))
    assert r.status_code == 200 and r.content == raw
    rec = wait_until(lambda: recs[-1] if recs else None, what="relayed row", timeout_s=5)
    assert (rec["cache_write_tokens"], rec["cache_marks"]) == (28434, 2)
    # Written tokens still count in the prompt, as before.
    assert rec["prompt_tokens"] == 3 + 28434


def test_relay_without_marks_counts_zero_marks(tmp_data_dir, client):
    raw = anthropic_sse_bytes()
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    _post(client, sonnet_body(stream=True))
    rec = wait_until(lambda: recs[-1] if recs else None, what="relayed row", timeout_s=5)
    assert (rec["cache_write_tokens"], rec["cache_marks"]) == (40, 0)


def test_relay_miss_without_marks_is_provider_no_marks(tmp_data_dir, client):
    """A stable relayed prompt that reused nothing and carries no marks: the
    provider's rule explains the miss (spec 6), not the prompt."""
    raw = anthropic_sse_bytes(text="answer one", usage_in=5000, cache_read=0, cache_create=0)
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    m1 = [{"role": "user", "content": "hello"}]
    _post(client, sonnet_body(m1, stream=True))
    wait_until(lambda: recs[0] if recs else None, what="turn 1", timeout_s=5)
    m2 = [
        *m1,
        {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
        {"role": "user", "content": "more"},
    ]
    _post(client, sonnet_body(m2, stream=True))
    t2 = wait_until(lambda: recs[1] if len(recs) > 1 else None, what="turn 2", timeout_s=5)
    assert t2["cobs2_edge"] == "continue"
    assert (t2["lens_cause"], t2["lens_break"], t2["cache_marks"]) == (
        "provider_no_marks",
        None,
        0,
    )
    assert t2["lens_cf_tokens"] > 1024 and t2["lens_lost_tokens"] == t2["lens_cf_tokens"]


def test_relay_renders_tools_first(tmp_data_dir, client):
    """On a relay to Anthropic the tool list is rendered before the system
    prompt: a changed tool list breaks first, the system change is counted."""
    raw = anthropic_sse_bytes(text="answer one")
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    recs = _spy_history(client)
    tool = {"name": "read", "input_schema": {"type": "object"}}
    m1 = [{"role": "user", "content": "hello"}]
    _post(client, sonnet_body(m1, stream=True, tools=[tool]))
    wait_until(lambda: recs[0] if recs else None, what="turn 1", timeout_s=5)
    m2 = [
        *m1,
        {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
        {"role": "user", "content": "more"},
    ]
    tool2 = {"name": "write", "input_schema": {"type": "object"}}
    _post(client, sonnet_body(m2, stream=True, tools=[tool, tool2], system="be terse"))
    t2 = wait_until(lambda: recs[1] if len(recs) > 1 else None, what="turn 2", timeout_s=5)
    assert (t2["lens_break"], t2["lens_cause"], t2["lens_other_breaks"]) == (
        "tools",
        "tools_added_end",
        1,
    )


def test_relay_with_kvsig_off_counts_no_marks_and_no_lens(tmp_data_dir, client, monkeypatch):
    raw = anthropic_sse_bytes(cache_create=12)
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    client.app.state.cobs2_key = None
    calls = {"n": 0}
    from app.lens import causes, hot

    def count(*a, **k):
        calls["n"] += 1
        raise AssertionError("no lens call with kvsig off")

    monkeypatch.setattr(causes, "evaluate", count)
    monkeypatch.setattr(hot, "cache_marks_of", count)
    recs = _spy_history(client)
    r = _post(client, _marked_body(stream=True))
    assert r.content == raw
    rec = wait_until(lambda: recs[-1] if recs else None, what="relayed row", timeout_s=5)
    assert calls["n"] == 0
    assert (rec["cache_write_tokens"], rec["cache_marks"], rec["lens_cause"]) == (12, None, None)


def test_isolated_key_relay_is_never_provider_no_marks(tmp_data_dir, client):
    """A-11: for an isolated key the warden strips cache_control itself, so the
    same stable miss without marks names no provider cause against the prompt."""
    import sqlite3

    raw = anthropic_sse_bytes(text="answer one", usage_in=5000, cache_read=0, cache_create=0)
    _setup(client, tmp_data_dir, anthropic=_sse_handler(raw))
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE api_tokens SET kv_isolation='isolated' WHERE id='tok1'")
        db.commit()
    from app.proxy import token_cache

    token_cache.invalidate_all()
    recs = _spy_history(client)
    m1 = [{"role": "user", "content": "hello"}]
    _post(client, _marked_body(m1, stream=True))
    wait_until(lambda: recs[0] if recs else None, what="turn 1", timeout_s=5)
    m2 = [
        *m1,
        {"role": "assistant", "content": [{"type": "text", "text": "answer one"}]},
        {"role": "user", "content": "more"},
    ]
    _post(client, _marked_body(m2, stream=True))
    t2 = wait_until(lambda: recs[1] if len(recs) > 1 else None, what="turn 2", timeout_s=5)
    assert t2["cobs2_edge"] == "continue"
    # The marks left stripped (0 sent), and the lens names no provider cause.
    assert t2["cache_marks"] == 0
    assert t2["lens_cause"] is None
    assert t2["lens_cf_tokens"] > 1024
