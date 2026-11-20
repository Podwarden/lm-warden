"""The guard wired into _forward (spec phase 1): the engine never sees a value, the client gets
it back, the block reply holds no slot, the log and history hold no value, and a key without a
guard forwards byte-identically."""

from __future__ import annotations

import json
import sqlite3

from tests.conftest import wait_until
from tests.unit.guard.corpus import github_token
from tests.unit.proxy.test_messages import _json_upstream, _ready, _sse_upstream
from tests.unit.proxy.test_thinking_policy import _go, _sent

VALUE = github_token()


def _set_guard(db_path, guard: dict | None) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET guard = ?", (json.dumps(guard) if guard else None,))
        db.commit()


def _chat(stream=False):
    return {
        "model": "qwen",
        "stream": stream,
        "messages": [{"role": "user", "content": f"deploy with {VALUE} please"}],
    }


def _placeholder(sent: dict) -> str:
    text = sent["messages"][-1]["content"]
    i = text.index("⟦LMW-SECRET-")
    return text[i : i + len("⟦LMW-SECRET-") + 11]


SANITIZE = {"override": {"mode": "sanitize", "packs": ["cloud-api-keys"], "trailer": "always"}}
BLOCK = {"override": {"mode": "block", "packs": ["cloud-api-keys"]}}


def test_no_guard_forwards_as_before(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, send = _go(client, _chat())
    assert r.status_code == 200 and VALUE in _sent(send).decode()


def test_sanitize_hides_the_value_from_the_engine_and_restores_it_in_the_json_answer(
    tmp_data_dir, client
):
    _ready(client, tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    _set_guard(db, SANITIZE)
    # the engine echoes the placeholder it was given
    r1, send1 = _go(client, _chat())
    sent = json.loads(_sent(send1))
    assert VALUE not in json.dumps(sent)
    ph = _placeholder(sent)
    upstream = _json_upstream(
        {
            "id": "x",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"export TOKEN={ph}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    r, _ = _go(client, _chat(), upstream=upstream)
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content.startswith(f"export TOKEN={VALUE}")
    assert "[LM Warden]" in content and VALUE not in content.split("\n\n", 1)[1]

    # The action log is written off the request path, one background write per log call: the
    # first request's `sanitized` and the second's `restored` race, so neither the row order nor
    # "some event is there" says the `sanitized` one landed. Wait for both, pick by action.
    def _rows():
        with sqlite3.connect(db) as c:
            rows = c.execute("SELECT guard FROM request_history WHERE guard IS NOT NULL").fetchall()
            ev = c.execute("SELECT action, rule_id, length FROM guard_events").fetchall()
        return (rows, ev) if rows and {e[0] for e in ev} >= {"sanitized", "restored"} else None

    rows, ev = wait_until(_rows, what="guard history and sanitized + restored events")
    dump = json.dumps(rows) + json.dumps(ev)
    assert VALUE not in dump
    assert [e[2] for e in ev if e[0] == "sanitized"] == [len(VALUE)]


def test_sanitize_restores_a_stream_and_puts_the_trailer_before_finish(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    _set_guard(db, SANITIZE)
    _, send = _go(client, _chat())
    ph = _placeholder(json.loads(_sent(send)))
    half = len(ph) // 2
    events = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "use " + ph[:half]}, "finish_reason": None}
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": ph[half:] + " ok"}, "finish_reason": None}
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    r, _ = _go(client, _chat(stream=True), upstream=_sse_upstream(events))
    assert r.status_code == 200
    text, finished_after = "", False
    for ev in r.content.split(b"\n\n"):
        if not ev.startswith(b"data: {"):
            continue
        ch = json.loads(ev[6:])["choices"][0]
        if ch.get("finish_reason"):
            finished_after = True
        else:
            assert not finished_after, "content after the finish chunk"
            text += ch["delta"].get("content") or ""
    assert text.startswith(f"use {VALUE} ok\n\n[LM Warden]")


def test_block_answers_from_lm_warden_without_calling_the_engine(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", BLOCK)
    r, send = _go(client, _chat())
    assert r.status_code == 200 and send.call_count == 0
    assert r.headers["x-lmwarden-guard"].startswith("blocked")
    body = r.json()
    assert body["model"] == "lm-warden-guard" and VALUE not in json.dumps(body)
    # the same conversation again: sanitized, the engine answers
    r2, send2 = _go(client, _chat())
    assert r2.status_code == 200 and send2.call_count == 1
    assert VALUE not in _sent(send2).decode()


def test_a_guard_fault_fails_closed(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", SANITIZE)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.guard.proxy.inspect_body", boom)
    r, send = _go(client, _chat())
    assert r.status_code == 422 and send.call_count == 0
    assert r.headers["x-lmwarden-guard"] == "failed;reason=scanner_error"


def test_translated_messages_path_is_guarded_and_restored(tmp_data_dir, client):
    from tests.unit.proxy.test_messages import _post

    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", SANITIZE)
    # learn the placeholder this key gives the value
    _, send = _go(client, _chat())
    ph = _placeholder(json.loads(_sent(send)))
    upstream = _json_upstream(
        {
            "id": "x",
            "object": "chat.completion",
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"set {ph}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    r, send = _post(
        client,
        {
            "model": "qwen",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": f"use {VALUE}"}],
        },
        upstream=upstream,
    )
    assert r.status_code == 200
    assert VALUE not in send.call_args.args[0].content.decode()
    text = r.json()["content"][0]["text"]
    assert text.startswith(f"set {VALUE}") and "[LM Warden]" in text


def test_a_blocked_request_gets_a_history_row_without_a_slot(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    _set_guard(db, BLOCK)
    r, send = _go(client, _chat())
    assert r.status_code == 200 and send.call_count == 0

    def _row():
        with sqlite3.connect(db) as c:
            return c.execute("SELECT guard, completion_tokens FROM request_history").fetchone()

    guard, completion = wait_until(_row, what="history row for the block")
    g = json.loads(guard)
    assert g["b"] == 1 and g["f"][0][0].startswith("github") and VALUE not in guard
    assert not completion


def test_a_list_prompt_on_completions_is_guarded(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", SANITIZE)
    body = {
        "model": "qwen",
        "prompt": [f"GITHUB_TOKEN={VALUE}\n# repeat it: ", "second"],
        "max_tokens": 5,
    }
    r, send = _go(client, body, path="/v1/completions")
    assert r.status_code == 200
    sent = json.loads(_sent(send))
    assert VALUE not in json.dumps(sent) and "⟦LMW-SECRET-" in sent["prompt"][0]


def test_a_token_id_prompt_cannot_be_checked_and_fails_closed(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", SANITIZE)
    r, send = _go(
        client, {"model": "qwen", "prompt": [1, 2, 3], "max_tokens": 5}, path="/v1/completions"
    )
    assert r.status_code == 422 and send.call_count == 0
    assert r.headers["x-lmwarden-guard"] == "failed;reason=unscannable"


def test_a_block_reply_names_its_history_row(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", BLOCK)
    r, _ = _go(client, _chat())
    assert r.headers["x-lmwarden-guard"].startswith("blocked") and r.headers.get("x-lmw-request-id")


def test_a_block_on_translated_messages_is_recognisable(tmp_data_dir, client):
    from tests.unit.proxy.test_messages import _post

    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", BLOCK)
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": f"use {VALUE}"}],
    }
    r, send = _post(client, body, upstream=None)
    assert send.call_count == 0
    assert r.headers["x-lmwarden-guard"].startswith("blocked")
    msg = r.json()
    assert msg["model"] == "lm-warden-guard" and msg["type"] == "message"
    assert VALUE not in json.dumps(msg)


def test_a_block_on_translated_responses_is_recognisable(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _set_guard(tmp_data_dir / "vllm-warden.db", BLOCK)
    body = {"model": "qwen", "input": f"use {VALUE}"}
    r, send = _go(client, body, path="/v1/responses")
    assert send.call_count == 0
    assert r.headers["x-lmwarden-guard"].startswith("blocked")
    out = r.json()
    assert out["model"] == "lm-warden-guard" and VALUE not in json.dumps(out)


def test_a_non_streamed_answer_logs_its_restored_values(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    _set_guard(db, SANITIZE)
    _, send = _go(client, _chat())
    ph = _placeholder(json.loads(_sent(send)))
    upstream = _json_upstream(
        {
            "id": "x",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"curl -H 'Authorization: {ph}'"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    r, _ = _go(client, _chat(), upstream=upstream)
    assert VALUE in r.json()["choices"][0]["message"]["content"]

    def _restored():
        with sqlite3.connect(db) as c:
            return (
                c.execute(
                    "SELECT rule_id, length FROM guard_events WHERE action='restored'"
                ).fetchall()
                or None
            )

    rows = wait_until(_restored, what="restored event")
    assert rows[0][1] == len(VALUE)
