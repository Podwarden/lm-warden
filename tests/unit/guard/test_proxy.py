"""The guard on the request path (spec phase 1): the sanitize / block decision, the stream and
JSON restorers, the trailer, the block reply and fail-closed. Pure: no app, no engine."""

from __future__ import annotations

import json
import random

import pytest

from app.guard.config import compile_guard
from app.guard.placeholders import placeholder_hex, render
from app.guard.proxy import (
    ChatStreamRestorer,
    GuardFailed,
    SeenSet,
    block_reply,
    guard_for,
    restore_json,
    run_guard,
)
from tests.unit.guard.corpus import github_token

SECRET = b"k" * 32
TOKEN = "tok1"


def _g(**kw):
    return compile_guard({"mode": "sanitize", "packs": ["cloud-api-keys"], **kw})


def _chat(*msgs, **extra):
    return {"model": "m", "messages": [{"role": r, "content": c} for r, c in msgs], **extra}


class _Tok:
    def __init__(self, guard=None):
        self.id = TOKEN
        self.guard = guard


# --- settings --------------------------------------------------------------------------------


def test_no_platform_and_no_key_setting_means_no_guard():
    assert guard_for(None, _Tok()) is None


def test_key_off_wins_over_the_platform():
    assert guard_for({"mode": "block"}, _Tok(json.dumps({"off": True}))) is None


def test_key_override_turns_the_guard_on():
    g = guard_for(None, _Tok(json.dumps({"override": {"mode": "block"}})))
    assert g is not None and g.mode == "block"


# --- the decision ----------------------------------------------------------------------------


def test_sanitize_replaces_the_value_notes_it_and_never_carries_it_in_the_report():
    value = github_token()
    body = _chat(("system", "be brief"), ("user", f"my token is {value} ok"))
    out = run_guard(
        body, "openai", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=SeenSet()
    )
    assert out.action == "sanitize"
    text = json.dumps(out.body)
    assert value not in text
    assert out.restore.restore(text)[0].count(value) == 1
    assert "[LM Warden]" in out.body["messages"][-1]["content"]
    assert value not in (out.trailer or "") and value not in out.history_json
    assert out.trailer and "github" in out.trailer
    assert body["messages"][1]["content"].endswith(" ok"), "the client's body is not mutated"


def test_nothing_found_changes_nothing():
    body = _chat(("user", "hello"))
    out = run_guard(
        body, "openai", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=SeenSet()
    )
    assert out.action == "pass" and out.body is body and not out.restore and out.trailer is None


def test_block_once_then_sanitize_in_the_same_session():
    value = github_token()
    body = _chat(("user", f"look {value}"))
    seen = SeenSet()
    g = _g(mode="block")
    first = run_guard(body, "openai", g, token_id=TOKEN, secret=SECRET, session="s", seen=seen)
    assert first.action == "block"
    body2 = _chat(("user", f"look {value}"), ("assistant", "no"), ("user", "go on"))
    second = run_guard(body2, "openai", g, token_id=TOKEN, secret=SECRET, session="s", seen=seen)
    assert second.action == "sanitize"
    other = run_guard(body, "openai", g, token_id=TOKEN, secret=SECRET, session="t", seen=seen)
    assert other.action == "block", "a different session blocks again"


def test_a_value_in_the_system_prompt_never_blocks():
    value = github_token()
    body = _chat(("system", f"deploy with {value}"), ("user", "hi"))
    out = run_guard(
        body, "openai", _g(mode="block"), token_id=TOKEN, secret=SECRET, session="s", seen=SeenSet()
    )
    assert out.action == "sanitize"


def test_trailer_new_only_is_quiet_when_nothing_is_new():
    value = github_token()
    seen = SeenSet()
    body = _chat(("user", f"x {value}"))
    run_guard(body, "openai", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=seen)
    again = run_guard(body, "openai", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=seen)
    assert again.action == "sanitize" and again.trailer is None
    always = run_guard(
        body, "openai", _g(trailer="always"), token_id=TOKEN, secret=SECRET, session="s", seen=seen
    )
    assert always.trailer is not None


def test_json_mode_and_echo_suppress_the_trailer():
    value = github_token()
    for extra in ({"response_format": {"type": "json_object"}},):
        body = _chat(("user", f"x {value}"), **extra)
        out = run_guard(
            body, "openai", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=SeenSet()
        )
        assert out.trailer is None
    body = {"model": "m", "prompt": f"x {value}", "echo": True}
    out = run_guard(
        body, "openai", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=SeenSet()
    )
    assert out.action == "sanitize" and out.trailer is None


def test_too_big_a_body_fails_closed_or_open_as_set():
    body = _chat(("user", "a" * 64))
    with pytest.raises(GuardFailed):
        run_guard(
            body,
            "openai",
            _g(),
            token_id=TOKEN,
            secret=SECRET,
            session="s",
            seen=SeenSet(),
            max_scan_bytes=10,
        )
    out = run_guard(
        body,
        "openai",
        _g(fail="open"),
        token_id=TOKEN,
        secret=SECRET,
        session="s",
        seen=SeenSet(),
        max_scan_bytes=10,
    )
    assert out.action == "pass"


def test_events_carry_no_value():
    value = github_token()
    out = run_guard(
        _chat(("user", f"x {value}")),
        "openai",
        _g(),
        token_id=TOKEN,
        secret=SECRET,
        session="s",
        seen=SeenSet(),
    )
    assert len(out.events) == 1
    ev = out.events[0]
    assert ev.action == "sanitized" and ev.length == len(value) and value not in repr(ev)


# --- restoring -------------------------------------------------------------------------------


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _delta(content=None, finish=None, args=None):
    d = {}
    if content is not None:
        d["content"] = content
    if args is not None:
        d["tool_calls"] = [{"index": 0, "function": {"arguments": args}}]
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
    }


def _client_text(raw: bytes) -> tuple[str, str, list[str]]:
    content, args, order = "", "", []
    for ev in raw.split(b"\n\n"):
        if not ev.startswith(b"data: ") or ev == b"data: [DONE]":
            continue
        obj = json.loads(ev[6:])
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content += d["content"]
                order.append("content")
            for tc in d.get("tool_calls") or []:
                args += tc["function"].get("arguments") or ""
            if ch.get("finish_reason"):
                order.append("finish")
    return content, args, order


@pytest.mark.parametrize("seed", range(40))
def test_stream_restores_across_any_chunking_and_puts_the_trailer_before_finish(seed):
    rnd = random.Random(seed)
    value, hx = "ghp_SECRETVALUE", "0123456789"
    from app.guard.placeholders import RestoreMap

    forms = [render(hx), f"[[LMW-SECRET-{hx}]]", f"LMW-SECRET-{hx}", f"lmw-secret-{hx.upper()}"]
    form = rnd.choice(forms)
    text = f"use {form} now and {form}."
    frames = b""
    i = 0
    while i < len(text):
        n = rnd.randint(1, 5)
        frames += _sse(_delta(text[i : i + n]))
        i += n
    frames += _sse(_delta(args='{"k": "' + form + '"}'))
    frames += _sse(_delta(finish="stop")) + b"data: [DONE]\n\n"
    r = ChatStreamRestorer(RestoreMap({hx: value}), trailer="TRAILER", restore_mode="all")
    out = b""
    j = 0
    while j < len(frames):
        n = rnd.randint(1, 40)
        out += r.feed(frames[j : j + n])
        j += n
    out += r.flush()
    content, args, order = _client_text(out)
    assert content == f"use {value} now and {value}.TRAILER".replace(".TRAILER", ".\n\nTRAILER")
    assert args == '{"k": "' + value + '"}'
    assert order[-1] == "finish" and order.index("finish") > max(
        i for i, o in enumerate(order) if o == "content"
    )
    assert r.restored == 3
    assert out.rstrip().endswith(b"data: [DONE]")


def test_stream_with_text_only_leaves_tool_arguments():
    from app.guard.placeholders import RestoreMap

    hx = "0123456789"
    r = ChatStreamRestorer(RestoreMap({hx: "V"}), trailer=None, restore_mode="text_only")
    out = r.feed(_sse(_delta(content=render(hx))) + _sse(_delta(args=render(hx)))) + r.flush()
    content, args, _ = _client_text(out)
    assert content == "V" and args == render(hx)


def test_stream_without_a_finish_chunk_flushes_what_it_held():
    from app.guard.placeholders import RestoreMap

    r = ChatStreamRestorer(RestoreMap({"0123456789": "V"}), trailer="T", restore_mode="all")
    out = r.feed(_sse(_delta(content="end ⟦LMW-SEC"))) + r.flush()
    content, _, _ = _client_text(out)
    assert content == "end ⟦LMW-SEC", "no trailer without a finish chunk; held text is not lost"


def test_restore_json_chat_and_legacy():
    from app.guard.placeholders import RestoreMap

    m = RestoreMap({"0123456789": "V"})
    body = {
        "choices": [
            {
                "message": {
                    "content": f"a {render('0123456789')}",
                    "tool_calls": [{"function": {"arguments": render("0123456789")}}],
                }
            },
            {"text": f"b {render('0123456789')}"},
        ]
    }
    raw, n = restore_json(json.dumps(body).encode(), m, trailer="T", restore_mode="all")
    out = json.loads(raw)
    assert out["choices"][0]["message"]["content"] == "a V\n\nT"
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "V"
    assert out["choices"][1]["text"] == "b V\n\nT"
    assert n == 3


# --- the block reply -------------------------------------------------------------------------


def test_block_reply_json_and_stream_say_what_was_found_masked():
    value = github_token()
    out = run_guard(
        _chat(("user", f"x {value}")),
        "openai",
        _g(mode="block"),
        token_id=TOKEN,
        secret=SECRET,
        session="s",
        seen=SeenSet(),
    )
    resp = block_reply(out, stream=False, path="/v1/chat/completions")
    assert resp.status_code == 200 and resp.headers["x-lmwarden-guard"].startswith("blocked;n=1")
    msg = json.loads(resp.body)
    text = msg["choices"][0]["message"]["content"]
    assert msg["model"] == "lm-warden-guard" and "github" in text and value not in text
    sresp = block_reply(out, stream=True, path="/v1/chat/completions")
    assert sresp.media_type == "text/event-stream"
    legacy = json.loads(block_reply(out, stream=False, path="/v1/completions").body)
    assert "github" in legacy["choices"][0]["text"]


def test_a_value_with_quotes_restores_into_valid_tool_arguments():
    from app.guard.anthropic import AnthropicStreamRewriter, restore_message_json
    from app.guard.placeholders import RestoreMap

    value = 'pa"ss\\word'
    m = RestoreMap({"0123456789": value})
    args = '{"k": "' + render("0123456789") + '"}'
    r = ChatStreamRestorer(m, trailer=None, restore_mode="all")
    out = r.feed(_sse(_delta(args=args))) + r.flush()
    _, restored, _ = _client_text(out)
    assert json.loads(restored) == {"k": value}
    raw, _ = restore_json(
        json.dumps(
            {
                "choices": [
                    {"message": {"content": "", "tool_calls": [{"function": {"arguments": args}}]}}
                ]
            }
        ).encode(),
        m,
        trailer=None,
    )
    assert json.loads(
        json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    ) == {"k": value}
    a = AnthropicStreamRewriter(m, trailer=None)
    ev = lambda e, d: f"event: {e}\ndata: {json.dumps(d)}\n\n".encode()  # noqa: E731
    out = (
        a.feed(
            ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "input": {}},
                },
            )
            + ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                },
            )
            + ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        )
        + a.flush()
    )
    pj = "".join(
        json.loads(f.split(b"data: ", 1)[1])["delta"]["partial_json"]
        for f in out.split(b"\n\n")
        if b"input_json_delta" in f
    )
    assert json.loads(pj) == {"k": value}
    body = {"content": [{"type": "tool_use", "input": {"k": render("0123456789")}}]}
    raw, _ = restore_message_json(json.dumps(body).encode(), m, trailer=None)
    assert json.loads(raw)["content"][0]["input"] == {"k": value}


def test_a_value_in_a_signed_thinking_block_blocks_even_in_sanitize_mode():
    value = github_token()
    body = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": f"the key is {value}", "signature": "sig"},
                    {"type": "text", "text": "ok"},
                ],
            },
            {"role": "user", "content": "go on"},
        ],
    }
    out = run_guard(
        body, "anthropic", _g(), token_id=TOKEN, secret=SECRET, session="s", seen=SeenSet()
    )
    assert out.action == "block"
    assert any(h.where.kind == "thinking" for h in out.findings)
    assert body["messages"][1]["content"][0]["thinking"].endswith(value), "never rewritten"


def test_the_trailer_never_lands_inside_the_answer_when_the_finish_chunk_has_text():
    from app.guard.placeholders import RestoreMap

    r = ChatStreamRestorer(RestoreMap({"0123456789": "V"}), trailer="TRAILER", restore_mode="all")
    out = (
        r.feed(_sse(_delta(content="curl https://api.github.com")))
        + r.feed(_sse(_delta(content="/user", finish="stop")))
        + r.flush()
    )
    content, _, order = _client_text(out)
    assert content == "curl https://api.github.com/user\n\nTRAILER"
    assert order[-1] == "finish"


def _gen():
    return compile_guard({"mode": "sanitize", "packs": ["generic-credentials", "cloud-api-keys"]})


def test_a_value_found_once_is_caught_again_without_its_context_in_the_same_session():
    pw = "Zx9qLm2vB7nR4tKw"
    seen = SeenSet()
    first = _chat(("user", f"my .env:\nDB_PASSWORD={pw}\n"))
    one = run_guard(first, "openai", _gen(), token_id=TOKEN, secret=SECRET, session="s", seen=seen)
    assert one.action == "sanitize" and pw not in json.dumps(one.body)
    # Later: the .env is gone (compaction), the value is in prose and in a restored answer.
    later = _chat(
        ("assistant", f"The database password is {pw}."),
        ("user", f"use {pw} with psql"),
    )
    two = run_guard(later, "openai", _gen(), token_id=TOKEN, secret=SECRET, session="s", seen=seen)
    assert two.action == "sanitize"
    text = json.dumps(two.body, ensure_ascii=False)
    assert pw not in text
    ph = render(placeholder_hex(SECRET, TOKEN, pw))
    assert text.count(ph) == 2
    assert all(h.rule_id == "password-assignment" for h in two.findings)


def test_another_session_does_not_inherit_seen_values():
    pw = "Zx9qLm2vB7nR4tKw"
    seen = SeenSet()
    run_guard(
        _chat(("user", f"DB_PASSWORD={pw}")),
        "openai",
        _gen(),
        token_id=TOKEN,
        secret=SECRET,
        session="s",
        seen=seen,
    )
    other = run_guard(
        _chat(("user", f"the password is {pw}")),
        "openai",
        _gen(),
        token_id=TOKEN,
        secret=SECRET,
        session="t",
        seen=seen,
    )
    assert other.action == "pass"


def test_the_seen_state_holds_no_value():
    pw = "Zx9qLm2vB7nR4tKw"
    seen = SeenSet()
    run_guard(
        _chat(("user", f"DB_PASSWORD={pw}")),
        "openai",
        _gen(),
        token_id=TOKEN,
        secret=SECRET,
        session="s",
        seen=seen,
    )
    assert pw not in repr(seen.get((TOKEN, "s"))) and pw not in repr(seen._d)
