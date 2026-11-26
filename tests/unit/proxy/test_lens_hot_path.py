"""Prompt Lens layer 1 on the proxy hot path (analysis plan, Task 1.3).

The cause is computed in the kvsig pass, stored text-free in
``request_history`` and carried on the kvsig event's ``lens`` block. With
kvsig off (no network key) the lens makes zero calls; a fault anywhere in it
leaves the proxied response byte-identical and drops only the lens fields.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from app.proxy.prompt_lens_scope import for_key_holder
from tests.conftest import wait_until
from tests.unit.proxy.test_prompt_lens_forward import _json_resp, _kvsigs, _ready_gm

SESSION = {"X-Session-Id": "5f0c2a9e-7d41-4b8e-9c3a-1e2f3a4b5c6d"}
CANARY = "CANARY-7f3e9b21-lens"
BODY = {"model": "qwen", "messages": [{"role": "user", "content": "hi there"}]}
LENS_COLS = (
    "lens_break",
    "lens_cause",
    "lens_span_class",
    "lens_span_line",
    "lens_lost_tokens",
    "lens_cf_tokens",
    "lens_other_breaks",
    "cache_write_tokens",
    "cache_marks",
)


def _reply(text="ok here", cached=0):
    usage = {"prompt_tokens": 40, "completion_tokens": 2}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return _json_resp(
        {
            "id": "c1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    )


def _post(client, plaintext, body, headers=None, reply="ok here", cached=0):
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_reply(reply, cached))):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {plaintext}", **(headers or {})},
            json=body,
        )
    assert r.status_code == 200, r.text
    return r


def _rows(tmp_data_dir, n):
    def _get():
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM request_history ORDER BY finished_at").fetchall()
        return [dict(r) for r in rows] if len(rows) >= n else None

    return wait_until(_get, what=f"{n} request_history rows")


def _owner(target):
    """(object, attribute) for ``pkg.mod.name`` or ``pkg.mod.Class.name``."""
    import importlib

    mod, name = target.rsplit(".", 1)
    try:
        return importlib.import_module(mod), name
    except ModuleNotFoundError:
        mod, cls = mod.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls), name


def _spy(monkeypatch, target):
    """Count calls through ``target`` (a plain function, so a method still binds)."""
    owner, name = _owner(target)
    orig = getattr(owner, name)
    seen = {"n": 0}

    def spy(*a, **k):
        seen["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(owner, name, spy)
    return type("Calls", (), {"count": property(lambda self: seen["n"])})()


def _boom(*a, **k):
    raise RuntimeError("lens exploded")


def _sys(*lines):
    return {"role": "system", "content": "\n".join(lines)}


def _two_turns(client, plaintext, sys1, sys2, headers=SESSION):
    """Turn 1, then turn 2 continuing it (the reply carried) with ``sys2``."""
    u = {"role": "user", "content": "list the files"}
    _post(client, plaintext, {"model": "qwen", "messages": [sys1, u]}, headers)
    t2 = [
        sys2,
        u,
        {"role": "assistant", "content": "ok here"},
        {"role": "user", "content": "and now"},
    ]
    _post(client, plaintext, {"model": "qwen", "messages": t2}, headers)


# ---- fail-open ------------------------------------------------------------------


def test_kvsig_off_makes_zero_lens_calls(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    client.app.state.cobs2_key = None
    calls = _spy(monkeypatch, "app.lens.causes.evaluate")
    probes = _spy(monkeypatch, "app.lens.probes.message_probes")
    digests = _spy(monkeypatch, "app.lens.probes.sys_line_digests")
    remember = _spy(monkeypatch, "app.lens.heads.RecentHeads.remember")
    _post(client, plaintext, BODY, SESSION)
    _post(client, plaintext, BODY, SESSION)
    assert (calls.count, probes.count, digests.count, remember.count) == (0, 0, 0, 0)
    (row, _) = _rows(tmp_data_dir, 2)
    assert all(row[c] is None for c in LENS_COLS)


@pytest.fixture(autouse=True)
def _cold_memos():
    """Every test starts with empty lens memos (they are process-wide)."""
    from app.lens import hot

    hot.clear_memos()
    yield
    hot.clear_memos()


_FAULT_TARGETS = [
    "app.lens.causes.evaluate",
    "app.lens.probes.message_probes",
    "app.lens.probes.sys_line_digests",
    "app.lens.probes.tool_def_hashes",
    "app.lens.heads.RecentHeads.remember",
    "app.lens.heads.RecentHeads.lookup",
    "app.lens.heads.RecentHeads.lookup_by_tools",
    # Task 1.4: the reply verdicts and the schema lint.
    "app.lens.reply_verdicts.call_verdicts",
    "app.lens.reply_verdicts.reply_counts",
    "app.lens.schema_lint.lint_tool",
]


def _fresh_body(tag):
    """A body no earlier request of the test sent: a new system line, a new
    tool and a new message, so every lens step misses its memo."""
    return {
        "model": "qwen",
        "tools": [{"type": "function", "function": {"name": f"tool_{tag}"}}],
        "messages": [
            _sys("You help.", f"Working directory: /w/{tag}"),
            {"role": "user", "content": f"first message {tag}"},
        ],
    }


@pytest.mark.parametrize("target", _FAULT_TARGETS)
def test_fault_in_lens_leaves_response_byte_identical(tmp_data_dir, client, monkeypatch, target):
    plaintext = _ready_gm(client, tmp_data_dir)
    # Warm the key's recent heads with another head of the same tools, so the
    # faulted turn (a new session) also reaches both head lookups.
    warm = _fresh_body("warm")
    warm["tools"] = _fresh_body("faulted")["tools"]
    _post(client, plaintext, warm, {"X-Session-Id": "33333333-3333-4333-8333-333333333333"})
    # The reference: the faulted request's own body with the lens out of the
    # path (kvsig off: no lens call at all).
    key = client.app.state.cobs2_key
    client.app.state.cobs2_key = None
    ref = _post(
        client,
        plaintext,
        _fresh_body("faulted"),
        {"X-Session-Id": "4" * 8 + "-4444-4444-8444-" + "4" * 12},
    )
    client.app.state.cobs2_key = key
    from app.lens import hot

    hot.clear_memos()

    owner, name = _owner(target)
    calls = {"n": 0}

    def raising(*a, **k):
        calls["n"] += 1
        raise RuntimeError("lens exploded")

    monkeypatch.setattr(owner, name, raising)
    again = _post(
        client,
        plaintext,
        _fresh_body("faulted"),
        {"X-Session-Id": "5" * 8 + "-5555-4555-8555-" + "5" * 12},
    )
    assert calls["n"] >= 1, f"{target} was never reached"
    assert again.content == ref.content
    assert again.headers["content-type"] == ref.headers["content-type"]
    rows = _rows(tmp_data_dir, 3)
    # The kvsig ids still land: only the lens fields may drop.
    assert rows[2]["cobs2_head"] is not None and rows[2]["cobs2_edge"] == "new"


def test_fault_in_evaluate_drops_only_the_lens_fields(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    monkeypatch.setattr("app.lens.causes.evaluate", _boom)
    _two_turns(client, plaintext, _sys("You help.", "dir /w/1"), _sys("You help.", "dir /w/2"))
    _, row = _rows(tmp_data_dir, 2)
    assert all(row[c] is None for c in LENS_COLS)
    assert row["cobs2_edge"] == "continue"


# ---- the stored cause ---------------------------------------------------------------


def test_second_turn_with_changed_path_line_stores_cause(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _two_turns(
        client,
        plaintext,
        _sys("You help.", "Working directory: /w/1"),
        _sys("You help.", "Working directory: /w/2"),
    )
    first, row = _rows(tmp_data_dir, 2)
    assert (row["lens_cause"], row["lens_span_class"], row["lens_span_line"]) == (
        "sys_changed",
        "path",
        2,
    )
    assert row["lens_break"] == "sys"
    assert row["lens_other_breaks"] == 0
    # Sys first: the change sits in the first part, so nothing is reusable
    # after it; the counterfactual is the head and the parent's message.
    assert row["lens_cf_tokens"] > 0
    # The engine reused nothing (cached 0): every counterfactual token is lost.
    assert row["lens_lost_tokens"] == row["lens_cf_tokens"]
    # Turn 1 is a new conversation with no earlier head of this key.
    assert first["lens_cause"] == "new_conversation" and first["lens_break"] is None


def test_line_ending_only_change_locates_no_line(tmp_data_dir, client):
    """M1: a CRLF/LF flip of the system prompt is a break, but no line moved:
    span line and class stay null."""
    plaintext = _ready_gm(client, tmp_data_dir)
    _two_turns(
        client,
        plaintext,
        {"role": "system", "content": "You help.\r\nWorking directory: /w/1\r\nBe brief."},
        {"role": "system", "content": "You help.\nWorking directory: /w/1\nBe brief."},
    )
    _, row = _rows(tmp_data_dir, 2)
    assert row["lens_break"] == "sys" and row["lens_cause"] in ("sys_noise", "sys_changed")
    assert (row["lens_span_line"], row["lens_span_class"]) == (None, None)


def test_lost_tokens_need_the_engines_reuse_figure(tmp_data_dir, client):
    """No cached_tokens from the engine: the lost figure is unknown, not 0."""
    plaintext = _ready_gm(client, tmp_data_dir)
    u = {"role": "user", "content": "list the files"}
    s1, s2 = _sys("You help.", "dir /w/1"), _sys("You help.", "dir /w/2")
    _post(client, plaintext, {"model": "qwen", "messages": [s1, u]}, SESSION, cached=None)
    t2 = [s2, u, {"role": "assistant", "content": "ok here"}, {"role": "user", "content": "n"}]
    _post(client, plaintext, {"model": "qwen", "messages": t2}, SESSION, cached=None)
    _, row = _rows(tmp_data_dir, 2)
    assert row["lens_cause"] == "sys_changed"
    assert row["lens_cf_tokens"] > 0 and row["lens_lost_tokens"] is None


def test_stable_continuation_stores_no_cause_but_the_counterfactual(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    s = _sys("You help.", "Working directory: /w/1")
    _two_turns(client, plaintext, s, s)
    _, row = _rows(tmp_data_dir, 2)
    assert row["lens_cause"] is None and row["lens_break"] is None
    assert row["lens_cf_tokens"] > 0 and row["lens_other_breaks"] == 0


def test_new_session_with_a_new_date_line_is_sys_noise_against_the_recent_head(
    tmp_data_dir, client
):
    """Cross-session (lens spec 4.6): the key's recent head with the same
    loose system prompt and tools; RecentHeads is fed on every turn."""
    plaintext = _ready_gm(client, tmp_data_dir)
    u = {"role": "user", "content": "start"}
    _post(
        client,
        plaintext,
        {"model": "qwen", "messages": [_sys("You help.", "2026-11-22"), u]},
        {"X-Session-Id": "11111111-1111-4111-8111-111111111111"},
    )
    _post(
        client,
        plaintext,
        {"model": "qwen", "messages": [_sys("You help.", "2026-11-23"), u]},
        {"X-Session-Id": "22222222-2222-4222-8222-222222222222"},
    )
    _, row = _rows(tmp_data_dir, 2)
    assert (row["lens_cause"], row["lens_break"], row["lens_span_line"]) == (
        "sys_noise",
        "sys",
        2,
    )
    assert row["lens_span_class"] == "date"


def test_another_keys_head_is_never_the_recent_head(tmp_data_dir, client):
    from tests.unit.proxy.test_prompt_lens_forward import _second_key

    plaintext = _ready_gm(client, tmp_data_dir)
    other = _second_key(tmp_data_dir)
    u = {"role": "user", "content": "start"}
    _post(client, other, {"model": "qwen", "messages": [_sys("You help.", "2026-11-22"), u]})
    _post(client, plaintext, {"model": "qwen", "messages": [_sys("You help.", "2026-11-23"), u]})
    _, row = _rows(tmp_data_dir, 2)
    # No earlier head of THIS key: a new conversation, nothing compared.
    assert row["lens_cause"] == "new_conversation" and row["lens_break"] is None
    assert row["lens_cf_tokens"] is None


def test_multi_turn_body_without_parent_is_not_attributed(tmp_data_dir, client):
    """After a restart (no lineage): never a false new_conversation."""
    plaintext = _ready_gm(client, tmp_data_dir)
    msgs = [
        _sys("You help."),
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    _post(client, plaintext, {"model": "qwen", "messages": msgs}, SESSION)
    (row,) = _rows(tmp_data_dir, 1)
    assert row["lens_cause"] is None and row["lens_break"] is None


def test_tool_loop_body_without_parent_is_not_a_new_conversation(tmp_data_dir, client):
    """C1: an agent loop's only user message is its first one. After a restart
    or an eviction (edge new, empty lineage) its body still has prior turns."""
    plaintext = _ready_gm(client, tmp_data_dir)
    calls = [_call("ls", "{}", "c1")]
    loop = [
        _sys("You help."),
        {"role": "user", "content": "list the files"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": "c1", "content": "a b"},
        {"role": "assistant", "content": None, "tool_calls": [_call("ls", "{}", "c2")]},
        {"role": "tool", "tool_call_id": "c2", "content": "c d"},
    ]
    _post(client, plaintext, {"model": "qwen", "messages": loop}, SESSION)
    (row,) = _rows(tmp_data_dir, 1)
    assert row["cobs2_edge"] == "new"
    assert row["lens_cause"] is None and row["lens_break"] is None
    # M2: unattributed, so every lens field is null (other_breaks too).
    assert all(row[c] is None for c in LENS_COLS[:7]), {c: row[c] for c in LENS_COLS}


def test_anthropic_tool_loop_body_without_parent_is_not_a_new_conversation(tmp_data_dir, client):
    """C1 on /v1/messages: the tool_result blocks inside the user message are
    a prior turn. A first-turn body on the same route still is one."""
    plaintext = _ready_gm(client, tmp_data_dir)
    loop = {
        "model": "qwen",
        "max_tokens": 64,
        "system": "You help.",
        "messages": [
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "fine"}],
            },
        ],
    }
    _post_resp(client, plaintext, loop, _tool_reply(None, content="ok"), path="/v1/messages")
    first = {
        "model": "qwen",
        "max_tokens": 64,
        "system": "You help.",
        "messages": [{"role": "user", "content": "something else entirely"}],
    }
    _post_resp(
        client,
        plaintext,
        first,
        _tool_reply(None, content="ok"),
        path="/v1/messages",
        headers={"X-Session-Id": "c" * 8 + "-cccc-4ccc-8ccc-" + "c" * 12},
    )
    a, b = _rows(tmp_data_dir, 2)
    assert a["cobs2_edge"] == "new"
    assert a["lens_cause"] is None and a["lens_break"] is None
    assert b["lens_cause"] == "new_conversation"


def test_history_rewrite_by_reminder_is_history_noise(tmp_data_dir, client):
    """The break message's probes (computed for the new node and memoised)
    name the cause: a reminder block changed in an earlier user message."""
    plaintext = _ready_gm(client, tmp_data_dir)
    s = _sys("You help.")
    u1 = {"role": "user", "content": "do it <system-reminder>r1</system-reminder>"}
    _post(client, plaintext, {"model": "qwen", "messages": [s, u1]}, SESSION)
    u1b = {"role": "user", "content": "do it <system-reminder>r2</system-reminder>"}
    t2 = [s, u1b, {"role": "assistant", "content": "ok here"}, {"role": "user", "content": "next"}]
    _post(client, plaintext, {"model": "qwen", "messages": t2}, SESSION)
    _, row = _rows(tmp_data_dir, 2)
    assert (row["lens_cause"], row["lens_break"]) == ("history_noise", "node:0")


# ---- the kvsig event ------------------------------------------------------------------


def test_kvsig_event_carries_the_lens_block(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _two_turns(
        client,
        plaintext,
        _sys("You help.", "Working directory: /w/1"),
        _sys("You help.", "Working directory: /w/2"),
    )
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    ev = _kvsigs(snap)[-1]
    lens = ev["lens"]
    assert set(lens) == {
        "brk",
        "cause",
        "span_class",
        "span_line",
        "cf_tokens",
        "lost_tokens",
        "other_breaks",
    }
    assert (lens["brk"], lens["cause"], lens["span_class"], lens["span_line"]) == (
        "sys",
        "sys_changed",
        "path",
        2,
    )
    assert lens["cf_tokens"] > 0 and lens["other_breaks"] == 0
    # At admission the reuse is not known yet.
    assert lens["lost_tokens"] is None
    assert for_key_holder(ev)["lens"] == lens


def test_kvsig_event_lens_is_null_when_the_lens_failed(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    monkeypatch.setattr("app.lens.causes.evaluate", _boom)
    _post(client, plaintext, BODY, SESSION)
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    ev = _kvsigs(snap)[-1]
    assert ev["available"] is True and ev["lens"] is None


# ---- text-free --------------------------------------------------------------------------


def test_canary_never_reaches_a_lens_column_event_or_log(tmp_data_dir, client, caplog):
    caplog.set_level(logging.DEBUG)
    plaintext = _ready_gm(client, tmp_data_dir)
    _two_turns(
        client,
        plaintext,
        _sys("You help.", f"Working directory: /w/{CANARY}/1"),
        _sys("You help.", f"Working directory: /w/{CANARY}/2"),
    )
    _, row = _rows(tmp_data_dir, 2)
    assert row["lens_cause"] == "sys_changed"
    lens_vals = {c: row[c] for c in LENS_COLS}
    assert CANARY not in json.dumps(lens_vals)
    _q, snap = client.app.state.prompt_lens_hub.subscribe()
    assert CANARY not in json.dumps(_kvsigs(snap)[-1]["lens"])
    assert CANARY not in caplog.text


def test_canary_never_reaches_a_log_when_the_lens_fails(tmp_data_dir, client, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    plaintext = _ready_gm(client, tmp_data_dir)

    def boom_quoting(*a, **k):
        raise RuntimeError(f"quoting the prompt {CANARY}")

    monkeypatch.setattr("app.lens.causes.evaluate", boom_quoting)
    monkeypatch.setattr("app.lens.probes.message_probes", boom_quoting)
    _two_turns(
        client,
        plaintext,
        _sys("You help.", f"dir {CANARY} 1"),
        _sys("You help.", f"dir {CANARY} 2"),
    )
    _rows(tmp_data_dir, 2)
    assert CANARY not in caplog.text


# ---- spec 4.5: a new session in another directory (fallback by tools) -----------


_TOOLS = [{"type": "function", "function": {"name": "read_file"}}]
_OPENCODE_SYS = [
    "You are opencode, an interactive CLI tool.",
    "Use the tools to help the user.",
    "Here is some useful information about the environment.",
]


def _session(client, plaintext, sys_lines, sid):
    body = {
        "model": "qwen",
        "tools": _TOOLS,
        "messages": [_sys(*sys_lines), {"role": "user", "content": "start the task"}],
    }
    _post(client, plaintext, body, {"X-Session-Id": sid})


def test_new_session_in_another_directory_is_sys_changed_path(tmp_data_dir, client):
    """OpenCode (spec 4.5): same tools, the system prompt differs only at the
    working-directory line, so its loose form differs too; the (model, tools)
    fallback still compares it with the key's last head."""
    plaintext = _ready_gm(client, tmp_data_dir)
    _session(
        client,
        plaintext,
        [*_OPENCODE_SYS, "Working directory: /home/u/wt/lens-1"],
        "6" * 8 + "-6666-4666-8666-" + "6" * 12,
    )
    _session(
        client,
        plaintext,
        [*_OPENCODE_SYS, "Working directory: /home/u/wt/lens-2"],
        "7" * 8 + "-7777-4777-8777-" + "7" * 12,
    )
    _, row = _rows(tmp_data_dir, 2)
    assert (
        row["lens_cause"],
        row["lens_break"],
        row["lens_span_class"],
        row["lens_span_line"],
    ) == (
        "sys_changed",
        "sys",
        "path",
        4,
    )
    assert row["lens_cf_tokens"] is not None


def test_fallback_is_only_for_a_first_turn_body(tmp_data_dir, client):
    """M6: the (model, tools) fallback compares only a new conversation's head.
    A body with earlier turns whose parent left memory stays unattributed."""
    plaintext = _ready_gm(client, tmp_data_dir)
    _session(
        client,
        plaintext,
        [*_OPENCODE_SYS, "Working directory: /home/u/wt/lens-1"],
        "6" * 8 + "-6666-4666-8666-" + "6" * 12,
    )
    body = {
        "model": "qwen",
        "tools": _TOOLS,
        "messages": [
            _sys(*_OPENCODE_SYS, "Working directory: /home/u/wt/lens-2"),
            {"role": "user", "content": "an older task"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "and the next one"},
        ],
    }
    _post(client, plaintext, body, {"X-Session-Id": "7" * 8 + "-7777-4777-8777-" + "7" * 12})
    _, row = _rows(tmp_data_dir, 2)
    assert row["cobs2_edge"] == "new"
    assert (row["lens_cause"], row["lens_break"], row["lens_span_line"]) == (None, None, None)


def test_fallback_needs_a_close_system_prompt(tmp_data_dir, client):
    """Over 3 lines and 5 % changed: another app's prompt, not a moved path."""
    plaintext = _ready_gm(client, tmp_data_dir)
    _session(
        client, plaintext, ["a1", "b1", "c1", "d1", "e1"], "8" * 8 + "-8888-4888-8888-" + "8" * 12
    )
    _session(
        client, plaintext, ["a2", "b2", "c2", "d2", "e1"], "9" * 8 + "-9999-4999-8999-" + "9" * 12
    )
    _, row = _rows(tmp_data_dir, 2)
    assert row["lens_cause"] == "new_conversation" and row["lens_break"] is None


def test_no_remember_without_digests(tmp_data_dir, client, monkeypatch):
    """A digest fault never stores an empty digest set as the key's head."""
    plaintext = _ready_gm(client, tmp_data_dir)
    monkeypatch.setattr("app.lens.probes.sys_line_digests", _boom)
    remember = _spy(monkeypatch, "app.lens.heads.RecentHeads.remember")
    _post(client, plaintext, BODY, SESSION)
    _rows(tmp_data_dir, 1)
    assert remember.count == 0


# ---- Task 1.4: reply verdicts and the layer 3 capture fields ------------------------

from tests.unit.proxy.test_forest_history_fields import _chunk, _sse  # noqa: E402

ARGS_CANARY = "CANARY-1d4b7e-args"
RESULT_CANARY = "CANARY-1d4b7e-result"
L3_COLS = (
    "lens_repeat_results",
    "max_tokens_req",
    "reasoning_tokens",
    "reply_chars",
    "calls_n",
    "calls_bad",
    "tools_lint",
)
_LONG_DESC = "Search the documentation index for pages that match a query and return the best ones."
_L3_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": _LONG_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "query"},
                    "limit": {"type": "integer", "description": "how many"},
                },
                "required": ["q", "limit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"description": "p"}, "a": {}, "b": {}},
            },
        },
    },
]


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _tool_reply(calls, *, content=None, reasoning=None, usage_extra=None):
    usage = {"prompt_tokens": 40, "completion_tokens": 9, **(usage_extra or {})}
    msg = {"role": "assistant", "content": content, "tool_calls": calls}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return _json_resp(
        {
            "id": "c1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            "usage": usage,
        }
    )


def _post_resp(client, plaintext, body, resp, path="/v1/chat/completions", headers=SESSION):
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=resp)):
        r = client.post(
            path,
            headers={"Authorization": f"Bearer {plaintext}", **(headers or {})},
            json=body,
        )
    assert r.status_code == 200, r.text
    return r


def _hk(client, name):
    from app.proxy.forest_fields import hkey

    return hkey(client.app.state.settings.cookie_secret, name, 8)


def _l3_body(**kw):
    return {
        "model": "qwen",
        "tools": _L3_TOOLS,
        "messages": [{"role": "user", "content": "find it"}],
        **kw,
    }


_L3_CALLS = [
    _call("search_docs", json.dumps({"q": ARGS_CANARY, "limit": 2}), "a"),
    _call("search_docs", json.dumps({"q": ARGS_CANARY}), "b"),
    _call("nope", "{}", "c"),
    _call("search_docs", '{"q": "' + ARGS_CANARY, "d"),
    _call("search_docs", json.dumps({"q": "x", "limit": "2"}), "e"),
]


def _assert_l3_verdicts(client, row):
    h = _hk(client, "search_docs")
    # [k, hash8, code] for the non-ok calls only; k is the reply order index.
    assert json.loads(row["calls_bad"]) == [
        [1, h, "missing_arg"],
        [2, _hk(client, "nope"), "unknown_tool"],
        [3, h, "json"],
        [4, h, "type"],
    ]
    assert row["calls_n"] == 5
    # search_docs is clean; read_file: no type, no description on two
    # parameters, and a short description with three parameters.
    assert json.loads(row["tools_lint"]) == [0, 1 | 2 | 8]


def test_reply_verdicts_are_stored(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _post_resp(client, plaintext, _l3_body(), _tool_reply(_L3_CALLS, content="ok"))
    (row,) = _rows(tmp_data_dir, 1)
    _assert_l3_verdicts(client, row)
    assert row["reply_chars"] == 2
    from app.lens.probes import args8

    secret = client.app.state.settings.cookie_secret
    tout = json.loads(row["tools_out"])
    assert [len(e) for e in tout] == [3] * 5
    assert tout[0][2] == args8(_L3_CALLS[0]["function"]["arguments"], secret)
    assert ARGS_CANARY not in row["tools_out"] + row["calls_bad"]


def test_reply_counts_for_a_reasoning_only_reply(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _post_resp(client, plaintext, _l3_body(), _tool_reply(None, content="", reasoning="hmm"))
    (row,) = _rows(tmp_data_dir, 1)
    assert (row["reply_chars"], row["calls_n"], row["calls_bad"]) == (0, 0, "[]")


def test_leaked_markup_is_stored(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    leaked = '<tool_call>{"name": "read_file", "arguments": {}}</tool_call>'
    _post_resp(client, plaintext, _l3_body(), _tool_reply(None, content=leaked))
    (row,) = _rows(tmp_data_dir, 1)
    assert json.loads(row["calls_bad"]) == [[None, _hk(client, "read_file"), "leaked_markup"]]
    assert row["calls_n"] == 0


def _l3_stream():
    """Two parallel calls whose argument fragments interleave."""
    return [
        _chunk({"role": "assistant", "content": "Looking"}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "a",
                        "type": "function",
                        "function": {"name": "search_docs", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "b",
                        "type": "function",
                        "function": {"name": "nope", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q": "' + ARGS_CANARY}}]}),
        _chunk({"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]}),
        _chunk({}, finish="tool_calls"),
        _chunk(
            usage={
                "prompt_tokens": 40,
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 5},
            }
        ),
    ]


def _stream_post(client, plaintext, path, body, frames):
    with patch("httpx.AsyncClient.send", new=AsyncMock(return_value=_sse(frames))):
        with client.stream(
            "POST",
            path,
            headers={"Authorization": f"Bearer {plaintext}", **SESSION},
            json=body,
        ) as r:
            raw = b"".join(r.iter_bytes())
            assert r.status_code == 200, raw
    return raw


def test_streamed_parallel_calls_each_get_a_verdict(tmp_data_dir, client):  # Review Focus 2
    plaintext = _ready_gm(client, tmp_data_dir)
    _stream_post(
        client,
        plaintext,
        "/v1/chat/completions",
        _l3_body(stream=True, max_completion_tokens=300),
        _l3_stream(),
    )
    (row,) = _rows(tmp_data_dir, 1)
    assert json.loads(row["calls_bad"]) == [
        [0, _hk(client, "search_docs"), "missing_arg"],
        [1, _hk(client, "nope"), "unknown_tool"],
    ]
    assert (row["calls_n"], row["reply_chars"]) == (2, len("Looking"))
    assert (row["reasoning_tokens"], row["max_tokens_req"]) == (5, 300)
    from app.lens.probes import args8

    secret = client.app.state.settings.cookie_secret
    tout = json.loads(row["tools_out"])
    assert [e[2] for e in tout] == [
        args8('{"q": "' + ARGS_CANARY + '"}', secret),
        args8("{}", secret),
    ]
    assert ARGS_CANARY not in row["tools_out"] + row["calls_bad"]


def test_anthropic_tool_use_translated_reply_is_judged(tmp_data_dir, client):
    """/v1/messages: the Anthropic tools are translated into the chat body the
    verdict reads; the engine's tool calls become the client's tool_use blocks."""
    plaintext = _ready_gm(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "tools": [
            {
                "name": f["function"]["name"],
                "description": f["function"]["description"],
                "input_schema": f["function"]["parameters"],
            }
            for f in _L3_TOOLS
        ],
        "messages": [{"role": "user", "content": "find it"}],
    }
    r = _post_resp(client, plaintext, body, _tool_reply(_L3_CALLS), path="/v1/messages")
    assert any(b["type"] == "tool_use" for b in r.json()["content"])
    (row,) = _rows(tmp_data_dir, 1)
    _assert_l3_verdicts(client, row)
    assert row["max_tokens_req"] is not None and row["max_tokens_req"] <= 64


def test_streamed_anthropic_messages_reply_is_judged(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "stream": True,
        "tools": [
            {
                "name": f["function"]["name"],
                "description": f["function"]["description"],
                "input_schema": f["function"]["parameters"],
            }
            for f in _L3_TOOLS
        ],
        "messages": [{"role": "user", "content": "find it"}],
    }
    raw = _stream_post(client, plaintext, "/v1/messages", body, _l3_stream())
    assert b"tool_use" in raw
    (row,) = _rows(tmp_data_dir, 1)
    assert [e[-1] for e in json.loads(row["calls_bad"])] == ["missing_arg", "unknown_tool"]
    assert row["reasoning_tokens"] == 5


def test_reasoning_and_max_tokens_stored(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _post_resp(
        client,
        plaintext,
        _l3_body(max_tokens=128),
        _tool_reply(
            None, content="fine", usage_extra={"completion_tokens_details": {"reasoning_tokens": 7}}
        ),
    )
    _post_resp(
        client, plaintext, _l3_body(max_completion_tokens=99), _tool_reply(None, content="x")
    )
    _post_resp(client, plaintext, _l3_body(max_tokens="lots"), _tool_reply(None, content="x"))
    a, b, c = _rows(tmp_data_dir, 3)
    assert (a["reasoning_tokens"], a["max_tokens_req"]) == (7, 128)
    assert (b["reasoning_tokens"], b["max_tokens_req"]) == (None, 99)
    assert c["max_tokens_req"] is None


def _results_body(contents, user="go on"):
    calls = [
        _call("read_file", json.dumps({"path": f"f{i}"}), f"r{i}") for i in range(len(contents))
    ]
    msgs = [
        {"role": "system", "content": "You help."},
        {"role": "user", "content": "read them"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        *({"role": "tool", "tool_call_id": f"r{i}", "content": c} for i, c in enumerate(contents)),
        {"role": "user", "content": user},
    ]
    return {"model": "qwen", "messages": msgs}


def _expected_repeat(body, repeated_chars, prompt_tokens=40):
    from app.cobs2.canonical import canonicalize
    from app.cobs2.shape import _parts_chars

    canon = canonicalize(body)
    total = sum(_parts_chars(m["parts"]) for m in canon.head + canon.messages) + sum(
        len(t) for t in canon.tools
    )
    return int(repeated_chars * prompt_tokens / total)


def test_repeat_result_counted_once_per_repeat(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    a = f"file a {RESULT_CANARY} " + "a" * 400
    b = "file b " + "b" * 300
    once = _results_body([a, b])
    _post(client, plaintext, once, SESSION)
    # The same content three times: the first never counts, each repeat once.
    thrice = _results_body([a, b, a, [{"type": "text", "text": a}]], user="again")
    _post(client, plaintext, thrice, {"X-Session-Id": "a" * 8 + "-aaaa-4aaa-8aaa-" + "a" * 12})
    r1, r2 = _rows(tmp_data_dir, 2)
    assert r1["lens_repeat_results"] == 0
    assert r2["lens_repeat_results"] == _expected_repeat(thrice, 2 * len(a)) > 0
    # No tool result at all: nothing repeated.
    _post(client, plaintext, BODY, SESSION)
    assert _rows(tmp_data_dir, 3)[2]["lens_repeat_results"] == 0


def test_tool_error_heuristic_on_openai_results(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    body = _results_body(["Traceback (most recent call last):\n  boom", "all good"])
    _post(client, plaintext, body, SESSION)
    client.app.state.cobs2_key = None
    _post(client, plaintext, body, {"X-Session-Id": "b" * 8 + "-bbbb-4bbb-8bbb-" + "b" * 12})
    on, off = _rows(tmp_data_dir, 2)
    assert [x[2] for x in json.loads(on["tools_in"])] == [True, False]
    # kvsig off: the heuristic is a lens call, so only the old prefix rule.
    assert [x[2] for x in json.loads(off["tools_in"])] == [False, False]


def test_tool_error_heuristic_skips_anthropic_results(tmp_data_dir, client):
    """An Anthropic tool_result says is_error itself: no guessing on its text."""
    plaintext = _ready_gm(client, tmp_data_dir)
    body = {
        "model": "qwen",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Traceback: but fine"},
                ],
            },
        ],
    }
    _post_resp(client, plaintext, body, _tool_reply(None, content="ok"), path="/v1/messages")
    (row,) = _rows(tmp_data_dir, 1)
    assert [x[2] for x in json.loads(row["tools_in"])] == [False]


def test_lint_runs_once_per_toolset(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    lint = _spy(monkeypatch, "app.lens.schema_lint.lint_tool")
    for _ in range(3):
        _post_resp(client, plaintext, _l3_body(), _tool_reply(None, content="x"))
    rows = _rows(tmp_data_dir, 3)
    assert lint.count == len(_L3_TOOLS)
    assert all(json.loads(r["tools_lint"]) == [0, 11] for r in rows)
    from app.lens import hot

    assert len(hot._TOOLS_LINT) == 1
    hot.clear_memos()
    assert len(hot._TOOLS_LINT) == 0


def test_kvsig_off_stores_no_layer3_fields(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    client.app.state.cobs2_key = None
    spies = [
        _spy(monkeypatch, t)
        for t in (
            "app.lens.reply_verdicts.call_verdicts",
            "app.lens.reply_verdicts.reply_counts",
            "app.lens.schema_lint.lint_tool",
            "app.lens.tool_errors.looks_failed",
            "app.lens.probes.args8",
            "app.lens.probes.content_hash",
            "app.lens.probes.result_hash",
        )
    ]
    body = _results_body(["Traceback x", "same", "same"])
    body["tools"] = _L3_TOOLS
    body["max_tokens"] = 50
    _post_resp(
        client,
        plaintext,
        body,
        _tool_reply(_L3_CALLS, usage_extra={"completion_tokens_details": {"reasoning_tokens": 3}}),
    )
    _stream_post(client, plaintext, "/v1/chat/completions", _l3_body(stream=True), _l3_stream())
    rows = _rows(tmp_data_dir, 2)
    assert [s.count for s in spies] == [0] * len(spies)
    for row in rows:
        assert all(row[c] is None for c in L3_COLS), {c: row[c] for c in L3_COLS}
        assert all(len(e) == 2 for e in json.loads(row["tools_out"]))


def test_fault_in_the_verdicts_keeps_the_rest(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    monkeypatch.setattr("app.lens.reply_verdicts.call_verdicts", _boom)
    body = _l3_body(max_tokens=77)
    _post_resp(client, plaintext, body, _tool_reply(_L3_CALLS))
    (row,) = _rows(tmp_data_dir, 1)
    assert row["calls_bad"] is None
    # The other fields are their own steps.
    assert row["max_tokens_req"] == 77 and row["calls_n"] == 5
    assert row["tools_lint"] is not None and row["cobs2_head"] is not None


def test_canary_args_and_results_never_reach_a_column_or_log(tmp_data_dir, client, caplog):
    caplog.set_level(logging.DEBUG)
    plaintext = _ready_gm(client, tmp_data_dir)
    body = _results_body([f"Error: {RESULT_CANARY}", f"x {RESULT_CANARY}", f"x {RESULT_CANARY}"])
    body["tools"] = _L3_TOOLS
    _post_resp(client, plaintext, body, _tool_reply(_L3_CALLS, content=f"<tool_call>{ARGS_CANARY}"))
    _stream_post(client, plaintext, "/v1/chat/completions", _l3_body(stream=True), _l3_stream())
    rows = _rows(tmp_data_dir, 2)
    for row in rows:
        vals = json.dumps({c: row[c] for c in (*L3_COLS, "tools_out", "tools_in")})
        assert ARGS_CANARY not in vals and RESULT_CANARY not in vals
    assert rows[0]["lens_repeat_results"] > 0
    assert ARGS_CANARY not in caplog.text and RESULT_CANARY not in caplog.text


def test_canary_never_reaches_a_log_when_the_verdicts_fail(
    tmp_data_dir, client, caplog, monkeypatch
):
    caplog.set_level(logging.DEBUG)
    plaintext = _ready_gm(client, tmp_data_dir)

    def boom_quoting(*a, **k):
        raise RuntimeError(f"quoting the arguments {ARGS_CANARY}")

    monkeypatch.setattr("app.lens.reply_verdicts.call_verdicts", boom_quoting)
    monkeypatch.setattr("app.lens.reply_verdicts.reply_counts", boom_quoting)
    monkeypatch.setattr("app.lens.schema_lint.lint_tool", boom_quoting)
    _post_resp(client, plaintext, _l3_body(), _tool_reply(_L3_CALLS))
    _rows(tmp_data_dir, 1)
    assert ARGS_CANARY not in caplog.text


# ---- prompt versions (stage 3, Task 3.1) ------------------------------------------


def _versioned(client, plaintext, sys_text, tools=None, headers=SESSION, sid=None):
    body = {"model": "qwen", "messages": [_sys(*sys_text), {"role": "user", "content": "go"}]}
    if tools:
        body["tools"] = tools
    _post(client, plaintext, body, {**headers, **({"X-Session-Id": sid} if sid else {})})


def test_a_date_line_keeps_the_version_and_a_stable_line_starts_one(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    sid = "6a0c2a9e-7d41-4b8e-9c3a-1e2f3a4b5c6d"
    _versioned(client, plaintext, ("You help.", "Today is 2026-11-20."), sid=sid)
    _versioned(client, plaintext, ("You help.", "Today is 2026-11-21."), sid=sid + "1")
    _versioned(client, plaintext, ("You help sparingly.", "Today is 2026-11-21."), sid=sid + "2")
    a, b, c = _rows(tmp_data_dir, 3)
    assert a["lens_version"] and len(a["lens_version"]) == 8
    assert b["lens_version"] == a["lens_version"]
    assert c["lens_version"] != a["lens_version"]


def test_the_header_names_the_version_and_wins(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _versioned(client, plaintext, ("You help.",), headers={**SESSION, "X-Prompt-Version": "v4"})
    _versioned(
        client,
        plaintext,
        ("You help.",),
        headers={**SESSION, "X-Prompt-Version": "x" * 65},
        sid="7a0c2a9e-7d41-4b8e-9c3a-1e2f3a4b5c6d",
    )
    named, dropped = _rows(tmp_data_dir, 2)
    assert named["lens_version"] == "h:v4"
    assert len(dropped["lens_version"]) == 8  # an over-long header is dropped: detected


def test_kvsig_off_stores_no_version_and_computes_nothing(tmp_data_dir, client, monkeypatch):
    plaintext = _ready_gm(client, tmp_data_dir)
    client.app.state.cobs2_key = None
    calls = _spy(monkeypatch, "app.lens.versions.version_id")
    _versioned(client, plaintext, ("You help.",), headers={**SESSION, "X-Prompt-Version": "v4"})
    (row,) = _rows(tmp_data_dir, 1)
    assert calls.count == 0 and row["lens_version"] is None


def test_a_fault_in_version_detection_leaves_the_response_and_other_fields(
    tmp_data_dir, client, monkeypatch
):
    plaintext = _ready_gm(client, tmp_data_dir)
    monkeypatch.setattr("app.lens.versions.version_id", _boom)
    _two_turns(client, plaintext, _sys("You help.", "Dir: /w/1"), _sys("You help.", "Dir: /w/2"))
    first, row = _rows(tmp_data_dir, 2)
    assert row["lens_version"] is None
    assert row["lens_cause"] == "sys_changed"


def test_the_canary_never_reaches_the_version(tmp_data_dir, client):
    plaintext = _ready_gm(client, tmp_data_dir)
    _versioned(client, plaintext, (f"You help. {CANARY}",))
    (row,) = _rows(tmp_data_dir, 1)
    assert CANARY not in str(row["lens_version"])
