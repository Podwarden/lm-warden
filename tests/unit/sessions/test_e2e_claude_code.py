"""End-to-end synthetic Claude Code sessions through the whole stack (sessions pages, task 13):
_forward records, the history writer commits, the sessions API assembles the tree."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.sessions.replay import ENGINES
from tests.conftest import jwt_login, seed_admin_user
from tests.unit.proxy.test_cobs2_forward import SESSION, SUMMARY_ASK
from tests.unit.proxy.test_dp_routing_forward import _post, _ready, _resp

SYS_MAIN = "You are the main coding agent."
SYS_SUB = "You are an Explore subagent."


def _fn(name):
    return {
        "type": "function",
        "function": {"name": name, "description": "d", "parameters": {"type": "object"}},
    }


def _reply(text=None, calls=(), cached=0):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
            for i, n in enumerate(calls)
        ]
    return _resp(
        {
            "choices": [
                {"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    )


def _send(client, messages, tools, reply, headers=SESSION):
    r, _ = _post(
        client, {"model": "qwen", "messages": messages, "tools": tools}, headers=headers, resp=reply
    )
    assert r.status_code == 200, r.text
    return r.headers["X-LMW-Request-Id"]


def _committed(client, *ids):
    """Wait until every id's row is committed (the background writer batches for ~1 s)."""
    hist = client.app.state.request_history
    client.portal.call(hist.flush)
    for rid in ids:
        assert client.portal.call(hist.wait_for, rid, 5.0), rid


def _session(client, h, rid):
    r = client.get(f"/api/sessions/r_{rid}", headers=h)
    assert r.status_code == 200, r.text
    return r.json()


def _u(t):
    return {"role": "user", "content": t}


def _a(t=None, calls=()):
    m = {"role": "assistant", "content": t}
    if calls:
        m["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
            for i, n in enumerate(calls)
        ]
    return m


def _tool(i, text="ok"):
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": text}


def test_local_claude_code_session_tree_edges_inferred_links_and_causes(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = jwt_login(client)
    tools_main = [_fn("Read"), _fn("Task"), _fn("Bash")]
    tools_sub = [_fn("Read"), _fn("Grep")]
    sysm = {"role": "system", "content": SYS_MAIN}
    first = _u("fix the bug <system-reminder>Today is Monday.</system-reminder>")
    m1 = [sysm, first]
    t1 = _send(client, m1, tools_main, _reply("looking"))
    m2 = [*m1, _a("looking"), _u("go ahead")]
    t2 = _send(client, m2, tools_main, _reply(None, calls=("Task", "Task", "Task")))
    for k in range(3):
        ss = {"role": "system", "content": SYS_SUB}
        s1 = [ss, _u(f"explore area {k}")]
        _send(client, s1, tools_sub, _reply(None, calls=("Read",)))
        _send(client, [*s1, _a(None, calls=("Read",)), _tool(0)], tools_sub, _reply(f"found {k}"))
    # a Claude Code title side call: one turn, no tools
    _send(
        client,
        [{"role": "system", "content": "Write a title."}, _u("fix the bug")],
        [],
        _reply("Bug fix"),
    )
    m3 = [*m2, _a(None, calls=("Task", "Task", "Task")), _tool(0), _tool(1), _tool(2)]
    t3 = _send(client, m3, tools_main, _reply("all explored"))
    rewritten = _u(
        "fix the bug <system-reminder>Today is Tuesday. 3 files changed.</system-reminder>"
    )
    m4 = [sysm, rewritten, *m3[2:], _a("all explored"), _u("now patch it")]
    t4 = _send(client, m4, tools_main, _reply("patched"))
    m5 = [*m4, _a("patched"), _u(SUMMARY_ASK)]
    summary = " ".join(f"point{i} of the plan we agreed on" for i in range(60))
    t5 = _send(client, m5, tools_main, _reply(summary))
    t6 = _send(
        client, [sysm, _u("This session is being continued: " + summary)], tools_main, _reply("ok")
    )
    _committed(client, t1, t2, t3, t4, t5, t6)
    s = _session(client, h, t6)

    labels = {c["label"]: c for c in s["conversations"]}
    assert set(labels) == {
        "main",
        "subagent #1",
        "subagent #2",
        "subagent #3",
        "main after compaction",
    }
    for i in (1, 2, 3):
        assert labels[f"subagent #{i}"]["parent"]["link"] == "inferred"
        assert labels[f"subagent #{i}"]["parent"]["request_id"] == t2
    assert labels["main after compaction"]["parent"]["link"] == "compaction"
    assert labels["main after compaction"]["parent"]["request_id"] == t5
    turns = {t["request_id"]: t for t in s["turns"]}
    assert turns[t4]["divergence_cause"] == "harness_rewrite"
    assert turns[t4]["shape"]["brk"] == 0
    assert turns[t6]["edge"] == "compacted_from"
    assert turns[t3]["parent_request_id"] == t2 and turns[t2]["parent_request_id"] == t1
    assert len(s["side_calls"]) == 1 and s["totals"]["compactions"] == 1
    # main offers {Read, Task, Bash} and calls Task; subagents offer {Read, Grep} and call Read
    assert (s["tools"]["offered"], s["tools"]["called"], s["tools"]["unused"]) == (4, 2, 2)
    assert {r["name"] for r in s["tools"]["rows"]} == {"Read", "Task", "Bash", "Grep"}
    assert [t["request_id"] for t in s["turns"] if t["lineage_unknown"]] == []
    assert SYS_MAIN not in json.dumps(s) and "fix the bug" not in json.dumps(s)


def test_dead_weight_saved_and_replay_through_forward(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=3)
    client.app.state.prefill_model = SimpleNamespace(cold_rate_for=lambda _m: 1000.0)
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = jwt_login(client)
    tools = [_fn("Read"), _fn("Task"), _fn("Bash")]
    sysm = {"role": "system", "content": SYS_MAIN}
    m1 = [sysm, _u("fix it <system-reminder>Today is Monday, the repo is clean.</system-reminder>")]
    t1 = _send(client, m1, tools, _reply(None, calls=("Read",), cached=0))
    thought = {**_a(None, calls=("Read",)), "reasoning_content": "think " * 40}
    big = "FILE CONTENTS " * 20
    m2 = [*m1, thought, _tool(0, big), _u("again"), _a(None, calls=("Read",)), _tool(0, big)]
    t2 = _send(client, m2, tools, _reply("read twice", cached=60))
    m3 = [*m2, _a("read twice"), _u("done?")]
    t3 = _send(client, m3, tools, _reply("done", cached=80))
    _committed(client, t1, t2, t3)
    s = _session(client, h, t3)

    kinds = s["dead_weight"]["kinds"]
    assert kinds["boil"]["tokens"] > 0 and kinds["dup"]["tokens"] > 0
    assert kinds["reason"]["tokens"] > 0 and s["totals"]["replicas"] == 3
    assert s["totals"]["saved_vs_vanilla_s"] == pytest.approx((60 + 80) * (2 / 3) / 1000.0)
    assert s["harness"] == "claude_code"
    assert {e["engine"] for e in s["engine_replay"]["engines"]} == set(ENGINES)


def test_test_agent_flow_header_wait_and_read_back(tmp_data_dir, client):
    """The coding agent's recipe: forest login once with its key, chat with a session header, read
    X-LMW-Request-Id, wait for the row, then read the session tree back."""
    from tests.unit.proxy.test_dp_routing_forward import PLAINTEXT
    from tests.unit.stats.test_forest_routes import _forest_login

    _ready(client, tmp_data_dir, dp=1)
    rh = _forest_login(client, PLAINTEXT)
    rid = _send(client, [_u("task one")], [], _reply("hi"))
    got = client.get(f"/api/requests/{rid}?wait_ms=5000", headers=rh)
    assert got.status_code == 200, got.text
    sid = got.json()["session_id"]
    assert sid.startswith("s_")
    tree = client.get(f"/api/sessions/{sid}", headers=rh)
    assert tree.status_code == 200, tree.text
    assert [t["request_id"] for t in tree.json()["turns"]] == [rid]


def _anthropic_sse(*, text: str, tools: tuple[str, ...] = (), cache_read: int = 900) -> bytes:
    def ev(name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 0,
        "output_tokens": 1,
    }
    out = [
        ev(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "m", "role": "assistant", "content": [], "usage": usage},
            },
        ),
        ev(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        ev(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]
    for i, name in enumerate(tools, start=1):
        out += [
            ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"toolu_{i}",
                        "name": name,
                        "input": {},
                    },
                },
            ),
            ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
            ),
            ev("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    stop = "tool_use" if tools else "end_turn"
    out += [
        ev(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop},
                "usage": {"output_tokens": 7},
            },
        ),
        ev("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(out)


def test_relayed_main_agent_with_local_subagents(tmp_data_dir, client):
    import httpx

    from tests.unit.router.harness import CLAUDE_CODE, chat_completion, fake_engine
    from tests.unit.router.test_data_plane import _engine_returning, _setup

    replies = iter(
        [
            _anthropic_sse(text="planning", cache_read=500),
            _anthropic_sse(text="dispatching", tools=("Task", "Task", "Task"), cache_read=800),
        ]
    )

    def anthropic(_req):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=next(replies)
        )

    _setup(client, tmp_data_dir, anthropic=anthropic)
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    h = jwt_login(client)
    hdr = {**CLAUDE_CODE, **SESSION}

    def main_body(msgs):
        return {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "system": SYS_MAIN,
            "stream": True,
            "messages": msgs,
        }

    m1 = [{"role": "user", "content": "fix the bug"}]
    r1 = client.post("/v1/messages", headers=hdr, json=main_body(m1))
    m2 = [
        *m1,
        {"role": "assistant", "content": [{"type": "text", "text": "planning"}]},
        {"role": "user", "content": "go"},
    ]
    r2 = client.post("/v1/messages", headers=hdr, json=main_body(m2))
    rel = [r1.headers["X-LMW-Request-Id"], r2.headers["X-LMW-Request-Id"]]

    local: list[str] = []
    with fake_engine(client, _engine_returning(lambda: chat_completion("found it"))):
        for k in range(3):
            s1 = [{"role": "user", "content": f"explore area {k}"}]
            body = {
                "model": "claude-haiku-4-5",
                "max_tokens": 64,
                "system": SYS_SUB,
                "messages": s1,
            }
            a = client.post("/v1/messages", headers=hdr, json=body)
            s2 = [
                *s1,
                {"role": "assistant", "content": [{"type": "text", "text": "found it"}]},
                {"role": "user", "content": "more"},
            ]
            b = client.post("/v1/messages", headers=hdr, json={**body, "messages": s2})
            assert a.status_code == b.status_code == 200, (a.text, b.text)
            local += [a.headers["X-LMW-Request-Id"], b.headers["X-LMW-Request-Id"]]

    _committed(client, *rel, *local)
    s = _session(client, h, local[-1])
    main = next(c for c in s["conversations"] if c["kind"] == "main")
    assert main["relayed"] is True
    assert {t["request_id"] for t in s["turns"] if t["conversation_id"] == main["id"]} == set(rel)
    assert all(t["relayed"] for t in s["turns"] if t["conversation_id"] == main["id"])
    subs = [c for c in s["conversations"] if c["kind"] == "subagent"]
    assert len(subs) == 3
    assert [c["parent"]["request_id"] for c in subs] == [rel[1]] * 3
    assert all(c["parent"]["link"] == "inferred" for c in subs)
    assert s["totals"]["relayed"]["cached_tokens"] == 500 + 800
    local_cached = sum((t["cached_tokens"] or 0) for t in s["turns"] if not t["relayed"])
    assert s["totals"]["cached_tokens"] == local_cached  # the upstream's cache is never mixed in
