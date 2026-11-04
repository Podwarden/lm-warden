"""Prompt diet: a per-key, opt-in cut of unused tool definitions and carried reasoning
from chat bodies on their way to the engine."""

from __future__ import annotations

import copy
import json

from app.proxy.prompt_diet import Diet, apply_diet, diet_json, parse_diet


def _fn(name: str, desc: str = "d") -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {}}}


def _body() -> dict:
    return {
        "model": "m",
        "tools": [_fn("bash"), _fn("todowrite", "x" * 40), _fn("read"), _fn("webfetch", "y" * 10)],
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "think" * 10,
                "tool_calls": [],
            },
            {"role": "tool", "tool_call_id": "1", "content": "out"},
            {"role": "assistant", "content": "ok", "reasoning": "more"},
        ],
    }


def test_parse_diet_accepts_the_stored_shape_and_rejects_junk():
    d = parse_diet(json.dumps({"strip_tools": ["todowrite", "webfetch"], "strip_reasoning": True}))
    assert d == Diet(frozenset({"todowrite", "webfetch"}), True)
    assert parse_diet(None) is None and parse_diet("") is None and parse_diet("not json") is None
    assert parse_diet(json.dumps({"strip_tools": [], "strip_reasoning": False})) is None
    assert parse_diet(json.dumps({"strip_tools": "bash"})) is None


def test_diet_json_round_trips():
    d = Diet(frozenset({"b", "a"}), True)
    assert parse_diet(diet_json(d)) == d
    assert json.loads(diet_json(d)) == {"strip_tools": ["a", "b"], "strip_reasoning": True}


def test_strips_named_tools_and_carried_reasoning_and_reports_what_it_cut():
    b = _body()
    tools_before = copy.deepcopy(b["tools"])
    r = apply_diet(b, Diet(frozenset({"todowrite", "webfetch"}), True))
    assert [t["function"]["name"] for t in b["tools"]] == ["bash", "read"]
    assert all("reasoning_content" not in m and "reasoning" not in m for m in b["messages"])
    assert b["messages"][4]["content"] == "ok" and b["messages"][2]["tool_calls"] == []
    cut = [t for t in tools_before if t["function"]["name"] in ("todowrite", "webfetch")]
    assert r is not None and r.tools == 2
    assert r.tools_chars == sum(len(json.dumps(t, separators=(",", ":"))) for t in cut)
    assert r.reason_chars == len("think" * 10) + len("more")


def test_a_tool_named_by_tool_choice_is_never_cut():
    b = _body()
    b["tool_choice"] = {"type": "function", "function": {"name": "todowrite"}}
    r = apply_diet(b, Diet(frozenset({"todowrite"}), False))
    assert "todowrite" in [t["function"]["name"] for t in b["tools"]] and r is None


def test_cutting_every_tool_drops_tools_and_an_auto_tool_choice():
    b = _body()
    b["tool_choice"] = "auto"
    apply_diet(b, Diet(frozenset({"bash", "todowrite", "read", "webfetch"}), False))
    assert "tools" not in b and "tool_choice" not in b


def test_nothing_to_cut_leaves_the_body_untouched():
    b = _body()
    before = copy.deepcopy(b)
    assert apply_diet(b, Diet(frozenset({"nope"}), False)) is None and b == before
    nb = {"prompt": "x"}
    assert apply_diet(nb, Diet(frozenset({"bash"}), True)) is None and nb == {"prompt": "x"}


def test_the_cut_is_the_same_every_turn_so_the_prefix_cache_holds():
    a, b = _body(), _body()
    b["messages"].append({"role": "user", "content": "next"})
    d = Diet(frozenset({"todowrite"}), True)
    apply_diet(a, d)
    apply_diet(b, d)
    assert b["messages"][: len(a["messages"])] == a["messages"] and a["tools"] == b["tools"]


# -- forward and key settings ---------------------------------------------------

import sqlite3  # noqa: E402

from tests.conftest import csrf_header, seed_admin_user, wait_until  # noqa: E402
from tests.unit.proxy.test_messages import _ready  # noqa: E402
from tests.unit.proxy.test_thinking_policy import _go, _sent  # noqa: E402

DIET = json.dumps({"strip_tools": ["todowrite"], "strip_reasoning": True})


def _set_diet(db_path, diet: str | None) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET prompt_diet = ?", (diet,))
        db.commit()


def _chat() -> dict:
    b = _body()
    b["model"] = "qwen"
    return b


def test_forward_cuts_per_the_keys_diet_and_records_what_it_cut(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    _set_diet(db_path, DIET)
    r, send = _go(client, _chat())
    assert r.status_code == 200
    sent = json.loads(_sent(send))
    assert [t["function"]["name"] for t in sent["tools"]] == ["bash", "read", "webfetch"]
    assert all("reasoning_content" not in m and "reasoning" not in m for m in sent["messages"])

    def _row():
        with sqlite3.connect(db_path) as db:
            return db.execute("SELECT prompt_diet, cobs2_shape FROM request_history").fetchone()

    diet, shape = wait_until(_row, what="history row")
    assert json.loads(diet) == {
        "tools": 1,
        "tools_chars": len(json.dumps(_fn("todowrite", "x" * 40), separators=(",", ":"))),
        "reason_chars": 54,
    }
    # lineage and the recorded shape describe what the client sent, before the cut
    assert json.loads(shape)["tools"] == 4 and json.loads(shape)["dw"]["reason"] == 54


def test_no_diet_forwards_the_body_as_before(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    r, send = _go(client, _chat())
    sent = json.loads(_sent(send))
    assert len(sent["tools"]) == 4 and sent["messages"][2]["reasoning_content"] == "think" * 10


def test_diet_failure_fails_open(tmp_data_dir, client, monkeypatch):
    _ready(client, tmp_data_dir)
    _set_diet(tmp_data_dir / "vllm-warden.db", DIET)

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.proxy.routes.apply_diet", _boom)
    r, send = _go(client, _chat())
    assert r.status_code == 200 and len(json.loads(_sent(send))["tools"]) == 4


def _admin(client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    return {"Authorization": f"Bearer {r.json()['access_token']}", **csrf_header(client)}


def test_patch_sets_shows_and_clears_a_keys_diet(tmp_data_dir, client):
    hdrs = _admin(client, tmp_data_dir)
    tid = client.post("/api/tokens", json={"name": "k"}, headers=hdrs).json()["id"]
    assert client.get(f"/api/tokens/{tid}", headers=hdrs).json()["prompt_diet"] is None
    want = {"strip_tools": ["todowrite", "webfetch"], "strip_reasoning": True}
    r = client.patch(f"/api/tokens/{tid}", json={"prompt_diet": want}, headers=hdrs)
    assert r.status_code == 200, r.text
    assert r.json()["prompt_diet"] == want
    assert (
        client.patch(f"/api/tokens/{tid}", json={"priority": 3}, headers=hdrs).json()["prompt_diet"]
        == want
    )
    (mine,) = [i for i in client.get("/api/tokens", headers=hdrs).json()["items"] if i["id"] == tid]
    assert mine["prompt_diet"] == want
    r = client.patch(f"/api/tokens/{tid}", json={"prompt_diet": None}, headers=hdrs)
    assert r.status_code == 200 and r.json()["prompt_diet"] is None
    bad = client.patch(
        f"/api/tokens/{tid}", json={"prompt_diet": {"strip_tools": "bash"}}, headers=hdrs
    )
    assert bad.status_code == 422


# -- reasoning retention: drop only before the last real user message (DeepSeek V3.2 / Qwen preserve_thinking=false)


def _chain() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "task 1"},
            {"role": "assistant", "content": None, "reasoning_content": "r1", "tool_calls": []},
            {"role": "tool", "tool_call_id": "a", "content": "out"},
            {"role": "assistant", "content": "done 1", "reasoning_content": "r2"},
            {"role": "user", "content": "task 2"},
            {"role": "assistant", "content": None, "reasoning_content": "r3", "tool_calls": []},
            {"role": "user", "content": "<tool_response>\nresult\n</tool_response>"},
            {"role": "assistant", "content": "x", "reasoning_content": "r4"},
        ]
    }


def test_parse_accepts_the_before_last_user_mode():
    d = parse_diet(json.dumps({"strip_reasoning": "before_last_user"}))
    assert d == Diet(frozenset(), "before_last_user")
    assert json.loads(diet_json(d))["strip_reasoning"] == "before_last_user"
    assert parse_diet(json.dumps({"strip_reasoning": "sometimes"})) is None


def test_before_last_user_keeps_the_current_chain_and_skips_tool_response_users():
    b = _chain()
    r = apply_diet(b, Diet(frozenset(), "before_last_user"))
    kept = [m.get("reasoning_content") for m in b["messages"] if m["role"] == "assistant"]
    # task 2 is the last real user message; the <tool_response> user message does not count
    assert kept == [None, None, "r3", "r4"]
    assert r is not None and r.reason_chars == len("r1") + len("r2")


def test_before_last_user_cuts_nothing_inside_one_task():
    b = {"messages": _chain()["messages"][:4]}
    assert apply_diet(b, Diet(frozenset(), "before_last_user")) is None


def test_patch_accepts_the_before_last_user_mode(tmp_data_dir, client):
    hdrs = _admin(client, tmp_data_dir)
    tid = client.post("/api/tokens", json={"name": "k"}, headers=hdrs).json()["id"]
    want = {"strip_tools": [], "strip_reasoning": "before_last_user"}
    r = client.patch(f"/api/tokens/{tid}", json={"prompt_diet": want}, headers=hdrs)
    assert r.status_code == 200, r.text
    assert r.json()["prompt_diet"] == want
    bad = {"strip_tools": [], "strip_reasoning": "often"}
    assert (
        client.patch(f"/api/tokens/{tid}", json={"prompt_diet": bad}, headers=hdrs).status_code
        == 422
    )
