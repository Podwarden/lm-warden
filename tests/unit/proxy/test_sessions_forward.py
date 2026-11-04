"""_forward records the sessions-pages fields (task 3)."""

import json

from app.proxy import routes
from app.proxy.forest_fields import hkey
from app.sessions.refs import sess_ref
from tests.unit.proxy.test_cache_estimate_forward import _live_rows, _spy_history, _spy_registry
from tests.unit.proxy.test_cobs2_forward import SESSION, _turn
from tests.unit.proxy.test_dp_routing_forward import _post, _ready


def _sys(t):
    return {"role": "system", "content": t}


def _u(t):
    return {"role": "user", "content": t}


def _a(t):
    return {"role": "assistant", "content": t}


def _fn(name, desc="", params=None):
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params or {}},
    }


def test_forward_records_session_conv_shape_parent_req(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t1 = _turn(client, recs, [_sys("be brief"), _u("a")], reply="ra")
    t2 = _turn(client, recs, [_sys("be brief"), _u("a"), _a("ra"), _u("b")])
    assert t1["cobs2_session"] == t1["session_key"] and t1["cobs2_session"]
    assert t1["cobs2_conv"] == t2["cobs2_conv"] and len(t1["cobs2_conv"]) == 34
    assert t2["cobs2_parent_req"] == t1["id"]
    assert json.loads(t2["cobs2_shape"])["msgs"] == {"user": 2, "assistant": 1, "tool": 0}
    secret = client.app.state.settings.cookie_secret
    assert t1["cobs2_sess_ref"] == sess_ref(secret, "tok1", t1["session_key"])


def test_reminder_rewrite_records_brk(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    _turn(client, recs, [_sys("s"), _u("q <system-reminder>Mon</system-reminder>")], reply="ra")
    t2 = _turn(
        client, recs, [_sys("s"), _u("q <system-reminder>Tue</system-reminder>"), _a("ra"), _u("b")]
    )
    assert json.loads(t2["cobs2_shape"])["brk"] == 0


def test_sessionless_request_gets_fallback_session(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t = _turn(client, recs, [_sys("s"), _u("a")], headers={})
    assert t["cobs2_session"].startswith("f:")
    assert t["cobs2_sess_ref"] and len(t["cobs2_sess_ref"]) == 32


def test_toolset_hashed_names_only_for_session_tools(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    body = {
        "model": "qwen",
        "messages": [_sys("s"), _u("a")],
        "tools": [_fn("Read", desc="SECRET-DESC"), _fn("Task", params={"SECRET_PARAM": {}})],
    }
    r, _ = _post(client, body, headers=SESSION)
    assert r.status_code == 200
    rec = recs[-1]
    secret = client.app.state.settings.cookie_secret
    assert rec["cobs2_toolset_tools"][1][:2] == [hkey(secret, "Task", 8), "agent"]
    assert rec["cobs2_toolset_names"] == ["Read", "Task"]
    assert "Task" not in rec["cobs2_shape"]
    assert "SECRET" not in json.dumps(
        {k: v for k, v in rec.items() if k != "tools_in"}, default=str
    )


def test_over_cap_body_keeps_harness_session(tmp_data_dir, client, monkeypatch):
    monkeypatch.setattr(routes._cobs_canonical, "MAX_CHAIN_BYTES", 10)
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t = _turn(client, recs, [_sys("s"), _u("a long enough message")])
    assert t["cobs2_node"] is None
    assert t["cobs2_session"] == t["session_key"] and t["cobs2_session"]
    assert t["cobs2_sess_ref"]


def test_live_row_id_equals_history_id(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _spy_registry(client)
    recs = _spy_history(client)
    _turn(client, recs, [_u("a")])
    assert _live_rows(client)[0].id == recs[-1]["id"]


def test_session_fields_fail_open(tmp_data_dir, client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(routes, "shape_of", boom)
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t = _turn(client, recs, [_sys("s"), _u("a")])
    assert t["cobs2_shape"] is None
    assert t["cobs2_conv"] and t["cobs2_session"]


def test_rewritten_first_message_stays_in_its_lane(tmp_data_dir, client):
    """A harness that changes the first message after turn 1 (Gleb's subagents, 2026-10-26):
    turn 2 carries turn 1's reply, so it continues turn 1 in the same conversation lane."""
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t1 = _turn(client, recs, [_sys("s"), _u("task, agent id 111")], reply="calling tools")
    t2 = _turn(
        client, recs, [_sys("s"), _u("task, agent id 222"), _a("calling tools"), _u("results")]
    )
    assert t2["cobs2_edge"] == "continue" and t2["cobs2_parent_req"] == t1["id"]
    assert t2["cobs2_conv"] == t1["cobs2_conv"]


def test_forward_records_dead_weight_in_shape(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t = _turn(client, recs, [_sys("S"), _u("<system-reminder>x</system-reminder>a")])
    assert json.loads(t["cobs2_shape"])["dw"] == {
        "dup": 0,
        "boil": len("<system-reminder>x</system-reminder>"),
        "reason": 0,
    }


def test_dead_weight_failure_keeps_the_shape(tmp_data_dir, client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(routes, "dead_weight_of", boom)
    _ready(client, tmp_data_dir, dp=1)
    recs = _spy_history(client)
    t = _turn(client, recs, [_sys("S"), _u("a")])
    shape = json.loads(t["cobs2_shape"])
    assert "dw" not in shape and shape["msgs"]["user"] == 1
