"""The Prompt Lens MCP endpoint (analysis Task 4.3; owner decisions 2026-11-26)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sqlite3
import time
from typing import Any

from app.lens import mcp as mcp_mod
from app.lens import skills
from tests.conftest import bearer, csrf_header, seed_admin_token
from tests.unit.lens.test_api import KEY_A, KEY_B, _a
from tests.unit.lens.test_routes_ui import (  # noqa: F401
    A,
    B,
    Env,
    env,
    req,
    score_all,
    sid,
)

URL = "/api/lens/mcp"
LENS_TOOLS = {
    "lens_list_sessions", "lens_list_requests", "lens_explain_request", "lens_turn_diff",
    "lens_session_cost", "lens_scorecard", "lens_findings", "lens_trends", "lens_versions",
    "lens_compare_versions", "lens_score", "lens_wontfix", "lens_wait_for_request",
}  # fmt: skip
GUIDE_TOOLS = {"list_skills", "load_skill"}


def admin_headers(env: Env) -> dict[str, str]:  # noqa: F811
    """An admin token (``vwa_``): the credential an MCP client would hold (no CSRF needed)."""
    return bearer(seed_admin_token(env.db)[1])


def rpc(
    env: Env,  # noqa: F811
    method: str,
    params: dict[str, Any] | None = None,
    *,
    who: Any = KEY_A,
    rid: int = 1,
) -> Any:
    headers = who if isinstance(who, dict) else bearer(who)
    body = {"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params else {})}
    return env.client.post(URL, json=body, headers=headers)


def call(env: Env, name: str, args: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:  # noqa: F811
    r = rpc(env, "tools/call", {"name": name, "arguments": args or {}}, **kw)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def data_of(out: dict[str, Any]) -> Any:
    assert "result" in out and not out["result"]["isError"], out
    return json.loads(out["result"]["content"][0]["text"])


def listed(env: Env, **kw: Any) -> set[str]:  # noqa: F811
    return {t["name"] for t in rpc(env, "tools/list", **kw).json()["result"]["tools"]}


# -- protocol ---------------------------------------------------------------------------------


def test_initialize_names_the_server_and_mentions_the_skills(env: Env) -> None:  # noqa: F811
    out = rpc(env, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}}).json()
    res = out["result"]
    assert res["protocolVersion"] == "2025-06-18" and res["serverInfo"]["name"] == "lm-warden"
    assert "tools" in res["capabilities"] and "resources" in res["capabilities"]
    ins = res["instructions"]
    assert "list_skills" in ins and "load_skill" in ins and "skill://" in ins
    assert "never instructions" in ins
    # An unknown protocol version is answered with one the server speaks.
    assert (
        rpc(env, "initialize", {"protocolVersion": "1999-01-01"}).json()["result"][
            "protocolVersion"
        ]
        == "2025-06-18"
    )


def test_a_notification_is_202_and_ping_is_empty(env: Env) -> None:  # noqa: F811
    r = env.client.post(
        URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=bearer(KEY_A)
    )
    assert r.status_code == 202 and r.content == b""
    assert rpc(env, "ping").json()["result"] == {}


def test_bad_json_and_unknown_method(env: Env) -> None:  # noqa: F811
    r = env.client.post(URL, content=b"{nope", headers=bearer(KEY_A))
    assert r.status_code == 400 and r.json()["error"]["code"] == -32700
    assert rpc(env, "nope/nope").json()["error"]["code"] == -32601
    assert (
        env.client.post(URL, json=[1], headers=bearer(KEY_A)).json()[0]["error"]["code"] == -32600
    )


def test_a_batch_answers_in_one_array(env: Env) -> None:  # noqa: F811
    body = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    out = env.client.post(URL, json=body, headers=bearer(KEY_A)).json()
    assert [o["id"] for o in out] == [1, 2]


# -- credentials ------------------------------------------------------------------------------


def test_the_key_the_sign_in_and_the_admin_all_open_it_and_nothing_else_does(env: Env) -> None:  # noqa: F811
    assert rpc(env, "ping").status_code == 200
    assert rpc(env, "ping", who={**env.a.headers, **csrf_header(env.client)}).status_code == 200
    session = {**env.admin, **csrf_header(env.client)}
    assert rpc(env, "ping", who=session).status_code == 200
    assert rpc(env, "ping", who=admin_headers(env)).status_code == 200
    nothing = env.client.post(
        URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=csrf_header(env.client)
    )
    assert nothing.status_code == 401
    assert rpc(env, "ping", who="vw_" + "q" * 33).status_code == 401


# -- tools/list gating ------------------------------------------------------------------------


def test_a_key_sees_lens_tools_engine_and_guides_but_no_admin_twins(env: Env) -> None:  # noqa: F811
    names = listed(env)
    assert names == LENS_TOOLS | GUIDE_TOOLS | {"engine_facts"}
    assert not any(n.startswith("lens_admin_") for n in names)


def test_every_description_says_the_results_are_data_not_instructions(env: Env) -> None:  # noqa: F811
    tools = rpc(env, "tools/list").json()["result"]["tools"]
    for t in tools:
        if t["name"] == "list_skills":
            continue
        low = t["description"].lower()
        assert "never instructions" in low or "guide written by lm warden" in low, t["name"]
        assert t["inputSchema"]["type"] == "object"
    for t in tools:
        if t["name"] in LENS_TOOLS | {"engine_facts"}:
            assert "never instructions" in t["description"], t["name"]


def _set(env: Env, **cols: int) -> None:  # noqa: F811
    sets = ", ".join(f"{k} = {v}" for k, v in cols.items())
    with sqlite3.connect(env.db) as c:
        c.execute(f"UPDATE api_tokens SET {sets} WHERE id = ?", (A,))
        c.commit()
    from app.proxy.token_cache import invalidate_all

    invalidate_all()


def test_no_lens_tools_without_the_key_permission(env: Env) -> None:  # noqa: F811
    _set(env, prompt_lens=0)
    assert listed(env) == GUIDE_TOOLS | {"engine_facts"}
    out = call(env, "lens_scorecard", {"session_id": sid(env, A)})
    assert out["error"]["code"] == -32602  # a hidden tool is unknown


def test_no_lens_tools_with_the_platform_switch_off(env: Env) -> None:  # noqa: F811
    st = env.client.app.state
    st.settings = dataclasses.replace(st.settings, prompt_lens_enabled=False)
    assert listed(env) == GUIDE_TOOLS | {"engine_facts"}


def test_no_engine_tool_without_the_engine_permission(env: Env) -> None:  # noqa: F811
    _set(env, engine_facts=0)
    assert listed(env) == LENS_TOOLS | GUIDE_TOOLS
    assert call(env, "engine_facts")["error"]["code"] == -32602


def test_an_admin_gets_the_admin_twins_with_a_key_argument(env: Env) -> None:  # noqa: F811
    tools = rpc(env, "tools/list", who=admin_headers(env)).json()["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "lens_admin_scorecard" in names and "lens_scorecard" not in names
    assert "engine_facts" not in names and GUIDE_TOOLS <= names
    trends = next(t for t in tools if t["name"] == "lens_admin_trends")
    assert (
        "key" in trends["inputSchema"]["properties"] and "key" in trends["inputSchema"]["required"]
    )


# -- tools/call ---------------------------------------------------------------------------------


def test_scorecard_equals_the_rest_route(env: Env) -> None:  # noqa: F811
    score_all(env)
    s = sid(env, A)
    got = data_of(call(env, "lens_scorecard", {"session_id": s}))
    assert got == _a(env, f"/sessions/{s}/scorecard").json()
    out = call(env, "lens_scorecard", {"session_id": s})
    assert out["result"]["structuredContent"] == got


def test_each_tool_matches_its_rest_twin(env: Env) -> None:  # noqa: F811
    score_all(env)
    s, r = sid(env, A), req("reqa")
    pairs = [
        ("lens_list_sessions", {"since": 0}, "/sessions?since=0"),
        ("lens_list_requests", {"since": 0, "limit": 5}, "/requests?since=0&limit=5"),
        ("lens_explain_request", {"request_id": r}, f"/requests/{r}"),
        ("lens_turn_diff", {"request_id": r}, f"/requests/{r}/diff"),
        ("lens_session_cost", {"session_id": s}, f"/sessions/{s}/cost"),
        ("lens_trends", {}, "/trends"),
        ("lens_versions", {}, "/versions"),
        ("lens_score", {"since": 0}, "/score?since=0"),
        ("lens_wontfix", {}, "/wontfix"),
    ]
    for tool, args, path in pairs:
        via = data_of(call(env, tool, args))
        rest = _a(env, path).json()
        assert via == rest, tool
    f = data_of(call(env, "lens_findings", {"session_id": s, "layer": "R"}))
    assert f == _a(env, f"/sessions/{s}/findings?layer=R").json()
    assert (
        data_of(call(env, "lens_findings", {"request_id": r}))
        == _a(env, f"/requests/{r}/findings").json()
    )


def test_unknown_tool_and_bad_arguments_are_32602(env: Env) -> None:  # noqa: F811
    assert call(env, "nope")["error"]["code"] == -32602
    for args in ({}, {"session_id": 5}, {"session_id": "s_x", "extra": 1}):
        assert call(env, "lens_scorecard", args)["error"]["code"] == -32602, args
    assert call(env, "lens_trends", {"by": "year"})["error"]["code"] == -32602
    both = call(env, "lens_findings", {"session_id": "s_x", "request_id": "r"})
    assert both["error"]["code"] == -32602


def test_foreign_and_unknown_sessions_are_one_error_result(env: Env) -> None:  # noqa: F811
    score_all(env)
    foreign = call(env, "lens_scorecard", {"session_id": sid(env, B)})
    unknown = call(env, "lens_scorecard", {"session_id": "s_" + "0" * 32})
    assert foreign == unknown
    assert foreign["result"]["isError"] is True


def test_a_key_never_sees_another_keys_data_through_the_tools(env: Env) -> None:  # noqa: F811
    score_all(env)
    text = json.dumps(call(env, "lens_list_requests", {"since": 0, "limit": 200})) + json.dumps(
        call(env, "lens_list_sessions", {"since": 0})
    )
    assert "reqb-fake" not in text and B not in text


def test_wontfix_through_the_tool(env: Env) -> None:  # noqa: F811
    score_all(env)
    out = data_of(call(env, "lens_wontfix", {"action": "mark", "rule": "C-01"}))
    assert out["changed"] is True and [m["rule"] for m in out["marks"]] == ["C-01"]
    assert data_of(call(env, "lens_wontfix", {}, who=KEY_B))["marks"] == []
    assert (
        call(env, "lens_wontfix", {"action": "mark", "rule": "R-09"})["result"]["isError"] is True
    )
    assert data_of(call(env, "lens_wontfix", {"action": "unmark", "rule": "C-01"}))["marks"] == []


def test_engine_facts_tool_equals_the_endpoint(env: Env) -> None:  # noqa: F811
    via = data_of(call(env, "engine_facts"))
    rest = env.client.get("/api/me/engine", headers=bearer(KEY_A)).json()
    assert via == rest and via["v"] == 1


def test_admin_tools_need_a_key_and_read_any(env: Env) -> None:  # noqa: F811
    score_all(env)
    assert call(env, "lens_admin_trends", {}, who=admin_headers(env))["error"]["code"] == -32602
    got = data_of(call(env, "lens_admin_trends", {"key": A}, who=admin_headers(env)))
    assert got == _a(env, "/trends").json()
    card = data_of(
        call(env, "lens_admin_scorecard", {"session_id": sid(env, B)}, who=admin_headers(env))
    )
    assert card["session"]["key_id"] == B


def test_no_answer_carries_the_canary(env: Env) -> None:  # A-10  # noqa: F811
    score_all(env)
    for name, args in (
        ("lens_scorecard", {"session_id": sid(env, A)}),
        ("lens_explain_request", {"request_id": req("reqa")}),
        ("lens_list_requests", {"since": 0}),
        ("lens_turn_diff", {"request_id": req("reqa")}),
    ):
        assert "CANARY" not in json.dumps(call(env, name, args)), name


def test_the_rate_limit_is_shared_with_rest(env: Env) -> None:  # noqa: F811
    st = env.client.app.state
    st.settings = dataclasses.replace(st.settings, lens_rate_key_per_min=3)
    st.lens_limiter = None
    assert _a(env, "/sessions").status_code == 200  # REST spends one
    assert call(env, "lens_list_sessions")["result"]["isError"] is False  # MCP spends one
    assert call(env, "lens_list_sessions")["result"]["isError"] is False
    r = rpc(env, "tools/call", {"name": "lens_list_sessions", "arguments": {}})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


# -- lens_wait_for_request ----------------------------------------------------------------------


def test_wait_returns_the_next_finished_request(env: Env) -> None:  # noqa: F811
    out = data_of(call(env, "lens_wait_for_request", {"timeout_s": 5, "after": 0}))
    assert out["request_id"].startswith("reqa-fake") and out["session_id"].startswith("s_")


def test_wait_times_out_with_timeout_true(env: Env, monkeypatch: Any) -> None:  # noqa: F811
    monkeypatch.setattr(mcp_mod, "_POLL_S", 0.05)
    t0 = time.monotonic()
    out = data_of(call(env, "lens_wait_for_request", {"timeout_s": 1, "after": time.time() + 3600}))
    assert out == {"timeout": True} and time.monotonic() - t0 < 5


def test_wait_wakes_when_a_request_lands(env: Env) -> None:  # noqa: F811
    from tests.unit.lens.test_routes_ui import put_rows, session_rows

    later = time.time() + 1000

    def land() -> None:
        time.sleep(0.4)
        rows = [
            dataclasses.replace(r, finished_at=later, cobs2_session=None)
            for r in session_rows(A, "reqw", turns=1)
        ]  # noqa: E501
        put_rows(env.db, rows[:1], secret=env.secret)

    import threading

    th = threading.Thread(target=land)
    th.start()
    out = data_of(call(env, "lens_wait_for_request", {"timeout_s": 5, "after": later - 1}))
    th.join()
    assert out["request_id"].startswith("reqw-fake")


def test_wait_is_capped_at_30_seconds() -> None:
    assert mcp_mod.WAIT_MAX_S == 30.0
    schema = next(t for t in mcp_mod._TOOLS if t.name == "lens_wait_for_request").properties
    assert schema["timeout_s"]["maximum"] == 30


# -- guides: skills, resources, tools agree ---------------------------------------------------


def test_list_skills_and_resources_list_agree(env: Env) -> None:  # noqa: F811
    via_tool = data_of(call(env, "list_skills"))["skills"]
    res = rpc(env, "resources/list").json()["result"]["resources"]
    assert [(s["name"], s["description"]) for s in via_tool] == [
        (r["name"], r["description"]) for r in res
    ]
    assert [r["uri"] for r in res] == [f"skill://{s['name']}" for s in via_tool]
    assert {s["name"] for s in via_tool} == {"prompt-lens", "leanprompt", "kvsig", "engine-facts"}
    assert all(r["mimeType"] == "text/markdown" for r in res)


def test_every_skill_loads_by_tool_and_by_resource_and_they_match(env: Env) -> None:  # noqa: F811
    for s in data_of(call(env, "list_skills"))["skills"]:
        via_tool = data_of(call(env, "load_skill", {"name": s["name"]}))
        via_res = rpc(env, "resources/read", {"uri": f"skill://{s['name']}"}).json()["result"][
            "contents"
        ][0]
        assert via_res["text"] == via_tool["markdown"] and len(via_tool["markdown"]) > 500
        assert via_res["uri"] == f"skill://{s['name']}" and via_res["mimeType"] == "text/markdown"
        assert via_tool["markdown"].startswith("# ")


def test_an_unknown_skill_and_resource(env: Env) -> None:  # noqa: F811
    assert call(env, "load_skill", {"name": "nope"})["result"]["isError"] is True
    out = rpc(env, "resources/read", {"uri": "skill://nope"}).json()
    assert out["error"]["code"] == -32002
    assert (
        rpc(env, "resources/read", {"uri": "file:///etc/passwd"}).json()["error"]["code"] == -32002
    )


def test_guides_follow_the_credentials_permissions(env: Env) -> None:  # noqa: F811
    _set(env, prompt_lens=0, engine_facts=0)
    names = {s["name"] for s in data_of(call(env, "list_skills"))["skills"]}
    assert names == {"leanprompt", "kvsig"}
    assert call(env, "load_skill", {"name": "prompt-lens"})["result"]["isError"] is True
    assert (
        rpc(env, "resources/read", {"uri": "skill://prompt-lens"}).json()["error"]["code"] == -32002
    )
    assert {r["name"] for r in rpc(env, "resources/list").json()["result"]["resources"]} == names


def test_every_listed_tool_is_callable_and_every_called_name_is_listed(env: Env) -> None:  # noqa: F811
    score_all(env)
    for t in rpc(env, "tools/list").json()["result"]["tools"]:
        out = call(env, t["name"], {})
        # Either it ran, or it asked for a missing argument: never "unknown tool".
        if "error" in out:
            assert "unknown tool" not in out["error"]["message"], (t["name"], out)


# -- the skill texts -----------------------------------------------------------------------------


HOSTNAME = re.compile(
    r"\b[\w-]+\.(?:h|lan|local|internal|home|corp)\b|\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+(?:\.\d+)?\b"
    # Named estate hosts are not listed here: the publish scan (publish/) blocks them in
    # the shipped skill files, and naming them in this test would ship them too.
    r"|ollama|lmwarden\.com|podwarden\.h",
    re.I,
)


def test_no_skill_text_has_an_internal_hostname_or_address() -> None:
    for k in skills.all_skills():
        assert not HOSTNAME.search(k.body), (k.name, HOSTNAME.search(k.body))
        assert "god mode" not in k.body.lower() and "godmode" not in k.body.lower()
        for url in re.findall(r"https?://[^\s)`]+", k.body):
            assert url.startswith(("https://github.com/Podwarden/", "https://<")), (k.name, url)


def test_the_skills_state_what_the_brief_asks() -> None:
    by = {k.name: k.body for k in skills.all_skills()}
    assert set(by) == {"prompt-lens", "leanprompt", "kvsig", "engine-facts"}
    assert "X-Prompt-Version" in by["prompt-lens"] and "won't fix" in by["prompt-lens"].lower()
    assert "10 minutes" in by["prompt-lens"] and "0.45 E + 0.15 C + 0.40 R" in by["prompt-lens"]
    assert "leanprompt profile detect --lmwarden" in by["leanprompt"]
    assert (
        "LMWARDEN_API_KEY" in by["leanprompt"]
        and "github.com/Podwarden/leanprompt" in by["leanprompt"]
    )
    assert "pip install kvsig" in by["kvsig"] and "npm install @kvsig/core" in by["kvsig"]
    assert "github.com/Podwarden/kvsig" in by["kvsig"]
    for field in ("engine_block_tokens", "engine_match_unit_tokens", "engine_drops_last_block"):
        assert field in by["engine-facts"]


def test_the_prompt_lens_skill_names_every_catalogue_rule() -> None:
    from app.lens.catalogue import RULES

    body = {k.name: k.body for k in skills.all_skills()}["prompt-lens"]
    missing = [r for r in RULES if f"| {r} |" not in body]
    assert missing == []


def test_the_engine_skill_names_every_field_the_endpoint_returns() -> None:
    from app.tokens.routes_me_engine import EngineModel

    body = {k.name: k.body for k in skills.all_skills()}["engine-facts"]
    assert [f for f in EngineModel.model_fields if f"`{f}`" not in body] == []


def test_the_mcp_tool_names_in_the_skills_exist() -> None:
    names = {t.name for t in mcp_mod._TOOLS}
    for k in skills.all_skills():
        for used in re.findall(r"`((?:lens_[a-z_]+|engine_facts|list_skills|load_skill))`", k.body):
            assert used in names, (k.name, used)


def test_the_prompt_lens_skill_covers_every_lens_tool() -> None:
    body = {k.name: k.body for k in skills.all_skills()}["prompt-lens"]
    for t in mcp_mod._TOOLS:
        if t.needs == "lens":
            assert f"`{t.name}`" in body, t.name


def test_asyncio_is_not_needed() -> None:
    assert asyncio.iscoroutinefunction(mcp_mod.mcp)
