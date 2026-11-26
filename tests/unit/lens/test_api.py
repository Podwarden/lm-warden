"""The Prompt Lens REST API (analysis Tasks 4.1, 4.2): ``/api/lens/me/*`` with an inference
key or a key holder's sign-in, ``/api/lens/admin/*`` with an admin credential, the per-key
``prompt_lens`` permission, rate limits and logs."""

from __future__ import annotations

import dataclasses
import logging
import re
import sqlite3
from typing import Any

from app.lens.limits import LensLimiter
from tests.conftest import bearer, csrf_header, seed_admin_token
from tests.unit.lens.test_routes_ui import (  # noqa: F401
    A,
    B,
    Env,
    env,
    req,
    score_all,
    sid,
)

KEY_A = "vw_validtoken1234567890abcdef12345"
KEY_B = "vw_secondtoken1234567890abcdef1234"
ME = "/api/lens/me"
ADM = "/api/lens/admin"


def _as_key(env: Env, plain: str, path: str, **kw: Any) -> Any:  # noqa: F811
    return env.client.get(path, headers=bearer(plain), **kw)


def _a(env: Env, path: str, **kw: Any) -> Any:  # noqa: F811
    return _as_key(env, KEY_A, ME + path, **kw)


def _admin(env: Env, path: str, **kw: Any) -> Any:  # noqa: F811
    return env.client.get(ADM + path, headers=env.admin, **kw)


def _ids(env: Env) -> dict[str, str]:  # noqa: F811
    return {
        "a_req": req("reqa"),
        "b_req": req("reqb"),
        "a_sess": sid(env, A),
        "b_sess": sid(env, B),
    }


# -- credentials -----------------------------------------------------------------------------


def test_an_inference_key_and_a_sign_in_both_reach_the_me_routes(env: Env) -> None:  # noqa: F811
    score_all(env)
    by_key = _a(env, "/sessions?since=0")
    by_forest = env.client.get(ME + "/sessions?since=0", headers=env.a.headers)
    assert by_key.status_code == by_forest.status_code == 200
    assert by_key.json() == by_forest.json() and by_key.json()["sessions"]


def test_the_lmwarden_key_header_works_like_it_does_for_inference(env: Env) -> None:  # noqa: F811
    r = env.client.get(ME + "/sessions", headers={"X-LMWarden-Key": KEY_A})
    assert r.status_code == 200


def test_an_admin_session_and_an_admin_token_get_403_on_me(env: Env) -> None:  # noqa: F811
    _, admin_token = seed_admin_token(env.db)
    assert env.client.get(ME + "/sessions", headers=env.admin).status_code == 403
    assert env.client.get(ME + "/sessions", headers=bearer(admin_token)).status_code == 403


def test_no_credential_and_a_wrong_key_are_401(env: Env) -> None:  # noqa: F811
    assert env.client.get(ME + "/sessions").status_code == 401
    assert _as_key(env, "vw_" + "q" * 33, ME + "/sessions").status_code == 401


def test_a_paused_key_is_403_and_a_rotated_away_key_is_401_at_once(env: Env) -> None:  # noqa: F811
    h = {**env.admin, **csrf_header(env.client)}
    r = env.client.patch("/api/tokens/tok1", json={"paused": True}, headers=h)
    assert r.status_code == 200, r.text
    assert _a(env, "/sessions").status_code == 403
    env.client.patch("/api/tokens/tok1", json={"paused": False}, headers=h)
    assert _a(env, "/sessions").status_code == 200
    r = env.client.post("/api/tokens/tok1/rotate", json={"grace_hours": 0}, headers=h)
    assert r.status_code == 201, r.text
    assert _a(env, "/sessions").status_code == 401  # rotated away, no grace
    new = r.json()["plaintext"]
    assert _as_key(env, new, ME + "/sessions").status_code == 200  # the successor inherits the flag


def test_a_key_without_the_prompt_lens_permission_gets_the_one_404(env: Env) -> None:  # noqa: F811
    unknown = _a(env, "/requests/nope")
    assert unknown.status_code == 404
    with sqlite3.connect(env.db) as c:
        c.execute("UPDATE api_tokens SET prompt_lens = 0 WHERE id = ?", (A,))
        c.commit()
    from app.proxy.token_cache import invalidate_all

    invalidate_all()
    for path in (
        "/sessions",
        "/requests",
        "/trends",
        "/score",
        "/wontfix",
        "/requests/" + req("reqa"),
    ):
        r = _a(env, path)
        assert r.status_code == 404, path
    assert _a(env, "/requests/" + req("reqa")).content == unknown.content
    # Engine facts are a separate permission: this key still has it.
    assert _as_key(env, KEY_A, "/api/me/engine").status_code == 200


def test_the_platform_switch_off_is_the_same_404(env: Env) -> None:  # noqa: F811
    unknown = _a(env, "/requests/nope").content
    st = env.client.app.state
    st.settings = dataclasses.replace(st.settings, prompt_lens_enabled=False)
    off = _a(env, "/requests/" + req("reqa"))
    assert off.status_code == 404 and off.content == unknown


# -- limits and logs ---------------------------------------------------------------------------


def test_the_61st_call_in_a_minute_is_429_with_retry_after(env: Env) -> None:  # noqa: F811
    for _ in range(60):
        assert _a(env, "/sessions").status_code == 200
    r = _a(env, "/sessions")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    # Another key is not limited by it.
    assert _as_key(env, KEY_B, ME + "/sessions").status_code == 200


def test_the_limit_comes_from_the_settings_and_admins_get_more(env: Env) -> None:  # noqa: F811
    st = env.client.app.state
    st.settings = dataclasses.replace(
        st.settings, lens_rate_key_per_min=2, lens_rate_admin_per_min=5
    )
    st.lens_limiter = None
    assert [_a(env, "/sessions").status_code for _ in range(3)] == [200, 200, 429]
    assert [_admin(env, "/sessions").status_code for _ in range(6)] == [200] * 5 + [429]


def test_limiter_bucket_refills_and_diffs_are_capped() -> None:
    now = [0.0]
    lim = LensLimiter(2, 10, clock=lambda: now[0])
    assert lim.check("k", admin=False) is None and lim.check("k", admin=False) is None
    wait = lim.check("k", admin=False)
    assert wait is not None and wait >= 1
    now[0] += 31
    assert lim.check("k", admin=False) is None
    lim2 = LensLimiter(100, 100)
    assert lim2.check("k", admin=False, diff=True) is None
    assert lim2.check("k", admin=False, diff=True) is None
    assert lim2.check("k", admin=False, diff=True) is not None  # a third at once
    lim2.diff_done("k")
    assert lim2.check("k", admin=False, diff=True) is None


def test_logs_carry_route_key_hash_status_and_time_only(env: Env, caplog: Any) -> None:  # noqa: F811
    caplog.set_level(logging.INFO, logger="app.lens.api")
    _a(env, "/requests/" + req("reqa") + "?against=zzz")
    lines = [r.getMessage() for r in caplog.records if r.name == "app.lens.api"]
    assert lines, "no log line"
    for line in lines:
        assert re.fullmatch(r"lens-api route=\S+ key=[0-9a-f]{8} status=\d+ ms=\d+", line), line
        assert KEY_A not in line and A not in line and req("reqa") not in line and "zzz" not in line


# -- isolation (A-1) -----------------------------------------------------------------------------


def test_every_me_route_answers_own_200_and_foreign_unknown_one_404(env: Env) -> None:  # noqa: F811
    score_all(env)
    i = _ids(env)
    routes = [
        "/requests/{}",
        "/requests/{}/findings",
        "/requests/{}/diff",
    ]
    for tpl in routes:
        assert _a(env, tpl.format(i["a_req"])).status_code == 200, tpl
        foreign = _a(env, tpl.format(i["b_req"]))
        unknown = _a(env, tpl.format("nope"))
        assert foreign.status_code == unknown.status_code == 404, tpl
        assert foreign.content == unknown.content, tpl
    for tpl in (
        "/sessions/{}/scorecard",
        "/sessions/{}/cost",
        "/sessions/{}/findings",
        "/sessions/{}/report",
    ):
        assert _a(env, tpl.format(i["a_sess"])).status_code == 200, tpl
        foreign = _a(env, tpl.format(i["b_sess"]))
        unknown = _a(env, tpl.format("s_" + "0" * 32))
        assert foreign.status_code == unknown.status_code == 404, tpl
        assert foreign.content == unknown.content, tpl


def test_a_diff_against_a_foreign_request_is_the_same_404(env: Env) -> None:  # noqa: F811
    ok = _a(env, f"/requests/{req('reqa')}/diff?against={req('reqa', 8)}")
    assert ok.status_code == 200
    foreign = _a(env, f"/requests/{req('reqa')}/diff?against={req('reqb', 8)}")
    unknown = _a(env, f"/requests/{req('reqa')}/diff?against=nope")
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.content == unknown.content


def test_a_key_parameter_is_refused_on_every_me_route(env: Env) -> None:  # noqa: F811
    for path in ("/requests", "/sessions", "/trends", "/versions", "/score", "/wontfix"):
        r = _a(env, f"{path}?key={B}")
        assert r.status_code == 422, path


def test_key_a_never_sees_key_b(env: Env) -> None:  # noqa: F811
    score_all(env)
    body = _a(env, "/requests?since=0&limit=200").text + _a(env, "/sessions?since=0").text
    assert B not in body and "reqb-fake" not in body
    ids = {r["request_id"] for r in _a(env, "/requests?since=0&limit=200").json()["requests"]}
    assert ids and all(i.startswith("reqa-") for i in ids)


def test_no_me_answer_holds_a_fleet_figure(env: Env) -> None:  # A-14  # noqa: F811
    score_all(env)
    i = _ids(env)
    paths = [
        "/sessions?since=0", "/requests", "/trends", "/versions", "/score?since=0", "/wontfix",
        f"/sessions/{i['a_sess']}/scorecard", f"/sessions/{i['a_sess']}/cost",
        f"/requests/{i['a_req']}", f"/sessions/{i['a_sess']}/report",
    ]  # fmt: skip
    for p in paths:
        r = _a(env, p)
        assert r.status_code == 200, p
        low = r.text.lower()
        for word in ("percentile", "leaderboard", '"rank"', '"fleet"', "distribution"):
            assert word not in low, (p, word)
    assert _a(env, "/fleet").status_code in (404, 405)


def test_the_fleet_route_is_the_admins_alone(env: Env) -> None:  # noqa: F811
    assert _admin(env, "/fleet").status_code == 200
    assert env.client.get(ADM + "/fleet", headers=bearer(KEY_A)).status_code == 401
    assert env.client.get(ADM + "/fleet", headers=env.a.headers).status_code == 401


# -- canary and shape ----------------------------------------------------------------------------


def test_no_answer_carries_the_canary(env: Env) -> None:  # A-10  # noqa: F811
    score_all(env)
    i = _ids(env)
    for p in (f"/requests/{i['a_req']}", f"/requests/{i['a_req']}/diff", "/requests",
              f"/sessions/{i['a_sess']}/scorecard", f"/sessions/{i['a_sess']}/report?format=html"):  # fmt: skip
        assert "CANARY" not in _a(env, p).text, p


def test_the_requests_list_and_one_request(env: Env) -> None:  # noqa: F811
    r = _a(env, "/requests?since=0&limit=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["requests"]) == 5 and body["next_before"] is not None
    page2 = _a(env, f"/requests?since=0&limit=5&before={body['next_before']}").json()
    assert not {x["request_id"] for x in page2["requests"]} & {
        x["request_id"] for x in body["requests"]
    }
    one = _a(env, f"/requests/{req('reqa')}").json()
    assert one["request_id"] == req("reqa") and one["turn"] == 9
    assert {f["rule"] for f in one["findings"]} >= {"E-01", "R-04"}
    assert one["session_id"].startswith("s_")


def test_a_diff_names_the_first_broken_part_without_text(env: Env) -> None:  # noqa: F811
    d = _a(env, f"/requests/{req('reqa')}/diff").json()  # against: the parent turn
    assert d["against"] == req("reqa", 8) and d["continues"] is True
    assert d["first_change"]["part"] == "sys" and d["first_change"]["line"] == 2
    assert set(d["tokens"]) == {"prompt_delta", "cached_delta"}


def test_session_findings_filter_by_layer_and_rule(env: Env) -> None:  # noqa: F811
    score_all(env)
    s = sid(env, A)
    allf = _a(env, f"/sessions/{s}/findings").json()["findings"]
    assert {"E-01", "R-04"} <= {f["rule"] for f in allf}
    only_r = _a(env, f"/sessions/{s}/findings?layer=R").json()["findings"]
    assert only_r and {f["layer"] for f in only_r} == {"R"}
    one = _a(env, f"/sessions/{s}/findings?rule=E-01").json()["findings"]
    assert {f["rule"] for f in one} == {"E-01"}
    cost = _a(env, f"/sessions/{s}/cost").json()
    assert set(cost) == {"cost", "turns"} and len(cost["turns"]) == 24


def test_score_says_not_scored_under_50_calls_and_gives_no_number(env: Env) -> None:  # noqa: F811
    r = _a(env, "/score?since=0").json()
    assert r["scored"] is False and r["calls"] == 24 and "leanprompt" not in r
    assert "reason" in r


def test_score_returns_leanprompt_fields_with_enough_calls(env: Env) -> None:  # noqa: F811
    from tests.unit.lens.test_routes_ui import put_rows, session_rows

    for n in range(3):  # 3 more sessions of 24 turns: 96 calls in all
        rows = session_rows(A, f"reqx{n}")
        rows = [
            dataclasses.replace(r, cobs2_session=f"more-{n}", cobs2_conv=f"c-{n}", session_key=f"more-{n}")
            for r in rows
        ]  # fmt: skip
        put_rows(env.db, rows, secret=env.secret)
    r = _a(env, "/score?since=0").json()
    assert r["scored"] is True and r["calls"] == 96 and r["kind"] == "measured"
    assert set(r) >= {"prefill_efficiency", "stability", "generation_waste", "leanprompt", "calls"}
    assert 0 <= r["leanprompt"] <= 100
    for k in ("prefill_efficiency", "stability", "generation_waste"):
        assert 0.0 <= r[k] <= 1.0


# -- the same function as the page --------------------------------------------------------------


def test_trends_versions_compare_and_wontfix_match_the_page_routes(env: Env) -> None:  # noqa: F811
    score_all(env)
    page = env.a.get("/api/me/prompt-lens/trends").json()
    assert _a(env, "/trends").json() == page
    assert _a(env, "/versions").json() == env.a.get("/api/me/prompt-lens/versions").json()
    assert _a(env, "/wontfix").json() == env.a.get("/api/me/prompt-lens/wontfix").json()
    card = env.a.get(f"/api/me/prompt-lens/sessions/{sid(env, A)}/scorecard").json()
    assert _a(env, f"/sessions/{sid(env, A)}/scorecard").json() == card


def test_wontfix_through_the_api_marks_and_unmarks_for_the_key_only(env: Env) -> None:  # noqa: F811
    score_all(env)
    h = bearer(KEY_A)
    r = env.client.put(ME + "/wontfix/C-01", json={"version": None}, headers=h)
    assert r.status_code == 200 and r.json()["changed"] is True
    assert [m["rule"] for m in _a(env, "/wontfix").json()["marks"]] == ["C-01"]
    assert _as_key(env, KEY_B, ME + "/wontfix").json()["marks"] == []
    assert (
        env.client.put(ME + "/wontfix/R-09", json={"version": None}, headers=h).status_code == 422
    )
    r = env.client.delete(ME + "/wontfix/C-01", headers=h)
    assert r.status_code == 200 and r.json()["marks"] == []


def test_admin_twins_name_a_key_and_see_any(env: Env) -> None:  # noqa: F811
    score_all(env)
    assert _admin(env, f"/trends?key={A}").status_code == 200
    assert _admin(env, "/trends").status_code == 422  # a key is needed
    both = _admin(env, "/sessions?since=0").json()["sessions"]
    assert {s["key_id"] for s in both} == {A, B}
    r = _admin(env, f"/sessions/{sid(env, B)}/scorecard")
    assert r.status_code == 200
    assert _admin(env, f"/requests/{req('reqb')}").status_code == 200
    # The admin's key-scoped reads and the key holder's agree.
    assert _admin(env, f"/versions?key={A}").json() == _a(env, "/versions").json()


def test_the_admin_routes_refuse_every_key_credential(env: Env) -> None:  # noqa: F811
    for h in (bearer(KEY_A), env.a.headers):
        assert env.client.get(ADM + "/sessions", headers=h).status_code == 401
