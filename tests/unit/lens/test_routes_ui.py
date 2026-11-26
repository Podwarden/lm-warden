"""Prompt Lens request findings routes (analysis Task 1.8).

``GET /api/me/prompt-lens/requests/{req_id}/findings`` (the key holder's own
requests only, decided on the server, A-1) and
``GET /api/admin/prompt-lens/requests/{req_id}/findings`` (any key). Two keys,
A (``tok1``) and B (``tok2``); their sessions are seeded straight into
``request_history`` from the lens seeds (codes, counts and keyed hashes only).
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.lens.probes import SPAN_CLASSES
from app.sessions import store
from app.sessions.refs import sess_ref
from app.sessions.store import Row
from tests.conftest import csrf_header, jwt_login
from tests.unit.lens import seed
from tests.unit.proxy.test_prompt_lens_forward import _ready_gm, _second_key
from tests.unit.stats.test_forest_routes import _forest_login

ME = "/api/me/prompt-lens/requests/{}/findings"
ADMIN = "/api/admin/prompt-lens/requests/{}/findings"
A, B = "tok1", "tok2"
#: The turn of A's session that has an E-01 (sys line 2, a time) and an R-04.
TURN = 9

_ATTR = {
    "cobs2_divergence_cause": "divergence_cause",
    "cobs2_divergence_cause_own": "divergence_cause_own",
    "cobs2_shape": "shape",
    "cobs2_toolset": "toolset",
}


def _value(v: Any) -> Any:
    if isinstance(v, dict):
        return json.dumps(v)
    if isinstance(v, tuple):
        return json.dumps([list(e) for e in v])
    if isinstance(v, bool):
        return int(v)
    return v


def put_rows(db: Path, rows: list[Row], *, secret: str) -> None:
    """Store ``rows`` as the hot path would have (every column ``store`` reads)."""
    with sqlite3.connect(db) as c:
        for r in rows:
            d = {col: _value(getattr(r, _ATTR.get(col, col))) for col in store._COLS}
            d["started_iso"] = "2023-11-14T22:13:20Z"
            d["orphan"] = 0
            if r.cobs2_session:
                d["cobs2_sess_ref"] = sess_ref(secret, r.token_id, r.cobs2_session)
            c.execute(
                f"INSERT INTO request_history({', '.join(d)}) VALUES ({', '.join('?' * len(d))})",
                tuple(d.values()),
            )
        c.commit()


def put_toolset(db: Path, token: str, names: dict[str, str], *, tools: int) -> None:
    ents = seed.seeded_toolsets("bad_schema", tools=tools)[seed.TOOLSET]
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT OR REPLACE INTO cobs2_toolsets(digest, tools, last_seen) VALUES (?, ?, ?)",
            (seed.TOOLSET, json.dumps([list(e) for e in ents]), seed.T0),
        )
        for h, name in names.items():
            c.execute(
                "INSERT OR REPLACE INTO session_tools(token_id, tool_hmac, name, family, size,"
                " first_seen, last_seen) VALUES (?, ?, ?, 'fn', 100, ?, ?)",
                (token, h, name, seed.T0, seed.T0),
            )
        c.commit()


def session_rows(token: str, prefix: str, **over: Any) -> list[Row]:
    """A's faulty session: bad tool arguments (R-04 on turns 7, 9, 12, 14) and a
    system prompt line that changes every turn (E-01). Ids get ``prefix``."""
    rows = seed.seeded_rows("bad_schema", tools=4, **over)
    rows = seed._sys_line(rows, seed._params("timestamp_line", {}))

    def rid(x: str | None) -> str | None:
        return None if x is None else x.replace("req-fake", f"{prefix}-fake")

    return [
        replace(r, id=rid(r.id) or "", cobs2_parent_req=rid(r.cobs2_parent_req), token_id=token)
        for r in rows
    ]


def req(prefix: str, n: int = TURN) -> str:
    return f"{prefix}-fake-{n:04d}"


@dataclasses.dataclass
class As:
    client: Any = dataclasses.field(repr=False)
    #: Never in a failure message: a sign-in token.
    headers: dict[str, str] = dataclasses.field(repr=False)
    req: str = ""

    def get(self, url: str) -> Any:
        return self.client.get(url, headers=self.headers)


@dataclasses.dataclass
class Env:
    client: Any = dataclasses.field(repr=False)
    db: Path
    secret: str = dataclasses.field(repr=False)
    a: As
    b: As
    admin: dict[str, str] = dataclasses.field(repr=False)

    def as_admin(self, url: str) -> Any:
        return self.client.get(url, headers=self.admin)

    def findings(self, who: As, rid: str | None = None) -> Any:
        r = who.get(ME.format(rid or who.req))
        assert r.status_code == 200, r.text
        return r


@pytest.fixture
def env(tmp_data_dir: Path, client: Any) -> Env:
    plain_a = _ready_gm(client, tmp_data_dir)
    plain_b = _second_key(tmp_data_dir)
    db = tmp_data_dir / "vllm-warden.db"
    secret = client.app.state.settings.cookie_secret
    put_toolset(db, A, {seed.tool_hash(i): f"a_tool_{i}" for i in range(4)}, tools=4)
    put_rows(db, session_rows(A, "reqa"), secret=secret)
    put_rows(db, session_rows(B, "reqb", start=seed.T0 + 7), secret=secret)
    ha, hb = _forest_login(client, plain_a), _forest_login(client, plain_b)
    return Env(
        client,
        db,
        secret,
        As(client, ha, req("reqa")),
        As(client, hb, req("reqb")),
        jwt_login(client),
    )


# -- isolation (A-1) -----------------------------------------------------------------


def test_key_holder_gets_own_findings_and_404_for_foreign(env: Env) -> None:
    a, b = env.a, env.b
    own = a.get(ME.format(a.req)).status_code
    assert own == 200
    foreign = a.get(ME.format(b.req))
    unknown = a.get(ME.format("nope"))
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.content == unknown.content  # one identical 404


def test_off_is_the_same_404(env: Env) -> None:
    unknown = env.a.get(ME.format("nope")).content
    env.client.app.state.settings = dataclasses.replace(
        env.client.app.state.settings, prompt_lens_enabled=False
    )
    off = env.a.get(ME.format(env.a.req))
    assert off.status_code == 404 and off.content == unknown


def test_admin_session_refused_on_me_route(env: Env) -> None:
    status = env.as_admin(ME.format(env.a.req)).status_code
    assert status == 403


def test_key_holder_refused_on_admin_route(env: Env) -> None:
    status = env.a.get(ADMIN.format(env.a.req)).status_code
    assert status == 401


def test_admin_reads_any_keys_findings(env: Env) -> None:
    for rid in (env.a.req, env.b.req):
        r = env.as_admin(ADMIN.format(rid))
        assert r.status_code == 200, r.text
        assert {f["rule"] for f in r.json()["findings"]} >= {"E-01", "R-04"}
    status = env.as_admin(ADMIN.format("nope")).status_code
    assert status == 404


def test_cross_key_seed_does_not_change_findings(env: Env) -> None:  # A-1
    before = env.findings(env.a).content
    # B: the same session id, toolset, tools and failures as A, on the same seconds.
    twin = [replace(r, cobs2_session=seed.SESSION) for r in session_rows(B, "reqc")]
    put_rows(env.db, twin, secret=env.secret)
    put_toolset(env.db, B, {seed.tool_hash(i): f"b_secret_{i}" for i in range(4)}, tools=4)
    with sqlite3.connect(env.db) as c:
        c.execute(
            "INSERT INTO guard_events(ts, token_id, request_id, action, rule_id, role, relayed)"
            " VALUES (?, ?, ?, 'sanitized', 'b_rule', 'system', 0)",
            (seed.T0 + 250, B, req("reqc")),
        )
        c.commit()
    after = env.findings(env.a).content
    assert after == before
    assert b"b_secret" not in after and b"b_rule" not in after


# -- the answer ------------------------------------------------------------------------


def test_findings_carry_parts_hashes_and_own_names(env: Env) -> None:
    body = env.findings(env.a).json()
    assert set(body) >= {"findings", "names"}
    by = {f["rule"]: f for f in body["findings"]}
    e01 = by["E-01"]
    assert e01["part"] == "sys"
    assert e01["counts"]["line"] == 2
    assert SPAN_CLASSES[e01["counts"]["class"]] == "time"
    r04 = by["R-04"]
    assert r04["part"] == "call:0"
    assert r04["tool"] == seed.tool_hash(3)  # a hash, never a name
    assert body["names"] == {seed.tool_hash(3): "a_tool_3"}
    for f in body["findings"]:
        assert TURN in f["turns"]
        assert set(f) >= {"rule", "turns", "counts", "tool", "cause", "saving", "confidence"}


def test_a_request_without_findings_answers_an_empty_list(env: Env) -> None:
    body = env.findings(env.a, req("reqa", 1)).json()
    assert body["findings"] == []


def test_misrouted_never_reaches_a_key_holder(env: Env) -> None:
    with sqlite3.connect(env.db) as c:
        c.execute(
            "UPDATE request_history SET lens_cause = 'misrouted', lens_break = NULL"
            " WHERE id IN (?, ?)",
            (req("reqa", 10), req("reqa", 11)),
        )
        c.commit()
    assert b"misrouted" not in env.findings(env.a, req("reqa", 10)).content


def test_r09_role_comes_from_the_guard_log(env: Env) -> None:
    rid = req("reqa", 5)
    with sqlite3.connect(env.db) as c:
        c.execute(
            "UPDATE request_history SET guard = ? WHERE id = ?",
            (json.dumps({"m": "sanitize", "b": 0, "f": [["fake_secret_rule", 2, 2]], "r": 0}), rid),
        )
        c.execute(
            "INSERT INTO guard_events(ts, token_id, request_id, action, rule_id, role, kind,"
            " relayed) VALUES (?, ?, ?, 'sanitized', 'fake_secret_rule', 'system', NULL, 0)",
            (seed.T0 + 125, A, rid),
        )
        c.commit()
    r09 = [f for f in env.findings(env.a, rid).json()["findings"] if f["rule"] == "R-09"]
    assert len(r09) == 1
    assert r09[0]["cause"] == "secret_in_system" and r09[0]["part"] == "sys"
    assert r09[0]["guard"] == {"rules": ["fake_secret_rule"], "where": ["system"]}


# -- the frontend's copy fixture -------------------------------------------------------

FIXTURE = (
    Path(__file__).parents[3] / "frontend" / "tests" / "unit" / "prompt-lens-findings-fixtures.json"
)
#: fixture key -> (seed fault, turn, extra context): the turns the vitest pins.
FIXTURE_CASES: dict[str, tuple[str, int, dict[str, Any]]] = {
    "path_line": ("path_line", 5, {}),
    "tools_reordered": ("tools_reordered", 13, {}),
    "unused_tools": ("unused_tools", 10, {"ts": {"tools": 12, "tool_chars": 1600}}),
    "loop": ("loop", 8, {}),
    "empty_reply_template": ("empty_reply_template", 11, {}),
    "truncating_max_tokens": ("truncating_max_tokens", 15, {"mml": 32768}),
    "evicted_pauses": ("evicted_pauses", 15, {}),
    "no_marks_anthropic": ("no_marks_anthropic", 10, {}),
}


def _answer(rows: list[Row], fault: str, n: int, kw: dict[str, Any]) -> dict[str, Any]:
    from app.lens.analyse import AnalyseContext, analyse_session, findings_for_request
    from app.lens.service import part_of

    ctx = AnalyseContext(
        toolsets=seed.seeded_toolsets(fault, **kw.get("ts", {})),
        recent_calls={},
        prefill_rate=None,
        max_model_len=kw.get("mml"),
        route_tool_limit=None,
    )
    a = analyse_session(rows, own=True, ctx=ctx)
    rid = f"req-fake-{n:04d}"
    t = next(t for t in a.turns if t.req_id == rid)
    out = []
    for f in findings_for_request(a, rid):
        d = f.to_json()
        d["part"] = part_of(f, t, None)
        d["guard"] = None
        out.append(d)
    return {"turn": t.n, "findings": out}


def test_frontend_fixture_is_the_analysis_answer() -> None:
    """frontend/tests/unit/prompt-lens-findings-fixtures.json pins the UI copy
    (prompt-lens-findings.test.tsx); it must stay what the server answers."""
    fx = json.loads(FIXTURE.read_text())
    rows = seed._sys_line(
        seed.seeded_rows("bad_schema", tools=4), seed._params("timestamp_line", {})
    )
    got = {"turn9": _answer(rows, "bad_schema", TURN, {"ts": {"tools": 4}})}
    for key, (fault, n, kw) in FIXTURE_CASES.items():
        got[key] = _answer(seed.seeded_rows(fault), fault, n, kw)
    want = {k: {"turn": v["turn"], "findings": v["findings"]} for k, v in fx.items() if k[0] != "_"}
    assert got == want


# ---- I4: a call finding sits on the call the stored verdict names ------------------


def _r04(h: str) -> Any:
    from app.lens.catalogue import Finding, Saving

    return Finding("R-04", (7,), {"bad": 2}, h, None, Saving(0, None, None, None), "high")


def test_call_part_uses_the_stored_call_index() -> None:
    """Two parallel calls of one tool, the second bad: the chip goes on call:1."""
    from app.lens.service import part_of
    from app.lens.turns import turns_of

    t = turns_of(seed.seeded_rows("bad_schema"), own=True)[6]
    h = seed.tool_hash(3)
    two = replace(
        t, tools_out=(("fn", h, "a1"), ("fn", h, "a2")), calls_bad=((1, h, "missing_arg"),)
    )
    assert part_of(_r04(h), two, None) == "call:1"
    # An old row ([hash8, code]: k unknown) falls back to the tool's first call.
    old = replace(two, calls_bad=((None, h, "missing_arg"),))
    assert part_of(_r04(h), old, None) == "call:0"
    # Another tool's bad call never places this finding.
    other = replace(two, calls_bad=((1, "ffffffff", "missing_arg"),))
    assert part_of(_r04(h), other, None) is None


# ---- I5: E-01 across sessions, from the key's own rows ------------------------------


def _one_turn(token: str, rid: str, session: str, at: float) -> Row:
    """A new session's first turn whose system prompt changed at line 4 (a
    path) against the key's recent head (spec 4.5)."""
    r = seed.seeded_rows("clean", turns=1)[0]
    return replace(
        r,
        id=rid,
        token_id=token,
        session_key=session,
        cobs2_session=session,
        cobs2_conv=f"conv-{session}",
        cobs2_node=f"node-{rid}",
        started_at=at,
        finished_at=at + 1,
        lens_break="sys",
        lens_cause="sys_changed",
        lens_span_class="path",
        lens_span_line=4,
        lens_lost_tokens=3_000,
        lens_cf_tokens=3_000,
    )


def _e01(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in body["findings"] if f["rule"] == "E-01"]


def test_e01_counts_the_keys_own_earlier_sessions(env: Env) -> None:
    """Two sessions of key A, each one turn with a changed path at line 4: the
    second has E-01 (sessions 2). Key B's identical rows never count for A."""
    t = seed.T0 + 50_000
    # B's identical earlier session, then A's first: no earlier session of A.
    put_rows(env.db, [_one_turn(B, "x5b-fake-0001", "sess-x5b-1", t)], secret=env.secret)
    put_rows(env.db, [_one_turn(A, "x5a-fake-0002", "sess-x5a-2", t + 600)], secret=env.secret)
    for rid in ("x5a-fake-0002",):
        assert _e01(env.findings(env.a, rid).json()) == []
        r = env.as_admin(ADMIN.format(rid))
        assert r.status_code == 200 and _e01(r.json()) == []
    # A's own earlier session with the same line: E-01 on the later one.
    put_rows(env.db, [_one_turn(A, "x5a-fake-0001", "sess-x5a-1", t + 60)], secret=env.secret)
    (f,) = _e01(env.findings(env.a, "x5a-fake-0002").json())
    assert (f["counts"]["line"], f["counts"]["sessions"]) == (4, 2)
    assert SPAN_CLASSES[f["counts"]["class"]] == "path"
    (adm,) = _e01(env.as_admin(ADMIN.format("x5a-fake-0002")).json())
    assert adm["counts"]["sessions"] == 2
    # Another line in the earlier session does not count for line 4.
    put_rows(
        env.db,
        [replace(_one_turn(A, "x5a-fake-0003", "sess-x5a-3", t + 300), lens_span_line=7)],
        secret=env.secret,
    )
    (g,) = _e01(env.findings(env.a, "x5a-fake-0002").json())
    assert g["counts"]["sessions"] == 2


# -- scores and the scorecard (analysis Task 2.4) ----------------------------------------

ME_LIST = "/api/me/prompt-lens/sessions?since=0"
ME_CARD = "/api/me/prompt-lens/sessions/{}/scorecard"
ADMIN_LIST = "/api/admin/prompt-lens/sessions?since=0"
ADMIN_CARD = "/api/admin/prompt-lens/sessions/{}/scorecard"
SHORT = "sess-short-0001"


def sid(env: Env, token: str, sess: str = seed.SESSION) -> str:
    from app.sessions.refs import session_id

    return session_id(sess_ref(env.secret, token, sess))


def put_short_session(env: Env, token: str, turns: int = 3) -> str:
    rows = [
        replace(r, cobs2_session=SHORT, cobs2_conv="conv-short", session_key=SHORT)
        for r in session_rows(token, "reqs", turns=turns)
    ]
    put_rows(env.db, rows, secret=env.secret)
    return sid(env, token, SHORT)


def score_all(env: Env) -> int:
    from app.lens.scorer_job import ScorerJob

    st = env.client.app.state
    st.settings = dataclasses.replace(st.settings, lens_quiet_s=600.0)
    return ScorerJob(st).run_once_sync(seed.T0 + 24 * 30 + 3600)


def test_scorecard_shape_for_a_seeded_session(env: Env) -> None:
    r = env.a.get(ME_CARD.format(sid(env, A)))
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) >= {"session", "scores", "top", "findings", "turns", "cost", "weights"}
    assert d["score_version"] == 1 and d["weights"] == {"E": 0.45, "C": 0.15, "R": 0.4}
    assert d["session"]["turns"] == 24 and d["session"]["route"] == "local"
    assert len(d["turns"]) == 24 and d["review"] is None
    assert d["turns"][8]["req_id"] == req("reqa", 9) and d["turns"][8]["n"] == 9
    sc = d["scores"]
    assert set(sc) == {"overall", "E", "C", "R"}
    assert all(sc[k]["scored"] for k in sc) and sc["E"]["value"] < 100
    assert 0 <= sc["E"]["lo"] <= sc["E"]["value"] <= sc["E"]["hi"] <= 100
    assert sc["overall"]["inputs"]["from"] == ["E", "C", "R"]
    rules = {f["rule"] for f in d["findings"]}
    assert {"E-01", "R-04"} <= rules
    assert {f["rule"] for f in d["top"]} <= rules and len(d["top"]) <= 3
    # The unit is tokens or seconds with no card: no money anywhere.
    assert all(f["saving"]["money"] is None for f in d["findings"])
    assert d["cost"]["card"] is None


def test_not_scored_layers_carry_a_reason_and_no_number(env: Env) -> None:  # A-7
    ref = put_short_session(env, A)
    d = env.a.get(ME_CARD.format(ref)).json()
    e = d["scores"]["E"]
    assert e == {"layer": "E", "scored": False, "turns": 2, "reason": "2 of 3 continuing turns"}
    assert "value" not in d["scores"]["R"] and "value" not in d["scores"]["overall"]
    assert d["not_scored"]["E"] == "2 of 3 continuing turns"
    assert d["not_scored"]["R"] == "3 of 5 assistant turns"
    assert d["scores"]["overall"]["scored"] is False


def test_scorecard_foreign_and_unknown_are_one_404(env: Env) -> None:
    foreign = env.a.get(ME_CARD.format(sid(env, B)))
    unknown = env.a.get(ME_CARD.format("s_" + "0" * 32))
    junk = env.a.get(ME_CARD.format("nope"))
    assert foreign.status_code == unknown.status_code == junk.status_code == 404
    assert foreign.content == unknown.content == junk.content


def test_scorecard_admin_reads_any_key_and_key_holder_is_refused_there(env: Env) -> None:
    for token in (A, B):
        r = env.as_admin(ADMIN_CARD.format(sid(env, token)))
        assert r.status_code == 200, r.text
    assert env.as_admin(ADMIN_CARD.format("s_" + "0" * 32)).status_code == 404
    assert env.a.get(ADMIN_CARD.format(sid(env, A))).status_code == 401
    assert env.as_admin(ME_CARD.format(sid(env, A))).status_code == 403
    assert env.as_admin(ME_LIST).status_code == 403


def test_only_the_admins_scorecard_names_the_key(env: Env) -> None:
    mine = env.a.get(ME_CARD.format(sid(env, A))).json()["session"]
    assert "key_id" not in mine
    assert env.as_admin(ADMIN_CARD.format(sid(env, A))).json()["session"]["key_id"] == A


def test_scorecard_is_the_same_for_the_key_holder_and_the_admin(env: Env) -> None:
    mine = env.a.get(ME_CARD.format(sid(env, A))).json()["scores"]
    admin = env.as_admin(ADMIN_CARD.format(sid(env, A))).json()["scores"]
    assert mine == admin  # same own-lens figures, so a mark can not move a board (D-3)


def test_scorecard_ignores_other_keys_traffic(env: Env) -> None:  # A-1
    before = env.a.get(ME_CARD.format(sid(env, A))).content
    twin = [replace(r, cobs2_session=seed.SESSION) for r in session_rows(B, "reqc")]
    put_rows(env.db, twin, secret=env.secret)
    with sqlite3.connect(env.db) as c:
        c.execute(
            "INSERT INTO guard_events(ts, token_id, request_id, action, rule_id, role, relayed)"
            " VALUES (?, ?, ?, 'sanitized', 'b_rule', 'system', 0)",
            (seed.T0 + 250, B, req("reqc")),
        )
        c.commit()
    assert env.a.get(ME_CARD.format(sid(env, A))).content == before


def _name_of(env: Env, token_id: str) -> str:
    with sqlite3.connect(env.db) as c:
        return str(c.execute("SELECT name FROM api_tokens WHERE id = ?", (token_id,)).fetchone()[0])


def test_session_list_names_the_key_and_tolerates_a_deleted_one(env: Env) -> None:
    score_all(env)
    allr = env.as_admin(ADMIN_LIST).json()["sessions"]
    assert {s["key_id"]: s["key_name"] for s in allr} == {
        A: _name_of(env, A),
        B: _name_of(env, B),
    }
    with sqlite3.connect(env.db) as c:
        c.execute("DELETE FROM api_tokens WHERE id = ?", (B,))
        c.commit()
    gone = {s["key_id"]: s["key_name"] for s in env.as_admin(ADMIN_LIST).json()["sessions"]}
    assert gone == {A: _name_of(env, A), B: None}


def test_session_list_is_own_and_ignores_a_key_parameter(env: Env) -> None:  # A-1
    assert score_all(env) == 2
    plain = env.a.get(ME_LIST)
    assert plain.status_code == 200
    got = plain.json()["sessions"]
    assert [s["session_id"] for s in got] == [sid(env, A)]
    assert set(got[0]) >= {"overall", "e", "c", "r", "confidence", "turns", "reasons"}
    assert "key_id" not in got[0]
    assert got[0]["key_name"] == _name_of(env, A)  # its own key's name, never another's
    assert _name_of(env, B) not in plain.text
    assert env.a.get(ME_LIST + f"&key={B}").content == plain.content  # ignored, still its own


def test_session_list_admin_sees_all_and_can_filter(env: Env) -> None:
    score_all(env)
    allr = env.as_admin(ADMIN_LIST).json()["sessions"]
    assert {s["key_id"] for s in allr} == {A, B}
    only = env.as_admin(ADMIN_LIST + f"&key={B}").json()["sessions"]
    assert [s["key_id"] for s in only] == [B]
    page = env.as_admin(ADMIN_LIST + "&limit=1").json()
    assert len(page["sessions"]) == 1 and page["next_cursor"]
    rest = env.as_admin(ADMIN_LIST + f"&limit=1&cursor={page['next_cursor']}").json()
    assert (
        len(rest["sessions"]) == 1
        and rest["sessions"][0]["session_id"] != page["sessions"][0]["session_id"]
    )


def _stored(env: Env) -> list[tuple[str, str, str]]:
    with sqlite3.connect(env.db) as c:
        return [
            (str(k), str(m), str(h))
            for k, m, h in c.execute("SELECT key_id, model, harness FROM lens_session_scores")
        ]


def test_session_list_filters_by_several_keys_models_and_clients(env: Env) -> None:
    score_all(env)
    (ka, model_a, harness_a), (kb, _, _) = sorted(_stored(env), key=lambda t: t[0] != A)
    assert (ka, kb) == (A, B)
    both = env.as_admin(ADMIN_LIST + f"&key={A}&key={B}").json()["sessions"]
    assert {s["key_id"] for s in both} == {A, B}
    only = env.as_admin(ADMIN_LIST + f"&key={A}&key={A}").json()["sessions"]
    assert {s["key_id"] for s in only} == {A}
    # Model and client: any of the listed; a value nobody has matches nothing.
    by_model = env.as_admin(ADMIN_LIST + f"&model={model_a}&model=nope").json()["sessions"]
    assert by_model and all(s["model"] == model_a for s in by_model)
    assert env.as_admin(ADMIN_LIST + "&model=nope").json()["sessions"] == []
    by_client = env.as_admin(ADMIN_LIST + f"&harness={harness_a}").json()["sessions"]
    assert by_client and all(s["harness"] == harness_a for s in by_client)
    assert env.as_admin(ADMIN_LIST + "&harness=nope").json()["sessions"] == []
    # They combine (and).
    assert env.as_admin(ADMIN_LIST + f"&key={A}&harness=nope").json()["sessions"] == []


def test_key_holder_filters_never_reach_another_key(env: Env) -> None:  # A-1
    score_all(env)
    own = env.a.get(ME_LIST).json()["sessions"]
    assert len(own) == 1
    # key= (one or several) is ignored; model and harness narrow its own only.
    assert env.a.get(ME_LIST + f"&key={B}&key={A}").json()["sessions"] == own
    assert env.a.get(ME_LIST + f"&key={B}&harness={own[0]['harness']}").json()["sessions"] == own
    assert env.a.get(ME_LIST + "&harness=nope").json()["sessions"] == []
    assert env.a.get(ME_LIST + "&model=nope").json()["sessions"] == []
    assert _name_of(env, B) not in env.a.get(ME_LIST + f"&key={B}").text


def test_scorecard_has_no_text(env: Env) -> None:  # A-10
    score_all(env)
    body = env.a.get(ME_CARD.format(sid(env, A))).text + env.a.get(ME_LIST).text
    assert "CANARY" not in body
    assert "a_tool_" not in env.a.get(ME_LIST).text  # names only where a finding names a tool


# -- won't fix (analysis Task 3.3, D-3) ------------------------------------------------

WF = "/api/me/prompt-lens/wontfix"
WF_ADMIN = "/api/admin/prompt-lens/wontfix"
V1, V2 = "aaaa1111", "bbbb2222"


def send(env: Env, who: As | None, method: str, url: str, **kw: Any) -> Any:
    headers = {**(who.headers if who is not None else env.admin), **csrf_header(env.client)}
    return env.client.request(method, url, headers=headers, **kw)


def card(env: Env, token: str = A) -> dict[str, Any]:
    who = env.a if token == A else env.b
    r = who.get(ME_CARD.format(sid(env, token)))
    assert r.status_code == 200, r.text
    return r.json()


def put_versioned(
    env: Env, token: str, sess: str, prefix: str, version: str | None, start: float
) -> str:
    """A second session of the key, as ``session_rows`` shapes it, stamped with a version."""
    rows = [
        replace(
            r,
            cobs2_session=sess,
            cobs2_conv=f"conv-{sess}",
            session_key=sess,
            lens_version=version,
        )
        for r in session_rows(token, prefix, start=start)
    ]
    put_rows(env.db, rows, secret=env.secret)
    return sid(env, token, sess)


def stamp(env: Env, version: str) -> None:
    with sqlite3.connect(env.db) as c:
        c.execute("UPDATE request_history SET lens_version = ?", (version,))
        c.commit()


def test_mark_removes_term_and_top3_but_keeps_finding_muted(env: Env) -> None:
    before = card(env)
    assert "E-01" in {f["rule"] for f in before["top"]} or before["scores"]["E"]["value"] < 100
    e0 = before["scores"]["E"]["value"]
    r = send(env, env.a, "PUT", f"{WF}/E-01", json={"version": None})
    assert r.status_code == 200 and r.json()["changed"] is True
    assert [(m["rule"], m["version"], m["by"]) for m in r.json()["marks"]] == [
        ("E-01", None, "key")
    ]
    after = card(env)
    e1 = after["scores"]["E"]
    assert e1["value"] > e0 and "E-01" in e1["excluded_rules"]
    assert e1["inputs"]["with_marks"] == e0  # "with them: <the old value>"
    assert all(f["rule"] != "E-01" for f in after["top"])
    muted = [f for f in after["findings"] if f["rule"] == "E-01"]
    assert muted and all(f["muted"] is True and f["muted_by"] == "key" for f in muted)
    unmuted = [f for f in after["findings"] if f["rule"] != "E-01"]
    assert unmuted and all(not f["muted"] and "muted_by" not in f for f in unmuted)
    assert after["scores"]["overall"]["value"] > before["scores"]["overall"]["value"]


def test_unmark_restores_scorecard_byte_identically(env: Env) -> None:
    before = env.a.get(ME_CARD.format(sid(env, A))).content
    assert send(env, env.a, "PUT", f"{WF}/E-01", json={}).status_code == 200
    assert env.a.get(ME_CARD.format(sid(env, A))).content != before
    r = send(env, env.a, "DELETE", f"{WF}/E-01")
    assert r.status_code == 200 and r.json() == {"marks": [], "changed": True}
    assert env.a.get(ME_CARD.format(sid(env, A))).content == before
    assert send(env, env.a, "DELETE", f"{WF}/E-01").json()["changed"] is False


def test_version_limited_mark_stops_on_the_next_version(env: Env) -> None:
    stamp(env, V1)
    second = put_versioned(env, A, "sess-v2", "reqv", V2, seed.T0 + 5_000)
    assert card(env)["session"]["version_id"] == V1
    assert send(env, env.a, "PUT", f"{WF}/E-01", json={"version": V1}).status_code == 200
    first = card(env)
    assert any(f["rule"] == "E-01" and f["muted"] for f in first["findings"])
    nxt = env.a.get(ME_CARD.format(second)).json()
    assert nxt["session"]["version_id"] == V2
    assert all(not f["muted"] for f in nxt["findings"])  # judged afresh
    assert "with_marks" not in nxt["scores"]["E"]["inputs"]
    # the same mark for every version reaches both
    assert send(env, env.a, "PUT", f"{WF}/E-01", json={}).status_code == 200
    assert any(
        f["rule"] == "E-01" and f["muted"]
        for f in env.a.get(ME_CARD.format(second)).json()["findings"]
    )


def test_each_mark_and_unmark_writes_one_audit_row(env: Env, flush_audit) -> None:  # noqa: F811
    flush_audit()

    def rows() -> list[tuple[str, str, str]]:
        flush_audit()
        with sqlite3.connect(env.db) as c:
            return c.execute(
                "SELECT token_id, username, path FROM admin_audit WHERE path LIKE '%lens.wontfix%'"
                " ORDER BY id"
            ).fetchall()

    assert rows() == []
    send(env, env.a, "PUT", f"{WF}/E-01", json={"version": V1})
    send(env, env.a, "PUT", f"{WF}/E-01", json={"version": V1})  # nothing changed: no row
    got = rows()
    assert len(got) == 1 and got[0][:2] == (A, "key-holder")
    assert got[0][2].endswith(f"lens.wontfix.mark:E-01@{V1}")
    send(env, env.a, "DELETE", f"{WF}/E-01?version={V1}")
    got = rows()
    assert len(got) == 2 and got[1][2].endswith(f"lens.wontfix.unmark:E-01@{V1}")
    assert "{rule}" in got[0][2]  # the route template, never the concrete URL
    send(env, None, "PUT", f"{WF_ADMIN}/C-01?key={A}", json={})
    got = rows()
    assert len(got) == 3 and got[2][0] == A and got[2][2].endswith("lens.wontfix.mark:C-01@all")


def test_r09_and_unknown_rules_are_422(env: Env) -> None:
    assert send(env, env.a, "PUT", f"{WF}/R-09", json={}).status_code == 422
    assert send(env, env.a, "PUT", f"{WF}/X-99", json={}).status_code == 422
    assert send(env, env.a, "DELETE", f"{WF}/X-99").status_code == 422
    assert send(env, None, "PUT", f"{WF_ADMIN}/R-09?key={A}", json={}).status_code == 422
    assert env.a.get(WF).json() == {"marks": [], "changed": False}


def test_a_key_holder_cannot_name_a_key(env: Env) -> None:  # A-1
    for method, url in (
        ("PUT", f"{WF}/E-01?key={B}"),
        ("DELETE", f"{WF}/E-01?key={B}"),
        ("GET", f"{WF}?key={B}"),
    ):
        r = send(env, env.a, method, url, **({"json": {}} if method == "PUT" else {}))
        assert r.status_code == 422, (method, r.status_code)
    # a body cannot carry one either
    assert send(env, env.a, "PUT", f"{WF}/E-01", json={"key": B}).status_code == 422
    assert env.b.get(WF).json()["marks"] == [] and env.a.get(WF).json()["marks"] == []


def test_a_mark_of_key_a_never_touches_key_b(env: Env) -> None:  # A-1
    before = env.b.get(ME_CARD.format(sid(env, B))).content
    send(env, env.a, "PUT", f"{WF}/E-01", json={})
    send(env, env.a, "PUT", f"{WF}/R-04", json={})
    assert env.b.get(ME_CARD.format(sid(env, B))).content == before
    assert env.b.get(WF).json()["marks"] == []


def test_admin_marks_any_key_and_the_holder_sees_who(env: Env) -> None:
    r = send(env, None, "PUT", f"{WF_ADMIN}/E-01?key={B}", json={})
    assert r.status_code == 200 and r.json()["marks"][0]["by"] == "admin"
    assert send(env, None, "PUT", f"{WF_ADMIN}/E-01?key=nope", json={}).status_code == 404
    assert send(env, None, "PUT", f"{WF_ADMIN}/E-01", json={}).status_code == 422  # key needed
    got = card(env, B)
    assert any(f["muted_by"] == "admin" for f in got["findings"] if f["muted"])
    assert env.as_admin(f"{WF_ADMIN}?key={B}").json()["marks"][0]["rule"] == "E-01"
    assert send(env, None, "DELETE", f"{WF_ADMIN}/E-01?key={B}").json()["marks"] == []
    # the admin's own door is closed to a key holder, and theirs to the admin
    assert send(env, env.a, "PUT", f"{WF_ADMIN}/E-01?key={A}", json={}).status_code == 401
    assert send(env, None, "PUT", f"{WF}/E-01", json={}).status_code == 403


def test_marks_change_the_scorecard_the_same_for_holder_and_admin(env: Env) -> None:
    send(env, env.a, "PUT", f"{WF}/E-01", json={})
    mine = env.a.get(ME_CARD.format(sid(env, A))).json()["scores"]
    admin = env.as_admin(ADMIN_CARD.format(sid(env, A))).json()["scores"]
    assert mine == admin


def test_marks_follow_the_rotation_lineage(env: Env) -> None:  # Review Focus 4
    with sqlite3.connect(env.db) as c:  # B's key is A's predecessor
        c.execute("UPDATE api_tokens SET rotated_from = ? WHERE id = ?", (B, A))
        c.commit()
    send(env, None, "PUT", f"{WF_ADMIN}/E-01?key={B}", json={})
    assert [m["rule"] for m in env.a.get(WF).json()["marks"]] == ["E-01"]
    assert any(f["muted"] for f in card(env)["findings"] if f["rule"] == "E-01")


def test_the_session_list_and_stored_rows_carry_no_mark(env: Env) -> None:
    """The stored score (what a leaderboard reads) is without the marks (spec 11.4)."""
    score_all(env)
    send(env, env.a, "PUT", f"{WF}/E-01", json={})
    stored = env.as_admin(ADMIN_LIST + f"&key={A}").json()["sessions"][0]["e"]
    with sqlite3.connect(env.db) as c:
        assert (
            c.execute("SELECT e FROM lens_session_scores WHERE key_id = ?", (A,)).fetchone()[0]
            == stored
        )
    assert stored < card(env)["scores"]["E"]["value"]


def test_marks_leave_the_canary_out_of_every_answer(env: Env) -> None:  # A-10
    send(env, env.a, "PUT", f"{WF}/E-01", json={"version": V1})
    body = env.a.get(WF).text + env.a.get(ME_CARD.format(sid(env, A))).text
    assert "CANARY" not in body


def test_a_mark_is_the_same_as_rescoring_without_the_rule(env: Env) -> None:
    """``apply_marks`` on the stored row equals the scorecard's exact re-score (E, C, R)."""
    from app.lens import wontfix
    from app.lens.store import LensStore

    score_all(env)
    with sqlite3.connect(env.db) as c:
        row = LensStore(c).session(A, sid(env, A))
    assert row is not None
    for rule in ("E-01", "R-04", "C-01"):
        send(env, env.a, "PUT", f"{WF}/{rule}", json={})
        live = card(env)["scores"]
        got = wontfix.apply_marks(row, frozenset({rule}))
        for layer, value in (("E", got.e), ("C", got.c), ("R", got.r)):
            if live[layer]["scored"]:
                assert abs(live[layer]["value"] - (value or 0)) <= 1, (rule, layer)
        send(env, env.a, "DELETE", f"{WF}/{rule}")


# -- trends, versions and the comparison (analysis Task 3.2) ---------------------------

ME_TRENDS = "/api/me/prompt-lens/trends"
ME_VERSIONS = "/api/me/prompt-lens/versions"
ME_COMPARE = "/api/me/prompt-lens/compare"
ADMIN_TRENDS = "/api/admin/prompt-lens/trends"
FLEET = "/api/admin/prompt-lens/fleet"
#: A Monday, 00:00 UTC (the seeds' T0 is a Tuesday).
WEEK0 = 1_700_438_400.0 - (1_700_438_400.0 % 604_800) + 4 * 86_400.0


def put_score(
    env: Env,
    key: str,
    n: int,
    at: float,
    *,
    version: str | None = V1,
    overall: int | None = 60,
    model: str = "fake-model",
    route: str = "local",
    harness: str = "opencode",
    turns: int = 20,
    rules: dict[str, Any] | None = None,
) -> None:
    layers = {"E": overall, "C": overall, "R": overall}
    terms = {
        k: {"scored": v is not None, "turns": turns, "by_rule": {}, "excluded": []}
        for k, v in layers.items()
    }
    with sqlite3.connect(env.db) as c:
        c.execute(
            "INSERT INTO lens_session_scores(key_id, sess_ref, started, last_at, model, route,"
            " harness, version, turns, e, c, r, overall, ranges, confidence, coverage, terms,"
            " rules, score_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '{}', '{}',"
            " ?, ?, 1)",
            (
                key,
                f"s_{key}_{n:04d}",
                at,
                at + 60,
                model,
                route,
                harness,
                version,
                turns,
                overall,
                overall,
                overall,
                overall,
                json.dumps(terms),
                json.dumps(rules or {}),
            ),
        )
        c.commit()


def fill(env: Env, key: str, start: int, scores: list[int | None], at: float, **kw: Any) -> None:
    for i, v in enumerate(scores):
        put_score(env, key, start + i, at + i * 3600, overall=v, **kw)


def test_weekly_trend_with_unscored_buckets_and_version_markers(env: Env) -> None:
    fill(env, A, 0, [40, 50, 60, 70, 80], WEEK0, version=V1, rules={"C": ["C-01"], "E": ["E-01"]})
    fill(env, A, 10, [85, 90, 95], WEEK0 + 7 * 86_400, version=V2, rules={"E": ["E-01"]})
    fill(env, A, 20, [30, 35], WEEK0 + 14 * 86_400, version=V2)  # two sessions: no number
    r = env.a.get(ME_TRENDS + "?by=week&since=0")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"] == "fake-model" and d["models"] == ["fake-model"]
    pts = d["points"]
    assert [p["n"] for p in pts] == [5, 3, 2]
    assert pts[0]["medians"]["overall"] == 60 and pts[0]["band"] == [50, 70]
    assert pts[1]["medians"]["overall"] == 90
    assert pts[2]["medians"] == {"overall": None, "E": None, "C": None, "R": None}  # A-7
    assert pts[2]["band"] is None
    v1, v2 = d["versions"]
    assert (v1["id"], v1["label"], v1["n"]) == (V1, "v1", 5)
    assert (v2["id"], v2["label"], v2["n"]) == (V2, "v2", 5)
    assert v2["resolved"] == ["C-01"] and v2["new"] == [] and v1["rules"] == {"C-01": 5, "E-01": 5}
    by_ver = env.a.get(ME_TRENDS + "?by=version&since=0").json()["points"]
    assert [(p["bucket"], p["label"]) for p in by_ver] == [(V1, "v1"), (V2, "v2")]
    day = env.a.get(ME_TRENDS + "?by=day&since=0&tz=-480").json()["points"]
    assert sum(p["n"] for p in day) == 10 and all(p["n"] < 3 or p["band"] for p in day)


def test_trends_default_to_the_newest_model_and_can_pick_another(env: Env) -> None:
    fill(env, A, 0, [50, 60, 70], WEEK0, model="old-model")
    fill(env, A, 10, [80, 85, 90], WEEK0 + 86_400, model="new-model")
    d = env.a.get(ME_TRENDS + "?since=0").json()
    assert d["model"] == "new-model" and d["models"] == ["new-model", "old-model"]
    assert d["points"][0]["medians"]["overall"] == 85
    old = env.a.get(ME_TRENDS + "?since=0&model=old-model").json()
    assert old["points"][0]["medians"]["overall"] == 60


def test_compare_gives_a_plain_verdict_and_resolved_rules(env: Env) -> None:
    fill(env, A, 0, [30, 35, 40, 45, 50], WEEK0, version=V1, rules={"C": ["C-01"]})
    fill(env, A, 10, [70, 75, 80, 85, 90], WEEK0 + 7 * 86_400, version=V2, rules={"R": ["R-04"]})
    r = env.a.get(ME_COMPARE + f"?a={V1}&b={V2}&since=0")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["verdict"] == "better_clear" and (d["n_a"], d["n_b"]) == (5, 5)
    assert d["diffs"]["overall"] == 40 and d["like_for_like"] is True and d["note"] is None
    assert d["resolved"] == ["C-01"] and d["new"] == ["R-04"] and d["p"] < 0.05
    assert (d["a_label"], d["b_label"]) == ("v1", "v2")
    back = env.a.get(ME_COMPARE + f"?a={V2}&b={V1}&since=0").json()
    assert back["verdict"] == "worse_clear"
    assert env.a.get(ME_COMPARE + f"?a={V1}&b=nope&since=0").status_code == 404
    assert env.a.get(ME_COMPARE + f"?a={V1}&since=0").status_code == 422


def test_compare_with_four_sessions_is_not_enough_and_unlike_is_said(env: Env) -> None:
    fill(env, A, 0, [30, 35, 40, 45, 50], WEEK0, version=V1)
    fill(env, A, 10, [70, 75, 80, 85], WEEK0 + 7 * 86_400, version=V2, harness="aider")
    d = env.a.get(ME_COMPARE + f"?a={V1}&b={V2}&since=0").json()
    assert d["verdict"] == "not_enough" and d["p"] is None
    assert d["like_for_like"] is False and "harness" in d["note"]


def test_header_named_versions_keep_their_own_label(env: Env) -> None:
    fill(env, A, 0, [40, 50, 60], WEEK0, version="h:spring prompt")
    fill(env, A, 10, [70, 75, 80], WEEK0 + 7 * 86_400, version=V1)
    labels = [v["label"] for v in env.a.get(ME_VERSIONS + "?since=0").json()["versions"]]
    assert labels == ["spring prompt", "v1"]  # an ordinal counts detected versions only


def test_a_rotated_keys_trend_includes_its_earlier_keys_sessions(
    env: Env,
) -> None:  # Review Focus 4
    fill(env, B, 0, [40, 50, 60], WEEK0)  # the predecessor
    fill(env, A, 10, [70, 80, 90], WEEK0 + 86_400)
    only = env.a.get(ME_TRENDS + "?since=0").json()["points"]
    assert [p["n"] for p in only] == [3]
    with sqlite3.connect(env.db) as c:
        c.execute("UPDATE api_tokens SET rotated_from = ? WHERE id = ?", (B, A))
        c.commit()
    pts = env.a.get(ME_TRENDS + "?since=0").json()["points"]
    assert [p["n"] for p in pts] == [6] and pts[0]["medians"]["overall"] == 65
    assert env.b.get(ME_TRENDS + "?since=0").json()["points"][0]["n"] == 6  # the chain is one


def test_key_routes_refuse_a_key_parameter(env: Env) -> None:  # A-1
    for url in (ME_TRENDS, ME_VERSIONS, ME_COMPARE + f"?a={V1}&b={V2}"):
        sep = "&" if "?" in url else "?"
        assert env.a.get(f"{url}{sep}key={B}").status_code == 422, url
    assert env.a.get(ME_TRENDS + "?key=").status_code == 422


def test_trends_ignore_other_keys_traffic(env: Env) -> None:  # A-1
    fill(env, A, 0, [40, 50, 60, 70], WEEK0)
    before = [env.a.get(u).content for u in (ME_TRENDS + "?since=0", ME_VERSIONS + "?since=0")]
    fill(env, B, 0, [10, 20, 30, 40, 50], WEEK0 + 3_600, rules={"C": ["C-01"]})
    fill(env, B, 10, [90, 95], WEEK0 + 7 * 86_400, version=V2)
    after = [env.a.get(u).content for u in (ME_TRENDS + "?since=0", ME_VERSIONS + "?since=0")]
    assert after == before


def test_won_t_fix_marks_move_a_trend_and_unmark_restores_it(env: Env) -> None:
    for i in range(4):
        put_score(env, A, i, WEEK0 + i * 3600, overall=60)
    with sqlite3.connect(env.db) as c:  # give the stored E a lost term, so a mark has weight
        terms = json.dumps(
            {
                "E": {
                    "scored": True,
                    "turns": 20,
                    "total": 10_000.0,
                    "base": 0.0,
                    "by_rule": {"E-01": 4_000.0},
                    "excluded": [],
                }
            }
        )
        c.execute("UPDATE lens_session_scores SET terms = ?, e = 60", (terms,))
        c.commit()
    before = env.a.get(ME_TRENDS + "?since=0").content
    send(env, env.a, "PUT", f"{WF}/E-01", json={})
    moved = env.a.get(ME_TRENDS + "?since=0").json()["points"][0]["medians"]
    assert moved["E"] == 100
    send(env, env.a, "DELETE", f"{WF}/E-01")
    assert env.a.get(ME_TRENDS + "?since=0").content == before


def test_fleet_is_the_admins_alone(env: Env) -> None:  # A-14, D-4
    fill(env, A, 0, [40, 50, 60, 70, 80], WEEK0)
    fill(env, B, 0, [20, 30, 40, 50, 60], WEEK0)
    d = env.as_admin(FLEET + "?since=0").json()
    assert d["sessions"] == 10 and d["keys"] == 2 and d["overall"]["n"] == 10
    assert d["overall"]["p50"] == 50 and d["overall"]["p10"] is not None
    assert env.a.get(FLEET + "?since=0").status_code == 401  # a key holder's sign-in
    assert env.client.get(FLEET).status_code == 401
    inference = env.client.get(FLEET, headers={"Authorization": "Bearer vw_" + "x" * 40})
    assert inference.status_code == 401
    for url in (ME_TRENDS, ME_VERSIONS, ME_COMPARE + f"?a={V1}&b={V2}", WF):
        assert env.as_admin(url).status_code == 403  # an admin has no key on a /me route


def test_no_key_holder_answer_holds_a_fleet_figure(env: Env) -> None:  # A-14
    fill(env, A, 0, [40, 50, 60, 70], WEEK0)
    fill(env, B, 0, [10, 20, 30, 40], WEEK0)
    for url in (ME_TRENDS + "?since=0", ME_VERSIONS + "?since=0", WF):
        text = env.a.get(url).text.lower()
        for word in ("percentile", "rank", "fleet", "leaderboard", "distribution", "p90"):
            assert word not in text, (url, word)


def test_admin_trends_need_a_key_and_read_any(env: Env) -> None:
    fill(env, B, 0, [40, 50, 60], WEEK0)
    assert env.as_admin(ADMIN_TRENDS).status_code == 422
    d = env.as_admin(ADMIN_TRENDS + f"?key={B}&since=0").json()
    assert d["points"][0]["n"] == 3
    assert env.a.get(ADMIN_TRENDS + f"?key={B}").status_code == 401


def test_the_scorer_stores_the_sessions_newest_version(env: Env) -> None:
    stamp(env, V1)
    with sqlite3.connect(env.db) as c:  # the last two turns went out with a newer version
        c.execute(
            "UPDATE request_history SET lens_version = ? WHERE token_id = ? AND id IN (?, ?)",
            (V2, A, req("reqa", 23), req("reqa", 24)),
        )
        c.commit()
    assert score_all(env) == 2
    with sqlite3.connect(env.db) as c:
        got = dict(c.execute("SELECT key_id, version FROM lens_session_scores").fetchall())
    assert got == {A: V2, B: V1}


def test_trends_hold_no_text(env: Env) -> None:  # A-10
    fill(env, A, 0, [40, 50, 60], WEEK0)
    for url in (ME_TRENDS + "?since=0", ME_VERSIONS + "?since=0"):
        assert "CANARY" not in env.a.get(url).text
