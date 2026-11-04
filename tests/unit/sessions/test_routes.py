"""Sessions API (sessions pages, task 8): scoping, refs, tool names, wait_ms."""

from __future__ import annotations

import json
import sqlite3
import time

from app.auth.policy import FOREST_ROUTES
from app.sessions.refs import sess_ref
from tests.unit.stats.test_forest_routes import _forest_login, _ready, _ready_key


def _secret(client):
    return client.app.state.settings.cookie_secret


def _put(
    db_path, rid, *, tok="tok1", sess="S", conv="c1", at=None, edge="continue", secret, **cols
):
    at = time.time() - 50 if at is None else at
    row = {
        "id": rid,
        "finished_at": at,
        "model_id": "m",
        "model": "m",
        "token_id": tok,
        "token_name": "name-" + tok,
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "cached_tokens": 800,
        "duration_s": 1.0,
        "ttft_s": 0.2,
        "started_iso": "2026-10-25T10:00:00Z",
        "orphan": 0,
        "session_key": sess,
        "cobs2_session": sess,
        "cobs2_sess_ref": sess_ref(secret, tok, sess),
        "cobs2_conv": conv,
        "cobs2_edge": edge,
        "cobs2_node": "n-" + rid,
    }
    row.update(cols)
    with sqlite3.connect(db_path) as db:
        db.execute(
            f"INSERT INTO request_history({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",
            tuple(row.values()),
        )
        db.commit()


RID = "a" * 31


def test_forest_routes_allowlist_includes_sessions():
    assert {
        ("GET", "/api/sessions"),
        ("GET", "/api/sessions/{session_id}"),
        ("GET", "/api/requests/{request_id}"),
    } <= FOREST_ROUTES


def test_admin_lists_all_sessions_key_holder_only_own(tmp_data_dir, client):
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    s = _secret(client)
    _put(db_path, RID + "1", secret=s, edge="new")
    _put(db_path, RID + "2", tok="tok2", sess="T", secret=s, edge="new")
    a = client.get("/api/sessions?range=24h", headers=admin).json()
    assert {x["token_id"] for x in a["sessions"]} == {"tok1", "tok2"}
    k = client.get("/api/sessions?range=24h", headers=_forest_login(client, plaintext)).json()
    assert {x["token_id"] for x in k["sessions"]} == {"tok1"}
    k2 = client.get(
        "/api/sessions?range=24h&token=tok2", headers=_forest_login(client, plaintext)
    ).json()
    assert {x["token_id"] for x in k2["sessions"]} == {"tok1"}


def test_bad_range_and_limit_are_refused(tmp_data_dir, client):
    _, h = _ready(client, tmp_data_dir)
    assert client.get("/api/sessions?range=99d", headers=h).status_code == 400
    assert client.get("/api/sessions?limit=101", headers=h).status_code == 422


def test_list_pages_by_cursor(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    now = time.time()
    for i in range(120):
        _put(db_path, f"{i:032x}", sess=f"S{i}", at=now - 1000 + i, secret=s, edge="new")
    p1 = client.get("/api/sessions?range=24h&limit=100", headers=h).json()
    assert len(p1["sessions"]) == 100 and p1["next_cursor"]
    p2 = client.get(
        f"/api/sessions?range=24h&limit=100&cursor={p1['next_cursor']}", headers=h
    ).json()
    assert len(p2["sessions"]) == 20 and p2["next_cursor"] is None
    ids = [x["session_id"] for x in p1["sessions"] + p2["sessions"]]
    assert len(set(ids)) == 120


def test_session_by_s_and_r_refs_agree_and_canonicalise(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    _put(db_path, RID + "1", secret=s, edge="new", at=time.time() - 60)
    _put(
        db_path,
        RID + "2",
        secret=s,
        at=time.time() - 50,
        cobs2_parent_req=RID + "1",
        cobs2_parent_node="n-" + RID + "1",
    )
    sid = "s_" + sess_ref(s, "tok1", "S")
    a = client.get(f"/api/sessions/{sid}", headers=h).json()
    b = client.get(f"/api/sessions/r_{RID}2", headers=h).json()
    assert a == b and a["session_id"] == sid
    assert [t["request_id"] for t in a["turns"]] == [RID + "1", RID + "2"]


def test_other_tokens_session_and_request_are_404(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    s = _secret(client)
    _put(db_path, RID + "9", tok="tok2", sess="T", secret=s, edge="new")
    h = _forest_login(client, plaintext)
    sid = "s_" + sess_ref(s, "tok2", "T")
    r1 = client.get(f"/api/sessions/{sid}", headers=h)
    r2 = client.get(f"/api/sessions/r_{RID}9", headers=h)
    r3 = client.get(f"/api/requests/{RID}9", headers=h)
    assert (r1.status_code, r2.status_code, r3.status_code) == (404, 404, 404)
    assert r1.json() == {"detail": "session not found"}


def test_bad_ref_is_404(tmp_data_dir, client):
    _, h = _ready(client, tmp_data_dir)
    assert client.get("/api/sessions/../etc", headers=h).status_code == 404
    assert client.get("/api/sessions/s_zz", headers=h).status_code == 404


def test_raw_session_key_never_in_responses(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    raw = "RAWSESSIONKEY0000"
    _put(db_path, RID + "1", sess=raw, secret=s, edge="new")
    lst = client.get("/api/sessions?range=24h", headers=h).text
    sid = json.loads(lst)["sessions"][0]["session_id"]
    one = client.get(f"/api/sessions/{sid}", headers=h).text
    req = client.get(f"/api/requests/{RID}1", headers=h).text
    assert raw not in lst and raw not in one and raw not in req


def test_tool_names_shown_to_owner_and_admin_only(tmp_data_dir, client):
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    s = _secret(client)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO cobs2_toolsets VALUES (?,?,?)",
            ("d" * 32, json.dumps([["h1234567", "agent", 40]]), 1.0),
        )
        db.executemany(
            "INSERT INTO session_tools VALUES (?,?,?,?,?,?,?)",
            [
                ("tok1", "h1234567", "Task", "agent", 40, 1.0, 1.0),
                ("tok2", "h1234567", "Other", "agent", 40, 1.0, 1.0),
            ],
        )
        db.commit()
    _put(db_path, RID + "1", secret=s, edge="new", cobs2_toolset="d" * 32)
    _put(db_path, RID + "2", tok="tok2", sess="T", secret=s, edge="new", cobs2_toolset="d" * 32)
    mine = "s_" + sess_ref(s, "tok1", "S")
    theirs = "s_" + sess_ref(s, "tok2", "T")
    kh = _forest_login(client, plaintext)
    for h in (kh, admin):
        body = client.get(f"/api/sessions/{mine}", headers=h).json()
        assert body["tools"]["rows"][0]["name"] == "Task"
    assert client.get(f"/api/sessions/{theirs}", headers=kh).status_code == 404
    other = client.get(f"/api/sessions/{theirs}", headers=admin).json()
    assert other["tools"]["rows"][0]["name"] == "Other"


def test_requests_wait_ms_waits_for_commit(tmp_data_dir, client):
    _, h = _ready(client, tmp_data_dir)
    rid = "b" * 32
    hist = client.app.state.request_history
    hist.record(
        {
            "id": rid,
            "model": "m",
            "model_id": "m",
            "token_id": None,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "duration_s": 1.0,
            "started_iso": "2026-10-25T10:00:00Z",
            "cobs2_session": "S",
            "cobs2_conv": "c1",
        }
    )
    r = client.get(f"/api/requests/{rid}?wait_ms=3000", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == rid and body["session_id"].startswith("s_")
    t0 = time.monotonic()
    miss = client.get(f"/api/requests/{'c' * 32}?wait_ms=100", headers=h)
    assert miss.status_code == 404 and time.monotonic() - t0 < 2.0


def test_wait_ms_is_bounded(tmp_data_dir, client):
    _, h = _ready(client, tmp_data_dir)
    assert client.get(f"/api/requests/{RID}1?wait_ms=999999", headers=h).status_code == 422


def test_read_failure_is_503_and_stats_still_work(tmp_data_dir, client, monkeypatch):
    from app.sessions import service

    _, h = _ready(client, tmp_data_dir)

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(service, "_connect", boom)
    r = client.get("/api/sessions?range=24h", headers=h)
    assert r.status_code == 503 and r.json() == {"detail": "sessions unavailable"}
    assert client.get(f"/api/sessions/s_{'a' * 32}", headers=h).status_code == 503
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 200


def test_live_ref_of_uncommitted_first_turn_is_not_recorded_yet(tmp_data_dir, client):
    from tests.unit.proxy.test_cache_estimate_forward import _live_rows  # noqa: F401

    _, h = _ready(client, tmp_data_dir)
    reg = client.app.state.request_registry

    class Live:
        id = "e" * 32
        token_id = None
        cobs2_session = "NEWSESS"

    monkey = reg.snapshot
    reg.snapshot = lambda: [Live()]
    try:
        r = client.get(f"/api/sessions/r_{'e' * 32}", headers=h)
    finally:
        reg.snapshot = monkey
    assert r.status_code == 404 and r.json() == {"detail": "not recorded yet"}


def test_raw_vw_key_is_refused(tmp_data_dir, client):
    _, plaintext, _ = _ready_key(client, tmp_data_dir)
    r = client.get("/api/sessions", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 401


def test_wait_ms_miss_for_an_in_flight_request_is_not_recorded_yet(tmp_data_dir, client):
    """Final review I3: a wait that ends while the request is still in flight says so."""
    _, h = _ready(client, tmp_data_dir)
    reg = client.app.state.request_registry

    class Live:
        id = "e" * 32
        token_id = None
        cobs2_session = "S"

    real = reg.snapshot
    reg.snapshot = lambda: [Live()]
    try:
        r = client.get(f"/api/requests/{'e' * 32}?wait_ms=50", headers=h)
    finally:
        reg.snapshot = real
    assert r.status_code == 404 and r.json() == {"detail": "not recorded yet"}
    assert client.get(f"/api/requests/{'d' * 32}?wait_ms=50", headers=h).json() == {
        "detail": "request not found"
    }


def test_fallback_session_listed_in_a_short_range_opens(tmp_data_dir, client):
    """Final review I1: a session-less client sending without an idle gap for longer than the
    range: the list's segment id starts at the first IN-WINDOW row; the detail must resolve it."""
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    now = time.time()
    for i in range(18):
        _put(
            db_path,
            f"{i:02x}" + "a" * 30,
            sess="f:abc",
            conv="c1",
            at=now - 3 * 3600 + i * 600,
            secret=s,
            edge="new" if i == 0 else "continue",
        )
    lst = client.get("/api/sessions?range=1h", headers=h).json()["sessions"]
    assert len(lst) == 1
    r = client.get(f"/api/sessions/{lst[0]['session_id']}", headers=h)
    assert r.status_code == 200, r.text
    assert len(r.json()["turns"]) == 18


def test_list_items_never_leak_private_keys_and_detail_has_costs(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    _put(
        db_path,
        RID + "1",
        secret=s,
        edge="new",
        cobs2_shape='{"chars":4000,"dw":{"dup":0,"boil":40,"reason":0}}',
    )
    body = client.get("/api/sessions?range=24h", headers=h).json()
    assert body["sessions"] and all(not k.startswith("_") for k in body["sessions"][0])
    sid = body["sessions"][0]["session_id"]
    d = client.get(f"/api/sessions/{sid}", headers=h).json()
    assert {"dead_weight", "anatomy", "engine_replay", "pauses", "harness"} <= set(d)
    assert d["totals"]["where"] == "local"


# -- sessions redesign, task 7 ------------------------------------------------


def test_list_grouped_response_shape(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    _put(db_path, RID + "1", secret=s, edge="new")
    _put(db_path, RID + "2", tok="tok2", sess="T", secret=s, edge="new")
    body = client.get("/api/sessions?range=24h&group=key", headers=h).json()
    assert {"groups", "views", "histogram", "pause_summary", "total"} <= set(body)
    assert {g["key"] for g in body["groups"]} == {"tok1", "tok2"} and body["views"]["all"] == 2


def test_list_bad_enums_are_400(tmp_data_dir, client):
    _, h = _ready(client, tmp_data_dir)
    for q in ("group=foo", "view=foo", "where=foo", "sort=foo"):
        assert client.get(f"/api/sessions?range=24h&{q}", headers=h).status_code == 400, q


def test_key_holder_gets_one_group(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    s = _secret(client)
    _put(db_path, RID + "1", secret=s, edge="new")
    _put(db_path, RID + "2", tok="tok2", sess="T", secret=s, edge="new")
    body = client.get(
        "/api/sessions?range=24h&group=key", headers=_forest_login(client, plaintext)
    ).json()
    assert [g["key"] for g in body["groups"]] == ["tok1"]


def test_legacy_cursor_path_gains_new_keys(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    _put(db_path, RID + "1", secret=_secret(client), edge="new")
    body = client.get("/api/sessions?range=24h", headers=h).json()
    assert body["sessions"] and "next_cursor" in body and "views" in body and "histogram" in body


def test_scan_cache_serves_filter_changes_within_5s(tmp_data_dir, client, monkeypatch):
    from app.sessions import service, store

    service._list_cache.clear()
    db_path, h = _ready(client, tmp_data_dir)
    _put(db_path, RID + "1", secret=_secret(client), edge="new")
    calls = []
    real = store.list_rows

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(store, "list_rows", spy)
    client.get("/api/sessions?range=24h&group=key", headers=h)
    client.get("/api/sessions?range=24h&group=harness&view=running", headers=h)
    assert len(calls) == 1


def test_fallback_detail_returns_the_requested_id(tmp_data_dir, client):
    """Verified finding #20: list and detail must name a segment the same way."""
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    now = time.time()
    for i in range(18):
        _put(
            db_path,
            f"{i:02x}" + "a" * 30,
            sess="f:abc",
            conv="c1",
            at=now - 3 * 3600 + i * 600,
            secret=s,
            edge="new" if i == 0 else "continue",
        )
    sid = client.get("/api/sessions?range=1h", headers=h).json()["sessions"][0]["session_id"]
    assert client.get(f"/api/sessions/{sid}", headers=h).json()["session_id"] == sid


def test_long_fallback_segment_keeps_opening_as_rows_arrive(tmp_data_dir, client, monkeypatch):
    """#20: a continuous session-less segment longer than the scan window must not 404 on the
    next poll (the old code named the window's oldest row, which then dropped out)."""
    import app.sessions.service as svc

    monkeypatch.setattr(svc, "_FALLBACK_SIDE", 6)
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    now = time.time()
    for i in range(30):
        _put(
            db_path,
            f"{i:02x}" + "b" * 30,
            sess="f:long",
            conv="c1",
            at=now - 3600 + i * 60,
            secret=s,
            edge="new" if i == 0 else "continue",
        )
    sid = client.get("/api/sessions?range=24h", headers=h).json()["sessions"][0]["session_id"]
    r = client.get(f"/api/sessions/{sid}", headers=h)
    assert r.status_code == 200, r.text
    got = r.json()["session_id"]
    for j in range(30, 36):
        _put(
            db_path,
            f"{j:02x}" + "b" * 30,
            sess="f:long",
            conv="c1",
            at=now - 3600 + j * 60,
            secret=s,
            edge="continue",
        )
        assert client.get(f"/api/sessions/{got}", headers=h).status_code == 200


def test_older_fallback_segment_beyond_the_newest_rows_opens(tmp_data_dir, client, monkeypatch):
    """#20: an older segment of a busy session-less key must open from the list."""
    import app.sessions.service as svc

    monkeypatch.setattr(svc, "_FALLBACK_SIDE", 6)
    db_path, h = _ready(client, tmp_data_dir)
    s = _secret(client)
    now = time.time()
    for i in range(5):  # old segment
        _put(
            db_path,
            f"{i:02x}" + "c" * 30,
            sess="f:busy",
            conv="c1",
            at=now - 6 * 3600 + i * 60,
            secret=s,
            edge="new" if i == 0 else "continue",
        )
    for i in range(20):  # a newer segment, after a gap, with more rows than the window
        _put(
            db_path,
            f"{0x40 + i:02x}" + "c" * 30,
            sess="f:busy",
            conv="c1",
            at=now - 3600 + i * 60,
            secret=s,
            edge="new" if i == 0 else "continue",
        )
    lst = client.get("/api/sessions?range=24h", headers=h).json()["sessions"]
    old = min(lst, key=lambda x: x["started_at"])
    r = client.get(f"/api/sessions/{old['session_id']}", headers=h)
    assert r.status_code == 200, r.text
    assert len(r.json()["turns"]) == 5
