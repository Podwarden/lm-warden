import asyncio
import gzip
import json
import sqlite3
import time

import pytest

from app.stats import forest, routes_forest
from tests.conftest import jwt_login, seed_admin_user


@pytest.fixture(autouse=True)
def _no_state_ttl(monkeypatch):
    """Every request refreshes its state, so rows inserted between requests are seen at once.
    The tests of the TTL itself set it back."""
    monkeypatch.setattr(routes_forest, "FOREST_TTL_S", 0.0)


COLS = (
    "id, finished_at, model_id, model, token_name, token_id, client_ip, prompt_tokens, completion_tokens, "
    "duration_s, ttft_s, finish_reason, orphan, started_iso, variant_id, cached_tokens, session_key, batch_id, tools_out"
)


def _insert(db_path, rows):
    with sqlite3.connect(db_path) as db:
        db.executemany(f"INSERT INTO request_history({COLS}) VALUES ({','.join('?' * 19)})", rows)
        db.commit()


def _r(i, at, *, tok="tok1", name="farm", sk="S1", batch=None, p=1000):
    return (
        f"r{i}",
        at,
        "id-m",
        "m",
        name,
        tok,
        "10.0.0.1",
        p,
        50,
        2.0,
        0.5,
        "tool_calls",
        0,
        "2026-10-05T00:00:00Z",
        "v1",
        900,
        sk,
        batch,
        json.dumps([["read", "abcd1234"]]),
    )


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path, jwt_login(client)


def test_forest_returns_trees_for_admin(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 100), _r(2, now - 50, p=1500)])
    r = client.get("/api/stats/forest?range=6h", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["range"] == "6h" and len(body["trees"]) == 1
    assert len(body["trees"][0]["sessions"][0]["turns"]) == 2
    assert body["cursor"] >= now - 50


def test_forest_1h_range(tmp_data_dir, client):
    """1h is a range (the Stats page's default): it picks the trees with a turn in [now - 3600, now]."""
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 7200, sk="old"), _r(2, now - 100, sk="new")])
    r = client.get("/api/stats/forest?range=1h", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["range"] == "1h"
    assert len(body["trees"]) == 1
    assert all(t["last_at"] >= now - 3600 - 5 for t in body["trees"])
    six = client.get("/api/stats/forest?range=6h", headers=h).json()
    assert len(six["trees"]) == 2


def _straddler(now, n=160, every=600.0, sk="LONG"):
    """One session: a turn every `every` s, the last 5 min ago, reaching back across the 1h, 6h and 24h edges."""
    return [_r(1000 + k, now - 300 - (n - 1 - k) * every, sk=sk) for k in range(n)]


def _abs_turns(body, tid):
    t0 = body["t0"]
    tree = next(t for t in body["trees"] if t["id"] == tid)
    out = []

    def walk(s):
        out.append((s["id"], [[round(x[0] + t0, 1), *x[1:]] for x in s["turns"]]))
        for c in s["children"]:
            walk(c)

    for s in tree["sessions"]:
        walk(s)
    return out


def test_a_tree_straddling_the_window_edge_comes_back_whole(tmp_data_dir, client):
    """C1: the window picks trees, it never trims them. A session reaching back 26 h is sent whole for 1h, 6h, 24h."""
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, _straddler(now))
    bodies = {
        r: client.get(f"/api/stats/forest?range={r}", headers=h).json()
        for r in ("1h", "6h", "24h", "7d")
    }
    ids = {r: [t["id"] for t in b["trees"]] for r, b in bodies.items()}
    assert all(len(v) == 1 for v in ids.values()) and len({v[0] for v in ids.values()}) == 1
    tid = ids["1h"][0]
    whole = _abs_turns(bodies["7d"], tid)
    assert sum(len(t) for _, t in whole) == 160
    for r in ("1h", "6h", "24h"):
        assert _abs_turns(bodies[r], tid) == whole, r
    # a tree whose turns all finished before the window is not picked
    _insert(
        db_path, [_r(5000 + k, now - 3 * 3600 - k * 60, sk="GONE", tok="tok9") for k in range(3)]
    )
    one = client.get("/api/stats/forest?range=1h", headers=h).json()
    assert [t["id"] for t in one["trees"]] == [tid]
    assert len(client.get("/api/stats/forest?range=6h", headers=h).json()["trees"]) == 2


def test_whole_tree_send_keeps_the_lod_blocks(tmp_data_dir, client):
    """C1 x LOD: a 2100-turn session straddling the 1h edge has the same merged blocks (and child `at`) in every range."""
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, _straddler(now, n=2100, every=30.0))
    b1 = client.get("/api/stats/forest?range=1h", headers=h).json()
    b24 = client.get("/api/stats/forest?range=24h", headers=h).json()
    tid = b1["trees"][0]["id"]
    t1, t24 = _abs_turns(b1, tid), _abs_turns(b24, tid)
    assert t1 == t24
    turns = t1[0][1]
    assert (
        len(turns)
        == forest.LOD_CAP
        + (2100 - forest.LOD_CAP) // forest.LOD_BLOCK
        + (2100 - forest.LOD_CAP) % forest.LOD_BLOCK
    )
    assert turns[forest.LOD_CAP][7] == forest.LOD_BLOCK


@pytest.mark.parametrize("rng", ["30d", "14d", "8d"])
def test_ranges_wider_than_7d_are_refused(tmp_data_dir, client, rng):
    """Re-review 2, R2: the forest never holds rows older than 7 d; anything wider is a 400."""
    _, h = _ready(client, tmp_data_dir)
    r = client.get(f"/api/stats/forest?range={rng}", headers=h)
    assert r.status_code == 400
    assert (
        "30d" not in routes_forest.RANGES
        and max(routes_forest.RANGES.values()) == routes_forest.WHOLE_S
    )


def test_bad_range_is_400(tmp_data_dir, client):
    _, h = _ready(client, tmp_data_dir)
    r = client.get("/api/stats/forest?range=99d", headers=h)
    assert r.status_code == 400
    assert "range must be one of" in r.json()["detail"]


def test_unauthenticated_is_401(tmp_data_dir, client):
    client.get("/healthz")
    assert client.get("/api/stats/forest").status_code == 401


def test_since_returns_whole_grown_trees_with_stable_ids(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 300, sk="A"), _r(2, now - 400, sk="B", tok="tok2", name="other")])
    full = client.get("/api/stats/forest?range=6h", headers=h).json()
    ids = {t["id"] for t in full["trees"]}
    _insert(db_path, [_r(3, now - 10, sk="A", p=1600)])
    delta = client.get(f"/api/stats/forest?range=6h&since={full['cursor']}", headers=h).json()
    (tree,) = delta["trees"]
    assert tree["id"] in ids and len(tree["sessions"][0]["turns"]) == 2


def test_ids_lists_every_tree_in_full_and_since_responses(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 300, sk="A"), _r(2, now - 400, sk="B", tok="tok2", name="other")])
    full = client.get("/api/stats/forest?range=6h", headers=h).json()
    full_ids = {t["id"] for t in full["trees"]}
    assert len(full_ids) == 2
    assert set(full["ids"]) == full_ids and len(full["ids"]) == len(full_ids)
    _insert(db_path, [_r(3, now - 10, sk="A", p=1600)])
    delta = client.get(f"/api/stats/forest?range=6h&since={full['cursor']}", headers=h).json()
    assert len(delta["trees"]) == 1
    # the unchanged tree is not sent, but its id is still listed so the client keeps it
    assert set(delta["ids"]) == full_ids


def test_since_unchanged_old_tree_not_returned_but_recent_overlap_is(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    cursor = now - 100
    _insert(
        db_path,
        [
            _r(1, cursor - 6.5, sk="A", tok="tok1", name="old"),  # >= 6 s before cursor
            _r(2, cursor - 2, sk="B", tok="tok2", name="recent"),  # inside the 5 s overlap
        ],
    )
    assert client.get("/api/stats/forest?range=6h", headers=h).json()["full"] is True
    d = client.get(f"/api/stats/forest?range=6h&since={cursor}", headers=h).json()
    assert [t["key"] for t in d["trees"]] == ["recent"]
    assert len(d["ids"]) == 2 and d["full"] is False


def test_since_without_a_previous_state_is_full_and_complete(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 500, sk="A"), _r(2, now - 400, sk="B", tok="tok2", name="o")])
    d = client.get(f"/api/stats/forest?range=6h&since={now - 10}", headers=h).json()
    assert d["full"] is True and len(d["trees"]) == 2


def test_throughput_query_cached_between_calls(tmp_data_dir, client, monkeypatch):
    _, h = _ready(client, tmp_data_dir)
    calls = []
    real = forest.model_throughput
    monkeypatch.setattr(forest, "model_throughput", lambda rows: calls.append(1) or real(rows))
    assert routes_forest.TPUT_TTL_S == 60.0
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 200
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 200
    assert len(calls) == 1


def test_viewer_filter_limits_trees_to_token(tmp_data_dir, client):
    from app.stats.routes_forest import ForestViewer, forest_viewer

    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 20, sk="A"), _r(2, now - 10, sk="B", tok="tok2", name="other")])
    client.app.dependency_overrides[forest_viewer] = lambda: ForestViewer(token_id="tok1")
    try:
        body = client.get("/api/stats/forest?range=6h", headers=h).json()
    finally:
        client.app.dependency_overrides.pop(forest_viewer, None)
    assert [t["key"] for t in body["trees"]] == ["farm"]


def test_malformed_tools_in_does_not_500(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 20)])
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE request_history SET tools_in = '{not json'")
        db.commit()
    r = client.get("/api/stats/forest?range=6h", headers=h)
    assert r.status_code == 200 and len(r.json()["trees"]) == 1


def test_malformed_tools_in_entries_are_dropped_not_500(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 20), _r(2, now - 10, p=1100)])
    with sqlite3.connect(db_path) as db:
        db.execute('UPDATE request_history SET tools_in = \'[1, "x", ["read", 3, false]]\'')
        db.execute(
            'UPDATE request_history SET tools_in = \'[{"a": 1}, ["shell", 9, true], [2]]\','
            ' tools_out = \'[["edit", null], 7, ["read", "abcd1234"]]\' WHERE id = \'r2\''
        )
        db.commit()
    r = client.get("/api/stats/forest?range=6h", headers=h)
    assert r.status_code == 200
    (tree,) = r.json()["trees"]
    turns = tree["sessions"][0]["turns"]
    assert turns[0][4] == [["read", 3, False]]
    assert turns[1][4] == [["shell", 9, True]] and turns[1][5] == ["read"]
    assert tree["traits"]["fail"] == 0.5  # one valid failed entry of two valid ones


def test_tools_in_rows_with_only_garbage_entries_compute_fail_from_valid_entry(
    tmp_data_dir, client
):
    db_path, h = _ready(client, tmp_data_dir)
    _insert(db_path, [_r(1, time.time() - 20)])
    with sqlite3.connect(db_path) as db:
        db.execute('UPDATE request_history SET tools_in = \'[1, "x", ["read", 3, false]]\'')
        db.commit()
    r = client.get("/api/stats/forest?range=6h", headers=h)
    assert r.status_code == 200
    assert r.json()["trees"][0]["traits"]["fail"] == 0.0


from tests.unit.proxy.test_queue_wait_recorded import _seed_loaded  # noqa: E402


def _ready_key(client, tmp_data_dir):
    """Like _ready, but seeded by _seed_loaded (which also creates the admin and token "tok1")."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintext = _seed_loaded(db_path)
    return db_path, plaintext, jwt_login(client)


def _forest_login(client, plaintext):
    r = client.post("/api/forest/login", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.mark.parametrize("rng", ["1h", "6h", "24h", "7d"])
def test_key_holder_sees_only_own_trees(tmp_data_dir, client, rng):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)  # token row id "tok1"
    now = time.time()
    _insert(
        db_path,
        [_r(1, now - 100, tok="tok1", sk="A"), _r(2, now - 90, tok="tok2", name="other", sk="B")],
    )
    h = _forest_login(client, plaintext)
    body = client.get(f"/api/stats/forest?range={rng}", headers=h).json()
    assert [t["id"].split(":")[0] for t in body["trees"]] == ["tok1"]


def test_forest_token_is_refused_elsewhere(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    h = _forest_login(client, plaintext)
    assert client.get("/api/stats/v2/overview?range=1h", headers=h).status_code == 401


def test_revoked_key_ends_forest_session(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    h = _forest_login(client, plaintext)
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET revoked_at = datetime('now') WHERE id = 'tok1'")
        db.commit()
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 401


def test_paused_key_ends_forest_session(tmp_data_dir, client):
    db_path, plaintext, _ = _ready_key(client, tmp_data_dir)
    h = _forest_login(client, plaintext)
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE api_tokens SET paused_at = datetime('now') WHERE id = 'tok1'")
        db.commit()
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 403


def test_login_with_a_bad_key_is_401(tmp_data_dir, client):
    _ready_key(client, tmp_data_dir)
    r = client.post("/api/forest/login", headers={"Authorization": "Bearer vw_nope"})
    assert r.status_code == 401


def test_login_refuses_session_jwt_and_admin_token(tmp_data_dir, client):
    _, _, h = _ready_key(client, tmp_data_dir)
    assert client.post("/api/forest/login", headers=h).status_code == 401
    r = client.post("/api/forest/login", headers={"Authorization": "Bearer vwa_whatever"})
    assert r.status_code == 401


def test_expired_forest_jwt_is_401(tmp_data_dir, client):
    from app.auth.jwt import mint_forest

    _ready_key(client, tmp_data_dir)
    h = {"Authorization": f"Bearer {mint_forest('tok1', client.app.state.jwt_secret, -1)}"}
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 401


def test_forest_jwt_for_an_admin_token_row_is_401(tmp_data_dir, client):
    from app.auth.jwt import mint_forest
    from tests.conftest import seed_admin_token

    db_path, _, _ = _ready_key(client, tmp_data_dir)
    admin_id, _ = seed_admin_token(db_path)
    h = {"Authorization": f"Bearer {mint_forest(admin_id, client.app.state.jwt_secret, 5)}"}
    assert client.get("/api/stats/forest?range=6h", headers=h).status_code == 401


# ---- performance: off-loop, single-flight, cheap deltas, gzip ---------------------------------


def _count_builds(monkeypatch):
    """Per refresh that loaded rows: the rows it loaded (full or new). ``calls.trees``: the trees built (token, id)."""
    calls = _Calls()
    trees: list = []
    calls.trees = trees
    real_load, real_new, real_tree = (
        routes_forest._load,
        routes_forest._load_new,
        forest._build_tree,
    )

    def load(*a, **k):
        rows = real_load(*a, **k)
        calls.append(rows)
        return rows

    def load_new(*a, **k):
        rows = real_new(*a, **k)
        if rows:
            calls.append(rows)
        return rows

    def build_tree(token, tid, *a, **k):
        trees.append((token, tid))
        return real_tree(token, tid, *a, **k)

    monkeypatch.setattr(routes_forest, "_load", load)
    monkeypatch.setattr(routes_forest, "_load_new", load_new)
    monkeypatch.setattr(forest, "_build_tree", build_tree)
    return calls


class _Calls(list):
    trees: list


async def test_concurrent_requests_share_one_build(tmp_path, monkeypatch):
    from app.db.database import open_db
    from app.db.migrations import apply_migrations

    db_path = tmp_path / "f.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    now = time.time()
    _insert(db_path, [_r(1, now - 100), _r(2, now - 50, tok="tok2", sk="B")])
    monkeypatch.setattr(routes_forest, "FOREST_TTL_S", 3.0)
    calls: list = []
    real = routes_forest._store_refresh
    monkeypatch.setattr(routes_forest, "_store_refresh", lambda *a: calls.append(1) or real(*a))
    cache = routes_forest.ForestCache()
    key = ("6h", None)
    a, b = await asyncio.gather(cache.state(key, db_path), cache.state(key, db_path))
    assert a is b and len(calls) == 1
    # and a repeat inside the TTL reuses it
    assert await cache.state(key, db_path) is a and len(calls) == 1


async def test_cache_holds_a_bounded_number_of_keys(tmp_path, monkeypatch):
    from app.db.database import open_db
    from app.db.migrations import apply_migrations

    db_path = tmp_path / "f.db"
    async with open_db(db_path) as db:
        await apply_migrations(db)
    monkeypatch.setattr(routes_forest, "MAX_CACHED_KEYS", 2)
    cache = routes_forest.ForestCache()
    for tok in ("a", "b", "c"):
        await cache.state(("6h", tok), db_path)
    assert list(cache._states) == [("6h", "b"), ("6h", "c")]


def test_since_poll_builds_only_the_changed_token(tmp_data_dir, client, monkeypatch):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    rows = [
        _r(i, now - 3000 - k * 100 + i, tok=t, name=t, sk=f"{t}-S")
        for k, t in enumerate(("tok3", "tok2", "tok1"))
        for i in range(5)
    ]
    for i, row in enumerate(rows):
        rows[i] = (f"{row[0]}-{row[5]}", *row[1:])
    _insert(db_path, rows)
    full = client.get("/api/stats/forest?range=6h", headers=h).json()
    calls = _count_builds(monkeypatch)
    _insert(db_path, [_r(99, now - 5, tok="tok2", name="tok2", sk="tok2-S", p=2000)])
    d = client.get(f"/api/stats/forest?range=6h&since={full['cursor']}", headers=h).json()
    # only the new row is loaded, and only its tree is rebuilt
    assert len(calls) == 1 and [r["id"] for r in calls[0]] == ["r99"]
    assert {t for t, _ in calls.trees} == {"tok2"} and len(calls.trees) == 1
    # tok3 wrote the cursor's row, so its tree is inside the 5 s overlap and re-sent unbuilt
    assert sorted(t["key"] for t in d["trees"]) == ["tok2", "tok3"] and d["full"] is False
    assert sorted(d["ids"]) == sorted(full["ids"])


def test_since_poll_with_one_changed_token_of_fifty_is_cheap(tmp_data_dir, client, monkeypatch):
    """Perf smoke test: ~20k rows over 50 tokens, one token changes, and the poll loads and builds
    only that token's rows (counted, not timed)."""
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    rows = []
    for k in range(50):
        for i in range(400):
            row = _r(
                i, now - 20000 + i * 40 + k * 10, tok=f"t{k}", name=f"t{k}", sk=f"S{k}-{i // 50}"
            )
            rows.append((f"r{k}-{i}", *row[1:]))
    _insert(db_path, rows)
    full = client.get("/api/stats/forest?range=6h", headers=h).json()
    assert len({i.split(":")[0] for i in full["ids"]}) == 50
    calls = _count_builds(monkeypatch)
    _insert(db_path, [("new", *_r(0, now - 1, tok="t7", name="t7", sk="S7-7")[1:])])
    d = client.get(f"/api/stats/forest?range=6h&since={full['cursor']}", headers=h).json()
    assert len(calls) == 1
    assert len(calls[0]) == 1 and {r["token_id"] for r in calls[0]} == {"t7"}  # the new row only
    assert calls.trees == [("t7", calls.trees[0][1])]  # one tree rebuilt
    # t49 wrote the cursor's row: its tree is in the 5 s overlap (re-sent from cache, not rebuilt)
    assert sorted(t["key"] for t in d["trees"]) == ["t49", "t7"]
    assert sorted(d["ids"]) == sorted(full["ids"])
    # a poll with nothing new builds nothing at all
    d2 = client.get(f"/api/stats/forest?range=6h&since={d['cursor']}", headers=h).json()
    assert len(calls) == 1 and d2["full"] is False


def test_gzip_response_decodes_to_the_same_json(tmp_data_dir, client):
    db_path, h = _ready(client, tmp_data_dir)
    now = time.time()
    _insert(db_path, [_r(1, now - 100), _r(2, now - 50, p=1500, tok="tok2", sk="B")])
    routes_forest.FOREST_TTL_S = 60.0  # monkeypatched by the autouse fixture: both see one state
    plain = client.get("/api/stats/forest?range=6h", headers={**h, "Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    with client.stream(
        "GET", "/api/stats/forest?range=6h", headers={**h, "Accept-Encoding": "gzip"}
    ) as r:
        raw = b"".join(r.iter_raw())
        assert r.headers["content-encoding"] == "gzip"
        assert r.headers["vary"] == "Accept-Encoding"
    # ``now`` is each request's own time (C1); the rest is the one cached state's body.
    assert {**json.loads(gzip.decompress(raw)), "now": 0} == {**plain.json(), "now": 0}
    assert plain.json()["trees"]


def test_window_slide_that_moves_a_spell_id_sets_full(tmp_data_dir, client, monkeypatch):
    db_path, h = _ready(client, tmp_data_dir)
    T = time.time()
    monkeypatch.setattr(routes_forest, "_now", lambda: T)
    six = routes_forest.WHOLE_S  # trees are built over 7 d (sent whole): ids move at that edge
    # One spell: X just inside the 7 d build window, Y ten minutes later.
    # (tok2's row only moves the cursor past Y, so Y's tree is not in the delta's overlap.)
    _insert(
        db_path,
        [
            _r(1, T - six + 30, sk="X"),
            _r(2, T - six + 600, sk="Y"),
            _r(4, T - 10, sk="W", tok="tok2", name="other"),
        ],
    )
    full = client.get("/api/stats/forest?range=7d", headers=h).json()
    (old_id,) = [i for i in full["ids"] if i.startswith("tok1:")]
    # Two minutes later X has left the window, so the spell's id now follows Y, and the token
    # changed elsewhere (a new session far from the old spell).
    monkeypatch.setattr(routes_forest, "_now", lambda: T + 120)
    _insert(db_path, [_r(3, T + 100, sk="Z")])
    d = client.get(f"/api/stats/forest?range=7d&since={full['cursor']}", headers=h).json()
    assert old_id not in d["ids"] and len(d["ids"]) == 3
    assert d["full"] is True
    # the full=true response is complete: both current trees, the unchanged-but-renamed one too
    assert sorted(t["id"] for t in d["trees"]) == sorted(d["ids"])


async def test_inflight_is_scoped_for_forest_token(tmp_data_dir, client):
    from tests.unit.proxy.test_request_registry import _mk as _live

    _, plaintext, h_admin = _ready_key(client, tmp_data_dir)  # token row id "tok1"
    reg = client.app.state.request_registry
    await reg.register(_live(id="a", token_id="tok1"))
    await reg.register(_live(id="b", token_id="tok2"))
    h_forest = _forest_login(client, plaintext)
    admin = client.get("/api/stats/requests", headers=h_admin).json()
    forest_body = client.get("/api/stats/requests", headers=h_forest).json()
    assert {r["id"] for r in admin["requests"]} >= {"a", "b"}
    assert {r["id"] for r in forest_body["requests"]} == {"a"}
    assert forest_body["count"] == 1
    assert [t["token_id"] for t in forest_body["by_token"]] == ["tok1"]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("gzip", True),
        ("gzip;q=0", False),
        ("gzip;q=0.0", False),
        ("gzip;q=0.", False),
        ("*", True),
        ("*;q=0", False),
        ("*;q=1, gzip;q=0", False),
        ("*;q=0, gzip", True),
        ("gzip;q=0.5", True),
        ("identity", False),
        ("", False),
    ],
)
def test_accepts_gzip(header, expected):
    assert routes_forest._accepts_gzip(header) is expected


@pytest.mark.parametrize("enc", ["identity", "gzip"])
@pytest.mark.parametrize("delta", [False, True])
def test_now_is_the_time_of_each_request_even_from_the_cache(
    tmp_data_dir, client, monkeypatch, enc, delta
):
    """C1: a cached state answers for FOREST_TTL_S, but every response's ``now`` is that request's
    own time (a stale ``now`` makes the client's clock stall, then race)."""
    db_path, h = _ready(client, tmp_data_dir)
    T = time.time()
    _insert(db_path, [_r(1, T - 100), _r(2, T - 50, p=1500, tok="tok2", sk="B")])
    monkeypatch.setattr(routes_forest, "FOREST_TTL_S", 60.0)
    clock = {"t": T}
    monkeypatch.setattr(routes_forest, "_now", lambda: clock["t"])
    calls = []
    real = routes_forest._store_refresh
    monkeypatch.setattr(routes_forest, "_store_refresh", lambda *a: calls.append(1) or real(*a))
    hdr = {**h, "Accept-Encoding": enc}

    def get(since=None):
        q = "" if since is None else f"&since={since}"
        with client.stream("GET", f"/api/stats/forest?range=6h{q}", headers=hdr) as r:
            raw = b"".join(r.iter_raw())
            gz = r.headers.get("content-encoding") == "gzip"
        assert gz is (enc == "gzip")
        return json.loads(gzip.decompress(raw) if gz else raw)

    first = get()
    clock["t"] = T + 2.0
    second = get(first["cursor"] if delta else None)
    assert len(calls) == 1  # the second request was answered from the cache
    assert first["now"] == pytest.approx(T, abs=1e-3)
    assert second["now"] == pytest.approx(T + 2.0, abs=1e-3)
    assert second["t0"] == first["t0"]
    if not delta:
        assert {**second, "now": 0} == {**first, "now": 0}  # the rest of the body is the cached one


def test_spliced_gzip_is_a_standard_single_member_stream():
    """The cached tail's deflate behind a stored head block decodes with plain zlib as well."""
    import zlib

    head = b'{"range":"6h","now":123.5'
    tail = routes_forest._tail(b',"trees":[' + b'{"id":"x"},' * 5000 + b'{"id":"y"}]}')
    raw = routes_forest._gzip_spliced(head * 3000, tail)
    d = zlib.decompressobj(wbits=31)
    assert d.decompress(raw) + d.flush() == head * 3000 + tail.raw
    assert d.eof and not d.unused_data
    assert gzip.decompress(raw) == head * 3000 + tail.raw
