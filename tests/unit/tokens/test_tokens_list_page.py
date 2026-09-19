"""GET /api/tokens: server-side sort, paging, name search, and the counts.

The list is built for millions of keys (app/db/repos/tokens.py,
list_token_page), so beyond "the right rows in the right order" these tests pin
that a page is a bounded number of statements whatever the table size.

Expected orders are computed here from the API's own item fields -- the status
rank from the same fields the badge reads (token-row.tsx deriveStatus) -- so
the SQL ORDER BY is checked against the badge, not against a copy of itself.
"""

import sqlite3
import time
import typing
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from app.db.repos import tokens as token_repo
from app.db.repos.tokens import sqlite_utc_in
from app.tokens import routes_api
from tests.conftest import jwt_login, seed_admin_user

SORTS = [
    "name", "prefix", "created", "expires", "last_used", "priority", "usage_24h", "status",
]

Item = dict[str, Any]


def _ts(days: float) -> str:
    return sqlite_utc_in(timedelta(days=days))


def _setup(client: Any, db_path: Path) -> dict[str, str]:
    client.get("/healthz")
    seed_admin_user(db_path)
    return jwt_login(client)


def _write(db_path: Path, tokens: list[dict[str, Any]], usage: list[tuple] = ()) -> None:
    """Insert rows straight into api_tokens (timestamps the API cannot set),
    through the same WAL PRAGMAs the app uses (see seed_admin_user)."""
    cols = (
        "id", "name", "prefix", "created_at", "last_used_at", "revoked_at",
        "expires_at", "rotated_at", "rotated_from", "priority", "paused_at",
    )
    rows = [
        (
            t["id"], t.get("name", t["id"]), t.get("prefix", t["id"][:8]),
            t.get("created_at", _ts(-10)), t.get("last_used_at"), t.get("revoked_at"),
            t.get("expires_at"), t.get("rotated_at"), t.get("rotated_from"),
            t.get("priority", 5), t.get("paused_at"),
        )
        for t in tokens
    ]
    with sqlite3.connect(db_path, isolation_level=None) as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("BEGIN IMMEDIATE")
        db.executemany(
            f"INSERT INTO api_tokens({', '.join(cols)}, hash) "
            f"VALUES ({', '.join('?' for _ in cols)}, 'h-' || ?1)",
            rows,
        )
        db.executemany(
            "INSERT INTO token_usage_minute"
            "(token_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?)",
            list(usage),
        )
        db.execute("COMMIT")
        db.execute("PRAGMA wal_checkpoint(FULL)")


def _get(client: Any, auth: dict[str, str], **params: Any) -> Any:
    return client.get("/api/tokens", headers=auth, params=params)


def _page(client: Any, auth: dict[str, str], **params: Any) -> dict[str, Any]:
    r = _get(client, auth, **params)
    assert r.status_code == 200, r.text
    return typing.cast(dict[str, Any], r.json())


def _ids(items: list[Item]) -> list[str]:
    return [it["id"] for it in items]


# ---------------------------------------------------------------------------
# A fixture set that exercises every key: ties, NULLs, and every badge.
# ---------------------------------------------------------------------------

NOW_MIN = int(time.time() // 60)


def _fixture_tokens() -> list[dict[str, Any]]:
    same = _ts(-20)
    return [
        # name ties under NOCASE (Bravo / bravo), created_at ties (same),
        # priority ties, NULL last_used / expires.
        {"id": "a01", "name": "alpha", "created_at": same, "priority": 5,
         "expires_at": _ts(200), "last_used_at": _ts(-1)},
        {"id": "a02", "name": "Bravo", "created_at": same, "priority": 5,
         "prefix": "vw_same0"},
        {"id": "a03", "name": "bravo", "created_at": same, "priority": 9,
         "prefix": "vw_same0", "last_used_at": _ts(-1)},
        {"id": "a04", "name": "charlie", "created_at": _ts(-5), "priority": 0,
         "expires_at": _ts(10)},  # Expiring soon
        {"id": "a05", "name": "delta", "created_at": _ts(-30), "priority": 3,
         "expires_at": _ts(-1), "last_used_at": _ts(-2)},  # Expired
        {"id": "a06", "name": "echo", "created_at": _ts(-40), "priority": 7,
         "paused_at": _ts(-1), "expires_at": _ts(5)},  # Paused (and near expiry)
        # Rotation chain with the grace window still open: a07 -> a08.
        {"id": "a07", "name": "fox (old 1)", "created_at": _ts(-50), "rotated_at": _ts(-1),
         "revoked_at": _ts(0.5)},  # Grace
        {"id": "a08", "name": "fox", "created_at": _ts(-1), "rotated_from": "a07"},
        # Grace over: a09 -> a10.
        {"id": "a09", "name": "golf (old 1)", "created_at": _ts(-60), "rotated_at": _ts(-3),
         "revoked_at": _ts(-2)},  # Rotated (revoked)
        {"id": "a10", "name": "golf", "created_at": _ts(-3), "rotated_from": "a09",
         "last_used_at": _ts(-0.5)},
        # Orphan: rotated, successor deleted (never inserted).
        {"id": "a11", "name": "hotel (old 1)", "created_at": _ts(-70), "rotated_at": _ts(-1),
         "revoked_at": _ts(1)},  # Rotated (orphan)
        # Expired AND past its grace -> the badge (and rank) says Expired.
        {"id": "a12", "name": "india (old 1)", "created_at": _ts(-80), "rotated_at": _ts(-4),
         "revoked_at": _ts(-3), "expires_at": _ts(-1)},
        {"id": "a13", "name": "india", "created_at": _ts(-4), "rotated_from": "a12"},
        # Hidden: revoked without a rotation.
        {"id": "a14", "name": "juliet", "created_at": _ts(-2), "revoked_at": _ts(-1),
         "expires_at": _ts(3)},
    ]


def _fixture_usage() -> list[tuple]:
    return [
        ("a01", NOW_MIN - 10, 2, 100, 50),  # 150
        ("a01", NOW_MIN - 2000, 9, 9000, 9000),  # older than 24h: not counted
        ("a03", NOW_MIN - 5, 1, 150, 0),  # 150 -- ties a01
        ("a08", NOW_MIN - 1, 3, 1000, 1),  # 1001
        ("a14", NOW_MIN - 1, 3, 5000, 5000),  # hidden row
    ]


VISIBLE = [f"a{i:02d}" for i in range(1, 14)]


def _status_rank(it: Item) -> int:
    """The badge ladder (token-row.tsx deriveStatus) as a rank."""
    if it["is_paused"]:
        return 0
    if it["revoked_at"] is not None and it["rotated_at"] is None:
        return 1
    if it["is_expired"]:
        return 2
    if it["rotated_at"] is not None and (it["successor_deleted"] or it["is_revoked"]):
        return 1
    if it["rotated_at"] is not None:
        return 3
    if it["is_near_expiry"]:
        return 4
    return 5


KEYS: dict[str, Callable[[Item], Any]] = {
    "name": lambda it: it["name"].lower(),
    "prefix": lambda it: it["prefix"],
    "created": lambda it: it["created_at"],
    "expires": lambda it: it["expires_at"],
    "last_used": lambda it: it["last_used_at"],
    "priority": lambda it: it["priority"],
    "usage_24h": lambda it: it["usage_24h"]["total_tokens"],
    "status": _status_rank,
}


def _expected(items: list[Item], sort: str, desc: bool) -> list[str]:
    key = KEYS[sort]
    present = [it for it in items if key(it) is not None]
    absent = [it for it in items if key(it) is None]
    present.sort(key=lambda it: (key(it), it["id"]), reverse=desc)
    absent.sort(key=lambda it: it["id"], reverse=desc)
    return _ids(present + absent)


@pytest.fixture
def seeded(client: Any, tmp_data_dir: Path) -> tuple[Any, dict[str, str]]:
    db_path = tmp_data_dir / "vllm-warden.db"
    auth = _setup(client, db_path)
    _write(db_path, _fixture_tokens(), _fixture_usage())
    return client, auth


def test_route_literal_and_repo_keys_agree() -> None:
    hints = typing.get_type_hints(routes_api.list_tokens)
    assert set(typing.get_args(hints["sort"])) == set(token_repo.TOKEN_SORT_KEYS) == set(SORTS)


def test_defaults_and_response_shape(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    body = _page(client, auth)
    assert set(body) == {"items", "total", "limit", "offset", "near_expiry"}
    assert (body["limit"], body["offset"], body["total"]) == (50, 0, len(VISIBLE))
    # Default sort: created, newest first, id breaking the three-way tie.
    assert _ids(body["items"]) == _expected(body["items"], "created", True)
    # Every field the UI reads is still there.
    for field in (
        "id", "name", "prefix", "preview", "created_at", "last_used_at", "expires_at",
        "rotated_at", "rotated_from", "successor_id", "successor_deleted", "is_expired",
        "is_near_expiry", "revoked_at", "is_revoked", "priority", "usage_24h",
        "paused_at", "is_paused",
    ):
        assert field in body["items"][0], field


@pytest.mark.parametrize("sort", SORTS)
@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_every_sort_orders_with_nulls_last_and_id_ties(
    seeded: tuple[Any, dict[str, str]], sort: str, direction: str,
) -> None:
    client, auth = seeded
    body = _page(client, auth, sort=sort, dir=direction, limit=500)
    items = body["items"]
    assert sorted(_ids(items)) == VISIBLE
    assert _ids(items) == _expected(items, sort, direction == "desc")
    # Walking the list three rows at a time reproduces it exactly: the id
    # tie-break makes every page boundary stable.
    walked: list[str] = []
    for offset in range(0, len(VISIBLE), 3):
        walked += _ids(_page(client, auth, sort=sort, dir=direction, limit=3,
                             offset=offset)["items"])
    assert walked == _ids(items)


@pytest.mark.parametrize("sort", ["expires", "last_used"])
def test_nulls_sort_last_in_both_directions(
    seeded: tuple[Any, dict[str, str]], sort: str,
) -> None:
    client, auth = seeded
    field = "expires_at" if sort == "expires" else "last_used_at"
    for direction in ("asc", "desc"):
        vals = [it[field] for it in _page(client, auth, sort=sort, dir=direction)["items"]]
        first_null = vals.index(None)
        assert all(v is None for v in vals[first_null:]), (direction, vals)
        assert all(v is not None for v in vals[:first_null])


def test_status_sort_follows_the_badge_ladder(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    ids = _ids(_page(client, auth, sort="status", dir="asc")["items"])
    # Paused, then the red rotated badges (grace over, orphan), Expired (incl.
    # an expired key past its grace), Grace, Expiring soon, Active.
    assert ids[0] == "a06"
    assert ids[1:3] == ["a09", "a11"]
    assert ids[3:5] == ["a05", "a12"]
    assert ids[5] == "a07"
    assert ids[6] == "a04"
    assert ids[7:] == sorted(["a01", "a02", "a03", "a08", "a10", "a13"])


def test_name_sort_is_case_insensitive_with_id_tiebreak(
    seeded: tuple[Any, dict[str, str]],
) -> None:
    client, auth = seeded
    names = [(it["name"], it["id"]) for it in
             _page(client, auth, sort="name", dir="asc", limit=3)["items"]]
    assert names == [("alpha", "a01"), ("Bravo", "a02"), ("bravo", "a03")]


def test_usage_24h_sort_counts_only_the_last_day(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    items = _page(client, auth, sort="usage_24h", dir="desc", limit=3)["items"]
    assert _ids(items) == ["a08", "a03", "a01"]  # 1001, then the 150 tie by id desc
    a01 = items[2]["usage_24h"]
    assert a01 == {"requests": 2, "prompt_tokens": 100, "completion_tokens": 50,
                   "total_tokens": 150}


def test_filter_parity_hides_plain_revoked_only(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    body = _page(client, auth, limit=500)
    ids = set(_ids(body["items"]))
    assert "a14" not in ids  # revoked, never rotated
    assert {"a07", "a09", "a11", "a12"} <= ids  # rotated predecessors stay
    assert body["total"] == len(VISIBLE)


def test_successor_fields(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    by_id = {it["id"]: it for it in _page(client, auth, limit=500)["items"]}
    assert by_id["a07"]["successor_id"] == "a08"
    assert by_id["a07"]["successor_deleted"] is False
    assert by_id["a11"]["successor_id"] is None
    assert by_id["a11"]["successor_deleted"] is True
    assert by_id["a08"]["successor_id"] is None
    assert by_id["a08"]["successor_deleted"] is False


def test_near_expiry_counts_the_whole_visible_set(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    # a04 (10 days) and a06 (5 days, paused) -- not a01 (200 days), not the
    # expired a05/a12, not the hidden a14 -- and not just the page's rows.
    body = _page(client, auth, limit=1, sort="name", dir="asc")
    assert body["near_expiry"] == 2
    assert _ids(body["items"]) == ["a01"]


def test_near_expiry_filter_is_the_near_expiry_count(
    seeded: tuple[Any, dict[str, str]],
) -> None:
    client, auth = seeded
    body = _page(client, auth, near_expiry=1, sort="expires", dir="asc")
    # a06 (5 days) then a04 (10 days); not the hidden a14, the expired a05 /
    # a12, the 200-day a01 or anything that never expires.
    assert _ids(body["items"]) == ["a06", "a04"]
    assert body["total"] == body["near_expiry"] == 2
    assert all(it["is_near_expiry"] for it in body["items"])
    paged = _page(client, auth, near_expiry=1, sort="expires", dir="asc", limit=1, offset=1)
    assert _ids(paged["items"]) == ["a04"] and paged["total"] == 2
    assert _page(client, auth, near_expiry=0)["total"] == len(VISIBLE)


def test_near_expiry_filter_combines_with_search(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    body = _page(client, auth, near_expiry=1, q="CHAR")
    assert _ids(body["items"]) == ["a04"]
    assert body["total"] == body["near_expiry"] == 1


def test_pagination_bounds_and_totals(seeded: tuple[Any, dict[str, str]]) -> None:
    client, auth = seeded
    body = _page(client, auth, limit=5, offset=10)
    assert (len(body["items"]), body["total"], body["limit"], body["offset"]) == (3, 13, 5, 10)
    past = _page(client, auth, limit=5, offset=100)
    assert past["items"] == [] and past["total"] == 13
    assert len(_page(client, auth, limit=500)["items"]) == 13
    assert len(_page(client, auth, limit=1)["items"]) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0}, {"limit": 501}, {"limit": -1}, {"offset": -1}, {"limit": "x"},
        {"sort": "hash"}, {"sort": ""}, {"dir": "up"}, {"dir": "DESC"}, {"q": "x" * 65},
        {"near_expiry": 2}, {"near_expiry": -1}, {"near_expiry": "yes"},
    ],
)
def test_bad_params_are_422(seeded: tuple[Any, dict[str, str]], params: dict[str, Any]) -> None:
    client, auth = seeded
    assert _get(client, auth, **params).status_code == 422


# ---------------------------------------------------------------------------
# Name search (q)
# ---------------------------------------------------------------------------


@pytest.fixture
def named(client: Any, tmp_data_dir: Path) -> tuple[Any, dict[str, str]]:
    db_path = tmp_data_dir / "vllm-warden.db"
    auth = _setup(client, db_path)
    names = {
        "s01": "Prod-Bot", "s02": "prod-bot (old 1)", "s03": "staging-bot",
        "s04": "50%_off", "s05": "50xxoff", "s06": "a_b", "s07": "axb",
        "s08": "back\\slash", "s09": "backslash", "s10": "PRODUCTION",
    }
    rows: list[dict[str, Any]] = [{"id": k, "name": v} for k, v in names.items()]
    rows[1].update(rotated_at=_ts(-1), revoked_at=_ts(1))
    rows[9].update(expires_at=_ts(3))
    # A hidden key that matches "prod" must not be found or counted.
    rows.append({"id": "s11", "name": "prod-hidden", "revoked_at": _ts(-1), "expires_at": _ts(2)})
    _write(db_path, rows)
    return client, auth


def _names(body: dict[str, Any]) -> list[str]:
    return sorted(it["name"] for it in body["items"])


def test_search_is_a_case_insensitive_substring(named: tuple[Any, dict[str, str]]) -> None:
    client, auth = named
    body = _page(client, auth, q="PROD")
    assert _names(body) == ["PRODUCTION", "Prod-Bot", "prod-bot (old 1)"]
    assert body["total"] == 3
    assert body["near_expiry"] == 1  # PRODUCTION; the hidden match is not counted
    assert _names(_page(client, auth, q="-bot")) == [
        "Prod-Bot", "prod-bot (old 1)", "staging-bot",
    ]


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("%", ["50%_off"]),
        ("_", ["50%_off", "a_b"]),
        ("50%_", ["50%_off"]),
        ("a_b", ["a_b"]),
        ("\\", ["back\\slash"]),
        ("k\\s", ["back\\slash"]),
    ],
)
def test_search_wildcards_are_literal(
    named: tuple[Any, dict[str, str]], q: str, expected: list[str],
) -> None:
    client, auth = named
    assert _names(_page(client, auth, q=q)) == expected


def test_search_trims_and_blank_means_no_filter(named: tuple[Any, dict[str, str]]) -> None:
    client, auth = named
    assert _page(client, auth, q="   ")["total"] == 10
    assert _page(client, auth, q="")["total"] == 10
    assert _names(_page(client, auth, q="  staging ")) == ["staging-bot"]


def test_search_pages_and_sorts(named: tuple[Any, dict[str, str]]) -> None:
    client, auth = named
    first = _page(client, auth, q="bot", sort="name", dir="desc", limit=2)
    second = _page(client, auth, q="bot", sort="name", dir="desc", limit=2, offset=2)
    assert [it["name"] for it in first["items"]] == ["staging-bot", "prod-bot (old 1)"]
    assert [it["name"] for it in second["items"]] == ["Prod-Bot"]
    assert first["total"] == second["total"] == 3


# ---------------------------------------------------------------------------
# Cost: a page is a bounded number of statements, not one per token.
# ---------------------------------------------------------------------------


def test_a_page_of_200_tokens_is_a_constant_number_of_statements(
    client: Any, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_data_dir / "vllm-warden.db"
    auth = _setup(client, db_path)
    rows: list[dict[str, Any]] = [{"id": f"t{i:04d}"} for i in range(200)]
    for i in range(0, 200, 2):  # half of them rotated, each with a successor
        rows[i].update(rotated_at=_ts(-1), revoked_at=_ts(1))
        rows[i + 1]["rotated_from"] = rows[i]["id"]
    _write(db_path, rows, [(f"t{i:04d}", NOW_MIN - 1, 1, 10, 10) for i in range(200)])

    statements: list[str] = []
    real_open_db = routes_api.open_db

    @asynccontextmanager
    async def counting_open_db(path: Path) -> Any:
        async with real_open_db(path) as db:
            await db.set_trace_callback(statements.append)
            yield db

    async def no_per_token_totals(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the list must not query usage per token")

    monkeypatch.setattr(routes_api, "open_db", counting_open_db)
    monkeypatch.setattr(token_repo.TokenUsageRepo, "totals", no_per_token_totals)

    for sort in SORTS:
        statements.clear()
        body = _page(client, auth, sort=sort, limit=200)
        assert len(body["items"]) == 200
        assert all(it["usage_24h"]["total_tokens"] == 20 for it in body["items"])
        selects = [s for s in statements if s.lstrip().upper().startswith(("SELECT", "WITH"))]
        assert len(selects) == 2, (sort, selects)


# ---------------------------------------------------------------------------
# Scale sanity: 20k keys, every page well under a second.
# ---------------------------------------------------------------------------


def _bulk(n: int) -> Iterator[dict[str, Any]]:
    for i in range(n):
        yield {
            "id": f"{i:032x}",
            "name": f"key-{(i * 7919) % n}",
            "created_at": f"2026-0{1 + i % 9}-{1 + i % 28:02d} {i % 24:02d}:00:00",
            "last_used_at": None if i % 3 == 0 else f"2026-09-{1 + i % 17:02d} 00:00:00",
            "expires_at": None if i % 4 == 0 else f"2027-0{1 + i % 9}-01 00:00:00",
            "priority": i % 10,
            "revoked_at": _ts(-1) if i % 20 == 0 else None,
        }


@pytest.mark.perf
def test_twenty_thousand_tokens_page_quickly(client: Any, tmp_data_dir: Path) -> None:
    db_path = tmp_data_dir / "vllm-warden.db"
    auth = _setup(client, db_path)
    n = 20_000
    _write(db_path, list(_bulk(n)),
           [(f"{i:032x}", NOW_MIN - i % 1000, 1, i, i) for i in range(0, n, 3)])
    visible = n - n // 20
    for params in (
        {"sort": "created", "dir": "desc"},
        {"sort": "name", "dir": "asc", "offset": n // 2},
        {"sort": "expires", "dir": "asc"},
        {"sort": "last_used", "dir": "desc"},
        {"sort": "usage_24h", "dir": "desc"},
        {"sort": "status", "dir": "asc", "offset": n // 3},
        {"sort": "priority", "dir": "desc", "q": "key-1"},
    ):
        start = time.perf_counter()
        body = _page(client, auth, limit=100, **params)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, (params, elapsed)
        assert len(body["items"]) == 100
        if "q" not in params:
            assert body["total"] == visible
