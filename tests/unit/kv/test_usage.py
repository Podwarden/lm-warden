"""Per-key and per-model KV cache reports.

Spec 2026-11-21 (Plan 2) section 6. A key's report has tokens served per tier, tokens
recomputed and the hit rate over 1 h, 24 h and 7 days (from ``request_history``), plus the
store's figures for the key (since the store started). Only an admin sees who evicted the
key's blocks and the isolation cost. ``GET /api/kv/usage`` (admin only) has usage, share
and evictions per model and per key.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import socketserver
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.tokens import TokenRepo
from app.kv import usage as usage_mod
from app.kv.usage import fetch_stats, key_cache_report, kv_usage, tenant_names
from lmw_kv.scopes import tenant_tag
from tests.conftest import csrf_header
from tests.unit.stats.test_forest_routes import _forest_login, _ready_key

NK = bytes(range(32))
STARTED = 1_700_000_000.0
STARTED_STR = "2023-11-14 22:13:20"
MODEL = "aa" * 16


def _tt(nk: bytes, tenant: str) -> str:
    return tenant_tag(nk, tenant).hex()


def _stats(nk: bytes) -> dict[str, Any]:
    """A daemon ``stats`` reply: tok1's family, a deleted family and the internal tenant."""
    t1, gone, internal = _tt(nk, "tok1"), _tt(nk, "gone-family"), _tt(nk, "internal")
    ev_t1 = {gone: 2, internal: 1, t1: 1}
    t1_row = {
        "ram_bytes": 300, "nvme_bytes": 2000, "ram_share": 500.0, "nvme_share": 4500.0,
        "ram_evictions": 3, "nvme_evictions": 1, "evicted_by": ev_t1,
    }  # fmt: skip
    gone_row = {
        "ram_bytes": 200, "nvme_bytes": 1000, "ram_share": 250.0, "nvme_share": 2250.0,
        "ram_evictions": 0, "nvme_evictions": 0, "evicted_by": {},
    }  # fmt: skip
    internal_row = {
        "ram_bytes": 100, "nvme_bytes": 0, "ram_share": 250.0, "nvme_share": 2250.0,
        "ram_evictions": 1, "nvme_evictions": 0, "evicted_by": {"": 1},
    }  # fmt: skip
    hits = {"ram_tokens": 0, "nvme_tokens": 0}
    return {
        "ram_bytes": 1000,
        "ram_used": 600,
        "nvme_used": 5000,
        "nvme_enabled": True,
        "blocks": 3,
        "evictions": 4,
        "nvme_evictions": 1,
        "policy": "lru",
        "started_at": STARTED,
        "nvme_cap_bytes": 9000,
        "retired_scopes": 0,
        "isolation_refused": False,
        "retired_persist_degraded": False,
        "dead_bytes": {"ram": 10, "nvme": 20},
        "by_tenant": {
            t1: {**t1_row, **hits},
            gone: {**gone_row, **hits},
            internal: {**internal_row, **hits},
        },
        "by_model": {
            MODEL: {
                "owners": ["qwen"],
                "weight": 1,
                "ram_part": 1000.0,
                "nvme_part": 9000.0,
                "ram_bytes": 600,
                "nvme_bytes": 3000,
                "tenants": {t1: t1_row, gone: gone_row, internal: internal_row},
            }
        },
        "by_namespace": {},
    }


def _history(db_path: Path, rows: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as db:
        for r in rows:
            base = {
                "model_id": "qwen",
                "model": "qwen",
                "token_id": "tok1",
                "token_name": "test",
                "completion_tokens": 5,
                "duration_s": 1.0,
                "started_iso": "2026-11-22T00:00:00Z",
                **r,
            }
            cols = ", ".join(base)
            db.execute(
                f"INSERT INTO request_history({cols}) VALUES ({', '.join('?' * len(base))})",
                tuple(base.values()),
            )
        db.commit()


def _tiers(g: int | None, r: int | None, n: int | None) -> dict[str, int | None]:
    return {"served_gpu_tokens": g, "served_ram_tokens": r, "served_nvme_tokens": n}


def _seed_history(db_path: Path, now: float) -> None:
    _history(
        db_path,
        [
            {"id": "a", "finished_at": now - 100, "prompt_tokens": 1000, **_tiers(100, 200, 300)},
            {"id": "b", "finished_at": now - 7200, "prompt_tokens": 500, **_tiers(50, 0, 0)},
            {"id": "c", "finished_at": now - 3 * 86400, "prompt_tokens": 100, **_tiers(10, 10, 10)},
            # Nothing known (neither the tiers nor the engine's cached count): left out.
            {"id": "d", "finished_at": now - 200, "prompt_tokens": 999, **_tiers(None, None, None)},
            # Store off on that turn: the GPU count is the engine's cached_tokens (R12).
            {
                "id": "j",
                "finished_at": now - 300,
                "prompt_tokens": 400,
                "cached_tokens": 300,
                **_tiers(None, None, None),
            },
            # Another key, and a row older than 7 days.
            {
                "id": "e",
                "finished_at": now - 100,
                "prompt_tokens": 70,
                "token_id": "tok2",
                **_tiers(7, 7, 7),
            },
            {"id": "f", "finished_at": now - 8 * 86400, "prompt_tokens": 9, **_tiers(1, 1, 1)},
            # Isolation cost: 24 h counts 40 + 2, the 3-day-old 1000 is outside.
            {"id": "g", "finished_at": now - 50, "kv_isolated": 1, "kv_iso_cost_tokens": 40},
            {"id": "h", "finished_at": now - 60, "kv_isolated": 1, "kv_iso_cost_tokens": 2},
            {
                "id": "i",
                "finished_at": now - 3 * 86400,
                "kv_isolated": 1,
                "kv_iso_cost_tokens": 1000,
            },
        ],
    )


#: Exact: integer sums, and the hit rate is the same division done in Python.
EXPECTED_WINDOWS = {
    "1h": {"gpu": 400, "ram": 200, "nvme": 300, "recomputed": 500, "hit_rate": 900 / 1400},
    "24h": {"gpu": 450, "ram": 200, "nvme": 300, "recomputed": 950, "hit_rate": 950 / 1900},
    "7d": {"gpu": 460, "ram": 210, "nvme": 310, "recomputed": 1020, "hit_rate": 980 / 2000},
}
WINDOWS_APPROX = EXPECTED_WINDOWS


@pytest.fixture
async def db(tmp_data_dir):
    async with open_db(tmp_data_dir / "vllm-warden.db") as conn:
        await apply_migrations(conn)
        await TokenRepo(conn).create("tok1", "test", "vw_tok1_" + "x" * 30)
        yield conn


# ---- the pure parts ------------------------------------------------------------------------


async def test_windows_give_exact_sums(db, tmp_data_dir):
    _seed_history(tmp_data_dir / "vllm-warden.db", time.time())
    rep = await key_cache_report(db, "tok1", None, NK, admin=False)
    assert rep["isolation"] == "sharing"
    assert rep["windows"] == WINDOWS_APPROX
    assert rep["store"] is None  # no daemon: windows intact


async def test_a_window_with_no_known_rows_has_no_hit_rate(db):
    rep = await key_cache_report(db, "tok1", None, NK, admin=True)
    assert rep["windows"]["1h"] == {
        "gpu": 0,
        "ram": None,
        "nvme": None,
        "recomputed": 0,
        "hit_rate": None,
    }
    assert rep["isolation_cost_tokens"] == {"24h": 0}


async def test_store_off_rows_count_the_gpu_and_leave_ram_and_nvme_null(db, tmp_data_dir):
    """R12: with the store off the GPU column is the engine's cached count, RAM and NVMe
    are null (not 0), and the hit rate comes from what is known."""
    now = 1_800_000_000.0
    _history(
        tmp_data_dir / "vllm-warden.db",
        [
            {"id": "s1", "finished_at": now - 10, "prompt_tokens": 1000, "cached_tokens": 600},
            {"id": "s2", "finished_at": now - 20, "prompt_tokens": 200, "cached_tokens": 0},
        ],
    )
    rep = await key_cache_report(db, "tok1", None, NK, admin=False, now=now)
    assert rep["windows"]["1h"] == {
        "gpu": 600,
        "ram": None,
        "nvme": None,
        "recomputed": 600,
        "hit_rate": 600 / 1200,
    }


async def test_counts_above_the_prompt_are_clamped(db, tmp_data_dir):
    """A row whose served counts exceed its prompt adds nothing to recomputed and at most
    its prompt to the hit rate, so hit_rate * prompt + recomputed == prompt."""
    now = 1_800_000_000.0
    _history(
        tmp_data_dir / "vllm-warden.db",
        [
            {"id": "o1", "finished_at": now - 10, "prompt_tokens": 40, **_tiers(50, 0, 0)},
            {"id": "o2", "finished_at": now - 20, "prompt_tokens": 60, **_tiers(0, 0, 0)},
        ],
    )
    w = (await key_cache_report(db, "tok1", None, NK, admin=False, now=now))["windows"]["1h"]
    assert w == {"gpu": 50, "ram": 0, "nvme": 0, "recomputed": 60, "hit_rate": 40 / 100}


async def test_window_edges_are_inclusive(db, tmp_data_dir):
    now = 1_800_000_000.0
    _history(
        tmp_data_dir / "vllm-warden.db",
        [
            {"id": "e1", "finished_at": now - 3600, "prompt_tokens": 10, **_tiers(1, 0, 0)},
            {"id": "e2", "finished_at": now - 3600.5, "prompt_tokens": 20, **_tiers(2, 0, 0)},
            {"id": "e3", "finished_at": now - 86400, "prompt_tokens": 40, **_tiers(4, 0, 0)},
            {"id": "e4", "finished_at": now - 7 * 86400, "prompt_tokens": 80, **_tiers(8, 0, 0)},
            {"id": "e5", "finished_at": now - 7 * 86400 - 1, "prompt_tokens": 1, **_tiers(1, 0, 0)},
        ],
    )
    w = (await key_cache_report(db, "tok1", None, NK, admin=False, now=now))["windows"]
    assert (w["1h"]["gpu"], w["24h"]["gpu"], w["7d"]["gpu"]) == (1, 7, 15)


async def test_admin_report_has_evicted_by_and_isolation_cost(db, tmp_data_dir):
    _seed_history(tmp_data_dir / "vllm-warden.db", time.time())
    rep = await key_cache_report(db, "tok1", _stats(NK), NK, admin=True)
    assert rep["store"] == {
        "ram_bytes": 300,
        "nvme_bytes": 2000,
        "ram_share": 0.5,
        "nvme_share": 0.5,
        "ram_share_bytes": 500,
        "nvme_share_bytes": 4500,
        "ram_evictions": 3,
        "nvme_evictions": 1,
        "since": STARTED_STR,
        # the deleted family's tag is "unknown"; most first, then by id
        "evicted_by": [
            {"token_id": "unknown", "name": "unknown", "count": 2},
            {"token_id": "internal", "name": "internal", "count": 1},
            {"token_id": "tok1", "name": "test", "count": 1},
        ],
    }
    assert rep["isolation_cost_tokens"] == {"24h": 42}


async def test_holder_report_has_own_totals_only(db, tmp_data_dir):
    _seed_history(tmp_data_dir / "vllm-warden.db", time.time())
    rep = await key_cache_report(db, "tok1", _stats(NK), NK, admin=False)
    assert set(rep) == {"isolation", "windows", "store"}
    assert rep["store"] == {
        "ram_bytes": 300,
        "nvme_bytes": 2000,
        "ram_share": 0.5,
        "nvme_share": 0.5,
        "ram_evictions": 3,
        "nvme_evictions": 1,
        "since": STARTED_STR,
    }


async def test_a_key_the_store_has_never_seen_reports_zeros(db):
    st = _stats(NK)
    st["by_tenant"] = {}
    rep = await key_cache_report(db, "tok1", st, NK, admin=True)
    assert rep["store"]["ram_bytes"] == 0 and rep["store"]["evicted_by"] == []


async def test_no_network_key_gives_no_store_block(db):
    rep = await key_cache_report(db, "tok1", _stats(NK), None, admin=True)
    assert rep["store"] is None


async def test_tenant_names_newest_of_the_family_internal_and_unknown(db):
    repo = TokenRepo(db)
    succ, _plain, _renamed = await repo.rotate("tok1", grace_hours=24)
    await repo.update(succ, name="test-2")
    names = tenant_names(await repo.list_all(), NK)
    assert names[tenant_tag(NK, "tok1")] == (succ, "test-2")
    assert names[tenant_tag(NK, "internal")] == ("internal", "internal")
    assert tenant_tag(NK, "gone-family") not in names


async def test_usage_report(db):
    names = tenant_names(await TokenRepo(db).list_all(), NK)
    u = kv_usage(_stats(NK), names)
    assert u["available"] is True
    assert u["policy"] == "lru" and u["since"] == STARTED_STR
    assert u["tiers"] == {
        "ram": {"capacity": 1000, "used": 600, "dead": 10},
        "nvme": {"capacity": 9000, "used": 5000, "dead": 20},
    }
    (m,) = u["models"]
    assert m["model_id"] == "qwen" and m["model_ids"] == ["qwen"]
    assert m["weight"] == 1
    assert m["part"] == {"ram": 1000, "nvme": 9000}
    assert m["used"] == {"ram": 600, "nvme": 3000, "total": 3600}
    assert [t["token_id"] for t in m["tenants"]] == ["tok1", "unknown", "internal"]
    t1 = m["tenants"][0]
    assert t1 == {
        "token_id": "tok1",
        "name": "test",
        "used": {"ram": 300, "nvme": 2000, "total": 2300},
        "share": {"ram": 500, "nvme": 4500},
        "evictions": {"ram": 3, "nvme": 1},
        "evicted_by": [
            {"token_id": "unknown", "name": "unknown", "count": 2},
            {"token_id": "internal", "name": "internal", "count": 1},
            {"token_id": "tok1", "name": "test", "count": 1},
        ],
    }
    # an eviction with no taker (b"" from the store) is "unknown" too
    assert m["tenants"][2]["evicted_by"] == [{"token_id": "unknown", "name": "unknown", "count": 1}]


def test_a_model_with_no_owner_left_is_named_by_its_last_owner():
    st = _stats(NK)
    st["by_model"][MODEL].update(owners=[], last_owner="qwen-old")
    assert kv_usage(st, {})["models"][0]["model_id"] == "qwen-old"
    st["by_model"][MODEL]["last_owner"] = ""
    assert kv_usage(st, {})["models"][0]["model_id"] == "unknown"


def test_usage_with_no_daemon():
    assert kv_usage(None, {}) == {
        "available": False,
        "policy": None,
        "since": None,
        "tiers": None,
        "models": [],
    }


def test_an_old_daemon_reply_reads_as_unknown_since_and_zero_evictions():
    st = _stats(NK)
    del st["started_at"]
    for row in st["by_model"][MODEL]["tenants"].values():
        for k in ("ram_evictions", "nvme_evictions", "evicted_by"):
            del row[k]
    u = kv_usage(st, {})
    assert u["since"] is None
    t = u["models"][0]["tenants"][0]
    assert t["evictions"] == {"ram": 0, "nvme": 0} and t["evicted_by"] == []


@pytest.mark.parametrize("bad", ["nope", {"by_model": "x", "by_tenant": [1]}, {"ram_bytes": "1"}])
def test_a_malformed_reply_never_fails_and_is_never_logged(bad, caplog):
    caplog.set_level(logging.DEBUG)
    u = kv_usage(bad, {})  # type: ignore[arg-type]
    assert u["models"] == []
    assert all("nope" not in r.getMessage() for r in caplog.records)


# ---- the routes ----------------------------------------------------------------------------


class _Client:
    def __init__(self, reply: dict[str, Any] | None, delay_s: float = 0.0) -> None:
        self.reply = reply
        self.calls = 0
        self.delay_s = delay_s

    def stats(self) -> dict[str, Any] | None:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.reply


class _HotClient:
    """The hot-path client: a report must never call ``stats`` on it."""

    def stats(self) -> None:
        raise AssertionError("stats on the hot-path client")


class _Sup:
    def __init__(self, client: Any) -> None:
        self._client = client

    def client(self) -> Any:
        return _HotClient() if self._client is not None else None

    def stats_client(self) -> Any:
        return self._client

    async def stop(self) -> None:
        return None


async def test_the_stats_reply_is_cached_with_one_call_in_flight(monkeypatch):
    c = _Client({"policy": "lru"}, delay_s=0.05)
    state = SimpleNamespace(kv_store=_Sup(c))
    got = await asyncio.gather(*(fetch_stats(state) for _ in range(5)))
    assert got == [{"policy": "lru"}] * 5 and c.calls == 1
    assert await fetch_stats(state) == {"policy": "lru"} and c.calls == 1  # within the TTL
    monkeypatch.setattr(usage_mod, "STATS_TTL_S", 0.0)
    await fetch_stats(state)
    assert c.calls == 2
    state.kv_store = _Sup(_Client(None))  # a new supervisor is never served the old reply
    assert await fetch_stats(state) is None


async def test_a_slow_stats_reply_never_nulls_the_tiers_of_turns_finishing_meanwhile(tmp_path):
    """A real ``lmw_kv`` client against a socket whose ``stats`` takes 0.3 s: the report
    waits on its own client, and the hot-path client on the same worker thread still gets
    every turn's hits (before the fix the 50 ms client timed out on stats, closed its
    connection and returned None for the next 0.5 s)."""
    from lmw_kv.client import Client

    sock = str(Path(tempfile.mkdtemp(prefix="kvs")) / "s.sock")

    class H(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for line in self.rfile:
                r = json.loads(line)
                if r["op"] == "stats":
                    time.sleep(0.3)
                out = {"op": r["op"], "id": r["id"], "ok": True, "stats": {"policy": "lru"}}
                out["hits"] = {}
                self.wfile.write(json.dumps(out).encode() + b"\n")

    server = socketserver.ThreadingUnixStreamServer(sock, H)
    server.daemon_threads = True
    server.block_on_close = False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    hot = Client(sock, timeout_s=0.05)
    slow = Client(sock, timeout_s=2.0)

    class Sup:
        def client(self) -> Client:
            return hot

        def stats_client(self) -> Client:
            return slow

    loop = asyncio.get_running_loop()
    one = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # one worker thread for all
    loop.set_default_executor(one)
    try:
        assert await fetch_stats(SimpleNamespace(kv_store=Sup())) == {"policy": "lru"}
        assert await asyncio.to_thread(hot.pop_hits, "req-1") == {}
    finally:
        hot.close()
        slow.close()
        server.shutdown()
        server.server_close()


def _routes(client, tmp_data_dir, reply: Any = "stats"):
    db_path, plaintext, admin = _ready_key(client, tmp_data_dir)
    nk = client.app.state.cobs2_key.key
    stats = _stats(nk) if reply == "stats" else reply
    client.app.state.kv_store = _Sup(_Client(stats) if stats is not None else None)
    _seed_history(db_path, time.time())
    return plaintext, admin


def test_admin_and_holder_key_cache_routes(tmp_data_dir, client):
    plaintext, admin = _routes(client, tmp_data_dir)
    a = client.get("/api/tokens/tok1/cache", headers=admin)
    assert a.status_code == 200, a.text
    body = a.json()
    assert "models" in body  # the prompt-cache observation is still there
    assert body["windows"] == WINDOWS_APPROX
    assert body["store"]["evicted_by"][0] == {"token_id": "unknown", "name": "unknown", "count": 2}
    assert body["isolation_cost_tokens"] == {"24h": 42}

    k = _forest_login(client, plaintext)
    h = client.get("/api/me/key/cache", headers=k)
    assert h.status_code == 200, h.text
    hb = h.json()
    assert set(hb) == {"isolation", "windows", "store", "top_reused"}
    assert "evicted_by" not in hb["store"] and "ram_share_bytes" not in hb["store"]
    assert hb["windows"] == WINDOWS_APPROX
    # the holder's older prompt-cache page shows no KV store figures
    assert "store" not in client.get("/api/me/cache", headers=k).json()


@pytest.mark.parametrize("reply", [None, "down"])
def test_daemon_down_gives_store_null_with_windows_intact(tmp_data_dir, client, reply):
    plaintext, admin = _routes(client, tmp_data_dir, None if reply is None else "stats")
    if reply == "down":
        client.app.state.kv_store = _Sup(_Client(None))  # stats call failed
    body = client.get("/api/tokens/tok1/cache", headers=admin).json()
    assert body["store"] is None
    assert body["windows"] == WINDOWS_APPROX
    hb = client.get("/api/me/key/cache", headers=_forest_login(client, plaintext)).json()
    assert hb["store"] is None and hb["windows"] == WINDOWS_APPROX
    u = client.get("/api/kv/usage", headers=admin)
    assert u.status_code == 200 and u.json()["available"] is False


def test_a_store_that_raises_is_down(tmp_data_dir, client):
    _plain, admin = _routes(client, tmp_data_dir)

    class Boom:
        def stats(self) -> None:
            raise OSError("socket gone")

    client.app.state.kv_store = _Sup(Boom())
    assert client.get("/api/tokens/tok1/cache", headers=admin).json()["store"] is None
    assert client.get("/api/kv/usage", headers=admin).json()["available"] is False


def test_usage_route_names_keys_and_carries_the_nvme_capacity(tmp_data_dir, client):
    _plain, admin = _routes(client, tmp_data_dir)
    r = client.get("/api/kv/usage", headers=admin)
    assert r.status_code == 200, r.text
    u = r.json()
    assert u["tiers"]["nvme"]["capacity"] == 9000
    names = [(t["token_id"], t["name"]) for t in u["models"][0]["tenants"]]
    assert names == [("tok1", "test"), ("unknown", "unknown"), ("internal", "internal")]


def test_usage_route_is_admin_only(tmp_data_dir, client):
    """A key holder's sign-in and an inference key are refused. The answer is 401, as on
    every other admin route (tests/unit/auth/test_auth_matrix.py pins it)."""
    plaintext, _admin = _routes(client, tmp_data_dir)
    holder = _forest_login(client, plaintext)
    assert client.get("/api/kv/usage", headers=holder).status_code == 401
    key = {"Authorization": f"Bearer {plaintext}"}
    assert client.get("/api/kv/usage", headers=key).status_code == 401
    assert client.get("/api/kv/usage").status_code == 401


def test_a_deleted_key_shows_as_unknown_and_nothing_fails(tmp_data_dir, client):
    _plain, admin = _routes(client, tmp_data_dir)
    r = client.delete("/api/tokens/tok1", headers={**admin, **csrf_header(client)})
    assert r.status_code == 204, r.text
    u = client.get("/api/kv/usage", headers=admin)
    assert u.status_code == 200
    tenants = u.json()["models"][0]["tenants"]
    assert [t["token_id"] for t in tenants] == ["unknown", "unknown", "internal"]
    assert client.get("/api/tokens/tok1/cache", headers=admin).status_code == 404


@pytest.mark.parametrize(
    ("path", "schema"),
    [
        ("/api/kv/usage", "KvUsage"),
        ("/api/me/key/cache", "KvKeyReport"),
        ("/api/tokens/{token_id}/cache", "TokenCacheOut"),
    ],
)
def test_the_report_routes_have_typed_responses(client, path, schema):
    """R13: the generated api types get the bodies, not ``{[key: string]: unknown}``."""
    doc = client.app.openapi()
    body = doc["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]
    assert body["schema"] == {"$ref": f"#/components/schemas/{schema}"}
    windows = doc["components"]["schemas"]["KvWindows"]
    assert sorted(windows["properties"]) == ["1h", "24h", "7d"]
