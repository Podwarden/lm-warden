"""Reuse and disk write figures in the KV cache reports (disk-only spec 6, "Reports").

The daemon's ``stats.reuse`` maps and its ``top_reused`` op reach three reports:
``GET /api/kv/usage`` (admin: per model and per key, and a top list per model),
``GET /api/tokens/{id}/cache`` (the key's admin figures) and ``GET /api/me/key/cache``
(the holder's own figures and own top list). The holder reads only the saver view and
its own top list, so another key's reloads never move it (counters design 7.4), and its
response models have no field for ``tenants_seen``, the charged key, total reloads or
regret. A minor 1 daemon answers ``top_reused`` with ``unknown_op``: "not available".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.kv import usage as usage_mod
from app.kv.schemas import KvReuseHolder, KvTopRow
from app.kv.usage import key_cache_report, kv_usage, tenant_names
from lmw_kv.reuse import HALF_LIFE_S, empty_report
from lmw_kv.scopes import tenant_tag
from lmw_kv.store import Store
from tests.unit.kv.test_usage import (
    MODEL,
    NK,
    STARTED,
    _Client,
    _routes,
    _stats,
    _Sup,
    _tt,
    db,  # noqa: F401  (fixture)
)
from tests.unit.lmw_kv.test_store import ROOT, SCOPE, _k, opened  # noqa: F401  (fixture)
from tests.unit.lmw_kv.test_store_reuse import _reg, _store
from tests.unit.stats.test_forest_routes import _forest_login

HOLDER_REUSE = {
    "positions",
    "positions_reused",
    "reused_pct",
    "reloads",
    "nvme_bytes_written",
    "never_reused_nvme_bytes",
    "half_life_s",
}
HOLDER_TOP = {"label", "depth_tokens", "reloads"}


def _hist(**cells: tuple[int, int]) -> dict[str, dict[str, dict[str, int]]]:
    """An ``evicted_hist``: zeros, with ``tier_bucket=(positions, bytes)`` cells set."""
    h = empty_report()["saver"]["evicted_hist"]
    for name, (n, b) in cells.items():
        tier, bucket = name.split("_", 1)
        h[tier][bucket.replace("_", "-")] = {"positions": n, "bytes": b}
    return h


def _reuse_map(
    *, charged: dict[str, Any] | None = None, saver: dict[str, Any] | None = None
) -> dict[str, Any]:
    r = empty_report()
    r["charged"].update(charged or {})
    r["saver"].update(saver or {})
    return r


def _t1_reuse() -> dict[str, Any]:
    return _reuse_map(
        charged={
            "positions": {"ram": 2, "nvme": 5},
            "positions_reloaded": {"ram": 1, "nvme": 3},
            "reloads": 11,
            "ram_reloads": 4,
            "nvme_reloads": 7,
            "evicted_hist": _hist(nvme_0=(6, 6000), nvme_2_3=(1, 1000)),
        },
        saver={
            "positions": 4,
            "positions_reloaded": 1,
            "saver_reloads": 3,
            "evicted_hist": _hist(nvme_0=(7, 7000)),
            "nvme_bytes_written": 12_000,
            "nvme_bytes_read": 4000,
            "wasted_nvme_bytes": 7000,
            "regret": 2,
        },
    )


def _gone_reuse() -> dict[str, Any]:
    return _reuse_map(
        charged={
            "positions": {"ram": 0, "nvme": 1},
            "positions_reloaded": {"ram": 0, "nvme": 1},
            "reloads": 5,
            "nvme_reloads": 5,
        },
        saver={
            "positions": 3,
            "positions_reloaded": 3,
            "saver_reloads": 8,
            "nvme_bytes_written": 3000,
            "nvme_bytes_read": 1000,
        },
    )


def _reuse_stats(nk: bytes) -> dict[str, Any]:
    """``_stats`` from a minor 2 daemon: ``reuse`` store-wide, per key and per model."""
    st = _stats(nk)
    t1, gone = _tt(nk, "tok1"), _tt(nk, "gone-family")
    st["reuse"] = {
        "reuse_persist_degraded": False,
        "half_life_s": HALF_LIFE_S,
        "order": "lru",
        "nvme_bytes_written": 15_000,
        "nvme_bytes_read": 5000,
    }
    st["by_tenant"][t1]["reuse"] = _t1_reuse()
    st["by_tenant"][gone]["reuse"] = _gone_reuse()
    st["by_model"][MODEL]["reuse"] = {t1: _t1_reuse(), gone: _gone_reuse()}
    st["by_namespace"] = {
        "11" * 32: {"model": MODEL, "role": "attention", "rank": 0, "block_size": 16},
    }
    return st


T1_ADMIN = {
    "positions": 4,
    "positions_reused": 1,
    "reused_pct": 25.0,
    "reloads": 3,
    "nvme_bytes_written": 12_000,
    "never_reused_nvme_bytes": 7000,
    "half_life_s": HALF_LIFE_S,
    "charged_positions": {"ram": 2, "nvme": 5},
    "charged_positions_reused": {"ram": 1, "nvme": 3},
    "all_reloads": 11,
    "ram_reloads": 4,
    "nvme_reloads": 7,
    "nvme_bytes_read": 4000,
    "evicted_hist": _hist(nvme_0=(6, 6000), nvme_2_3=(1, 1000)),
    "regret": 2,
}


def _admin_rows(nk: bytes) -> list[dict[str, Any]]:
    """``top_reused`` admin rows as the daemon sends them."""
    t1, gone = _tt(nk, "tok1"), _tt(nk, "gone-family")
    common = {"model": MODEL, "decayed": 4.5, "first_saved": STARTED, "tiers": ["nvme"]}
    return [
        {**common, "label": "a1" * 6, "depth": 3, "reloads": 9, "last_reload": STARTED + 60,
         "tenants_seen": 4, "many": False, "charged": t1},
        {**common, "label": "b2" * 6, "depth": -1, "reloads": 5, "last_reload": STARTED + 1,
         "tenants_seen": 2, "many": False, "charged": gone},
        {**common, "label": "c3" * 6, "depth": 0, "reloads": 1, "last_reload": 0,
         "tenants_seen": 4, "many": True, "charged": t1},
    ]  # fmt: skip


class _TopClient(_Client):
    """A stats client whose ``top_reused`` gives ``rows`` (None: ``unknown_op``)."""

    def __init__(self, reply: Any, rows: list[dict[str, Any]] | None) -> None:
        super().__init__(reply)
        self.rows = rows
        self.top_calls: list[dict[str, Any]] = []

    def top_reused(self, **kw: Any) -> list[dict[str, Any]] | None:
        self.top_calls.append(kw)
        return self.rows


# ---- admin ------------------------------------------------------------------------------


async def test_admin_usage_shows_reuse_and_write_volume_per_model_and_key(db):  # noqa: F811
    names = tenant_names(await usage_mod.TokenRepo(db).list_all(), NK)
    st = _reuse_stats(NK)
    (m,) = kv_usage(st, names, top={MODEL: []})["models"]
    # per model: the keys' maps summed
    assert m["reuse"]["nvme_bytes_written"] == 15_000
    assert m["reuse"]["never_reused_nvme_bytes"] == 7000
    assert m["reuse"]["positions"] == 7 and m["reuse"]["positions_reused"] == 4
    assert m["reuse"]["reused_pct"] == pytest.approx(400 / 7)
    assert m["reuse"]["all_reloads"] == 16 and m["reuse"]["reloads"] == 11
    assert m["reuse"]["evicted_hist"] == T1_ADMIN["evicted_hist"]
    assert m["top_reused"] == []
    # per key inside the model
    t1 = next(t for t in m["tenants"] if t["token_id"] == "tok1")
    assert t1["reuse"] == T1_ADMIN
    internal = next(t for t in m["tenants"] if t["token_id"] == "internal")
    assert internal["reuse"] == {**_zero_admin(), "reused_pct": None}
    # the key's admin report: the same figures from its by_tenant row
    rep = await key_cache_report(db, "tok1", st, NK, admin=True)
    assert rep["store"]["reuse"] == T1_ADMIN


def _zero_admin() -> dict[str, Any]:
    out: dict[str, Any] = {k: 0 for k in T1_ADMIN}
    out.update(
        half_life_s=HALF_LIFE_S,
        charged_positions={"ram": 0, "nvme": 0},
        charged_positions_reused={"ram": 0, "nvme": 0},
        evicted_hist=_hist(),
    )
    return out


async def test_admin_top_rows_name_the_charged_key_and_cap_keys_seen(db):  # noqa: F811
    names = tenant_names(await usage_mod.TokenRepo(db).list_all(), NK)
    (m,) = kv_usage(_reuse_stats(NK), names, top={MODEL: _admin_rows(NK)})["models"]
    assert m["top_reused"] == [
        {"label": "a1" * 6, "depth_tokens": 48, "reloads": 9, "decayed": 4.5,
         "last_reload": "2023-11-14 22:14:20", "first_saved": "2023-11-14 22:13:20",
         "tiers": ["nvme"], "keys_seen": "4 or more", "charged_name": "test"},
        {"label": "b2" * 6, "depth_tokens": None, "reloads": 5, "decayed": 4.5,
         "last_reload": "2023-11-14 22:13:21", "first_saved": "2023-11-14 22:13:20",
         "tiers": ["nvme"], "keys_seen": "2", "charged_name": "unknown"},
        {"label": "c3" * 6, "depth_tokens": 0, "reloads": 1, "decayed": 4.5,
         "last_reload": None, "first_saved": "2023-11-14 22:13:20",
         "tiers": ["nvme"], "keys_seen": "4 or more", "charged_name": "test"},
    ]  # fmt: skip
    for row in m["top_reused"]:
        assert set(row) >= HOLDER_TOP
        assert MODEL not in row.values() and _tt(NK, "tok1") not in row.values()


def test_admin_usage_route_asks_each_model_for_its_top_ten(tmp_data_dir, client):
    _plain, admin = _routes(client, tmp_data_dir)
    nk = client.app.state.cobs2_key.key
    c = _TopClient(_reuse_stats(nk), _admin_rows(nk))
    client.app.state.kv_store = _Sup(c)
    r = client.get("/api/kv/usage", headers=admin)
    assert r.status_code == 200, r.text
    (m,) = r.json()["models"]
    assert [row["label"] for row in m["top_reused"]] == ["a1" * 6, "b2" * 6, "c3" * 6]
    assert m["top_reused"][0]["charged_name"] == "test"
    assert m["reuse"]["nvme_bytes_written"] == 15_000
    assert c.top_calls == [
        {"model": bytes.fromhex(MODEL), "tenant": None, "view": "admin", "n": 10}
    ]
    a = client.get("/api/tokens/tok1/cache", headers=admin).json()
    assert a["store"]["reuse"]["regret"] == 2
    assert a["store"]["reuse"]["evicted_hist"]["nvme"]["2-3"] == {"positions": 1, "bytes": 1000}


# ---- holder -----------------------------------------------------------------------------


def test_holder_report_has_no_admin_fields(tmp_data_dir, client, monkeypatch):
    assert set(KvReuseHolder.model_fields) == HOLDER_REUSE
    assert set(KvTopRow.model_fields) == HOLDER_TOP
    plaintext, _admin = _routes(client, tmp_data_dir)
    k = _forest_login(client, plaintext)
    leak = {"tenants_seen": 3, "charged": "ab" * 16, "all_reloads": 99, "regret": 7}
    reuse = {
        "positions": 4, "positions_reused": 1, "reused_pct": 25.0, "reloads": 3,
        "nvme_bytes_written": 1, "never_reused_nvme_bytes": 1, "half_life_s": HALF_LIFE_S,
    }  # fmt: skip

    async def leaky(*_a: Any, **_kw: Any) -> dict[str, Any]:
        return {
            "isolation": "sharing",
            "windows": {w: {"gpu": 0, "ram": None, "nvme": None, "recomputed": 0,
                            "hit_rate": None} for w in ("1h", "24h", "7d")},
            "store": {
                "ram_bytes": 0, "nvme_bytes": 0, "ram_share": 0.0, "nvme_share": 0.0,
                "ram_evictions": 0, "nvme_evictions": 0, "since": None,
                "evicted_by": [], "reuse": {**reuse, **leak},
            },
            "top_reused": [{"label": "a1" * 6, "depth_tokens": 16, "reloads": 2, **leak,
                            "charged_name": "test", "keys_seen": "2"}],
        }  # fmt: skip

    monkeypatch.setattr(usage_mod, "key_cache_report", leaky)
    body = client.get("/api/me/key/cache", headers=k).json()
    assert set(body["store"]["reuse"]) == HOLDER_REUSE
    assert set(body["top_reused"][0]) == HOLDER_TOP
    assert "evicted_by" not in body["store"]


async def test_holder_figures_ignore_other_keys_reloads(
    db,  # noqa: F811
    tmp_path: Path,
    opened: list[Any],  # noqa: F811
    monkeypatch,
):
    """Sharing key A saves a position and reloads it once; key B then reloads it twice.
    B is charged for it now and its own figures grow, A's holder report does not move."""
    monkeypatch.setattr(usage_mod, "STATS_TTL_S", 0.0)
    a, b = tenant_tag(NK, "tok1"), tenant_tag(NK, "tok2")
    s: Store = _store(opened, tmp_path / "store", nvme=False)
    ns_id = _reg(s)
    key = _k("shared-prefix")
    assert s.begin_save(ns_id, key, ROOT, SCOPE, tenant=a, depth=0) is not None
    s.commit_save(ns_id, key)

    def use(tenant: bytes, tag: str) -> None:
        h = s.begin_load(ns_id, [key], tenant=tenant, tag=tag)
        s.end_load(h.id, used=True)

    class StoreClient:
        def stats(self) -> dict[str, Any]:
            return s.stats()

        def top_reused(self, *, model: Any, tenant: Any, view: str, n: int) -> Any:
            return s.top_reused(model, tenant, view, n)

    state = SimpleNamespace(kv_store=_Sup(StoreClient()))

    async def holder() -> dict[str, Any]:
        st = await usage_mod.fetch_stats(state)
        return await key_cache_report(db, "tok1", st, NK, admin=False, app_state=state)

    use(a, "a-1")
    before = await holder()
    assert before["store"]["reuse"]["reloads"] == 1
    assert [r["reloads"] for r in before["top_reused"]] == [1]
    use(b, "b-1")
    use(b, "b-2")
    st = s.stats()
    assert st["by_tenant"][b.hex()]["reuse"]["charged"]["reloads"] == 3  # B pays now
    after = await holder()
    assert after["store"]["reuse"] == before["store"]["reuse"]
    assert after["top_reused"] == before["top_reused"]


# ---- an old daemon, no daemon ------------------------------------------------------------


def test_reuse_not_available_with_a_minor1_daemon(tmp_data_dir, client):
    """A minor 1 daemon: no ``reuse`` in stats and ``unknown_op`` for ``top_reused``. The
    reuse fields read "not available" (null) and every other figure is as before."""
    plaintext, admin = _routes(client, tmp_data_dir)
    nk = client.app.state.cobs2_key.key
    old = client.get("/api/kv/usage", headers=admin).json()
    old_key = client.get("/api/tokens/tok1/cache", headers=admin).json()
    k = _forest_login(client, plaintext)
    old_holder = client.get("/api/me/key/cache", headers=k).json()
    client.app.state.kv_store = _Sup(_TopClient(_stats(nk), None))
    client.app.state.kv_stats_cache = None
    u = client.get("/api/kv/usage", headers=admin).json()
    assert u["models"][0]["top_reused"] is None and u["models"][0]["reuse"] is None
    assert u == old
    new_key = client.get("/api/tokens/tok1/cache", headers=admin).json()
    assert {**new_key, "since_epoch": 0} == {**old_key, "since_epoch": 0}
    h = client.get("/api/me/key/cache", headers=k).json()
    assert h["top_reused"] is None and h["store"]["reuse"] is None
    assert h == old_holder
    # a minor 2 stats reply but top_reused refused: only the top lists are missing
    client.app.state.kv_store = _Sup(_TopClient(_reuse_stats(nk), None))
    client.app.state.kv_stats_cache = None
    (m,) = client.get("/api/kv/usage", headers=admin).json()["models"]
    assert m["top_reused"] is None and m["reuse"]["positions"] == 7
    h = client.get("/api/me/key/cache", headers=k).json()
    assert h["top_reused"] is None and h["store"]["reuse"]["positions"] == 4


def test_no_daemon_gives_no_reuse(tmp_data_dir, client):
    plaintext, admin = _routes(client, tmp_data_dir, None)
    assert client.get("/api/kv/usage", headers=admin).json() == {
        "available": False,
        "policy": None,
        "since": None,
        "tiers": None,
        "models": [],
        "pool_error": None,
        "nvme_error": None,
    }
    assert client.get("/api/tokens/tok1/cache", headers=admin).json()["store"] is None
    h = client.get("/api/me/key/cache", headers=_forest_login(client, plaintext)).json()
    assert h["store"] is None and h["top_reused"] is None


async def test_fetch_top_fails_open():
    class Boom:
        def top_reused(self, **_kw: Any) -> None:
            raise OSError("socket gone")

    for sup in (None, _Sup(None), _Sup(Boom()), _Sup(_Client({}))):
        state = SimpleNamespace(kv_store=sup)
        assert await usage_mod.fetch_top(state, model=None, tenant=b"\x01" * 16,
                                         view="holder", n=10) is None  # fmt: skip
    rows = [{"label": "x"}]
    state = SimpleNamespace(kv_store=_Sup(_TopClient({}, rows)))
    assert await usage_mod.fetch_top(state, model=None, tenant=None, view="admin", n=10) == rows
