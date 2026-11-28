"""The price catalogue, its refresh, overrides and the Prices routes (#347)."""

from __future__ import annotations

import sqlite3

import httpx
import pytest

from app.db.database import open_db
from app.lens import price_sources as src
from app.lens import price_store as store
from app.lens import prices
from app.lens.catalogue import Saving
from tests.conftest import csrf_header, jwt_login
from tests.unit.proxy.test_prompt_lens_forward import _ready_gm

HOST = "api.anthropic.com"
MODEL = "claude-sonnet-4-5-20250929"


def _admin(client, tmp_data_dir):
    _ready_gm(client, tmp_data_dir)
    return {**jwt_login(client), **csrf_header(client)}, tmp_data_dir / "vllm-warden.db"


def _openrouter(price_in: str, n: int = 25):
    models = [
        {"id": f"openai/filler-{i}", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
        for i in range(n)
    ]
    models.append(
        {
            "id": "anthropic/claude-sonnet-4.5",
            "pricing": {
                "prompt": price_in,
                "completion": "0.000015",
                "input_cache_read": "0.0000003",
                "input_cache_write": "0.00000375",
            },
        }
    )
    return {"data": models}


def _sync_card(db_path, provider=HOST, model=MODEL):
    with sqlite3.connect(db_path) as db:
        return prices.find_card(db, provider, model)


async def test_offline_snapshot_is_used_with_no_network(client, tmp_data_dir, monkeypatch):
    _h, db_path = _admin(client, tmp_data_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert store.network_allowed() is False
    # Before the catalogue is seeded Prompt Lens already prices from the snapshot in memory.
    prices._snapshot_index.cache_clear()
    early = _sync_card(db_path)
    assert early is not None and early.input_per_m_micros > 0
    # The app seeds the catalogue from the bundled snapshot, no network involved;
    # seeding again changes nothing.
    async with open_db(db_path) as db:
        assert await store.seed_snapshot(db) > 100
        assert await store.seed_snapshot(db) == 0
        cur = await db.execute("SELECT COUNT(*), MIN(source), MAX(source) FROM price_catalogue")
        n, lo, hi = await cur.fetchone()
    assert n > 100 and lo == hi == "snapshot"
    card = _sync_card(db_path)
    assert card is not None and card.input_per_m_micros == early.input_per_m_micros
    assert card is not None and card.input_per_m_micros > 0 and card.output_per_m_micros > 0
    meta, _ = store.load_snapshot()
    assert meta["source"] and meta["fetched_at"]  # the snapshot says where and when it is from
    # A refresh with no network fails softly and changes no price.
    st = await store.refresh_once(db_path)
    assert st["last_error"] and not st.get("last_success")
    assert _sync_card(db_path) == card


async def test_refresh_replaces_prices_and_a_failure_keeps_the_last_good_table(
    client, tmp_data_dir, monkeypatch
):
    _h, db_path = _admin(client, tmp_data_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("VW_PRICE_REFRESH", raising=False)
    async with open_db(db_path) as db:
        await store.seed_snapshot(db)

    ok = httpx.MockTransport(lambda r: httpx.Response(200, json=_openrouter("0.000004")))
    st = await store.refresh_once(db_path, transport=ok)
    assert st["last_error"] is None and st["source"] == "openrouter" and st["last_success"]
    fetched = _sync_card(db_path)
    assert fetched.input_per_m_micros == 4_000_000

    down = httpx.MockTransport(lambda r: httpx.Response(500))
    st = await store.refresh_once(db_path, transport=down)
    assert "RuntimeError" in st["last_error"] and st["last_success"]  # the earlier success stays
    assert _sync_card(db_path) == fetched  # the table is as it was

    # Garbage that parses but is nearly empty is a failure too, not a price cut.
    tiny = httpx.MockTransport(lambda r: httpx.Response(200, json=_openrouter("0.000001", n=2)))
    await store.refresh_once(db_path, transport=tiny)
    assert _sync_card(db_path) == fetched

    # The network never overwrites a row with the snapshot: seeding again keeps the fetched one.
    async with open_db(db_path) as db:
        await store.seed_snapshot(db)
    assert _sync_card(db_path) == fetched


def test_due_follows_the_interval_and_retries_a_failure_after_an_hour():
    now = 1_000_000.0
    assert store.due({}, 24, now) is True
    assert store.due({}, 0, now) is False  # interval 0: automatic refresh off
    assert store.due({"last_success": now - 3600, "last_attempt": now - 3600}, 24, now) is False
    day = 24 * 3600
    assert store.due({"last_success": now - day - 1, "last_attempt": now - day - 1}, 24, now)
    assert not store.due({"last_success": now - day - 1, "last_attempt": now - 60}, 24, now)


async def test_override_wins_and_is_audited_and_clearing_restores_the_list_price(
    client, tmp_data_dir
):
    h, db_path = _admin(client, tmp_data_dir)
    async with open_db(db_path) as db:
        await store.seed_snapshot(db)
    list_price = _sync_card(db_path).input_per_m_micros

    body = {
        "provider": HOST,
        "model": "claude-sonnet-4-5",
        "input_per_m_micros": 123_000,
        "output_per_m_micros": 456_000,
    }
    r = client.put("/api/admin/prices/override", json=body, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "override" and r.json()["price"]["input_per_m_micros"] == 123_000
    assert r.json()["list_price"]["input_per_m_micros"] == list_price
    # Prompt Lens' lookup, by another spelling of the same model.
    assert _sync_card(db_path).input_per_m_micros == 123_000

    audit = client.get("/api/admin/prices/audit", headers=h).json()["entries"]
    assert audit[0]["action"] == "override_set" and audit[0]["actor"] == "admin"
    assert (
        audit[0]["model"] == "claude-sonnet-4-5"
        and audit[0]["new"]["input_per_m_micros"] == 123_000
    )

    assert (
        client.delete(
            f"/api/admin/prices/override?provider={HOST}&model=claude-sonnet-4-5", headers=h
        ).status_code
        == 200
    )
    assert _sync_card(db_path).input_per_m_micros == list_price
    audit = client.get("/api/admin/prices/audit", headers=h).json()["entries"]
    assert [a["action"] for a in audit[:2]] == ["override_clear", "override_set"]
    assert audit[0]["old"]["input_per_m_micros"] == 123_000
    assert (
        client.delete(
            f"/api/admin/prices/override?provider={HOST}&model=nope", headers=h
        ).status_code
        == 404
    )
    assert client.put("/api/admin/prices/override", json=body).status_code in (401, 403)


async def test_prompt_lens_savings_use_the_fetched_price(client, tmp_data_dir, monkeypatch):
    _h, db_path = _admin(client, tmp_data_dir)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    ok = httpx.MockTransport(lambda r: httpx.Response(200, json=_openrouter("0.000004")))
    await store.refresh_once(db_path, transport=ok)
    with sqlite3.connect(db_path) as db:
        card = prices.card_for(db, relayed=True, upstream=HOST, model=MODEL, model_id="x")
        assert card is not None and card.effective_from == src.now_iso()[:10]
        s = prices.price_saving(Saving(1_000_000, None, None, None), card, cached_share=1.0)
        # 1M tokens move from 4.00 to the 0.30 cache-read price per million.
        assert s.money == pytest.approx(3.70)
        # No price known for a provider: no money, tokens only.
        assert (
            prices.card_for(db, relayed=True, upstream="other.example", model="m", model_id="x")
            is None
        )


def test_the_old_rate_card_routes_still_work_and_are_audited(client, tmp_data_dir):
    h, _db = _admin(client, tmp_data_dir)
    got = client.get("/api/admin/lens/rate-cards", headers=h).json()
    assert got["cards"] == [] and got["defaults"] == [] and got["reference"] == {}
    card = {
        "provider": "upstream.example",
        "model": "big",
        "input_per_m_micros": 100_000,
        "output_per_m_micros": 400_000,
        "effective_from": "2026-11-21",
    }
    assert (
        client.put("/api/admin/lens/rate-cards", json={"cards": [card]}, headers=h).status_code
        == 200
    )
    assert client.get("/api/admin/lens/rate-cards", headers=h).json()["cards"][0]["model"] == "big"
    entries = client.get("/api/admin/prices/audit", headers=h).json()["entries"]
    assert entries[0]["action"] == "override_set" and entries[0]["provider"] == "upstream.example"


async def test_prices_lists_the_providers_the_install_uses_and_the_local_models(
    client, tmp_data_dir
):
    h, db_path = _admin(client, tmp_data_dir)
    async with open_db(db_path) as db:
        await store.seed_snapshot(db)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES ('m1','local-llm','o/r','main','[0]',1,0.9,0,'[]','registered',0)"
        )
        db.execute(
            "INSERT INTO settings(key, value) VALUES ('router_upstream_url', 'https://api.openai.com')"
        )
    got = client.get("/api/admin/prices", headers=h).json()
    hosts = {p["host"]: p for p in got["providers"]}
    assert "api.openai.com" in hosts and hosts["api.openai.com"]["label"] == "OpenAI"
    assert "router upstream" in hosts["api.openai.com"]["reasons"]
    assert got["local"][0]["name"] == "local-llm" and got["local"][0]["reference"] is None
    assert got["refresh"]["interval_hours"] == 24 and got["refresh"]["snapshot"]["rows"] > 100

    url = "/api/admin/lens/reference-card/m1"
    assert (
        client.put(
            url, json={"provider": HOST, "model": "claude-sonnet-4.5"}, headers=h
        ).status_code
        == 200
    )
    local = client.get("/api/admin/prices", headers=h).json()["local"][0]
    assert local["reference"] == {"provider": HOST, "model": "claude-sonnet-4.5"}
    assert local["priced_as"]["source"] in ("snapshot", "openrouter", "litellm")
    assert (
        client.get("/api/admin/prices/audit", headers=h).json()["entries"][0]["action"]
        == "reference"
    )

    found = client.get(f"/api/admin/prices/models?provider={HOST}&q=sonnet", headers=h).json()
    assert found["models"] and all("sonnet" in m["model"].lower() for m in found["models"])
    assert all(m["price"]["input_per_m_micros"] > 0 for m in found["models"])


def test_refresh_interval_setting_and_refresh_now_offline(client, tmp_data_dir):
    h, _db = _admin(client, tmp_data_dir)
    r = client.put("/api/admin/prices/settings", json={"refresh_interval_hours": 0}, headers=h)
    assert r.status_code == 200 and r.json()["interval_hours"] == 0
    assert (
        client.put(
            "/api/admin/prices/settings", json={"refresh_interval_hours": -1}, headers=h
        ).status_code
        == 422
    )
    assert (
        client.get("/api/admin/prices/audit", headers=h).json()["entries"][0]["action"]
        == "interval"
    )
    r = client.post("/api/admin/prices/refresh", headers=h)  # tests are offline (HF_HUB_OFFLINE=1)
    assert r.status_code == 200 and r.json()["last_error"]
