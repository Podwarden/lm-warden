"""Prompt Lens price cards and money savings (analysis Task 2.2)."""

import sqlite3

import pytest

from app.chat2.ledger import RateCard
from app.lens.catalogue import Saving
from app.lens.prices import (
    card_for,
    load_defaults,
    price_saving,
    session_input_cost,
)

CARD = RateCard(
    "c1", None, "prov", "m", 100_000, 400_000, None, 10_000, 125_000, 0, "USD", "2026-11-21"
)


def test_worked_example_of_spec_5_2():
    prompt, reusable = 165_668, 145_755
    actual = session_input_cost(prompt, CARD)
    assert actual == pytest.approx(0.0166, abs=1e-4)
    s = price_saving(Saving(reusable, None, None, None), CARD, cached_share=reusable / prompt)
    assert s.money is not None
    assert (actual - s.money) == pytest.approx(0.0040, abs=2e-4)  # cost with breakpoints
    assert s.money / actual == pytest.approx(0.76, abs=0.01)


def test_no_write_price_no_premium():
    card = RateCard("c", None, "p", "m", 260_000, 1, None, 130_000, None, 0, "USD", "2026-11-21")
    s = price_saving(Saving(1_000_000, None, None, None), card, cached_share=0.9)
    assert s.money == pytest.approx(0.13)


def test_output_side_uses_the_output_price():
    s = price_saving(Saving(10_000, None, None, None), CARD, cached_share=1.0, kind="output")
    assert s.money == pytest.approx(0.004)


def test_no_card_means_no_money():
    s = Saving(5_000, 1.0, None, None)
    assert price_saving(s, None, cached_share=1.0) == s


def test_card_without_read_price_gives_no_money():
    card = RateCard("c", None, "p", "m", 100_000, 1, None, None, None, 0, "USD", "2026-11-21")
    assert price_saving(Saving(1_000, None, None, None), card, cached_share=1.0).money is None


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute(
        "CREATE TABLE rate_cards (id TEXT PRIMARY KEY, org_id TEXT, provider TEXT, model TEXT,"
        " input_per_m_micros INTEGER, output_per_m_micros INTEGER, reasoning_per_m_micros INTEGER,"
        " cache_read_per_m_micros INTEGER, cache_write_per_m_micros INTEGER,"
        " markup_pct INTEGER, currency TEXT, effective_from TEXT)"
    )
    db.execute(
        "INSERT INTO rate_cards VALUES ('a', NULL, 'upstream.example', 'big', 100000, 400000,"
        " NULL, 10000, NULL, 0, 'USD', '2026-01-01')"
    )
    db.execute(
        "INSERT INTO rate_cards VALUES ('b', NULL, 'upstream.example', 'big', 200000, 400000,"
        " NULL, 20000, NULL, 0, 'USD', '2026-06-01')"
    )
    return db


def test_card_for_relay_is_the_provider_card_newest_effective():
    db = _db()
    c = card_for(
        db, relayed=True, upstream="upstream.example", model="big", model_id="x", at="2026-03-01"
    )
    assert c is not None and c.id == "a"
    c = card_for(
        db, relayed=True, upstream="upstream.example", model="big", model_id="x", at="2026-07-01"
    )
    assert c is not None and c.id == "b"
    assert card_for(db, relayed=True, upstream=None, model="big", model_id="x") is None


def test_card_for_own_engine_needs_a_reference_card():
    db = _db()
    assert card_for(db, relayed=False, upstream=None, model="local", model_id="m1") is None
    db.execute(
        "INSERT INTO settings VALUES ('lens_reference_card:m1',"
        ' \'{"provider": "upstream.example", "model": "big"}\')'
    )
    c = card_for(db, relayed=False, upstream=None, model="local", model_id="m1", at="2026-07-01")
    assert c is not None and c.id == "b"


def test_missing_tables_read_as_no_card():
    db = sqlite3.connect(":memory:")
    assert card_for(db, relayed=True, upstream="u", model="m", model_id="x") is None
    assert card_for(db, relayed=False, upstream=None, model="m", model_id="x") is None


def test_shipped_defaults_carry_source_and_date():
    cards = load_defaults()
    assert cards and all(c["source"] and c["effective_from"] and c["currency"] for c in cards)


# -- admin routes ---------------------------------------------------------------------


def _admin(client, tmp_data_dir):
    from tests.conftest import csrf_header, jwt_login
    from tests.unit.proxy.test_prompt_lens_forward import _ready_gm

    _ready_gm(client, tmp_data_dir)
    return {**jwt_login(client), **csrf_header(client)}


def test_rate_card_routes(client, tmp_data_dir):
    h = _admin(client, tmp_data_dir)
    got = client.get("/api/admin/lens/rate-cards", headers=h).json()
    assert got["cards"] == [] and got["defaults"] and got["reference"] == {}
    card = {
        "provider": "upstream.example",
        "model": "big",
        "input_per_m_micros": 100_000,
        "output_per_m_micros": 400_000,
        "cache_read_per_m_micros": 10_000,
        "effective_from": "2026-11-21",
    }
    assert (
        client.put("/api/admin/lens/rate-cards", json={"cards": [card]}, headers=h).status_code
        == 200
    )
    card["input_per_m_micros"] = 120_000  # same key: replaced, not duplicated
    assert (
        client.put("/api/admin/lens/rate-cards", json={"cards": [card]}, headers=h).status_code
        == 200
    )
    cards = client.get("/api/admin/lens/rate-cards", headers=h).json()["cards"]
    assert len(cards) == 1 and cards[0]["input_per_m_micros"] == 120_000
    bad = {**card, "effective_from": "yesterday"}
    assert (
        client.put("/api/admin/lens/rate-cards", json={"cards": [bad]}, headers=h).status_code
        == 422
    )


def test_reference_card_route(client, tmp_data_dir):
    import sqlite3

    h = _admin(client, tmp_data_dir)
    url = "/api/admin/lens/reference-card/m1"
    body = {"provider": "upstream.example", "model": "big"}
    assert client.put(url, json=body, headers=h).status_code == 404  # no such model
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes) VALUES ('m1','m1','o/r','main','[0]',1,0.9,0,'[]','registered',0)"
        )
    assert client.put(url, json={"provider": "x"}, headers=h).status_code == 422
    assert client.put(url, json=body, headers=h).status_code == 200
    ref = client.get("/api/admin/lens/rate-cards", headers=h).json()["reference"]
    assert ref == {"m1": body}
    assert client.put(url, json={}, headers=h).status_code == 200
    assert client.get("/api/admin/lens/rate-cards", headers=h).json()["reference"] == {}
    assert client.put(url, json=body).status_code in (401, 403)


def test_waste_is_billed_at_the_read_price_for_the_cached_share():
    s = Saving(1_000_000, None, None, None)
    assert price_saving(s, CARD, cached_share=1.0, kind="waste").money == pytest.approx(0.01)
    assert price_saving(s, CARD, cached_share=0.0, kind="waste").money == pytest.approx(0.10)
    assert price_saving(s, CARD, cached_share=0.5, kind="waste").money == pytest.approx(0.055)
