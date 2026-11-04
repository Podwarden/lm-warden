"""Guard settings: platform default, per-key inherit / off / override, the packs list, the
try-it endpoint (stores nothing) and the action log (no values)."""

from __future__ import annotations

import sqlite3

from app.guard.resolve import effective
from tests.conftest import csrf_header, seed_admin_user
from tests.unit.guard.corpus import positives

PLATFORM = {"mode": "sanitize", "packs": ["cloud-api-keys", "private-keys"], "restore": "all"}


def test_effective_inherits_turns_off_and_overrides_field_by_field():
    assert effective(None, None) is None
    # Every field in force is shown, defaults included, and `source` says where each came from.
    assert effective(PLATFORM, None) == {
        **PLATFORM,
        "rules": [],
        "overrides": {},
        "trailer": "new_only",
        "fail": "closed",
        "source": {
            "mode": "platform",
            "packs": "platform",
            "rules": "default",
            "overrides": "default",
            "restore": "platform",
            "trailer": "default",
            "fail": "default",
        },
    }
    assert effective(PLATFORM, {"off": True}) is None
    got = effective(
        PLATFORM,
        {
            "override": {
                "mode": "block",
                "rules": [{"id": "x", "type": "fixed", "values": ["Halcyon"]}],
            }
        },
    )
    assert (
        got["mode"] == "block"
        and got["packs"] == PLATFORM["packs"]
        and got["rules"][0]["id"] == "x"
    )
    assert got["source"]["mode"] == "key" and got["source"]["packs"] == "platform"
    # a key can enable the guard even when the platform has none
    assert effective(None, {"override": {"mode": "sanitize"}})["mode"] == "sanitize"


def _admin(client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    return {"Authorization": f"Bearer {r.json()['access_token']}", **csrf_header(client)}


def test_platform_settings_round_trip_and_validate(tmp_data_dir, client):
    h = _admin(client, tmp_data_dir)
    assert client.get("/api/guard/settings", headers=h).json() == {"guard": None}
    r = client.put("/api/guard/settings", json={"guard": PLATFORM}, headers=h)
    assert r.status_code == 200, r.text
    assert (
        client.get("/api/guard/settings", headers=h).json()["guard"]["packs"] == PLATFORM["packs"]
    )
    bad = client.put(
        "/api/guard/settings",
        json={
            "guard": {
                "mode": "sanitize",
                "rules": [{"id": "r", "type": "regex", "regex": "(a+)+$"}],
            }
        },
        headers=h,
    )
    assert bad.status_code == 422 and "exponential" in bad.text
    assert client.get("/api/guard/settings").status_code == 401


def test_key_inherit_off_override_and_effective(tmp_data_dir, client):
    h = _admin(client, tmp_data_dir)
    client.put("/api/guard/settings", json={"guard": PLATFORM}, headers=h)
    tid = client.post("/api/tokens", json={"name": "k"}, headers=h).json()["id"]
    d = client.get(f"/api/tokens/{tid}", headers=h).json()
    assert d["guard"] is None and d["guard_effective"]["mode"] == "sanitize"
    d = client.patch(f"/api/tokens/{tid}", json={"guard": {"off": True}}, headers=h).json()
    assert d["guard"] == {"off": True} and d["guard_effective"] is None
    d = client.patch(
        f"/api/tokens/{tid}", json={"guard": {"override": {"mode": "block"}}}, headers=h
    ).json()
    assert (
        d["guard_effective"]["mode"] == "block" and d["guard_effective"]["source"]["mode"] == "key"
    )
    d = client.patch(f"/api/tokens/{tid}", json={"guard": None}, headers=h).json()
    assert d["guard"] is None and d["guard_effective"]["mode"] == "sanitize"
    assert (
        client.patch(
            f"/api/tokens/{tid}", json={"guard": {"override": {"mode": "nope"}}}, headers=h
        ).status_code
        == 422
    )


def test_packs_list_and_try_it_store_nothing(tmp_data_dir, client):
    h = _admin(client, tmp_data_dir)
    packs = client.get("/api/guard/packs", headers=h).json()["packs"]
    assert {p["id"] for p in packs} >= {
        "cloud-api-keys",
        "private-keys",
        "generic-credentials",
        "crypto-keys",
        "pii-contact",
    }
    assert all(r["label"] for p in packs for r in p["rules"])
    carrier, value = positives()["aws-access-key"][0]
    r = client.post(
        "/api/guard/test", json={"text": carrier, "guard": {"mode": "sanitize"}}, headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert value not in r.text
    assert body["findings"][0]["rule_id"] == "aws-access-key" and body["findings"][0][
        "length"
    ] == len(value)
    assert "⟦LMW-SECRET-" in body["preview"]
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        dump = "\n".join(db.iterdump())
    assert value not in dump


def test_action_log_records_no_values(tmp_data_dir, client):
    from app.guard.events import GuardEvent, record_events

    h = _admin(client, tmp_data_dir)
    db_path = tmp_data_dir / "vllm-warden.db"
    record_events(
        db_path,
        [
            GuardEvent(
                ts=1000.0,
                token_id="tok1",
                request_id="r1",
                sess_ref="s_x",
                action="sanitized",
                rule_id="aws-access-key",
                role="tool",
                msg_index=14,
                kind="tool result",
                length=20,
                ph="7c41e09b2a",
                relayed=False,
            ),
            GuardEvent(
                ts=1001.0,
                token_id="tok1",
                request_id="r1",
                sess_ref="s_x",
                action="restored",
                rule_id="aws-access-key",
                role="assistant",
                msg_index=0,
                kind="tool call",
                length=20,
                ph="7c41e09b2a",
                relayed=False,
            ),
        ],
    )
    r = client.get("/api/guard/events?token=tok1&since=0", headers=h)
    assert r.status_code == 200, r.text
    ev = r.json()
    assert [e["action"] for e in ev["events"]] == ["restored", "sanitized"]
    assert ev["totals"] == {
        "sanitized": 1,
        "blocked": 0,
        "restored": 1,
        "error": 0,
        "distinct_values": 1,
    }
    assert set(ev["events"][0]) == {
        "ts",
        "token_id",
        "request_id",
        "sess_ref",
        "action",
        "rule_id",
        "role",
        "msg_index",
        "kind",
        "length",
        "ph",
        "relayed",
    }


def test_guard_effective_shows_every_field_in_force_even_without_a_platform_setting():
    from app.guard.resolve import effective

    eff = effective(None, {"override": {"mode": "block"}})
    assert eff["mode"] == "block"
    assert eff["packs"] == ["cloud-api-keys", "private-keys", "generic-credentials", "crypto-keys"]
    assert eff["restore"] == "all" and eff["trailer"] == "new_only" and eff["fail"] == "closed"
    assert eff["source"]["mode"] == "key" and eff["source"]["packs"] == "default"


def test_the_packs_list_names_each_rule_once(tmp_data_dir, client):
    h = _admin(client, tmp_data_dir)
    packs = client.get("/api/guard/packs", headers=h).json()["packs"]
    for p in packs:
        ids = [r["id"] for r in p["rules"]]
        assert len(ids) == len(set(ids)), (p["id"], ids)


def test_a_key_can_be_created_with_its_guard_settings(tmp_data_dir, client):
    h = _admin(client, tmp_data_dir)
    own = {
        "override": {
            "mode": "block",
            "packs": ["cloud-api-keys"],
            "overrides": {"cloud-api-keys": "sanitize"},
        }
    }
    r = client.post("/api/tokens", headers=h, json={"name": "with-guard", "guard": own})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["guard"] == own and body["guard_effective"]["mode"] == "block"
    detail = client.get(f"/api/tokens/{body['id']}", headers=h).json()
    assert detail["guard"] == own and detail["guard_effective"]["packs"] == ["cloud-api-keys"]

    off = client.post(
        "/api/tokens", headers=h, json={"name": "guard-off", "guard": {"off": True}}
    ).json()
    assert off["guard"] == {"off": True} and off["guard_effective"] is None

    plain = client.post("/api/tokens", headers=h, json={"name": "inherits"}).json()
    assert plain["guard"] is None


def test_creating_a_key_with_bad_guard_settings_is_refused_and_creates_nothing(
    tmp_data_dir, client
):
    h = _admin(client, tmp_data_dir)
    before = len(client.get("/api/tokens", headers=h).json()["items"])
    r = client.post(
        "/api/tokens", headers=h, json={"name": "bad", "guard": {"override": {"packs": ["cloud"]}}}
    )
    assert r.status_code == 422 and "unknown pack" in r.text
    r2 = client.post(
        "/api/tokens",
        headers=h,
        json={
            "name": "bad2",
            "guard": {"override": {"rules": [{"id": "r", "type": "regex", "regex": "(a+)+$"}]}},
        },
    )
    assert r2.status_code == 422
    assert len(client.get("/api/tokens", headers=h).json()["items"]) == before
