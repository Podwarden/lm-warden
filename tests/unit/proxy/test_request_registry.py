"""Plane B — live per-request registry + GET /api/stats/requests snapshot.

Covers: register/deregister/count/snapshot, the endpoint's by-token (grouped
by token_id, not name) and by-IP aggregation, context_pct math, the orphan
flag surfacing, and the invariant that neither the registry nor the endpoint
ever emits a token secret (token_id is fine — it's an opaque row id, not a
hash/plaintext column, and the frontend links it to /tokens/{id}).
"""

import dataclasses
import time

from app.proxy.request_registry import LiveRequest, RequestRegistry
from app.stats.live_requests import _aggregate, _serialize
from tests.conftest import jwt_login, seed_admin_user


def _mk(**over) -> LiveRequest:
    base = dict(
        id="req1",
        token_id="tok1",
        token_name="hermes-bot",
        client_ip="10.42.5.185",
        model="qwen",
        model_row_id="qwen",
        path="/v1/chat/completions",
        prompt_tokens=100,
        max_model_len=1000,
        started_monotonic=time.monotonic(),
        started_iso="2026-07-19T20:01:02.000Z",
    )
    base.update(over)
    return LiveRequest(**base)


# --------------------------------------------------------------------------- #
# Registry core
# --------------------------------------------------------------------------- #

async def test_register_deregister_count():
    reg = RequestRegistry()
    assert reg.count() == 0
    await reg.register(_mk(id="a"))
    await reg.register(_mk(id="b"))
    assert reg.count() == 2
    assert {r.id for r in reg.snapshot()} == {"a", "b"}
    await reg.deregister("a")
    assert reg.count() == 1
    # Deregistering an unknown id is a no-op, not an error.
    await reg.deregister("ghost")
    assert reg.count() == 1


async def test_snapshot_is_a_copy():
    reg = RequestRegistry()
    await reg.register(_mk(id="a"))
    snap = reg.snapshot()
    await reg.deregister("a")
    # The list captured before deregister is unaffected by later mutation.
    assert len(snap) == 1
    assert reg.count() == 0


async def test_live_field_updates_are_in_place():
    reg = RequestRegistry()
    req = _mk(id="a")
    await reg.register(req)
    # Streaming loop mutates the object directly (no lock, no registry call).
    req.phase = "decode"
    req.completion_tokens = 42
    req.orphan = True
    got = reg.get("a")
    assert got.phase == "decode"
    assert got.completion_tokens == 42
    assert got.orphan is True


# --------------------------------------------------------------------------- #
# Serialization + aggregation (endpoint helpers)
# --------------------------------------------------------------------------- #

def test_serialize_context_pct_math():
    now = time.monotonic()
    req = _mk(prompt_tokens=200, completion_tokens=98, max_model_len=1000,
              started_monotonic=now - 5.0)
    row = _serialize(req, now)
    assert row["context_tokens"] == 298
    assert row["context_pct"] == 0.298
    assert row["elapsed_s"] == 5.0
    assert row["orphan"] is False


def test_serialize_context_pct_none_when_no_max_len():
    now = time.monotonic()
    row = _serialize(_mk(max_model_len=None, completion_tokens=10), now)
    assert row["context_tokens"] == 110
    assert row["context_pct"] is None


def test_serialize_surfaces_orphan_flag():
    now = time.monotonic()
    row = _serialize(_mk(orphan=True, phase="decode"), now)
    assert row["orphan"] is True
    assert row["phase"] == "decode"


def test_aggregate_by_token_and_ip():
    now = time.monotonic()
    rows = [
        _serialize(_mk(id="1", token_id="tok1", token_name="hermes-bot", client_ip="10.0.0.1",
                       prompt_tokens=100, completion_tokens=10), now),
        _serialize(_mk(id="2", token_id="tok1", token_name="hermes-bot", client_ip="10.0.0.2",
                       prompt_tokens=200, completion_tokens=20), now),
        _serialize(_mk(id="3", token_id="tok2", token_name="other", client_ip="10.0.0.1",
                       prompt_tokens=50, completion_tokens=5), now),
    ]
    by_token, by_ip = _aggregate(rows)

    hermes = next(t for t in by_token if t["token_id"] == "tok1")
    assert hermes["token_name"] == "hermes-bot"
    assert hermes["requests"] == 2
    assert hermes["prompt_tokens"] == 300
    assert hermes["completion_tokens"] == 30
    assert hermes["context_tokens"] == 330

    ip1 = next(p for p in by_ip if p["client_ip"] == "10.0.0.1")
    assert ip1["requests"] == 2
    assert ip1["context_tokens"] == (100 + 10) + (50 + 5)


def test_aggregate_by_token_groups_same_name_different_id_separately():
    # Names are reused across rotations (migration 0033) -- two distinct
    # tokens sharing a display name must not collapse into one by-token row.
    now = time.monotonic()
    rows = [
        _serialize(_mk(id="1", token_id="tok-old", token_name="shared-name",
                       client_ip="10.0.0.1", prompt_tokens=100, completion_tokens=10), now),
        _serialize(_mk(id="2", token_id="tok-new", token_name="shared-name",
                       client_ip="10.0.0.2", prompt_tokens=200, completion_tokens=20), now),
    ]
    by_token, _ = _aggregate(rows)
    assert len(by_token) == 2
    ids = {t["token_id"] for t in by_token}
    assert ids == {"tok-old", "tok-new"}
    for t in by_token:
        assert t["token_name"] == "shared-name"
        assert t["requests"] == 1


def test_aggregate_by_token_anonymous_bucket_has_no_id():
    now = time.monotonic()
    rows = [
        _serialize(_mk(id="1", token_id=None, token_name=None, client_ip="10.0.0.1"), now),
        _serialize(_mk(id="2", token_id=None, token_name=None, client_ip="10.0.0.2"), now),
    ]
    by_token, _ = _aggregate(rows)
    assert len(by_token) == 1
    assert by_token[0]["token_id"] is None
    assert by_token[0]["requests"] == 2


# --------------------------------------------------------------------------- #
# Secret-leak guard
# --------------------------------------------------------------------------- #

def test_live_request_has_no_secret_fields():
    # The dataclass itself carries only metadata (token_id/name), never a hash
    # or plaintext column from TokenRow.
    fields = {f.name for f in dataclasses.fields(LiveRequest)}
    assert "hash" not in fields
    assert "prefix" not in fields
    assert "scope" not in fields


def test_serialized_row_leaks_no_token_secret():
    now = time.monotonic()
    row = _serialize(_mk(), now)
    # token_id is an opaque row id (used by the frontend to link to the
    # token's details page) -- fine to expose. Actual secret columns aren't.
    assert row["token_id"] == "tok1"
    assert set(row) & {"hash", "prefix", "scope", "plaintext", "secret"} == set()
    assert row["token_name"] == "hermes-bot"


# --------------------------------------------------------------------------- #
# Endpoint end-to-end
# --------------------------------------------------------------------------- #

async def test_requests_endpoint_snapshot(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = jwt_login(client)

    reg = client.app.state.request_registry
    await reg.register(_mk(id="a", token_id="tok1", token_name="hermes-bot", client_ip="10.0.0.1",
                           prompt_tokens=300, completion_tokens=20,
                           max_model_len=1000, phase="decode"))
    await reg.register(_mk(id="b", token_id="tok1", token_name="hermes-bot", client_ip="10.0.0.1",
                           prompt_tokens=100, completion_tokens=0, orphan=True))

    r = client.get("/api/stats/requests", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["requests"]) == 2

    a = next(x for x in body["requests"] if x["id"] == "a")
    assert a["context_tokens"] == 320
    assert a["context_pct"] == 0.32
    assert a["phase"] == "decode"
    assert a["token_id"] == "tok1"

    b = next(x for x in body["requests"] if x["id"] == "b")
    assert b["orphan"] is True

    assert body["by_token"][0]["token_id"] == "tok1"
    assert body["by_token"][0]["token_name"] == "hermes-bot"
    assert body["by_token"][0]["requests"] == 2
    assert body["by_ip"][0]["client_ip"] == "10.0.0.1"
    assert body["by_ip"][0]["requests"] == 2

    # No secret column anywhere in the payload (token_id itself is fine).
    for row in body["requests"]:
        assert set(row) & {"hash", "prefix", "scope", "plaintext", "secret"} == set()


async def test_requests_endpoint_requires_jwt(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/stats/requests")
    assert r.status_code in (401, 403)
