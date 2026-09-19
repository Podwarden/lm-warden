"""Token details page, management half (spec 2026-09-18 §3.2-3.3):

  * GET /api/tokens/{id} -- the list item's fields plus pause state and the
    rotation lineage; 404 on an unknown id
  * PATCH /api/tokens/{id} -- rename (trimmed, 1..64) and pause/resume, the 409
    for a dead key, and the full-token response
  * GET /api/tokens -- items carry paused_at / is_paused
"""

import sqlite3

import pytest

from tests.conftest import csrf_header, jwt_login, seed_admin_user

UNKNOWN = "deadbeefdeadbeefdeadbeefdeadbeef"


def _ready(client, tmp_data_dir):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path, {**jwt_login(client), **csrf_header(client)}


def _create(client, hdrs, name="harness") -> str:
    r = client.post("/api/tokens", json={"name": name}, headers=hdrs)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _rotate(client, hdrs, tid, grace_hours=24) -> str:
    r = client.post(f"/api/tokens/{tid}/rotate", json={"grace_hours": grace_hours}, headers=hdrs)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _set(db_path, tid, assignment: str) -> None:
    # `assignment` is a trusted SQL fragment written in this file, never input.
    with sqlite3.connect(db_path) as db:
        db.execute(f"UPDATE api_tokens SET {assignment} WHERE id = ?", (tid,))
        db.commit()


def _patch(client, hdrs, tid, body):
    return client.patch(f"/api/tokens/{tid}", json=body, headers=hdrs)


# ---- GET /api/tokens/{id} ----------------------------------------------------


def test_get_returns_the_list_item_plus_pause_and_lineage(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)

    r = client.get(f"/api/tokens/{tid}", headers=hdrs)
    assert r.status_code == 200, r.text
    detail = r.json()
    items = {it["id"]: it for it in client.get("/api/tokens", headers=hdrs).json()["items"]}
    # One shape: everything but lineage is exactly what the list says.
    assert {k: v for k, v in detail.items() if k != "lineage"} == items[tid]
    assert detail["paused_at"] is None
    assert detail["is_paused"] is False
    assert detail["lineage"] == [{
        "id": tid,
        "name": "harness",
        "created_at": detail["created_at"],
        "rotated_at": None,
        "is_revoked": False,
        "in_grace": False,
        "is_self": True,
    }]


def test_get_unknown_token_is_404(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    assert client.get(f"/api/tokens/{UNKNOWN}", headers=hdrs).status_code == 404


def test_get_requires_a_session(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert client.get(f"/api/tokens/{tid}").status_code == 401


def test_lineage_across_a_two_rotation_chain(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    a = _create(client, hdrs, "harness")
    b = _rotate(client, hdrs, a, grace_hours=24)  # a: grace window open
    c = _rotate(client, hdrs, b, grace_hours=0)   # b: cut off now

    body = client.get(f"/api/tokens/{b}", headers=hdrs).json()
    lineage = body["lineage"]
    assert [e["id"] for e in lineage] == [a, b, c]  # oldest first
    assert [e["name"] for e in lineage] == ["harness (old 1)", "harness (old 2)", "harness"]
    assert [e["is_self"] for e in lineage] == [False, True, False]
    assert (lineage[0]["in_grace"], lineage[0]["is_revoked"]) == (True, False)
    assert (lineage[1]["in_grace"], lineage[1]["is_revoked"]) == (False, True)
    assert (lineage[2]["in_grace"], lineage[2]["is_revoked"]) == (False, False)
    assert lineage[0]["rotated_at"] is not None and lineage[2]["rotated_at"] is None
    assert body["successor_id"] == c

    newest = client.get(f"/api/tokens/{c}", headers=hdrs).json()
    assert [e["id"] for e in newest["lineage"]] == [a, b, c]
    assert [e["is_self"] for e in newest["lineage"]] == [False, False, True]
    assert newest["successor_id"] is None


@pytest.mark.parametrize(
    "assignment",
    [
        "paused_at = '2000-01-01 00:00:00'",  # paused: require_bearer answers 403
        "expires_at = '2000-01-01 00:00:00'",  # expired inside its grace: 401
    ],
)
def test_a_paused_or_expired_predecessor_is_not_in_grace(tmp_data_dir, client, assignment):
    # #251: in_grace used to read only revoked_at > now, so a predecessor
    # that is refused anyway still said "in grace" on the lineage card.
    db_path, hdrs = _ready(client, tmp_data_dir)
    a = _create(client, hdrs)
    b = _rotate(client, hdrs, a, grace_hours=24)
    before = client.get(f"/api/tokens/{b}", headers=hdrs).json()["lineage"][0]
    assert (before["in_grace"], before["is_revoked"]) == (True, False)

    _set(db_path, a, assignment)
    after = client.get(f"/api/tokens/{b}", headers=hdrs).json()["lineage"][0]
    assert after["id"] == a
    # Not in grace -- and not "revoked" either: its revoked_at is still ahead.
    assert (after["in_grace"], after["is_revoked"]) == (False, False)


# ---- PATCH: rename -----------------------------------------------------------


def test_rename_is_trimmed(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    r = _patch(client, hdrs, tid, {"name": "  renamed  "})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "renamed"


def test_rename_bounds_are_422(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert _patch(client, hdrs, tid, {"name": "   "}).status_code == 422
    assert _patch(client, hdrs, tid, {"name": "x" * 65}).status_code == 422
    assert _patch(client, hdrs, tid, {"name": None}).status_code == 422
    # 64 characters after trimming is fine.
    assert _patch(client, hdrs, tid, {"name": " " + "x" * 64 + " "}).status_code == 200


def test_patch_returns_the_full_token(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    body = _patch(client, hdrs, tid, {"priority": 8}).json()
    assert body == client.get(f"/api/tokens/{tid}", headers=hdrs).json()
    assert body["priority"] == 8


# ---- PATCH: pause / resume ---------------------------------------------------


def test_pause_then_resume(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    body = _patch(client, hdrs, tid, {"paused": True}).json()
    assert body["is_paused"] is True
    assert body["paused_at"] is not None
    body = _patch(client, hdrs, tid, {"paused": False}).json()
    assert body["is_paused"] is False
    assert body["paused_at"] is None


def test_pausing_again_keeps_the_first_timestamp(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert _patch(client, hdrs, tid, {"paused": True}).status_code == 200
    _set(db_path, tid, "paused_at = '2026-01-01 00:00:00'")
    body = _patch(client, hdrs, tid, {"paused": True}).json()
    assert body["paused_at"] == "2026-01-01 00:00:00"


def test_paused_null_is_422(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    assert _patch(client, hdrs, tid, {"paused": None}).status_code == 422


def test_pausing_an_expired_key_is_409_and_applies_nothing(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _set(db_path, tid, "expires_at = datetime('now', '-1 day')")
    r = _patch(client, hdrs, tid, {"paused": True, "name": "should-not-stick"})
    assert r.status_code == 409
    assert "expired" in r.json()["detail"]
    body = client.get(f"/api/tokens/{tid}", headers=hdrs).json()
    assert (body["is_paused"], body["name"]) == (False, "harness")


def test_pausing_a_revoked_key_is_409(tmp_data_dir, client):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _set(db_path, tid, "revoked_at = datetime('now', '-1 second')")
    r = _patch(client, hdrs, tid, {"paused": True})
    assert r.status_code == 409
    assert "revoked" in r.json()["detail"]


def test_pausing_a_key_in_its_grace_window_is_allowed(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    old = _create(client, hdrs)
    _rotate(client, hdrs, old, grace_hours=24)
    r = _patch(client, hdrs, old, {"paused": True})
    assert r.status_code == 200, r.text
    assert r.json()["is_paused"] is True


def test_resuming_a_dead_key_is_allowed(tmp_data_dir, client):
    # Only PAUSING a dead key is refused; clearing a stale pause is harmless.
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)
    _set(db_path, tid, "paused_at = datetime('now'), expires_at = datetime('now', '-1 day')")
    r = _patch(client, hdrs, tid, {"paused": False})
    assert r.status_code == 200, r.text
    assert r.json()["is_paused"] is False


def test_rotating_a_paused_key_gives_an_unpaused_successor(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    old = _create(client, hdrs)
    _patch(client, hdrs, old, {"paused": True})
    new = _rotate(client, hdrs, old)
    assert client.get(f"/api/tokens/{new}", headers=hdrs).json()["is_paused"] is False
    assert client.get(f"/api/tokens/{old}", headers=hdrs).json()["is_paused"] is True


# ---- GET /api/tokens ---------------------------------------------------------


def test_list_items_carry_the_pause_fields(tmp_data_dir, client):
    _db, hdrs = _ready(client, tmp_data_dir)
    paused = _create(client, hdrs, "paused-one")
    live = _create(client, hdrs, "live-one")
    _patch(client, hdrs, paused, {"paused": True})
    items = {it["id"]: it for it in client.get("/api/tokens", headers=hdrs).json()["items"]}
    assert items[paused]["is_paused"] is True
    assert items[paused]["paused_at"] is not None
    assert items[live]["is_paused"] is False
    assert items[live]["paused_at"] is None
