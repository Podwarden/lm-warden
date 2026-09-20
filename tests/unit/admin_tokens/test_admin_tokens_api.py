"""Settings -> Admin tokens API (spec 2026-09-19, "API"): issue, list, refresh
and revoke. Every route is session-only; the auth matrix
(tests/unit/auth/test_auth_matrix.py) checks the admin-token refusal on each."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.db.repos.tokens import sqlite_utc_in
from tests.conftest import csrf_header, jwt_login, seed_admin_user

URL = "/api/admin-tokens"


class _FakeTask:
    """Stands in for an asyncio.Task in the stream registry (cf. test_logout.py)."""

    def __init__(self) -> None:
        self.cancelled_called = False

    def cancel(self) -> bool:
        self.cancelled_called = True
        return True


def _ready(client, tmp_data_dir: Path) -> tuple[Path, dict[str, str]]:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path, {**jwt_login(client), **csrf_header(client)}


def _issue(client, hdrs, name="ci", expires_in_days=90) -> dict:
    r = client.post(URL, json={"name": name, "expires_in_days": expires_in_days}, headers=hdrs)
    assert r.status_code == 201, r.text
    return r.json()


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def _days_from_now(ts: str) -> float:
    at = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return (at - datetime.now(UTC)).total_seconds() / 86400


def _set(db_path: Path, tid: str, **cols: str | None) -> None:
    sets = ", ".join(f"{k} = ?" for k in cols)
    with sqlite3.connect(db_path) as db:
        db.execute(f"UPDATE api_tokens SET {sets} WHERE id = ?", (*cols.values(), tid))
        db.commit()


def test_issue_returns_the_secret_once_and_the_list_never_does(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    body = _issue(client, hdrs, name="ci")
    assert body["plaintext"].startswith("vwa_") and len(body["plaintext"]) == 60
    assert body["prefix"] == body["plaintext"][:8]
    assert (body["name"], body["created_by"], body["status"]) == ("ci", "admin", "active")
    assert 89.9 < _days_from_now(body["expires_at"]) <= 90.0
    listed = client.get(URL, headers=hdrs).json()["items"]
    assert [t["id"] for t in listed] == [body["id"]]
    assert "plaintext" not in listed[0]


def test_issue_needs_an_explicit_expiry(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    assert client.post(URL, json={"name": "ci"}, headers=hdrs).status_code == 422
    assert client.post(URL, json={"name": "ci", "expires_in_days": 45}, headers=hdrs).status_code == 422
    never = _issue(client, hdrs, expires_in_days=None)
    assert (never["expires_at"], never["status"]) == (None, "active")


def test_issue_trims_and_bounds_the_name(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    assert _issue(client, hdrs, name="  deploy  ")["name"] == "deploy"
    for bad in ("", "   ", "x" * 65):
        r = client.post(URL, json={"name": bad, "expires_in_days": 30}, headers=hdrs)
        assert r.status_code == 422, bad


def test_the_issued_token_calls_the_control_api(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    secret = _issue(client, hdrs)["plaintext"]
    assert client.get("/api/tokens", headers=_bearer(secret)).status_code == 200


def test_the_list_puts_live_tokens_first_and_hides_long_dead_ones(client, tmp_data_dir):
    db_path, hdrs = _ready(client, tmp_data_dir)
    old = _issue(client, hdrs, name="old")
    recent = _issue(client, hdrs, name="recent")
    live = _issue(client, hdrs, name="live")
    assert client.delete(f"{URL}/{recent['id']}", headers=hdrs).status_code == 204
    assert client.delete(f"{URL}/{old['id']}", headers=hdrs).status_code == 204
    _set(db_path, old["id"], revoked_at=sqlite_utc_in(timedelta(days=-40)))
    items = client.get(URL, headers=hdrs).json()["items"]
    assert [(t["name"], t["status"]) for t in items] == [("live", "active"), ("recent", "revoked")]
    assert live["id"] == items[0]["id"]


def test_the_list_never_shows_inference_tokens(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    assert client.post("/api/tokens", json={"name": "bot"}, headers=hdrs).status_code == 201
    assert client.get(URL, headers=hdrs).json()["items"] == []


def test_refresh_issues_a_new_secret_and_keeps_the_old_one_for_the_grace(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    old = _issue(client, hdrs, name="ci")
    r = client.post(f"{URL}/{old['id']}/rotate", json={}, headers=hdrs)  # default grace: 1 h
    assert r.status_code == 201, r.text
    new = r.json()
    assert new["plaintext"].startswith("vwa_") and new["plaintext"] != old["plaintext"]
    assert (new["name"], new["rotated_from"], new["status"]) == ("ci", old["id"], "active")
    assert client.get("/api/tokens", headers=_bearer(old["plaintext"])).status_code == 200
    assert client.get("/api/tokens", headers=_bearer(new["plaintext"])).status_code == 200
    by_id = {t["id"]: t for t in client.get(URL, headers=hdrs).json()["items"]}
    assert (by_id[old["id"]]["name"], by_id[old["id"]]["status"]) == ("ci (old 1)", "grace")


def test_refresh_with_no_grace_cuts_the_old_secret_now(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    old = _issue(client, hdrs)
    registry = client.app.state.stream_registry
    stream = _FakeTask()
    registry.register(f"admin_token:{old['id']}", stream)
    r = client.post(f"{URL}/{old['id']}/rotate", json={"grace_hours": 0}, headers=hdrs)
    assert r.status_code == 201, r.text
    r = client.get("/api/tokens", headers=_bearer(old["plaintext"]))
    assert (r.status_code, r.json()["detail"]) == (401, "token revoked")
    assert stream.cancelled_called


def test_refresh_restarts_the_tokens_term(client, tmp_data_dir):
    db_path, hdrs = _ready(client, tmp_data_dir)
    old = _issue(client, hdrs, expires_in_days=30)
    # Twenty days into a thirty-day term: the successor gets thirty from now.
    _set(db_path, old["id"], created_at=sqlite_utc_in(timedelta(days=-20)),
         expires_at=sqlite_utc_in(timedelta(days=10)))
    new = client.post(f"{URL}/{old['id']}/rotate", json={"grace_hours": 1}, headers=hdrs).json()
    assert 29.9 < _days_from_now(new["expires_at"]) <= 30.0
    never = _issue(client, hdrs, name="forever", expires_in_days=None)
    again = client.post(f"{URL}/{never['id']}/rotate", json={}, headers=hdrs).json()
    assert again["expires_at"] is None


def test_refresh_refuses_a_refreshed_revoked_or_expired_token(client, tmp_data_dir):
    db_path, hdrs = _ready(client, tmp_data_dir)
    rotated = _issue(client, hdrs, name="rotated")
    client.post(f"{URL}/{rotated['id']}/rotate", json={}, headers=hdrs)
    revoked = _issue(client, hdrs, name="revoked")
    client.delete(f"{URL}/{revoked['id']}", headers=hdrs)
    expired = _issue(client, hdrs, name="expired")
    _set(db_path, expired["id"], expires_at=sqlite_utc_in(timedelta(minutes=-1)))
    for t in (rotated, revoked, expired):
        r = client.post(f"{URL}/{t['id']}/rotate", json={}, headers=hdrs)
        assert r.status_code == 409, (t["name"], r.text)


def test_refresh_takes_only_the_three_grace_choices(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    t = _issue(client, hdrs)
    r = client.post(f"{URL}/{t['id']}/rotate", json={"grace_hours": 2}, headers=hdrs)
    assert r.status_code == 422


def test_revoke_is_immediate_idempotent_and_keeps_the_row(client, tmp_data_dir):
    db_path, hdrs = _ready(client, tmp_data_dir)
    t = _issue(client, hdrs)
    stream = _FakeTask()
    client.app.state.stream_registry.register(f"admin_token:{t['id']}", stream)
    assert client.delete(f"{URL}/{t['id']}", headers=hdrs).status_code == 204
    assert stream.cancelled_called
    r = client.get("/api/tokens", headers=_bearer(t["plaintext"]))
    assert (r.status_code, r.json()["detail"]) == (401, "token revoked")
    first = client.get(URL, headers=hdrs).json()["items"][0]
    assert (first["id"], first["status"]) == (t["id"], "revoked")
    _set(db_path, t["id"], revoked_at="2026-01-01 00:00:00")
    assert client.delete(f"{URL}/{t['id']}", headers=hdrs).status_code == 204
    with sqlite3.connect(db_path) as db:
        (revoked_at,) = db.execute(
            "SELECT revoked_at FROM api_tokens WHERE id = ?", (t["id"],)
        ).fetchone()
    assert revoked_at == "2026-01-01 00:00:00"  # the original time is kept


def test_admin_token_routes_404_inference_and_unknown_ids(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    inference_id = client.post("/api/tokens", json={"name": "bot"}, headers=hdrs).json()["id"]
    for tid in (inference_id, "deadbeefdeadbeefdeadbeefdeadbeef"):
        assert client.post(f"{URL}/{tid}/rotate", json={}, headers=hdrs).status_code == 404
        assert client.delete(f"{URL}/{tid}", headers=hdrs).status_code == 404
    # The inference key was not touched.
    assert client.get(f"/api/tokens/{inference_id}", headers=hdrs).json()["is_revoked"] is False


def test_an_admin_token_cannot_issue_admin_tokens(client, tmp_data_dir):
    _, hdrs = _ready(client, tmp_data_dir)
    secret = _issue(client, hdrs)["plaintext"]
    r = client.post(URL, json={"name": "copy", "expires_in_days": None}, headers=_bearer(secret))
    assert r.status_code == 403
    assert r.json()["detail"]["error_code"] == "session_only"
