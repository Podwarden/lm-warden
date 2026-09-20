"""The inference-token API never sees admin tokens (spec 2026-09-19, decision
2): no /api/tokens route lists, shows, edits, rotates, pauses, tests, reports
on or deletes an admin row -- each answers 404 as if it did not exist -- and
the list's counts leave them out."""

import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import pytest

from app.db.repos.tokens import sqlite_utc_in, sqlite_utc_now
from tests.conftest import csrf_header, jwt_login, seed_admin_token, seed_admin_user


def _ready(client, tmp_data_dir: Path) -> tuple[Path, dict[str, str]]:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path, {**jwt_login(client), **csrf_header(client)}


def test_every_token_route_404s_an_admin_row(client, tmp_data_dir):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid, _ = seed_admin_token(db_path, name="ops")
    now = time.time()
    calls = [
        ("GET", f"/api/tokens/{tid}", None),
        ("PATCH", f"/api/tokens/{tid}", {"name": "renamed"}),
        ("PATCH", f"/api/tokens/{tid}", {"paused": True}),
        ("POST", f"/api/tokens/{tid}/rotate", {"grace_hours": 0}),
        ("POST", f"/api/tokens/{tid}/test", None),
        ("GET", f"/api/tokens/{tid}/usage", None),
        ("GET", f"/api/tokens/{tid}/series?from={now - 3600}&to={now}", None),
        ("DELETE", f"/api/tokens/{tid}", None),
    ]
    for method, url, body in calls:
        r = client.request(method, url, json=body, headers=hdrs)
        assert r.status_code == 404, (method, url, r.text)
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT name, revoked_at, rotated_at, paused_at FROM api_tokens WHERE id = ?", (tid,)
        ).fetchone()
    assert row == ("ops", None, None, None)  # untouched


@pytest.mark.parametrize(
    "params",
    [{}, {"q": "alpha"}, {"near_expiry": 1}, {"near_expiry": 1, "q": "alpha"},
     {"sort": "status"}, {"sort": "usage_24h"}, {"sort": "name", "dir": "asc"}],
)
def test_the_list_and_its_counts_never_include_admin_rows(client, tmp_data_dir, params):
    db_path, hdrs = _ready(client, tmp_data_dir)
    for name in ("alpha", "beta"):
        r = client.post("/api/tokens", json={"name": name, "expires_in_days": 10}, headers=hdrs)
        assert r.status_code == 201, r.text
    soon = sqlite_utc_in(timedelta(days=10))
    seed_admin_token(db_path, name="alpha-admin", expires_at=soon)
    # Visible under the inference list's own rule (rotated, in grace)...
    seed_admin_token(db_path, name="alpha-admin-rotated", expires_at=soon,
                     rotated_at=sqlite_utc_now(), revoked_at=sqlite_utc_in(timedelta(hours=1)))
    # ...and hidden by it anyway (revoked without a rotation).
    seed_admin_token(db_path, name="alpha-admin-revoked", expires_at=soon,
                     revoked_at=sqlite_utc_now())
    body = client.get("/api/tokens", params=params, headers=hdrs).json()
    names = [i["name"] for i in body["items"]]
    assert not [n for n in names if "admin" in n], names
    expected = 1 if "q" in params else 2
    assert (body["total"], body["near_expiry"]) == (expected, expected)


def test_the_playground_sweep_leaves_admin_rows_alone(client, tmp_data_dir):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid, _ = seed_admin_token(db_path, name="vw-playground")
    r = client.post("/api/chat/playground/ensure", headers=hdrs)
    assert r.status_code == 200, r.text
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM api_tokens WHERE id = ?", (tid,)).fetchone() == (1,)
