"""Every admin-token request is audited (spec 2026-09-19, decision 7): one row
with the ROUTE TEMPLATE, status, duration and client IP, written shortly after
the response. Session requests are not audited.

The middleware enqueues and a background flusher batches (#258), so every test
that reads the table calls the ``flush_audit`` fixture first -- an explicit
drain on the app's own event loop, never a sleep.
"""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from app.db.repos.admin_audit import AdminAuditRepo
from app.db.repos.tokens import sqlite_utc_in
from tests.conftest import csrf_header, jwt_login, seed_admin_token, seed_admin_user


def _ready(client, tmp_data_dir: Path) -> tuple[Path, dict[str, str]]:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    seed_admin_user(db_path)
    return db_path, {**jwt_login(client), **csrf_header(client)}


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def _rows(db_path: Path) -> list[tuple]:
    with sqlite3.connect(db_path) as db:
        return db.execute(
            "SELECT token_id, username, method, path, status, duration_ms, client_ip, peer_ip "
            "FROM admin_audit ORDER BY id"
        ).fetchall()


def test_a_request_is_audited_with_the_route_template(client, tmp_data_dir, flush_audit):
    db_path, _ = _ready(client, tmp_data_dir)
    tid, secret = seed_admin_token(db_path)
    r = client.get("/api/models/abc123", headers=_bearer(secret))
    assert r.status_code == 404
    assert flush_audit() == 1
    [(token_id, username, method, path, status, duration_ms, client_ip, peer_ip)] = _rows(db_path)
    assert (token_id, username, method, path, status) == (
        tid, "admin", "GET", "/api/models/{model_id}", 404,
    )
    assert duration_ms >= 0
    assert client_ip == "testclient"  # TestClient's socket peer
    assert peer_ip == "testclient"


def test_the_client_ip_honours_x_forwarded_for(client, tmp_data_dir, flush_audit):
    db_path, _ = _ready(client, tmp_data_dir)
    _, secret = seed_admin_token(db_path)
    client.get("/api/tokens", headers={**_bearer(secret), "X-Forwarded-For": "198.51.100.7, 10.0.0.1"})
    flush_audit()
    row = _rows(db_path)[0]
    # client_ip trusts X-Forwarded-For (forgeable); peer_ip is the raw socket
    # peer next to it (the ruling's mitigation -- Task 1/decision 7).
    assert (row[6], row[7]) == ("198.51.100.7", "testclient")


def test_session_requests_are_not_audited(client, tmp_data_dir, flush_audit):
    db_path, hdrs = _ready(client, tmp_data_dir)
    client.get("/api/tokens", headers=hdrs)
    assert flush_audit() == 0
    assert _rows(db_path) == []


def test_a_refused_attempt_with_a_known_token_is_audited(client, tmp_data_dir, flush_audit):
    db_path, _ = _ready(client, tmp_data_dir)
    tid, secret = seed_admin_token(db_path, revoked_at=sqlite_utc_in(timedelta(minutes=-1)))
    assert client.get("/api/tokens", headers=_bearer(secret)).status_code == 401
    flush_audit()
    assert [(r[0], r[3], r[4]) for r in _rows(db_path)] == [(tid, "/api/tokens", 401)]
    # A secret matching no admin row has no token to file it under.
    client.get("/api/tokens", headers=_bearer("vwa_" + "z" * 56))
    assert flush_audit() == 0
    assert len(_rows(db_path)) == 1


def test_a_failing_audit_write_does_not_fail_the_request(
    client, tmp_data_dir, monkeypatch, flush_audit
):
    db_path, _ = _ready(client, tmp_data_dir)
    _, secret = seed_admin_token(db_path)

    async def boom(self, rows: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(AdminAuditRepo, "record_many", boom)
    assert client.get("/api/tokens", headers=_bearer(secret)).status_code == 200
    # The flush swallows it too -- a broken volume must not fail a shutdown.
    assert flush_audit() == 0
    assert _rows(db_path) == []


def test_the_audit_endpoint_pages_newest_first(client, tmp_data_dir, flush_audit):
    db_path, hdrs = _ready(client, tmp_data_dir)
    tid, secret = seed_admin_token(db_path)
    for path in ("/api/tokens", "/api/models", "/api/version"):
        client.get(path, headers=_bearer(secret))
    assert flush_audit() == 3
    r = client.get(f"/api/admin-tokens/{tid}/audit?limit=2", headers=hdrs)
    assert r.status_code == 200, r.text
    page = r.json()
    assert [i["path"] for i in page["items"]] == ["/api/version", "/api/models"]
    assert set(page["items"][0]) == {
        "id", "ts", "method", "path", "status", "duration_ms", "client_ip", "peer_ip", "username",
    }
    assert page["next_before"] == page["items"][-1]["ts"]
    older = client.get(
        f"/api/admin-tokens/{tid}/audit", params={"limit": 2, "before": page["next_before"]},
        headers=hdrs,
    ).json()
    assert ([i["path"] for i in older["items"]], older["next_before"]) == (["/api/tokens"], None)


@pytest.mark.parametrize("kind", ["inference", "unknown"])
def test_the_audit_endpoint_404s_what_is_not_an_admin_token(client, tmp_data_dir, kind):
    _, hdrs = _ready(client, tmp_data_dir)
    tid = (
        client.post("/api/tokens", json={"name": "bot"}, headers=hdrs).json()["id"]
        if kind == "inference" else "deadbeefdeadbeefdeadbeefdeadbeef"
    )
    assert client.get(f"/api/admin-tokens/{tid}/audit", headers=hdrs).status_code == 404


def test_an_admin_token_probing_a_session_only_route_is_audited(
    client, tmp_data_dir, flush_audit
):
    """Decision 7: "a leaked token probing token management leaves a trail" --
    the session-only 403 (require_session) is identified before it refuses,
    so it lands in the trail too (M2)."""
    db_path, _ = _ready(client, tmp_data_dir)
    tid, secret = seed_admin_token(db_path)
    r = client.get("/api/admin-tokens", headers=_bearer(secret))
    assert r.status_code == 403
    flush_audit()
    assert [(row[0], row[3], row[4]) for row in _rows(db_path)] == [
        (tid, "/api/admin-tokens", 403)
    ]


def test_a_batch_of_requests_produces_every_row(client, tmp_data_dir, flush_audit):
    """#258: the middleware enqueues, one flusher writes. Every request is
    still one row -- nothing is coalesced, nothing is lost. More requests than
    one batch holds, so the flush spans several transactions."""
    db_path, _ = _ready(client, tmp_data_dir)
    tid, secret = seed_admin_token(db_path)
    # Small batches, so the burst spans several transactions -- and the
    # background flusher's full-batch path fires during the requests, which is
    # why the count below is asserted on the TABLE and not on this flush's
    # return (it only writes whatever the flusher had not already taken).
    client.app.state.admin_audit.batch_max = 7
    for _ in range(40):
        assert client.get("/api/version", headers=_bearer(secret)).status_code == 200
    flush_audit()
    rows = _rows(db_path)
    assert len(rows) == 40
    assert {(r[0], r[3], r[4]) for r in rows} == {(tid, "/api/version", 200)}
    # Every row keeps its own ts, so the trail is still ordered by time.
    with sqlite3.connect(db_path) as db:
        ts = [r[0] for r in db.execute("SELECT ts FROM admin_audit ORDER BY id")]
    assert len(set(ts)) == 40
    assert ts == sorted(ts)


def test_shutdown_flushes_what_is_still_queued(tmp_data_dir, migrated_db_template):
    """The lifespan cancels the flusher, whose final flush writes the rest --
    a request served a moment before shutdown is not lost. The interval is
    pinned out of the way so only the shutdown flush can be the writer."""
    import shutil

    from fastapi.testclient import TestClient

    from app.main import build_app

    db_path = tmp_data_dir / "vllm-warden.db"
    shutil.copyfile(migrated_db_template, db_path)
    with TestClient(build_app()) as c:
        c.get("/healthz")
        seed_admin_user(db_path)
        _, secret = seed_admin_token(db_path)
        c.app.state.admin_audit.flush_interval_s = 3600.0
        assert c.get("/api/version", headers=_bearer(secret)).status_code == 200
        assert _rows(db_path) == [], "the flusher is parked; nothing should be written yet"
    assert [(r[3], r[4]) for r in _rows(db_path)] == [("/api/version", 200)]
