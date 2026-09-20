import sqlite3

import bcrypt


def _seed_admin(db_path, username="admin", pw="hunter2"):
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO users(username, password_hash) VALUES (?, ?)",
            (username, h),
        )
        db.execute("UPDATE setup_state SET step='done' WHERE id=1")
        db.commit()


def test_logs_stream_rejects_without_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")  # mark setup done; otherwise gate_setup redirects
    r = client.get("/api/models/some-id/logs/stream")
    # `ticket` became optional when admin tokens could authenticate a stream
    # by header (spec 2026-09-19, decision 9), so a request with neither is
    # refused by require_sse_ticket itself: 401, no longer FastAPI's 422.
    assert r.status_code == 401
    assert r.json()["detail"] == "missing ticket"


def test_logs_stream_rejects_wrong_path_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    # Mint a ticket bound to a DIFFERENT path; consuming it on /some-id/... must fail.
    ticket = client.app.state.sse_tickets.mint(
        "admin", "/api/models/OTHER/logs/stream"
    )
    r = client.get(f"/api/models/some-id/logs/stream?ticket={ticket}")
    assert r.status_code == 401


def test_logs_stream_rejects_bogus_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models/some-id/logs/stream?ticket=not-a-valid-ticket")
    assert r.status_code == 401
