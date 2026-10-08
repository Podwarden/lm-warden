from tests.conftest import jwt_login, seed_admin_user


# #55 fix — route through the shared barrier+retry helpers.
def _seed_admin(db_path, username="admin", pw="hunter2"):
    seed_admin_user(db_path, username=username, password=pw)


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def test_mint_ticket_requires_jwt(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    # No Authorization header -> 401
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/models/abc/logs/stream"},
    )
    assert r.status_code == 401


def test_mint_returns_ticket(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/models/abc/logs/stream"},
        headers=auth,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "ticket" in body
    assert isinstance(body["ticket"], str) and len(body["ticket"]) > 0


def test_minted_ticket_is_consumable_for_its_path(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.post(
        "/api/auth/sse-ticket",
        json={"path": "/api/models/abc/logs/stream"},
        headers=auth,
    )
    ticket = r.json()["ticket"]
    # Round-trip through the in-process TicketStore to prove the endpoint
    # produced a valid ticket bound to the requested path and the user.
    sub = client.app.state.sse_tickets.consume(ticket, "/api/models/abc/logs/stream")
    assert sub == "admin"
