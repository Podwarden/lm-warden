from tests.conftest import jwt_login, seed_admin_user


# #55 fix — route through the shared barrier+retry helpers.
def _seed_admin(db_path, username="admin", pw="hunter2"):
    seed_admin_user(db_path, username=username, password=pw)


def _jwt_login(client):
    # Module-local wrapper returns just the token string (legacy shape).
    auth = jwt_login(client)
    return auth["Authorization"].removeprefix("Bearer ")


def test_no_authorization_header_401(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/tokens")
    assert r.status_code == 401


def test_malformed_header_401(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/tokens", headers={"Authorization": "Token foo"})
    assert r.status_code == 401


def test_valid_bearer_jwt_200(tmp_data_dir, client):
    client.get("/healthz")
    _seed_admin(tmp_data_dir / "vllm-warden.db")
    access = _jwt_login(client)
    r = client.get(
        "/api/tokens",
        headers={"Authorization": f"Bearer {access}"},
    )
    assert r.status_code == 200
