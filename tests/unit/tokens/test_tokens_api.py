from tests.conftest import csrf_header, jwt_login, seed_admin_user


# #55 fix — local shims route through the shared barrier+retry helpers so
# all callers in this module pick up the WAL-race mitigation.
def _seed_done(db_path):
    seed_admin_user(db_path)


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def test_create_token_returns_plaintext_once(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    r = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **csrf_header(client)})
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "ci-bot"
    assert body["plaintext"].startswith("vw_")
    assert len(body["plaintext"]) >= 30


def test_list_tokens_does_not_return_plaintext(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert "plaintext" not in items[0]
    assert "hash" not in items[0]
    assert items[0]["name"] == "ci-bot"
    assert items[0]["preview"]


def test_list_tokens_surfaces_created_at(tmp_data_dir, client):
    # The UI's tokens table renders a "Created" column from this field
    # (see frontend/src/components/tokens/token-row.tsx). Pin it here so a
    # silent removal from _enrich() doesn't surface as a blank column.
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    item = r.json()["items"][0]
    assert "created_at" in item
    assert isinstance(item["created_at"], str)
    assert item["created_at"]


def test_delete_token(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post("/api/tokens", json={"name": "x"}, headers={**auth, **h})
    tid = create.json()["id"]
    r = client.delete(f"/api/tokens/{tid}", headers={**auth, **h})
    assert r.status_code == 204
    r = client.get("/api/tokens", headers=auth)
    assert r.json()["items"] == []


def test_create_requires_session(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    # not logged in — CSRF token is valid but session is missing, so route returns 401
    r = client.post("/api/tokens", json={"name": "ci-bot"}, headers=csrf_header(client))
    assert r.status_code == 401
