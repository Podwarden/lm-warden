"""``anthropic_relay`` flag on inference keys (#287)."""

from tests.conftest import csrf_header, seed_admin_user


def _ready(client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}", **csrf_header(client)}


def _create(client, hdrs, **body):
    r = client.post("/api/tokens", json={"name": "k", **body}, headers=hdrs)
    assert r.status_code == 201, r.text
    return r.json()


def test_default_is_false(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    detail = client.get(f"/api/tokens/{tid}", headers=hdrs).json()
    assert detail["anthropic_relay"] is False


def test_create_true_shows_on_detail_and_list(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    created = _create(client, hdrs, anthropic_relay=True)
    assert created["anthropic_relay"] is True
    tid = created["id"]
    assert client.get(f"/api/tokens/{tid}", headers=hdrs).json()["anthropic_relay"] is True
    items = client.get("/api/tokens", headers=hdrs).json()["items"]
    (mine,) = [i for i in items if i["id"] == tid]
    assert mine["anthropic_relay"] is True


def test_patch_toggles(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs, anthropic_relay=True)["id"]
    r = client.patch(f"/api/tokens/{tid}", json={"anthropic_relay": False}, headers=hdrs)
    assert r.status_code == 200, r.text
    assert r.json()["anthropic_relay"] is False
    r = client.patch(f"/api/tokens/{tid}", json={"anthropic_relay": True}, headers=hdrs)
    assert r.json()["anthropic_relay"] is True
    # an unrelated PATCH leaves it alone
    r = client.patch(f"/api/tokens/{tid}", json={"priority": 3}, headers=hdrs)
    assert r.json()["anthropic_relay"] is True


def test_patch_null_is_422(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    r = client.patch(f"/api/tokens/{tid}", json={"anthropic_relay": None}, headers=hdrs)
    assert r.status_code == 422


def test_rotation_inherits_the_flag(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs, anthropic_relay=True)["id"]
    r = client.post(f"/api/tokens/{tid}/rotate", json={}, headers=hdrs)
    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    assert client.get(f"/api/tokens/{new_id}", headers=hdrs).json()["anthropic_relay"] is True
