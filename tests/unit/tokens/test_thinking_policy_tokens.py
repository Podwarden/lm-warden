"""``thinking_policy`` on inference keys: 'client' (default) | 'off'."""

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


def test_create_defaults_to_client(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    created = _create(client, hdrs)
    assert created["thinking_policy"] == "client"
    detail = client.get(f"/api/tokens/{created['id']}", headers=hdrs).json()
    assert detail["thinking_policy"] == "client"


def test_create_off(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    created = _create(client, hdrs, thinking_policy="off")
    assert created["thinking_policy"] == "off"


def test_patch_token_to_off_and_back(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    r = client.patch(f"/api/tokens/{tid}", json={"thinking_policy": "off"}, headers=hdrs)
    assert r.status_code == 200, r.text
    assert r.json()["thinking_policy"] == "off"
    # an unrelated PATCH leaves it alone
    r = client.patch(f"/api/tokens/{tid}", json={"priority": 3}, headers=hdrs)
    assert r.json()["thinking_policy"] == "off"
    r = client.patch(f"/api/tokens/{tid}", json={"thinking_policy": "client"}, headers=hdrs)
    assert r.json()["thinking_policy"] == "client"


def test_bad_value_is_422(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    for bad in ("on", "", None, 1):
        r = client.patch(f"/api/tokens/{tid}", json={"thinking_policy": bad}, headers=hdrs)
        assert r.status_code == 422, (bad, r.text)
    r = client.post("/api/tokens", json={"name": "x", "thinking_policy": "bogus"}, headers=hdrs)
    assert r.status_code == 422


def test_rotate_carries_thinking_policy(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs, thinking_policy="off")["id"]
    r = client.post(f"/api/tokens/{tid}/rotate", json={}, headers=hdrs)
    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    assert client.get(f"/api/tokens/{new_id}", headers=hdrs).json()["thinking_policy"] == "off"


def test_list_item_shows_thinking_policy(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs, thinking_policy="off")["id"]
    items = client.get("/api/tokens", headers=hdrs).json()["items"]
    (mine,) = [i for i in items if i["id"] == tid]
    assert mine["thinking_policy"] == "off"
