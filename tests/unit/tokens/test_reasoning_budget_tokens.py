"""``reasoning_budget`` / ``reasoning_budget_mode`` on inference keys (#348)."""

import pytest

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


def _detail(client, hdrs, tid):
    return client.get(f"/api/tokens/{tid}", headers=hdrs).json()


def test_defaults_are_unset(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    created = _create(client, hdrs)
    assert created["reasoning_budget"] is None
    assert created["reasoning_budget_mode"] is None
    d = _detail(client, hdrs, created["id"])
    assert d["reasoning_budget"] is None and d["reasoning_budget_mode"] is None


def test_create_with_budget_and_mode(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    created = _create(client, hdrs, reasoning_budget=2048, reasoning_budget_mode="stop")
    assert created["reasoning_budget"] == 2048
    assert created["reasoning_budget_mode"] == "stop"
    d = _detail(client, hdrs, created["id"])
    assert (d["reasoning_budget"], d["reasoning_budget_mode"]) == (2048, "stop")


def test_patch_sets_changes_and_clears(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    r = client.patch(
        f"/api/tokens/{tid}",
        json={"reasoning_budget": 1000, "reasoning_budget_mode": "force_answer"},
        headers=hdrs,
    )
    assert r.status_code == 200, r.text
    assert (r.json()["reasoning_budget"], r.json()["reasoning_budget_mode"]) == (
        1000,
        "force_answer",
    )
    # an unrelated PATCH leaves both alone
    r = client.patch(f"/api/tokens/{tid}", json={"priority": 3}, headers=hdrs)
    assert (r.json()["reasoning_budget"], r.json()["reasoning_budget_mode"]) == (
        1000,
        "force_answer",
    )
    # null clears: no limit, and the model's mode again
    r = client.patch(
        f"/api/tokens/{tid}",
        json={"reasoning_budget": None, "reasoning_budget_mode": None},
        headers=hdrs,
    )
    assert r.status_code == 200, r.text
    assert r.json()["reasoning_budget"] is None and r.json()["reasoning_budget_mode"] is None


@pytest.mark.parametrize("bad", [0, -1, 1.5, "7", True])
def test_bad_budget_is_422(tmp_data_dir, client, bad):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    r = client.patch(f"/api/tokens/{tid}", json={"reasoning_budget": bad}, headers=hdrs)
    assert r.status_code == 422, r.text


def test_bad_mode_is_422(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs)["id"]
    r = client.patch(f"/api/tokens/{tid}", json={"reasoning_budget_mode": "bogus"}, headers=hdrs)
    assert r.status_code == 422
    r = client.post("/api/tokens", json={"name": "x", "reasoning_budget": 0}, headers=hdrs)
    assert r.status_code == 422


def test_rotate_carries_the_budget(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs, reasoning_budget=512, reasoning_budget_mode="stop")["id"]
    r = client.post(f"/api/tokens/{tid}/rotate", json={}, headers=hdrs)
    assert r.status_code == 201, r.text
    d = _detail(client, hdrs, r.json()["id"])
    assert (d["reasoning_budget"], d["reasoning_budget_mode"]) == (512, "stop")


def test_list_item_shows_the_budget(tmp_data_dir, client):
    hdrs = _ready(client, tmp_data_dir)
    tid = _create(client, hdrs, reasoning_budget=512)["id"]
    items = client.get("/api/tokens", headers=hdrs).json()["items"]
    (mine,) = [i for i in items if i["id"] == tid]
    assert mine["reasoning_budget"] == 512 and mine["reasoning_budget_mode"] is None
