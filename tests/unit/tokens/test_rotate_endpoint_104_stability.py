"""Stability re-confirmation for #104.

The original flake (test_list_includes_rotated_predecessor_during_grace)
fired intermittently on the CI shared runner: sync sqlite3.connect in
``_seed_done`` raced with the async aiosqlite WAL writer that the
TestClient lifespan held open, surfacing as 401 "invalid credentials".

The fix (tests/conftest.py::seed_admin_user) routes the seed through an
explicit WAL-aware sqlite3 connection with PRAGMA journal_mode = WAL +
PRAGMA wal_checkpoint(FULL), aligning the journal mode with the async
reader. This test runs the exact failing test body 10 times in a row
in the same process so CI catches a regression even when the runner
is fast and the race window is short.
"""

import pytest

from tests.conftest import csrf_header, seed_admin_user


def _jwt_login(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "hunter2"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.mark.parametrize("iteration", list(range(10)))
def test_104_rotate_predecessor_visible_during_grace_stable(iteration, tmp_data_dir, client):
    """Replays the previously-flaky scenario 10x with fresh fixtures each
    time (tmp_data_dir is function-scoped). Any 401 from _jwt_login here
    is a regression of #104.
    """
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)

    create = client.post("/api/tokens", json={"name": "ci-bot"}, headers={**auth, **h})
    assert create.status_code == 201, create.text
    old_id = create.json()["id"]

    rot = client.post(
        f"/api/tokens/{old_id}/rotate",
        json={"grace_hours": 24},
        headers={**auth, **h},
    )
    assert rot.status_code == 201, rot.text
    new_id = rot.json()["id"]

    r = client.get("/api/tokens", headers=auth)
    assert r.status_code == 200
    items = {it["id"]: it for it in r.json()["items"]}
    assert old_id in items, f"iteration {iteration}: predecessor missing during grace"
    assert new_id in items, f"iteration {iteration}: successor missing"
    assert items[old_id]["rotated_at"] is not None
    assert items[old_id]["successor_id"] == new_id
    assert items[new_id]["rotated_from"] == old_id
