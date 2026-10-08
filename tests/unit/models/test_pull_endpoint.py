import sqlite3

from tests.conftest import csrf_header, jwt_login, seed_admin_user, wait_until


# #55 fix — route through the shared barrier+retry helpers.
def _seed_login(client, tmp_data_dir):
    seed_admin_user(tmp_data_dir / "vllm-warden.db", allowed_gpu_indices=[0, 1])


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def test_pull_endpoint_accepts_and_runs_task(tmp_data_dir, client, monkeypatch):
    from app.models import routes_api as ra

    invoked = []

    async def fake_run_pull(model_id, settings, force=False):
        invoked.append((model_id, force))
        with sqlite3.connect(settings.db_path) as db:
            db.execute("UPDATE models SET status = 'pulled' WHERE id = ?", (model_id,))
            db.commit()

    monkeypatch.setattr(ra, "run_pull", fake_run_pull)

    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={"served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]},
        headers={**auth, **h},
    )
    mid = create.json()["id"]

    r = client.post(f"/api/models/{mid}/pull", headers={**auth, **h})
    assert r.status_code == 202

    def _status():
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            (s,) = db.execute("SELECT status FROM models WHERE id = ?", (mid,)).fetchone()
        return s if s == "pulled" else None

    assert wait_until(_status, what=f"model '{mid}' to reach 'pulled'") == "pulled"
    assert (mid, False) in invoked


def test_pull_endpoint_404_on_missing(tmp_data_dir, client):
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    auth = _jwt_login(client)
    r = client.post("/api/models/does-not-exist/pull", headers={**auth, **csrf_header(client)})
    assert r.status_code == 404


def test_pull_endpoint_passes_force_flag(tmp_data_dir, client, monkeypatch):
    """Regression: ?force=true must thread through to run_pull(force=True)
    so the disk-space safeguard can be skipped intentionally."""
    from app.models import routes_api as ra

    invoked = []

    async def fake_run_pull(model_id, settings, force=False):
        invoked.append((model_id, force))
        with sqlite3.connect(settings.db_path) as db:
            db.execute("UPDATE models SET status = 'pulled' WHERE id = ?", (model_id,))
            db.commit()

    monkeypatch.setattr(ra, "run_pull", fake_run_pull)

    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={"served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]},
        headers={**auth, **h},
    )
    mid = create.json()["id"]

    r = client.post(f"/api/models/{mid}/pull?force=true", headers={**auth, **h})
    assert r.status_code == 202
    assert r.json()["force"] is True

    def _status():
        with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
            (s,) = db.execute("SELECT status FROM models WHERE id = ?", (mid,)).fetchone()
        return s if s == "pulled" else None

    wait_until(_status, what=f"model '{mid}' to reach 'pulled' (force)")
    assert (mid, True) in invoked


def test_pull_endpoint_409_when_already_loaded(tmp_data_dir, client):
    client.get("/healthz")
    _seed_login(client, tmp_data_dir)
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={"served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]},
        headers={**auth, **h},
    )
    mid = create.json()["id"]
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status = 'loaded' WHERE id = ?", (mid,))
        db.commit()
    r = client.post(f"/api/models/{mid}/pull", headers={**auth, **h})
    assert r.status_code == 409
