import sqlite3

from tests.conftest import csrf_header, seed_admin_user


def _setup_state(db_path):
    with sqlite3.connect(db_path) as db:
        return db.execute("SELECT step, draft FROM setup_state").fetchone()


def test_get_state_fresh_db_reports_welcome(tmp_data_dir, client):
    # Fresh install: no auth, no CSRF — the entry-gate fetch must work for an
    # anonymous first-run visitor.
    client.get("/healthz")
    r = client.get("/api/setup/state")
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "welcome", "done": False}


def test_get_state_after_setup_complete_reports_done(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/setup/state")
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "done", "done": True}


def test_post_welcome_advances_to_gpus(tmp_data_dir, client):
    client.get("/healthz")
    r = client.post("/api/setup/welcome", follow_redirects=False, headers=csrf_header(client))
    assert r.status_code == 200
    assert r.json() == {"step": "gpus"}
    step, _ = _setup_state(tmp_data_dir / "vllm-warden.db")
    assert step == "gpus"


def test_post_welcome_is_idempotent_mid_wizard(tmp_data_dir, client):
    # Regression: browser Back / reload lands a mid-wizard visitor on the
    # welcome page again; re-posting welcome must NOT 400 (that stranded
    # first-run setup) — it reports the real current step so the client
    # can resume there, and leaves state untouched.
    client.get("/healthz")
    h = csrf_header(client)
    client.post("/api/setup/welcome", headers=h)  # welcome -> gpus

    r = client.post("/api/setup/welcome", headers=h)
    assert r.status_code == 200, r.text
    assert r.json() == {"step": "gpus"}
    step, _ = _setup_state(tmp_data_dir / "vllm-warden.db")
    assert step == "gpus"  # unchanged — no double-advance


def test_post_gpus_validates_subset_and_persists(tmp_data_dir, client, monkeypatch):
    from app.system import gpu as gpu_mod
    from app.system.gpu import GpuInfo

    async def fake_query():
        return [
            GpuInfo(0, "A100", 40960, 0, 0),
            GpuInfo(1, "A100", 40960, 0, 0),
            GpuInfo(2, "A100", 40960, 0, 0),
            GpuInfo(3, "A100", 40960, 0, 0),
        ]

    monkeypatch.setattr(gpu_mod, "query_gpus", fake_query)

    client.get("/healthz")
    h = csrf_header(client)
    client.post("/api/setup/welcome", headers=h)
    r = client.post("/api/setup/gpus", json={"allowed_gpu_indices": [1, 2]}, headers=h)
    assert r.status_code == 200, r.text

    import json

    step, draft = _setup_state(tmp_data_dir / "vllm-warden.db")
    assert step == "hf_token"
    assert json.loads(draft)["allowed_gpu_indices"] == [1, 2]


def test_post_gpus_rejects_indices_out_of_range(tmp_data_dir, client, monkeypatch):
    from app.system import gpu as gpu_mod
    from app.system.gpu import GpuInfo

    async def fake_query():
        return [GpuInfo(i, "A100", 40960, 0, 0) for i in range(2)]

    monkeypatch.setattr(gpu_mod, "query_gpus", fake_query)

    client.get("/healthz")
    h = csrf_header(client)
    client.post("/api/setup/welcome", headers=h)
    r = client.post("/api/setup/gpus", json={"allowed_gpu_indices": [0, 5]}, headers=h)
    assert r.status_code == 400
