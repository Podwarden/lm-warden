"""try-stack trial-and-error routes (#162).

Adapted to this repo's sync ``TestClient`` + JWT login fixtures (the plan's
async-client / ``make_model`` snippet does not match the real fixtures). A
local ``_make_model`` helper creates a registered model via the create-model
route and returns its id. Routes hang off ``/api/models``, so paths are
``/api/models/{id}/try-stack``.
"""
from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _seed_done(db_path, allowed=None):
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client):
    return jwt_login(client, username="admin", password="hunter2")


def _make_model(client, auth, h, served="ts1"):
    r = client.post("/api/models", json={
        "served_model_name": served,
        "hf_repo": "o/r",
        "gpu_indices": [0],
    }, headers={**auth, **h})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_try_stack_records_attempt_and_sets_engine(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid = _make_model(client, auth, h, served="ts1")

    r = client.post(f"/api/models/{mid}/try-stack", json={
        "channel": "cuda-stable", "vllm_version": "0.20.0",
    }, headers={**auth, **h})
    assert r.status_code == 201, r.text
    attempt_id = r.json()["attempt_id"]

    # engine axis now set on the model
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"]["channel"] == "cuda-stable"

    # history shows the pending attempt
    hist = client.get(f"/api/models/{mid}/try-stack", headers=auth).json()
    assert hist["attempts"][0]["result"] == "pending"

    # post a failed result → classifier fills category + suggestion
    res = client.post(f"/api/models/{mid}/try-stack/{attempt_id}", json={
        "result": "failed",
        "error": "CUDA error: no kernel image is available (sm_86)",
    }, headers={**auth, **h})
    assert res.status_code == 200, res.text
    assert res.json()["category"] == "cuda_arch_unsupported"
    assert res.json()["suggestion"]


# ---------------------------------------------------------------------------
# #265: try-stack goes through the one row writer
#
# It used to write `engine_channel`, `engine_vllm_version` and `engine_image`
# with a raw `UPDATE models` -- `resolve_image` returns an explicit `image`
# unchanged, so `engine_image` was any string the caller liked, landing on a row
# the supervisor hands straight to `containers.run`. It also rewrote a model
# that was currently serving, silently, because a pinned image only takes
# effect on the next load.
# ---------------------------------------------------------------------------


def _force_status(db_path, model_id, status):
    import sqlite3

    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE models SET status = ? WHERE id = ?", (status, model_id))
        db.commit()


def test_try_stack_refuses_a_loaded_model(tmp_data_dir, client):
    db_path = tmp_data_dir / "vllm-warden.db"
    client.get("/healthz")
    _seed_done(db_path, allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid = _make_model(client, auth, h, served="ts-loaded")
    _force_status(db_path, mid, "loaded")

    r = client.post(f"/api/models/{mid}/try-stack", json={
        "channel": "cuda-stable", "vllm_version": "0.20.0",
    }, headers={**auth, **h})
    assert r.status_code == 409, r.text
    assert "unload" in r.text.lower()
    # And no attempt was recorded either -- the refusal is before any write.
    hist = client.get(f"/api/models/{mid}/try-stack", headers=auth).json()
    assert hist["attempts"] == []


def test_try_stack_refuses_a_row_register_would_refuse(tmp_data_dir, client):
    """The engine axis is written onto the whole row, so the whole row is
    validated -- a model carrying a value register never accepted cannot have an
    engine pinned onto it until that value is fixed."""
    import json
    import sqlite3

    db_path = tmp_data_dir / "vllm-warden.db"
    client.get("/healthz")
    _seed_done(db_path, allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid = _make_model(client, auth, h, served="ts-legacy")
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE models SET extra_env = ? WHERE id = ?",
            (json.dumps({"LD_PRELOAD": "/tmp/evil.so"}), mid),
        )
        db.commit()

    r = client.post(f"/api/models/{mid}/try-stack", json={
        "channel": "cuda-stable", "vllm_version": "0.20.0",
    }, headers={**auth, **h})
    assert r.status_code == 422, r.text
    assert "hard-locked" in r.text.lower()
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"] is None
