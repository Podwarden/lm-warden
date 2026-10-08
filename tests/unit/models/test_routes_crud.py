import json
import sqlite3

from tests.conftest import csrf_header, jwt_login, seed_admin_user


# #55 fix — local shims route through the shared barrier+retry helpers
# so callers in this module pick up the WAL-race mitigation while keeping
# the existing ``allowed`` kwarg shape.
def _seed_done(db_path, allowed=None):
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def test_create_model_persists_and_returns_id(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[1, 2])
    auth = _jwt_login(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "qwen3.5-9b",
            "hf_repo": "Qwen/Qwen3.5-9B",
            "gpu_indices": [1, 2],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 201, r.text
    model_id = r.json()["id"]
    assert len(model_id) > 0

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        row = db.execute(
            "SELECT served_model_name, gpu_indices, tensor_parallel_size, status "
            "FROM models WHERE id = ?",
            (model_id,),
        ).fetchone()
    assert row[0] == "qwen3.5-9b"
    assert json.loads(row[1]) == [1, 2]
    assert row[2] == 2
    assert row[3] == "registered"


def test_create_model_rejects_gpus_outside_allowed(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "o/r",
            "gpu_indices": [2, 3],
        },
        headers={**auth, **csrf_header(client)},
    )
    assert r.status_code == 400
    assert "not in allowed_gpu_indices" in r.text.lower()


def test_create_rejects_duplicate_served_name(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    body = {"served_model_name": "x", "hf_repo": "o/r", "gpu_indices": [0]}
    assert client.post("/api/models", json=body, headers={**auth, **h}).status_code == 201
    assert client.post("/api/models", json=body, headers={**auth, **h}).status_code == 409


def test_list_models(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    client.post(
        "/api/models",
        json={"served_model_name": "a", "hf_repo": "o/r", "gpu_indices": [0]},
        headers={**auth, **h},
    )
    client.post(
        "/api/models",
        json={"served_model_name": "b", "hf_repo": "o/r2", "gpu_indices": [1]},
        headers={**auth, **h},
    )
    r = client.get("/api/models", headers=auth)
    names = sorted(m["served_model_name"] for m in r.json()["models"])
    assert names == ["a", "b"]


def test_delete_model_only_when_unloaded(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
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
    r = client.delete(f"/api/models/{mid}", headers={**auth, **h})
    assert r.status_code == 409

    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET status = 'registered' WHERE id = ?", (mid,))
        db.commit()
    r = client.delete(f"/api/models/{mid}", headers={**auth, **h})
    assert r.status_code == 204


def test_unauthed_models_api_401(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/models")
    assert r.status_code == 401


def test_get_model_returns_extra_env_and_extra_args(tmp_data_dir, client):
    """Operators must be able to read back what they wrote.

    Until now POST /api/models accepted extra_env / extra_args but
    GET /api/models/{id} omitted both — asymmetric, blind from the UI.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={
            "served_model_name": "x",
            "hf_repo": "o/r",
            "gpu_indices": [0],
            "extra_env": {"VLLM_USE_V1": "1"},
            "extra_args": ["--enforce-eager"],
        },
        headers={**auth, **h},
    )
    assert create.status_code == 201, create.text
    mid = create.json()["id"]

    r = client.get(f"/api/models/{mid}", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extra_env"] == {"VLLM_USE_V1": "1"}
    assert body["extra_args"] == ["--enforce-eager"]


def test_get_model_returns_persisted_backend_for_llamacpp_row(tmp_data_dir, client):
    """The read path must report the backend the row was registered with.

    Regression for the operator report "every model reports Engine: vllm".
    ``GET /api/models/{id}`` hand-built its response dict and never grew the
    ``backend`` / ``mmproj_filename`` / ``n_gpu_layers`` keys that migrations
    0027 and 0028 added, so the UI saw ``backend: undefined`` on a model that
    was demonstrably running ``llama-server`` and fell back to "vllm".
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={
            "served_model_name": "qwen-gguf",
            "hf_repo": "o/r-GGUF",
            "gpu_indices": [0],
            "backend": "llamacpp",
            "filename": "model-IQ3_XXS.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
            "n_gpu_layers": 20,
        },
        headers={**auth, **h},
    )
    assert create.status_code == 201, create.text
    mid = create.json()["id"]

    r = client.get(f"/api/models/{mid}", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "llamacpp"
    assert body["mmproj_filename"] == "mmproj-F16.gguf"
    assert body["n_gpu_layers"] == 20


def test_get_model_defaults_backend_to_vllm_for_legacy_row(tmp_data_dir, client):
    """A pre-migration-0027 row has backend NULL and must read back as vllm.

    ``_decode_row`` already applies DEFAULT_BACKEND; this pins that the read
    path never leaks the raw NULL to the UI (which is what made the UI's
    client-side ``?? "vllm"`` fallback look harmless in the first place).
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={
            "served_model_name": "legacy",
            "hf_repo": "o/r",
            "gpu_indices": [0],
        },
        headers={**auth, **h},
    )
    mid = create.json()["id"]
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET backend = NULL WHERE id = ?", (mid,))
        db.commit()

    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["backend"] == "vllm"
    assert body["mmproj_filename"] is None
    assert body["n_gpu_layers"] is None


def test_list_models_reports_backend_per_row(tmp_data_dir, client):
    """The list endpoint feeds the models table, which also labels engines.

    Same hand-built-dict defect as ``get_model``: without ``backend`` here the
    table cannot distinguish a vLLM row from a llama.cpp one either.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    h = csrf_header(client)
    client.post(
        "/api/models",
        json={
            "served_model_name": "a-vllm",
            "hf_repo": "o/r",
            "gpu_indices": [0],
        },
        headers={**auth, **h},
    )
    client.post(
        "/api/models",
        json={
            "served_model_name": "b-gguf",
            "hf_repo": "o/r2",
            "gpu_indices": [1],
            "backend": "llamacpp",
            "filename": "w.gguf",
        },
        headers={**auth, **h},
    )

    rows = client.get("/api/models", headers=auth).json()["models"]
    by_name = {m["served_model_name"]: m for m in rows}
    assert by_name["a-vllm"]["backend"] == "vllm"
    assert by_name["b-gguf"]["backend"] == "llamacpp"


def test_list_and_detail_agree_on_engine_settings_and_capability_flags(tmp_data_dir, client):
    """#237: list dropped engine settings AND every capability flag that
    detail already returned, and neither endpoint returned the capability
    flags at all -- so a ``PATCH .../settings`` write-then-verify against
    either read endpoint looked silently ignored.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0, 1])
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models",
        json={
            "served_model_name": "cap-test",
            "hf_repo": "o/r",
            "gpu_indices": [0],
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.95,
            "extra_args": ["--enforce-eager"],
        },
        headers={**auth, **h},
    )
    assert create.status_code == 201, create.text
    mid = create.json()["id"]

    patch = client.patch(
        f"/api/models/{mid}/settings",
        json={"supports_reasoning": True, "supports_vision": True},
        headers={**auth, **h},
    )
    assert patch.status_code == 200, patch.text

    detail = client.get(f"/api/models/{mid}", headers=auth).json()
    rows = client.get("/api/models", headers=auth).json()["models"]
    summary = next(m for m in rows if m["id"] == mid)

    assert summary == detail
    assert detail["max_model_len"] == 32768
    assert detail["gpu_memory_utilization"] == 0.95
    assert detail["extra_args"] == ["--enforce-eager"]
    assert detail["supports_reasoning"] is True
    assert detail["supports_vision"] is True
    assert detail["supports_tools"] is None  # never written -> stays null
