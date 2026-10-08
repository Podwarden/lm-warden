"""Engine-aware create-model + user-template CRUD routes (#162).

Uses the shared sync ``TestClient`` fixture + JWT login pattern from
``tests/unit/models/test_routes_crud.py`` (the plan's async-client snippet
does not match this repo's fixtures). Routes hang off the existing
``/api/models`` router, so the real paths are ``/api/models/templates``.
"""

from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _seed_done(db_path, allowed=None):
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client):
    return jwt_login(client, username="admin", password="hunter2")


def test_create_from_template_prefills_engine(tmp_data_dir, client):
    # #215: the template's engine pin is only persisted when the active
    # driver can honor it. The default driver is the in-container subprocess
    # driver (supports_engine_image = False), so this test — which asserts
    # the pin lands on the row — runs under a driver that swaps images, like
    # the docker driver. The imageless-driver half is covered by
    # tests/unit/models/test_template_engine_pin.py.
    from tests.fakes.fake_engine import FakeDriver

    class _ImageSwappingDriver(FakeDriver):
        supports_engine_image = True

    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    client.app.state.supervisor._driver = _ImageSwappingDriver()
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "from-tpl",
            "gpu_indices": [0],
            "template_id": "gpt-oss-20b",
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"] == {
        "channel": "cuda-stable",
        "vllm_version": "0.20.0",
        "image": "vllm/vllm-openai:v0.20.0",
    }
    assert body["hf_repo"] == "openai/gpt-oss-20b"


def test_explicit_engine_overrides_template(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/models",
        json={
            "served_model_name": "ovr",
            "gpu_indices": [0],
            "template_id": "gpt-oss-20b",
            "engine_channel": "cuda-edge",
            "engine_vllm_version": "0.21.0",
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    body = client.get(f"/api/models/{mid}", headers=auth).json()
    assert body["engine"]["channel"] == "cuda-edge"
    assert body["engine"]["vllm_version"] == "0.21.0"


def test_template_crud(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    create = client.post(
        "/api/models/templates",
        json={
            "id": "my-mistral",
            "label": "My Mistral",
            "hf_repo": "mistralai/Mistral-7B",
            "hf_revision": "main",
            "dtype": "auto",
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": False,
            "extra_args": [],
            "extra_env": {},
            "engine": {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None},
        },
        headers={**auth, **h},
    )
    assert create.status_code == 201, create.text
    listed = client.get("/api/models/templates", headers=auth).json()
    ids = {t["id"] for t in listed}
    assert {"gpt-oss-20b", "my-mistral"} <= ids
    assert any(t["id"] == "gpt-oss-20b" and t["source"] == "builtin" for t in listed)
    delr = client.delete("/api/models/templates/my-mistral", headers={**auth, **h})
    assert delr.status_code == 204
    delbuiltin = client.delete("/api/models/templates/gpt-oss-20b", headers={**auth, **h})
    assert delbuiltin.status_code == 400


def test_create_template_minimal_panel_body(tmp_data_dir, client):
    """The try-stack 'save working combo' POST omits max_model_len + tp.

    Regression for the merge-blocker where these were required and every
    save-as-template call 422'd. They now default; a minimal body must 201.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    r = client.post(
        "/api/models/templates",
        json={
            "id": "saved-combo",
            "label": "openai/gpt-oss-20b on cuda-stable vLLM 0.20.0",
            "hf_repo": "openai/gpt-oss-20b",
            "engine": {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None},
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    listed = client.get("/api/models/templates", headers=auth).json()
    saved = next(t for t in listed if t["id"] == "saved-combo")
    assert saved["max_model_len"] == 8192
    assert saved["tensor_parallel_size"] == 1
    assert saved["source"] == "user"


def test_save_combo_captures_live_extra_args_and_gpu_mem(tmp_data_dir, client):
    """#170: the save-working-combo POST must capture the live model's
    ``extra_args`` + ``gpu_memory_utilization`` from the model row, not the
    schema defaults.

    Repro: an AWQ single-GPU model that only works with --enforce-eager and
    gpu_memory_utilization=0.92 was saved as a template that came back with
    extra_args=[] and gpu_memory_utilization=0.9 (defaults), so re-instantiating
    from the template OOM'd. The backend now sources both from the live model
    row referenced by ``model_id``.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)

    # Create a model carrying the non-default tuning that worked.
    mr = client.post(
        "/api/models",
        json={
            "served_model_name": "awq-model",
            "hf_repo": "TheBloke/awq-model",
            "gpu_indices": [0],
            "gpu_memory_utilization": 0.92,
            "extra_args": ["--enforce-eager"],
        },
        headers={**auth, **h},
    )
    assert mr.status_code == 201, mr.text
    mid = mr.json()["id"]

    # Save-as-template the same way the try-stack panel does: a minimal body
    # that references the live model via model_id (no explicit extra_args /
    # gpu_memory_utilization in the body).
    r = client.post(
        "/api/models/templates",
        json={
            "id": "awq-saved",
            "label": "TheBloke/awq-model on cuda-stable vLLM 0.20.0",
            "hf_repo": "TheBloke/awq-model",
            "model_id": mid,
            "engine": {"channel": "cuda-stable", "vllm_version": "0.20.0", "image": None},
        },
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text

    listed = client.get("/api/models/templates", headers=auth).json()
    saved = next(t for t in listed if t["id"] == "awq-saved")
    assert saved["extra_args"] == ["--enforce-eager"], saved
    assert saved["gpu_memory_utilization"] == 0.92, saved


def test_template_with_parallel_flags_in_extra_args_is_refused(tmp_data_dir, client):
    """#286 review #5: a saved template carrying them would 422 on every register."""
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db", allowed=[0])
    auth = _jwt_login(client)
    h = csrf_header(client)
    for extra in (["--data-parallel-size", "7"], ["-tp=2"], ["--pipeline_parallel_size", "2"]):
        r = client.post(
            "/api/models/templates",
            json={"id": "bad-tpl", "label": "bad", "hf_repo": "org/m", "extra_args": extra},
            headers={**auth, **h},
        )
        assert r.status_code == 422, (extra, r.text)
        assert "data_parallel_size" in r.text
    ok = client.post(
        "/api/models/templates",
        json={
            "id": "ok-tpl",
            "label": "ok",
            "hf_repo": "org/m",
            "extra_args": ["--max-num-seqs", "8"],
        },
        headers={**auth, **h},
    )
    assert ok.status_code == 201, ok.text
