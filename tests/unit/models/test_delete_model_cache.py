"""R39 (issue #273): DELETE /api/models/{model_id} must reclaim the
model's HF cache directory once no other row references the same
hf_repo.

Pre-fix, delete_model only removed the DB row; the weight tree under
``hf_cache_dir/models--{org}--{name}/`` (or the ``hub/``-parent variant,
HF layout drift) sat on disk indefinitely with no API-visible trace.
"""

from pathlib import Path

from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _seed_done(db_path, allowed=None):
    seed_admin_user(db_path, allowed_gpu_indices=allowed)


def _jwt_login(client, username="admin", password="hunter2"):
    return jwt_login(client, username=username, password=password)


def _register_model(client, auth, h, served: str, repo: str, gpu: int) -> str:
    r = client.post(
        "/api/models",
        json={"served_model_name": served, "hf_repo": repo, "gpu_indices": [gpu]},
        headers={**auth, **h},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_cache_dir(hf_cache_dir: Path, repo: str, under_hub: bool = False) -> Path:
    """Write a fake blob tree at ``models--{org}--{name}/``, optionally
    under the ``hub/`` parent (HF layout drift). Returns the repo root."""
    safe = "models--" + repo.replace("/", "--", 1)
    root = Path(hf_cache_dir) / ("hub" if under_hub else "") / safe
    blob = root / "blobs" / "fake-weights.bin"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    return root


def test_delete_model_reclaims_cache_when_repo_has_no_other_row(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid = _register_model(client, auth, h, "x", "o/r", 0)
    cache_dir = _make_cache_dir(tmp_data_dir / "hf-cache", "o/r")
    assert cache_dir.exists()

    r = client.delete(f"/api/models/{mid}", headers={**auth, **h})
    assert r.status_code == 204, r.text

    assert client.get(f"/api/models/{mid}", headers=auth).status_code == 404
    assert not cache_dir.exists(), "last row for the repo is gone; its cache dir must be reclaimed"


def test_delete_model_keeps_cache_shared_by_a_surviving_row(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid_a = _register_model(client, auth, h, "a", "o/r", 0)
    mid_b = _register_model(client, auth, h, "b", "o/r", 1)
    cache_dir = _make_cache_dir(tmp_data_dir / "hf-cache", "o/r")
    marker = cache_dir / "blobs" / "fake-weights.bin"
    assert marker.exists()

    r = client.delete(f"/api/models/{mid_a}", headers={**auth, **h})
    assert r.status_code == 204, r.text
    assert cache_dir.exists(), "a surviving row still references the repo; shared blobs must stay"
    assert marker.exists()

    r = client.delete(f"/api/models/{mid_b}", headers={**auth, **h})
    assert r.status_code == 204, r.text
    assert not cache_dir.exists(), "no rows left for the repo; its cache dir must be reclaimed"


def test_delete_model_reclaims_cache_in_hub_layout(tmp_data_dir, client):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    auth = _jwt_login(client)
    h = csrf_header(client)
    mid = _register_model(client, auth, h, "x", "o/r", 0)
    cache_dir = _make_cache_dir(tmp_data_dir / "hf-cache", "o/r", under_hub=True)
    assert cache_dir.exists()

    r = client.delete(f"/api/models/{mid}", headers={**auth, **h})
    assert r.status_code == 204, r.text
    assert not cache_dir.exists(), "the hub-layout cache dir must be reclaimed too"
