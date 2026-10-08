"""Endpoint tests for ``GET /api/templates/engine-versions`` (#177).

Hermetic — no Docker Hub. We install a fake ``EngineVersionsCache`` on
``app.state.engine_versions_cache`` (same seam the production route lazily
creates) so the route's job — channel→family resolution + envelope shaping —
is exercised without any network. The fetch/cache mechanics themselves are
covered in ``test_engine_versions.py``.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import jwt_login, seed_admin_user


class _FakeCache:
    """Drop-in for ``EngineVersionsCache`` — returns a canned ``(versions, error)``."""

    def __init__(self, versions: list[str], error: str | None = None) -> None:
        self._versions = versions
        self._error = error
        self.families: list[str] = []

    async def get(self, family: str) -> tuple[list[str], str | None]:
        self.families.append(family)
        return self._versions, self._error


def _seed(db_path: Path) -> None:
    seed_admin_user(db_path)


def test_engine_versions_requires_auth(client, tmp_data_dir):
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/api/templates/engine-versions?channel=cuda-stable")
    assert r.status_code == 401


def test_engine_versions_resolvable_channel_returns_versions(client, tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path)
    auth = jwt_login(client)
    fake = _FakeCache(["0.21.0", "0.20.0"])
    client.app.state.engine_versions_cache = fake

    r = client.get("/api/templates/engine-versions?channel=cuda-stable", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["channel"] == "cuda-stable"
    assert body["family"] == "vllm/vllm-openai"
    assert body["versions"] == ["0.21.0", "0.20.0"]
    assert body["error"] is None
    # The route resolved the channel to the upstream family before hitting cache.
    assert fake.families == ["vllm/vllm-openai"]


def test_engine_versions_non_resolvable_channel_returns_empty(client, tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path)
    auth = jwt_login(client)
    # Even if a cache is present, a non-resolvable channel must short-circuit
    # to an empty list WITHOUT touching the cache (no Docker Hub family).
    fake = _FakeCache(["should-not-appear"])
    client.app.state.engine_versions_cache = fake

    for channel in ("rocm", "cpu", "xpu", "unknown"):
        r = client.get(f"/api/templates/engine-versions?channel={channel}", headers=auth)
        assert r.status_code == 200, channel
        body = r.json()
        assert body["channel"] == channel
        assert body["family"] is None
        assert body["versions"] == []
    assert fake.families == []  # cache never consulted for non-resolvable channels


def test_engine_versions_reflects_cache_error_without_failing(client, tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path)
    auth = jwt_login(client)
    fake = _FakeCache([], error="docker_hub_unavailable")
    client.app.state.engine_versions_cache = fake

    r = client.get("/api/templates/engine-versions?channel=cuda-edge", headers=auth)
    assert r.status_code == 200  # third-party hiccup never 500s the page
    body = r.json()
    assert body["versions"] == []
    assert body["error"] == "docker_hub_unavailable"


def test_engine_versions_defaults_channel_to_cuda_stable(client, tmp_data_dir):
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed(db_path)
    auth = jwt_login(client)
    fake = _FakeCache(["0.20.0"])
    client.app.state.engine_versions_cache = fake

    r = client.get("/api/templates/engine-versions", headers=auth)
    assert r.status_code == 200
    assert r.json()["channel"] == "cuda-stable"
