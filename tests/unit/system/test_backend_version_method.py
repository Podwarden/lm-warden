"""``Backend.version_string()``: each backend reports its OWN engine version.

BACKLOG-macos P5 / Metal design s10 'Capability flagging', exception 1. The
``/api/system/backends`` loop used to branch on the backend NAME
(``_backend_version`` in routes_engine.py), which made the MLX backend report
nothing even though mlx-lm is installed and knowable. Moving the version to a
method on each backend is what lets a new backend -- MLX included -- report a
version without the route learning about it.

Every value is a DECLARED one (package metadata or a build stamp), never a
subprocess or an ``import`` of the engine: the route answers a metadata question
and must not acquire a failure mode for it.

Backend classes are imported INSIDE each test. Importing one at module level
would make the whole file an ImportError on a tree that has no ``version_string``
yet, which runs nothing and is red for the wrong reason.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

from app.runtime.backends import registry
from tests.conftest import jwt_login, seed_admin_user


def test_mlx_version_comes_from_mlx_lm_metadata(monkeypatch):
    from app.runtime.backends.mlx import MlxBackend

    def _fake_version(name: str) -> str:
        if name == "mlx-lm":
            return "0.28.3"
        raise PackageNotFoundError(name)

    monkeypatch.setattr("app.runtime.backends.mlx._dist_version", _fake_version)
    assert MlxBackend().version_string() == "0.28.3"


def test_mlx_version_is_none_without_the_package(monkeypatch):
    from app.runtime.backends.mlx import MlxBackend

    def _missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("app.runtime.backends.mlx._dist_version", _missing)
    assert MlxBackend().version_string() is None


def test_vllm_version_comes_from_vllm_metadata(monkeypatch):
    from app.runtime.backends.vllm import VllmBackend

    def _fake_version(name: str) -> str:
        if name == "vllm":
            return "0.26.0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr("app.runtime.backends.vllm._dist_version", _fake_version)
    assert VllmBackend().version_string() == "0.26.0"


def test_llamacpp_version_comes_from_the_build_stamp(monkeypatch):
    from app.runtime.backends.llamacpp import LlamaCppBackend

    monkeypatch.setenv("VW_LLAMACPP_BUILD", "b10731")
    assert LlamaCppBackend().version_string() == "b10731"

    monkeypatch.delenv("VW_LLAMACPP_BUILD", raising=False)
    assert LlamaCppBackend().version_string() is None


def test_every_registered_backend_implements_version_string():
    for name in registry.available():
        backend = registry.get(name)
        assert callable(
            getattr(backend, "version_string", None)
        ), f"registered backend {name!r} does not implement version_string()"


def test_version_string_is_a_declared_protocol_member():
    """Declared on the ``Backend`` Protocol itself, so the ``@runtime_checkable``
    membership check requires it: a backend whose ``version_string`` is blanked
    out is no longer a Backend, the unchanged one still is."""
    from app.runtime.backends import Backend
    from app.runtime.backends.vllm import VllmBackend

    assert "version_string" in Backend.__dict__

    class NoVersion(VllmBackend):
        version_string = None  # type: ignore[assignment]

    assert isinstance(VllmBackend(), Backend)
    assert not isinstance(NoVersion(), Backend)


def test_every_profile_backend_implements_version_string():
    """Both profiles' backend sets, not only the running one: MLX registers only
    under the apple profile, so the registry the suite sees never holds it."""
    from app.runtime.backends.registry import backends_for

    for profile in ("linux", "apple"):
        for name, backend in backends_for(profile).items():
            assert callable(
                getattr(backend, "version_string", None)
            ), f"{profile} backend {name!r} does not implement version_string()"


def test_backend_version_alias_delegates_to_the_backend(monkeypatch):
    """The stress subject builder's name-keyed alias reports what the backend says."""
    from app.system.routes_engine import _backend_version

    monkeypatch.setattr(
        type(registry.get("llamacpp")),
        "version_string",
        lambda self: "sentinel-b1",
        raising=False,
    )
    assert _backend_version("llamacpp") == "sentinel-b1"


def test_backend_version_alias_unregistered_name_is_none(monkeypatch):
    """A name the profile does not register reports None, never raises -- the apple
    profile has no vllm backend, and the stress builder passes ``backend or "vllm"``."""
    from app.runtime.backends.registry import backends_for
    from app.system.routes_engine import _backend_version

    monkeypatch.setattr(registry, "_BACKENDS", backends_for("apple"))
    assert _backend_version("vllm") is None
    assert _backend_version("no-such-backend") is None


def test_backend_version_docstring_not_stale():
    """The docstring must not claim the stress builder passes ``or "vllm"``: it
    passes ``_subject_backend(model)``, which resolves a NULL backend to
    ``registry.DEFAULT_BACKEND`` -- the docstring said otherwise since eeaa50ee."""
    from app.system.routes_engine import _backend_version

    assert 'or "vllm"' not in _backend_version.__doc__
    assert "_subject_backend" in _backend_version.__doc__


def test_route_reports_what_the_backend_says(tmp_data_dir, client, monkeypatch):
    auth = _ready(tmp_data_dir, client)
    monkeypatch.setattr(
        type(registry.get("vllm")),
        "version_string",
        lambda self: "sentinel-9.9",
        raising=False,
    )
    body = client.get("/api/system/backends", headers=auth).json()
    vllm = next(b for b in body["backends"] if b["name"] == "vllm")
    assert vllm["version"] == "sentinel-9.9"


def _seed_done(db_path) -> None:
    seed_admin_user(db_path, allowed_gpu_indices=[0])


def _ready(tmp_data_dir, client):
    """Boot the app, seed an admin, and return the auth header.

    Mirrors the fixture in test_routes_backends.py: the default app runs the
    in-container subprocess driver, and the route is exercised with a real DB.
    """
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    return jwt_login(client)
