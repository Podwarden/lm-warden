"""``baked_engine_version`` asks the backend (via the registry), not a name switch.

BACKLOG-macos P5 follow-up: the function used to branch on the backend NAME and
returned None for ``mlx``; the ``/api/system/load`` fingerprint passed
``backend or "vllm"``, which was wrong on the apple profile where NULL means MLX.
After the fix the body is a thin delegation to ``registry.get(name).version_string()``
-- mirroring ``_backend_version`` in ``app/system/routes_engine.py`` -- and the
call site passes the raw ``model.backend`` (possibly None) straight through.

The NEW behaviour is imported INSIDE each test: on the base tree the name is still
``baked_engine_version`` but its signature and body are different, and a module-level
import here would only fail on the base if the name were removed -- which it is not.
So the import is done inside a test to keep red-first semantics honest.
"""

from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError

import pytest

from app.runtime import variants
from app.runtime.backends import registry


@pytest.fixture(autouse=True)
def _clear_lru():
    variants.baked_engine_version.cache_clear()
    yield
    variants.baked_engine_version.cache_clear()


class _FakeBackend:
    def __init__(self, ver: str):
        self._ver = ver

    def version_string(self) -> str | None:
        return self._ver


def test_delegates_to_the_backend(monkeypatch):
    """A name the profile does register goes to ``registry.get(name)`` and the
    function returns what the backend's ``version_string()`` says -- no name branch."""
    from app.runtime.variants import baked_engine_version

    seen: list[str | None] = []

    def _fake_get(name: str | None):
        seen.append(name)
        return _FakeBackend("9.9.9")

    monkeypatch.setattr(registry, "get", _fake_get)
    assert baked_engine_version("mlx") == "9.9.9"
    assert seen == ["mlx"]


def test_none_resolves_through_the_registry_default(monkeypatch):
    """A NULL ``models.backend`` is passed through as None; the registry's D6
    default (vLLM on linux, MLX on apple) resolves it. The function itself does
    not substitute -- that is the registry's one-line job."""
    from app.runtime.variants import baked_engine_version

    seen: list[str | None] = []

    def _fake_get(name: str | None):
        seen.append(name)
        return _FakeBackend("9.9.9")

    monkeypatch.setattr(registry, "get", _fake_get)
    assert baked_engine_version(None) == "9.9.9"
    assert seen == [None]


def test_unregistered_name_is_none(monkeypatch):
    """A name this profile does not register reports None, never raises -- the
    apple profile has no vllm backend, and the fingerprint must not crash on it."""
    from app.runtime.variants import baked_engine_version

    def _fake_get(name: str | None):
        raise registry.UnknownBackendError(f"unknown backend {name!r}")

    monkeypatch.setattr(registry, "get", _fake_get)
    assert baked_engine_version("no-such-backend") is None


def test_vllm_on_linux_is_the_package_version(monkeypatch):
    """No fake: on the linux profile the vllm backend's declared version (its
    package metadata) is what comes back. The test suite's linux profile
    registers vllm; the value is whatever ``importlib.metadata`` reports."""
    from app.runtime.variants import baked_engine_version

    def _fake_dist(name: str) -> str:
        if name == "vllm":
            return "0.11.0"
        raise PackageNotFoundError(name)

    monkeypatch.setattr("app.runtime.backends.vllm._dist_version", _fake_dist)
    assert baked_engine_version("vllm") == "0.11.0"


def test_llamacpp_is_its_build_stamp(monkeypatch):
    """No fake: llamacpp's declared version is the build stamp it was compiled
    with (env ``VW_LLAMACPP_BUILD``). The function must agree with the backend."""
    from app.runtime.variants import baked_engine_version

    monkeypatch.setenv("VW_LLAMACPP_BUILD", "b9999")
    assert baked_engine_version("llamacpp") == registry.get("llamacpp").version_string()
    assert baked_engine_version("llamacpp") == "b9999"


def test_no_name_branch_left():
    """The function body must not contain a comparison against a specific
    backend name. The fix is a delegation, not a switch with a different set of
    arms -- a leftover ``if backend == "vllm"`` would silently regress mlx."""
    from app.runtime.variants import baked_engine_version

    src = inspect.getsource(baked_engine_version)
    assert '"vllm"' not in src
    assert "'vllm'" not in src
    assert '"llamacpp"' not in src
    assert "'llamacpp'" not in src
    assert '"mlx"' not in src
    assert "'mlx'" not in src


def test_load_call_site_passes_the_raw_backend():
    """The load fingerprint's call site hands ``model.backend`` (possibly None)
    straight through. A re-added ``or "vllm"`` there is invisible on linux and
    wrong on apple, where a NULL row is MLX -- pin the line itself."""
    from app.models import routes_api

    calls = [
        line.strip()
        for line in inspect.getsource(routes_api).splitlines()
        if "baked_engine_version(" in line and "import" not in line
    ]
    assert calls == ['_baked_ver = baked_engine_version(getattr(model, "backend", None))']
