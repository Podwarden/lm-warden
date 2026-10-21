"""The load tracker's backend resolves NULL to the profile default.

D6: ``models.backend`` is nullable and NULL means the profile's default —
vLLM on linux, MLX on apple. A hardcoded ``or "vllm"`` at the tracker
builder is wrong on the apple profile, where a NULL row's MLX load
progress never gets the MLX parser.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


def test_null_backend_is_profile_default_apple(monkeypatch):
    from app.models.routes_api import _tracker_backend
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _tracker_backend(SimpleNamespace(backend=None)) == "mlx"


def test_empty_backend_is_profile_default(monkeypatch):
    from app.models.routes_api import _tracker_backend
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _tracker_backend(SimpleNamespace(backend="")) == "mlx"


def test_missing_attribute_is_profile_default(monkeypatch):
    from app.models.routes_api import _tracker_backend
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _tracker_backend(SimpleNamespace()) == "mlx"


def test_null_backend_on_linux_default_is_vllm(monkeypatch):
    from app.models.routes_api import _tracker_backend
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "vllm")
    assert _tracker_backend(SimpleNamespace(backend=None)) == "vllm"


def test_explicit_backend_wins(monkeypatch):
    from app.models.routes_api import _tracker_backend
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _tracker_backend(SimpleNamespace(backend="llamacpp")) == "llamacpp"


def test_module_has_no_hardcoded_vllm_fallback():
    import app.models.routes_api

    assert 'or "vllm"' not in inspect.getsource(app.models.routes_api)


class _TrackerBuilt(Exception):
    """Raised by the stand-in LoadTracker so start_engine stops right there."""


def _tracker_backend_at_call_site(monkeypatch, tmp_path, backend):
    """Drive the real start_engine up to its LoadTracker(...) call and return
    the backend it passes, so the call site itself is pinned, not just the helper."""
    import asyncio

    import app.models.routes_api as routes_api

    seen: dict[str, object] = {}

    def fake_tracker(**kwargs):
        seen.update(kwargs)
        raise _TrackerBuilt

    async def fake_load(*a, **k):
        return None

    monkeypatch.setattr(routes_api, "LoadTracker", fake_tracker)
    sup = SimpleNamespace(
        load=fake_load, get_generation=lambda _id: 1, get_host=lambda _id: "127.0.0.1"
    )
    settings = SimpleNamespace(
        logs_dir=tmp_path,
        db_path=tmp_path / "db.sqlite",
        load_stall_warn_s=30,
        load_stall_timeout_s=60,
        load_max_s=600,
    )
    model = SimpleNamespace(id="m1", backend=backend)
    try:
        asyncio.run(routes_api.start_engine(settings, sup, SimpleNamespace(), model, 8000))
    except _TrackerBuilt:
        pass
    return seen.get("backend")


def test_start_engine_tracker_null_backend_is_profile_default(monkeypatch, tmp_path):
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "mlx")
    assert _tracker_backend_at_call_site(monkeypatch, tmp_path, None) == "mlx"


def test_start_engine_tracker_explicit_backend_wins(monkeypatch, tmp_path):
    from app.runtime.backends import registry

    monkeypatch.setattr(registry, "DEFAULT_BACKEND", "vllm")
    assert _tracker_backend_at_call_site(monkeypatch, tmp_path, "llamacpp") == "llamacpp"
